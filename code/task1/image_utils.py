#!/usr/bin/env python3
"""
Image fetch + cache + base64 utilities for vision-enabled benchmark runs.

Used by run_task1.py when --input-mode is text_image or image_only.
Fetches each row's image_url over HTTP, validates and normalizes it, caches
the raw bytes on disk (keyed by SHA-256 of the URL), and returns a
`data:<mime>;base64,...` URL ready to drop into an OpenAI-compatible
`image_url` content block (which is what vLLM also accepts).

Design notes
------------
- Cache layout:  <cache_dir>/<sha256(url)>.bin  +  <sha256(url)>.mime
  (one file holds raw bytes, sibling file holds the MIME type)
- WebP is re-encoded to JPEG up front because not every VLM/processor
  handles webp reliably.
- A process-local lock dedups concurrent fetches of the same URL when
  the runner uses a ThreadPoolExecutor.
- Failures raise ImageFetchError with a short human-readable reason
  (HTTP code, timeout, oversize, non-image content-type, decode failure).
  Callers are expected to record the reason and skip the sample.
"""
from __future__ import annotations

import base64
import hashlib
import io
import threading
from pathlib import Path
from typing import Optional

import requests
from PIL import Image, UnidentifiedImageError


class ImageFetchError(Exception):
    """Raised when an image URL cannot be fetched or decoded."""


_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; food-as-medicine-benchmark/1.0; +https://example.local)"
)
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB hard cap

_EXT_BY_MIME = {
    "image/jpeg": "jpg",
    "image/jpg":  "jpg",
    "image/png":  "png",
    "image/gif":  "gif",
    "image/webp": "webp",
}

_inflight: dict[str, threading.Lock] = {}
_inflight_guard = threading.Lock()


def _url_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _get_lock(key: str) -> threading.Lock:
    with _inflight_guard:
        lock = _inflight.get(key)
        if lock is None:
            lock = threading.Lock()
            _inflight[key] = lock
        return lock


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    return cache_dir / f"{key}.bin", cache_dir / f"{key}.mime"


def _read_cache(cache_dir: Path, key: str) -> Optional[tuple[bytes, str]]:
    bin_path, mime_path = _cache_paths(cache_dir, key)
    if not (bin_path.exists() and mime_path.exists()):
        return None
    try:
        return bin_path.read_bytes(), mime_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _write_cache(cache_dir: Path, key: str, data: bytes, mime: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    bin_path, mime_path = _cache_paths(cache_dir, key)
    bin_tmp = bin_path.with_suffix(bin_path.suffix + ".tmp")
    mime_tmp = mime_path.with_suffix(mime_path.suffix + ".tmp")
    bin_tmp.write_bytes(data)
    mime_tmp.write_text(mime, encoding="utf-8")
    bin_tmp.replace(bin_path)
    mime_tmp.replace(mime_path)


def _normalize_payload(raw: bytes, content_type: str) -> tuple[bytes, str]:
    """Validate bytes as an image; re-encode webp -> jpeg.

    Returns (bytes, mime). Raises ImageFetchError on any decode problem.
    """
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime not in _EXT_BY_MIME:
        # Some CDNs report octet-stream; let PIL decide.
        mime = ""

    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
    except (UnidentifiedImageError, Exception) as exc:
        raise ImageFetchError(f"not a decodable image: {exc}") from exc

    # Re-open for actual format / conversion (verify() exhausts the file).
    try:
        img = Image.open(io.BytesIO(raw))
        pil_format = (img.format or "").lower()
    except Exception as exc:
        raise ImageFetchError(f"pillow open failed: {exc}") from exc

    if not mime:
        if pil_format in ("jpeg", "jpg"):
            mime = "image/jpeg"
        elif pil_format == "png":
            mime = "image/png"
        elif pil_format == "webp":
            mime = "image/webp"
        elif pil_format == "gif":
            mime = "image/gif"
        else:
            mime = "image/jpeg"  # fall through to re-encode

    if mime == "image/webp" or pil_format == "webp":
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"

    return raw, mime


def fetch_image_as_data_url(
    url: str,
    *,
    cache_dir: Path,
    timeout: float = 15.0,
    max_retries: int = 1,
    session: Optional[requests.Session] = None,
) -> str:
    """Return a ``data:<mime>;base64,...`` URL for ``url``.

    Uses an on-disk cache keyed by sha256(url). Raises ``ImageFetchError`` on
    any failure (network / HTTP / size / decode); caller should record the
    failure and skip the sample rather than retry blindly.
    """
    if not url or not isinstance(url, str):
        raise ImageFetchError("empty or non-string url")

    cache_dir = Path(cache_dir)
    key = _url_key(url)

    # Fast path: already cached.
    cached = _read_cache(cache_dir, key)
    if cached is not None:
        data, mime = cached
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    # Serialize concurrent fetches of the *same* URL.
    lock = _get_lock(key)
    with lock:
        # Re-check after acquiring the lock — another thread may have filled the cache.
        cached = _read_cache(cache_dir, key)
        if cached is not None:
            data, mime = cached
            return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

        sess = session or requests
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = sess.get(
                    url,
                    timeout=timeout,
                    allow_redirects=True,
                    headers={"User-Agent": _DEFAULT_USER_AGENT, "Accept": "image/*,*/*;q=0.8"},
                )
            except requests.RequestException as exc:
                last_err = exc
                continue

            if resp.status_code != 200:
                last_err = ImageFetchError(f"HTTP {resp.status_code}")
                # 4xx is permanent; don't retry.
                if 400 <= resp.status_code < 500:
                    break
                continue

            raw = resp.content
            if not raw:
                last_err = ImageFetchError("empty body")
                continue
            if len(raw) > _MAX_BYTES:
                raise ImageFetchError(f"body too large: {len(raw)} bytes")

            ctype = resp.headers.get("Content-Type", "")
            try:
                data, mime = _normalize_payload(raw, ctype)
            except ImageFetchError:
                raise

            _write_cache(cache_dir, key, data, mime)
            return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

        # All attempts exhausted.
        if isinstance(last_err, ImageFetchError):
            raise last_err
        raise ImageFetchError(f"fetch failed: {last_err!r}")
