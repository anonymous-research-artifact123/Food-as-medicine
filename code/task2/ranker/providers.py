"""Multi-provider call layer for the Task 2 rank task.

Supports openai (Responses API), anthropic, gemini and vllm (any OpenAI-compatible
server). Every builder takes the system prompt and the precomputed user text;
image captions use the SHUFFLED options' letters (what the model sees).
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .io_utils import ROOT, clean_text


# ---------------------------------------------------------------------------
# Provider detection and token budgets
# ---------------------------------------------------------------------------

def model_provider(model: str, provider_override: str = "") -> str:
    override = clean_text(provider_override).lower()
    if override:
        return override
    normalized = clean_text(model).lower()
    if normalized.startswith("gemini"):
        return "gemini"
    if normalized.startswith("claude"):
        return "anthropic"
    # Open-weight models (Qwen3-VL, Qwen3.5, Gemma 3/4) are served locally via vLLM.
    if "qwen" in normalized or "gemma" in normalized:
        return "vllm"
    return "openai"


def max_output_tokens(prompt_mode: str, model: str = "") -> int:
    mode = clean_text(prompt_mode).lower()
    normalized_model = clean_text(model).lower()
    uses_cot = mode in {"cot", "cot_ki"}
    # Gemini 2.5 family (pro/flash) silently burns part of the output budget on
    # internal reasoning, so the visible JSON gets truncated on stingy limits.
    if normalized_model.startswith("gemini-2.5"):
        return 32768 if "pro" in normalized_model else 8192
    if "gemma" in normalized_model:
        return 2048
    # CoT: 900 tokens truncated ~0.3% of Qwen3-VL CoT answers mid-sentence;
    # 1500 leaves headroom.
    return 1500 if uses_cot else 400


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def iter_image_options(shuffled_options: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and return options in the order they should appear in the prompt."""
    out: List[Dict[str, Any]] = []
    for option in shuffled_options:
        if option.get("is_abstention"):
            continue
        image_url = clean_text(option.get("image_url"))
        if not image_url and not clean_text(option.get("cached_image_path")):
            raise ValueError(
                f"option {clean_text(option.get('letter')).upper()}: missing image_url/cached_image_path"
            )
        out.append(option)
    return out


def _sniff_mime_type(payload: bytes, fallback: str = "") -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    guessed, _ = mimetypes.guess_type(fallback)
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/jpeg"


IMAGE_SOURCES = ("cached_first", "url_only")
_IMAGE_SOURCE = "cached_first"
# Same UA / Accept as code/task1/image_utils.py so both tasks see the same
# 403 behaviour from picky recipe sites.
_FETCH_USER_AGENT = "Mozilla/5.0 (compatible; food-as-medicine-benchmark/1.0; +https://example.local)"
_FETCH_TIMEOUT_SECONDS = 15.0
_URL_CACHE: Dict[str, Tuple[str, bytes]] = {}
_URL_CACHE_LOCK = threading.Lock()


class ImageFetchError(RuntimeError):
    """Raised when an option's image cannot be obtained under the active image source."""


def set_image_source(mode: str) -> str:
    """Process-wide image source. cached_first (default): local cached_image_path,
    falling back to image_url. url_only: always download image_url (the local
    cache is ignored) -- use this to measure how many images are unobtainable."""
    global _IMAGE_SOURCE
    key = clean_text(mode).lower() or "cached_first"
    if key not in IMAGE_SOURCES:
        raise ValueError(f"Unsupported image source '{mode}'. Use one of: {', '.join(IMAGE_SOURCES)}.")
    _IMAGE_SOURCE = key
    return key


def image_source() -> str:
    return _IMAGE_SOURCE


def _local_image(option: Dict[str, Any]) -> Optional[Tuple[str, bytes]]:
    cached_path = clean_text(option.get("cached_image_path"))
    if not cached_path:
        return None
    path = Path(cached_path)
    if not path.is_absolute():
        path = (ROOT / cached_path).resolve()
    if not (path.exists() and path.is_file()):
        return None
    data = path.read_bytes()
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = _sniff_mime_type(data, str(path))
    return mime_type, data


_FETCH_MAX_ATTEMPTS = 5
# 403 is retried: recipes.heart.org (a large share of the benchmark images)
# returns 403 intermittently from bot protection and usually succeeds on a
# later attempt. 404/410 and other 4xx are permanent.
_RETRYABLE_HTTP = {403, 408, 425, 429, 500, 502, 503, 504}


def _fetch_image_url(image_url: str) -> Tuple[str, bytes]:
    """Download once per process (memoised). Retries transient failures
    (network errors, 5xx, 403/429) with linear backoff, up to
    _FETCH_MAX_ATTEMPTS; other 4xx are permanent. Raises ImageFetchError."""
    with _URL_CACHE_LOCK:
        cached = _URL_CACHE.get(image_url)
    if cached is not None:
        return cached
    last_reason = ""
    for attempt in range(1, _FETCH_MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(
                image_url,
                headers={"User-Agent": _FETCH_USER_AGENT, "Accept": "image/*,*/*;q=0.8"},
            )
            with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:
                mime_type = response.headers.get_content_type() or "image/jpeg"
                data = response.read()
            if not data:
                raise ImageFetchError("empty body")
            if not mime_type.startswith("image/") or mime_type == "application/octet-stream":
                mime_type = _sniff_mime_type(data, image_url)
            with _URL_CACHE_LOCK:
                _URL_CACHE[image_url] = (mime_type, data)
            return mime_type, data
        except urllib.error.HTTPError as exc:
            last_reason = f"HTTP {exc.code}"
            if exc.code not in _RETRYABLE_HTTP:
                break
        except urllib.error.URLError as exc:
            last_reason = f"URLError {clean_text(getattr(exc, 'reason', exc))[:80]}"
        except ImageFetchError as exc:
            last_reason = str(exc)
            break
        except Exception as exc:  # timeouts, SSL, decode
            last_reason = f"{type(exc).__name__} {clean_text(str(exc))[:80]}"
        time.sleep(1.5 * attempt)
    raise ImageFetchError(f"{last_reason or 'unknown error'} (after {_FETCH_MAX_ATTEMPTS} attempts)")


def image_payload(option: Dict[str, Any]) -> Tuple[str, bytes]:
    """(mime_type, bytes) for an option under the active image source."""
    letter = clean_text(option.get("letter")).upper()
    if _IMAGE_SOURCE == "cached_first":
        local = _local_image(option)
        if local is not None:
            return local
    image_url = clean_text(option.get("image_url"))
    if not image_url:
        raise ImageFetchError(f"option {letter}: missing image_url")
    try:
        return _fetch_image_url(image_url)
    except ImageFetchError as exc:
        raise ImageFetchError(f"option {letter}: {exc}") from exc


def image_input_reference(option: Dict[str, Any]) -> str:
    """data: URL for the option image (always inlined so the provider never
    fetches the remote URL itself; that keeps url_only failures visible here)."""
    mime_type, payload = image_payload(option)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_images(shuffled_options: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Pre-fetch every option image under the active source. Returns one entry
    per FAILED option: {letter, image_url, reason}. Empty list = all good."""
    failures: List[Dict[str, str]] = []
    for option in shuffled_options:
        if option.get("is_abstention"):
            continue
        try:
            image_payload(option)
        except ImageFetchError as exc:
            failures.append(
                {
                    "letter": clean_text(option.get("letter")).upper(),
                    "image_url": clean_text(option.get("image_url")),
                    "reason": str(exc).split(": ", 1)[-1],
                }
            )
    return failures


def jpeg_data_url_for_provider(option: Dict[str, Any]) -> str:
    """RGB JPEG data URL for providers that reject other encodings (used for vllm)."""
    try:
        from PIL import Image, ImageOps
    except Exception:
        return image_input_reference(option)

    _mime_type, payload = image_payload(option)
    try:
        with Image.open(BytesIO(payload)) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode in {"RGBA", "LA"}:
                background = Image.new("RGB", image.size, (255, 255, 255))
                alpha = image.getchannel("A")
                background.paste(image.convert("RGB"), mask=alpha)
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")
            output = BytesIO()
            image.save(output, format="JPEG", quality=92, optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return image_input_reference(option)


def image_grid_data_url(shuffled_options: Sequence[Dict[str, Any]]) -> str:
    """Single 2x2 labeled JPEG grid for providers that accept only one image.

    Labels use the SHUFFLED letters (option["letter"]), which is what the model
    sees in the prompt.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except Exception as exc:
        raise RuntimeError(
            "Pillow is required for vLLM single-image grid mode. `pip install pillow`."
        ) from exc

    tile_size = 512
    label_height = 56
    gap = 12
    background = (255, 255, 255)
    label_bg = (20, 20, 20)
    label_fg = (255, 255, 255)
    canvas_width = tile_size * 2 + gap * 3
    canvas_height = (tile_size + label_height) * 2 + gap * 3
    canvas = Image.new("RGB", (canvas_width, canvas_height), background)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, option in enumerate(iter_image_options(shuffled_options)[:4]):
        letter = clean_text(option.get("letter")).upper() or chr(ord("A") + index)
        _mime_type, payload = image_payload(option)
        with Image.open(BytesIO(payload)) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode in {"RGBA", "LA"}:
                image_background = Image.new("RGB", image.size, background)
                alpha = image.getchannel("A")
                image_background.paste(image.convert("RGB"), mask=alpha)
                image = image_background
            elif image.mode != "RGB":
                image = image.convert("RGB")
            image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_size, tile_size), background)
            left = (tile_size - image.width) // 2
            top = (tile_size - image.height) // 2
            tile.paste(image, (left, top))

        col = index % 2
        row = index // 2
        x = gap + col * (tile_size + gap)
        y = gap + row * (tile_size + label_height + gap)
        draw.rectangle([x, y, x + tile_size, y + label_height], fill=label_bg)
        draw.text((x + 18, y + 18), f"Option {letter}", fill=label_fg, font=font)
        canvas.paste(tile, (x, y + label_height))

    output = BytesIO()
    canvas.save(output, format="JPEG", quality=92, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# ---------------------------------------------------------------------------
# Per-provider payload builders
# ---------------------------------------------------------------------------

def _responses_payload(
    system_prompt: str,
    user_text: str,
    shuffled_options: Sequence[Dict[str, Any]],
    *,
    attach_images: bool = True,
) -> Dict[str, Any]:
    """OpenAI Responses API input list. The Responses API doesn't have a
    distinct 'system' role on input; we prepend a developer message.
    """
    input_messages: List[Dict[str, Any]] = [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": system_prompt}],
        }
    ]
    user_content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_text}]
    for option in iter_image_options(shuffled_options) if attach_images else []:
        letter = clean_text(option.get("letter")).upper()
        user_content.append({"type": "input_text", "text": f"Option {letter}:"})
        user_content.append({"type": "input_image", "image_url": image_input_reference(option)})
    input_messages.append({"role": "user", "content": user_content})
    return input_messages


def _gemini_payload(
    model: str,
    prompt_mode: str,
    system_prompt: str,
    user_text: str,
    shuffled_options: Sequence[Dict[str, Any]],
    *,
    attach_images: bool = True,
) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = [{"text": user_text}]
    for option in iter_image_options(shuffled_options) if attach_images else []:
        letter = clean_text(option.get("letter")).upper()
        mime_type, payload = image_payload(option)
        parts.append({"text": f"Option {letter}:"})
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64.b64encode(payload).decode("ascii"),
                }
            }
        )
    out_tokens = max(max_output_tokens(prompt_mode, model), 256)
    return {
        "contents": [{"role": "user", "parts": parts}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "temperature": 0,
            "topP": 1,
            "topK": 1,
            "maxOutputTokens": out_tokens,
            "responseMimeType": "text/plain",
        },
    }


def _anthropic_payload(
    model: str,
    prompt_mode: str,
    system_prompt: str,
    user_text: str,
    shuffled_options: Sequence[Dict[str, Any]],
    *,
    attach_images: bool = True,
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    for option in iter_image_options(shuffled_options) if attach_images else []:
        letter = clean_text(option.get("letter")).upper()
        mime_type, payload = image_payload(option)
        content.append({"type": "text", "text": f"Option {letter}:"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64.b64encode(payload).decode("ascii"),
                },
            }
        )
    return {
        "model": model,
        "max_tokens": max_output_tokens(prompt_mode, model),
        "temperature": 0,
        "system": system_prompt,
        "messages": [{"role": "user", "content": content}],
    }


def _openai_chat_payload(
    model: str,
    prompt_mode: str,
    system_prompt: str,
    user_text: str,
    shuffled_options: Sequence[Dict[str, Any]],
    *,
    normalize_images_to_jpeg: bool = False,
    single_image_grid: bool = False,
    attach_images: bool = True,
) -> Dict[str, Any]:
    """OpenAI chat-completions-compatible payload (used for vllm)."""
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    if not attach_images:
        pass
    elif single_image_grid:
        user_content.append(
            {
                "type": "text",
                "text": "The single attached image is a 2x2 grid labeled Option A, Option B, Option C, and Option D.",
            }
        )
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_grid_data_url(shuffled_options)},
            }
        )
    else:
        for option in iter_image_options(shuffled_options):
            letter = clean_text(option.get("letter")).upper()
            user_content.append({"type": "text", "text": f"Option {letter}:"})
            url = jpeg_data_url_for_provider(option) if normalize_images_to_jpeg else image_input_reference(option)
            user_content.append({"type": "image_url", "image_url": {"url": url}})
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_output_tokens(prompt_mode, model),
    }


# ---------------------------------------------------------------------------
# Response extractors
# ---------------------------------------------------------------------------

def _extract_responses_text(response: Any) -> str:
    text = clean_text(getattr(response, "output_text", None))
    if text:
        return text
    try:
        payload = response.model_dump()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        for item in payload.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []) or []:
                if not isinstance(content, dict):
                    continue
                value = clean_text(content.get("text"))
                if value:
                    return value
    return ""


def _extract_gemini_text(payload: Dict[str, Any]) -> str:
    def _walk(node: Any) -> Optional[str]:
        if isinstance(node, dict):
            text = clean_text(node.get("text"))
            if text:
                return text
            for value in node.values():
                found = _walk(value)
                if found:
                    return found
            return None
        if isinstance(node, list):
            for item in node:
                found = _walk(item)
                if found:
                    return found
        return None

    for candidate in payload.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = clean_text(part.get("text"))
            if text:
                return text
    return _walk(payload) or ""


def _extract_anthropic_text(payload: Dict[str, Any]) -> str:
    for item in payload.get("content") or []:
        if isinstance(item, dict):
            text = clean_text(item.get("text"))
            if text:
                return text
    return ""


def _extract_openai_chat_text(payload: Dict[str, Any]) -> str:
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        text = clean_text(message.get("content"))
        if text:
            return text
    return ""


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

def create_client(model: str, provider_override: str = "") -> Dict[str, Any]:
    provider = model_provider(model, provider_override)
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is required for Gemini models.")
        return {"provider": provider, "api_key": api_key}
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY is required for Anthropic models.")
        return {"provider": provider, "api_key": api_key}
    if provider == "vllm":
        base_url = os.getenv("VLLM_BASE_URL", "").strip().rstrip("/") or "http://localhost:8000/v1"
        api_key = os.getenv("VLLM_API_KEY", "").strip() or "EMPTY"
        return {"provider": provider, "api_key": api_key, "base_url": base_url}
    if provider != "openai":
        raise SystemExit(
            f"Unsupported provider '{provider}'. Use openai, gemini, anthropic, or vllm."
        )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for OpenAI models.")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise SystemExit(f"Failed to import OpenAI client: {exc}") from exc
    return {"provider": provider, "client": OpenAI(api_key=api_key)}


# ---------------------------------------------------------------------------
# Top-level call
# ---------------------------------------------------------------------------

_MAX_RETRY_SLEEP_SECONDS = 60.0


def _is_retryable(exc: Exception) -> bool:
    """Whether another attempt at the same request could succeed.

    Every failure is retried (exponential backoff, up to --max-retries); a
    question whose calls all fail is reported as request_error and not scored.
    """
    return True


def call_model(
    client: Dict[str, Any],
    *,
    model: str,
    prompt_mode: str,
    system_prompt: str,
    user_text: str,
    shuffled_options: Sequence[Dict[str, Any]],
    max_retries: int,
    retry_sleep_seconds: float,
    request_timeout_seconds: float,
    attach_images: bool = True,
    vllm_image_layout: str = "separate",
) -> Dict[str, Any]:
    """Synchronous call with retries. `attach_images=False` sends a text-only
    request (no image parts) for every provider.

    `vllm_image_layout` (vllm provider only):
      - "separate": four image parts, one per option (server needs
        --limit-mm-per-prompt '{"image":4}').
      - "grid": one 2x2 labeled JPEG grid (server needs image:1). Returns:
        {ok, raw_output, elapsed_ms, attempts, response_id, error, debug}
    """
    provider = client["provider"]
    last_error = ""
    last_debug: Dict[str, Any] = {}

    for attempt in range(1, max(1, max_retries) + 1):
        try:
            started = time.perf_counter()
            if provider == "gemini":
                payload = _gemini_payload(model, prompt_mode, system_prompt, user_text, shuffled_options, attach_images=attach_images)
                url = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{urllib.parse.quote(model, safe='')}:generateContent"
                )
                request = urllib.request.Request(
                    url=f"{url}?key={urllib.parse.quote(client['api_key'], safe='')}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=request_timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                raw_text = _extract_gemini_text(response_payload)
                response_id = clean_text(response_payload.get("responseId"))
                last_debug = {
                    "provider": "gemini",
                    "request_url": url,
                    "request_payload": payload,
                    "response_payload": response_payload,
                }
            elif provider == "anthropic":
                payload = _anthropic_payload(model, prompt_mode, system_prompt, user_text, shuffled_options, attach_images=attach_images)
                request = urllib.request.Request(
                    url="https://api.anthropic.com/v1/messages",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": client["api_key"],
                        "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=request_timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                raw_text = _extract_anthropic_text(response_payload)
                response_id = clean_text(response_payload.get("id"))
                last_debug = {
                    "provider": "anthropic",
                    "request_payload": payload,
                    "response_payload": response_payload,
                }
            elif provider == "vllm":
                payload = _openai_chat_payload(
                    model,
                    prompt_mode,
                    system_prompt,
                    user_text,
                    shuffled_options,
                    normalize_images_to_jpeg=True,
                    single_image_grid=(clean_text(vllm_image_layout).lower() == "grid"),
                    attach_images=attach_images,
                )
                url = f"{client['base_url']}/chat/completions"
                request = urllib.request.Request(
                    url=url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {client['api_key']}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=request_timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                raw_text = _extract_openai_chat_text(response_payload)
                response_id = clean_text(response_payload.get("id"))
                last_debug = {
                    "provider": "vllm",
                    "request_url": url,
                    "request_payload": payload,
                    "response_payload": response_payload,
                }
            else:  # openai
                input_messages = _responses_payload(system_prompt, user_text, shuffled_options, attach_images=attach_images)
                response = client["client"].responses.create(
                    model=model,
                    input=input_messages,
                    temperature=0,
                    max_output_tokens=max_output_tokens(prompt_mode, model),
                )
                raw_text = _extract_responses_text(response)
                response_id = clean_text(getattr(response, "id", ""))
                last_debug = {"provider": "openai", "request_input": input_messages}
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return {
                "ok": True,
                "raw_output": raw_text,
                "elapsed_ms": elapsed_ms,
                "attempts": attempt,
                "response_id": response_id,
                "error": "",
                "debug": last_debug,
            }
        except Exception as exc:
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    body = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    body = ""
                last_debug = {
                    "provider": provider,
                    "http_status": getattr(exc, "code", None),
                    "error_body": body,
                }
                last_error = f"HTTPError: {exc.code} {body[:1000]}" if body else f"HTTPError: {exc}"
            else:
                last_error = f"{exc.__class__.__name__}: {exc}"
                last_debug = {
                    "provider": provider,
                    "exception": f"{exc.__class__.__name__}: {exc}",
                }
            if not _is_retryable(exc):
                break
            if attempt < max(1, max_retries):
                time.sleep(min(_MAX_RETRY_SLEEP_SECONDS, retry_sleep_seconds * 2 ** (attempt - 1)))

    return {
        "ok": False,
        "raw_output": "",
        "elapsed_ms": None,
        "attempts": attempt,
        "response_id": "",
        "error": last_error,
        "debug": last_debug,
    }
