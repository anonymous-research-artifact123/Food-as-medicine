#!/usr/bin/env python3
"""
Unified LLM provider layer for Q2 benchmark experiments.

Goal
----
Let `run_task1.py` talk to many commercial LLMs through one
`chat(...)` call. Each provider is configured below; most providers are
OpenAI-compatible so we reuse the `openai` SDK with a different base_url
and API key. Anthropic gets its own native SDK because of message-format
differences.

Adding a new provider
---------------------
1. Add an entry to `PROVIDERS` with: env var for the API key, base_url
   (None for OpenAI), and optional default model.
2. If the provider follows the OpenAI Chat Completions contract (almost
   everyone does these days), nothing else is needed.
3. If it needs a custom client, extend `_NATIVE_PROVIDERS` and add a
   `_chat_<provider>` function below.

Provider keys recognized
------------------------
- openai      OPENAI_API_KEY              (GPT-5.4, native OpenAI endpoint)
- anthropic   ANTHROPIC_API_KEY           (Claude Sonnet 4.6, native SDK)
- gemini      GEMINI_API_KEY              (Gemini 2.5 Pro, OpenAI-compat)
- vllm        VLLM_API_KEY                (local vLLM: Qwen3-VL-8B-Instruct, Gemma-3-12B-IT)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderSpec:
    name: str
    api_key_env: str
    base_url: Optional[str]          # None => native OpenAI endpoint
    default_model: str
    sdk: str = "openai"              # "openai" | "anthropic"


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        default_model="gpt-5.4",
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        base_url=None,
        default_model="claude-sonnet-4-6",
        sdk="anthropic",
    ),
    "gemini": ProviderSpec(
        name="gemini",
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-2.5-pro",
    ),
    # Open-weight models (Qwen3-VL-8B-Instruct, Gemma-3-12B-IT) are served
    # locally via vLLM rather than through any vendor cloud API.
    "vllm": ProviderSpec(
        name="vllm",
        api_key_env="VLLM_API_KEY",
        base_url="http://localhost:8000/v1",
        default_model="gemma-3-12b-it",
    ),
}


# Providers whose Chat Completions accept `response_format={"type":"json_object"}`.
# The rest fall back to prompt-only JSON instruction.
_JSON_MODE_PROVIDERS = {"openai", "gemini", "vllm"}


def resolve_provider(model: str, explicit: Optional[str] = None) -> ProviderSpec:
    """Pick a provider from --provider, else infer from the model name."""
    if explicit:
        key = explicit.lower()
        if key not in PROVIDERS:
            raise ValueError(f"Unknown provider '{explicit}'. Known: {sorted(PROVIDERS)}")
        return PROVIDERS[key]

    lower = (model or "").lower()
    inferred_order = [
        ("claude", "anthropic"),
        ("gemini", "gemini"),
        ("gpt", "openai"),
        # Open-weight models are served locally through vLLM.
        ("qwen", "vllm"),
        ("gemma", "vllm"),
    ]
    for needle, key in inferred_order:
        if needle in lower:
            return PROVIDERS[key]
    return PROVIDERS["openai"]


# ---------------------------------------------------------------------------
# Client construction (lazy import; do not require all SDKs to be installed)
# ---------------------------------------------------------------------------

_CLIENT_CACHE: dict[str, Any] = {}


def build_client(provider: ProviderSpec) -> Any:
    if provider.name in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider.name]

    api_key = os.getenv(provider.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider '{provider.name}'. "
            f"Set environment variable {provider.api_key_env}."
        )

    if provider.sdk == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic SDK not installed. `pip install anthropic`"
            ) from exc
        client = Anthropic(api_key=api_key)
    else:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI SDK not installed. `pip install openai`"
            ) from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if provider.base_url:
            kwargs["base_url"] = provider.base_url
        # Per-provider base_url overrides (e.g. remote vLLM host).
        if provider.name == "vllm":
            override = os.getenv("VLLM_BASE_URL", "").strip()
            if override:
                kwargs["base_url"] = override
        client = OpenAI(**kwargs)

    _CLIENT_CACHE[provider.name] = client
    return client


# ---------------------------------------------------------------------------
# Unified chat entry point
# ---------------------------------------------------------------------------

def chat(
    *,
    provider: ProviderSpec,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    json_mode: bool = True,
    max_tokens: int = 4096,
    image_data_urls: Optional[list[str]] = None,
) -> str:
    """Run one chat completion. Returns the raw assistant text.

    ``image_data_urls`` (optional) is a list of ``data:<mime>;base64,...`` URLs
    that are passed as image content blocks alongside the text user prompt.
    Both the OpenAI-compatible path (incl. vLLM, Gemini) and the Anthropic
    native path consume them.
    """
    client = build_client(provider)

    if provider.sdk == "anthropic":
        return _chat_anthropic(
            client,
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            image_data_urls=image_data_urls,
        )

    return _chat_openai_compatible(
        client,
        provider=provider,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        json_mode=json_mode,
        max_tokens=max_tokens,
        image_data_urls=image_data_urls,
    )


def _chat_openai_compatible(
    client: Any,
    *,
    provider: ProviderSpec,
    model: str,
    system: str,
    user: str,
    temperature: float,
    json_mode: bool,
    max_tokens: int,
    image_data_urls: Optional[list[str]] = None,
) -> str:
    if image_data_urls:
        user_content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": data_url}}
            for data_url in image_data_urls
        ]
        if user:
            user_content.append({"type": "text", "text": user})
        user_payload: Any = user_content
    else:
        user_payload = user

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_payload},
    ]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if json_mode and provider.name in _JSON_MODE_PROVIDERS:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens:
        # Some providers reject `max_tokens` for newer models; tolerate it below.
        kwargs["max_tokens"] = max_tokens

    try:
        completion = client.chat.completions.create(**kwargs, temperature=temperature)
    except Exception as exc:
        err = str(exc).lower()
        # Newer OpenAI reasoning models reject custom temperature.
        if "temperature" in err and "unsupported" in err:
            kwargs.pop("temperature", None)
            completion = client.chat.completions.create(**kwargs)
        # Some providers (e.g. certain Gemini models via OpenAI-compat) reject max_tokens.
        elif "max_tokens" in err and ("unsupported" in err or "unknown" in err):
            kwargs.pop("max_tokens", None)
            completion = client.chat.completions.create(**kwargs, temperature=temperature)
        # Some providers/models reject json_object response_format.
        elif "response_format" in err or "json_object" in err:
            kwargs.pop("response_format", None)
            completion = client.chat.completions.create(**kwargs, temperature=temperature)
        else:
            raise

    choice = completion.choices[0]
    return getattr(choice.message, "content", "") or ""


_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.*)$", flags=re.DOTALL)


def _parse_data_url(data_url: str) -> tuple[str, str]:
    """Split a ``data:<mime>;base64,<payload>`` URL into (media_type, base64_data)."""
    match = _DATA_URL_RE.match(data_url)
    if not match:
        raise ValueError(f"Unsupported image URL (expected data:<mime>;base64,...): {data_url[:64]}...")
    return match.group("mime"), match.group("data")


def _chat_anthropic(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    image_data_urls: Optional[list[str]] = None,
) -> str:
    if image_data_urls:
        content_blocks: list[dict[str, Any]] = []
        for data_url in image_data_urls:
            media_type, b64 = _parse_data_url(data_url)
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            })
        if user:
            content_blocks.append({"type": "text", "text": user})
        user_content: Any = content_blocks
    else:
        user_content = user

    response = client.messages.create(
        model=model,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": user_content}],
    )
    parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Helpers exposed to runners
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def extract_first_json_object(text: str) -> Optional[str]:
    """Find the first balanced top-level JSON object in `text`.

    Some providers (Anthropic when not in JSON mode, or any model that
    emits reasoning before the JSON) wrap the JSON with extra prose.
    We greedily search for the outermost balanced braces.
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def list_available_providers() -> list[str]:
    """Providers whose API key is currently present in the environment."""
    available: list[str] = []
    for name, spec in PROVIDERS.items():
        if os.getenv(spec.api_key_env, "").strip():
            available.append(name)
    return available
