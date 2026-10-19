#!/usr/bin/env python3
"""
Unified LLM provider layer for the FAM-Bench Task 1 runner.

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
- openai      OPENAI_API_KEY              (gpt-4.1 / gpt-4o / gpt-5 ...)
- anthropic   ANTHROPIC_API_KEY           (claude-* models, native SDK)
- gemini      GEMINI_API_KEY              (Google AI Studio, OpenAI-compat)
- deepseek    DEEPSEEK_API_KEY            (deepseek-chat / deepseek-reasoner)
- moonshot    MOONSHOT_API_KEY            (Kimi: moonshot-v1-*)
- qwen        DASHSCOPE_API_KEY           (Alibaba Qwen: qwen-max / qwen-plus)
- xai         XAI_API_KEY                 (grok-* models)
- mistral     MISTRAL_API_KEY             (mistral-large-latest, etc.)
- groq        GROQ_API_KEY                (Groq hosted Llama / Mixtral)
- vllm        VLLM_API_KEY                (local vLLM OpenAI-compatible server at
                                           VLLM_BASE_URL, default http://localhost:8000/v1)
"""
from __future__ import annotations

import json
import os
import re
import threading
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
        default_model="gemini-2.5-flash",
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
    ),
    "moonshot": ProviderSpec(
        name="moonshot",
        api_key_env="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.cn/v1",
        default_model="moonshot-v1-32k",
    ),
    "qwen": ProviderSpec(
        name="qwen",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
    ),
    "xai": ProviderSpec(
        name="xai",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        default_model="grok-2-latest",
    ),
    "mistral": ProviderSpec(
        name="mistral",
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-large-latest",
    ),
    "groq": ProviderSpec(
        name="groq",
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
    ),
    "vllm": ProviderSpec(
        name="vllm",
        api_key_env="VLLM_API_KEY",
        base_url="http://localhost:8000/v1",
        default_model="gemma-3-12b-it",
    ),
}


# Providers whose Chat Completions accept `response_format={"type":"json_object"}`.
# The rest fall back to prompt-only JSON instruction.
_JSON_MODE_PROVIDERS = {"openai", "deepseek", "moonshot", "qwen", "mistral", "groq", "gemini", "vllm"}


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
        ("deepseek", "deepseek"),
        ("moonshot", "moonshot"),
        ("kimi", "moonshot"),
        # Open-weight models (Qwen3-VL, Qwen3.5, Gemma 3/4) are served locally
        # through vLLM; pass --provider qwen to use Alibaba's hosted API instead.
        ("qwen", "vllm"),
        ("gemma", "vllm"),
        ("grok", "xai"),
        ("mistral", "mistral"),
        ("mixtral", "groq"),
        ("llama", "groq"),
        ("gpt", "openai"),
        ("o1", "openai"),
        ("o3", "openai"),
        ("o4", "openai"),
    ]
    for needle, key in inferred_order:
        if needle in lower:
            return PROVIDERS[key]
    return PROVIDERS["openai"]


# ---------------------------------------------------------------------------
# Client construction (lazy import; do not require all SDKs to be installed)
# ---------------------------------------------------------------------------

_CLIENT_CACHE: dict[str, Any] = {}

# Both SDKs retry transient failures themselves (connection errors, timeouts,
# 408/409/429/5xx) with exponential backoff and honour Retry-After, but only
# twice by default -- too few for a 16-way sweep that hits a rate limit. A call
# that still fails is not scored: the runner refuses to write results until
# every row has a real answer (see run_task1.py --resume).
_SDK_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "8"))


def build_client(provider: ProviderSpec) -> Any:
    if provider.name in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider.name]

    api_key = os.getenv(provider.api_key_env, "").strip()
    if not api_key and provider.name == "vllm":
        api_key = "EMPTY"  # a local vLLM server does not check the key
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
        client = Anthropic(api_key=api_key, max_retries=_SDK_MAX_RETRIES)
    else:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI SDK not installed. `pip install openai`"
            ) from exc
        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": _SDK_MAX_RETRIES}
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
# Per-model parameter adaptation
# ---------------------------------------------------------------------------
# Providers disagree about which Chat Completions parameters they accept, and
# the disagreement is per *model*, not per provider: gpt-4.1 takes `max_tokens`
# while gpt-5.x rejects it and wants `max_completion_tokens`. Instead of a
# hard-coded model table that goes stale, we learn each rejection from the
# API's own 400 on the first call and remember it for the rest of the process,
# so a 1500-sample sweep pays the probe once per worker rather than once per
# sample.
#
# The token cap is *renamed*, never silently dropped, so runs stay comparable
# across providers: a model that only knows `max_completion_tokens` still runs
# under the cap the caller asked for. The cap is given up only if the provider
# rejects it under both spellings, and that is recorded as "dropped" so the
# runner can say so in its metadata.

_ADAPT_LOCK = threading.Lock()
_PARAM_ADAPTATIONS: dict[str, dict[str, Any]] = {}

_MAX_PARAM_RETRIES = 4


def get_param_adaptations(provider_name: str, model: str) -> dict[str, Any]:
    """Report which parameters this (provider, model) actually accepted.

    Empty dict means everything the caller requested went through as-is.
    Otherwise keys are ``token_param`` ("max_completion_tokens" or "dropped"),
    ``temperature_dropped`` and ``response_format_dropped``. Runners record
    this so saved metadata reflects the config the run really executed under,
    not merely the one it asked for.
    """
    with _ADAPT_LOCK:
        return dict(_PARAM_ADAPTATIONS.get(f"{provider_name}:{model}", {}))


def _record_adaptation(provider_name: str, model: str, field: str, value: Any) -> None:
    with _ADAPT_LOCK:
        _PARAM_ADAPTATIONS.setdefault(f"{provider_name}:{model}", {})[field] = value


def _param_rejected(err: str, param: str) -> bool:
    return param in err and (
        "unsupported" in err or "unknown" in err or "not supported" in err
    )


def _adapt_kwargs_after_error(
    provider_name: str,
    model: str,
    kwargs: dict[str, Any],
    exc: Exception,
) -> bool:
    """Rename or strip one rejected parameter in `kwargs` (in place).

    Returns True if something changed and the call is worth retrying, False if
    the error is not a parameter-compatibility problem and should propagate.
    """
    err = str(exc).lower()

    # gpt-5.x / o-series: the completion cap moved to `max_completion_tokens`.
    # Rename rather than drop so the run keeps the cap it asked for.
    if "max_completion_tokens" in err and "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        _record_adaptation(provider_name, model, "token_param", "max_completion_tokens")
        return True

    # Some providers (certain Gemini models via OpenAI-compat) reject the cap
    # under either spelling; give it up and let the model's own default apply.
    if _param_rejected(err, "max_tokens") or _param_rejected(err, "max_completion_tokens"):
        for key in ("max_tokens", "max_completion_tokens"):
            if key in kwargs:
                kwargs.pop(key)
                _record_adaptation(provider_name, model, "token_param", "dropped")
                return True

    # Newer reasoning models only accept their default temperature.
    if _param_rejected(err, "temperature") and "temperature" in kwargs:
        kwargs.pop("temperature")
        _record_adaptation(provider_name, model, "temperature_dropped", True)
        return True

    if ("response_format" in err or "json_object" in err) and "response_format" in kwargs:
        kwargs.pop("response_format")
        _record_adaptation(provider_name, model, "response_format_dropped", True)
        return True

    return False


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

    # Apply whatever this model already told us it rejects, so only the first
    # call of a sweep pays for the discovery round trip.
    adapt = get_param_adaptations(provider.name, model)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if json_mode and provider.name in _JSON_MODE_PROVIDERS and not adapt.get("response_format_dropped"):
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens and adapt.get("token_param") != "dropped":
        kwargs[adapt.get("token_param", "max_tokens")] = max_tokens
    if not adapt.get("temperature_dropped"):
        kwargs["temperature"] = temperature

    last_exc: Optional[Exception] = None
    for _ in range(_MAX_PARAM_RETRIES):
        try:
            completion = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            last_exc = exc
            # Each adaptation fixes one parameter, so a model that rejects
            # several (gpt-5.x: max_tokens and temperature) needs a loop here,
            # not the single retry this used to do.
            if not _adapt_kwargs_after_error(provider.name, model, kwargs, exc):
                raise
    else:
        raise last_exc  # type: ignore[misc]

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

    kwargs: dict[str, Any] = {
        "model": model,
        "system": system,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_content}],
    }
    # anthropic>=1.0 removed `temperature` from the messages.create() signature
    # (TypeError if passed), but the API still honours it on Sonnet/Opus 4.6 and
    # older, so it goes in as raw JSON. Opus 4.7+ / Sonnet 5+ reject it with a
    # 400; drop it once and record that, same as the OpenAI-compatible path.
    if not get_param_adaptations("anthropic", model).get("temperature_dropped"):
        kwargs["extra_body"] = {"temperature": temperature}
    try:
        response = client.messages.create(**kwargs)
    except Exception as exc:
        if "extra_body" not in kwargs or "temperature" not in str(exc).lower():
            raise
        kwargs.pop("extra_body")
        _record_adaptation("anthropic", model, "temperature_dropped", True)
        response = client.messages.create(**kwargs)
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
