#!/usr/bin/env python3
"""Evaluate the image recipe-choice benchmark with a multimodal model."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]


def _ascii_safe(value: Any) -> str:
    return str(value or "").encode("ascii", "ignore").decode("ascii")

ANSWER_RE = re.compile(r"\b([A-D])\b", flags=re.I)
VALID_LETTERS = {"A", "B", "C", "D"}
_RESULTS_DIR = Path(__file__).resolve().parents[2] / "runs" / "task2"
DEFAULT_OUTPUT_CSV = _RESULTS_DIR / "image_recipe_choice_eval.csv"
DEFAULT_OUTPUT_JSONL = _RESULTS_DIR / "image_recipe_choice_eval.jsonl"
DEFAULT_OUTPUT_SUMMARY = _RESULTS_DIR / "image_recipe_choice_eval_summary.json"
DEFAULT_OUTPUT_DEBUG = _RESULTS_DIR / "image_recipe_choice_eval_debug.jsonl"
KI_DATA_DIR = Path(__file__).resolve().parent / "ki_data"
DISEASE_RULES_PATH = KI_DATA_DIR / "condition_rules.json"
NUTRITION_GUIDELINES_PATH = KI_DATA_DIR / "nutrition_guidelines.json"
PROMPT_MODES = {"baseline", "cot", "ki", "cot_ki"}
LEGACY_PROMPT_MODE_ALIASES = {"rationale": "cot", "ki_rationale": "cot_ki"}
PROMPT_MODE_CHOICES = tuple(sorted(PROMPT_MODES | set(LEGACY_PROMPT_MODE_ALIASES)))
_KI_CONTEXT_CACHE: Optional[Dict[str, Any]] = None


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_model() -> str:
    return (
        os.getenv("OPENAI_VISION_MODEL", "").strip()
        or os.getenv("GEMINI_VISION_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or "gpt-5.4"
    )


def _slugify(value: Any) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "value"


def _normalize_prompt_mode(prompt_mode: str) -> str:
    normalized = _clean_text(prompt_mode).lower() or "baseline"
    return LEGACY_PROMPT_MODE_ALIASES.get(normalized, normalized)


def _condition_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("’", "").replace("'", "")
    text = re.sub(r"[()/|]+", " ", text)
    text = re.sub(r"[^a-z0-9:+\-\s]", " ", text)
    return " ".join(text.split())


def _resolve_default_output_paths(
    *,
    benchmark_path: Path,
    model: str,
    provider: str,
    task: str,
    prompt_mode: str,
    csv_output_path: Path,
    jsonl_output_path: Path,
    summary_output_path: Path,
    debug_output_path: Path,
) -> tuple[Path, Path, Path, Path]:
    benchmark_name = _slugify(benchmark_path.stem)
    model_name = _slugify(model)
    provider_name = _slugify(provider) if provider else ""
    model_stem = f"{provider_name}_{model_name}" if provider_name else model_name
    if task == "rank":
        model_stem = f"{model_stem}_ranking"
    elif task == "decision_rationale":
        model_stem = f"{model_stem}_decision_rationale"
    if prompt_mode != "baseline":
        model_stem = f"{model_stem}_{_slugify(prompt_mode)}"
    stamp = _utc_now_stamp()
    stem = f"{benchmark_name}__{model_stem}__{stamp}"
    results_dir = DEFAULT_OUTPUT_CSV.parent

    resolved_csv = (
        results_dir / f"{stem}.csv"
        if csv_output_path.resolve() == DEFAULT_OUTPUT_CSV.resolve()
        else csv_output_path
    )
    resolved_jsonl = (
        results_dir / f"{stem}.jsonl"
        if jsonl_output_path.resolve() == DEFAULT_OUTPUT_JSONL.resolve()
        else jsonl_output_path
    )
    resolved_summary = (
        results_dir / f"{stem}_summary.json"
        if summary_output_path.resolve() == DEFAULT_OUTPUT_SUMMARY.resolve()
        else summary_output_path
    )
    resolved_debug = (
        results_dir / f"{stem}_debug.jsonl"
        if debug_output_path.resolve() == DEFAULT_OUTPUT_DEBUG.resolve()
        else debug_output_path
    )
    return resolved_csv, resolved_jsonl, resolved_summary, resolved_debug


def _load_rag_context() -> Dict[str, Any]:
    global _KI_CONTEXT_CACHE
    if _KI_CONTEXT_CACHE is not None:
        return _KI_CONTEXT_CACHE
    disease_payload = json.loads(DISEASE_RULES_PATH.read_text(encoding="utf-8"))
    guideline_payload = json.loads(NUTRITION_GUIDELINES_PATH.read_text(encoding="utf-8"))
    condition_rules: Dict[str, Dict[str, Any]] = {}
    for item in disease_payload.get("conditions") or []:
        if not isinstance(item, dict):
            continue
        key = _condition_key(item.get("value"))
        if key:
            condition_rules[key] = item
    alias_map = {
        _condition_key(alias): _condition_key(target)
        for alias, target in (disease_payload.get("alias_map") or {}).items()
        if _condition_key(alias) and _condition_key(target)
    }
    guideline_rows_by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in guideline_payload.get("rules") or []:
        if not isinstance(row, dict):
            continue
        key = _condition_key(row.get("condition_key") or row.get("condition"))
        if key:
            guideline_rows_by_condition[key].append(row)
    _KI_CONTEXT_CACHE = {
        "condition_rules": condition_rules,
        "alias_map": alias_map,
        "guideline_rows_by_condition": guideline_rows_by_condition,
    }
    return _KI_CONTEXT_CACHE


def _compact_items(values: Sequence[Any], limit: int = 8) -> str:
    items = [_clean_text(item) for item in values if _clean_text(item)]
    if not items:
        return "none listed"
    clipped = items[:limit]
    suffix = f" (+{len(items) - limit} more)" if len(items) > limit else ""
    return ", ".join(clipped) + suffix


def _format_guideline_row(row: Dict[str, Any]) -> str:
    nutrient = _clean_text(row.get("nutrient"))
    target_type = _clean_text(row.get("target_type"))
    min_value = row.get("min")
    max_value = row.get("max")
    unit = _clean_text(row.get("unit"))
    basis = _clean_text(row.get("basis"))
    population = _clean_text(row.get("population"))
    if max_value not in (None, ""):
        target = f"{target_type} {max_value:g} {unit}".strip()
    elif min_value not in (None, ""):
        target = f"{target_type} {min_value:g} {unit}".strip()
    else:
        target = target_type
    return " ".join(part for part in [nutrient, target, basis, f"({population})" if population else ""] if part)


def _rag_context_for_question(question: Dict[str, Any]) -> str:
    context = _load_rag_context()
    prompt = question.get("prompt") or {}
    conditions = [_condition_key(item) for item in (prompt.get("conditions") or []) if _condition_key(item)]
    age_group = _clean_text(prompt.get("age_group"))
    lines = [
        "Disease-nutrition rule context from the app's generated rule artifacts:",
        "Use these rules to judge health-condition fit. Do not ignore the images or ingredient prompt.",
    ]
    for condition in conditions:
        target = context["alias_map"].get(condition, condition)
        rule = context["condition_rules"].get(target)
        if not rule:
            continue
        label = _clean_text(rule.get("label") or target)
        lines.append(f"- {label}:")
        lines.append(f"  Favor: {_compact_items(rule.get('beneficial_raw') or rule.get('beneficial') or [])}.")
        lines.append(f"  Limit: {_compact_items(rule.get('limit_raw') or rule.get('limit') or [])}.")
        lines.append(f"  Avoid: {_compact_items(rule.get('avoid_raw') or rule.get('avoid') or [])}.")
        lines.append(f"  Helpful tags: {_compact_items(rule.get('dietary_tag_hints') or [])}.")
        guideline_keys = [target] + [_condition_key(item) for item in (rule.get("guideline_aliases") or [])]
        guideline_rows: List[Dict[str, Any]] = []
        for key in guideline_keys:
            guideline_rows.extend(context["guideline_rows_by_condition"].get(key, []))
        compact_guidelines = [_format_guideline_row(row) for row in guideline_rows[:6]]
        if compact_guidelines:
            lines.append(f"  Nutrient targets: {'; '.join(compact_guidelines)}.")
    general_rows = context["guideline_rows_by_condition"].get("general", [])
    age_rows = [
        row
        for row in general_rows
        if not age_group or age_group.replace("-", "") in _clean_text(row.get("population_key")).replace("_", "")
    ][:4]
    if age_rows:
        lines.append(f"- General age-context targets: {'; '.join(_format_guideline_row(row) for row in age_rows)}.")
    return "\n".join(lines)


def _option_ingredients_block(question: Dict[str, Any]) -> str:
    lines = [
        "Recipe ingredient lists for the four options:",
        "Use these ingredient lists together with the images. Do not assume an ingredient is absent just because it is not visually obvious.",
    ]
    for option in _iter_image_options(question):
        letter = _clean_text(option.get("letter")).upper()
        title = _clean_text(option.get("title"))
        source = option.get("full_ingredients") or option.get("ingredients") or option.get("key_ingredients") or []
        ingredients = [_clean_text(item) for item in source if _clean_text(item)]
        ingredient_text = "; ".join(ingredients) if ingredients else "No ingredient list available."
        title_text = f" ({title})" if title else ""
        lines.append(f"- Option {letter}{title_text}: {ingredient_text}")
    return "\n".join(lines)


def _text_only_options_block(question: Dict[str, Any], option_fields: str) -> str:
    option_fields = _clean_text(option_fields).lower() or "title_ingredients"
    lines = ["Text-only recipe options:"]
    for option in _iter_image_options(question):
        letter = _clean_text(option.get("letter")).upper()
        title = _clean_text(option.get("title"))
        source = option.get("full_ingredients") or option.get("ingredients") or option.get("key_ingredients") or []
        ingredients = [_clean_text(item) for item in source if _clean_text(item)]
        ingredient_text = "; ".join(ingredients) if ingredients else "No ingredient list available."
        if option_fields == "ingredients_only":
            lines.append(f"- Option {letter}: {ingredient_text}")
        else:
            title_text = title or "Untitled recipe"
            lines.append(f"- Option {letter}: {title_text}. Ingredients: {ingredient_text}")
    return "\n".join(lines)


def _build_prompt(
    question: Dict[str, Any],
    task: str = "answer",
    prompt_mode: str = "baseline",
    include_recipe_ingredients: bool = False,
    text_only: bool = False,
    text_only_option_fields: str = "title_ingredients",
) -> str:
    prompt_mode = _normalize_prompt_mode(prompt_mode)
    prompt = question.get("prompt") or {}
    condition_labels = [_clean_text(item) for item in (prompt.get("condition_labels") or []) if _clean_text(item)]
    available_ingredients = [
        _clean_text(item) for item in (prompt.get("available_ingredients") or []) if _clean_text(item)
    ]
    evidence_name = "recipe text options" if text_only else "recipe images"
    base = (
        f"You are evaluating {evidence_name} for a food-as-medicine benchmark.\n"
        f"Judge only from the provided prompt, {evidence_name}, and any supplied reference rules.\n"
        "Prioritize clinical safety: avoiding clearly harmful foods is more important than selecting an appealing dish.\n"
        "When citing ingredients, cite only ingredients that are present in the provided prompt evidence.\n\n"
        f"Health condition(s): {', '.join(condition_labels) or 'None specified'}\n"
        f"Available ingredient(s): {', '.join(available_ingredients) or 'None specified'}\n\n"
        f"You will see four recipe {'text options' if text_only else 'images'} labeled A, B, C, and D.\n"
    )
    if text_only:
        base += f"\n{_text_only_options_block(question, text_only_option_fields)}\n"
    elif include_recipe_ingredients:
        base += f"\n{_option_ingredients_block(question)}\n"
    context_block = ""
    if prompt_mode in {"ki", "cot_ki"}:
        context_block = f"\n\n{_rag_context_for_question(question)}\n"
    uses_cot = prompt_mode in {"cot", "cot_ki"}
    evidence_instruction = (
        "\nFor each option, explicitly check condition fit, avoid/limit concerns, ingredient match, "
        "and overall basis before finalizing. Keep this concise and structured in JSON.\n"
        if uses_cot
        else ""
    )
    json_requirement = "\nReturn only valid JSON. Do not include markdown, prose, or code fences.\n"
    if task == "decision_rationale":
        evidence_schema = ""
        if uses_cot:
            evidence_schema = (
                ' Include "evidence_checks" inside each option object with keys '
                '"condition_fit", "avoid_limit_concern", "ingredient_match", and "overall_basis".'
            )
        return (
            base
            + context_block
            + "For each option, decide whether it should be recommended for the health condition(s).\n"
            + evidence_instruction
            + json_requirement
            + "Use this schema for each of A, B, C, and D:\n"
            + '{"option_decisions": {"A": {"decision": "recommend|not_recommend", '
            + '"risk_level": "low|moderate|high", '
            + '"rationale_ingredients": [{"condition": "condition name", '
            + '"supporting_ingredients": ["ingredient"], '
            + '"concerning_ingredients": ["ingredient"]}]}, '
            + '"B": {"same fields": "..."}, "C": {"same fields": "..."}, "D": {"same fields": "..."}}}\n'
            + evidence_schema
            + "\nEach option must contain a decision, risk_level, and rationale_ingredients."
        )
    if task == "rank":
        basis_schema = (
            ', "ranking_basis": {"A": "short basis", "B": "short basis", "C": "short basis", "D": "short basis"}'
            if uses_cot
            else ""
        )
        return (
            base
            + context_block
            + f"Rank all four recipe {'text options' if text_only else 'images'} from best to worst for the health condition(s) and available ingredients.\n\n"
            + evidence_instruction
            + json_requirement
            + "Use this exact schema:\n"
            + '{"ranking": ["best", "second", "third", "worst"], "best_option": "A|B|C|D"'
            + basis_schema
            + "}\n"
            + "The ranking array must contain each of A, B, C, and D exactly once."
        )
    silent_instruction = (
        "\nBefore finalizing, silently check condition fit, avoid/limit foods, ingredient coverage, "
        "and obvious nutrition suitability. Do not output the checklist or explanation.\n"
        if uses_cot
        else ""
    )
    return (
        base
        + context_block
        + f"Choose the one {'recipe option' if text_only else 'image'} that best fits the health condition(s) and available ingredients.\n\n"
        + silent_instruction
        + "Output requirement:\n"
        "- Return exactly one capital letter\n"
        "- Allowed outputs: A, B, C, D\n"
        "- Do not output words, punctuation, or explanation"
    )


def _question_condition_key(question: Dict[str, Any]) -> str:
    prompt = question.get("prompt") or {}
    labels = [_clean_text(item) for item in (prompt.get("condition_labels") or []) if _clean_text(item)]
    return " + ".join(labels)


def _option_map(options: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        _clean_text(option.get("letter")).upper(): option
        for option in options
        if _clean_text(option.get("letter"))
    }


def _parse_letter(raw_text: str) -> str:
    text = _clean_text(raw_text).upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).upper()
    return ""


def _parse_ranking(raw_text: str) -> List[str]:
    letters = [match.upper() for match in re.findall(r"[A-D]", _clean_text(raw_text).upper())]
    ranking: List[str] = []
    for letter in letters:
        if letter not in ranking:
            ranking.append(letter)
    if len(ranking) == 4 and set(ranking) == VALID_LETTERS:
        return ranking
    return []


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return {}
                return payload if isinstance(payload, dict) else {}
    return {}


def _parse_json_ranking(raw_text: str) -> List[str]:
    payload = _extract_json_object(raw_text)
    ranking = payload.get("ranking") if isinstance(payload, dict) else None
    if isinstance(ranking, list):
        letters = [_clean_text(item).upper() for item in ranking]
    else:
        letters = []
    if len(letters) == 4 and set(letters) == VALID_LETTERS:
        return letters
    best = _clean_text(payload.get("best_option")).upper() if isinstance(payload, dict) else ""
    fallback = _parse_ranking(raw_text)
    if best in VALID_LETTERS and len(fallback) == 4 and fallback[0] != best:
        fallback = [best] + [letter for letter in fallback if letter != best]
    return fallback


def _parse_decision_payload(raw_text: str) -> Dict[str, Any]:
    payload = _extract_json_object(raw_text)
    decisions = payload.get("option_decisions") if isinstance(payload, dict) else None
    if not isinstance(decisions, dict):
        return {}
    parsed: Dict[str, Any] = {}
    for letter in sorted(VALID_LETTERS):
        item = decisions.get(letter) or decisions.get(letter.lower())
        if not isinstance(item, dict):
            return {}
        decision = _clean_text(item.get("decision")).lower().replace("-", "_").replace(" ", "_")
        if decision not in {"recommend", "not_recommend"}:
            return {}
        risk_level = _clean_text(item.get("risk_level")).lower()
        if risk_level not in {"low", "moderate", "high"}:
            risk_level = "moderate"
        parsed[letter] = {
            **item,
            "decision": decision,
            "risk_level": risk_level,
        }
    return parsed


def _iter_image_options(question: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for option in question.get("options") or []:
        if option.get("is_abstention"):
            continue
        image_url = _clean_text(option.get("image_url"))
        if not image_url:
            raise ValueError(
                f"{_clean_text(question.get('question_id'))}: option "
                f"{_clean_text(option.get('letter')).upper()} is missing image_url."
            )
        yield option


def _jpeg_data_url_for_provider(option: Dict[str, Any]) -> str:
    """Return an RGB JPEG data URL for providers that reject otherwise-valid image encodings."""
    mime_type, payload = _normalized_jpeg_payload_for_provider(option)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _image_grid_data_url_for_provider(question: Dict[str, Any]) -> str:
    """Return one labeled 2x2 JPEG grid for providers that allow only one image."""
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except Exception as exc:
        raise RuntimeError(
            "Pillow is required for vLLM/Gemma single-image grid mode. Install with `pip install pillow`."
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

    for index, option in enumerate(_iter_image_options(question)):
        if index >= 4:
            break
        letter = _clean_text(option.get("letter")).upper() or chr(ord("A") + index)
        _mime_type, payload = _image_payload(option)
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


def _image_input_reference(option: Dict[str, Any]) -> str:
    cached_path = _clean_text(option.get("cached_image_path"))
    if cached_path:
        path = Path(cached_path)
        if not path.is_absolute():
            path = (ROOT / cached_path).resolve()
        if path.exists() and path.is_file():
            mime_type, _ = mimetypes.guess_type(str(path))
            mime_type = mime_type or "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"
    image_url = _clean_text(option.get("image_url"))
    if not image_url:
        raise ValueError(f"Option {_clean_text(option.get('letter')).upper()} is missing image_url.")
    return image_url


def _image_url_from_option(option: Dict[str, Any]) -> str:
    image_url = _clean_text(option.get("image_url"))
    if not image_url:
        raise ValueError(f"Option {_clean_text(option.get('letter')).upper()} is missing image_url.")
    return image_url


def _max_output_tokens(task: str, prompt_mode: str, model: str = "") -> int:
    task = _clean_text(task).lower()
    prompt_mode = _normalize_prompt_mode(prompt_mode)
    normalized_model = _clean_text(model).lower()
    if normalized_model.startswith("gemini-2.5-pro"):
        if task == "decision_rationale":
            return 32768
        if task == "rank":
            return 32768
        return 4096
    if "gemma" in normalized_model:
        if task == "decision_rationale":
            return 4096
        if task == "rank":
            return 2048
        return 512
    if task == "decision_rationale":
        return 2200 if prompt_mode in {"cot", "cot_ki"} else 1600
    if task == "rank":
        return 900 if prompt_mode in {"cot", "cot_ki"} else 400
    return 16


def _model_provider(model: str, provider_override: str = "") -> str:
    override = _clean_text(provider_override).lower()
    if override:
        return override
    normalized = _clean_text(model).lower()
    if normalized.startswith("gemini"):
        return "gemini"
    if normalized.startswith("claude"):
        return "anthropic"
    if normalized.startswith("qwen"):
        return "vllm"
    if normalized.startswith("gemma"):
        return "vllm"
    return "openai"


def _image_payload(option: Dict[str, Any], image_source: str = "cached_first") -> Tuple[str, bytes]:
    def _sniff_mime_type(payload: bytes, fallback: str = "") -> str:
        if payload.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if payload[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
            return "image/webp"
        if len(payload) >= 12 and payload[4:12] == b"ftypavif":
            return "image/avif"
        guessed, _ = mimetypes.guess_type(fallback)
        if guessed and guessed.startswith("image/"):
            return guessed
        return "image/jpeg"

    image_source = _clean_text(image_source).lower() or "cached_first"
    if image_source not in {"cached_first", "url_only"}:
        raise ValueError("Unsupported image_source. Use cached_first or url_only.")

    if image_source == "cached_first":
        cached_path = _clean_text(option.get("cached_image_path"))
        if cached_path:
            path = Path(cached_path)
            if not path.is_absolute():
                path = (ROOT / cached_path).resolve()
            if path.exists() and path.is_file():
                payload = path.read_bytes()
                mime_type, _ = mimetypes.guess_type(str(path))
                sniffed_mime_type = _sniff_mime_type(payload, str(path))
                if (
                    not mime_type
                    or not mime_type.startswith("image/")
                    or mime_type == "image/jpeg"
                    and sniffed_mime_type not in {"image/jpeg", "image/jpg"}
                ):
                    mime_type = sniffed_mime_type
                return mime_type, payload

    image_url = _image_url_from_option(option)
    request = urllib.request.Request(
        image_url,
        headers={"User-Agent": "FAMBench-Eval/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        mime_type = response.headers.get_content_type() or "image/jpeg"
        payload = response.read()
    if not mime_type.startswith("image/") or mime_type == "application/octet-stream":
        mime_type = _sniff_mime_type(payload, image_url)
    return mime_type, payload


def _normalized_jpeg_payload_for_provider(option: Dict[str, Any], image_source: str = "cached_first") -> Tuple[str, bytes]:
    """Return JPEG bytes for providers with stricter image decoders."""
    try:
        from PIL import Image, ImageOps
        try:
            import pillow_avif  # type: ignore  # noqa: F401
        except Exception:
            try:
                import pillow_avif_plugin  # type: ignore  # noqa: F401
            except Exception:
                pass
    except Exception as exc:
        mime_type, payload = _image_payload(option, image_source=image_source)
        if mime_type in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
            return mime_type, payload
        raise RuntimeError(
            "Pillow with AVIF support is required to convert cached image "
            f"{_clean_text(option.get('cached_image_path')) or _clean_text(option.get('image_url'))} "
            f"from {mime_type} to JPEG. Install with: pip install pillow pillow-avif-plugin"
        ) from exc

    mime_type, payload = _image_payload(option, image_source=image_source)
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
        return "image/jpeg", output.getvalue()
    except Exception as exc:
        if mime_type in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
            return mime_type, payload
        raise RuntimeError(
            "Could not convert cached image "
            f"{_clean_text(option.get('cached_image_path')) or _clean_text(option.get('image_url'))} "
            f"from {mime_type} to JPEG. Install AVIF support with: pip install pillow pillow-avif-plugin"
        ) from exc


def _responses_payload(
    question: Dict[str, Any],
    task: str,
    prompt_mode: str,
    include_recipe_ingredients: bool,
    text_only: bool = False,
    text_only_option_fields: str = "title_ingredients",
    image_source: str = "cached_first",
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": _build_prompt(
                question,
                task,
                prompt_mode,
                include_recipe_ingredients,
                text_only=text_only,
                text_only_option_fields=text_only_option_fields,
            ),
        }
    ]
    if text_only:
        return [{"role": "user", "content": content}]
    for option in _iter_image_options(question):
        letter = _clean_text(option.get("letter")).upper()
        content.append({"type": "input_text", "text": f"Option {letter}:"})
        if _clean_text(image_source).lower() == "url_only":
            content.append({"type": "input_image", "image_url": _image_url_from_option(option)})
        else:
            content.append({"type": "input_image", "image_url": _jpeg_data_url_for_provider(option)})
    return [{"role": "user", "content": content}]


def _gemini_payload(
    question: Dict[str, Any],
    model: str,
    task: str,
    prompt_mode: str,
    include_recipe_ingredients: bool,
    text_only: bool = False,
    text_only_option_fields: str = "title_ingredients",
    image_source: str = "cached_first",
) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = [
        {
            "text": _build_prompt(
                question,
                task,
                prompt_mode,
                include_recipe_ingredients,
                text_only=text_only,
                text_only_option_fields=text_only_option_fields,
            )
        }
    ]
    if not text_only:
        for option in _iter_image_options(question):
            letter = _clean_text(option.get("letter")).upper()
            mime_type, payload = _image_payload(option, image_source=image_source)
            parts.append({"text": f"Option {letter}:"})
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(payload).decode("ascii"),
                    }
                }
            )
    max_output_tokens = max(_max_output_tokens(task, prompt_mode, model), 256)
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "topP": 1,
            "topK": 1,
            # Gemini models may spend part of this budget on internal reasoning.
            # Pro needs a larger ceiling so the final answer token is emitted.
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "text/plain",
        },
    }


def _anthropic_payload(
    question: Dict[str, Any],
    model: str,
    task: str,
    prompt_mode: str,
    include_recipe_ingredients: bool,
    text_only: bool = False,
    text_only_option_fields: str = "title_ingredients",
    image_source: str = "cached_first",
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": _build_prompt(
                question,
                task,
                prompt_mode,
                include_recipe_ingredients,
                text_only=text_only,
                text_only_option_fields=text_only_option_fields,
            ),
        }
    ]
    if not text_only:
        for option in _iter_image_options(question):
            letter = _clean_text(option.get("letter")).upper()
            mime_type, payload = _normalized_jpeg_payload_for_provider(option, image_source=image_source)
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
        "max_tokens": _max_output_tokens(task, prompt_mode, model),
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }


def _openai_chat_payload(
    question: Dict[str, Any],
    model: str,
    task: str,
    prompt_mode: str,
    include_recipe_ingredients: bool,
    normalize_images_to_jpeg: bool = False,
    single_image_grid: bool = False,
    text_only: bool = False,
    text_only_option_fields: str = "title_ingredients",
    image_source: str = "cached_first",
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": _build_prompt(
                question,
                task,
                prompt_mode,
                include_recipe_ingredients,
                text_only=text_only,
                text_only_option_fields=text_only_option_fields,
            ),
        }
    ]
    if text_only:
        return {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": _max_output_tokens(task, prompt_mode, model),
        }
    if single_image_grid:
        content.append(
            {
                "type": "text",
                "text": "The single attached image is a 2x2 grid labeled Option A, Option B, Option C, and Option D.",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_grid_data_url_for_provider(question),
                },
            }
        )
        return {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": _max_output_tokens(task, prompt_mode, model),
        }
    for option in _iter_image_options(question):
        letter = _clean_text(option.get("letter")).upper()
        content.append({"type": "text", "text": f"Option {letter}:"})
        if _clean_text(image_source).lower() == "url_only":
            image_url = _image_url_from_option(option)
        else:
            image_url = _jpeg_data_url_for_provider(option) if normalize_images_to_jpeg else _image_input_reference(option)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        )
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": _max_output_tokens(task, prompt_mode, model),
    }


def _extract_output_text(response: Any) -> str:
    text = _clean_text(getattr(response, "output_text", None))
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
                text_value = _clean_text(content.get("text"))
                if text_value:
                    return text_value
    return ""


def _extract_gemini_text(payload: Dict[str, Any]) -> str:
    def _walk(node: Any) -> Optional[str]:
        if isinstance(node, dict):
            text = _clean_text(node.get("text"))
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
            text = _clean_text(part.get("text"))
            if text:
                return text
    return _walk(payload) or ""


def _extract_anthropic_text(payload: Dict[str, Any]) -> str:
    for item in payload.get("content") or []:
        if isinstance(item, dict):
            text = _clean_text(item.get("text"))
            if text:
                return text
    return ""


def _extract_openai_chat_text(payload: Dict[str, Any]) -> str:
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        text = _clean_text(message.get("content"))
        if text:
            return text
    return ""


def _create_client(model: str, provider_override: str = "") -> Dict[str, Any]:
    provider = _model_provider(model, provider_override)
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
        raise SystemExit(f"Unsupported provider '{provider}'. Use openai, gemini, anthropic, or vllm.")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for OpenAI models.")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise SystemExit(f"Failed to import OpenAI client: {exc}") from exc
    return {"provider": provider, "client": OpenAI(api_key=api_key)}


def _call_model(
    client: Dict[str, Any],
    *,
    model: str,
    question: Dict[str, Any],
    task: str,
    prompt_mode: str,
    include_recipe_ingredients: bool,
    text_only: bool,
    text_only_option_fields: str,
    image_source: str,
    max_retries: int,
    retry_sleep_seconds: float,
    request_timeout_seconds: float,
) -> Dict[str, Any]:
    last_error = ""
    provider = client["provider"]
    last_debug: Dict[str, Any] = {}
    for attempt in range(1, max_retries + 1):
        try:
            started = time.perf_counter()
            if provider == "gemini":
                payload = _gemini_payload(
                    question,
                    model,
                    task,
                    prompt_mode,
                    include_recipe_ingredients,
                    text_only=text_only,
                    text_only_option_fields=text_only_option_fields,
                    image_source=image_source,
                )
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
                response_id = _clean_text(response_payload.get("responseId"))
                last_debug = {
                    "provider": "gemini",
                    "request_url": url,
                    "request_payload": payload,
                    "response_payload": response_payload,
                }
            elif provider == "anthropic":
                payload = _anthropic_payload(
                    question,
                    model,
                    task,
                    prompt_mode,
                    include_recipe_ingredients,
                    text_only=text_only,
                    text_only_option_fields=text_only_option_fields,
                    image_source=image_source,
                )
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
                response_id = _clean_text(response_payload.get("id"))
                last_debug = {
                    "provider": "anthropic",
                    "request_payload": payload,
                    "response_payload": response_payload,
                }
            elif provider == "vllm":
                payload = _openai_chat_payload(
                    question,
                    model,
                    task,
                    prompt_mode,
                    include_recipe_ingredients,
                    normalize_images_to_jpeg=True,
                    single_image_grid=True,
                    text_only=text_only,
                    text_only_option_fields=text_only_option_fields,
                    image_source=image_source,
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
                response_id = _clean_text(response_payload.get("id"))
                last_debug = {
                    "provider": "vllm",
                    "request_url": url,
                    "request_payload": payload,
                    "response_payload": response_payload,
                }
            else:
                response = client["client"].responses.create(
                    model=model,
                    input=_responses_payload(
                        question,
                        task,
                        prompt_mode,
                        include_recipe_ingredients,
                        text_only=text_only,
                        text_only_option_fields=text_only_option_fields,
                        image_source=image_source,
                    ),
                )
                raw_text = _extract_output_text(response)
                response_id = _clean_text(getattr(response, "id", ""))
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            decision_payload = _parse_decision_payload(raw_text) if task == "decision_rationale" else {}
            ranking = _parse_json_ranking(raw_text) if task == "rank" else []
            letter = ranking[0] if task == "rank" and ranking else _parse_letter(raw_text)
            return {
                "ok": True,
                "raw_output": raw_text,
                "predicted_letter": letter,
                "predicted_ranking": ranking,
                "decision_payload": decision_payload,
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
                if body:
                    last_error = f"HTTPError: {exc.code} {body[:1000]}"
                else:
                    last_error = f"HTTPError: {exc}"
            else:
                last_error = f"{exc.__class__.__name__}: {exc}"
                last_debug = {
                    "provider": provider,
                    "exception": f"{exc.__class__.__name__}: {exc}",
                }
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)
    return {
        "ok": False,
        "raw_output": "",
        "predicted_letter": "",
        "predicted_ranking": [],
        "decision_payload": {},
        "elapsed_ms": None,
        "attempts": max_retries,
        "response_id": "",
        "error": last_error,
        "debug": last_debug,
    }


def _format_percentage(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def _limit_questions(
    questions: Sequence[Dict[str, Any]],
    *,
    max_questions: Optional[int],
    question_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    wanted = {_clean_text(item).upper() for item in question_ids if _clean_text(item)}
    filtered = []
    for question in questions:
        qid = _clean_text(question.get("question_id")).upper()
        if wanted and qid not in wanted:
            continue
        filtered.append(question)
    if max_questions is not None:
        return filtered[: max(0, int(max_questions))]
    return filtered


def _load_existing_rows(
    jsonl_output_path: Path,
    *,
    selected_question_ids: set[str],
    model: str,
    provider: str,
    task: str,
    prompt_mode: str,
    parsed_only: bool = False,
) -> List[Dict[str, Any]]:
    if not jsonl_output_path.exists():
        return []
    rows_by_question_id: Dict[str, Dict[str, Any]] = {}
    with jsonl_output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            qid = _clean_text(row.get("question_id")).upper()
            if qid not in selected_question_ids:
                continue
            if _clean_text(row.get("model")) != model:
                continue
            if _clean_text(row.get("provider")) != provider:
                continue
            if _clean_text(row.get("task") or "answer").lower() != task:
                continue
            if _normalize_prompt_mode(_clean_text(row.get("prompt_mode") or "baseline")) != prompt_mode:
                continue
            if parsed_only and _clean_text(row.get("parse_status")) != "parsed":
                continue
            rows_by_question_id[qid] = row
    return list(rows_by_question_id.values())


def _normalize_ingredient(value: Any) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    stop_words = {"and", "or", "of", "with", "fresh", "cooked", "diced", "minced", "chopped"}
    return " ".join(part for part in text.split() if part not in stop_words)


def _soft_contains(candidate: str, targets: Sequence[Any]) -> bool:
    normalized = _normalize_ingredient(candidate)
    if not normalized:
        return False
    candidate_tokens = set(normalized.split())
    for target in targets:
        target_norm = _normalize_ingredient(target)
        if not target_norm:
            continue
        if normalized in target_norm or target_norm in normalized:
            return True
        target_tokens = set(target_norm.split())
        if candidate_tokens and target_tokens and len(candidate_tokens & target_tokens) / max(1, min(len(candidate_tokens), len(target_tokens))) >= 0.67:
            return True
    return False


def _iter_predicted_ingredients(decision_payload: Dict[str, Any], letter: str) -> Iterable[Tuple[str, str, str]]:
    item = decision_payload.get(letter) if isinstance(decision_payload, dict) else None
    if not isinstance(item, dict):
        return
    for rationale in item.get("rationale_ingredients") or []:
        if not isinstance(rationale, dict):
            continue
        condition = _condition_key(rationale.get("condition"))
        for ingredient in rationale.get("supporting_ingredients") or []:
            if _clean_text(ingredient):
                yield condition, _clean_text(ingredient), "supports"
        for ingredient in rationale.get("concerning_ingredients") or []:
            if _clean_text(ingredient):
                yield condition, _clean_text(ingredient), "concern"


def _condition_attribution_match(
    *,
    condition: str,
    ingredient: str,
    polarity: str,
    targets: Sequence[Dict[str, Any]],
) -> bool:
    for target in targets:
        if not isinstance(target, dict):
            continue
        target_condition = _condition_key(target.get("condition"))
        target_polarity = _clean_text(target.get("polarity")).lower()
        if condition and target_condition and condition != target_condition:
            continue
        if target_polarity and polarity != target_polarity:
            continue
        if _soft_contains(ingredient, [target.get("term"), target.get("concept")]):
            return True
    return False


def _score_decision_rationale(question: Dict[str, Any], decision_payload: Dict[str, Any]) -> Dict[str, Any]:
    gold = question.get("gold") or {}
    gold_labels = gold.get("decision_labels") or {}
    option_by_letter = _option_map(question.get("options") or [])
    option_total = 0
    option_correct = 0
    false_recommend = 0
    false_recommend_cost = 0
    ingredient_mentions = 0
    ingredient_mentions_in_recipe = 0
    attribution_mentions = 0
    attribution_correct = 0

    for letter in sorted(VALID_LETTERS):
        predicted_item = decision_payload.get(letter) if isinstance(decision_payload, dict) else {}
        predicted_decision = _clean_text((predicted_item or {}).get("decision")).lower()
        gold_decision = _clean_text(gold_labels.get(letter)).lower()
        if gold_decision:
            option_total += 1
            option_correct += int(predicted_decision == gold_decision)
            if predicted_decision == "recommend" and gold_decision == "not_recommend":
                false_recommend += 1
                false_recommend_cost += int((gold.get("false_recommend_costs") or {}).get(letter) or 1)

        option = option_by_letter.get(letter) or {}
        targets = ((option.get("gold") or {}).get("rationale_targets") or {})
        allowed_mentions = targets.get("allowed_ingredient_mentions") or []
        attribution_targets = targets.get("condition_attribution_targets") or []
        for condition, ingredient, polarity in _iter_predicted_ingredients(decision_payload, letter):
            ingredient_mentions += 1
            if _soft_contains(ingredient, allowed_mentions):
                ingredient_mentions_in_recipe += 1
            attribution_mentions += 1
            if _condition_attribution_match(
                condition=condition,
                ingredient=ingredient,
                polarity=polarity,
                targets=attribution_targets,
            ):
                attribution_correct += 1

    return {
        "option_decision_total": option_total,
        "option_decision_correct": option_correct,
        "all_option_decisions_correct": int(option_total > 0 and option_correct == option_total),
        "false_recommend_count": false_recommend,
        "false_recommend_cost": false_recommend_cost,
        "ingredient_mentions": ingredient_mentions,
        "ingredient_mentions_in_recipe": ingredient_mentions_in_recipe,
        "condition_attribution_mentions": attribution_mentions,
        "condition_attribution_correct": attribution_correct,
    }


def _kendall_tau(predicted: Sequence[str], gold: Sequence[str]) -> float:
    pred = [item for item in predicted if item in VALID_LETTERS]
    target = [item for item in gold if item in VALID_LETTERS]
    if len(pred) != 4 or len(target) != 4 or set(pred) != VALID_LETTERS or set(target) != VALID_LETTERS:
        return 0.0
    pred_pos = {letter: index for index, letter in enumerate(pred)}
    gold_pos = {letter: index for index, letter in enumerate(target)}
    concordant = 0
    discordant = 0
    letters = sorted(VALID_LETTERS)
    for i, left in enumerate(letters):
        for right in letters[i + 1 :]:
            pred_order = pred_pos[left] - pred_pos[right]
            gold_order = gold_pos[left] - gold_pos[right]
            if pred_order * gold_order > 0:
                concordant += 1
            elif pred_order * gold_order < 0:
                discordant += 1
    return round((concordant - discordant) / 6.0, 6)


def _decision_inversion_rate(predicted: Sequence[str], decision_labels: Dict[str, Any]) -> float:
    ranking = [item for item in predicted if item in VALID_LETTERS]
    if len(ranking) != 4:
        return 0.0
    inversions = 0
    possible = 0
    for i, higher in enumerate(ranking):
        for lower in ranking[i + 1 :]:
            high_label = _clean_text(decision_labels.get(higher)).lower()
            low_label = _clean_text(decision_labels.get(lower)).lower()
            if {high_label, low_label} == {"recommend", "not_recommend"}:
                possible += 1
                if high_label == "not_recommend" and low_label == "recommend":
                    inversions += 1
    return round(inversions / possible, 6) if possible else 0.0


def _build_result_row(
    *,
    index: int,
    question: Dict[str, Any],
    model: str,
    provider: str,
    task: str,
    prompt_mode: str,
    result: Dict[str, Any],
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    qid = _clean_text(question.get("question_id")).upper()
    gold_letter = _clean_text((question.get("answer") or {}).get("letter")).upper()
    condition_key = _question_condition_key(question)
    option_by_letter = _option_map(question.get("options") or [])
    predicted_letter = _clean_text(result.get("predicted_letter")).upper()
    predicted_ranking = [
        _clean_text(item).upper()
        for item in (result.get("predicted_ranking") or [])
        if _clean_text(item).upper() in VALID_LETTERS
    ]
    decision_payload = result.get("decision_payload") or {}
    raw_output = _clean_text(result.get("raw_output"))

    if result.get("error"):
        parse_status = "request_error"
    elif task == "decision_rationale" and set(decision_payload) == VALID_LETTERS:
        parse_status = "parsed"
    elif task == "rank" and len(predicted_ranking) == 4 and set(predicted_ranking) == VALID_LETTERS:
        parse_status = "parsed"
    elif predicted_letter in {"A", "B", "C", "D"}:
        parse_status = "parsed"
    else:
        parse_status = "unparseable_output"

    decision_scores = _score_decision_rationale(question, decision_payload) if task == "decision_rationale" else {}
    gold_ranking = [
        _clean_text(item).upper()
        for item in ((question.get("gold") or {}).get("ranking") or [])
        if _clean_text(item).upper() in VALID_LETTERS
    ]
    if task == "rank":
        gold_rank = predicted_ranking.index(gold_letter) + 1 if gold_letter in predicted_ranking else 0
        top_prediction = predicted_ranking[0] if predicted_ranking else ""
        is_correct = gold_rank == 1
    elif task == "decision_rationale":
        gold_rank = 0
        top_prediction = ""
        is_correct = bool(decision_scores.get("all_option_decisions_correct"))
    else:
        gold_rank = 1 if predicted_letter == gold_letter else 0
        top_prediction = predicted_letter
        is_correct = bool(predicted_letter and predicted_letter == gold_letter)
    row = {
        "row_index": index,
        "question_id": qid,
        "task": task,
        "prompt_mode": prompt_mode,
        "condition_prompt": condition_key,
        "available_ingredients": ", ".join(
            _ascii_safe(item) for item in ((question.get("prompt") or {}).get("available_ingredients") or [])
        ),
        "gold_letter": gold_letter,
        "gold_ranking": " ".join(gold_ranking),
        "gold_title": _clean_text((option_by_letter.get(gold_letter) or {}).get("title")),
        "predicted_letter": predicted_letter,
        "predicted_title": _clean_text((option_by_letter.get(predicted_letter) or {}).get("title")),
        "predicted_ranking": " ".join(predicted_ranking),
        "top_prediction": top_prediction,
        "kendall_tau_gold": _kendall_tau(predicted_ranking, gold_ranking) if task == "rank" else 0.0,
        "ranking_decision_inversion_rate": _decision_inversion_rate(
            predicted_ranking,
            (question.get("gold") or {}).get("decision_labels") or {},
        )
        if task == "rank"
        else 0.0,
        "option_decisions_json": json.dumps(decision_payload, ensure_ascii=False, sort_keys=True)
        if task == "decision_rationale"
        else "",
        "option_decision_total": decision_scores.get("option_decision_total", 0),
        "option_decision_correct": decision_scores.get("option_decision_correct", 0),
        "all_option_decisions_correct": decision_scores.get("all_option_decisions_correct", 0),
        "false_recommend_count": decision_scores.get("false_recommend_count", 0),
        "false_recommend_cost": decision_scores.get("false_recommend_cost", 0),
        "ingredient_mentions": decision_scores.get("ingredient_mentions", 0),
        "ingredient_mentions_in_recipe": decision_scores.get("ingredient_mentions_in_recipe", 0),
        "condition_attribution_mentions": decision_scores.get("condition_attribution_mentions", 0),
        "condition_attribution_correct": decision_scores.get("condition_attribution_correct", 0),
        "gold_rank": gold_rank,
        "is_top2": int(0 < gold_rank <= 2),
        "is_top3": int(0 < gold_rank <= 3),
        "reciprocal_rank": round(1 / gold_rank, 6) if gold_rank else 0.0,
        "is_correct": int(is_correct),
        "parse_status": parse_status,
        "raw_output": raw_output,
        "response_id": _clean_text(result.get("response_id")),
        "attempts": result.get("attempts"),
        "elapsed_ms": result.get("elapsed_ms"),
        "error": _clean_text(result.get("error")),
        "model": model,
        "provider": provider,
    }
    debug_row = None
    if provider in {"gemini", "anthropic", "vllm"} and parse_status in {
        "request_error",
        "unparseable_output",
    }:
        debug_row = {
            "question_id": qid,
            "model": model,
            "provider": provider,
            "parse_status": parse_status,
            "raw_output": raw_output,
            "error": _clean_text(result.get("error")),
            "debug": result.get("debug") or {},
        }
    return row, debug_row


def _append_jsonl(path: Path, row: Dict[str, Any], lock: Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()


def _summarize_rows(
    *,
    rows: Sequence[Dict[str, Any]],
    questions: Sequence[Dict[str, Any]],
    benchmark_path: Path,
    metadata: Dict[str, Any],
    model: str,
    provider: str,
    task: str,
    prompt_mode: str,
    include_recipe_ingredients: bool,
    csv_output_path: Path,
    jsonl_output_path: Path,
    summary_output_path: Path,
    debug_output_path: Path,
    debug_rows: Sequence[Dict[str, Any]],
    concurrency: int,
    resume: bool,
    text_only: bool,
    text_only_option_fields: str,
    image_source: str,
) -> Dict[str, Any]:
    by_condition: Dict[str, Counter[str]] = defaultdict(Counter)
    prediction_distribution: Counter[str] = Counter()
    parse_status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    correct = 0
    answered = 0
    top2 = 0
    top3 = 0
    reciprocal_rank_sum = 0.0
    ranked = 0
    kendall_tau_sum = 0.0
    inversion_rate_sum = 0.0
    option_decision_total = 0
    option_decision_correct = 0
    all_option_correct = 0
    false_recommend_count = 0
    false_recommend_cost = 0
    ingredient_mentions = 0
    ingredient_mentions_in_recipe = 0
    attribution_mentions = 0
    attribution_correct = 0

    sorted_rows = sorted(rows, key=lambda item: int(item.get("row_index") or 0))
    for row in sorted_rows:
        parse_status = _clean_text(row.get("parse_status"))
        predicted_letter = _clean_text(row.get("top_prediction") or row.get("predicted_letter")).upper()
        condition_key = _clean_text(row.get("condition_prompt"))
        parse_status_counts[parse_status] += 1
        if parse_status == "parsed":
            prediction_distribution[predicted_letter] += 1
            answered += 1
            if task == "rank":
                ranked += 1
                top2 += int(row.get("is_top2") or 0)
                top3 += int(row.get("is_top3") or 0)
                reciprocal_rank_sum += float(row.get("reciprocal_rank") or 0.0)
                kendall_tau_sum += float(row.get("kendall_tau_gold") or 0.0)
                inversion_rate_sum += float(row.get("ranking_decision_inversion_rate") or 0.0)
        error = _clean_text(row.get("error"))
        if error:
            error_counts[error] += 1
        is_correct = int(row.get("is_correct") or 0)
        correct += is_correct
        if task == "decision_rationale":
            option_decision_total += int(row.get("option_decision_total") or 0)
            option_decision_correct += int(row.get("option_decision_correct") or 0)
            all_option_correct += int(row.get("all_option_decisions_correct") or 0)
            false_recommend_count += int(row.get("false_recommend_count") or 0)
            false_recommend_cost += int(row.get("false_recommend_cost") or 0)
            ingredient_mentions += int(row.get("ingredient_mentions") or 0)
            ingredient_mentions_in_recipe += int(row.get("ingredient_mentions_in_recipe") or 0)
            attribution_mentions += int(row.get("condition_attribution_mentions") or 0)
            attribution_correct += int(row.get("condition_attribution_correct") or 0)
        by_condition[condition_key]["questions"] += 1
        by_condition[condition_key]["correct"] += is_correct
        by_condition[condition_key]["answered"] += int(parse_status == "parsed")

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sorted_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted_rows)

    with jsonl_output_path.open("w", encoding="utf-8") as handle:
        for row in sorted_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    if debug_rows:
        with debug_output_path.open("w", encoding="utf-8") as handle:
            for row in debug_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    by_condition_summary = []
    for condition_key in sorted(by_condition):
        item = by_condition[condition_key]
        question_count = int(item.get("questions") or 0)
        by_condition_summary.append(
            {
                "condition_prompt": condition_key,
                "questions": question_count,
                "answered": int(item.get("answered") or 0),
                "correct": int(item.get("correct") or 0),
                "accuracy_pct": _format_percentage(int(item.get("correct") or 0), question_count),
            }
        )

    summary = {
        "benchmark_path": _display_path(benchmark_path),
        "benchmark_name": _clean_text(metadata.get("benchmark_name")),
        "benchmark_question_count": int(metadata.get("question_count") or len(questions)),
        "evaluated_question_count": len(sorted_rows),
        "selected_question_count": len(questions),
        "generated_at_utc": _utc_now_iso(),
        "model": model,
        "provider": provider,
        "task": task,
        "prompt_mode": prompt_mode,
        "include_recipe_ingredients": bool(include_recipe_ingredients),
        "text_only": bool(text_only),
        "text_only_option_fields": _clean_text(text_only_option_fields) if text_only else "",
        "image_source": _clean_text(image_source).lower() or "cached_first",
        "concurrency": concurrency,
        "resume": bool(resume),
        "accuracy": {
            "correct": correct,
            "evaluated": len(sorted_rows),
            "answered": answered,
            "accuracy_pct": _format_percentage(correct, len(sorted_rows)),
            "answer_rate_pct": _format_percentage(answered, len(sorted_rows)),
        },
        "ranking": {
            "top1_accuracy_pct": _format_percentage(correct, len(sorted_rows)),
            "top2_accuracy_pct": _format_percentage(top2, len(sorted_rows)),
            "top3_accuracy_pct": _format_percentage(top3, len(sorted_rows)),
            "mean_reciprocal_rank": round(reciprocal_rank_sum / ranked, 6) if ranked else 0.0,
            "mean_kendall_tau_against_gold_ranking": round(kendall_tau_sum / ranked, 6) if ranked else 0.0,
            "mean_ranking_decision_inversion_rate": round(inversion_rate_sum / ranked, 6) if ranked else 0.0,
            "valid_rankings": ranked,
        }
        if task == "rank"
        else {},
        "decision_rationale": {
            "option_decision_correct": option_decision_correct,
            "option_decision_total": option_decision_total,
            "option_decision_accuracy_pct": _format_percentage(option_decision_correct, option_decision_total),
            "all_options_exact_match_pct": _format_percentage(all_option_correct, len(sorted_rows)),
            "false_recommend_count": false_recommend_count,
            "false_recommend_rate_pct": _format_percentage(false_recommend_count, option_decision_total),
            "clinical_risk_score_total": false_recommend_cost,
            "clinical_risk_score_avg_per_question": round(false_recommend_cost / len(sorted_rows), 6)
            if sorted_rows
            else 0.0,
            "ingredient_in_recipe_rate_pct": _format_percentage(
                ingredient_mentions_in_recipe,
                ingredient_mentions,
            ),
            "ingredient_mentions": ingredient_mentions,
            "condition_attribution_accuracy_pct": _format_percentage(
                attribution_correct,
                attribution_mentions,
            ),
            "condition_attribution_mentions": attribution_mentions,
        }
        if task == "decision_rationale"
        else {},
        "prediction_distribution": dict(sorted(prediction_distribution.items())),
        "parse_status_counts": dict(sorted(parse_status_counts.items())),
        "top_errors": [{"error": error, "count": count} for error, count in error_counts.most_common(10)],
        "by_condition": by_condition_summary,
        "outputs": {
            "csv": _display_path(csv_output_path),
            "jsonl": _display_path(jsonl_output_path),
            "summary_json": _display_path(summary_output_path),
            "debug_jsonl": _display_path(debug_output_path) if debug_rows else "",
        },
    }
    summary_output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def evaluate(
    *,
    benchmark_path: Path,
    model: str,
    provider: str,
    task: str,
    prompt_mode: str,
    csv_output_path: Path,
    jsonl_output_path: Path,
    summary_output_path: Path,
    debug_output_path: Path,
    max_questions: Optional[int],
    question_ids: Sequence[str],
    max_retries: int,
    retry_sleep_seconds: float,
    request_timeout_seconds: float,
    concurrency: int,
    resume: bool,
    rerun_failed: bool,
    include_recipe_ingredients: bool,
    text_only: bool,
    text_only_option_fields: str,
    image_source: str,
) -> Dict[str, Any]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    metadata = benchmark.get("metadata") or {}
    questions = _limit_questions(
        benchmark.get("questions") or [],
        max_questions=max_questions,
        question_ids=question_ids,
    )
    if not questions:
        raise SystemExit("No questions selected for evaluation.")

    task = _clean_text(task).lower() or "answer"
    if task not in {"answer", "rank", "decision_rationale"}:
        raise SystemExit("Unsupported task. Use answer, rank, or decision_rationale.")
    prompt_mode = _normalize_prompt_mode(prompt_mode)
    if prompt_mode not in PROMPT_MODES:
        raise SystemExit(f"Unsupported prompt mode. Use one of: {', '.join(sorted(PROMPT_MODES))}.")
    text_only_option_fields = _clean_text(text_only_option_fields).lower() or "title_ingredients"
    if text_only_option_fields not in {"title_ingredients", "ingredients_only"}:
        raise SystemExit("Unsupported --text-only-option-fields. Use title_ingredients or ingredients_only.")
    image_source = _clean_text(image_source).lower() or "cached_first"
    if image_source not in {"cached_first", "url_only"}:
        raise SystemExit("Unsupported --image-source. Use cached_first or url_only.")
    provider = _model_provider(model, provider)
    concurrency = max(1, int(concurrency))
    client = _create_client(model, provider)
    rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    selected_question_ids = {_clean_text(question.get("question_id")).upper() for question in questions}
    if not resume:
        jsonl_output_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_output_path.write_text("", encoding="utf-8")
        if debug_output_path.exists():
            debug_output_path.unlink()
    if resume:
        rows.extend(
            _load_existing_rows(
                jsonl_output_path,
                selected_question_ids=selected_question_ids,
                model=model,
                provider=provider,
                task=task,
                prompt_mode=prompt_mode,
                parsed_only=rerun_failed,
            )
        )

    completed_question_ids = {_clean_text(row.get("question_id")).upper() for row in rows}
    pending_questions = [
        (index, question)
        for index, question in enumerate(questions, start=1)
        if _clean_text(question.get("question_id")).upper() not in completed_question_ids
    ]
    jsonl_lock = Lock()
    debug_lock = Lock()

    def run_one(index: int, question: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        result = _call_model(
            client,
            model=model,
            question=question,
            task=task,
            prompt_mode=prompt_mode,
            include_recipe_ingredients=include_recipe_ingredients,
            text_only=text_only,
            text_only_option_fields=text_only_option_fields,
            image_source=image_source,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )
        return _build_result_row(
            index=index,
            question=question,
            model=model,
            provider=provider,
            task=task,
            prompt_mode=prompt_mode,
            result=result,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {executor.submit(run_one, index, question): index for index, question in pending_questions}
        for future in as_completed(future_map):
            row, debug_row = future.result()
            rows.append(row)
            _append_jsonl(jsonl_output_path, row, jsonl_lock)
            if debug_row:
                debug_rows.append(debug_row)
                _append_jsonl(debug_output_path, debug_row, debug_lock)

    return _summarize_rows(
        rows=rows,
        questions=questions,
        benchmark_path=benchmark_path,
        metadata=metadata,
        model=model,
        provider=provider,
        task=task,
        prompt_mode=prompt_mode,
        include_recipe_ingredients=include_recipe_ingredients,
        csv_output_path=csv_output_path,
        jsonl_output_path=jsonl_output_path,
        summary_output_path=summary_output_path,
        debug_output_path=debug_output_path,
        debug_rows=debug_rows,
        concurrency=concurrency,
        resume=resume,
        text_only=text_only,
        text_only_option_fields=text_only_option_fields,
        image_source=image_source,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "task2_comparative_analysis.json",
        help="Path to the image benchmark JSON.",
    )
    parser.add_argument(
        "--model",
        default=_default_model(),
        help="Vision-capable model name.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini", "anthropic", "vllm"),
        default="",
        help=(
            "Optional provider override. Defaults by model prefix: gemini->gemini, "
            "claude->anthropic, qwen->vllm, gemma->vllm, otherwise openai."
        ),
    )
    parser.add_argument(
        "--task",
        choices=("answer", "rank", "decision_rationale"),
        default="answer",
        help=(
            "Evaluation task: answer returns one best option; rank returns all four options best-to-worst; "
            "decision_rationale returns per-option recommend/not_recommend decisions and rationale evidence."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODE_CHOICES,
        default="baseline",
        help="Prompt variant: baseline, cot, ki, or cot_ki. Legacy rationale aliases are normalized.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path for per-question CSV output.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Path for per-question JSONL output.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY,
        help="Path for summary JSON output.",
    )
    parser.add_argument(
        "--output-debug",
        type=Path,
        default=DEFAULT_OUTPUT_DEBUG,
        help="Path for debug JSONL output used for Gemini unparseable/error cases.",
    )
    parser.add_argument("--max-questions", type=int, default=None, help="Optional maximum number of questions to run.")
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Specific question ID(s) to evaluate, e.g. CRMCQ_0001. Can be passed multiple times.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Max API attempts per question.")
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=2.0,
        help="Sleep time between failed attempts.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=120.0,
        help="Per-request API read timeout in seconds. Use a larger value for slow multimodal providers.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of questions to evaluate concurrently.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip question IDs already present in the output JSONL and finalize from accumulated rows.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help=(
            "With --resume, keep parsed rows but rerun existing rows whose parse_status is not parsed. "
            "Useful for retrying request_error or unparseable_output rows without rerunning successful questions."
        ),
    )
    parser.add_argument(
        "--include-recipe-ingredients",
        action="store_true",
        help=(
            "Add each option's recipe ingredient list to the prompt while keeping the same images, "
            "questions, categories, gold labels, and scoring."
        ),
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help=(
            "Run the same benchmark without attaching images. Options are supplied as recipe text in the prompt."
        ),
    )
    parser.add_argument(
        "--text-only-option-fields",
        choices=("title_ingredients", "ingredients_only"),
        default="title_ingredients",
        help=(
            "Text evidence for --text-only runs. title_ingredients includes recipe names and full ingredients; "
            "ingredients_only hides recipe names and uses only ingredient lists."
        ),
    )
    parser.add_argument(
        "--image-source",
        choices=("cached_first", "url_only"),
        default="url_only",
        help=(
            "Image source for image-bearing runs. url_only (default) always uses each option's image_url "
            "(passed directly to OpenAI/Gemini; fetched + base64-encoded for Anthropic). "
            "cached_first prefers a local cached_image_path and falls back to image_url -- note the cached "
            "image assets are NOT shipped with this release, so cached_first is only useful if you regenerate "
            "them locally."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_path = args.benchmark.resolve()
    model = str(args.model).strip()
    csv_output_path, jsonl_output_path, summary_output_path, debug_output_path = _resolve_default_output_paths(
        benchmark_path=benchmark_path,
        model=model,
        provider=str(args.provider or "").strip(),
        task=str(args.task or "answer").strip().lower(),
        prompt_mode=_normalize_prompt_mode(str(args.prompt_mode or "baseline")),
        csv_output_path=args.output_csv.resolve(),
        jsonl_output_path=args.output_jsonl.resolve(),
        summary_output_path=args.output_summary.resolve(),
        debug_output_path=args.output_debug.resolve(),
    )
    summary = evaluate(
        benchmark_path=benchmark_path,
        model=model,
        provider=str(args.provider or "").strip(),
        task=str(args.task or "answer").strip().lower(),
        prompt_mode=_normalize_prompt_mode(str(args.prompt_mode or "baseline")),
        csv_output_path=csv_output_path,
        jsonl_output_path=jsonl_output_path,
        summary_output_path=summary_output_path,
        debug_output_path=debug_output_path,
        max_questions=args.max_questions,
        question_ids=args.question_id,
        max_retries=max(1, int(args.max_retries)),
        retry_sleep_seconds=max(0.0, float(args.retry_sleep_seconds)),
        request_timeout_seconds=max(1.0, float(args.request_timeout_seconds)),
        concurrency=max(1, int(args.concurrency)),
        resume=bool(args.resume),
        rerun_failed=bool(args.rerun_failed),
        include_recipe_ingredients=bool(args.include_recipe_ingredients),
        text_only=bool(args.text_only),
        text_only_option_fields=str(args.text_only_option_fields),
        image_source=str(args.image_source),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
