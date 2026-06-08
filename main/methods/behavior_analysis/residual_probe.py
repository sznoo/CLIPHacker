# methods/behavior_analysis/residual_probe.py

import re
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from methods.behavior_analysis.configs import (
        DEFAULT_MAX_RESIDUAL_CASES,
        DEFAULT_RESIDUALS_PER_CASE,
    )
except Exception:
    DEFAULT_MAX_RESIDUAL_CASES = 5
    DEFAULT_RESIDUALS_PER_CASE = 2


def _fmt(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def _clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[-*•]\s*", "", line)
    line = re.sub(r"^\d+[\).\s]+", "", line)
    line = line.strip()
    return line


def parse_residuals(text: str, max_items: int) -> list[str]:
    lines = []
    for line in str(text).splitlines():
        line = _clean_line(line)
        if not line:
            continue
        if line.lower().startswith(("here are", "sure", "residual")):
            continue
        lines.append(line)

    if not lines:
        chunks = re.split(r"[;\n]", str(text))
        lines = [_clean_line(c) for c in chunks if _clean_line(c)]

    deduped = []
    seen = set()
    for line in lines:
        key = line.lower()
        if key not in seen:
            deduped.append(line)
            seen.add(key)

    return deduped[:max_items]


def build_residual_prompt(caption: str, n: int = 2) -> str:
    return "\n".join([
        "You are shown an image and a caption produced by a captioning instruction.",
        "",
        f"Current caption: {caption}",
        "",
        "Do not write a replacement caption.",
        "Do not mention image-specific entities, colors, text, places, or object names from this image.",
        "List general behavior changes that would make this caption better aligned with the image.",
        "Each behavior change must be reusable as an instruction for future image captions.",
        f"Return exactly {n} concise bullets.",
    ])


def build_mock_residuals(caption: str, n: int = 2) -> list[str]:
    caption_l = str(caption).lower()
    residuals = []

    generic_terms = [
        "something",
        "someone",
        "people",
        "person",
        "outside",
        "indoors",
        "scene",
        "image",
        "area",
        "thing",
    ]

    if len(str(caption).split()) < 10 or any(t in caption_l for t in generic_terms):
        residuals.append(
            "Make the caption more specific about the visible subjects, objects, and setting."
        )

    if not any(v in caption_l for v in ["standing", "sitting", "walking", "holding", "riding", "playing", "looking"]):
        residuals.append(
            "Describe what is visibly happening instead of giving only a broad scene summary."
        )

    residuals.append(
        "Name the important visible subjects and objects rather than using generic wording."
    )
    residuals.append(
        "Avoid unsupported interpretations and include only meanings clearly grounded in visible evidence."
    )

    deduped = []
    seen = set()
    for r in residuals:
        key = r.lower()
        if key not in seen:
            deduped.append(r)
            seen.add(key)

    return deduped[:n]


def load_qwen_captioner(model_name: str | None = None):
    from methods.qwen_prompt.models_qwen import DEFAULT_CAPTION_MODEL, load_captioner

    if model_name is None:
        model_name = DEFAULT_CAPTION_MODEL

    try:
        return load_captioner(model_name)
    except TypeError:
        try:
            return load_captioner(model_name=model_name)
        except TypeError:
            return load_captioner()


def _generate_with_qwen(captioner, image: Image.Image, prompt: str) -> str:
    from methods.qwen_prompt.models_qwen import generate_caption

    call_attempts = []

    if isinstance(captioner, tuple):
        call_attempts.extend([
            lambda: generate_caption(*captioner, image, prompt),
            lambda: generate_caption(*captioner, image=image, prompt=prompt),
            lambda: generate_caption(captioner, image, prompt),
        ])
    else:
        call_attempts.extend([
            lambda: generate_caption(captioner, image, prompt),
            lambda: generate_caption(captioner, image=image, prompt=prompt),
        ])

    last_error = None
    for fn in call_attempts:
        try:
            out = fn()
            if isinstance(out, dict):
                for key in ["caption", "text", "output", "response"]:
                    if key in out:
                        return str(out[key])
                return str(out)
            return str(out)
        except TypeError as e:
            last_error = e

    raise TypeError(
        "Could not call methods.qwen_prompt.models_qwen.generate_caption. "
        "Adjust _generate_with_qwen() to match the local generate_caption signature."
    ) from last_error


def _load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def generate_residual_for_case(
    case: dict,
    residuals_per_case: int = DEFAULT_RESIDUALS_PER_CASE,
    mode: str = "mock",
    captioner=None,
    model_name: str | None = None,
) -> dict:
    caption = str(case.get("caption", ""))
    image_path = case.get("image_path")

    if mode == "mock":
        raw_text = "\n".join(f"- {r}" for r in build_mock_residuals(caption, residuals_per_case))
        residuals = parse_residuals(raw_text, residuals_per_case)

    elif mode == "qwen":
        if not image_path:
            raise ValueError(f"Missing image_path for case: {case.get('image_id')}")

        if captioner is None:
            captioner = load_qwen_captioner(model_name=model_name)

        image = _load_image(image_path)
        residual_prompt = build_residual_prompt(caption, n=residuals_per_case)
        raw_text = _generate_with_qwen(captioner, image, residual_prompt)
        residuals = parse_residuals(raw_text, residuals_per_case)

    else:
        raise ValueError(f"Unknown residual probe mode: {mode}")

    return {
        "image_id": case.get("image_id"),
        "image_relpath": case.get("image_relpath"),
        "image_path": image_path,
        "caption": caption,
        "clip_score": case.get("clip_score"),
        "siglip_score": case.get("siglip_score"),
        "delta_clip": case.get("delta_clip"),
        "delta_siglip": case.get("delta_siglip"),
        "mode": mode,
        "raw_residual_text": raw_text,
        "residuals": residuals,
    }


def build_behavior_residuals(
    cases: list[dict],
    max_cases: int = DEFAULT_MAX_RESIDUAL_CASES,
    residuals_per_case: int = DEFAULT_RESIDUALS_PER_CASE,
    mode: str = "mock",
    model_name: str | None = None,
) -> list[dict]:
    selected_cases = cases[:max_cases]

    captioner = None
    if mode == "qwen":
        captioner = load_qwen_captioner(model_name=model_name)

    records = []
    for case in selected_cases:
        record = generate_residual_for_case(
            case=case,
            residuals_per_case=residuals_per_case,
            mode=mode,
            captioner=captioner,
            model_name=model_name,
        )
        records.append(record)

    return records


def flatten_residuals(records: list[dict]) -> list[str]:
    out = []
    for r in records:
        for residual in r.get("residuals") or []:
            residual = str(residual).strip()
            if residual:
                out.append(residual)
    return out