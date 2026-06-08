# methods/behavior_analysis/prompt_updater.py

import os
import re
from typing import Any

from openai import OpenAI

from methods.behavior_analysis.configs import (
    DEFAULT_OPTIMIZER_MODEL,
    OPENAI_API_KEY,
    PROMPT_UPDATE_CONSTRAINTS,
)


def _fmt(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def _clean_prompt(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = " ".join(lines)

    if (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith("'") and text.endswith("'"))
    ):
        text = text[1:-1].strip()

    prefixes = [
        "New prompt:",
        "Updated prompt:",
        "Prompt:",
        "Candidate prompt:",
    ]
    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()

    return text


def _format_summary(train_summary: dict | None) -> str:
    if not train_summary:
        return "N/A"

    fields = [
        ("n", train_summary.get("n")),
        ("mean_clip", train_summary.get("mean_clip")),
        ("mean_delta_clip", train_summary.get("mean_delta_clip")),
        ("clip_win_rate", train_summary.get("clip_win_rate")),
        ("mean_caption_len", train_summary.get("mean_caption_len")),
        ("generic_rate", train_summary.get("generic_rate")),
    ]

    return "\n".join(f"- {k}: {_fmt(v)}" for k, v in fields)


def _format_residual_records(
    residual_records: list[dict],
    max_records: int | None = None,
) -> str:
    records = residual_records if max_records is None else residual_records[:max_records]

    blocks = []
    for idx, r in enumerate(records, start=1):
        residuals = r.get("residuals") or []
        if isinstance(residuals, str):
            residuals = [residuals]

        lines = [
            f"Example {idx}",
            f"- image_id: {r.get('image_id', 'N/A')}",
            f"- current_caption: {r.get('caption', 'N/A')}",
            f"- clip_score: {_fmt(r.get('clip_score'))}",
            f"- delta_clip: {_fmt(r.get('delta_clip'))}",
            "- behavior_residuals:",
        ]

        for residual in residuals:
            residual = str(residual).strip().lstrip("-•").strip()
            if residual:
                lines.append(f"  - {residual}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_update_prompt(
    current_prompt: str,
    residual_records: list[dict],
    train_summary: dict | None = None,
    max_records: int | None = None,
) -> str:
    constraints = "\n".join(f"- {c}" for c in PROMPT_UPDATE_CONSTRAINTS)
    summary = _format_summary(train_summary)
    residual_text = _format_residual_records(residual_records, max_records=max_records)

    return "\n".join([
        "You are updating a global instruction prompt for a frozen image captioning model.",
        "",
        "The model receives an image and this prompt, then outputs one caption.",
        "The goal is to improve CLIP image-text similarity by addressing recurring behavior residuals.",
        "",
        "Current global prompt:",
        current_prompt,
        "",
        "Train score summary:",
        summary,
        "",
        "Behavior residuals from low-CLIP examples:",
        residual_text if residual_text else "N/A",
        "",
        "Update task:",
        "Rewrite the global prompt so that future captions address the recurring residuals.",
        "The new prompt must be general, reusable, and not tied to any specific image.",
        "",
        "Constraints:",
        constraints,
        "",
        "Return only the new prompt text.",
    ])


def update_prompt_from_residuals(
    current_prompt: str,
    residual_records: list[dict],
    train_summary: dict | None = None,
    model: str = DEFAULT_OPTIMIZER_MODEL,
    max_records: int | None = None,
) -> tuple[str, str]:
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    update_instruction = build_update_prompt(
        current_prompt=current_prompt,
        residual_records=residual_records,
        train_summary=train_summary,
        max_records=max_records,
    )

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": update_instruction,
            }
        ],
    )

    raw_text = response.output_text.strip()
    new_prompt = _clean_prompt(raw_text)

    if not new_prompt:
        raise ValueError("Optimizer returned an empty prompt.")

    return new_prompt, raw_text