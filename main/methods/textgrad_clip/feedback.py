# methods/textgrad_clip/feedback.py
import base64
import mimetypes
from pathlib import Path

from openai import OpenAI

from methods.textgrad_clip.configs import DEFAULT_MULTIMODAL_FEEDBACK_ENGINE

def _image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"

def _fmt(x):
    if x is None:
        return "N/A"
    return f"{float(x):.6f}"

def _build_case_critique_prompt(c: dict) -> str:
    return "\n".join([
        "You are generating image-grounded feedback for prompt optimization.",
        "The downstream captioner is frozen. Only its natural-language instruction prompt will be updated.",
        "The primary objective is CLIP image-text similarity, not general caption aesthetics.",
        "",
        "Look at the image and the generated caption.",
        "Identify visible, discriminative cues that the caption missed, under-specified, or made too generic.",
        "Do not write a replacement caption.",
        "Do not mention CLIP in the suggested final captioning prompt.",
        "Do not suggest external rerankers, verifiers, multiple candidates, tags, JSON, or non-caption protocols.",
        "",
        "Return 2-4 concise bullets. Each bullet should describe a prompt-level change that would help across similar images.",
        "",
        f"image_id: {c.get('image_id')}",
        f"generated_caption: {c.get('caption')}",
        f"clip_score: {_fmt(c.get('clip_score'))}",
        f"delta_clip_vs_baseline: {_fmt(c.get('delta_clip'))}",
    ])

def _generate_multimodal_case_feedback(
    c: dict,
    model: str = DEFAULT_MULTIMODAL_FEEDBACK_ENGINE,
) -> str:
    image_path = c.get("image_path")
    if not image_path:
        return "No image_path available; cannot generate image-grounded critique."

    image_path = Path(image_path)
    if not image_path.exists():
        return f"Image file not found: {image_path}"

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _build_case_critique_prompt(c),
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_to_data_url(image_path),
                    },
                ],
            }
        ],
    )
    return response.output_text.strip()

def build_textual_feedback(result: dict, max_cases: int = 5) -> str:
    s = result["summary"]
    cases = result["cases"][:max_cases]

    lines = [
        "You are evaluating a captioning prompt for a frozen image captioning model.",
        "The prompt is used to generate captions for images.",
        "The primary objective is to improve CLIP image-text similarity.",
        "SigLIP is reported only as a secondary diagnostic metric.",
        "",
        "Evaluate the current prompt using the score report below.",
        "Provide concise, critical feedback on how the prompt should be changed.",
        "Do not produce captions for the examples.",
        "",
        "Aggregate score report:",
        f"- mean_clip: {_fmt(s.get('mean_clip'))}",
        f"- mean_delta_clip: {_fmt(s.get('mean_delta_clip'))}",
        f"- clip_win_rate: {_fmt(s.get('clip_win_rate'))}",
        f"- mean_siglip: {_fmt(s.get('mean_siglip'))}",
        f"- mean_delta_siglip: {_fmt(s.get('mean_delta_siglip'))}",
        f"- siglip_win_rate: {_fmt(s.get('siglip_win_rate'))}",
        f"- mean_caption_len: {_fmt(s.get('mean_caption_len'))}",
        f"- generic_rate: {_fmt(s.get('generic_rate'))}",
        "",
        "Low-performing examples:",
    ]

    for c in cases:
        lines.extend([
            f"- image_id: {c.get('image_id')}",
            f"  caption: {c.get('caption')}",
            f"  siglip_score: {_fmt(c.get('siglip_score'))}",
            f"  clip_score: {_fmt(c.get('clip_score'))}",
            f"  delta_siglip: {_fmt(c.get('delta_siglip'))}",
            f"  delta_clip: {_fmt(c.get('delta_clip'))}",
        ])

    return "\n".join(lines)


def build_multimodal_feedback(result: dict, max_cases: int = 5) -> str:
    s = result["summary"]
    cases = result["cases"][:max_cases]

    critiques = []
    for c in cases:
        critiques.append(_generate_multimodal_case_feedback(c))

    lines = [
        "You are evaluating a captioning prompt for a frozen image captioning model.",
        "The prompt is used to generate captions for images.",
        "The primary objective is to improve CLIP image-text similarity.",
        "SigLIP is reported only as a secondary diagnostic metric.",
        "",
        "This is multimodal feedback mode.",
        "For each low-performing example, an image-grounded critique is provided.",
        "Use the aggregate scores and critiques to provide concise, critical feedback on how the prompt should be changed.",
        "Do not produce captions for the examples.",
        "Do not suggest external rerankers, verifiers, multiple candidates, tags, JSON, or non-caption protocols.",
        "",
        "Aggregate score report:",
        f"- mean_clip: {_fmt(s.get('mean_clip'))}",
        f"- mean_delta_clip: {_fmt(s.get('mean_delta_clip'))}",
        f"- clip_win_rate: {_fmt(s.get('clip_win_rate'))}",
        f"- mean_siglip: {_fmt(s.get('mean_siglip'))}",
        f"- mean_delta_siglip: {_fmt(s.get('mean_delta_siglip'))}",
        f"- siglip_win_rate: {_fmt(s.get('siglip_win_rate'))}",
        f"- mean_caption_len: {_fmt(s.get('mean_caption_len'))}",
        f"- generic_rate: {_fmt(s.get('generic_rate'))}",
        "",
        "Image-grounded low-performing examples:",
    ]

    for c, critique in zip(cases, critiques):
        lines.extend([
            f"- image_id: {c.get('image_id')}",
            f"  image_relpath: {c.get('image_relpath')}",
            f"  caption: {c.get('caption')}",
            f"  clip_score: {_fmt(c.get('clip_score'))}",
            f"  delta_clip: {_fmt(c.get('delta_clip'))}",
            "  image_grounded_critique:",
        ])
        for line in critique.splitlines():
            lines.append(f"    {line}")

    lines.extend([
        "",
        "Final instruction:",
        "Give feedback for updating the global captioning prompt.",
        "The feedback should target reusable prompt changes, not example-specific captions.",
    ])

    return "\n".join(lines)


def build_feedback_instruction(
    result: dict,
    mode: str,
    max_cases: int = 5,
) -> str:
    if mode == "textual":
        return build_textual_feedback(result, max_cases=max_cases)

    if mode == "multimodal":
        return build_multimodal_feedback(result, max_cases=max_cases)

    raise ValueError(f"Unknown feedback mode: {mode}")