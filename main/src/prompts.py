# src/prompts.py

PROMPTS = [
    {
        "prompt_id": "baseline",
        "prompt_name": "Baseline",
        "text": "Write a caption for this image.",
    },
    {
        "prompt_id": "observer_specific_supported",
        "prompt_name": "Observer Specific Supported",
        "text": (
            "Write one specific caption as a careful human observer would: "
            "state what is visibly happening, name the important subjects and objects, "
            "and include only scene meanings that are clearly supported."
        ),
    },
]


def get_prompts(prompt_ids=None):
    if prompt_ids is None:
        return PROMPTS

    prompt_map = get_prompt_map()
    missing = [pid for pid in prompt_ids if pid not in prompt_map]
    if missing:
        raise ValueError(f"Unknown prompt_id: {missing}")

    return [prompt_map[pid] for pid in prompt_ids]


def get_prompt_map():
    prompt_ids = [p["prompt_id"] for p in PROMPTS]
    if len(prompt_ids) != len(set(prompt_ids)):
        dup = sorted({x for x in prompt_ids if prompt_ids.count(x) > 1})
        raise ValueError(f"Duplicate prompt_id found: {dup}")

    return {p["prompt_id"]: p for p in PROMPTS}


if __name__ == "__main__":
    for p in get_prompts():
        print(f"{p['prompt_id']}: {p['text']}")