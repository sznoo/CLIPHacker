# prompts.py

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


def get_prompts():
    return PROMPTS


if __name__ == "__main__":
    for p in get_prompts():
        print(f"{p['prompt_id']}: {p['text']}")