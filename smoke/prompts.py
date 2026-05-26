# prompts.py

PROMPTS = [
    {
        "prompt_id": "baseline",
        "prompt_name": "Baseline",
        "text": "Describe the image in one concise sentence.",
    },
    {
        "prompt_id": "object_action",
        "prompt_name": "Object-action",
        "text": "Describe the image by naming the main visible subject, action, and setting in one sentence.",
    },
    {
        "prompt_id": "grounded_detail",
        "prompt_name": "Grounded detail",
        "text": "Generate a visually grounded caption. Mention only objects, actions, and scene details clearly visible in the image.",
    },
    {
        "prompt_id": "anti_generic",
        "prompt_name": "Anti-generic",
        "text": "Describe the image with concrete visible objects and actions. Avoid generic descriptions.",
    },
    {
        "prompt_id": "main_subject_first",
        "prompt_name": "Main subject first",
        "text": "Describe the image in one sentence. Start with the main subject, then describe the action and surrounding scene.",
    },
    {
        "prompt_id": "clip_friendly",
        "prompt_name": "CLIP-friendly",
        "text": "Write a concise caption that closely matches the visible content of the image, including the main subject, action, and scene.",
    },
    {
        "prompt_id": "no_inference",
        "prompt_name": "No inference",
        "text": "Describe only what can be directly seen in the image. Do not infer unseen events, intentions, or identities.",
    },
    {
        "prompt_id": "dense_visual",
        "prompt_name": "Dense visual caption",
        "text": "Describe the image in one specific sentence including the main subject, action, setting, and important visible objects.",
    },
    {
        "prompt_id": "short_phrase",
        "prompt_name": "Short phrase",
        "text": "Describe the image in a short phrase focusing on the main visible subject and action.",
    },
    {
        "prompt_id": "two_sentence_detail",
        "prompt_name": "Two-sentence detail",
        "text": "Describe the image in two short sentences. Mention the main subject, action, setting, and key visible objects.",
    },
]


def get_prompts():
    return PROMPTS


def get_prompt_ids():
    return [p["prompt_id"] for p in PROMPTS]


def get_prompt_text(prompt_id: str) -> str:
    for prompt in PROMPTS:
        if prompt["prompt_id"] == prompt_id:
            return prompt["text"]
    raise KeyError(f"Unknown prompt_id: {prompt_id}")


if __name__ == "__main__":
    for i, prompt in enumerate(PROMPTS):
        print(f"{i:02d} | {prompt['prompt_id']} | {prompt['text']}")