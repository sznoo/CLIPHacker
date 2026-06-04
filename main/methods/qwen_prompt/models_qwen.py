# methods/qwen_prompt/models_qwen.py

from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


DEFAULT_CAPTION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass
class Captioner:
    model_name: str
    model: torch.nn.Module
    processor: object

    @property
    def input_device(self):
        return next(self.model.parameters()).device


def load_captioner(
    model_name: str = DEFAULT_CAPTION_MODEL,
    device_map: str = "auto",
    dtype: str = "auto",
    attn_implementation: Optional[str] = None,
) -> Captioner:
    kwargs = {
        "dtype": dtype,
        "device_map": device_map,
    }

    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        **kwargs,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    model.eval()

    return Captioner(
        model_name=model_name,
        model=model,
        processor=processor,
    )


def build_caption_messages(image: Image.Image, prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        }
    ]


@torch.no_grad()
def generate_caption(
    captioner: Captioner,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
) -> str:
    messages = build_caption_messages(image, prompt)

    text = captioner.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = captioner.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(captioner.input_device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }

    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

    generated_ids = captioner.model.generate(
        **inputs,
        **gen_kwargs,
    )

    input_len = inputs["input_ids"].shape[-1]
    generated_trimmed = generated_ids[:, input_len:]

    caption = captioner.processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return caption.strip()


if __name__ == "__main__":
    image = Image.new("RGB", (512, 512), color="white")
    captioner = load_captioner()

    caption = generate_caption(
        captioner=captioner,
        image=image,
        prompt="Describe the image in one concise sentence.",
        max_new_tokens=64,
        temperature=0.0,
    )

    print(caption)