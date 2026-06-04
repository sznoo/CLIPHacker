# src/scoring.py

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
import open_clip
from PIL import Image


@dataclass
class ImageTextScorer:
    name: str
    model: torch.nn.Module
    preprocess: object
    tokenizer: object
    device: str

    @torch.no_grad()
    def score(self, image: Image.Image, text: str) -> float:
        image_input = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        text_input = self.tokenizer([text]).to(self.device)

        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_input)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return float((image_features @ text_features.T).item())

    @torch.no_grad()
    def score_batch(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        image_inputs = torch.stack(
            [self.preprocess(img.convert("RGB")) for img in images]
        ).to(self.device)
        text_inputs = self.tokenizer(texts).to(self.device)

        image_features = self.model.encode_image(image_inputs)
        text_features = self.model.encode_text(text_inputs)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        scores = (image_features * text_features).sum(dim=-1)
        return [float(x) for x in scores.cpu().tolist()]


def load_openclip_scorer(
    name: str,
    model_name: str,
    pretrained: str,
    device: str = "cuda",
) -> ImageTextScorer:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    return ImageTextScorer(
        name=name,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
    )


def load_clip_scorer(device: str = "cuda") -> ImageTextScorer:
    return load_openclip_scorer(
        name="clip",
        model_name="ViT-B-32",
        pretrained="openai",
        device=device,
    )


def load_siglip_scorer(device: str = "cuda") -> ImageTextScorer:
    return load_openclip_scorer(
        name="siglip",
        model_name="ViT-SO400M-14-SigLIP-384",
        pretrained="webli",
        device=device,
    )


def load_scorers(
    device: str = "cuda",
    use_clip: bool = True,
    use_siglip: bool = True,
) -> Dict[str, ImageTextScorer]:
    scorers = {}

    if use_clip:
        scorers["clip"] = load_clip_scorer(device=device)

    if use_siglip:
        scorers["siglip"] = load_siglip_scorer(device=device)

    return scorers


def caption_length(text: str) -> int:
    return len(str(text).strip().split())


def generic_caption_flag(text: str) -> bool:
    text_l = str(text).lower()

    vague_phrases = [
        "a person",
        "someone",
        "something",
        "an image of",
        "a photo of",
        "a picture of",
        "people are",
        "there is",
        "there are",
    ]

    return any(p in text_l for p in vague_phrases)


if __name__ == "__main__":
    image = Image.new("RGB", (224, 224), color="white")
    text = "A blank white image."

    scorers = load_scorers(
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_clip=True,
        use_siglip=True,
    )

    for name, scorer in scorers.items():
        print(name, scorer.score(image, text))