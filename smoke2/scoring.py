# scoring.py

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
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = self.tokenizer([text]).to(self.device)

        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_input)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        score = (image_features @ text_features.T).item()
        return float(score)

    @torch.no_grad()
    def score_batch(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
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
    return len(text.strip().split())


def generic_caption_flag(text: str) -> bool:
    text_l = text.lower()

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


def compute_prompt_summary(df, baseline_prompt_id: str = "baseline"):
    baseline = (
        df[df["prompt_id"] == baseline_prompt_id]
        [["image_id", "clip_score", "siglip_score"]]
        .rename(
            columns={
                "clip_score": "baseline_clip_score",
                "siglip_score": "baseline_siglip_score",
            }
        )
    )

    merged = df.merge(baseline, on="image_id", how="left")

    merged["delta_clip"] = merged["clip_score"] - merged["baseline_clip_score"]
    merged["delta_siglip"] = merged["siglip_score"] - merged["baseline_siglip_score"]
    merged["clip_win"] = merged["delta_clip"] > 0
    merged["siglip_win"] = merged["delta_siglip"] > 0

    summary = (
        merged.groupby(["prompt_id", "prompt_name"], as_index=False)
        .agg(
            mean_clip=("clip_score", "mean"),
            mean_delta_clip=("delta_clip", "mean"),
            clip_win_rate=("clip_win", "mean"),
            mean_siglip=("siglip_score", "mean"),
            mean_delta_siglip=("delta_siglip", "mean"),
            siglip_win_rate=("siglip_win", "mean"),
            mean_caption_len=("caption_len", "mean"),
            generic_rate=("generic_flag", "mean"),
        )
        .sort_values("mean_delta_clip", ascending=False)
    )

    return summary, merged


def make_score_matrix(df, score_col: str):
    return df.pivot(
        index="image_id",
        columns="prompt_id",
        values=score_col,
    )


if __name__ == "__main__":
    from PIL import Image

    scorers = load_scorers(device="cuda" if torch.cuda.is_available() else "cpu")

    image = Image.new("RGB", (224, 224), color="white")
    text = "A blank white image."

    for name, scorer in scorers.items():
        print(name, scorer.score(image, text))