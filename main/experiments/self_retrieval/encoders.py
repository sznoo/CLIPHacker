# experiments/self_retrieval/encoders.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from tqdm import tqdm


@dataclass
class EncoderPack:
    name: str
    model: torch.nn.Module
    preprocess: object
    tokenizer: object
    device: str


def load_openclip_encoder(
    name: str,
    model_name: str,
    pretrained: str,
    device: str = "cuda",
) -> EncoderPack:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    return EncoderPack(
        name=name,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
    )


def load_clip_encoder(device: str = "cuda") -> EncoderPack:
    return load_openclip_encoder(
        name="clip",
        model_name="ViT-B-32",
        pretrained="openai",
        device=device,
    )


def load_siglip_encoder(device: str = "cuda") -> EncoderPack:
    return load_openclip_encoder(
        name="siglip",
        model_name="ViT-SO400M-14-SigLIP-384",
        pretrained="webli",
        device=device,
    )


def load_encoder(scorer: str, device: str = "cuda") -> EncoderPack:
    scorer = scorer.lower()

    if scorer == "clip":
        return load_clip_encoder(device=device)

    if scorer == "siglip":
        return load_siglip_encoder(device=device)

    raise ValueError(f"Unknown scorer: {scorer}")


@torch.no_grad()
def encode_images(
    encoder: EncoderPack,
    image_paths: Sequence[str | Path],
    batch_size: int = 32,
) -> torch.Tensor:
    feats = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc=f"encode images ({encoder.name})"):
        batch_paths = image_paths[start : start + batch_size]

        images = [
            encoder.preprocess(Image.open(p).convert("RGB"))
            for p in batch_paths
        ]
        image_inputs = torch.stack(images).to(encoder.device)

        image_features = encoder.model.encode_image(image_inputs)
        image_features = F.normalize(image_features, dim=-1)

        feats.append(image_features.cpu())

    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_texts(
    encoder: EncoderPack,
    texts: Sequence[str],
    batch_size: int = 64,
) -> torch.Tensor:
    feats = []

    for start in tqdm(range(0, len(texts), batch_size), desc=f"encode texts ({encoder.name})"):
        batch_texts = [str(x) for x in texts[start : start + batch_size]]
        text_inputs = encoder.tokenizer(batch_texts).to(encoder.device)

        text_features = encoder.model.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=-1)

        feats.append(text_features.cpu())

    return torch.cat(feats, dim=0)


def compute_similarity_matrix(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
) -> torch.Tensor:
    text_features = F.normalize(text_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)
    return text_features @ image_features.T