# data.py

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image


DATASET_NAME = "nlphuji/flickr30k"
DEFAULT_SPLIT = "test"


def _get_image(row):
    image = row["image"]
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _get_image_id(row, dataset_index: int):
    for key in ["image_id", "img_id", "filename", "file_name"]:
        if key in row and row[key] is not None:
            return str(row[key])
    return f"idx_{dataset_index}"


def _get_reference_captions(row):
    for key in ["caption", "captions", "sentences", "raw"]:
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                return [str(x) for x in value]
    return []


def sample_flickr30k(
    num_images: int = 10,
    seed: int = 0,
    split: str = DEFAULT_SPLIT,
):
    ds = load_dataset(DATASET_NAME, split=split, trust_remote_code=True)
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), num_images)

    samples = []
    for order, idx in enumerate(indices):
        row = ds[idx]
        image = _get_image(row)
        image_id = _get_image_id(row, idx)
        refs = _get_reference_captions(row)

        samples.append(
            {
                "order": order,
                "dataset_index": idx,
                "image_id": image_id,
                "image": image,
                "reference_captions": refs,
            }
        )

    return samples


def save_samples(
    samples,
    output_dir: str = "results/sample_images",
):
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample in samples:
        image_name = f"{sample['order']:03d}_{sample['image_id']}.jpg"
        image_path = image_dir / image_name

        sample["image"].save(image_path, quality=95)

        rows.append(
            {
                "order": sample["order"],
                "dataset_index": sample["dataset_index"],
                "image_id": sample["image_id"],
                "image_path": str(image_path),
                "reference_captions": json.dumps(
                    sample["reference_captions"],
                    ensure_ascii=False,
                ),
            }
        )

    metadata = pd.DataFrame(rows)
    metadata_path = output_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    return metadata


def load_saved_samples(metadata_path: str = "results/sample_images/metadata.csv"):
    metadata_path = Path(metadata_path)
    metadata = pd.read_csv(metadata_path)

    samples = []
    for _, row in metadata.iterrows():
        image = Image.open(row["image_path"]).convert("RGB")
        refs = json.loads(row["reference_captions"])

        samples.append(
            {
                "order": int(row["order"]),
                "dataset_index": int(row["dataset_index"]),
                "image_id": str(row["image_id"]),
                "image_path": row["image_path"],
                "image": image,
                "reference_captions": refs,
            }
        )

    return samples


def get_or_create_samples(
    num_images: int = 10,
    seed: int = 0,
    split: str = DEFAULT_SPLIT,
    output_dir: str = "results/sample_images",
    force_resample: bool = False,
):
    metadata_path = Path(output_dir) / "metadata.csv"

    if metadata_path.exists() and not force_resample:
        return load_saved_samples(metadata_path)

    samples = sample_flickr30k(
        num_images=num_images,
        seed=seed,
        split=split,
    )
    save_samples(samples, output_dir=output_dir)
    return load_saved_samples(metadata_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=str, default="results/sample_images")
    parser.add_argument("--force-resample", action="store_true")
    args = parser.parse_args()

    samples = get_or_create_samples(
        num_images=args.num_images,
        seed=args.seed,
        split=args.split,
        output_dir=args.output_dir,
        force_resample=args.force_resample,
    )

    print(f"Saved/loaded {len(samples)} samples from {args.output_dir}")
    for s in samples:
        print(f"{s['order']:03d} | {s['image_id']} | {s['image_path']}")