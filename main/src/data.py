# src/data.py

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image


DATASET_NAME = "nlphuji/flickr30k"
DEFAULT_SOURCE_SPLIT = "test"


def get_image(row):
    image = row["image"]
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def get_reference_captions(row):
    for key in ["caption", "captions", "sentences", "raw"]:
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                return [str(x) for x in value]
    return []


def make_image_id(source_split: str, dataset_index: int) -> str:
    return f"flickr30k_{source_split}_{dataset_index:06d}"


def make_row(source_split: str, dataset_index: int, image_relpath: str, row):
    return {
        "dataset": "flickr30k",
        "source_split": source_split,
        "dataset_index": dataset_index,
        "image_id": make_image_id(source_split, dataset_index),
        "image_relpath": image_relpath,
        "reference_captions": json.dumps(
            get_reference_captions(row),
            ensure_ascii=False,
        ),
    }


def prepare_random_splits(args):
    output_root = Path(args.output_root)
    image_dir = output_root / "images" / "flickr30k"
    split_dir = output_root / "splits"

    image_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        DATASET_NAME,
        split=args.source_split,
        trust_remote_code=True,
    )

    total = args.n_train + args.n_val + args.n_test
    if total > len(ds):
        raise ValueError(f"Requested {total} images, but dataset has only {len(ds)}.")

    rng = random.Random(args.seed)
    sampled_indices = rng.sample(range(len(ds)), total)

    split_indices = {
        "train": sampled_indices[: args.n_train],
        "val": sampled_indices[args.n_train : args.n_train + args.n_val],
        "test": sampled_indices[args.n_train + args.n_val :],
    }

    all_rows = []

    for split_name, indices in split_indices.items():
        split_rows = []

        for order, dataset_index in enumerate(indices):
            row = ds[dataset_index]

            image_id = make_image_id(args.source_split, dataset_index)
            image_relpath = f"images/flickr30k/{image_id}.jpg"
            image_path = output_root / image_relpath

            if args.force or not image_path.exists():
                image = get_image(row)
                image.save(image_path, quality=95)

            item = make_row(
                source_split=args.source_split,
                dataset_index=dataset_index,
                image_relpath=image_relpath,
                row=row,
            )

            item.insert if False else None  # no-op; keeps structure simple

            split_id = f"{args.prefix}_{split_name}"
            split_item = {
                "split_id": split_id,
                "order": order,
                **item,
            }

            split_rows.append(split_item)
            all_rows.append(split_item)

        split_df = pd.DataFrame(split_rows)
        split_csv = split_dir / f"{args.prefix}_{split_name}.csv"
        split_df.to_csv(split_csv, index=False)

        print(f"Saved {split_name}: {split_csv} ({len(split_df)} rows)")

    metadata_csv = output_root / f"metadata_{args.prefix}.csv"
    pd.DataFrame(all_rows).to_csv(metadata_csv, index=False)

    print(f"Saved selected metadata: {metadata_csv}")
    print(f"Saved images:            {image_dir}")
    print(f"Total selected images:   {len(all_rows)}")


def load_split(split_csv: str, data_root: str = "data"):
    df = pd.read_csv(split_csv)
    data_root = Path(data_root)

    rows = []
    for _, row in df.iterrows():
        item = row.to_dict()
        item["image_path"] = str(data_root / item["image_relpath"])
        item["reference_captions"] = json.loads(item["reference_captions"])
        rows.append(item)

    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-random-splits")
    p.add_argument("--source-split", type=str, default=DEFAULT_SOURCE_SPLIT)
    p.add_argument("--prefix", type=str, default="flickr30k_main_seed1")
    p.add_argument("--n-train", type=int, required=True)
    p.add_argument("--n-val", type=int, required=True)
    p.add_argument("--n-test", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-root", type=str, default="data")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=prepare_random_splits)
    p = sub.add_parser("prepare-random-split")
    p.add_argument("--source-split", type=str, default=DEFAULT_SOURCE_SPLIT)
    p.add_argument("--split-id", type=str, required=True)
    p.add_argument("--num-images", type=int, required=True)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--output-root", type=str, default="data")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=prepare_random_split)
    return parser.parse_args()

def prepare_random_split(args):
    output_root = Path(args.output_root)
    image_dir = output_root / "images" / "flickr30k"
    split_dir = output_root / "splits"

    image_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        DATASET_NAME,
        split=args.source_split,
        trust_remote_code=True,
    )

    if args.num_images > len(ds):
        raise ValueError(f"Requested {args.num_images} images, but dataset has only {len(ds)}.")

    rng = random.Random(args.seed)
    sampled_indices = rng.sample(range(len(ds)), args.num_images)

    rows = []
    for order, dataset_index in enumerate(sampled_indices):
        row = ds[dataset_index]

        image_id = make_image_id(args.source_split, dataset_index)
        image_relpath = f"images/flickr30k/{image_id}.jpg"
        image_path = output_root / image_relpath

        if args.force or not image_path.exists():
            image = get_image(row)
            image.save(image_path, quality=95)

        rows.append(
            {
                "split_id": args.split_id,
                "order": order,
                "dataset": "flickr30k",
                "source_split": args.source_split,
                "dataset_index": dataset_index,
                "image_id": image_id,
                "image_relpath": image_relpath,
                "reference_captions": json.dumps(
                    get_reference_captions(row),
                    ensure_ascii=False,
                ),
            }
        )

    split_df = pd.DataFrame(rows)
    out_csv = split_dir / f"{args.split_id}.csv"
    split_df.to_csv(out_csv, index=False)

    print(f"Saved split: {out_csv} ({len(split_df)} rows)")
    print(f"Saved images: {image_dir}")

if __name__ == "__main__":
    args = parse_args()
    args.func(args)