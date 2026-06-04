# methods/qwen_prompt/run_qwen_prompt.py

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.prompts import get_prompts
from methods.qwen_prompt.models_qwen import (
    DEFAULT_CAPTION_MODEL,
    load_captioner,
    generate_caption,
)


def load_split(split_csv: str):
    df = pd.read_csv(split_csv)

    required = [
        "split_id",
        "order",
        "dataset_index",
        "image_id",
        "image_relpath",
        "reference_captions",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in split CSV: {missing}")

    return df


def make_run_id(args, split_id: str) -> str:
    if args.run_id is not None:
        return args.run_id
    return f"{args.method_id}__{split_id}"


def make_existing_key(row):
    return (
        str(row["image_id"]),
        str(row["prompt_id"]),
        int(row.get("candidate_id", 0)),
    )


def load_existing_rows(out_csv: str, overwrite: bool):
    path = Path(out_csv)
    if overwrite or not path.exists():
        return [], set()

    df = pd.read_csv(path)
    rows = df.to_dict("records")
    keys = {make_existing_key(r) for r in rows}

    print(f"Loaded existing predictions: {path} ({len(rows)} rows)")
    return rows, keys


def make_output_row(
    split_row,
    prompt,
    caption: str,
    args,
    run_id: str,
):
    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    return {
        "split_id": split_row["split_id"],
        "method_id": args.method_id,
        "run_id": run_id,
        "order": int(split_row["order"]),
        "dataset_index": int(split_row["dataset_index"]),
        "image_id": str(split_row["image_id"]),
        "image_relpath": str(split_row["image_relpath"]),
        "reference_captions": split_row["reference_captions"],
        "prompt_id": prompt["prompt_id"],
        "prompt_name": prompt["prompt_name"],
        "prompt_text": prompt["text"],
        "caption": caption,
        "candidate_id": 0,
        "selected": 1,
        "model": args.caption_model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "generation_config": json.dumps(generation_config, ensure_ascii=False),
    }


def save_rows(rows, out_csv: str):
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df = df.sort_values(["order", "prompt_id", "candidate_id"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)

    print(f"Saved predictions: {out_path} ({len(df)} rows)")


def run(args):
    split_df = load_split(args.split_csv)

    if args.limit is not None:
        split_df = split_df.head(args.limit).copy()

    split_ids = split_df["split_id"].unique().tolist()
    if len(split_ids) != 1:
        raise ValueError(f"Expected one split_id, found: {split_ids}")

    split_id = split_ids[0]
    run_id = make_run_id(args, split_id)

    prompts = get_prompts(args.prompt_ids)

    rows, existing_keys = load_existing_rows(
        out_csv=args.out_csv,
        overwrite=args.overwrite,
    )

    tasks = []
    for _, split_row in split_df.iterrows():
        for prompt in prompts:
            key = (str(split_row["image_id"]), str(prompt["prompt_id"]), 0)
            if key not in existing_keys:
                tasks.append((split_row, prompt))

    print(f"Split:       {split_id}")
    print(f"Run ID:      {run_id}")
    print(f"Images:      {len(split_df)}")
    print(f"Prompts:     {len(prompts)}")
    print(f"Tasks:       {len(split_df) * len(prompts)}")
    print(f"Existing:    {len(existing_keys)}")
    print(f"Missing:     {len(tasks)}")

    if not tasks:
        save_rows(rows, args.out_csv)
        return

    captioner = load_captioner(
        model_name=args.caption_model,
        device_map=args.caption_device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    data_root = Path(args.data_root)

    for split_row, prompt in tqdm(tasks, desc="Generating captions"):
        image_path = data_root / split_row["image_relpath"]
        image = Image.open(image_path).convert("RGB")

        caption = generate_caption(
            captioner=captioner,
            image=image,
            prompt=prompt["text"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        row = make_output_row(
            split_row=split_row,
            prompt=prompt,
            caption=caption,
            args=args,
            run_id=run_id,
        )

        rows.append(row)

        if args.save_every > 0 and len(rows) % args.save_every == 0:
            save_rows(rows, args.out_csv)

    save_rows(rows, args.out_csv)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-csv",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--prompt-ids",
        nargs="+",
        default=["baseline", "observer_specific_supported"],
    )

    parser.add_argument(
        "--method-id",
        type=str,
        default="qwen_prompt",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--caption-model",
        type=str,
        default=DEFAULT_CAPTION_MODEL,
    )
    parser.add_argument(
        "--caption-device-map",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)