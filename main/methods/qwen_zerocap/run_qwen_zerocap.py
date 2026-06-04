# methods/qwen_zerocap/run_qwen_zerocap.py

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from methods.qwen_zerocap.qwen_zerocap import (
    DEFAULT_CAPTION_MODEL,
    QwenZeroCapConfig,
    QwenZeroCapGenerator,
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


def build_generation_config_dict(args):
    return {
        "caption_model": args.caption_model,
        "device_map": args.caption_device_map,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "clip_model_name": args.clip_model_name,
        "clip_pretrained": args.clip_pretrained,
        "clip_device": args.clip_device,
        "prompt": args.prompt_text,
        "clip_text_prefix": args.clip_text_prefix,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "top_size": args.top_size,
        "num_iterations": args.num_iterations,
        "clip_loss_temperature": args.clip_loss_temperature,
        "clip_scale": args.clip_scale,
        "ce_scale": args.ce_scale,
        "stepsize": args.stepsize,
        "grad_norm_factor": args.grad_norm_factor,
        "fusion_factor": args.fusion_factor,
        "repetition_penalty": args.repetition_penalty,
        "end_factor": args.end_factor,
        "verbose_cache": args.verbose_cache,
    }


def make_output_row(
    split_row,
    caption: str,
    args,
    run_id: str,
):
    generation_config = build_generation_config_dict(args)

    return {
        "split_id": split_row["split_id"],
        "method_id": args.method_id,
        "run_id": run_id,
        "order": int(split_row["order"]),
        "dataset_index": int(split_row["dataset_index"]),
        "image_id": str(split_row["image_id"]),
        "image_relpath": str(split_row["image_relpath"]),
        "reference_captions": split_row["reference_captions"],
        "prompt_id": args.prompt_id,
        "prompt_name": args.prompt_name,
        "prompt_text": args.prompt_text,
        "caption": caption,
        "candidate_id": 0,
        "selected": 1,
        "model": args.caption_model,
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

    rows, existing_keys = load_existing_rows(
        out_csv=args.out_csv,
        overwrite=args.overwrite,
    )

    tasks = []
    for _, split_row in split_df.iterrows():
        key = (str(split_row["image_id"]), str(args.prompt_id), 0)
        if key not in existing_keys:
            tasks.append(split_row)

    print(f"Split:       {split_id}")
    print(f"Run ID:      {run_id}")
    print(f"Images:      {len(split_df)}")
    print(f"Tasks:       {len(split_df)}")
    print(f"Existing:    {len(existing_keys)}")
    print(f"Missing:     {len(tasks)}")
    print(f"Method:      {args.method_id}")
    print(f"Prompt ID:   {args.prompt_id}")

    if not tasks:
        save_rows(rows, args.out_csv)
        return

    config = QwenZeroCapConfig(
        caption_model=args.caption_model,
        device_map=args.caption_device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        clip_model_name=args.clip_model_name,
        clip_pretrained=args.clip_pretrained,
        clip_device=args.clip_device,
        prompt=args.prompt_text,
        clip_text_prefix=args.clip_text_prefix,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        top_size=args.top_size,
        num_iterations=args.num_iterations,
        clip_loss_temperature=args.clip_loss_temperature,
        clip_scale=args.clip_scale,
        ce_scale=args.ce_scale,
        stepsize=args.stepsize,
        grad_norm_factor=args.grad_norm_factor,
        fusion_factor=args.fusion_factor,
        repetition_penalty=args.repetition_penalty,
        end_factor=args.end_factor,
        verbose_cache=args.verbose_cache,
    )
    generator = QwenZeroCapGenerator(config)

    data_root = Path(args.data_root)

    for split_row in tqdm(tasks, desc="Generating ZeroCap captions"):
        image_path = data_root / split_row["image_relpath"]
        image = Image.open(image_path).convert("RGB")

        caption = generator.generate(
            image=image,
            prompt=args.prompt_text,
            max_new_tokens=args.max_new_tokens,
        )

        row = make_output_row(
            split_row=split_row,
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

    parser.add_argument("--split-csv", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--out-csv", type=str, required=True)

    parser.add_argument("--method-id", type=str, default="qwen_zerocap_greedy")
    parser.add_argument("--run-id", type=str, default=None)

    parser.add_argument("--prompt-id", type=str, default="qwen_zerocap")
    parser.add_argument("--prompt-name", type=str, default="Qwen ZeroCap")
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="Write a short, natural caption for this image.",
    )

    parser.add_argument("--caption-model", type=str, default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--caption-device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--attn-implementation", type=str, default=None)

    parser.add_argument("--clip-model-name", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--clip-device", type=str, default="cuda:0")
    parser.add_argument("--clip-text-prefix", type=str, default="A photo of ")

    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=5)
    parser.add_argument("--top-size", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=1)

    parser.add_argument("--clip-loss-temperature", type=float, default=0.01)
    parser.add_argument("--clip-scale", type=float, default=1.0)
    parser.add_argument("--ce-scale", type=float, default=0.5)
    parser.add_argument("--stepsize", type=float, default=0.05)
    parser.add_argument("--grad-norm-factor", type=float, default=0.9)
    parser.add_argument("--fusion-factor", type=float, default=0.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--end-factor", type=float, default=1.01)

    parser.add_argument("--verbose-cache", action="store_true")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)