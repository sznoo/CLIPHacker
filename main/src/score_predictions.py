# src/score_predictions.py

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.scoring import (
    load_scorers,
    caption_length,
    generic_caption_flag,
)


KEY_COLS = [
    "method_id",
    "run_id",
    "image_id",
    "prompt_id",
    "candidate_id",
    "caption",
]


def validate_predictions(df: pd.DataFrame):
    required = [
        "method_id",
        "run_id",
        "image_id",
        "image_relpath",
        "prompt_id",
        "caption",
        "candidate_id",
        "selected",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in prediction CSV: {missing}")


def make_key(row) -> tuple:
    return tuple(str(row[c]) for c in KEY_COLS)


def load_existing_scores(out_csv: str, overwrite: bool, target_keys: set):
    out_path = Path(out_csv)

    if overwrite or not out_path.exists():
        return [], set()

    df = pd.read_csv(out_path)
    if len(df) == 0:
        return [], set()

    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        print(f"Ignoring existing score file because key columns are missing: {missing}")
        return [], set()

    rows = []
    keys = set()

    for _, row in df.iterrows():
        key = make_key(row)
        if key in target_keys:
            rows.append(row.to_dict())
            keys.add(key)

    print(f"Loaded existing scores: {out_path} ({len(rows)} reusable rows)")
    return rows, keys


def save_rows(rows, out_csv: str):
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    sort_cols = [c for c in ["order", "prompt_id", "candidate_id"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"Saved scores: {out_path} ({len(df)} rows)")


def score_batch(batch, scorers, data_root: Path):
    images = []
    captions = []

    for row in batch:
        image_path = data_root / row["image_relpath"]
        image = Image.open(image_path).convert("RGB")

        images.append(image)
        captions.append(str(row["caption"]))

    clip_scores = None
    siglip_scores = None

    if "clip" in scorers:
        clip_scores = scorers["clip"].score_batch(images, captions)

    if "siglip" in scorers:
        siglip_scores = scorers["siglip"].score_batch(images, captions)

    scored_rows = []
    for i, row in enumerate(batch):
        out = dict(row)

        if clip_scores is not None:
            out["clip_score"] = clip_scores[i]

        if siglip_scores is not None:
            out["siglip_score"] = siglip_scores[i]

        out["caption_len"] = caption_length(out["caption"])
        out["generic_flag"] = generic_caption_flag(out["caption"])

        scored_rows.append(out)

    return scored_rows


def run(args):
    pred_df = pd.read_csv(args.pred_csv)
    validate_predictions(pred_df)

    if args.selected_only:
        pred_df = pred_df[pred_df["selected"].astype(int) == 1].copy()

    if args.limit is not None:
        pred_df = pred_df.head(args.limit).copy()

    target_rows = pred_df.to_dict("records")
    target_keys = {make_key(row) for row in target_rows}

    rows, existing_keys = load_existing_scores(
        out_csv=args.out_csv,
        overwrite=args.overwrite,
        target_keys=target_keys,
    )

    missing_rows = [
        row for row in target_rows
        if make_key(row) not in existing_keys
    ]

    print(f"Prediction CSV: {args.pred_csv}")
    print(f"Output CSV:     {args.out_csv}")
    print(f"Rows target:    {len(target_rows)}")
    print(f"Rows existing:  {len(rows)}")
    print(f"Rows missing:   {len(missing_rows)}")
    print(f"Device:         {args.device}")
    print(f"Batch size:     {args.batch_size}")

    if not missing_rows:
        save_rows(rows, args.out_csv)
        return

    scorers = load_scorers(
        device=args.device,
        use_clip=not args.no_clip,
        use_siglip=not args.no_siglip,
    )

    data_root = Path(args.data_root)

    for start in tqdm(range(0, len(missing_rows), args.batch_size), desc="Scoring"):
        batch = missing_rows[start : start + args.batch_size]
        scored = score_batch(batch, scorers=scorers, data_root=data_root)
        rows.extend(scored)

        if args.save_every > 0 and len(rows) % args.save_every == 0:
            save_rows(rows, args.out_csv)

    save_rows(rows, args.out_csv)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred-csv", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--out-csv", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--selected-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--no-siglip", action="store_true")

    parser.add_argument("--save-every", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)