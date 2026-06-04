# experiments/self_retrieval/run_eval.py

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.self_retrieval.io import load_self_retrieval_inputs
from experiments.self_retrieval.encoders import (
    load_encoder,
    encode_images,
    encode_texts,
    compute_similarity_matrix,
)
from experiments.self_retrieval.metrics import (
    compute_retrieval_rows,
    summarize_retrieval,
    compare_to_baseline,
)


def _safe_name(x: str) -> str:
    x = str(x)
    keep = []
    for ch in x:
        if ch.isalnum() or ch in ["-", "_", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def _file_hash(path: str | Path) -> str:
    p = Path(path).resolve()
    return hashlib.md5(str(p).encode("utf-8")).hexdigest()[:10]


def _text_hash(texts: list[str]) -> str:
    h = hashlib.md5()
    for t in texts:
        h.update(str(t).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:10]


def _load_pt(path: Path):
    return torch.load(path, map_location="cpu")


def _save_pt(x, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x, path)


def get_image_cache_paths(
    cache_dir: Path,
    split_id: str,
    scorer: str,
    split_csv: str | Path,
) -> tuple[Path, Path]:
    key = _file_hash(split_csv)
    base = cache_dir / split_id / scorer
    return (
        base / f"image_features__{key}.pt",
        base / f"image_meta__{key}.csv",
    )


def get_text_cache_paths(
    cache_dir: Path,
    split_id: str,
    scorer: str,
    method_id: str,
    prompt_id: str,
    prediction_csv: str | Path,
    captions: list[str],
) -> tuple[Path, Path]:
    pred_key = _file_hash(prediction_csv)
    text_key = _text_hash(captions)
    base = cache_dir / split_id / scorer
    name = f"text_features__{_safe_name(method_id)}__{_safe_name(prompt_id)}__{pred_key}__{text_key}"
    return (
        base / f"{name}.pt",
        base / f"{name}.csv",
    )


def maybe_encode_images(
    encoder,
    split_df: pd.DataFrame,
    split_csv: str | Path,
    cache_dir: Path,
    batch_size: int,
    overwrite_cache: bool,
) -> torch.Tensor:
    split_id = str(split_df["split_id"].iloc[0])
    feat_path, meta_path = get_image_cache_paths(
        cache_dir=cache_dir,
        split_id=split_id,
        scorer=encoder.name,
        split_csv=split_csv,
    )

    if feat_path.exists() and meta_path.exists() and not overwrite_cache:
        print(f"[cache hit] image features: {feat_path}")
        return _load_pt(feat_path)

    print("[encode] image features")
    image_features = encode_images(
        encoder=encoder,
        image_paths=split_df["image_path"].tolist(),
        batch_size=batch_size,
    )

    _save_pt(image_features, feat_path)
    split_df[["split_id", "order", "gallery_index", "image_id", "image_relpath", "image_path"]].to_csv(
        meta_path,
        index=False,
    )
    print(f"[cache save] image features: {feat_path}")

    return image_features


def maybe_encode_texts(
    encoder,
    group_meta: dict,
    group_df: pd.DataFrame,
    split_id: str,
    prediction_csv: str | Path,
    cache_dir: Path,
    batch_size: int,
    overwrite_cache: bool,
) -> torch.Tensor:
    method_id = str(group_meta.get("method_id", "unknown_method"))
    prompt_id = str(group_meta.get("prompt_id", "unknown_prompt"))
    captions = [str(x) for x in group_df["caption"].tolist()]

    feat_path, meta_path = get_text_cache_paths(
        cache_dir=cache_dir,
        split_id=split_id,
        scorer=encoder.name,
        method_id=method_id,
        prompt_id=prompt_id,
        prediction_csv=prediction_csv,
        captions=captions,
    )

    if feat_path.exists() and meta_path.exists() and not overwrite_cache:
        print(f"[cache hit] text features: {feat_path}")
        return _load_pt(feat_path)

    print(f"[encode] text features: method={method_id}, prompt={prompt_id}")
    text_features = encode_texts(
        encoder=encoder,
        texts=captions,
        batch_size=batch_size,
    )

    _save_pt(text_features, feat_path)

    meta_cols = [
        c
        for c in [
            "split_id",
            "method_id",
            "run_id",
            "prompt_id",
            "prompt_name",
            "order",
            "gallery_index",
            "image_id",
            "caption",
        ]
        if c in group_df.columns
    ]
    group_df[meta_cols].to_csv(meta_path, index=False)
    print(f"[cache save] text features: {feat_path}")

    return text_features


def save_config(args, output_dir: Path, split_df: pd.DataFrame) -> None:
    config = vars(args).copy()
    config["n_gallery"] = int(len(split_df))
    config["split_id"] = str(split_df["split_id"].iloc[0])
    config["output_dir"] = str(output_dir)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--scorer", choices=["clip", "siglip"], default="siglip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=64)

    parser.add_argument("--cache-dir", default="results/self_retrieval/cache")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--no-check-images", action="store_true")
    parser.add_argument("--include-unselected", action="store_true")

    parser.add_argument("--baseline-prompt-id", default="baseline")
    parser.add_argument("--save-similarity", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)

    split_df, pred_df, groups = load_self_retrieval_inputs(
        split_csv=args.split_csv,
        prediction_csv=args.prediction_csv,
        image_root=args.image_root,
        selected_only=not args.include_unselected,
        check_images=not args.no_check_images,
    )

    split_id = str(split_df["split_id"].iloc[0])

    print(f"[data] split rows: {len(split_df)}")
    print(f"[data] prediction rows: {len(pred_df)}")
    print(f"[data] groups: {len(groups)}")

    encoder = load_encoder(args.scorer, device=args.device)

    image_features = maybe_encode_images(
        encoder=encoder,
        split_df=split_df,
        split_csv=args.split_csv,
        cache_dir=cache_dir,
        batch_size=args.image_batch_size,
        overwrite_cache=args.overwrite_cache,
    )

    all_rows = []

    for group_meta, group_df in groups:
        method_id = str(group_meta.get("method_id", "unknown_method"))
        prompt_id = str(group_meta.get("prompt_id", "unknown_prompt"))

        text_features = maybe_encode_texts(
            encoder=encoder,
            group_meta=group_meta,
            group_df=group_df,
            split_id=split_id,
            prediction_csv=args.prediction_csv,
            cache_dir=cache_dir,
            batch_size=args.text_batch_size,
            overwrite_cache=args.overwrite_cache,
        )

        sim = compute_similarity_matrix(
            text_features=text_features,
            image_features=image_features,
        )

        rows = compute_retrieval_rows(
            similarity_matrix=sim,
            query_df=group_df,
            split_df=split_df,
        )

        for k, v in group_meta.items():
            if k not in rows.columns:
                rows[k] = v

        rows["scorer"] = encoder.name
        all_rows.append(rows)

        if args.save_similarity:
            sim_path = output_dir / f"similarity_matrix__{_safe_name(method_id)}__{_safe_name(prompt_id)}.pt"
            _save_pt(sim.cpu(), sim_path)

    per_caption = pd.concat(all_rows, ignore_index=True)
    summary = summarize_retrieval(per_caption)
    summary = compare_to_baseline(
        summary,
        baseline_prompt_id=args.baseline_prompt_id,
    )

    per_caption_path = output_dir / "retrieval_per_caption.csv"
    summary_path = output_dir / "retrieval_summary.csv"

    per_caption.to_csv(per_caption_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_config(args, output_dir, split_df)

    print(f"[save] {per_caption_path}")
    print(f"[save] {summary_path}")
    print("\n=== retrieval summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()