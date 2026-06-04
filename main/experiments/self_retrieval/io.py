# experiments/self_retrieval/io.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


SPLIT_REQUIRED_COLUMNS = [
    "split_id",
    "order",
    "dataset_index",
    "image_id",
    "image_relpath",
]

PRED_REQUIRED_COLUMNS = [
    "split_id",
    "method_id",
    "run_id",
    "order",
    "dataset_index",
    "image_id",
    "image_relpath",
    "prompt_id",
    "prompt_name",
    "prompt_text",
    "caption",
]


def _require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def load_split(split_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    _require_columns(df, SPLIT_REQUIRED_COLUMNS, "split_csv")

    df = df.sort_values("order").reset_index(drop=True)
    df["gallery_index"] = range(len(df))

    if df["image_id"].duplicated().any():
        dup = df.loc[df["image_id"].duplicated(), "image_id"].head().tolist()
        raise ValueError(f"split_csv has duplicated image_id: {dup}")

    if df["order"].duplicated().any():
        dup = df.loc[df["order"].duplicated(), "order"].head().tolist()
        raise ValueError(f"split_csv has duplicated order: {dup}")

    return df


def load_predictions(
    prediction_csv: str | Path,
    selected_only: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(prediction_csv)
    _require_columns(df, PRED_REQUIRED_COLUMNS, "prediction_csv")

    if selected_only and "selected" in df.columns:
        df = df[df["selected"].astype(int) == 1].copy()

    df = df.sort_values(["prompt_id", "order"]).reset_index(drop=True)
    return df


def resolve_image_paths(
    split_df: pd.DataFrame,
    image_root: str | Path,
    check_exists: bool = True,
) -> pd.DataFrame:
    image_root = Path(image_root)
    rows = []

    for _, row in split_df.iterrows():
        rel = Path(str(row["image_relpath"]))

        candidates = [
            image_root / rel,
            image_root / "data" / rel,
            image_root / rel.name,
        ]

        found = None
        for p in candidates:
            if p.exists():
                found = p
                break

        if found is None:
            if check_exists:
                raise FileNotFoundError(
                    f"Image not found for image_id={row['image_id']}, "
                    f"image_relpath={row['image_relpath']}, "
                    f"tried={[str(p) for p in candidates]}"
                )
            found = candidates[0]

        rows.append(str(found))

    out = split_df.copy()
    out["image_path"] = rows
    return out


def align_predictions_to_split(
    pred_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    meta = split_df[["image_id", "order", "gallery_index"]].copy()

    out = pred_df.merge(
        meta,
        on=["image_id", "order"],
        how="inner",
        validate="many_to_one",
    )

    if len(out) != len(pred_df):
        raise ValueError(
            f"Some predictions were not matched to split: "
            f"pred={len(pred_df)}, matched={len(out)}"
        )

    out = out.sort_values(["prompt_id", "gallery_index"]).reset_index(drop=True)
    return out


def group_predictions(
    pred_df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("method_id", "prompt_id"),
) -> list[tuple[dict, pd.DataFrame]]:
    available_cols = [c for c in group_cols if c in pred_df.columns]
    if not available_cols:
        raise ValueError("No valid group columns found.")

    groups = []
    for keys, g in pred_df.groupby(available_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        meta = dict(zip(available_cols, keys))

        for c in ["run_id", "prompt_name", "prompt_text", "model"]:
            if c in g.columns:
                meta[c] = g[c].iloc[0]

        g = g.sort_values("gallery_index").reset_index(drop=True)
        groups.append((meta, g))

    return groups


def load_self_retrieval_inputs(
    split_csv: str | Path,
    prediction_csv: str | Path,
    image_root: str | Path,
    selected_only: bool = True,
    check_images: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[dict, pd.DataFrame]]]:
    split_df = load_split(split_csv)
    split_df = resolve_image_paths(split_df, image_root, check_exists=check_images)

    pred_df = load_predictions(prediction_csv, selected_only=selected_only)
    pred_df = align_predictions_to_split(pred_df, split_df)

    groups = group_predictions(pred_df)
    return split_df, pred_df, groups