# experiments/self_retrieval/metrics.py

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch


META_COLUMNS = [
    "split_id",
    "method_id",
    "run_id",
    "prompt_id",
    "prompt_name",
    "order",
    "dataset_index",
    "image_id",
    "image_relpath",
    "caption",
]


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def compute_retrieval_rows(
    similarity_matrix,
    query_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    sim = _to_numpy(similarity_matrix)

    if sim.shape[0] != len(query_df):
        raise ValueError(f"sim rows != query rows: {sim.shape[0]} vs {len(query_df)}")
    if sim.shape[1] != len(split_df):
        raise ValueError(f"sim cols != gallery size: {sim.shape[1]} vs {len(split_df)}")

    gallery = split_df.sort_values("gallery_index").reset_index(drop=True)
    query = query_df.reset_index(drop=True)

    rows = []

    for q_idx, qrow in query.iterrows():
        source_idx = int(qrow["gallery_index"])
        scores = sim[q_idx]

        positive_score = float(scores[source_idx])

        neg_scores = scores.copy()
        neg_scores[source_idx] = -np.inf
        hardest_idx = int(np.argmax(neg_scores))
        hardest_negative_score = float(neg_scores[hardest_idx])

        rank = int(1 + np.sum(scores > positive_score))
        margin = positive_score - hardest_negative_score

        out = {}

        for c in META_COLUMNS:
            if c in qrow.index:
                out[c] = qrow[c]

        out.update(
            {
                "source_gallery_index": source_idx,
                "hardest_negative_gallery_index": hardest_idx,
                "hardest_negative_image_id": gallery.loc[hardest_idx, "image_id"],
                "positive_score": positive_score,
                "hardest_negative_score": hardest_negative_score,
                "margin": margin,
                "rank": rank,
                "hit@1": int(rank <= 1),
                "hit@5": int(rank <= 5),
                "hit@10": int(rank <= 10),
                "reciprocal_rank": 1.0 / rank,
            }
        )

        rows.append(out)

    return pd.DataFrame(rows)


def summarize_retrieval(
    rows_df: pd.DataFrame,
    group_cols: Iterable[str] = ("method_id", "prompt_id"),
) -> pd.DataFrame:
    group_cols = [c for c in group_cols if c in rows_df.columns]

    if group_cols:
        groups = rows_df.groupby(group_cols, sort=False)
    else:
        rows_df = rows_df.copy()
        rows_df["_all"] = "all"
        groups = rows_df.groupby("_all", sort=False)

    summaries = []

    for keys, g in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))

        for c in ["run_id", "prompt_name", "prompt_text", "model"]:
            if c in g.columns:
                row[c] = g[c].iloc[0]

        row.update(
            {
                "n": len(g),
                "r@1": float(g["hit@1"].mean()),
                "r@5": float(g["hit@5"].mean()),
                "r@10": float(g["hit@10"].mean()),
                "mrr": float(g["reciprocal_rank"].mean()),
                "median_rank": float(g["rank"].median()),
                "mean_rank": float(g["rank"].mean()),
                "mean_positive_score": float(g["positive_score"].mean()),
                "mean_hardest_negative_score": float(g["hardest_negative_score"].mean()),
                "mean_margin": float(g["margin"].mean()),
            }
        )

        summaries.append(row)

    return pd.DataFrame(summaries)


def compare_to_baseline(
    summary_df: pd.DataFrame,
    baseline_prompt_id: str = "baseline",
) -> pd.DataFrame:
    df = summary_df.copy()

    base = df[df["prompt_id"] == baseline_prompt_id]
    if len(base) != 1:
        return df

    base = base.iloc[0]

    for c in [
        "r@1",
        "r@5",
        "r@10",
        "mrr",
        "median_rank",
        "mean_rank",
        "mean_positive_score",
        "mean_hardest_negative_score",
        "mean_margin",
    ]:
        if c in df.columns:
            df[f"delta_{c}"] = df[c] - base[c]

    return df