# src/summarize_results.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "method_id",
    "run_id",
    "image_id",
    "prompt_id",
    "prompt_name",
    "caption",
    "clip_score",
    "siglip_score",
    "caption_len",
    "generic_flag",
]


def read_score_csvs(paths):
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df["score_csv"] = str(path)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in score CSVs: {missing}")

    if "selected" in df.columns:
        df = df[df["selected"].astype(int) == 1].copy()

    df["generic_flag"] = (
        df["generic_flag"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return df


def bootstrap_mean_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boots = np.array([
        rng.choice(x, size=len(x), replace=True).mean()
        for _ in range(n_boot)
    ])

    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    p_mean_gt_0 = float((boots > 0).mean())

    return float(ci_low), float(ci_high), p_mean_gt_0


def add_baseline_delta(df, baseline_method_id, baseline_prompt_id):
    baseline = df[
        (df["method_id"] == baseline_method_id)
        & (df["prompt_id"] == baseline_prompt_id)
    ].copy()

    if len(baseline) == 0:
        raise ValueError(
            f"No baseline rows found for "
            f"method_id={baseline_method_id}, prompt_id={baseline_prompt_id}"
        )

    dup = baseline["image_id"].duplicated()
    if dup.any():
        duplicated_ids = baseline.loc[dup, "image_id"].tolist()[:10]
        raise ValueError(f"Duplicate baseline rows for image_id: {duplicated_ids}")

    baseline = baseline[
        ["image_id", "clip_score", "siglip_score"]
    ].rename(
        columns={
            "clip_score": "baseline_clip_score",
            "siglip_score": "baseline_siglip_score",
        }
    )

    merged = df.merge(baseline, on="image_id", how="left")

    missing = merged["baseline_clip_score"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} rows have no matching baseline score.")

    merged["delta_clip"] = merged["clip_score"] - merged["baseline_clip_score"]
    merged["delta_siglip"] = merged["siglip_score"] - merged["baseline_siglip_score"]

    merged["clip_win"] = merged["delta_clip"] > 0
    merged["siglip_win"] = merged["delta_siglip"] > 0

    return merged


def summarize_group(g, n_boot, seed):
    sig_ci_low, sig_ci_high, sig_p = bootstrap_mean_ci(
        g["delta_siglip"].to_numpy(),
        n_boot=n_boot,
        seed=seed,
    )

    clip_ci_low, clip_ci_high, clip_p = bootstrap_mean_ci(
        g["delta_clip"].to_numpy(),
        n_boot=n_boot,
        seed=seed,
    )

    return pd.Series({
        "n": len(g),

        "mean_clip": g["clip_score"].mean(),
        "mean_delta_clip": g["delta_clip"].mean(),
        "median_delta_clip": g["delta_clip"].median(),
        "clip_win_rate": g["clip_win"].mean(),
        "clip_delta_ci_low": clip_ci_low,
        "clip_delta_ci_high": clip_ci_high,
        "clip_p_mean_gt_0": clip_p,

        "mean_siglip": g["siglip_score"].mean(),
        "mean_delta_siglip": g["delta_siglip"].mean(),
        "median_delta_siglip": g["delta_siglip"].median(),
        "siglip_win_rate": g["siglip_win"].mean(),
        "siglip_delta_ci_low": sig_ci_low,
        "siglip_delta_ci_high": sig_ci_high,
        "siglip_p_mean_gt_0": sig_p,

        "mean_caption_len": g["caption_len"].mean(),
        "generic_rate": g["generic_flag"].mean(),
    })


def make_summary(df, n_boot=10000, seed=0):
    group_cols = ["method_id", "prompt_id", "prompt_name"]

    summary = (
        df.groupby(group_cols, dropna=False)
        .apply(lambda g: summarize_group(g, n_boot=n_boot, seed=seed))
        .reset_index()
        .sort_values("mean_delta_siglip", ascending=False)
        .reset_index(drop=True)
    )

    return summary


def save_outputs(df_with_delta, summary, out_prefix):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    delta_path = Path(str(out_prefix) + "_with_delta.csv")
    summary_path = Path(str(out_prefix) + "_summary.csv")

    df_with_delta.to_csv(delta_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved delta rows: {delta_path}")
    print(f"Saved summary:    {summary_path}")

    print("\nSummary:")
    print(summary.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--score-csvs", nargs="+", required=True)
    parser.add_argument("--baseline-method-id", type=str, default="qwen_prompt")
    parser.add_argument("--baseline-prompt-id", type=str, default="baseline")
    parser.add_argument("--out-prefix", type=str, required=True)

    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    df = read_score_csvs(args.score_csvs)

    df_with_delta = add_baseline_delta(
        df,
        baseline_method_id=args.baseline_method_id,
        baseline_prompt_id=args.baseline_prompt_id,
    )

    summary = make_summary(
        df_with_delta,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    save_outputs(
        df_with_delta=df_with_delta,
        summary=summary,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()