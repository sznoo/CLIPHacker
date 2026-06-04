# visualization.py

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


REQUIRED_COLS = {
    "image_id",
    "prompt_id",
    "clip_score",
    "siglip_score",
    "temperature",
}


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return df


def get_image_order(df: pd.DataFrame):
    if "order" in df.columns:
        return (
            df[["order", "image_id"]]
            .drop_duplicates()
            .sort_values("order")["image_id"]
            .tolist()
        )
    return df[["image_id"]].drop_duplicates()["image_id"].tolist()


def get_prompt_order(df: pd.DataFrame):
    return df[["prompt_id"]].drop_duplicates()["prompt_id"].tolist()


def validate_baseline(df: pd.DataFrame, baseline_prompt_id: str):
    prompt_ids = set(df["prompt_id"].unique())
    if baseline_prompt_id not in prompt_ids:
        available = sorted(prompt_ids)
        raise ValueError(
            f"baseline_prompt_id='{baseline_prompt_id}' not found in input CSV.\n"
            f"Available prompt_ids:\n{available}\n\n"
            f"Fix: include '{baseline_prompt_id}' in PROMPTS, or pass "
            f"--baseline-prompt-id with one of the available prompt_ids."
        )

    image_ids = set(df["image_id"].unique())
    baseline_image_ids = set(
        df[df["prompt_id"] == baseline_prompt_id]["image_id"].unique()
    )
    missing_images = sorted(image_ids - baseline_image_ids)
    if missing_images:
        raise ValueError(
            f"Baseline prompt '{baseline_prompt_id}' is missing for "
            f"{len(missing_images)} images: {missing_images}"
        )


def compute_delta_columns(df: pd.DataFrame, baseline_prompt_id: str) -> pd.DataFrame:
    validate_baseline(df, baseline_prompt_id)

    baseline = df[df["prompt_id"] == baseline_prompt_id][
        ["image_id", "clip_score", "siglip_score"]
    ].rename(
        columns={
            "clip_score": "baseline_clip_score",
            "siglip_score": "baseline_siglip_score",
        }
    )

    out = df.merge(baseline, on="image_id", how="left")
    out["delta_clip"] = out["clip_score"] - out["baseline_clip_score"]
    out["delta_siglip"] = out["siglip_score"] - out["baseline_siglip_score"]
    return out


def summarize_prompts(df: pd.DataFrame, baseline_prompt_id: str) -> pd.DataFrame:
    df_delta = compute_delta_columns(df, baseline_prompt_id)

    summary = (
        df_delta.groupby("prompt_id")
        .agg(
            mean_clip=("clip_score", "mean"),
            mean_siglip=("siglip_score", "mean"),
            mean_delta_clip=("delta_clip", "mean"),
            mean_delta_siglip=("delta_siglip", "mean"),
            min_delta_clip=("delta_clip", "min"),
            max_delta_clip=("delta_clip", "max"),
            min_delta_siglip=("delta_siglip", "min"),
            max_delta_siglip=("delta_siglip", "max"),
            clip_win_rate=("delta_clip", lambda x: (x > 0).mean()),
            siglip_win_rate=("delta_siglip", lambda x: (x > 0).mean()),
            n=("image_id", "count"),
        )
        .reset_index()
    )

    if "prompt_name" in df_delta.columns:
        names = df_delta[["prompt_id", "prompt_name"]].drop_duplicates()
        summary = summary.merge(names, on="prompt_id", how="left")

    cols = ["prompt_id"]
    if "prompt_name" in summary.columns:
        cols.append("prompt_name")
    cols += [
        "mean_clip",
        "mean_delta_clip",
        "min_delta_clip",
        "max_delta_clip",
        "clip_win_rate",
        "mean_siglip",
        "mean_delta_siglip",
        "min_delta_siglip",
        "max_delta_siglip",
        "siglip_win_rate",
        "n",
    ]

    summary = summary[cols]
    summary = summary.sort_values("mean_delta_siglip", ascending=False)
    return summary


def select_prompt_order(
    df: pd.DataFrame,
    baseline_prompt_id: str,
    sort_by: str,
    top_k: int | None,
):
    base_order = get_prompt_order(df)

    if sort_by == "input":
        ordered = base_order
    else:
        summary = summarize_prompts(df, baseline_prompt_id)

        sort_col_map = {
            "mean_clip": "mean_clip",
            "mean_siglip": "mean_siglip",
            "delta_clip": "mean_delta_clip",
            "delta_siglip": "mean_delta_siglip",
        }
        sort_col = sort_col_map[sort_by]
        ordered = summary.sort_values(sort_col, ascending=False)["prompt_id"].tolist()

    if top_k is not None and top_k > 0:
        keep = ordered[:top_k]
        if baseline_prompt_id not in keep and baseline_prompt_id in ordered:
            keep = [baseline_prompt_id] + keep
        ordered = keep

    return ordered


def make_matrix(df: pd.DataFrame, value_col: str, prompt_order, image_order):
    matrix = df.pivot(index="image_id", columns="prompt_id", values=value_col)
    matrix = matrix.reindex(index=image_order, columns=prompt_order)
    return matrix


def blue_white_red_cmap():
    return LinearSegmentedColormap.from_list(
        "blue_white_red",
        ["#3b4cc0", "#ffffff", "#b40426"],
        N=256,
    )


def _safe_value_range(values: np.ndarray):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("All values are NaN/inf. Check baseline and input CSV.")
    return finite.min(), finite.max()


def plot_delta_grid(
    matrix: pd.DataFrame,
    title: str,
    save_path: Path,
    value_fmt: str = "+.3f",
):
    values = matrix.values.astype(float)
    _, vmax_abs_raw = _safe_value_range(np.abs(values))
    max_abs = vmax_abs_raw if vmax_abs_raw > 0 else 1e-6

    n_rows, n_cols = values.shape
    fig_w = max(10, n_cols * 1.15)
    fig_h = max(6, n_rows * 0.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = blue_white_red_cmap()
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Prompt ID")
    ax.set_ylabel("Image ID")
    ax.set_title(title)

    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]
            text = "nan" if np.isnan(val) else format(val, value_fmt)
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta vs baseline")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_abs_grid(
    matrix: pd.DataFrame,
    title: str,
    save_path: Path,
    cmap: str = "YlOrRd",
    value_fmt: str = ".3f",
):
    values = matrix.values.astype(float)
    vmin, vmax = _safe_value_range(values)
    threshold = (vmin + vmax) / 2.0

    n_rows, n_cols = values.shape
    fig_w = max(10, n_cols * 1.15)
    fig_h = max(6, n_rows * 0.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Prompt ID")
    ax.set_ylabel("Image ID")
    ax.set_title(title)

    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]
            if np.isnan(val):
                text = "nan"
                color = "black"
            else:
                text = format(val, value_fmt)
                color = "white" if val > threshold else "black"

            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Similarity")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_oracle_tables(df_delta: pd.DataFrame, output_dir: Path):
    clip_best = (
        df_delta.sort_values(["image_id", "clip_score"], ascending=[True, False])
        .groupby("image_id")
        .head(1)
        .sort_values("image_id")
    )
    siglip_best = (
        df_delta.sort_values(["image_id", "siglip_score"], ascending=[True, False])
        .groupby("image_id")
        .head(1)
        .sort_values("image_id")
    )

    clip_cols = [
        "image_id",
        "prompt_id",
        "clip_score",
        "baseline_clip_score",
        "delta_clip",
        "caption",
    ]
    siglip_cols = [
        "image_id",
        "prompt_id",
        "siglip_score",
        "baseline_siglip_score",
        "delta_siglip",
        "caption",
    ]

    clip_best[clip_cols].to_csv(output_dir / "oracle_best_clip.csv", index=False)
    siglip_best[siglip_cols].to_csv(output_dir / "oracle_best_siglip.csv", index=False)


def main(args):
    df = load_results(args.input_csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_order = get_image_order(df)
    prompt_order = select_prompt_order(
        df=df,
        baseline_prompt_id=args.baseline_prompt_id,
        sort_by=args.sort_prompts_by,
        top_k=args.top_k,
    )

    temperature = df["temperature"].iloc[0]

    if args.mode in ["delta", "both"] or args.save_summary or args.save_oracle:
        df_delta = compute_delta_columns(df, args.baseline_prompt_id)

        if args.save_summary:
            summary = summarize_prompts(df, args.baseline_prompt_id)
            summary_path = output_dir / "visual_prompt_summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"Saved visual summary: {summary_path}")
            print(summary.to_string(index=False))

        if args.save_oracle:
            save_oracle_tables(df_delta, output_dir)
            print(f"Saved oracle tables: {output_dir / 'oracle_best_clip.csv'}")
            print(f"Saved oracle tables: {output_dir / 'oracle_best_siglip.csv'}")

    if args.mode in ["delta", "both"]:
        clip_delta = make_matrix(df_delta, "delta_clip", prompt_order, image_order)
        siglip_delta = make_matrix(df_delta, "delta_siglip", prompt_order, image_order)

        clip_delta_path = output_dir / "clip_delta_grid.png"
        siglip_delta_path = output_dir / "siglip_delta_grid.png"

        plot_delta_grid(
            matrix=clip_delta,
            title=f"CLIP Delta Grid vs {args.baseline_prompt_id} (temperature={temperature})",
            save_path=clip_delta_path,
        )
        plot_delta_grid(
            matrix=siglip_delta,
            title=f"SigLIP Delta Grid vs {args.baseline_prompt_id} (temperature={temperature})",
            save_path=siglip_delta_path,
        )

        print(f"Saved CLIP delta figure: {clip_delta_path}")
        print(f"Saved SigLIP delta figure: {siglip_delta_path}")

    if args.mode in ["absolute", "both"]:
        clip_abs = make_matrix(df, "clip_score", prompt_order, image_order)
        siglip_abs = make_matrix(df, "siglip_score", prompt_order, image_order)

        clip_abs_path = output_dir / "clip_similarity_grid.png"
        siglip_abs_path = output_dir / "siglip_similarity_grid.png"

        plot_abs_grid(
            matrix=clip_abs,
            title=f"CLIP Similarity Grid (temperature={temperature})",
            save_path=clip_abs_path,
        )
        plot_abs_grid(
            matrix=siglip_abs,
            title=f"SigLIP Similarity Grid (temperature={temperature})",
            save_path=siglip_abs_path,
        )

        print(f"Saved CLIP absolute figure: {clip_abs_path}")
        print(f"Saved SigLIP absolute figure: {siglip_abs_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-csv",
        type=str,
        default="results/smoke_run/raw_generations.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/smoke_run/figures",
    )
    parser.add_argument(
        "--baseline-prompt-id",
        type=str,
        default="baseline",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["delta", "absolute", "both"],
        default="delta",
    )
    parser.add_argument(
        "--sort-prompts-by",
        type=str,
        choices=["input", "mean_clip", "mean_siglip", "delta_clip", "delta_siglip"],
        default="input",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="If set, visualize only top-k prompts after sorting. Baseline is kept.",
    )
    parser.add_argument("--save-summary", action="store_true")
    parser.add_argument("--save-oracle", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)