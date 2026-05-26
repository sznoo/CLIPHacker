# visualization.py

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {
        "image_id",
        "prompt_id",
        "clip_score",
        "siglip_score",
        "temperature",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return df


def get_prompt_order(df: pd.DataFrame):
    return df[["prompt_id"]].drop_duplicates()["prompt_id"].tolist()


def get_image_order(df: pd.DataFrame):
    if "order" in df.columns:
        return (
            df[["order", "image_id"]]
            .drop_duplicates()
            .sort_values("order")["image_id"]
            .tolist()
        )
    return df[["image_id"]].drop_duplicates()["image_id"].tolist()


def make_delta_matrix(
    df: pd.DataFrame,
    score_col: str,
    baseline_prompt_id: str,
    prompt_order=None,
    image_order=None,
):
    if prompt_order is None:
        prompt_order = get_prompt_order(df)
    if image_order is None:
        image_order = get_image_order(df)

    baseline = (
        df[df["prompt_id"] == baseline_prompt_id][["image_id", score_col]]
        .rename(columns={score_col: "baseline_score"})
    )

    merged = df.merge(baseline, on="image_id", how="left")
    merged["delta"] = merged[score_col] - merged["baseline_score"]

    matrix = merged.pivot(index="image_id", columns="prompt_id", values="delta")
    matrix = matrix.reindex(index=image_order, columns=prompt_order)
    return matrix


def make_abs_matrix(
    df: pd.DataFrame,
    score_col: str,
    prompt_order=None,
    image_order=None,
):
    if prompt_order is None:
        prompt_order = get_prompt_order(df)
    if image_order is None:
        image_order = get_image_order(df)

    matrix = df.pivot(index="image_id", columns="prompt_id", values=score_col)
    matrix = matrix.reindex(index=image_order, columns=prompt_order)
    return matrix


def blue_white_red_cmap():
    return LinearSegmentedColormap.from_list(
        "blue_white_red",
        ["#3b4cc0", "#ffffff", "#b40426"],
        N=256,
    )


def plot_delta_grid(
    matrix: pd.DataFrame,
    title: str,
    save_path: str,
    value_fmt: str = "+.3f",
):
    values = matrix.values.astype(float)

    n_rows, n_cols = values.shape
    fig_w = max(10, n_cols * 1.2)
    fig_h = max(6, n_rows * 0.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    max_abs = np.nanmax(np.abs(values))
    if max_abs == 0:
        max_abs = 1e-6

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
            if np.isnan(val):
                text = "nan"
                color = "black"
            else:
                text = format(val, value_fmt)
                color = "black"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta vs baseline")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_abs_grid(
    matrix: pd.DataFrame,
    title: str,
    save_path: str,
    cmap: str = "YlOrRd",
    value_fmt: str = ".3f",
):
    values = matrix.values.astype(float)

    n_rows, n_cols = values.shape
    fig_w = max(10, n_cols * 1.2)
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

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    threshold = (vmin + vmax) / 2.0

    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]
            if np.isnan(val):
                text = "nan"
                color = "black"
            else:
                text = format(val, value_fmt)
                color = "white" if val > threshold else "black"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Similarity")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(args):
    df = load_results(args.input_csv)

    prompt_order = get_prompt_order(df)
    image_order = get_image_order(df)
    temperature = df["temperature"].iloc[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ["delta", "both"]:
        clip_delta = make_delta_matrix(
            df=df,
            score_col="clip_score",
            baseline_prompt_id=args.baseline_prompt_id,
            prompt_order=prompt_order,
            image_order=image_order,
        )
        siglip_delta = make_delta_matrix(
            df=df,
            score_col="siglip_score",
            baseline_prompt_id=args.baseline_prompt_id,
            prompt_order=prompt_order,
            image_order=image_order,
        )

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
        clip_abs = make_abs_matrix(
            df=df,
            score_col="clip_score",
            prompt_order=prompt_order,
            image_order=image_order,
        )
        siglip_abs = make_abs_matrix(
            df=df,
            score_col="siglip_score",
            prompt_order=prompt_order,
            image_order=image_order,
        )

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)