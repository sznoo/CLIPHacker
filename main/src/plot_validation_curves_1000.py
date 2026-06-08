# /home/jinwoo/CLIPHacker/main/src/plot_validation_curves_1000.py

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _to_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _fmt(x):
    if x is None:
        return "N/A"
    return f"{float(x):+.4f}"


def load_history(run_dir: str | Path, label: str, mode: str) -> pd.DataFrame:
    run_dir = Path(run_dir)
    history_path = run_dir / "history.jsonl"

    rows = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue

            r = json.loads(line)
            step = r.get("step")
            if step is None or int(step) < 0:
                continue

            val_score = _to_float(r.get("val_selection_score"))
            best_before = _to_float(r.get("best_val_before_step"))
            rejected = bool(r.get("rejected_by_val_gate", False))

            # accepted_update = (not rejected) and (val_score is not None)
            accepted_update = (not rejected) and bool(r.get("improved", False))
            rows.append({
                "line_idx": line_idx,
                "label": label,
                "mode": mode,
                "step": int(step),
                "plot_step": int(step) + 1,
                "val_score": val_score,
                "best_val_before_step": best_before,
                "rejected": rejected,
                "improved": bool(r.get("improved", False)),
                "accepted_update": accepted_update,
            })

    if not rows:
        raise ValueError(f"No valid records found: {history_path}")

    df = pd.DataFrame(rows)

    # Reused output dirs may append duplicate steps. Keep the last record per step.
    df = (
        df.sort_values(["step", "line_idx"])
          .groupby("step", as_index=False)
          .tail(1)
          .sort_values("step")
          .reset_index(drop=True)
    )

    if mode == "per_step":
        df["plot_score"] = df["val_score"]

    elif mode == "best_so_far":
        best_vals = []
        cur = 0.0

        for _, row in df.iterrows():
            before = row["best_val_before_step"]
            val = row["val_score"]

            if pd.notna(before):
                cur = max(cur, float(before))
            if pd.notna(val):
                cur = max(cur, float(val))

            best_vals.append(cur)

        df["plot_score"] = best_vals

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return df


def load_final_csv(path: str | Path, metric: str) -> dict:
    path = Path(path)
    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise ValueError("final CSV must contain a 'label' column")
    if metric not in df.columns:
        raise ValueError(f"final CSV does not contain metric column: {metric}")

    out = {}
    for _, row in df.iterrows():
        out[str(row["label"])] = _to_float(row[metric])
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--final-csv", required=True)
    parser.add_argument("--final-metric", default="delta_clip")

    parser.add_argument("--title", default="Validation Curve + 1000-Set Final Performance")
    parser.add_argument("--ylabel", default="Validation score / 1000-set ΔCLIP")
    parser.add_argument("--include-init", action="store_true")
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument("--fig-w", type=float, default=10.0)
    parser.add_argument("--fig-h", type=float, default=5.5)
    parser.add_argument("--final-x-offset", type=float, default=1.0)
    parser.add_argument("--mark-accepted", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    if not (len(args.runs) == len(args.labels) == len(args.modes)):
        raise ValueError("--runs, --labels, and --modes must have the same length")

    final_scores = load_final_csv(args.final_csv, metric=args.final_metric)

    plt.figure(figsize=(args.fig_w, args.fig_h))
    ax = plt.gca()

    max_plot_step = 0
    final_points = []

    for run_dir, label, mode in zip(args.runs, args.labels, args.modes):
        df = load_history(run_dir, label, mode)
        plot_df = df.dropna(subset=["plot_score"]).copy()

        if args.include_init:
            init = pd.DataFrame([{
                "plot_step": 0,
                "plot_score": 0.0,
                "accepted_update": False,
                "improved": False,
            }])
            plot_df = pd.concat([init, plot_df], ignore_index=True)

        max_plot_step = max(max_plot_step, int(plot_df["plot_step"].max()))
        suffix = "per-step" if mode == "per_step" else "best-so-far"

        line, = ax.plot(
            plot_df["plot_step"],
            plot_df["plot_score"],
            marker="o",
            linewidth=2.2 if mode == "per_step" else 2.8,
            label=f"{label} ({suffix})",
        )
        color = line.get_color()

        if mode == "best_so_far" and args.mark_accepted:
            accepted_steps = set(df[df["accepted_update"]]["plot_step"].tolist())
            accepted_df = plot_df[plot_df["plot_step"].isin(accepted_steps)]

            if len(accepted_df) > 0:
                ax.scatter(
                    accepted_df["plot_step"],
                    accepted_df["plot_score"],
                    marker="*",
                    s=230,
                    color="green",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=6,
                    label=f"{label} accepted updates",
                )

        if label in final_scores and final_scores[label] is not None:
            final_points.append({
                "label": label,
                "y": final_scores[label],
                "color": color,
            })

        print("=" * 80)
        print(label, mode)
        print(plot_df[["plot_step", "plot_score"]].to_string(index=False))
        print(f"1000-set {args.final_metric}:", final_scores.get(label))

    if final_points:
        final_x = max_plot_step + args.final_x_offset

        for fp in final_points:
            ax.scatter(
                [final_x],
                [fp["y"]],
                marker="D",
                s=95,
                color=fp["color"],
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
                label=f"{fp['label']} 1000-set",
            )

            ax.annotate(
                _fmt(fp["y"]),
                xy=(final_x, fp["y"]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=fp["color"],
            )

        ax.axvline(final_x, linestyle="--", linewidth=1, alpha=0.35)
        y_top = args.ylim[1] if args.ylim is not None else ax.get_ylim()[1]
        ax.text(
            final_x,
            y_top,
            " 1000-set",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.axhline(0.0, linestyle=":", linewidth=1)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)

    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])

    ax.legend(loc="lower center")
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()