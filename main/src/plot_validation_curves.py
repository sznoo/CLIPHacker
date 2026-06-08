# /home/jinwoo/CLIPHacker/main/src/plot_validation_curves.py

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


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_final_metrics(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)

    opt = _load_json(run_dir / "optimization_summary.json")
    test = _load_json(run_dir / "test_result.json")
    best_record = _load_json(run_dir / "best_record.json")

    best_val = None
    if opt is not None:
        best_val = _to_float(opt.get("best_val_score"))
    if best_val is None and best_record is not None:
        best_val = _to_float(best_record.get("val_selection_score"))

    test_delta = None
    test_clip = None
    if test is not None:
        summary = test.get("summary", {})
        test_delta = _to_float(summary.get("mean_delta_clip"))
        test_clip = _to_float(summary.get("mean_clip"))

    return {
        "best_val": best_val,
        "test_delta_clip": test_delta,
        "test_mean_clip": test_clip,
    }


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
            improved = bool(r.get("improved", False))

            accepted_update = (not rejected) and (val_score is not None)

            rows.append({
                "line_idx": line_idx,
                "label": label,
                "mode": mode,
                "step": int(step),
                "plot_step": int(step) + 1,
                "train_score": _to_float(r.get("train_selection_score")),
                "val_score": val_score,
                "best_val_before_step": best_before,
                "rejected": rejected,
                "improved": improved,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Validation Curve")
    parser.add_argument("--ylabel", default="Validation score (mean_delta_clip)")
    parser.add_argument("--include-init", action="store_true")
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument("--fig-w", type=float, default=10.0)
    parser.add_argument("--fig-h", type=float, default=5.5)
    parser.add_argument("--mark-accepted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-final-points", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-x-offset", type=float, default=1.0)
    args = parser.parse_args()

    if not (len(args.runs) == len(args.labels) == len(args.modes)):
        raise ValueError("--runs, --labels, and --modes must have the same length")

    plt.figure(figsize=(args.fig_w, args.fig_h))
    ax = plt.gca()

    max_plot_step = 0
    final_points = []

    for run_dir, label, mode in zip(args.runs, args.labels, args.modes):
        df = load_history(run_dir, label, mode)
        final = load_final_metrics(run_dir)

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

        # No steps-post: keep Ours as ordinary point/line curve.
        line, = ax.plot(
            plot_df["plot_step"],
            plot_df["plot_score"],
            marker="o",
            linewidth=2.2 if mode == "per_step" else 2.8,
            label=f"{label} ({suffix})",
        )
        color = line.get_color()

        # Ours accepted updates: green star.
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

        final_y = final["test_delta_clip"]
        if final_y is None:
            final_y = final["test_mean_clip"]

        if final_y is not None:
            final_points.append({
                "label": label,
                "y": final_y,
                "color": color,
            })

        print("=" * 80)
        print(label, mode)
        print(plot_df[["plot_step", "plot_score"]].to_string(index=False))
        print("Final:", final)

    if args.show_final_points and final_points:
        final_x = max_plot_step + args.final_x_offset

        for i, fp in enumerate(final_points):
            y = fp["y"]
            color = fp["color"]
            label = fp["label"]

            ax.scatter(
                [final_x],
                [y],
                marker="D",
                s=95,
                color=color,
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
                label=f"{label} test",
            )

            ax.annotate(
                _fmt(y),
                xy=(final_x, y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=color,
            )

        ax.axvline(final_x, linestyle="--", linewidth=1, alpha=0.35)
        ax.text(
            final_x,
            ax.get_ylim()[1] if args.ylim is None else args.ylim[1],
            " test",
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