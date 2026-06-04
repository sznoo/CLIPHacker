# analyze_results.py

import argparse
import numpy as np
import pandas as pd


def bootstrap_mean_ci(x, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    boots = np.array(
        [
            rng.choice(x, size=len(x), replace=True).mean()
            for _ in range(n_boot)
        ]
    )
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    p_mean_gt_0 = float((boots > 0).mean())
    return float(ci_low), float(ci_high), p_mean_gt_0


def summarize_metric(x):
    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "win_rate": float((x > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="results/generalization_100_seed1/raw_generations_with_delta.csv",
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
        default="observer_specific_supported",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    x = df[df["prompt_id"] == args.prompt_id].copy()

    if len(x) == 0:
        raise ValueError(f"No rows found for prompt_id={args.prompt_id}")

    siglip_delta = x["delta_siglip"].dropna().to_numpy()
    clip_delta = x["delta_clip"].dropna().to_numpy()

    siglip = summarize_metric(siglip_delta)
    clip = summarize_metric(clip_delta)

    ci_low, ci_high, p_mean_gt_0 = bootstrap_mean_ci(
        siglip_delta,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    print("=" * 80)
    print("Generalization Benchmark Result")
    print("=" * 80)
    print(f"csv: {args.csv}")
    print(f"prompt_id: {args.prompt_id}")
    print(f"N: {len(x)}")
    print()

    print("[Primary metric: SigLIP delta]")
    print(f"mean_delta_siglip:   {siglip['mean']:.6f}")
    print(f"median_delta_siglip: {siglip['median']:.6f}")
    print(f"std_delta_siglip:    {siglip['std']:.6f}")
    print(f"min_delta_siglip:    {siglip['min']:.6f}")
    print(f"max_delta_siglip:    {siglip['max']:.6f}")
    print(f"siglip_win_rate:     {siglip['win_rate']:.3f}")
    print(f"bootstrap_95ci:      [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"P(mean > 0):         {p_mean_gt_0:.4f}")
    print()

    print("[Secondary metric: CLIP delta]")
    print(f"mean_delta_clip:     {clip['mean']:.6f}")
    print(f"median_delta_clip:   {clip['median']:.6f}")
    print(f"std_delta_clip:      {clip['std']:.6f}")
    print(f"min_delta_clip:      {clip['min']:.6f}")
    print(f"max_delta_clip:      {clip['max']:.6f}")
    print(f"clip_win_rate:       {clip['win_rate']:.3f}")
    print()

    print("[Caption statistics]")
    print(f"mean_caption_len:    {x['caption_len'].mean():.3f}")
    print(f"generic_rate:        {x['generic_flag'].mean():.3f}")
    print()

    print("[Interpretation]")
    if siglip["mean"] >= 0.005 and siglip["win_rate"] >= 0.60:
        print("Strong generalization signal.")
    elif siglip["mean"] > 0 and siglip["win_rate"] > 0.55:
        print("Weak but positive generalization signal.")
    elif abs(siglip["mean"]) < 0.002:
        print("Near-zero gain. Original 10-image result may be sample-sensitive.")
    else:
        print("Negative or unstable signal.")


if __name__ == "__main__":
    main()