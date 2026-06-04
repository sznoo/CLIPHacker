import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from scoring import (
    load_scorers,
    caption_length,
    generic_caption_flag,
    compute_prompt_summary,
    make_score_matrix,
)


def score_external_rows(df, score_device):
    scorers = load_scorers(
        device=score_device,
        use_clip=True,
        use_siglip=True,
    )

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring external captions"):
        image = Image.open(row["image_path"]).convert("RGB")
        caption = str(row["caption"])

        out = dict(row)
        out["clip_score"] = scorers["clip"].score(image, caption)
        out["siglip_score"] = scorers["siglip"].score(image, caption)
        out["caption_len"] = caption_length(caption)
        out["generic_flag"] = generic_caption_flag(caption)
        out["cache_hit"] = False

        if "reference_captions" not in out:
            out["reference_captions"] = json.dumps([], ensure_ascii=False)

        rows.append(out)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-csv", type=str, required=True)
    parser.add_argument("--base-run-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--score-device", type=str, default="cuda:3")
    parser.add_argument("--baseline-prompt-id", type=str, default="baseline")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(args.base_run_csv)
    ext_df = pd.read_csv(args.external_csv)

    scored_ext = score_external_rows(ext_df, args.score_device)

    common_cols = sorted(set(base_df.columns) | set(scored_ext.columns))
    for col in common_cols:
        if col not in base_df.columns:
            base_df[col] = None
        if col not in scored_ext.columns:
            scored_ext[col] = None

    combined = pd.concat(
        [base_df[common_cols], scored_ext[common_cols]],
        ignore_index=True,
    )

    raw_path = output_dir / "combined_raw_generations.csv"
    delta_path = output_dir / "combined_raw_generations_with_delta.csv"
    summary_path = output_dir / "combined_prompt_summary.csv"
    clip_matrix_path = output_dir / "combined_clip_matrix.csv"
    siglip_matrix_path = output_dir / "combined_siglip_matrix.csv"

    combined.to_csv(raw_path, index=False)

    summary, combined_delta = compute_prompt_summary(
        combined,
        baseline_prompt_id=args.baseline_prompt_id,
    )

    summary.to_csv(summary_path, index=False)
    combined_delta.to_csv(delta_path, index=False)
    make_score_matrix(combined, "clip_score").to_csv(clip_matrix_path)
    make_score_matrix(combined, "siglip_score").to_csv(siglip_matrix_path)

    print(f"Saved: {raw_path}")
    print(f"Saved: {delta_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()