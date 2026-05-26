# run_experiments.py

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from data import get_or_create_samples
from models import load_captioner, generate_caption, DEFAULT_CAPTION_MODEL
from prompts import get_prompts
from scoring import (
    load_scorers,
    caption_length,
    generic_caption_flag,
    compute_prompt_summary,
    make_score_matrix,
)


def save_results(df: pd.DataFrame, output_dir: Path, baseline_prompt_id: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw_generations.csv"
    df.to_csv(raw_path, index=False)

    summary, df_with_delta = compute_prompt_summary(
        df,
        baseline_prompt_id=baseline_prompt_id,
    )

    summary_path = output_dir / "prompt_summary.csv"
    delta_path = output_dir / "raw_generations_with_delta.csv"
    clip_matrix_path = output_dir / "clip_matrix.csv"
    siglip_matrix_path = output_dir / "siglip_matrix.csv"

    summary.to_csv(summary_path, index=False)
    df_with_delta.to_csv(delta_path, index=False)

    clip_matrix = make_score_matrix(df, "clip_score")
    siglip_matrix = make_score_matrix(df, "siglip_score")

    clip_matrix.to_csv(clip_matrix_path)
    siglip_matrix.to_csv(siglip_matrix_path)

    print(f"Saved raw results: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved CLIP matrix: {clip_matrix_path}")
    print(f"Saved SigLIP matrix: {siglip_matrix_path}")


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = get_or_create_samples(
        num_images=args.num_images,
        seed=args.seed,
        split=args.split,
        output_dir=args.sample_dir,
        force_resample=args.force_resample,
    )

    prompts = get_prompts()

    captioner = load_captioner(
        model_name=args.caption_model,
        device_map=args.caption_device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    scorers = load_scorers(
        device=args.score_device,
        use_clip=True,
        use_siglip=True,
    )

    rows = []

    total = len(samples) * len(prompts)
    pbar = tqdm(total=total, desc="Generating/scoring")

    for sample in samples:
        image = sample["image"]

        for prompt in prompts:
            caption = generate_caption(
                captioner=captioner,
                image=image,
                prompt=prompt["text"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            clip_score = scorers["clip"].score(image, caption)
            siglip_score = scorers["siglip"].score(image, caption)

            rows.append(
                {
                    "order": sample["order"],
                    "dataset_index": sample["dataset_index"],
                    "image_id": sample["image_id"],
                    "image_path": sample.get("image_path", ""),
                    "reference_captions": json.dumps(
                        sample.get("reference_captions", []),
                        ensure_ascii=False,
                    ),
                    "prompt_id": prompt["prompt_id"],
                    "prompt_name": prompt["prompt_name"],
                    "prompt_text": prompt["text"],
                    "caption": caption,
                    "clip_score": clip_score,
                    "siglip_score": siglip_score,
                    "caption_len": caption_length(caption),
                    "generic_flag": generic_caption_flag(caption),
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "caption_model": args.caption_model,
                    "seed": args.seed,
                    "split": args.split,
                }
            )

            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(rows)
    save_results(
        df=df,
        output_dir=output_dir,
        baseline_prompt_id=args.baseline_prompt_id,
    )

    print("\nPrompt summary:")
    summary = pd.read_csv(output_dir / "prompt_summary.csv")
    print(summary.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--sample-dir", type=str, default="results/sample_images")
    parser.add_argument("--output-dir", type=str, default="results/smoke_run")
    parser.add_argument("--force-resample", action="store_true")

    parser.add_argument("--caption-model", type=str, default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--caption-device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--attn-implementation", type=str, default=None)

    parser.add_argument("--score-device", type=str, default="cuda:3")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)

    parser.add_argument("--baseline-prompt-id", type=str, default="baseline")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.score_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --score-device uses cuda.")

    run(args)