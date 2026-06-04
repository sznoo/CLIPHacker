# run_experiments.py

import argparse
import hashlib
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


CACHE_KEY_FIELDS = [
    "cache_version",
    "split",
    "dataset_index",
    "image_id",
    "prompt_id",
    "prompt_text",
    "caption_model",
    "temperature",
    "top_p",
    "max_new_tokens",
]


def _norm_value(x):
    if x is None:
        return None
    if isinstance(x, float):
        return round(x, 8)
    return x


def make_cache_key(
    sample: dict,
    prompt: dict,
    args,
) -> str:
    payload = {
        "cache_version": args.cache_version,
        "split": args.split,
        "dataset_index": sample["dataset_index"],
        "image_id": sample["image_id"],
        "prompt_id": prompt["prompt_id"],
        "prompt_text": prompt["text"],
        "caption_model": args.caption_model,
        "temperature": _norm_value(args.temperature),
        "top_p": _norm_value(args.top_p),
        "max_new_tokens": args.max_new_tokens,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def get_cache_path(args) -> Path:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / args.cache_filename


def load_cache(args) -> pd.DataFrame:
    cache_path = get_cache_path(args)
    if args.ignore_cache or not cache_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(cache_path)
    if "cache_key" not in df.columns:
        return pd.DataFrame()

    df = df.drop_duplicates(subset=["cache_key"], keep="last")
    print(f"Loaded global cache: {cache_path} ({len(df)} rows)")
    return df


def save_cache(cache_df: pd.DataFrame, args):
    cache_path = get_cache_path(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if "cache_key" in cache_df.columns:
        cache_df = cache_df.drop_duplicates(subset=["cache_key"], keep="last")

    cache_df.to_csv(cache_path, index=False)


def build_cache_index(cache_df: pd.DataFrame):
    if cache_df.empty or "cache_key" not in cache_df.columns:
        return {}
    return {
        row["cache_key"]: row
        for _, row in cache_df.drop_duplicates("cache_key", keep="last").iterrows()
    }


def sample_metadata(sample: dict, args) -> dict:
    return {
        "order": sample["order"],
        "dataset_index": sample["dataset_index"],
        "image_id": sample["image_id"],
        "image_path": sample.get("image_path", ""),
        "reference_captions": json.dumps(
            sample.get("reference_captions", []),
            ensure_ascii=False,
        ),
        "seed": args.seed,
        "split": args.split,
    }


def make_output_row_from_cached(
    cached_row,
    sample: dict,
    prompt: dict,
    args,
    cache_key: str,
) -> dict:
    row = dict(cached_row)

    # Keep current experiment/sample metadata fresh.
    row.update(sample_metadata(sample, args))

    # Keep current prompt metadata fresh.
    row["prompt_id"] = prompt["prompt_id"]
    row["prompt_name"] = prompt["prompt_name"]
    row["prompt_text"] = prompt["text"]

    # Keep current config metadata fresh.
    row["temperature"] = args.temperature
    row["top_p"] = args.top_p
    row["max_new_tokens"] = args.max_new_tokens
    row["caption_model"] = args.caption_model
    row["cache_key"] = cache_key
    row["cache_hit"] = True

    return row


def make_new_row(
    sample: dict,
    prompt: dict,
    args,
    cache_key: str,
    caption: str,
    clip_score: float,
    siglip_score: float,
) -> dict:
    row = {}
    row.update(sample_metadata(sample, args))
    row.update(
        {
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
            "cache_version": args.cache_version,
            "cache_key": cache_key,
            "cache_hit": False,
        }
    )
    return row


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


def validate_prompts(prompts, baseline_prompt_id: str):
    prompt_ids = [p["prompt_id"] for p in prompts]
    if len(prompt_ids) != len(set(prompt_ids)):
        dup = sorted({x for x in prompt_ids if prompt_ids.count(x) > 1})
        raise ValueError(f"Duplicate prompt_id found: {dup}")

    if baseline_prompt_id not in set(prompt_ids):
        raise ValueError(
            f"baseline_prompt_id='{baseline_prompt_id}' is not in PROMPTS. "
            f"Add it to prompts.py or change --baseline-prompt-id."
        )


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
    validate_prompts(prompts, args.baseline_prompt_id)

    cache_df = load_cache(args)
    cache_index = build_cache_index(cache_df)

    rows = []
    missing_tasks = []

    for sample in samples:
        for prompt in prompts:
            cache_key = make_cache_key(sample, prompt, args)

            if cache_key in cache_index and not args.ignore_cache:
                rows.append(
                    make_output_row_from_cached(
                        cached_row=cache_index[cache_key],
                        sample=sample,
                        prompt=prompt,
                        args=args,
                        cache_key=cache_key,
                    )
                )
            else:
                missing_tasks.append((sample, prompt, cache_key))

    print(
        f"Total tasks: {len(samples) * len(prompts)} | "
        f"cache hits: {len(rows)} | missing: {len(missing_tasks)}"
    )

    if missing_tasks:
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

        new_cache_rows = []

        pbar = tqdm(total=len(missing_tasks), desc="Generating/scoring missing")
        for sample, prompt, cache_key in missing_tasks:
            image = sample["image"]

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

            row = make_new_row(
                sample=sample,
                prompt=prompt,
                args=args,
                cache_key=cache_key,
                caption=caption,
                clip_score=clip_score,
                siglip_score=siglip_score,
            )

            rows.append(row)
            new_cache_rows.append(row)

            if args.save_cache_every > 0 and len(new_cache_rows) % args.save_cache_every == 0:
                cache_df = pd.concat(
                    [cache_df, pd.DataFrame(new_cache_rows)],
                    ignore_index=True,
                )
                save_cache(cache_df, args)
                cache_index = build_cache_index(cache_df)
                new_cache_rows = []

            pbar.update(1)

        pbar.close()

        if new_cache_rows:
            cache_df = pd.concat(
                [cache_df, pd.DataFrame(new_cache_rows)],
                ignore_index=True,
            )
            save_cache(cache_df, args)

    df = pd.DataFrame(rows)

    # Preserve current sample/prompt order in the final output.
    prompt_order = {p["prompt_id"]: i for i, p in enumerate(prompts)}
    df["_prompt_order"] = df["prompt_id"].map(prompt_order)
    df = df.sort_values(["order", "_prompt_order"]).drop(columns=["_prompt_order"])

    save_results(
        df=df,
        output_dir=output_dir,
        baseline_prompt_id=args.baseline_prompt_id,
    )

    print("\nPrompt summary:")
    summary = pd.read_csv(output_dir / "prompt_summary.csv")
    print(summary.to_string(index=False))

    cache_path = get_cache_path(args)
    print(f"\nGlobal cache: {cache_path}")


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

    # Shared cache across experiment folders.
    parser.add_argument("--cache-dir", type=str, default="results/global_cache")
    parser.add_argument("--cache-filename", type=str, default="generation_score_cache.csv")
    parser.add_argument("--cache-version", type=str, default="clip_siglip_v1")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--save-cache-every", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.score_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --score-device uses cuda.")

    run(args)