# methods/behavior_analysis/evaluate_prompt.py

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def prompt_id_from_text(prompt: str) -> str:
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    return f"behavior_{h}"


def make_batch_split(
    split_csv: str | Path,
    output_csv: Path,
    batch_size: int | None,
    seed: int,
) -> Path:
    df = pd.read_csv(split_csv)

    if batch_size is not None and batch_size < len(df):
        df = df.sample(n=batch_size, random_state=seed).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def build_generation_cmd(
    split_csv: Path,
    output_csv: Path,
    prompt_id: str,
    prompt_text: str,
    prompt_name: str,
    script_path: str,
    data_root: str | Path,
    method_id: str = "behavior_analysis",
) -> list[str]:
    return [
        sys.executable,
        script_path,
        "--split-csv", str(split_csv),
        "--data-root", str(data_root),
        "--out-csv", str(output_csv),
        "--custom-prompt-id", prompt_id,
        "--custom-prompt-name", prompt_name,
        "--custom-prompt-text", prompt_text,
        "--method-id", method_id,
        "--run-id", f"{method_id}__{prompt_id}",
    ]


def build_scoring_cmd(
    prediction_csv: Path,
    score_csv: Path,
    script_path: str,
    data_root: str | Path,
) -> list[str]:
    return [
        sys.executable,
        script_path,
        "--pred-csv", str(prediction_csv),
        "--data-root", str(data_root),
        "--out-csv", str(score_csv),
        "--selected-only",
        "--overwrite",
    ]


def run_cmd(cmd: list[str]):
    subprocess.run(cmd, check=True)


def _selected(df: pd.DataFrame) -> pd.DataFrame:
    if "selected" in df.columns:
        mask = df["selected"].astype(str).str.lower().isin(["true", "1", "yes"])
        return df[mask]
    return df


def _safe_mean(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    return float(df[col].mean())


def _safe_rate(series: pd.Series):
    if len(series) == 0:
        return None
    return float(series.mean())


def summarize_scores(
    score_csv: Path,
    baseline_score_csv: str | Path | None = None,
    data_root: str | Path = "data",
    low_case_k: int = 12,
) -> tuple[dict, list[dict]]:
    df = _selected(pd.read_csv(score_csv)).copy()

    if "caption_len" not in df.columns and "caption" in df.columns:
        df["caption_len"] = df["caption"].astype(str).str.split().str.len()

    summary = {
        "n": int(len(df)),
        "mean_clip": _safe_mean(df, "clip_score"),
        "mean_siglip": _safe_mean(df, "siglip_score"),
        "mean_caption_len": _safe_mean(df, "caption_len"),
        "generic_rate": _safe_mean(df, "generic_flag"),
        "mean_delta_clip": None,
        "mean_delta_siglip": None,
        "clip_win_rate": None,
        "siglip_win_rate": None,
    }

    if baseline_score_csv is not None:
        base = _selected(pd.read_csv(baseline_score_csv)).copy()
        keep_cols = ["image_id", "clip_score", "siglip_score"]
        keep_cols = [c for c in keep_cols if c in base.columns]

        base = base[keep_cols].rename(
            columns={
                "clip_score": "baseline_clip_score",
                "siglip_score": "baseline_siglip_score",
            }
        )

        df = df.merge(base, on="image_id", how="left")

        if "clip_score" in df.columns and "baseline_clip_score" in df.columns:
            df["delta_clip"] = df["clip_score"] - df["baseline_clip_score"]
            valid = df["delta_clip"].dropna()
            summary["mean_delta_clip"] = float(valid.mean())
            summary["clip_win_rate"] = _safe_rate(valid > 0)
        else:
            df["delta_clip"] = None

        if "siglip_score" in df.columns and "baseline_siglip_score" in df.columns:
            df["delta_siglip"] = df["siglip_score"] - df["baseline_siglip_score"]
            valid = df["delta_siglip"].dropna()
            summary["mean_delta_siglip"] = float(valid.mean())
            summary["siglip_win_rate"] = _safe_rate(valid > 0)
        else:
            df["delta_siglip"] = None
    else:
        df["delta_clip"] = None
        df["delta_siglip"] = None

    sort_col = "delta_clip" if baseline_score_csv is not None else "clip_score"
    low = df.sort_values(sort_col, ascending=True).head(low_case_k).copy()

    if "image_relpath" in low.columns:
        root = Path(data_root)

        def to_image_path(p):
            p = Path(str(p))
            return str(p if p.is_absolute() else root / p)

        low["image_path"] = low["image_relpath"].apply(to_image_path)

    case_cols = [
        "image_id",
        "image_relpath",
        "image_path",
        "caption",
        "clip_score",
        "siglip_score",
        "delta_clip",
        "delta_siglip",
        "caption_len",
        "generic_flag",
    ]
    case_cols = [c for c in case_cols if c in low.columns]
    cases = low[case_cols].to_dict("records")

    return summary, cases


def evaluate_prompt(
    prompt: str,
    split_csv: str | Path,
    output_dir: str | Path,
    step: int,
    batch_size: int | None,
    seed: int,
    baseline_score_csv: str | Path | None,
    qwen_script: str,
    score_script: str,
    data_root: str | Path = "data",
    method_id: str = "behavior_analysis",
    prompt_name: str | None = None,
    low_case_k: int = 12,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_id = prompt_id_from_text(prompt)
    if prompt_name is None:
        prompt_name = f"BehaviorAnalysis Step {step}"

    batch_csv = make_batch_split(
        split_csv=split_csv,
        output_csv=output_dir / "batch_split.csv",
        batch_size=batch_size,
        seed=seed,
    )

    prediction_csv = output_dir / "predictions.csv"
    score_csv = output_dir / "scores.csv"

    generation_cmd = build_generation_cmd(
        split_csv=batch_csv,
        output_csv=prediction_csv,
        prompt_id=prompt_id,
        prompt_text=prompt,
        prompt_name=prompt_name,
        script_path=qwen_script,
        data_root=data_root,
        method_id=method_id,
    )
    run_cmd(generation_cmd)

    scoring_cmd = build_scoring_cmd(
        prediction_csv=prediction_csv,
        score_csv=score_csv,
        script_path=score_script,
        data_root=data_root,
    )
    run_cmd(scoring_cmd)

    summary, cases = summarize_scores(
        score_csv=score_csv,
        baseline_score_csv=baseline_score_csv,
        data_root=data_root,
        low_case_k=low_case_k,
    )

    result = {
        "prompt": prompt,
        "prompt_id": prompt_id,
        "prompt_name": prompt_name,
        "method_id": method_id,
        "split_csv": str(split_csv),
        "batch_csv": str(batch_csv),
        "prediction_csv": str(prediction_csv),
        "score_csv": str(score_csv),
        "data_root": str(data_root),
        "summary": summary,
        "cases": cases,
        "generation_cmd": generation_cmd,
        "scoring_cmd": scoring_cmd,
    }

    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result