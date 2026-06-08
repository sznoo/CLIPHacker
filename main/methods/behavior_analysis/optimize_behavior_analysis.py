# methods/behavior_analysis/optimize_behavior_analysis.py

import argparse
import json
import os
import time
from pathlib import Path

from methods.behavior_analysis import configs as cfg
from methods.behavior_analysis.evaluate_prompt import evaluate_prompt
from methods.behavior_analysis.prompt_updater import update_prompt_from_residuals
from methods.behavior_analysis.residual_probe import build_behavior_residuals


BASELINE_PROMPT = getattr(cfg, "BASELINE_PROMPT", "Write a caption for this image.")
DEFAULT_DATA_ROOT = getattr(cfg, "DEFAULT_DATA_ROOT", "data")
DEFAULT_STEPS = getattr(cfg, "DEFAULT_STEPS", 20)
DEFAULT_BATCH_SIZE = getattr(cfg, "DEFAULT_BATCH_SIZE", 24)
DEFAULT_MAX_RESIDUAL_CASES = getattr(cfg, "DEFAULT_MAX_RESIDUAL_CASES", 5)
DEFAULT_RESIDUALS_PER_CASE = getattr(cfg, "DEFAULT_RESIDUALS_PER_CASE", 2)
DEFAULT_OPTIMIZER_MODEL = getattr(cfg, "DEFAULT_OPTIMIZER_MODEL", "gpt-5-mini")
DEFAULT_RESIDUAL_MODEL = getattr(cfg, "DEFAULT_RESIDUAL_MODEL", None)

DEFAULT_USE_VAL_GATE = getattr(cfg, "DEFAULT_USE_VAL_GATE", True)
DEFAULT_VAL_GATE_BATCH_SIZE = getattr(cfg, "DEFAULT_VAL_GATE_BATCH_SIZE", 24)
DEFAULT_VAL_GATE_TOLERANCE = getattr(cfg, "DEFAULT_VAL_GATE_TOLERANCE", 0.0)
DEFAULT_SKIP_FULL_VAL_IF_REJECTED = getattr(cfg, "DEFAULT_SKIP_FULL_VAL_IF_REJECTED", True)
DEFAULT_INCLUDE_INIT_AS_CANDIDATE = getattr(cfg, "DEFAULT_INCLUDE_INIT_AS_CANDIDATE", True)

OPENAI_API_KEY = getattr(cfg, "OPENAI_API_KEY", None)


def _score_for_selection(summary: dict) -> float:
    if summary.get("mean_delta_clip") is not None:
        return float(summary["mean_delta_clip"])
    return float(summary["mean_clip"])


def _fmt_score(x) -> str:
    if x is None:
        return "N/A"
    return f"{float(x):+.6f}"


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"

    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _print_progress(
    step: int,
    total_steps: int,
    stage: str,
    train_score=None,
    gate_current_score=None,
    gate_candidate_score=None,
    val_score=None,
    best_val_score=None,
    eta_seconds=None,
):
    print(
        "[BehaviorAnalysis] "
        f"step {step}/{total_steps} | "
        f"stage={stage} | "
        f"train_score={_fmt_score(train_score)} | "
        f"gate_current={_fmt_score(gate_current_score)} | "
        f"gate_candidate={_fmt_score(gate_candidate_score)} | "
        f"val_score={_fmt_score(val_score)} | "
        f"best_val={_fmt_score(best_val_score)} | "
        f"eta={_fmt_time(eta_seconds)}",
        flush=True,
    )


def write_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_score_summary(title: str, summary: dict):
    print("=" * 80, flush=True)
    print(f"[BehaviorAnalysis] {title}", flush=True)
    print(f"[BehaviorAnalysis] n={summary.get('n')}", flush=True)
    print(f"[BehaviorAnalysis] mean_clip={_fmt_score(summary.get('mean_clip'))}", flush=True)
    print(f"[BehaviorAnalysis] mean_delta_clip={_fmt_score(summary.get('mean_delta_clip'))}", flush=True)
    print(f"[BehaviorAnalysis] clip_win_rate={_fmt_score(summary.get('clip_win_rate'))}", flush=True)
    print(f"[BehaviorAnalysis] mean_siglip={_fmt_score(summary.get('mean_siglip'))}", flush=True)
    print(f"[BehaviorAnalysis] mean_delta_siglip={_fmt_score(summary.get('mean_delta_siglip'))}", flush=True)
    print(f"[BehaviorAnalysis] siglip_win_rate={_fmt_score(summary.get('siglip_win_rate'))}", flush=True)
    print(f"[BehaviorAnalysis] mean_caption_len={_fmt_score(summary.get('mean_caption_len'))}", flush=True)
    print(f"[BehaviorAnalysis] generic_rate={_fmt_score(summary.get('generic_rate'))}", flush=True)
    print("=" * 80, flush=True)


def print_prompt_block(title: str, prompt: str):
    print("=" * 80, flush=True)
    print(f"[BehaviorAnalysis] {title}", flush=True)
    print(prompt, flush=True)
    print("=" * 80, flush=True)


def _run_val_gate(
    current_prompt: str,
    candidate_prompt: str,
    args,
    out_dir: Path,
    step: int,
) -> dict:
    gate_seed = args.seed + 100000 + step

    current_result = evaluate_prompt(
        prompt=current_prompt,
        split_csv=args.val_split_csv,
        output_dir=out_dir / f"step_{step:02d}" / "val_gate_current",
        step=step,
        batch_size=args.val_gate_batch_size,
        seed=gate_seed,
        baseline_score_csv=args.val_baseline_score_csv,
        qwen_script=args.qwen_script,
        score_script=args.score_script,
        data_root=args.data_root,
        low_case_k=args.low_case_k,
    )
    current_score = _score_for_selection(current_result["summary"])

    candidate_result = evaluate_prompt(
        prompt=candidate_prompt,
        split_csv=args.val_split_csv,
        output_dir=out_dir / f"step_{step:02d}" / "val_gate_candidate",
        step=step,
        batch_size=args.val_gate_batch_size,
        seed=gate_seed,
        baseline_score_csv=args.val_baseline_score_csv,
        qwen_script=args.qwen_script,
        score_script=args.score_script,
        data_root=args.data_root,
        low_case_k=args.low_case_k,
    )
    candidate_score = _score_for_selection(candidate_result["summary"])

    accepted = candidate_score >= current_score - args.val_gate_tolerance

    return {
        "accepted": accepted,
        "current_score": current_score,
        "candidate_score": candidate_score,
        "tolerance": args.val_gate_tolerance,
        "current_summary": current_result["summary"],
        "candidate_summary": candidate_result["summary"],
        "current_score_csv": current_result["score_csv"],
        "candidate_score_csv": candidate_result["score_csv"],
        "current_result_json": str(Path(current_result["score_csv"]).parent / "result.json"),
        "candidate_result_json": str(Path(candidate_result["score_csv"]).parent / "result.json"),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-split-csv", required=True)
    parser.add_argument("--val-split-csv", required=True)
    parser.add_argument("--test-split-csv", default=None)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--train-baseline-score-csv", default=None)
    parser.add_argument("--val-baseline-score-csv", default=None)
    parser.add_argument("--test-baseline-score-csv", default=None)

    parser.add_argument("--init-prompt", default=BASELINE_PROMPT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--optimizer-model", default=DEFAULT_OPTIMIZER_MODEL)
    parser.add_argument(
        "--residual-mode",
        choices=["mock", "qwen"],
        default="mock",
        help="Use mock residuals for loop debugging or Qwen image-conditioned residuals for the real method.",
    )
    parser.add_argument("--residual-model", default=DEFAULT_RESIDUAL_MODEL)
    parser.add_argument("--max-residual-cases", type=int, default=DEFAULT_MAX_RESIDUAL_CASES)
    parser.add_argument("--residuals-per-case", type=int, default=DEFAULT_RESIDUALS_PER_CASE)

    parser.add_argument(
        "--use-val-gate",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_VAL_GATE,
    )
    parser.add_argument("--val-gate-batch-size", type=int, default=DEFAULT_VAL_GATE_BATCH_SIZE)
    parser.add_argument("--val-gate-tolerance", type=float, default=DEFAULT_VAL_GATE_TOLERANCE)
    parser.add_argument(
        "--skip-full-val-if-rejected",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SKIP_FULL_VAL_IF_REJECTED,
    )

    parser.add_argument(
        "--include-init-as-candidate",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INCLUDE_INIT_AS_CANDIDATE,
        help="Initialize best prompt with the initial prompt. If val baseline score is provided, its delta is treated as 0.0.",
    )
    parser.add_argument(
        "--eval-init-full-val",
        action="store_true",
        help="Evaluate the initial prompt on the full val split before optimization.",
    )
    parser.add_argument(
        "--run-test-at-end",
        action="store_true",
        help="Evaluate the selected best-val prompt on the test split after optimization.",
    )

    parser.add_argument("--low-case-k", type=int, default=12)
    parser.add_argument("--qwen-script", default="methods/qwen_prompt/run_qwen_prompt.py")
    parser.add_argument("--score-script", default="src/score_predictions.py")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)

    return parser.parse_args()


def main():
    args = parse_args()

    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "history.jsonl"
    residuals_path = out_dir / "residuals.jsonl"
    prompt_updates_path = out_dir / "prompt_updates.jsonl"

    current_prompt = args.init_prompt
    best_prompt = None
    best_val_score = None
    best_record = None

    write_text(out_dir / "init_prompt.txt", args.init_prompt)

    if args.include_init_as_candidate:
        best_prompt = args.init_prompt
        best_val_score = 0.0 if args.val_baseline_score_csv is not None else None
        best_record = {
            "step": -1,
            "old_prompt": None,
            "candidate_prompt": args.init_prompt,
            "new_prompt": args.init_prompt,
            "gate": None,
            "residual_records": None,
            "update_text": None,
            "train_summary": None,
            "val_summary": None,
            "train_selection_score": None,
            "val_selection_score": best_val_score,
            "best_val_before_step": None,
            "improved": True,
            "note": "initial prompt included as candidate",
        }

        if best_val_score is not None:
            write_text(out_dir / "best_prompt.txt", best_prompt)
            write_json(out_dir / "best_record.json", best_record)

    if args.eval_init_full_val:
        print("=" * 80, flush=True)
        print("[BehaviorAnalysis] initial full val evaluation started", flush=True)
        print("=" * 80, flush=True)

        init_val_result = evaluate_prompt(
            prompt=args.init_prompt,
            split_csv=args.val_split_csv,
            output_dir=out_dir / "init_val",
            step=-1,
            batch_size=None,
            seed=args.seed,
            baseline_score_csv=args.val_baseline_score_csv,
            qwen_script=args.qwen_script,
            score_script=args.score_script,
            data_root=args.data_root,
            low_case_k=args.low_case_k,
        )
        init_val_score = _score_for_selection(init_val_result["summary"])

        best_prompt = args.init_prompt
        best_val_score = init_val_score
        best_record = {
            "step": -1,
            "old_prompt": None,
            "candidate_prompt": args.init_prompt,
            "new_prompt": args.init_prompt,
            "gate": None,
            "residual_records": None,
            "update_text": None,
            "train_summary": None,
            "val_summary": init_val_result["summary"],
            "train_selection_score": None,
            "val_selection_score": init_val_score,
            "best_val_before_step": None,
            "improved": True,
            "note": "initial prompt full val evaluated",
        }

        write_text(out_dir / "best_prompt.txt", best_prompt)
        write_json(out_dir / "best_record.json", best_record)
        write_json(out_dir / "init_val_result.json", init_val_result)
        print_score_summary("initial val result", init_val_result["summary"])

    run_start_time = time.time()

    print("=" * 80, flush=True)
    print("[BehaviorAnalysis] optimization started", flush=True)
    print(f"[BehaviorAnalysis] steps={args.steps}", flush=True)
    print(f"[BehaviorAnalysis] batch_size={args.batch_size}", flush=True)
    print(f"[BehaviorAnalysis] residual_mode={args.residual_mode}", flush=True)
    print(f"[BehaviorAnalysis] max_residual_cases={args.max_residual_cases}", flush=True)
    print(f"[BehaviorAnalysis] residuals_per_case={args.residuals_per_case}", flush=True)
    print(f"[BehaviorAnalysis] optimizer_model={args.optimizer_model}", flush=True)
    print(f"[BehaviorAnalysis] use_val_gate={args.use_val_gate}", flush=True)
    print(f"[BehaviorAnalysis] val_gate_batch_size={args.val_gate_batch_size}", flush=True)
    print(f"[BehaviorAnalysis] val_gate_tolerance={args.val_gate_tolerance}", flush=True)
    print(f"[BehaviorAnalysis] skip_full_val_if_rejected={args.skip_full_val_if_rejected}", flush=True)
    print(f"[BehaviorAnalysis] include_init_as_candidate={args.include_init_as_candidate}", flush=True)
    print(f"[BehaviorAnalysis] eval_init_full_val={args.eval_init_full_val}", flush=True)
    print(f"[BehaviorAnalysis] run_test_at_end={args.run_test_at_end}", flush=True)
    print(f"[BehaviorAnalysis] output_dir={out_dir}", flush=True)
    print("=" * 80, flush=True)

    for step in range(args.steps):
        step_num = step + 1
        step_start_time = time.time()

        completed_steps = step
        eta_seconds = None
        if completed_steps > 0:
            avg_step_time = (time.time() - run_start_time) / completed_steps
            eta_seconds = avg_step_time * (args.steps - completed_steps)

        _print_progress(
            step=step_num,
            total_steps=args.steps,
            stage="train_eval",
            best_val_score=best_val_score,
            eta_seconds=eta_seconds,
        )

        train_result = evaluate_prompt(
            prompt=current_prompt,
            split_csv=args.train_split_csv,
            output_dir=out_dir / f"step_{step:02d}" / "train",
            step=step,
            batch_size=args.batch_size,
            seed=args.seed + step,
            baseline_score_csv=args.train_baseline_score_csv,
            qwen_script=args.qwen_script,
            score_script=args.score_script,
            data_root=args.data_root,
            low_case_k=args.low_case_k,
        )
        train_score = _score_for_selection(train_result["summary"])

        _print_progress(
            step=step_num,
            total_steps=args.steps,
            stage="residual_probe",
            train_score=train_score,
            best_val_score=best_val_score,
            eta_seconds=eta_seconds,
        )

        residual_records = build_behavior_residuals(
            cases=train_result["cases"],
            max_cases=args.max_residual_cases,
            residuals_per_case=args.residuals_per_case,
            mode=args.residual_mode,
            model_name=args.residual_model,
        )

        for record in residual_records:
            write_jsonl(
                residuals_path,
                {
                    "step": step,
                    **record,
                },
            )

        _print_progress(
            step=step_num,
            total_steps=args.steps,
            stage="prompt_update",
            train_score=train_score,
            best_val_score=best_val_score,
            eta_seconds=eta_seconds,
        )

        candidate_prompt, update_text = update_prompt_from_residuals(
            current_prompt=current_prompt,
            residual_records=residual_records,
            train_summary=train_result["summary"],
            model=args.optimizer_model,
        )

        write_jsonl(
            prompt_updates_path,
            {
                "step": step,
                "old_prompt": current_prompt,
                "candidate_prompt": candidate_prompt,
                "update_text": update_text,
                "residual_records": residual_records,
                "train_summary": train_result["summary"],
            },
        )

        gate = None
        gate_accepted = True

        if args.use_val_gate:
            _print_progress(
                step=step_num,
                total_steps=args.steps,
                stage="val_gate",
                train_score=train_score,
                best_val_score=best_val_score,
                eta_seconds=eta_seconds,
            )

            gate = _run_val_gate(
                current_prompt=current_prompt,
                candidate_prompt=candidate_prompt,
                args=args,
                out_dir=out_dir,
                step=step,
            )
            gate_accepted = bool(gate["accepted"])

            _print_progress(
                step=step_num,
                total_steps=args.steps,
                stage="val_gate_done",
                train_score=train_score,
                gate_current_score=gate["current_score"],
                gate_candidate_score=gate["candidate_score"],
                best_val_score=best_val_score,
                eta_seconds=eta_seconds,
            )

        val_result = None
        val_score = None
        improved = False
        new_prompt = current_prompt
        rejected_by_val_gate = args.use_val_gate and not gate_accepted

        if rejected_by_val_gate:
            print(
                "[BehaviorAnalysis] "
                f"step {step_num}/{args.steps} rejected_by_val_gate | "
                f"gate_current={_fmt_score(gate['current_score'])} | "
                f"gate_candidate={_fmt_score(gate['candidate_score'])} | "
                f"tolerance={args.val_gate_tolerance}",
                flush=True,
            )

            if not args.skip_full_val_if_rejected:
                _print_progress(
                    step=step_num,
                    total_steps=args.steps,
                    stage="val_eval_rejected_current",
                    train_score=train_score,
                    best_val_score=best_val_score,
                    eta_seconds=eta_seconds,
                )

                val_result = evaluate_prompt(
                    prompt=current_prompt,
                    split_csv=args.val_split_csv,
                    output_dir=out_dir / f"step_{step:02d}" / "val_rejected_current",
                    step=step,
                    batch_size=None,
                    seed=args.seed,
                    baseline_score_csv=args.val_baseline_score_csv,
                    qwen_script=args.qwen_script,
                    score_script=args.score_script,
                    data_root=args.data_root,
                    low_case_k=args.low_case_k,
                )
                val_score = _score_for_selection(val_result["summary"])
                improved = best_val_score is None or val_score > best_val_score

        else:
            new_prompt = candidate_prompt
            current_prompt = new_prompt

            _print_progress(
                step=step_num,
                total_steps=args.steps,
                stage="val_eval",
                train_score=train_score,
                best_val_score=best_val_score,
                eta_seconds=eta_seconds,
            )

            val_result = evaluate_prompt(
                prompt=new_prompt,
                split_csv=args.val_split_csv,
                output_dir=out_dir / f"step_{step:02d}" / "val",
                step=step,
                batch_size=None,
                seed=args.seed,
                baseline_score_csv=args.val_baseline_score_csv,
                qwen_script=args.qwen_script,
                score_script=args.score_script,
                data_root=args.data_root,
                low_case_k=args.low_case_k,
            )
            val_score = _score_for_selection(val_result["summary"])
            improved = best_val_score is None or val_score > best_val_score

        record = {
            "step": step,
            "old_prompt": train_result["prompt"],
            "candidate_prompt": candidate_prompt,
            "new_prompt": new_prompt,
            "rejected_by_val_gate": rejected_by_val_gate,
            "gate": gate,
            "residual_mode": args.residual_mode,
            "residual_records": residual_records,
            "update_text": update_text,
            "train_summary": train_result["summary"],
            "val_summary": val_result["summary"] if val_result is not None else None,
            "train_selection_score": train_score,
            "val_selection_score": val_score,
            "best_val_before_step": best_val_score,
            "improved": improved,
            "step_seconds": time.time() - step_start_time,
        }

        write_jsonl(history_path, record)

        if improved:
            best_val_score = val_score
            best_prompt = new_prompt
            best_record = record

            write_text(out_dir / "best_prompt.txt", best_prompt)
            write_json(out_dir / "best_record.json", best_record)

        if best_prompt is None:
            best_prompt = current_prompt

        print_prompt_block(
            title=f"current best prompt after step {step_num}/{args.steps}",
            prompt=best_prompt,
        )

        completed_steps = step + 1
        elapsed = time.time() - run_start_time
        avg_step_time = elapsed / completed_steps
        eta_seconds = avg_step_time * (args.steps - completed_steps)

        print(
            "[BehaviorAnalysis] "
            f"step {step_num}/{args.steps} done | "
            f"train_score={_fmt_score(train_score)} | "
            f"val_score={_fmt_score(val_score)} | "
            f"best_val={_fmt_score(best_val_score)} | "
            f"gate_accepted={gate_accepted} | "
            f"improved={improved} | "
            f"step_time={_fmt_time(time.time() - step_start_time)} | "
            f"elapsed={_fmt_time(elapsed)} | "
            f"eta={_fmt_time(eta_seconds)}",
            flush=True,
        )
        print("-" * 80, flush=True)

    final_prompt = current_prompt

    if best_prompt is None:
        best_prompt = final_prompt

    write_text(out_dir / "final_prompt.txt", final_prompt)
    write_text(out_dir / "selected_prompt.txt", best_prompt)

    summary = {
        "best_val_score": best_val_score,
        "best_prompt": best_prompt,
        "final_prompt": final_prompt,
        "best_record": best_record,
        "elapsed_seconds": time.time() - run_start_time,
    }
    write_json(out_dir / "optimization_summary.json", summary)

    if args.run_test_at_end:
        if args.test_split_csv is None:
            raise ValueError("--run-test-at-end requires --test-split-csv")

        print("=" * 80, flush=True)
        print("[BehaviorAnalysis] test evaluation started", flush=True)
        print("[BehaviorAnalysis] selected prompt:", flush=True)
        print(best_prompt, flush=True)
        print("=" * 80, flush=True)

        test_result = evaluate_prompt(
            prompt=best_prompt,
            split_csv=args.test_split_csv,
            output_dir=out_dir / "test_selected",
            step=args.steps,
            batch_size=None,
            seed=args.seed,
            baseline_score_csv=args.test_baseline_score_csv,
            qwen_script=args.qwen_script,
            score_script=args.score_script,
            data_root=args.data_root,
            low_case_k=args.low_case_k,
        )

        write_json(out_dir / "test_result.json", test_result)
        print_score_summary("test selected prompt result", test_result["summary"])

    print("=" * 80, flush=True)
    print("[BehaviorAnalysis] optimization finished", flush=True)
    print(f"[BehaviorAnalysis] best_val_score={_fmt_score(best_val_score)}", flush=True)
    print("[BehaviorAnalysis] best_prompt:", flush=True)
    print(best_prompt, flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()