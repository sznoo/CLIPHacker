# methods/textgrad_clip/optimize_textgrad_clip.py

import argparse
import json
import os
import time
from pathlib import Path

import textgrad as tg

from methods.textgrad_clip.configs import (
    BASELINE_PROMPT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_ROOT,
    DEFAULT_ENGINE,
    DEFAULT_FEEDBACK_MODE,
    DEFAULT_GRADIENT_MEMORY,
    DEFAULT_MAX_FEEDBACK_CASES,
    DEFAULT_STEPS,
    OPENAI_API_KEY,
)
from methods.textgrad_clip.evaluate_prompt import evaluate_prompt
from methods.textgrad_clip.feedback import build_feedback_instruction
from methods.textgrad_clip.textgrad_adapter import (
    make_optimizer,
    make_prompt_variable,
    run_textgrad_step,
)


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
    val_score=None,
    best_val_score=None,
    accepted_val_score=None,
    eta_seconds=None,
):
    print(
        "[TextGrad] "
        f"step {step}/{total_steps} | "
        f"stage={stage} | "
        f"train_score={_fmt_score(train_score)} | "
        f"val_score={_fmt_score(val_score)} | "
        f"best_val={_fmt_score(best_val_score)} | "
        f"accepted_val={_fmt_score(accepted_val_score)} | "
        f"eta={_fmt_time(eta_seconds)}",
        flush=True,
    )


def write_jsonl(path: Path, obj: dict):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: dict):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_score_summary(title: str, summary: dict):
    print("=" * 80, flush=True)
    print(f"[TextGrad] {title}", flush=True)
    print(f"[TextGrad] n={summary.get('n')}", flush=True)
    print(f"[TextGrad] mean_clip={_fmt_score(summary.get('mean_clip'))}", flush=True)
    print(f"[TextGrad] mean_delta_clip={_fmt_score(summary.get('mean_delta_clip'))}", flush=True)
    print(f"[TextGrad] clip_win_rate={_fmt_score(summary.get('clip_win_rate'))}", flush=True)
    print(f"[TextGrad] mean_siglip={_fmt_score(summary.get('mean_siglip'))}", flush=True)
    print(f"[TextGrad] mean_delta_siglip={_fmt_score(summary.get('mean_delta_siglip'))}", flush=True)
    print(f"[TextGrad] siglip_win_rate={_fmt_score(summary.get('siglip_win_rate'))}", flush=True)
    print(f"[TextGrad] mean_caption_len={_fmt_score(summary.get('mean_caption_len'))}", flush=True)
    print(f"[TextGrad] generic_rate={_fmt_score(summary.get('generic_rate'))}", flush=True)
    print("=" * 80, flush=True)


def print_prompt_block(title: str, prompt: str | None):
    print("=" * 80, flush=True)
    print(f"[TextGrad] {title}", flush=True)
    print(prompt if prompt is not None else "None", flush=True)
    print("=" * 80, flush=True)


def _make_prompt_and_optimizer(prompt: str, gradient_memory: int):
    prompt_var = make_prompt_variable(prompt)
    optimizer = make_optimizer(
        prompt_var=prompt_var,
        gradient_memory=gradient_memory,
    )
    return prompt_var, optimizer


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
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient-memory", type=int, default=DEFAULT_GRADIENT_MEMORY)

    parser.add_argument(
        "--feedback-mode",
        choices=["textual", "multimodal"],
        default=DEFAULT_FEEDBACK_MODE,
    )
    parser.add_argument(
        "--max-feedback-cases",
        type=int,
        default=DEFAULT_MAX_FEEDBACK_CASES,
    )

    parser.add_argument(
        "--qwen-script",
        default="methods/qwen_prompt/run_qwen_prompt.py",
    )
    parser.add_argument(
        "--score-script",
        default="src/score_predictions.py",
    )

    parser.add_argument(
        "--include-init-as-candidate",
        action="store_true",
        help="Initialize best prompt with the initial prompt. If val baseline score is provided, its delta is treated as 0.0.",
    )
    parser.add_argument(
        "--run-test-at-end",
        action="store_true",
        help="Evaluate the selected prompt on the test split after optimization.",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--update-policy",
        choices=["sequential", "hillclimb"],
        default="sequential",
        help="sequential: keep every TextGrad update; hillclimb: accept only if full-val score improves.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tg.set_backward_engine(args.engine, override=True, cache=False)

    prompt_var, optimizer = _make_prompt_and_optimizer(
        prompt=args.init_prompt,
        gradient_memory=args.gradient_memory,
    )

    history_path = out_dir / "history.jsonl"

    best_prompt = None
    best_val_score = None
    best_record = None

    accepted_prompt = args.init_prompt
    accepted_val_score = None
    accepted_record = None

    if args.include_init_as_candidate:
        init_val_score = 0.0 if args.val_baseline_score_csv is not None else None

        best_prompt = args.init_prompt
        best_val_score = init_val_score

        accepted_prompt = args.init_prompt
        accepted_val_score = init_val_score

        best_record = {
            "step": -1,
            "feedback_mode": args.feedback_mode,
            "update_policy": args.update_policy,
            "old_prompt": None,
            "new_prompt": args.init_prompt,
            "accepted_prompt_before_step": None,
            "accepted_prompt_after_step": args.init_prompt,
            "feedback_instruction": None,
            "textgrad_loss_text": None,
            "train_summary": None,
            "val_summary": None,
            "train_selection_score": None,
            "val_selection_score": init_val_score,
            "accepted_val_before_step": None,
            "accepted_val_after_step": init_val_score,
            "best_val_before_step": None,
            "best_val_after_step": init_val_score,
            "accepted": True,
            "improved": True,
            "note": "initial prompt included as candidate",
        }
        accepted_record = best_record

        if init_val_score is not None:
            write_text(out_dir / "best_prompt.txt", best_prompt)
            write_text(out_dir / "accepted_prompt.txt", accepted_prompt)
            write_json(out_dir / "best_record.json", best_record)
            write_json(out_dir / "accepted_record.json", accepted_record)

    write_text(out_dir / "init_prompt.txt", args.init_prompt)

    run_start_time = time.time()

    print("=" * 80, flush=True)
    print("[TextGrad] optimization started", flush=True)
    print(f"[TextGrad] engine={args.engine}", flush=True)
    print(f"[TextGrad] feedback_mode={args.feedback_mode}", flush=True)
    print(f"[TextGrad] update_policy={args.update_policy}", flush=True)
    print(f"[TextGrad] steps={args.steps}", flush=True)
    print(f"[TextGrad] batch_size={args.batch_size}", flush=True)
    print(f"[TextGrad] max_feedback_cases={args.max_feedback_cases}", flush=True)
    print(f"[TextGrad] gradient_memory={args.gradient_memory}", flush=True)
    print(f"[TextGrad] data_root={args.data_root}", flush=True)
    print(f"[TextGrad] include_init_as_candidate={args.include_init_as_candidate}", flush=True)
    print(f"[TextGrad] run_test_at_end={args.run_test_at_end}", flush=True)
    print(f"[TextGrad] output_dir={out_dir}", flush=True)
    print("=" * 80, flush=True)

    for step in range(args.steps):
        step_num = step + 1
        step_start_time = time.time()

        if args.update_policy == "hillclimb":
            current_prompt = accepted_prompt
            prompt_var, optimizer = _make_prompt_and_optimizer(
                prompt=current_prompt,
                gradient_memory=args.gradient_memory,
            )
        else:
            current_prompt = prompt_var.get_value()

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
            accepted_val_score=accepted_val_score,
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
        )

        train_score = _score_for_selection(train_result["summary"])

        _print_progress(
            step=step_num,
            total_steps=args.steps,
            stage="textgrad_update",
            train_score=train_score,
            best_val_score=best_val_score,
            accepted_val_score=accepted_val_score,
            eta_seconds=eta_seconds,
        )

        feedback_instruction = build_feedback_instruction(
            train_result,
            mode=args.feedback_mode,
            max_cases=args.max_feedback_cases,
        )

        new_prompt, loss_text = run_textgrad_step(
            prompt_var=prompt_var,
            optimizer=optimizer,
            feedback_instruction=feedback_instruction,
        )

        _print_progress(
            step=step_num,
            total_steps=args.steps,
            stage="val_eval",
            train_score=train_score,
            best_val_score=best_val_score,
            accepted_val_score=accepted_val_score,
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
        )

        val_score = _score_for_selection(val_result["summary"])

        best_val_before_step = best_val_score
        accepted_val_before_step = accepted_val_score
        accepted_prompt_before_step = accepted_prompt

        improved = best_val_score is None or val_score > best_val_score

        if args.update_policy == "hillclimb":
            accepted = accepted_val_score is None or val_score > accepted_val_score
        else:
            accepted = True

        if accepted:
            accepted_prompt = new_prompt
            accepted_val_score = val_score
        else:
            prompt_var, optimizer = _make_prompt_and_optimizer(
                prompt=accepted_prompt,
                gradient_memory=args.gradient_memory,
            )

        record = {
            "step": step,
            "feedback_mode": args.feedback_mode,
            "update_policy": args.update_policy,
            "old_prompt": current_prompt,
            "new_prompt": new_prompt,
            "accepted_prompt_before_step": accepted_prompt_before_step,
            "accepted_prompt_after_step": accepted_prompt,
            "feedback_instruction": feedback_instruction,
            "textgrad_loss_text": loss_text,
            "train_summary": train_result["summary"],
            "val_summary": val_result["summary"],
            "train_selection_score": train_score,
            "val_selection_score": val_score,
            "accepted_val_before_step": accepted_val_before_step,
            "accepted_val_after_step": accepted_val_score,
            "best_val_before_step": best_val_before_step,
            "best_val_after_step": best_val_score,
            "accepted": accepted,
            "improved": improved,
            "step_seconds": time.time() - step_start_time,
        }

        if accepted:
            accepted_record = record
            write_text(out_dir / "accepted_prompt.txt", accepted_prompt)
            write_json(out_dir / "accepted_record.json", accepted_record)

        if improved:
            best_val_score = val_score
            best_prompt = new_prompt
            best_record = record

            record["best_val_after_step"] = best_val_score

            write_text(out_dir / "best_prompt.txt", best_prompt)
            write_json(out_dir / "best_record.json", best_record)

        write_jsonl(history_path, record)

        if args.update_policy == "hillclimb":
            prompt_to_show = accepted_prompt
            prompt_title = f"current accepted prompt after step {step_num}/{args.steps}"
        else:
            prompt_to_show = best_prompt
            prompt_title = f"current best prompt after step {step_num}/{args.steps}"

        print_prompt_block(
            title=prompt_title,
            prompt=prompt_to_show,
        )

        completed_steps = step + 1
        elapsed = time.time() - run_start_time
        avg_step_time = elapsed / completed_steps
        eta_seconds = avg_step_time * (args.steps - completed_steps)

        print(
            "[TextGrad] "
            f"step {step_num}/{args.steps} done | "
            f"train_score={_fmt_score(train_score)} | "
            f"val_score={_fmt_score(val_score)} | "
            f"best_val={_fmt_score(best_val_score)} | "
            f"accepted_val={_fmt_score(accepted_val_score)} | "
            f"accepted={accepted} | "
            f"improved={improved} | "
            f"step_time={_fmt_time(time.time() - step_start_time)} | "
            f"elapsed={_fmt_time(elapsed)} | "
            f"eta={_fmt_time(eta_seconds)}",
            flush=True,
        )
        print("-" * 80, flush=True)

    final_prompt = prompt_var.get_value()

    if best_prompt is None:
        best_prompt = final_prompt

    if args.update_policy == "hillclimb":
        selected_prompt = accepted_prompt
        selected_val_score = accepted_val_score
    else:
        selected_prompt = best_prompt
        selected_val_score = best_val_score

    write_text(out_dir / "final_prompt.txt", final_prompt)
    write_text(out_dir / "selected_prompt.txt", selected_prompt)
    write_text(out_dir / "accepted_prompt.txt", accepted_prompt)

    summary = {
        "update_policy": args.update_policy,
        "best_val_score": best_val_score,
        "accepted_val_score": accepted_val_score,
        "selected_val_score": selected_val_score,
        "best_prompt": best_prompt,
        "accepted_prompt": accepted_prompt,
        "final_prompt": final_prompt,
        "selected_prompt": selected_prompt,
        "best_record": best_record,
        "accepted_record": accepted_record,
        "elapsed_seconds": time.time() - run_start_time,
    }
    write_json(out_dir / "optimization_summary.json", summary)

    if args.run_test_at_end:
        if args.test_split_csv is None:
            raise ValueError("--run-test-at-end requires --test-split-csv")

        print("=" * 80, flush=True)
        print("[TextGrad] test evaluation started", flush=True)
        print("[TextGrad] selected prompt:", flush=True)
        print(selected_prompt, flush=True)
        print("=" * 80, flush=True)

        test_result = evaluate_prompt(
            prompt=selected_prompt,
            split_csv=args.test_split_csv,
            output_dir=out_dir / "test_selected",
            step=args.steps,
            batch_size=None,
            seed=args.seed,
            baseline_score_csv=args.test_baseline_score_csv,
            qwen_script=args.qwen_script,
            score_script=args.score_script,
            data_root=args.data_root,
        )

        write_json(out_dir / "test_result.json", test_result)
        print_score_summary("test selected prompt result", test_result["summary"])

    print("=" * 80, flush=True)
    print("[TextGrad] optimization finished", flush=True)
    print(f"[TextGrad] update_policy={args.update_policy}", flush=True)
    print(f"[TextGrad] best_val_score={_fmt_score(best_val_score)}", flush=True)
    print(f"[TextGrad] accepted_val_score={_fmt_score(accepted_val_score)}", flush=True)
    print("[TextGrad] selected_prompt:", flush=True)
    print(selected_prompt, flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()