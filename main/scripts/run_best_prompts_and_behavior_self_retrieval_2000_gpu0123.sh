#!/usr/bin/env bash
set -euo pipefail

cd /home/jinwoo/CLIPHacker/main
mkdir -p results/predictions logs results/tmp_prompts

export CUDA_VISIBLE_DEVICES=0,1,2,3

SPLIT_ID=flickr30k_retrieval_2000_seed2
SPLIT_CSV=data/splits/${SPLIT_ID}.csv
CACHE_DIR=results/self_retrieval/cache
BASE_OUT=results/self_retrieval/${SPLIT_ID}

CONSTRAINED_PROMPT_PATH=results/textgrad_clip/baseline_init_seed1_gpt5mini_s20_b24_constrained/best_prompt.txt
MULTIMODAL_PROMPT_PATH=results/textgrad_clip/baseline_init_seed1_gpt5mini_s20_b24_multimodal/best_prompt.txt
BEHAVIOR_RECORD=results/behavior_analysis/qwen_residual_full_seed1_s20_b24/best_record.json
BEHAVIOR_PROMPT_PATH=results/tmp_prompts/qwen_residual_full_seed1_s20_b24_best_prompt.txt

python - <<PY
import json
from pathlib import Path

record_path = Path("${BEHAVIOR_RECORD}")
out_path = Path("${BEHAVIOR_PROMPT_PATH}")
record = json.loads(record_path.read_text())

for key in ["new_prompt", "prompt", "best_prompt", "selected_prompt"]:
    if key in record and isinstance(record[key], str) and record[key].strip():
        prompt = record[key]
        break
else:
    raise KeyError(f"No prompt-like key found in {record_path}. Keys: {list(record.keys())}")

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(prompt)
print(f"Saved behavior prompt: {out_path}")
print(prompt[:500])
PY

run_infer () {
  local prompt_path="$1"
  local method_id="$2"
  local prompt_id="$3"
  local prompt_name="$4"

  python methods/qwen_prompt/run_qwen_prompt.py \
    --split-csv "$SPLIT_CSV" \
    --data-root data \
    --out-csv results/predictions/${method_id}__${SPLIT_ID}.csv \
    --method-id "$method_id" \
    --run-id ${method_id}__${SPLIT_ID} \
    --custom-prompt-id "$prompt_id" \
    --custom-prompt-name "$prompt_name" \
    --custom-prompt-text "$(cat "$prompt_path")" \
    --temperature 0.0 \
    --max-new-tokens 64 \
    --save-every 20 \
    2>&1 | tee logs/${method_id}__${SPLIT_ID}.log
}

run_eval () {
  local method_id="$1"
  local scorer="$2"
  local device="$3"

  python experiments/self_retrieval/run_eval.py \
    --split-csv "$SPLIT_CSV" \
    --image-root data \
    --prediction-csv results/predictions/${method_id}__${SPLIT_ID}.csv \
    --output-dir "$BASE_OUT/${method_id}__${scorer}" \
    --cache-dir "$CACHE_DIR" \
    --scorer "$scorer" \
    --device "$device"
}

echo "=== Inference: TextGrad constrained best ==="
run_infer "$CONSTRAINED_PROMPT_PATH" \
  qwen_textgrad_constrained_best \
  textgrad_constrained_best \
  "TextGrad Constrained Best"

echo "=== Inference: TextGrad multimodal best ==="
run_infer "$MULTIMODAL_PROMPT_PATH" \
  qwen_textgrad_multimodal_best \
  textgrad_multimodal_best \
  "TextGrad Multimodal Best"

echo "=== Inference: Behavior residual best ==="
run_infer "$BEHAVIOR_PROMPT_PATH" \
  qwen_behavior_residual_best \
  behavior_residual_best \
  "Behavior Residual Best"

echo "=== Retrieval eval batch 1 ==="
run_eval qwen_textgrad_constrained_best siglip cuda:0 &
run_eval qwen_textgrad_constrained_best clip   cuda:1 &
run_eval qwen_textgrad_multimodal_best   siglip cuda:2 &
run_eval qwen_textgrad_multimodal_best   clip   cuda:3 &
wait

echo "=== Retrieval eval batch 2 ==="
run_eval qwen_behavior_residual_best siglip cuda:0 &
run_eval qwen_behavior_residual_best clip   cuda:1 &
wait

echo "=== Combined summary ==="
python - <<'PY'
import pandas as pd
from pathlib import Path

split_id = "flickr30k_retrieval_2000_seed2"
base = Path("results/self_retrieval") / split_id

methods = [
    "qwen_prompt",
    "qwen_zerocap_weak_current",
    "qwen_textgrad_constrained_best",
    "qwen_textgrad_multimodal_best",
    "qwen_behavior_residual_best",
]

dfs = []
for m in methods:
    for scorer in ["clip", "siglip"]:
        p = base / f"{m}__{scorer}/retrieval_summary.csv"
        if not p.exists():
            print(f"[missing] {p}")
            continue
        df = pd.read_csv(p)
        df.insert(0, "scorer", scorer)
        df.insert(1, "source", m)
        dfs.append(df)

out = pd.concat(dfs, ignore_index=True)

cols = [
    "scorer", "source", "method_id", "prompt_id", "prompt_name", "n",
    "r@1", "r@5", "r@10", "mrr",
    "median_rank", "mean_rank",
    "mean_positive_score",
    "mean_hardest_negative_score",
    "mean_margin",
]
cols = [c for c in cols if c in out.columns]

out = out[cols].sort_values(
    ["scorer", "r@1", "mrr", "mean_margin"],
    ascending=[True, False, False, False],
)

print(out.to_string(index=False))

save_path = base / "combined_summary_best_and_behavior.csv"
out.to_csv(save_path, index=False)
print(f"\nSaved: {save_path}")
PY

echo "Done."
