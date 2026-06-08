#!/usr/bin/env bash
set -euo pipefail

cd /home/jinwoo/CLIPHacker/main
mkdir -p results/predictions logs

export CUDA_VISIBLE_DEVICES=0,1,2,3

SPLIT_ID=flickr30k_retrieval_2000_seed2
SPLIT_CSV=data/splits/${SPLIT_ID}.csv

CONSTRAINED_PROMPT="$(cat results/textgrad_clip/baseline_init_seed1_gpt5mini_s20_b24_constrained/best_prompt.txt)"
MULTIMODAL_PROMPT="$(cat results/textgrad_clip/baseline_init_seed1_gpt5mini_s20_b24_multimodal/best_prompt.txt)"

python methods/qwen_prompt/run_qwen_prompt.py \
  --split-csv "$SPLIT_CSV" \
  --data-root data \
  --out-csv results/predictions/qwen_textgrad_constrained_best__${SPLIT_ID}.csv \
  --method-id qwen_textgrad_constrained_best \
  --run-id qwen_textgrad_constrained_best__${SPLIT_ID} \
  --custom-prompt-id textgrad_constrained_best \
  --custom-prompt-name "TextGrad Constrained Best" \
  --custom-prompt-text "$CONSTRAINED_PROMPT" \
  --temperature 0.0 \
  --max-new-tokens 64 \
  --save-every 20 \
  2>&1 | tee logs/qwen_textgrad_constrained_best__${SPLIT_ID}.log

python methods/qwen_prompt/run_qwen_prompt.py \
  --split-csv "$SPLIT_CSV" \
  --data-root data \
  --out-csv results/predictions/qwen_textgrad_multimodal_best__${SPLIT_ID}.csv \
  --method-id qwen_textgrad_multimodal_best \
  --run-id qwen_textgrad_multimodal_best__${SPLIT_ID} \
  --custom-prompt-id textgrad_multimodal_best \
  --custom-prompt-name "TextGrad Multimodal Best" \
  --custom-prompt-text "$MULTIMODAL_PROMPT" \
  --temperature 0.0 \
  --max-new-tokens 64 \
  --save-every 20 \
  2>&1 | tee logs/qwen_textgrad_multimodal_best__${SPLIT_ID}.log

echo "Done: TextGrad best-prompt inference"
