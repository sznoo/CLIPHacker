#!/usr/bin/env bash
set -euo pipefail

cd /home/jinwoo/CLIPHacker/main

mkdir -p results/predictions results/scores

for SPLIT in train val; do
  SPLIT_ID="flickr30k_main_seed1_${SPLIT}"

  CUDA_VISIBLE_DEVICES=0,1,2,3 python methods/qwen_prompt/run_qwen_prompt.py \
    --split-csv "data/splits/${SPLIT_ID}.csv" \
    --out-csv "results/predictions/qwen_prompt__${SPLIT_ID}.csv" \
    --prompt-ids baseline \
    --method-id qwen_prompt \
    --run-id "qwen_prompt__${SPLIT_ID}"

  CUDA_VISIBLE_DEVICES=0 python src/score_predictions.py \
    --pred-csv "results/predictions/qwen_prompt__${SPLIT_ID}.csv" \
    --out-csv "results/scores/qwen_prompt__${SPLIT_ID}.csv" \
    --selected-only \
    --overwrite
done
