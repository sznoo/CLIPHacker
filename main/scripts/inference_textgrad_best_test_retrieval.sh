#!/usr/bin/env bash
set -euo pipefail

cd /home/jinwoo/CLIPHacker/main

RUN_DIR="results/textgrad_clip/baseline_init_seed1_gpt5mini_s20_b24"
PROMPT_TEXT="$(cat ${RUN_DIR}/best_prompt.txt)"

METHOD_ID="textgrad_clip"
PROMPT_ID="textgrad_best_s20_b24"
PROMPT_NAME="TextGrad Best s20 b24"

mkdir -p results/predictions results/scores

# 1. Test split inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python methods/qwen_prompt/run_qwen_prompt.py \
  --split-csv data/splits/flickr30k_main_seed1_test.csv \
  --out-csv results/predictions/textgrad_best_s20_b24__flickr30k_main_seed1_test.csv \
  --custom-prompt-id "${PROMPT_ID}" \
  --custom-prompt-name "${PROMPT_NAME}" \
  --custom-prompt-text "${PROMPT_TEXT}" \
  --method-id "${METHOD_ID}" \
  --run-id textgrad_best_s20_b24__flickr30k_main_seed1_test

# 2. Test split scoring
CUDA_VISIBLE_DEVICES=0 python src/score_predictions.py \
  --pred-csv results/predictions/textgrad_best_s20_b24__flickr30k_main_seed1_test.csv \
  --out-csv results/scores/textgrad_best_s20_b24__flickr30k_main_seed1_test.csv \
  --selected-only \
  --overwrite

# 3. Retrieval 2000 split inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python methods/qwen_prompt/run_qwen_prompt.py \
  --split-csv data/splits/flickr30k_retrieval_2000_seed2.csv \
  --out-csv results/predictions/textgrad_best_s20_b24__flickr30k_retrieval_2000_seed2.csv \
  --custom-prompt-id "${PROMPT_ID}" \
  --custom-prompt-name "${PROMPT_NAME}" \
  --custom-prompt-text "${PROMPT_TEXT}" \
  --method-id "${METHOD_ID}" \
  --run-id textgrad_best_s20_b24__flickr30k_retrieval_2000_seed2

# 4. Retrieval split scoring, optional but useful
CUDA_VISIBLE_DEVICES=0 python src/score_predictions.py \
  --pred-csv results/predictions/textgrad_best_s20_b24__flickr30k_retrieval_2000_seed2.csv \
  --out-csv results/scores/textgrad_best_s20_b24__flickr30k_retrieval_2000_seed2.csv \
  --selected-only \
  --overwrite
