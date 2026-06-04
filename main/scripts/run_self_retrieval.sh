#!/usr/bin/env bash
set -euo pipefail

cd /home/jinwoo/CLIPHacker/main

export CUDA_VISIBLE_DEVICES=0,1,2,3

SPLIT=data/splits/flickr30k_main_seed1_test.csv
IMAGE_ROOT=data
CACHE_DIR=results/self_retrieval/cache
BASE_OUT=results/self_retrieval/flickr30k_main_seed1_test

mkdir -p "$BASE_OUT" "$CACHE_DIR"

python experiments/self_retrieval/run_eval.py \
  --split-csv "$SPLIT" \
  --image-root "$IMAGE_ROOT" \
  --prediction-csv results/predictions/qwen_prompt__flickr30k_main_seed1_test.csv \
  --output-dir "$BASE_OUT/qwen_prompt__siglip" \
  --cache-dir "$CACHE_DIR" \
  --scorer siglip \
  --device cuda:0 \
  --baseline-prompt-id baseline &

python experiments/self_retrieval/run_eval.py \
  --split-csv "$SPLIT" \
  --image-root "$IMAGE_ROOT" \
  --prediction-csv results/predictions/qwen_prompt__flickr30k_main_seed1_test.csv \
  --output-dir "$BASE_OUT/qwen_prompt__clip" \
  --cache-dir "$CACHE_DIR" \
  --scorer clip \
  --device cuda:1 \
  --baseline-prompt-id baseline &

python experiments/self_retrieval/run_eval.py \
  --split-csv "$SPLIT" \
  --image-root "$IMAGE_ROOT" \
  --prediction-csv results/predictions/qwen_zerocap__flickr30k_main_seed1_test.csv \
  --output-dir "$BASE_OUT/qwen_zerocap__siglip" \
  --cache-dir "$CACHE_DIR" \
  --scorer siglip \
  --device cuda:2 \
  --baseline-prompt-id baseline &

python experiments/self_retrieval/run_eval.py \
  --split-csv "$SPLIT" \
  --image-root "$IMAGE_ROOT" \
  --prediction-csv results/predictions/qwen_zerocap__flickr30k_main_seed1_test.csv \
  --output-dir "$BASE_OUT/qwen_zerocap__clip" \
  --cache-dir "$CACHE_DIR" \
  --scorer clip \
  --device cuda:3 \
  --baseline-prompt-id baseline &

wait
echo "Done: self-retrieval evaluation"
