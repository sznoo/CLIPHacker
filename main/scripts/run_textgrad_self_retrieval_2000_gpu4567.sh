#!/usr/bin/env bash
set -euo pipefail

cd /home/jinwoo/CLIPHacker/main
export CUDA_VISIBLE_DEVICES=4,5,6,7

SPLIT_ID=flickr30k_retrieval_2000_seed2
SPLIT=data/splits/${SPLIT_ID}.csv
CACHE_DIR=results/self_retrieval/cache
BASE_OUT=results/self_retrieval/${SPLIT_ID}

mkdir -p "$BASE_OUT" "$CACHE_DIR"

python experiments/self_retrieval/run_eval.py \
  --split-csv "$SPLIT" \
  --image-root data \
  --prediction-csv results/predictions/qwen_textgrad_constrained__${SPLIT_ID}.csv \
  --output-dir "$BASE_OUT/qwen_textgrad_constrained__siglip" \
  --cache-dir "$CACHE_DIR" \
  --scorer siglip \
  --device cuda:0 &

python experiments/self_retrieval/run_eval.py \
  --split-csv "$SPLIT" \
  --image-root data \
  --prediction-csv results/predictions/qwen_textgrad_constrained__${SPLIT_ID}.csv \
  --output-dir "$BASE_OUT/qwen_textgrad_constrained__clip" \
  --cache-dir "$CACHE_DIR" \
  --scorer clip \
  --device cuda:1 &

python experiments/self_retrieval/run_eval.py \
  --split-csv "$SPLIT" \
  --image-root data \
  --prediction-csv results/predictions/qwen_textgrad_multimodal__${SPLIT_ID}.csv \
  --output-dir "$BASE_OUT/qwen_textgrad_multimodal__siglip" \
  --cache-dir "$CACHE_DIR" \
  --scorer siglip \
  --device cuda:2 &

python experiments/self_retrieval/run_eval.py \
  --split-csv "$SPLIT" \
  --image-root data \
  --prediction-csv results/predictions/qwen_textgrad_multimodal__${SPLIT_ID}.csv \
  --output-dir "$BASE_OUT/qwen_textgrad_multimodal__clip" \
  --cache-dir "$CACHE_DIR" \
  --scorer clip \
  --device cuda:3 &

wait
echo "Done: TextGrad self-retrieval on physical GPUs 4,5,6,7"
