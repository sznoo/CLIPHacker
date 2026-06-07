#!/usr/bin/env bash
set -euo pipefail

cd /home/jinwoo/CLIPHacker/main
export CUDA_VISIBLE_DEVICES=0,1,2,3

SPLIT_ID=flickr30k_retrieval_2000_seed2
SPLIT=data/splits/${SPLIT_ID}.csv
PRED=results/predictions/qwen_prompt__${SPLIT_ID}.csv
CACHE_DIR=results/self_retrieval/cache
BASE_OUT=results/self_retrieval/${SPLIT_ID}

mkdir -p "$BASE_OUT" "$CACHE_DIR"

python experiments/self_retrieval/run_eval.py \
  --split-csv "$SPLIT" \
  --image-root data \
  --prediction-csv "$PRED" \
  --output-dir "$BASE_OUT/qwen_prompt__siglip" \
  --cache-dir "$CACHE_DIR" \
  --scorer siglip \
  --device cuda:0 \
  --baseline-prompt-id baseline &

python experiments/self_retrieval/run_eval.py \
  --split-csv "$SPLIT" \
  --image-root data \
  --prediction-csv "$PRED" \
  --output-dir "$BASE_OUT/qwen_prompt__clip" \
  --cache-dir "$CACHE_DIR" \
  --scorer clip \
  --device cuda:1 \
  --baseline-prompt-id baseline &

wait
echo "Done: self-retrieval 2000"
