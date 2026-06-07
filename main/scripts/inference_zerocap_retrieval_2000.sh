#!/usr/bin/env bash
set -e

cd /home/jinwoo/CLIPHacker/main
mkdir -p results/predictions logs

CUDA_VISIBLE_DEVICES=0,1,2,3 python methods/qwen_zerocap/run_qwen_zerocap.py \
  --split-csv data/splits/flickr30k_retrieval_2000_seed2.csv \
  --data-root data \
  --out-csv results/predictions/qwen_zerocap_weak_current__flickr30k_retrieval_2000_seed2.csv \
  --method-id qwen_zerocap_weak_current \
  --prompt-id qwen_zerocap \
  --prompt-name "Qwen ZeroCap Weak Current" \
  --prompt-text "Write a short, natural caption for this image." \
  --clip-text-prefix "A photo of " \
  --max-new-tokens 32 \
  --min-new-tokens 5 \
  --top-size 64 \
  --num-iterations 1 \
  --clip-loss-temperature 0.01 \
  --clip-scale 1.0 \
  --ce-scale 0.5 \
  --stepsize 0.05 \
  --grad-norm-factor 0.9 \
  --fusion-factor 0.5 \
  --repetition-penalty 1.1 \
  --end-factor 1.01 \
  --clip-device cuda:0 \
  --save-every 20 \
  2>&1 | tee logs/qwen_zerocap_weak_current__flickr30k_retrieval_2000_seed2.log
