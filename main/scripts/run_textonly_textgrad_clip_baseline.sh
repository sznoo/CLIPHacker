#!/usr/bin/env bash
set -euo pipefail

cd /home/jinwoo/CLIPHacker/main

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m methods.textgrad_clip.optimize_textgrad_clip \
  --train-split-csv data/splits/flickr30k_main_seed1_train.csv \
  --val-split-csv data/splits/flickr30k_main_seed1_val.csv \
  --test-split-csv data/splits/flickr30k_main_seed1_test.csv \
  --train-baseline-score-csv results/scores/qwen_prompt__flickr30k_main_seed1_train.csv \
  --val-baseline-score-csv results/scores/qwen_prompt__flickr30k_main_seed1_val.csv \
  --test-baseline-score-csv results/scores/qwen_prompt__flickr30k_main_seed1_test.csv \
  --output-dir results/textgrad_clip/baseline_init_seed1_gpt5mini_s20_b24_constrained \
  --init-prompt "Write a caption for this image." \
  --engine "experimental:gpt-5-mini" \
  --feedback-mode textual \
  --steps 20 \
  --batch-size 24 \
  --max-feedback-cases 5 \
  --gradient-memory 0 \
  --include-init-as-candidate \
  --run-test-at-end