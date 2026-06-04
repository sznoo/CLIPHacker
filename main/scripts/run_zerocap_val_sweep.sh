#!/usr/bin/env bash
set -e

cd /home/jinwoo/CLIPHacker/main
mkdir -p results/predictions logs scripts

COMMON_ARGS=(
  --split-csv data/splits/flickr30k_main_seed1_val.csv
  --data-root data
  --prompt-id qwen_zerocap
  --prompt-text "Write a short, natural caption for this image."
  --clip-text-prefix "A photo of "
  --max-new-tokens 32
  --min-new-tokens 5
  --clip-loss-temperature 0.01
  --clip-scale 1.0
  --grad-norm-factor 0.9
  --repetition-penalty 1.1
  --end-factor 1.01
  --clip-device cuda:0
  --save-every 5
)

run_one () {
  NAME=$1
  TOP_SIZE=$2
  NUM_ITER=$3
  STEPSIZE=$4
  FUSION=$5
  CE=$6

  echo "===== Running $NAME ====="

  CUDA_VISIBLE_DEVICES=0,1,2,3 python methods/qwen_zerocap/run_qwen_zerocap.py \
    "${COMMON_ARGS[@]}" \
    --out-csv "results/predictions/${NAME}__flickr30k_main_seed1_val.csv" \
    --method-id "$NAME" \
    --prompt-name "$NAME" \
    --top-size "$TOP_SIZE" \
    --num-iterations "$NUM_ITER" \
    --stepsize "$STEPSIZE" \
    --fusion-factor "$FUSION" \
    --ce-scale "$CE" \
    2>&1 | tee "logs/${NAME}__flickr30k_main_seed1_val.log"
}

run_one qwen_zerocap_weak_current     64   1   0.05   0.5   0.5
run_one qwen_zerocap_top128_stable    128  1   0.05   0.7   0.5
run_one qwen_zerocap_medium_guidance  128  1   0.10   0.7   0.3
run_one qwen_zerocap_strong_guidance  128  2   0.10   0.9   0.2
run_one qwen_zerocap_aggressive_clip  256  2   0.15   0.95  0.1
