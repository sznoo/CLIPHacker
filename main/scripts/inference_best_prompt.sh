cd /home/jinwoo/CLIPHacker/main

CUDA_VISIBLE_DEVICES=0,1,2,3 python methods/qwen_prompt/run_qwen_prompt.py \
  --split-csv data/splits/flickr30k_main_seed1_test.csv \
  --data-root data \
  --out-csv results/predictions/qwen_prompt__flickr30k_main_seed1_test.csv \
  --prompt-ids baseline observer_specific_supported \
  --temperature 0.0 \
  --max-new-tokens 64