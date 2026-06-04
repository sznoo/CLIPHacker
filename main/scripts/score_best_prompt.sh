CUDA_VISIBLE_DEVICES=0,1,2,3 python src/scoring.pyCUDA_VISIBLE_DEVICES=0,1,2,3 python src/score_predictions.py \
  --pred-csv results/predictions/qwen_prompt__flickr30k_main_seed1_test.csv \
  --data-root data \
  --out-csv results/scores/qwen_prompt__flickr30k_main_seed1_test.csv \
  --device cuda:0 \
  --batch-size 8 \
  --selected-only