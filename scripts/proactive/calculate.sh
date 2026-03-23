# !/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/eval_proactive/calculate_proactive.py \
    --result-path ./results/proactive/result.csv \
    --embed-model Qwen/Qwen3-Embedding-0.6B \
    --fn_path ./results/proactive/false_negative.csv