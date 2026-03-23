# !/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_path="Qwen/Qwen3-VL-8B-Instruct"
device="cuda"
timestamp=$(date +"%m%d%H")

ambiguous=True
if [ "$ambiguous" = True ]; then
    save_path="./results/execution/Qwen3VL_ambiguous_${timestamp}.json"
else
    save_path="./results/execution/Qwen3VL_${timestamp}.json"
fi

python src/eval_execution/evaluation_qwen3vl.py \
    --device ${device} \
    --test-path ./data/personal_execution/personal_execution.json \
    --model-path ${model_path} \
    --ambiguous ${ambiguous} \
    --save-file ${save_path}