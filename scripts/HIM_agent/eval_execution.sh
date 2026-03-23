# !/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_path="Qwen/Qwen3-VL-8B-Instruct"

device="cuda"
timestamp=$(date +"%m%d%H")

save_path="./results/agent/Qwen3VL_HIMagent_${timestamp}.json"

python HIM_agent/eval_agent/execution_eval.py \
    --device ${device} \
    --test-path ./data/personal_execution/personal_execution.json \
    --model-path ${model_path} \
    --save-file ${save_path} \
    --HIM-path ./result/HIM_Agent/preference_memory