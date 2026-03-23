#!/bin/bash

llm_model="gpt-5.1"
timestamp=$(date +"%m%d%H")

eval_false=True

if [ "${eval_false}" = True ]; then
    test_path="./dataset/proactive_suggesstion/proactive_false_samples.csv"
    file_path="./results/proactive/${llm_model}_HIMagent_false_${timestamp}.csv"
else
    test_path="./dataset/proactive_suggesstion/proactive_samples.csv"
    file_path="./results/proactive/${llm_model}_HIMagent_orignal_${timestamp}.csv"
fi

python HIM_agent/eval_agent/proactive_eval.py \
    --llm-model ${llm_model} \
    --base-dir ./data/Fingertip \
    --test-path ${test_path} \
    --result-path ${file_path} \
    --HIM-path ./result/HIM_Agent/routine_memory
