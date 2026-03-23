# !/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_path="mPLUG/GUI-Owl-7B"
device="cuda"
timestamp=$(date +"%m%d%H")

ambiguous=True
if [ "$ambiguous" = True ]; then
    save_path="./results/execution/GUIOwl_ambiguous_${timestamp}.json"
else
    save_path="./results/execution/GUIOwl_${timestamp}.json"
fi

python src/eval_execution/evaluation_owl.py \
    --device ${device} \
    --test-path ./data/personal_execution/personal_execution.json \
    --model-path ${model_path} \
    --ambiguous ${ambiguous} \
    --save-file ${save_path}