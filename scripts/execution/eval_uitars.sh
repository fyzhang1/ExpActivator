# !/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_path="ByteDance-Seed/UI-TARS-1.5-7B"
device="cuda"
timestamp=$(date +"%m%d%H")

ambiguous=True
if [ "$ambiguous" = True ]; then
    save_path="./results/execution/UI_TARS_ambiguous_${timestamp}.json"
else
    save_path="./results/execution/UI_TARS_${timestamp}.json"
fi

python src/eval_execution/evaluation_ui_tars.py \
    --device ${device} \
    --test-path ./data/personal_execution/personal_execution.json \
    --model-path ${model_path} \
    --ambiguous ${ambiguous} \
    --save-file ${save_path}