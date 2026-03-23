# !/bin/bash

model_name="gpt-5.1"
timestamp=$(date +"%m%d%H")

ambiguous=True
if [ "$ambiguous" = True ]; then
    save_path="./results/execution/${model_name}_ambiguous_${timestamp}.json"
else
    save_path="./results/execution/${model_name}_${timestamp}.json"
fi

# we recommend you can sample 200 cases for evaluation api (seed=42).
python src/eval_execution/evaluation_api.py \
    --test-path ./data/personal_execution/personal_execution.json \
    --model-name ${model_name} \
    --ambiguous ${ambiguous} \
    --save-file ${save_path}