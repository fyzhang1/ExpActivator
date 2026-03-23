#!/bin/bash

# we find that many GUI Agent cannot handle the false proactive suggestions, so we evaluate the proactive suggestion in gpt-series
llm_model="gpt-5.1"
timestamp=$(date +"%m%d%H")

eval_false=True

if [ "${eval_false}" = True ]; then
    test_path="./dataset/proactive_suggestion/proactive_false_samples.csv"
    file_path="./results/proactive/${llm_model}_execution_false_${timestamp}.csv"
else
    test_path="./dataset/proactive_suggestion/proactive_samples.csv"
    file_path="./results/proactive/${llm_model}_execution_orignal_${timestamp}.csv"
fi

python src/eval_proactive/evaluation_proactive.py \
    --llm-model ${llm_model} \
    --base-dir ./data/Fingertip \
    --test-path ${test_path} \
    --result-path ${file_path} \
    --user-profile ./samples/user_profile.csv \
    --csv-file ./samples/total.csv

