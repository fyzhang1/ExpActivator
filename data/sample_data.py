import json
import os
import random

random.seed(42)
data_root = './data/Fingertip'
result_json = []
with open('./data/personal_execution/personal_samples.json', 'r') as f:
    rows = json.load(f)
sampled_rows = random.sample(rows, 200)
for user in sampled_rows:
    user_id = str(user['user_id'])
    time = str(user['time'])
    root_path = os.path.join(data_root, user_id, user_id, time) # 'Screenshots_' + time

    prompt = user['originalInstruction']
    ambiguous_prompt = user['ambiguousInstruction']
    with open(os.path.join(root_path, 'action.jsonl'), 'r') as f:
        open_app_data = {
            "user_id": user_id,
            "instruction": prompt,
            "ambiguous_instruction": ambiguous_prompt,
            "id": user_id + '_' + time + '_0',
            "answer": f"open_app(app_name='{user['app_name']}')",
            "img_path": '',
            "app_name": user['app_name']
        }
        result_json.append(open_app_data)
        i = 1
        for line in f:
            d = json.loads(line)
            img_name = list(d.keys())[0]
            action_type = list(d.values())[0]
            if os.path.exists(os.path.join(root_path, 'Screenshots_' + time, img_name)):
                row_data = {
                    "user_id": user_id,
                    "instruction": prompt,
                    'ambiguous_instruction': ambiguous_prompt,
                    "id": user_id + '_' + time + '_' + str(i),
                    "answer": action_type,
                    "img_path": os.path.join(root_path, 'Screenshots_' + time, img_name),
                    "app_name": user['app_name']
                }
            
                result_json.append(row_data)
                i += 1

with open('./data/personal_execution/personal_execution_200.json', 'w', encoding='utf-8') as f:
    json.dump(result_json, f, ensure_ascii=False, indent=4)