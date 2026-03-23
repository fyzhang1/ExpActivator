import pandas as pd
import os
import json
import csv
import traceback
import argparse
from tqdm import tqdm
from openai import AsyncOpenAI
import asyncio
from time import time
import re

from src.utils import remove_think_tags, clean_llm_json
from src.eval_execution.utils import load_user_profile

client = AsyncOpenAI(
    api_key="",
    base_url=""
)

parser = argparse.ArgumentParser()
parser.add_argument("--llm-model")
parser.add_argument("--base-dir")
parser.add_argument("--test-path")
parser.add_argument("--result-path")
parser.add_argument("--HIM-path")

args = parser.parse_args()

total_record = pd.read_csv('./samples/total.csv')
result = {}

def get_routine_memory(user_id):
    user_routines = []
    with open(os.path.join(args.HIM_path, f'user_{user_id}_routine_memory.json'), 'r', encoding='utf-8') as f:
        user_routines = json.load(f)
    if len(user_routines) == 0:
        return user_routines
    for row in user_routines:
        center_text = row['cluster_content']["example_intents"]
        time = row['collect_time']
        cluster_len = row['cluster_content']['cluster_len']

        user_routines.append(f"routine_intent: {center_text}, time_distribution: {time}, occurrence_count: {cluster_len}")

    return user_routines

def get_recent_intents(user_id, time):
    recent_record = []

    user_record = total_record[total_record['user_id'] == int(user_id)]
    current_date = str(time).split('_')[0]

    user_record = user_record[user_record['time'] < str(time)]
    user_record['date'] = user_record['time'].str.split('_').str[0]
    user_record = user_record[user_record['date'] == current_date]

    user_record = user_record.sort_values(by='time', ascending=False)
    user_record = user_record.head(3)

    for _, row in user_record.iterrows():
        r_time = row['time']
        r_scenario = row['scenario']
        r_intent = row['intentDescription']
        recent_record.append(f'Time: {r_time}, Scenario: {r_scenario}, intent: {r_intent}')

    return recent_record

def get_prompt(profile, time, scenario, recent_records, generalized_routine):
    prompt = f"""You are skilled at analyzing user history. You are given summarized user daily routine. Your task is to determine whether the current user state requires an proactive suggestions. 

## Note
- If you decide that a recommendation is needed, output a suitable user instruction for the GUI agent to execute.
- If neither the time nor the scenario matches well with any entry in the Summarized_routine, only output False.
- Express the user's intent unambiguously in one Chinese sentence, ensuring the expression faithfully reflects the user's original intent without adding any extra description.
- Do not output any explanation including time and scenario information. Only output the user's intent.  

## Input
User_profile: {profile}
Time: {time}
Scenario: {scenario}
Summarized_routine: {generalized_routine}

The user's intent:
"""
    
    return prompt


async def llm_client(prompt):
    try:
        completion = await client.chat.completions.create(
            model=args.llm_model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                ],
                    }
            ]
        )

        return completion.choices[0].message.content.strip(), completion.usage.total_tokens

    except Exception as e:

        print(f"Error occurred: {e}")
        print(traceback.format_exc())

        return 'ERROR', 0


async def main(episode_path, result_path, user_id, intent, record_time, scenario):
    user_profile = load_user_profile(user_id)
    recent_records = str(get_recent_intents(user_id, record_time))
    generalized_routine = str(get_routine_memory(user_id))
    
    prompt = get_prompt(user_profile, record_time, scenario, recent_records, generalized_routine)

    start_time = time()
    result, token = await llm_client(prompt)
    end_time = time()
    execution_time = round(end_time - start_time, 2)

    mode = 'a' if os.path.exists(result_path) else 'w'
    with open(result_path, mode=mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if mode == 'w':
            writer.writerow(['original_intent', 'predicted_intent', 'time', 'token'])
        writer.writerow([intent, result, execution_time, token])


async def process_episode_async(row, base_dir, result_path):
    user_id = str(row['user_id'])
    time = str(row['time'])
    scenario = str(row['scenario'])
    intent = str(row['originalInstruction'])

    episode_path = os.path.join(base_dir, user_id, user_id, time)

    await main(episode_path, result_path, user_id, intent, time, scenario)


async def execution(test_data, base_dir, result_path, max_workers=4):
    semaphore = asyncio.Semaphore(max_workers)

    async def sem_task(row):
        async with semaphore:
            await process_episode_async(row, base_dir, result_path)

    tasks = []
    for _, row in test_data.iterrows():
        tasks.append(sem_task(row))

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing episodes"):
        await future


if __name__ == "__main__":
    test_data = pd.read_csv(args.test_path)
    asyncio.run(execution(test_data, args.base_dir, args.result_path))