import pandas as pd
import os
import csv
import traceback
import argparse
from tqdm import tqdm
from openai import AsyncOpenAI
import asyncio
from time import time

client = AsyncOpenAI(
    api_key="",
    base_url=""
)

parser = argparse.ArgumentParser()
parser.add_argument("--llm-model")
parser.add_argument("--base-dir")
parser.add_argument("--test-path")
parser.add_argument("--result-path")
parser.add_argument("--user-profile")
parser.add_argument("--csv-file")
args = parser.parse_args()

task_level = 0

total_record = pd.read_csv(args.csv_file)
result = {}


def load_user_profile(user_id):

    df = pd.read_csv(args.user_profile)

    row = df[df['user_id'] == int(user_id)]

    if not row.empty:
        result = row.iloc[0, 1:].tolist()
        text = f'{result[0]}，{str(result[1])}岁，职业为{result[2]}，现居{result[3]}，{result[4]}，使用{result[5]}手机'
        return text
    else:
        print("not found user_id")

# get all history data
def get_all_intents(user_id):
    previous_record = []
    user_record = total_record[total_record['user_id'] == int(user_id)]

    user_record = user_record.sort_values(by='time')

    cutoff_index = int(len(user_record) * 0.8)
    user_record = user_record.iloc[:cutoff_index]

    user_record = user_record.tail(150).iloc[::-1]

    for _, row in user_record.iterrows():
        r_time = row['time']
        r_scenario = row['scenario']
        r_intent = row['intentDescription']
        previous_record.append(f'Time: {r_time}, Scenario: {r_scenario}, intent: {r_intent}')

    return previous_record

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


def get_prompt(profile, time, scenario, recent_records, previous_intents):    
    prompt = f"""You are skilled at analyzing user history. You are given several retrieved or full user history records. Your task is to determine whether the current user state requires an proactive suggestions. 

## Note
- If you decide that a recommendation is needed, output a suitable user instruction for the GUI agent to execute.
- If no highly repeated daily routines are found from History_records, only output False.
- Express the user's intent unambiguously in one Chinese sentence, including all necessary information.
- History_records contains the user's intents at certain times and in certain scenarios in the past.
- Do not output any explanation including time and scenario information. Only output the user's intent or instruction.    

## Input
User_profile: {profile}
Time: {time}
Scenario: {scenario}
History_records: {previous_intents}

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

    previous_intents = str(get_all_intents(user_id))
    
    prompt = get_prompt(user_profile, record_time, scenario, recent_records, previous_intents)

    start_time = time()
    result, token = await llm_client(task_level, prompt)
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