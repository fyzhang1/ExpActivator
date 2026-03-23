import json
import os
import base64
from openai import AsyncOpenAI
from tqdm import tqdm
import argparse
import asyncio

from prompt import API_PROMPT
from utils import load_user_profile, convert_format

# Here we use async API to accelerate the evaluation.
client = AsyncOpenAI(
    api_key="",
    base_url=""
)

parser = argparse.ArgumentParser()
parser.add_argument("--test-path")
parser.add_argument("--model-name")
parser.add_argument("--ambiguous", type=lambda x: x.lower() == 'true', default=False)
parser.add_argument("--data-path", type=str, default="./data")
parser.add_argument("--save-file", type=str, default="eval_results.json")
args = parser.parse_args()

with open(args.test_path, 'r') as file:
    test_data = json.load(file)

outputs = []


def resolve_img_path(img_path):
    """Resolve image path: desktop images use path as-is, others prepend data_path."""
    if img_path.startswith('./') or img_path.startswith('/'):
        return img_path
    return os.path.join(args.data_path, img_path)


async def generate_response(image_path, prompt, model):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    if model.startswith("gpt"):
        msg_model = model
        image_url = f"data:image/png;base64,{base64_image}"
    elif model == "glm":
        msg_model = "glm-4v-plus"
        image_url = base64_image
    elif model == "qwen":
        msg_model = "qwen-vl-max"
        image_url = f"data:image/png;base64,{base64_image}"
    else:
        raise ValueError(f"Unknown model: {model}")

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }]

    try:
        completion = await client.chat.completions.create(
            model=msg_model,
            messages=messages,
        )
        output_text = completion.choices[0].message.content
    except Exception as e:
        print(f"[Error] API call failed: {e}")
        return {}, f"Error: {e}"

    lines = output_text.split('\n')
    thought = ""
    action = {}

    for line in lines:
        if line.startswith("Thought:"):
            thought = line[len("Thought:"):].strip()
        elif line.startswith("Action:"):
            try:
                action = json.loads(line[len("Action: "):].strip())
            except json.JSONDecodeError:
                action = line
                print(f"Failed to parse action JSON: {line}")

    return action, thought


async def process_step(step, episode_state):
    user_profile = load_user_profile(step['user_id'])
    instruction = step["ambiguous_instruction"] if args.ambiguous else step["instruction"]
    answer = step["answer"]
    images = resolve_img_path(step["img_path"])

    response_action, thought = await generate_response(
        images,
        API_PROMPT.replace("{history_actions}", str(episode_state["history_actions"]))
                  .replace("{instruction}", instruction)
                  .replace("{user_profile}", user_profile),
        args.model_name
    )
    episode_state["history_actions"].append(convert_format(step["answer"]))

    result = {
        'episode_id': step["id"],
        'answer': answer,
        'output': response_action,
        'thought': thought,
    }

    outputs.append(result)
    return result


async def execution():
    episode_state = {"history_actions": []}
    tasks = []
    semaphore = asyncio.Semaphore(5)

    async def sem_task(step):
        async with semaphore:
            return await process_step(step, episode_state)

    for step in test_data:
        tasks.append(asyncio.create_task(sem_task(step)))

    # tqdm异步进度显示
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await future
        if len(outputs) % 10 == 0:
            with open(args.save_file, 'w', encoding='utf-8') as f:
                json.dump(outputs, f, indent=4, ensure_ascii=False)

    with open(args.save_file, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(execution())