import json
import os
import re
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from tqdm import tqdm
import argparse

from prompt import UI_TARS_PROMPT
from utils import load_user_profile, eval_action, convert_format, compute_CSR

outputs = []

min_pixels = 200704
max_pixels = 1048576

parser = argparse.ArgumentParser()
parser.add_argument("--device")
parser.add_argument("--test-path")
parser.add_argument("--model-path")
parser.add_argument("--ambiguous", type=lambda x: x.lower() == 'true', default=False)
parser.add_argument("--data-path", type=str, default="./data")
parser.add_argument("--save-file", type=str, default="eval_results.json")
args = parser.parse_args()
print(args.save_file)

device = args.device
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype="bfloat16", device_map="auto",attn_implementation="flash_attention_2"
)
processor = AutoProcessor.from_pretrained(args.model_path)
with open(args.test_path, 'r') as file:
    test_data = json.load(file)

def generate_grounding(image_path, query, min_pixel=256 * 28 * 28, max_pixel=1048576):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": min_pixel,
                    "max_pixels": max_pixel,
                },
                {"type": "text",
                 "text": query},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
#    resized_w, resized_h = image_inputs[0].size
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0][0:-10]

    pattern = r"Action:\s*(\S+)"
    thought = r"Thought:\s*(.+)"
    try:
        match = re.search(pattern, output_text)
        action = match.group(1)
    except:
        print(output_text)
        action = None
    
    try:
        match = re.search(thought, output_text)
        thought = match.group(1)
    except:
        print(output_text)
        thought = None


    return action, thought

def is_episode_start(step_id):
    """Check if the step is the first step of an episode (suffix _0)."""
    return step_id.rsplit('_', 1)[-1] == '0'


def execution():
    sr_total, type_total = 0, 0
    history_actions = []
    for i in tqdm(range(len(test_data))):
        step = test_data[i]
        user_profile = load_user_profile(step['user_id'])
        instruction = step["ambiguous_instruction"] if args.ambiguous else step["instruction"]
        answer = step["answer"]
        images = step["img_path"]

        # Reset history at the start of each episode
        if is_episode_start(step["id"]):
            history_actions = []

        response_action, thought = generate_grounding(
            images,
            UI_TARS_PROMPT.format(history_actions=history_actions, language='English' if is_episode_start(step["id"]) else 'Chinese', instruction=instruction, user_profile=user_profile)
        )

        # add GT(User Action) in history
        history_actions.append(convert_format(step["answer"]))

        try:
            with Image.open(images) as img:
                width, height = img.size
        except:
            print('Cannot get image size for ', images)
            width, height = 1000, 1000

        type_acc, sr = eval_action(response_action, answer, width, height)
        sr_total += sr
        type_total += type_acc
        outputs.append({
            'episode_id': step["id"],
            'answer': answer,
            'output': response_action,
            'thought': thought,
            'result': f"Type: {type_acc}, SR: {sr}"
        })

    CSR = compute_CSR(outputs)

    print(f"Type Accuracy: {type_total / len(outputs):.4f}")
    print(f"Success Rate: {sr_total / len(outputs):.4f}")
    print(f"Cumulative Success Rate (CSR): {CSR:.4f}")

    print(f"Saving predict result ...")
    json.dump(outputs, open(args.save_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

execution()
