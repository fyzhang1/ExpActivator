import json
import os
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from tqdm import tqdm
import argparse
import numpy as np

from src.eval_execution.prompt import QWEN3VL_SYSTEM_PROMPT, QWEN3_USER_PROMPT_GENERALIZED
from src.eval_execution.utils import load_user_profile, convert_action_qwen, convert_format, compute_CSR

outputs = []

min_pixels = 200704
max_pixels = 1048576

parser = argparse.ArgumentParser()
parser.add_argument("--device")
parser.add_argument("--test-path")
parser.add_argument("--model-path")
parser.add_argument("--save-file", type=str, default="eval_results.json")
parser.add_argument("--HIM-path", type=str, default="./result/HIM_Agent/preference_memory")

args = parser.parse_args()

print(args.save_file)
device = args.device
# offline inference
ambiguous_embed = np.load('./ambiguous_embed.npy', allow_pickle=True).item()

# online inference
# model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
# ambiguous_embed = model.encode(ambiguous_intent).reshape(1, -1)

total_token = 0
generate_token = 0


model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype="bfloat16", device_map="auto",attn_implementation="flash_attention_2"
)

processor = AutoProcessor.from_pretrained(args.model_path)
with open(args.test_path, 'r') as file:
    test_data = json.load(file)

with open(args.generalized_path, 'r', encoding='utf-8') as f:
    generalized_result = json.load(f)

# parse generalized data
generalized_data = {}
for item in generalized_result:
    uid = item['user_id']
    parsed = json.loads(item["LLM_output"])
    for item in parsed.get("summaries", []):
        if "reasoning" in item:
            del item["reasoning"]
    generalized_data[uid] = parsed

def cosine_similarity(A, B):
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B)
    return A_norm.dot(B_norm)

def generate_grounding(image_path, query, min_pixel=256 * 28 * 28, max_pixel=1048576):
    global total_token
    global generate_token
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": QWEN3VL_SYSTEM_PROMPT},
            ],
        },
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

    total_token += generated_ids.shape[1]
    generate_token += generated_ids.shape[1] - inputs.input_ids.shape[1]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0][0:-10]
    # print(output_text)

    try:
        action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
        action = action.get("arguments")
        temp = action.get("action")
    except:
        print(output_text)
        action = {'action': 'error'}

    return action

def get_HIM_summaries(user_id, exe_id=''):
    with open(os.path.join(args.HIM_path, f'user_{user_id}_clusters.json'), 'r', encoding='utf-8') as f:
        user_preference = json.load(f)
    embed_id = '_'.join(exe_id.split('_')[:-1])

    # To acceralate the evaluation, we stored the ambiguous instruction embeddings in advance
    ambiguous_embed_ = ambiguous_embed[embed_id]
    embeds = []
    cluster_refs = []

    for cluster in user_preference:
        if cluster.get("cluster_len", 0) >= 3:
            emb = cluster.get("center_text_emb", None)
            if emb is not None:
                embeds.append(emb)
                cluster_refs.append(cluster)

    embeds = np.array(embeds)

    sims = cosine_similarity(embeds, ambiguous_embed_.T).reshape(-1)
    top_idx = np.argsort(sims)[-3:]
    summaries = []
    for idx in top_idx:
        top_cluster = cluster_refs[idx]
        summaries.append([f'Intent: {top_cluster.get("center_text")}, Actions: {top_cluster.get("center_action")}'])

    return summaries


def is_episode_start(step_id):
    """Check if the step is the first step of an episode (suffix _0)."""
    return step_id.rsplit('_', 1)[-1] == '0'

def execution():
    sr_total, type_total = 0, 0
    history_actions = []
    for i in tqdm(range(len(test_data))):
        step = test_data[i]
        user_profile = load_user_profile(step['user_id'])
        instruction = step["ambiguous_instruction"]
        answer = step["answer"]
        images = step["img_path"]

        # Reset history at the start of each episode
        if is_episode_start(step["id"]):
            history_actions = []
            try:
                generalized_intents = get_HIM_summaries(step['user_id'], exe_id=step["id"])
            except Exception as e:
                print(e)
                generalized_intents = ''

        response_action = generate_grounding(
            images,
            QWEN3_USER_PROMPT_GENERALIZED.format(history_actions=history_actions, retrieved_intent=generalized_intents, instruction=instruction, user_profile=user_profile)
        )

        history_actions.append(convert_format(step["answer"]))

        try:
            with Image.open(images) as img:
                width, height = img.size
        except:
            print('Cannot get image size for ', images)
            width, height = 1000, 1000

        type_acc, sr = convert_action_qwen(response_action, answer, width, height)
        sr_total += sr
        type_total += type_acc
        outputs.append({
            'episode_id': step["id"],
            'answer': answer,
            'output': response_action,
            'result': f"Type: {type_acc}, SR: {sr}"
        })

    CSR = compute_CSR(outputs)

    print(f"Type Accuracy: {type_total / len(outputs):.4f}")
    print(f"Success Rate: {sr_total / len(outputs):.4f}")
    print(f"Cumulative Success Rate (CSR): {CSR:.4f}")

    json.dump(outputs, open(args.save_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f"Total tokens: {total_token/len(outputs)}, Generated tokens: {generate_token/len(outputs)}")



execution()
