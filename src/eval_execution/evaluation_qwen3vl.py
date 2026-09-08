import argparse
import json
import os
import re

from PIL import Image
from prompt import QWEN3_USER_PROMPT, QWEN3VL_SYSTEM_PROMPT
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from utils import compute_CSR, convert_action_qwen, convert_format, load_user_profile

outputs = []

min_pixels = 200704
max_pixels = 1048576

parser = argparse.ArgumentParser()
parser.add_argument("--device")
parser.add_argument("--test-path")
parser.add_argument("--model-path")
parser.add_argument("--ambiguous", type=lambda x: x.lower() == "true", default=False)
parser.add_argument("--data-path", type=str, default="./data")
parser.add_argument("--save-file", type=str, default="eval_results.json")
parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--first-step-only", action="store_true")
parser.add_argument("--checkpoint-every", type=int, default=100)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--batch-size", type=int, default=1)
args = parser.parse_args()
print(args.save_file)

device = args.device
model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype="bfloat16",
    device_map="auto",
    attn_implementation=args.attn_implementation,
)
processor = AutoProcessor.from_pretrained(args.model_path)
processor.tokenizer.padding_side = "left"
with open(args.test_path, "r") as file:
    test_data = json.load(file)
if args.limit is not None:
    test_data = test_data[: args.limit]
if args.first_step_only:
    test_data = [step for step in test_data if step["id"].endswith("_0")]

save_dir = os.path.dirname(os.path.abspath(args.save_file))
os.makedirs(save_dir, exist_ok=True)
if args.resume and os.path.exists(args.save_file):
    with open(args.save_file, "r", encoding="utf-8") as f:
        outputs.extend(json.load(f))
    print(f"Resuming from {len(outputs)} saved predictions")


def resolve_image_path(image_path):
    """Map the released ./data/Fingertip paths to a separately stored dataset."""
    if os.path.exists(image_path):
        return image_path

    normalized = image_path.replace("\\", "/")
    marker = "data/Fingertip/"
    if marker in normalized:
        relative_path = normalized.split(marker, 1)[1]
        candidate = os.path.join(args.data_path, relative_path)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Cannot resolve image path: {image_path}")


def save_outputs():
    with open(args.save_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)


def parse_action(output_text):
    try:
        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", output_text, re.DOTALL)
        if match is None:
            raise ValueError("Missing <tool_call> block")
        action = json.loads(match.group(1))
        arguments = action.get("arguments")
        return arguments if isinstance(arguments, dict) else {"action": "error"}
    except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
        return {"action": "error"}


def generate_grounding_batch(batch, min_pixel=256 * 28 * 28, max_pixel=1048576):
    texts = []
    image_inputs = []
    video_inputs = []
    for image_path, query in batch:
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
                    {"type": "text", "text": query},
                ],
            },
        ]
        texts.append(
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
        sample_images, sample_videos = process_vision_info(messages)
        image_inputs.extend(sample_images or [])
        video_inputs.extend(sample_videos or [])

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs or None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(
        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return [parse_action(output_text) for output_text in output_texts]


def is_episode_start(step_id):
    """Check if the step is the first step of an episode (suffix _0)."""
    return step_id.rsplit("_", 1)[-1] == "0"


def execution():
    sr_total = sum("SR: 1" in item.get("result", "") for item in outputs)
    type_total = sum("Type: 1" in item.get("result", "") for item in outputs)
    completed_ids = {item["episode_id"] for item in outputs}
    history_actions = []
    prepared = []
    for step in test_data:
        if is_episode_start(step["id"]):
            history_actions = []

        user_profile = load_user_profile(step["user_id"])
        instruction = (
            step["ambiguous_instruction"] if args.ambiguous else step["instruction"]
        )
        query = QWEN3_USER_PROMPT.format(
            history_actions=history_actions,
            instruction=instruction,
            user_profile=user_profile,
        )
        if step["id"] not in completed_ids:
            prepared.append((step, resolve_image_path(step["img_path"]), query))
        history_actions.append(convert_format(step["answer"]))

    batch_size = max(args.batch_size, 1)
    for batch_start in tqdm(range(0, len(prepared), batch_size)):
        prepared_batch = prepared[batch_start : batch_start + batch_size]
        response_actions = generate_grounding_batch(
            [(image_path, query) for _, image_path, query in prepared_batch]
        )

        for (step, images, _), response_action in zip(prepared_batch, response_actions):
            answer = step["answer"]

            try:
                with Image.open(images) as img:
                    width, height = img.size
            except (OSError, ValueError):
                print("Cannot get image size for ", images)
                width, height = 1000, 1000

            type_acc, sr = convert_action_qwen(response_action, answer, width, height)
            sr_total += sr
            type_total += type_acc
            outputs.append(
                {
                    "episode_id": step["id"],
                    "answer": answer,
                    "output": response_action,
                    "result": f"Type: {type_acc}, SR: {sr}",
                }
            )

            if args.checkpoint_every > 0 and len(outputs) % args.checkpoint_every == 0:
                save_outputs()

    CSR = compute_CSR(outputs)

    print(f"Type Accuracy: {type_total / len(outputs):.4f}")
    print(f"Success Rate: {sr_total / len(outputs):.4f}")
    print(f"Cumulative Success Rate (CSR): {CSR:.4f}")

    print("Saving predict result ...")
    save_outputs()


execution()
