import pandas as pd
import os
import json
import re
import base64


def get_all_previous(user_id, time, csv_file='./samples/total.csv'):
    df = pd.read_csv(csv_file)
    filtered_df = df[df['user_id'] == int(user_id)]
    filtered_df = filtered_df[filtered_df['time'] < str(time)]

    result = []
    for _, row in filtered_df.iterrows():
        r_time = row['time']
        r_scenario = row['scenario']
        r_intent = row['intentDescription']
        r_app = row['app']
        result.append(f'time: {r_time}, scenario: {r_scenario}, intent: {r_intent}, app: {r_app}')

    return result, filtered_df


def write_json(data, file_path):
    """Write data to a JSON file with indentation."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local image file to base64 format."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def remove_think_tags(text):
    """Remove <think>...</think> tags from text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def clean_llm_json(text):
    """Clean markdown code block wrappers from LLM JSON output."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    return cleaned
