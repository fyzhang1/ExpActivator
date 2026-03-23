import json
import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from collections import Counter
from scipy.stats import entropy as scipy_entropy
from math import log2

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--embed-model", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Path to embedding model")
parser.add_argument("--csv-file", type=str, default="./samples/test.csv", help="Path to test CSV file")
parser.add_argument("--preference-path", type=str, default="./result/HIM_Agent/preference_memory", help="Base directory for HPM cluster files")
parser.add_argument("--output-path", type=str, default="./result/HIM_Agent/routine_memory", help="Output path")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(args.embed_model, device=device)

df = pd.read_csv(args.csv_file)
df = df.dropna(subset=['intentDescription'])

def time_to_hour(time_str):
    return int(time_str.split('_')[1][:2])

def extract_topk_hours(hours, top_k):
    counter = Counter(hours.tolist())
    return counter.most_common(top_k)

def extract_topk_scenes(scene_list, top_k):
    counter = Counter(scene_list)
    return counter.most_common(top_k)

def calculate_cluster_scores(preference_cluster, n_clusters=5):
    cluster_scores = []
    raw_scores = []
    max_len = max([center['cluster_len'] for center in preference_cluster])

    for center in preference_cluster:
        time_list = center['time']
        scene_list = center['scenario']
        L = center['cluster_len']
        consist = center['consist']

        if L < n_clusters:
            continue

        hours = np.array([time_to_hour(t) for t in time_list])
        time_entropy = scipy_entropy(np.bincount(hours, minlength=24), base=2) / log2(24)
        # scenes = [scene_list[j] for j in top10_idx if not np.isnan(row[j])]
        
        scene_counts = pd.Series(scene_list).value_counts()
        num_classes = len(scene_counts)
        if num_classes > 1:
            scene_entropy = scipy_entropy(scene_counts, base=2) / np.log2(num_classes)
        else:
            scene_entropy = 0.0
        
        center_hours = extract_topk_hours(hours, top_k=2)
        center_scenes = extract_topk_scenes(scene_list, top_k=2)

        time_density = 1 - time_entropy
        scene_density = 1 - scene_entropy
        density = (time_density + scene_density) / 2
        length = L / max_len

        total_density = (length + density + consist) / 3

        raw_scores.append({
            'time_cluster_density': time_entropy,
            'scene_cluster_density': scene_entropy,
            'center_hours': center_hours,
            'center_scenes': center_scenes,
            'total_cluster_density': total_density,
            'cluster_consistency': consist
        })

    cluster_scores = raw_scores
    return cluster_scores

Heat = 0.6
for filename in os.listdir(args.preference_path):
    if not filename.endswith(".json"):
        continue

    user_id = filename.split("_")[1]
    file_path = os.path.join(args.preference_path, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        preference_cluster = json.load(f)
    
    cluster_scores = calculate_cluster_scores(preference_cluster)

    user_routine_results = []
    for idx, score in enumerate(cluster_scores):
        if score["total_cluster_density"] > Heat:
            user_routine_results.append({
                "user_id": user_id,
                "cluster_id": idx,
                "time_cluster_density": score["time_cluster_density"],
                "scene_cluster_density": score["scene_cluster_density"],
                "collect_time": score["center_hours"],
                "collect_scene": score["center_scenes"],
                "total_cluster_density": score["total_cluster_density"],
                "cluster_content": {
                    k: v
                    for k, v in preference_cluster[idx].items()
                    if k not in {"center_action", "center_text_emb"}
                }
            })

    if user_routine_results:
        user_save_path = os.path.join(args.output_path, f'user_{user_id}_routine_memory.json')
        with open(user_save_path, "w", encoding="utf-8") as f:
            json.dump(user_routine_results, f, ensure_ascii=False, indent=2)