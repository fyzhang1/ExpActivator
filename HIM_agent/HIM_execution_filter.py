import json
import re
import numpy as np
import pandas as pd
import torch
import jieba
from sentence_transformers import SentenceTransformer
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--embed-model", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Path to embedding model")
parser.add_argument("--save-base", type=str, default="./result/HIM_Agent/preference_memory", help="Base directory for saving results")
parser.add_argument("--csv-file", type=str, default="./samples/total.csv", help="Path to CSV data file")
parser.add_argument("--data-path", type=str, default="./data/Fingertip", help="Path to data directory")
args = parser.parse_args()

DECAY_LAMBDA = 0.99 # Optional: give a daily decay to each cluster
MAX_POOL_SIZE = 50
MAX_SAMPLES_PER_CLUSTER = 30
CORE_CLUSTER_RADIUS = 0.6

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(args.embed_model, device=device)

csv_file = args.csv_file
data_path = args.data_path
full_data = pd.read_csv(csv_file)

def extract_date(t):
    m = re.match(r"(\d{8})", str(t))
    return m.group(1) if m else t

def embed_texts(texts):
    return model.encode(texts, normalize_embeddings=True)

def parse_action(text):
    if "click" in text:
        content = re.search(r"content='(.*?)'", text)
        coord = re.search(r"coordinates=\((\d+),\s*(\d+)\)", text)
        return {
            "type": "click",
            "content": content.group(1) if content else "",
            "coordinates": tuple(map(int, coord.groups())) if coord else None
        }
    elif "long_click" in text:
        content = re.search(r"content='(.*?)'", text)
        coord = re.search(r"coordinates=\((\d+),\s*(\d+)\)", text)
        return {
            "type": "long_click",
            "content": content.group(1) if content else "",
            "coordinates": tuple(map(int, coord.groups())) if coord else None
        }
    elif "type" in text:
        return {"type": "type"}
    elif "scroll" in text:
        return {"type": "scroll"}
    elif "finished" in text:
        return {"type": "finished"}
    else:
        return {"type": "other"}

def action_distance(a, b, screen_size=(1200, 2640), threshold=0.14):
    if a["type"] != b["type"]:
        return 1

    if a["type"] == "click" and a.get("coordinates") and b.get("coordinates"):
        coord_a = np.array(a["coordinates"])
        coord_b = np.array(b["coordinates"])
        screen_w, screen_h = screen_size

        norm_dist = np.linalg.norm(coord_a - coord_b) / np.linalg.norm([screen_w, screen_h])

        return 0.0 if norm_dist < threshold else norm_dist

    return 0.0

def dtw_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = action_distance(seq1[i - 1], seq2[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return dp[n, m]

def dtw_action_similarity(seq1_texts, seq2_texts):
    seq1 = [parse_action(t) for t in seq1_texts]
    seq2 = [parse_action(t) for t in seq2_texts]
    distance = dtw_distance(seq1, seq2)
    similarity = 1 / (1 + distance)
    return round(similarity, 3)

def jaccard_similarity(a, b):
    set_a = set(jieba.lcut(a))
    set_b = set(jieba.lcut(b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def cosine(a, b):
    return float(np.dot(np.array(a), np.array(b)))

class DailyUpdate:
    def __init__(self, emb, record, is_core=False):
        self.center = emb
        self.weight = 1.0
        self.samples = [record]
        self.embed_list = [emb]
        self.action_list = [record.get('action_sequence', [])]

        self.is_core = is_core
        self.last_update = record['time'][:8]

        self.center_text = record['text']
        self.center_action = record.get('action_sequence', [])
        self.center_text_emb = emb

    def update_center_text(self):
        best_dist = float("inf")
        best_idx = 0

        for i, emb_i in enumerate(self.embed_list):
            dist = torch.norm(emb_i - self.center).item()
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        self.center_text = self.samples[best_idx]["text"]
        self.center_text_emb = self.embed_list[best_idx]

    def update_center_action(self):
        best_seq = None
        best_score = float("inf")

        n = len(self.action_list)
        for seq_i in self.action_list:
            total_dist = 0.0
            for seq_j in self.action_list:
                total_dist += dtw_action_similarity(seq_i, seq_j)

            mean_dist = total_dist / n

            if mean_dist < best_score:
                best_score = mean_dist
                best_seq = seq_i
        if best_score < 0.3:
            best_seq = []

        self.center_action = best_seq

    def update(self, emb, record, sim):
        w = sim

        self.center = (self.center * self.weight + emb * w) / (self.weight + w)
        self.weight += w

        self.samples.append(record)
        self.embed_list.append(emb)
        self.action_list.append(record.get('action_sequence', []))

        if len(self.samples) > MAX_SAMPLES_PER_CLUSTER:
            self.samples = self.samples[-MAX_SAMPLES_PER_CLUSTER:]
            self.embed_list = self.embed_list[-MAX_SAMPLES_PER_CLUSTER:]
            self.action_list = self.action_list[-MAX_SAMPLES_PER_CLUSTER:]

        self.last_update = record['time'][:8]

    def decay(self):
        self.weight *= DECAY_LAMBDA

def get_user_cluster(user_id):
    user_record = full_data[full_data['user_id'] == int(user_id)]
    user_record = user_record.sort_values(by='time', ascending=True)
    n = int(len(user_record) * 0.8)
    user_record = user_record.iloc[:n]
    
    user_record['date'] = user_record['time'].apply(extract_date)
    pool = []

    # get_action
    all_actions = []
    for _, row in user_record.iterrows():
        action_path = os.path.join(
            data_path, str(row['user_id']), str(row['user_id']),
            str(row['time']), 'action.jsonl'
        )

        if not os.path.exists(action_path):
            action_list = []
        else:
            try:
                with open(action_path, 'r', encoding='utf-8') as f:
                    action_list = [list(json.loads(line).values())[0] for line in f]
            except:
                print(action_path)
                action_list = []
        all_actions.append(action_list)
    user_record['action_sequence'] = all_actions

    for _, group in user_record.groupby('date'):
        descriptions = group['intentDescription'].tolist()
        apps = group['app'].tolist()
        times = group['time'].tolist()
        scenarios = group['scenario'].tolist()
        action = group['action_sequence'].tolist()
        
        embeddings = embed_texts(descriptions)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

        new_records = [
            {'text': t, 'app': a, 'time': tm, 'scenario': sc, 'action_sequence': act_seq}
            for t, a, tm, sc, act_seq in zip(descriptions, apps, times, scenarios, action)
        ]

        for c in pool:
            c.decay()
            c.update_center_text()
            c.update_center_action()

        for emb, record in zip(embeddings, new_records):
            assigned = False

            core_clusters = [c for c in pool if c.is_core]
            for c in core_clusters:
                cluster_text = c.center_text
                dtw_score = dtw_action_similarity(record.get('action_sequence', []), c.center_action)
                sim = 0.7 * cosine(emb, c.center) + 0.15 * jaccard_similarity(record['text'], cluster_text) + 0.15 * dtw_score
                if sim >= CORE_CLUSTER_RADIUS:
                    c.update(emb, record, sim)
                    assigned = True
                    break

            if not assigned:
                edge_clusters = [c for c in pool if not c.is_core]
                for c in edge_clusters:
                    cluster_text = c.center_text
                    dtw_score = dtw_action_similarity(record.get('action_sequence', []), c.center_action)
                    sim = 0.7 * cosine(emb, c.center) + 0.15 * jaccard_similarity(record['text'], cluster_text) + 0.15 * dtw_score
                    if sim >= CORE_CLUSTER_RADIUS:
                        c.update(emb, record, sim)
                        if c.weight > 3:
                            c.is_core = True
                        assigned = True
                        break

            if not assigned:
                pool.append(DailyUpdate(emb, record, is_core=False))

        pool = sorted(pool, key=lambda x: x.weight, reverse=True)[:MAX_POOL_SIZE]

    preference_cluster = []
    for i, c in enumerate(pool):
        samples = [s['text'] for s in c.samples]
        apps = list(set([s['app'] for s in c.samples]))
        times = [s['time'] for s in c.samples]
        scenes = [s['scenario'] for s in c.samples]

        preference_cluster.append({
            'cluster_id': i,
            'app_preferences': apps,
            'example_intents': samples,
            'time': times,
            'scenario': scenes,
            'last_update': c.last_update,
            'cluster_len': len(samples),
            'consist': c.weight / len(samples),
            'center_action': c.center_action,
            'center_text': c.center_text,
            'center_text_emb': c.center_text_emb.tolist(),
            'is_core': c.is_core
        })
    return preference_cluster

for user_id in full_data.user_id.unique():
    preference_cluster = get_user_cluster(user_id)
    save_path = os.path.join(args.save_base, f'user_{user_id}_preference_memory.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(preference_cluster, f, ensure_ascii=False, indent=2)