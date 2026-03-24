import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch

model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

data_path = './data/personal_execution/personal_samples.csv'
ambiguous_data = pd.read_csv(data_path)

save_history = {}

for idx, row in ambiguous_data.iterrows():
    # print(row)
    ambiguous_intent = row['ambiguousInstruction']
    embed_result = model.encode(ambiguous_intent).reshape(1, -1)
    save_str = str(row['user_id']) + '_' + str(row['time'])
    
    save_history[save_str] = embed_result

np.save('./ambiguous_embed.npy', save_history)