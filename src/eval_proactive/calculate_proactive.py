import argparse
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def text_similarity_batch(original_texts, predicted_texts, model):
    all_texts = original_texts + predicted_texts
    embeddings = model.encode(all_texts)
    n = len(original_texts)
    
    orig_emb = embeddings[:n]
    pred_emb = embeddings[n:]
    
    edit_scores = [fuzz.ratio(o, p) / 100 for o, p in zip(original_texts, predicted_texts)]
    semantic_scores = [
        np.dot(orig, pred) / (np.linalg.norm(orig) * np.linalg.norm(pred))
        for orig, pred in zip(orig_emb, pred_emb)
    ]
    
    combined_scores = [(e + s) / 2 for e, s in zip(edit_scores, semantic_scores)]
    return np.round(combined_scores, 2)


def calculate_similarity_mean(eval_data, model):
    if 'original_intent' not in eval_data.columns or 'predicted_intent' not in eval_data.columns:
        raise ValueError("CSV must contain 'original_intent' and 'predicted_intent' columns")

    mask = eval_data['predicted_intent'] != "False"
    eval_data = eval_data[mask].copy()
    similarities = text_similarity_batch(
        eval_data['original_intent'].tolist(),
        eval_data['predicted_intent'].tolist(),
        model
    )
    
    eval_data['similarity'] = similarities
    mean_score = similarities.mean()

    time_mean = eval_data['time'].mean()
    token_mean = eval_data['token'].mean()

    print(f"Mean similar: {mean_score}")
    print(f"Mean time: {time_mean}")
    print(f"Mean token: {token_mean}")
    # return round(mean_score, 2), time_mean, token_mean, eval_data
    
def calculate_f1_recall(eval_data, fn_data=None):
    y_pred_eval = (eval_data["predicted_intent"].astype(str).str.lower() != "false").astype(int)
    y_true_eval = [1] * len(eval_data)
    
    y_pred_fn = (fn_data["predicted_intent"].astype(str).str.lower() != "false").astype(int)
    y_true_fn = [0] * len(fn_data)

    y_true = y_true_eval + y_true_fn
    y_pred = y_pred_eval.tolist() + y_pred_fn.tolist()

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"TP (True Positive):  {tp}")
    print(f"FP (False Positive): {fp}")
    print(f"FN (False Negative): {fn}")
    print(f"TN (True Negative):  {tn}")
    print("-" * 40)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"FPR (False Positive Rate): {fpr:.4f}")
    print(f"F1-score:  {f1:.4f}")


parser = argparse.ArgumentParser()
parser.add_argument("--result-path")
parser.add_argument("--embed-model")
parser.add_argument("--fn_path", default=None)
args = parser.parse_args()

eval_data = pd.read_csv(args.result_path, encoding="utf-8")
eval_data["predicted_intent"] = eval_data["predicted_intent"].astype(str).str.replace("**", "", regex=False).str.strip()

if args.fn_path:
    fn_data = pd.read_csv(args.fn_path, encoding="utf-8") 
    calculate_f1_recall(eval_data, fn_data)

model = SentenceTransformer(args.embed_model)
calculate_similarity_mean(eval_data, model)

