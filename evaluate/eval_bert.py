import os
import json
from typing import List, Dict
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, set_seed


CLOZE_PATH = "../datasets_lg/processed/cloze_100_llm.jsonl"
OUTPUT_DIR = "results_lg"

MODEL_NAME = "bert-base-cased"
SEED = 42
set_seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_cloze(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            text = obj.get("cloze")
            ans  = obj.get("answer")

            if text is None or ans is None:
                print(f"Skipping line, missing cloze/answer: {obj}")
                continue

            data.append(
                {
                    "id": obj.get("id"),
                    "text": text,
                    "answer": ans,
                }
            )

    print(f"Loaded {len(data)} cloze questions from {path}")
    return data


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    device = torch.device("cuda")

    print(f"Evaluating model: {MODEL_NAME}")
    
    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    vocab_size = model.config.vocab_size
    mask_id = tokenizer.mask_token_id

    # load cloze
    cloze_data = load_cloze(CLOZE_PATH)

    total = 0
    correct = 0
    sum_rr = 0.0
    sum_cos = 0.0

    emb_weight = model.get_input_embeddings().weight  # [V, H]

    for ex in cloze_data:
        text = ex["text"]
        gold_word = ex["answer"]

        gold_tokens = tokenizer.tokenize(gold_word)
        if len(gold_tokens) != 1:
            print(
                f"[WARN] example id={ex.get('id')} answer '{gold_word}' "
                f"tokenized to {gold_tokens}, taking first piece."
            )
        gold_token = gold_tokens[0]
        gold_id = tokenizer.convert_tokens_to_ids(gold_token)

        enc = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)        # [1, L]
        attention_mask = enc["attention_mask"].to(device)

        mask_pos = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
        if mask_pos.numel() == 0:
            print(f"[WARN] no [MASK] found in example id={ex.get('id')}, skip.")
            continue
        if mask_pos.numel() > 1:
            mask_idx = mask_pos[0].item()
        else:
            mask_idx = mask_pos.item()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = outputs.logits                 # [1, L, V]
            hidden_states = outputs.hidden_states[-1]  # last layer [1, L, H]


        # Top-1 Accuracy
        mask_logits = logits[0, mask_idx, :]          # [V]
        pred_id = int(mask_logits.argmax(dim=-1).item())
        top1_correct = (pred_id == int(gold_id))
        if top1_correct:
            correct += 1

        # MRR
        gold_score = mask_logits[gold_id]
        rank = int((mask_logits > gold_score).sum().item()) + 1
        rr = 1.0 / rank
        sum_rr += rr

        # Cosine similarity
        h_mask = hidden_states[0, mask_idx, :]      # [H]
        e_gold = emb_weight[gold_id, :]      # [H]
        cos = F.cosine_similarity(
            h_mask.unsqueeze(0),
            e_gold.unsqueeze(0),
            dim=-1,
        ).item()
        sum_cos += cos

        total += 1


    if total == 0:
        print("No valid examples evaluated.")
        return

    acc = correct / total
    mrr = sum_rr / total
    avg_cos = sum_cos / total

    print("========== Evaluation on Cloze (BERT-base-cased) ==========")
    print(f"Total examples:         {total}")
    print(f"Top-1 Accuracy:         {acc:.4f}")
    print(f"Mean Reciprocal Rank:   {mrr:.4f}")
    print(f"Avg Cosine Similarity:  {avg_cos:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"eval_bert_base_cased_{timestamp}.json")

    results = {
        "mode": "bert_base_cased",
        "model_name": MODEL_NAME,
        "cloze": CLOZE_PATH,
        "timestamp": timestamp,
        "n": total,
        "n_correct": correct,
        "mrr_sum": sum_rr,
        "cos_sum": sum_cos,
        "accuracy": acc,
        "mrr": mrr,
        "avg_cos": avg_cos,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
