import os
import sys
import json
from typing import List, Dict
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, set_seed


STAGE1_DIR = "../kadapter/kadapter_stage1"
STAGE2_DIR = "../kadapter/kadapter_stage2_fusion"
STAGE2_WEIGHTS = f"{STAGE2_DIR}/stage2_kadapter_fusion.pt"

CLOZE_PATH = "../datasets_lg/processed/cloze_100_llm.jsonl"
OUTPUT_DIR = "results_lg"

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

            cloze = obj.get("cloze")
            ans = obj.get("answer")
            head = obj.get("triple_head")
            rel = obj.get("triple_relation")
            tail = obj.get("triple_tail")

            if cloze is None or ans is None:
                print(f"Skipping line, missing cloze/answer: {obj}")
                continue

            data.append(
                {
                    "id": obj.get("id"),
                    "text": cloze,
                    "answer": ans,
                    "triple_head": head,
                    "triple_relation": rel,
                    "triple_tail": tail,
                }
            )

    print(f"Loaded {len(data)} cloze questions from {path}")
    return data


KADAPTER_DIR = "../kadapter"
if KADAPTER_DIR not in sys.path:
    sys.path.append(KADAPTER_DIR)

from kadapter_model import BertKAdapterForMaskedLM
from kadapter_stage1 import BERT_DAPT_PATH, BOTTLE_NECK


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")

    print("Evaluating: K-Adapter")

    # use tokenizer saved from Stage2 training
    tokenizer = AutoTokenizer.from_pretrained(STAGE2_DIR, use_fast=True)
    mask_id = tokenizer.mask_token_id

    # load entity/relation mappings from Stage1
    ent2id_path = os.path.join(STAGE1_DIR, "ent2id.json")
    rel2id_path = os.path.join(STAGE1_DIR, "rel2id.json")
    with open(ent2id_path, "r", encoding="utf-8") as f:
        ent2id = json.load(f)
    with open(rel2id_path, "r", encoding="utf-8") as f:
        rel2id = json.load(f)
    num_entities = len(ent2id)
    num_relations = len(rel2id)
    print(f"num_entities={num_entities}, num_relations={num_relations}")

    # initialize model and load Stage2 weights
    model = BertKAdapterForMaskedLM(
        backbone_path=BERT_DAPT_PATH,
        bottleneck_dim=BOTTLE_NECK,
        num_entities=num_entities,
        num_relations=num_relations,
        freeze_bert=False,
    )
    state_dict = torch.load(STAGE2_WEIGHTS, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # load cloze data
    cloze_data = load_cloze(CLOZE_PATH)

    total = 0
    correct = 0
    sum_rr = 0.0
    sum_cos = 0.0

    emb_weight = model.bert_mlm.get_input_embeddings().weight

    for ex in cloze_data:
        text = ex["text"]
        gold_word = ex["answer"]

        head_str = ex.get("triple_head")
        rel_str = ex.get("triple_relation")
        tail_str = ex.get("triple_tail")

        use_kg = (
            head_str is not None and
            tail_str is not None and
            rel_str is not None and
            head_str in ent2id and
            tail_str in ent2id and
            rel_str in rel2id
        )

        if use_kg:
            head_id = ent2id[head_str]
            tail_id = ent2id[tail_str]
            rel_id = rel2id[rel_str]

        # process gold token
        gold_tokens = tokenizer.tokenize(gold_word)
        if len(gold_tokens) != 1:
            print(
                f"[WARN] example id={ex.get('id')} answer '{gold_word}' "
                f"tokenized to {gold_tokens}, taking first piece."
            )
        gold_token = gold_tokens[0]
        gold_id = tokenizer.convert_tokens_to_ids(gold_token)

        gold_norm = gold_word.strip().lower()

        # encode input
        enc = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)        # [1, L]
        attention_mask = enc["attention_mask"].to(device)

        # find mask position
        mask_pos = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
        if mask_pos.numel() == 0:
            print(f"[WARN] no [MASK] found in example id={ex.get('id')}, skip.")
            continue
        mask_idx = mask_pos[0].item() if mask_pos.numel() > 1 else mask_pos.item()
        mask_positions = torch.tensor([mask_idx], device=device, dtype=torch.long)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                head_ids=torch.tensor([head_id], device=device) if use_kg else None,
                rel_ids=torch.tensor([rel_id], device=device) if use_kg else None,
                tail_ids=torch.tensor([tail_id], device=device) if use_kg else None,
                mask_positions=mask_positions if use_kg else None,
                use_kg=use_kg,
            )
            logits = outputs["logits"]      # [1, L, V]
            hidden_states = outputs["hidden_states"]    # [1, L, H]

        # Top-1 Accuracy
        mask_logits = logits[0, mask_idx, :]          # [V]
        pred_id = int(mask_logits.argmax(dim=-1).item())

        pred_token = tokenizer.convert_ids_to_tokens(pred_id)
        if pred_token.startswith("##"):
            pred_token_clean = pred_token[2:]
        else:
            pred_token_clean = pred_token
        pred_norm = pred_token_clean.strip().lower()

        top1_correct = (pred_norm == gold_norm)
        if top1_correct:
            correct += 1

        # MRR
        gold_score = mask_logits[gold_id]
        rank = int((mask_logits > gold_score).sum().item()) + 1
        rr = 1.0 / rank
        sum_rr += rr

        # Cosine similarity
        h_mask = hidden_states[0, mask_idx, :]     # [H]
        e_gold = emb_weight[gold_id, :]      # [H]
        cos = F.cosine_similarity(
            h_mask.unsqueeze(0),
            e_gold.unsqueeze(0),
            dim=-1,
        ).item()
        sum_cos += cos

        total += 1

    # summary
    if total == 0:
        print("No valid examples evaluated.")
        return

    accuracy = correct / total
    mrr = sum_rr / total
    avg_cos = sum_cos / total

    print("========== Evaluation on Cloze (K-Adapter, KG-injected) ==========")
    print(f"Total examples:         {total}")
    print(f"Top-1 Accuracy:         {accuracy:.4f}  (case-insensitive)")
    print(f"Mean Reciprocal Rank:   {mrr:.4f}")
    print(f"Avg Cosine Similarity:  {avg_cos:.4f}")

    # save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        OUTPUT_DIR,
        f"eval_kadapter_{timestamp}.json",
    )

    results = {
        "mode": "kadapter_kg_injected",
        "cloze": CLOZE_PATH,
        "timestamp": timestamp,
        "n": total,
        "n_correct": correct,
        "mrr_sum": sum_rr,
        "cos_sum": sum_cos,
        "accuracy": accuracy,
        "mrr": mrr,
        "avg_cos": avg_cos,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
