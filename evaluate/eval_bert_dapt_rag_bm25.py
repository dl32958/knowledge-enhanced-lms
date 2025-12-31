import os
import re
import json
from typing import List, Dict
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, set_seed
from rank_bm25 import BM25Okapi


MODEL_PATH = "../bert_dapt/bert_base_dapt/epoch_3"
CLOZE_PATH = "../datasets_lg/processed/cloze_100_llm.jsonl"
RAG_FACTS_PATH = "../datasets_lg/processed/rag_facts.jsonl"
OUTPUT_DIR = "results_lg"


RAG_TOPK = 3
SEED = 42
set_seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def simple_word_tokenize(text):
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if len(t) > 2]


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

            if cloze is None or ans is None:
                print(f"[WARN] skip line, missing cloze/answer: {obj}")
                continue

            data.append(
                {
                    "id": obj.get("id"),
                    "text": cloze,
                    "answer": ans,
                }
            )

    print(f"Loaded {len(data)} cloze questions from {path}")
    return data


def load_rag_facts(path):
    facts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj:
                facts.append({"text": obj["text"]})

    print(f"Loaded {len(facts)} RAG facts from {path}")
    return facts


def build_bm25_index(facts):
    corpus_tokens = [simple_word_tokenize(f["text"]) for f in facts]
    bm25 = BM25Okapi(corpus_tokens)

    return bm25


def retrieve_facts_bm25(query, facts: List[Dict], bm25, topk = 3):
    """
    Use BM25 to retrieve top-k natural language facts from facts.
    """
    if not facts:
        return []

    # remove mask in query
    q_tokens = simple_word_tokenize(query.replace("[MASK]", ""))
    if not q_tokens:
        return [f["text"] for f in facts[:topk]]

    scores = bm25.get_scores(q_tokens)  # numpy array, shape [N_docs]
    idx_scores = list(enumerate(scores))
    idx_scores.sort(key=lambda x: x[1], reverse=True)

    top = idx_scores[:topk]

    # f highest score <= 0, matching is very weak, also fallback to first topk
    if top and top[0][1] <= 0:
        return [facts[i]["text"] for i in range(min(topk, len(facts)))]

    return [facts[i]["text"] for (i, s) in top]



def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    device = torch.device("cuda")

    print("Evaluating: BERT-DAPT + RAG (BM25)")

    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    vocab_size = model.config.vocab_size
    mask_id = tokenizer.mask_token_id

    # load data
    cloze_data = load_cloze(CLOZE_PATH)
    rag_facts = load_rag_facts(RAG_FACTS_PATH)

    # build BM25 index
    bm25 = build_bm25_index(rag_facts)

    total = 0
    correct = 0
    sum_rr = 0.0
    sum_cos = 0.0

    emb_weight = model.get_input_embeddings().weight  # [V, H]
    sep_token = tokenizer.sep_token or "[SEP]"

    for ex in cloze_data:
        cloze_text = ex["text"]
        gold_word = ex["answer"]

        retrieved_facts = retrieve_facts_bm25(
            query=cloze_text,
            facts=rag_facts,
            bm25=bm25,
            topk=RAG_TOPK,
        )

        # combine facts and cloze for model input, fact1 [SEP] fact2 [SEP] ... [SEP] Question: cloze
        context_parts = []
        for ft in retrieved_facts:
            context_parts.append(ft)

        context_str = f" {sep_token} ".join(context_parts) if context_parts else ""
        if context_str:
            full_input = f"{context_str} {sep_token} Question: {cloze_text}"
        else:
            full_input = cloze_text

        gold_tokens = tokenizer.tokenize(gold_word)
        if len(gold_tokens) != 1:
            print(
                f"Example id={ex.get('id')} answer '{gold_word}' "
                f"tokenized to {gold_tokens}, taking first piece."
            )
        gold_token = gold_tokens[0]
        gold_id = tokenizer.convert_tokens_to_ids(gold_token)

        enc = tokenizer(
            full_input,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)        # [1, L]
        attention_mask = enc["attention_mask"].to(device)

        # find mask position
        mask_pos = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
        if mask_pos.numel() == 0:
            print(f"No [MASK] found in example id={ex.get('id')}, skip.")
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
            logits = outputs.logits      # [1, L, V]
            hidden_states = outputs.hidden_states[-1]  # last layer [1, L, H]

        # Top-1 Accuracy
        mask_logits = logits[0, mask_idx, :]        # [V]
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
        h_mask = hidden_states[0, mask_idx, :]   # [H]
        e_gold = emb_weight[gold_id, :]   # [H]
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

    accuracy = correct / total
    mrr = sum_rr / total
    avg_cos = sum_cos / total

    print("========== Evaluation on Cloze (BERT-DAPT + RAG-BM25) ==========")
    print(f"Total examples:         {total}")
    print(f"Top-1 Accuracy:         {accuracy:.4f}")
    print(f"Mean Reciprocal Rank:   {mrr:.4f}")
    print(f"Avg Cosine Similarity:  {avg_cos:.4f}")


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        OUTPUT_DIR,
        f"eval_bert_dapt_rag_bm25_{timestamp}.json",
    )

    results = {
        "mode": "bert_dapt_rag_bm25",
        "cloze": CLOZE_PATH,
        "timestamp": timestamp,
        "n": total,
        "n_correct": correct,
        "mrr_sum": sum_rr,
        "cos_sum": sum_cos,
        "accuracy": accuracy,
        "mrr": mrr,
        "avg_cos": avg_cos,
        "rag_topk": RAG_TOPK,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
