#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate BERT-DAPT + RAG (Model B') on 200 cloze questions.

Pipeline:
  - Retrieval: from rag_facts.jsonl using a simple token-overlap scorer
  - Fusion: [fact_1] [SEP] fact_2 [SEP] ... [SEP] cloze_with_[MASK]
  - Model: BERT (same DAPT checkpoint as Model A)
  - Metrics: Top-1 Accuracy, MRR, Avg Cosine Similarity (same as eval_bert_dapt.py)
"""

import os
import re
import json
from typing import List, Dict
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, set_seed

from rank_bm25 import BM25Okapi


# ================== 路径配置 ==================

ROOT = "/home/lu.dong1/rag-embed-study-v4"

# 与 Model A 一样的 DAPT checkpoint
MODEL_PATH = os.path.join(
    ROOT, "bert_dapt", "bert_base_dapt", "epoch_3"
)

# 200 道 Cloze
CLOZE_PATH = os.path.join(
    ROOT, "datasets", "processed", "cloze_200.jsonl"
)

# RAG facts（之前用 gen_rag_facts.py 生成）
RAG_FACTS_PATH = os.path.join(
    ROOT, "datasets", "processed", "rag_facts.jsonl"
)

OUTPUT_DIR = os.path.join(ROOT, "evaluate", "results")

# RAG 检索 top-K
RAG_TOPK = 5

SEED = 42
set_seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ================== 工具函数 ==================

def simple_word_tokenize(text: str) -> List[str]:
    """
    非 BERT 的简易分词，用于 RAG 检索：
    - 小写
    - 只保留字母数字
    - 去掉长度 <= 2 的词
    """
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if len(t) > 2]


def load_cloze(path: str) -> List[Dict]:
    """
    cloze_200.jsonl 结构示例：
    {
      "id": "cloze_001",
      "cloze": "... [MASK] ...",
      "answer": "cognitive bias",
      ...
    }
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            text = obj.get("cloze")
            ans = obj.get("answer")

            if text is None or ans is None:
                print(f"[WARN] skip line, missing cloze/answer: {obj}")
                continue

            data.append(
                {
                    "id": obj.get("id"),
                    "text": text,
                    "answer": ans,
                }
            )

    print(f"[INFO] Loaded {len(data)} cloze questions from {path}")
    return data


def load_rag_facts(path: str) -> List[Dict]:
    """
    rag_facts.jsonl 大致结构（根据之前脚本）：
      { "triple_id": ..., "head": ..., "relation": ..., "tail": ..., "fact": "... sentence ..." }
    这里只要拿到一个 natural-language fact 文本即可；
    尝试 fact / text / sentence 三个字段名。
    """
    facts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            fact_text = obj.get("fact") or obj.get("text") or obj.get("sentence")
            if not fact_text:
                continue

            tokens = simple_word_tokenize(fact_text)
            facts.append(
                {
                    "text": fact_text,
                    "tokens": tokens,
                }
            )

    print(f"[INFO] Loaded {len(facts)} RAG facts from {path}")
    return facts


def retrieve_facts(
    query: str,
    facts: List[Dict],
    topk: int = 5,
) -> List[str]:
    """
    非常简单的检索器：
      - query / fact 都做 simple_word_tokenize
      - 得分 = |交集词集|
      - 取 top-k score 最大的 fact
      - 如果所有分数都为 0，就退化为前 top-k 条
    以后如果你有 BM25 / FAISS，可以把这里换掉，外面逻辑可以不动。
    """
    if not facts:
        return []

    q_tokens = simple_word_tokenize(query.replace("[MASK]", ""))
    if not q_tokens:
        # 没有有效 query 词，直接退化
        return [f["text"] for f in facts[:topk]]

    q_set = set(q_tokens)
    scored = []

    for idx, f in enumerate(facts):
        f_set = set(f["tokens"])
        score = len(q_set & f_set)
        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [(s, i) for (s, i) in scored[:topk]]

    # 如果得分全 0，退化为前 topk 条
    if top and top[0][0] == 0:
        return [facts[i]["text"] for i in range(min(topk, len(facts)))]

    return [facts[i]["text"] for (s, i) in top]


# ================== 主评测逻辑 ==================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---- 模型 & tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    mask_id = tokenizer.mask_token_id
    emb_weight = model.get_input_embeddings().weight  # [V, H]

    # ---- 数据 ----
    cloze_data = load_cloze(CLOZE_PATH)
    rag_facts = load_rag_facts(RAG_FACTS_PATH)

    total = 0
    correct = 0
    sum_rr = 0.0
    sum_cos = 0.0

    for ex in cloze_data:
        base_text = ex["text"]         # cloze with [MASK]
        gold_word = ex["answer"]

        # ===== RAG: 检索 + 构造输入 =====
        retrieved = retrieve_facts(base_text, rag_facts, topk=RAG_TOPK)
        if retrieved:
            # 用 [SEP] 连接 facts 和 cloze
            sep = f" {tokenizer.sep_token} "
            fused_text = sep.join(retrieved + [base_text])
        else:
            fused_text = base_text

        # ===== gold token id（仍假设单 subword；多 token 时取第一个 piece）=====
        gold_tokens = tokenizer.tokenize(gold_word)
        if len(gold_tokens) != 1:
            print(f"[WARN] example id={ex.get('id')} answer '{gold_word}' "
                  f"tokenized to {gold_tokens}, taking first piece.")
        gold_token = gold_tokens[0]
        gold_id = tokenizer.convert_tokens_to_ids(gold_token)

        # ===== 编码输入 =====
        enc = tokenizer(
            fused_text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(device)        # [1, L]
        attention_mask = enc["attention_mask"].to(device)

        # 找 [MASK] 位置
        mask_pos = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
        if mask_pos.numel() == 0:
            print(f"[WARN] no [MASK] found in example id={ex.get('id')}, skip.")
            continue
        if mask_pos.numel() > 1:
            mask_idx = mask_pos[0].item()
        else:
            mask_idx = mask_pos.item()

        # ===== forward =====
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = outputs.logits      # [1, L, V]
            hidden_states = outputs.hidden_states[-1]  # last layer [1, L, H]

        mask_logits = logits[0, mask_idx, :]          # [V]

        # ---------- 1) Top-1 Accuracy ----------
        pred_id = int(mask_logits.argmax(dim=-1).item())
        if pred_id == int(gold_id):
            correct += 1

        # ---------- 2) MRR ----------
        gold_score = mask_logits[gold_id]
        rank = int((mask_logits > gold_score).sum().item()) + 1
        rr = 1.0 / rank
        sum_rr += rr

        # ---------- 3) Cosine similarity ----------
        h_mask = hidden_states[0, mask_idx, :]            # [H]
        e_gold = emb_weight[gold_id, :]                   # [H]
        cos = F.cosine_similarity(
            h_mask.unsqueeze(0),
            e_gold.unsqueeze(0),
            dim=-1,
        ).item()
        sum_cos += cos

        total += 1

    # ================== 汇总 ==================

    if total == 0:
        print("[ERROR] No valid examples evaluated.")
        return

    acc = correct / total
    mrr = sum_rr / total
    avg_cos = sum_cos / total

    print("========== Evaluation on Cloze with RAG ==========")
    print(f"Total examples:         {total}")
    print(f"Top-1 Accuracy:         {acc:.4f}")
    print(f"Mean Reciprocal Rank:   {mrr:.4f}")
    print(f"Avg Cosine Similarity:  {avg_cos:.4f}")

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"eval_bert_dapt_rag_{timestamp}.json")

    results = {
        "mode": "bert_dapt_rag",
        "timestamp": timestamp,
        "n": total,
        "n_correct": correct,
        "mrr_sum": sum_rr,
        "cos_sum": sum_cos,
        "accuracy": acc,
        "mrr": mrr,
        "avg_cos": avg_cos,
        "rag_topk": RAG_TOPK,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
