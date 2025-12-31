"""
Extract KG triples from domain articles using REBEL model
"""

import os
import json
from typing import List, Tuple
import re

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

INPUT_CORPUS = "../raw/domain_articles.jsonl"
OUTPUT_TRIPLES = "../processed/kg_triples.jsonl"

# hyperparameters
MAX_INPUT_TOKENS = 512   # for REBEL
CHUNK_TOKENS = 320       # single chunk length
CHUNK_OVERLAP = 128
BATCH_SIZE = 32
GEN_MAX_LEN = 384     # max generation length
NUM_BEAMS = 5

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required but not available.")
device = "cuda"


MODEL_NAME = "Babelscape/rebel-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()


def make_sliding_windows(text):
    """
    Split long text into overlapping chunks using sliding window. Each chunk is fed to REBEL model separately.
    """
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return []

    chunks = []
    i = 0
    while i < len(tokens):
        window = tokens[i: i + CHUNK_TOKENS]
        if not window:
            break
        chunk_text = tokenizer.convert_tokens_to_string(window)
        chunks.append(chunk_text)

        if i + CHUNK_TOKENS >= len(tokens):
            break
        i += CHUNK_TOKENS - CHUNK_OVERLAP

    return chunks


# head, relation, tail
def parse_rebel_output(text):
    """
    REBEL format: <triplet> RELATION <subj> HEAD_ENTITY <obj> TAIL_ENTITY
    Returns list of (head, relation, tail) triplets.
    """
    triplets = []
    current_relation = ""
    current_head = ""
    current_tail = ""
    parsing_mode = "none"  # "relation", "head", "tail", "none"

    text = text.strip()
    # remove special tokens
    text = (
        text.replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .strip()
    )

    for token in text.split():
        if token == "<triplet>":
            # Save previous triplet if complete
            if current_relation and current_head and current_tail:
                triplets.append(
                    (current_head.strip(), current_relation.strip(), current_tail.strip())
                )
            parsing_mode = "relation"
            current_relation = ""
            current_head = ""
            current_tail = ""
        elif token == "<subj>":
            parsing_mode = "head"
        elif token == "<obj>":
            parsing_mode = "tail"
        else:
            if parsing_mode == "relation":
                current_relation += " " + token
            elif parsing_mode == "head":
                current_head += " " + token
            elif parsing_mode == "tail":
                current_tail += " " + token

    # Save final triplet
    if current_relation and current_head and current_tail:
        triplets.append((current_head.strip(), current_relation.strip(), current_tail.strip()))

    return triplets


def extract_triples_from_chunks(chunks):
    all_triples = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        if not batch:
            continue

        enc = tokenizer(
            batch,
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_length=GEN_MAX_LEN,
                num_beams=NUM_BEAMS,
                num_return_sequences=1,
                length_penalty=1.0,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        for seq in decoded:
            all_triples.extend(parse_rebel_output(seq))

    return all_triples


BAD_CHARS = set("*{}[]()\\^_")  # LaTeX/formula

def is_noisy_token(s: str) -> bool:
    s = s.strip()
    if len(s) <= 2:
        return True
    if sum(c.isalpha() for c in s) <= 1:
        return True
    if any(c in BAD_CHARS for c in s):
        return True
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        return True
    return False

def dedup_and_clean(triples):
    seen = set()
    out = []
    for h, r, t in triples:
        h = h.strip(); r = r.strip(); t = t.strip()
        if not h or not r or not t:
            continue

        if is_noisy_token(h) or is_noisy_token(t):
            continue

        key = (h.lower(), r.lower(), t.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((h, r, t))
    return out

# main loop
triple_id = 0
os.makedirs(os.path.dirname(OUTPUT_TRIPLES), exist_ok=True)
out_f = open(OUTPUT_TRIPLES, "w", encoding="utf-8")

with open(INPUT_CORPUS, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="extract KG"):
        rec = json.loads(line)
        art_id = rec["id"]
        text = rec["text"]

        chunks = make_sliding_windows(text)
        if not chunks:
            continue

        triples = extract_triples_from_chunks(chunks)
        triples = dedup_and_clean(triples)

        for h, r, t in triples:
            record = {
                "triple_id": triple_id,
                "article_id": art_id,
                "head": h,
                "relation": r,
                "tail": t,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            triple_id += 1

out_f.close()

print(f"Saved {triple_id} triples to: {OUTPUT_TRIPLES}")
