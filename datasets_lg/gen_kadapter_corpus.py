"""
Generate K-Adapter MLM corpus from KG triples.
"""
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple

KG_PATH = "processed/kg_triples.jsonl"
OUT_PATH = "processed/kadapter_corpus.txt"


def verbalize_triple(head, relation, tail):
    """
    Convert triplet (head, relation, tail) to sentence, suitable for MLM training.
    """
    h = head.strip()
    r = relation.strip().lower()
    t = tail.strip()

    if not h or not t:
        return ""

    # rule-based verbalization patterns
    if r == "instance of":
        return f"{h} is a {t}."
    if r in ("subclass of", "type of", "kind of"):
        return f"{h} is a type of {t}."
    if r == "facet of":
        return f"{h} is a facet of {t}."
    if r == "part of":
        return f"{h} is part of {t}."
    if r == "has part":
        return f"{h} has {t} as a part."
    if r in ("cause of", "causes"):
        return f"{h} can cause {t}."
    if r == "has cause":
        return f"{h} can be caused by {t}."
    if r in ("opposite of", "antonym of"):
        return f"{h} is the opposite of {t}."
    if r in ("related to", "see also"):
        return f"{h} is related to {t}."

    # direct concatenation
    return f"{h} {relation} {t}."


def main():
    if not os.path.isfile(KG_PATH):
        raise FileNotFoundError(f"not found: {KG_PATH}")

    print(f"Loading triples from: {KG_PATH}")

    # Group triples by article_id
    article_triples = defaultdict(list)    # {"article_1": [(h, r, t)...]}

    with open(KG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            art_id = rec["article_id"]
            h = rec["head"]
            r = rec["relation"]
            t = rec["tail"]
            article_triples[art_id].append((h, r, t))
    print(f"Total articles with triples: {len(article_triples)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    num_lines = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for art_id, triples in article_triples.items():
            sentences: List[str] = []
            seen_sentences = set()

            for h, r, t in triples:
                sentence = verbalize_triple(h, r, t)
                sentence = sentence.strip()
                if not sentence or sentence in seen_sentences:
                    continue
                seen_sentences.add(sentence)
                sentences.append(sentence)

            if not sentences:
                continue

            # Combine all facts for this article into a compact paragraph
            paragraph = " ".join(sentences)
            out_f.write(paragraph + "\n")
            num_lines += 1
            
    print(f"Saved K-Adapter MLM corpus to: {OUT_PATH}  lines: {num_lines}")


if __name__ == "__main__":
    main()
