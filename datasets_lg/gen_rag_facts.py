import os
import json
from typing import List

KG_PATH = "processed/kg_triples.jsonl"
OUT_PATH = "processed/rag_facts.jsonl"

def verbalize_triple(head: str, relation: str, tail: str) -> str:
    """
    Convert triplet to sentences for RAG retrieval.
    """
    h = head.strip()
    r = relation.strip().lower()
    t = tail.strip()

    if not h or not t:
        return ""

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

    return f"{h} {relation} {t}."

"""
Generate RAG fact from KG triples.
{
    "fact_id",
    "triple_id",
    "article_id",
    "head",
    "relation",
    "tail",
    "text": sentence,
}
"""
def main():
    if not os.path.isfile(KG_PATH):
        raise FileNotFoundError(KG_PATH)

    print("Loading triples from:", KG_PATH)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    fact_id = 0
    num_written = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        with open(KG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)

                triple_id = rec["triple_id"]
                art_id = rec["article_id"]
                h = rec["head"]
                r = rec["relation"]
                t = rec["tail"]

                text = verbalize_triple(h, r, t).strip()
                if not text:
                    continue

                fact = {
                    "fact_id": fact_id,
                    "triple_id": triple_id,
                    "article_id": art_id,
                    "head": h,
                    "relation": r,
                    "tail": t,
                    "text": text,
                }
                out_f.write(json.dumps(fact, ensure_ascii=False) + "\n")
                fact_id += 1
                num_written += 1

    print(f"Saved RAG fact sentences to: {OUT_PATH}  total facts: {num_written}")


if __name__ == "__main__":
    main()
