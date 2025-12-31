import os
import json
import random
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment")

KG_PATH = "processed/kg_triples_cleaned.jsonl"
OUT_PATH = "processed/cloze_200_llm.jsonl"

client = OpenAI(api_key=OPENAI_API_KEY)
random.seed(42)


def last_content_word(phrase):
    toks = phrase.strip().split()
    if not toks:
        return None
    return toks[-1]


def mask_last_word(phrase: str, answer_word: str) -> str:
    phrase = phrase.strip()
    if not phrase.lower().endswith(answer_word.lower()):
        return phrase.replace(answer_word, "[MASK]")

    idx = phrase.lower().rfind(answer_word.lower())
    return phrase[:idx] + "[MASK]" + phrase[idx + len(answer_word):]


def build_prompt(
    head: str,
    relation: str,
    tail: str,
    mask_target: str,
    masked_phrase: str,
    other_phrase: str,
    answer_word: str,
) -> str:
    if mask_target == "head":
        masked_label = "HEAD"
        other_label = "TAIL"
    else:
        masked_label = "TAIL"
        other_label = "HEAD"

    return f"""
You are generating a cloze-style fact sentence.

FACT:
- HEAD = "{head}"
- RELATION = "{relation}"
- TAIL = "{tail}"

TASK:
1. Write a **single English sentence** (minimum 10 words) that accurately expresses this FACT.
2. The sentence **must mention BOTH** the head concept and the tail concept:
   - Use the following **masked phrase** for the {masked_label} concept **exactly once**:
       >>> {masked_phrase}
   - Also include the {other_label} concept **verbatim as a phrase**:
       >>> {other_phrase}
3. The masked phrase must appear randomly at the BEGINNING or MIDDLE or END of the sentence
   (choose randomly; do not always put it at the end).
4. The sentence must remain grammatical and natural.
5. Do NOT reveal the answer. Do NOT include the unmasked original word "{answer_word}" anywhere.
6. Output ONLY the sentence (no explanations, no quotes).

The answer is the masked word: "{answer_word}"
Generate the sentence now:
""".strip()


def call_gpt(prompt, max_retry = 5):
    for _ in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"GPT error: {e}")
            time.sleep(2)
    return None


"""
Generate Cloze from KG using GPT
"""
def main():
    print(f"Loading KG triples from: {KG_PATH}")

    triples = []
    with open(KG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            triples.append(json.loads(line))

    print(f"Loaded {len(triples)} KG triples (cleaned).")

    random.shuffle(triples)
    target_n = 200
    results = []   # list of dicts
    existing_clozes = set()

    for tri in triples:
        if len(results) >= target_n:
            break

        head = tri["head"]
        tail = tri["tail"]
        relation = tri["relation"]

        # randomly decide to mask h/t
        mask_target = random.choice(["head", "tail"])
        phrase = head if mask_target == "head" else tail
        other_phrase = tail if mask_target == "head" else head

        answer = last_content_word(phrase)
        if not answer or len(answer) < 3:
            continue

        masked_phrase = mask_last_word(phrase, answer)

        prompt = build_prompt(
            head=head,
            relation=relation,
            tail=tail,
            mask_target=mask_target,
            masked_phrase=masked_phrase,
            other_phrase=other_phrase,
            answer_word=answer,
        )

        sentence = call_gpt(prompt)
        if not sentence:
            continue

        # quality checks
        sent_lower = sentence.lower()
        masked_lower = masked_phrase.lower()
        head_lower = head.lower()
        tail_lower = tail.lower()
        answer_lower = answer.lower()

        if len(sentence.split()) < 10:
            continue
        if sentence in existing_clozes:
            continue
        if masked_lower not in sent_lower:
            continue
        if mask_target == "head":
            if tail_lower not in sent_lower:
                continue
        else:
            if head_lower not in sent_lower:
                continue
        if answer_lower in sent_lower.replace("[mask]", ""):
            continue

        out = {
            "id": f"cloze_{len(results) + 1:03d}",
            "triple_head": head,
            "triple_relation": relation,
            "triple_tail": tail,
            "masked_target": mask_target,
            "cloze": sentence,
            "answer": answer,
        }

        results.append(out)
        existing_clozes.add(sentence)
        print(f"[OK] ({len(results)}/{target_n}) {sentence}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} cloze questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
