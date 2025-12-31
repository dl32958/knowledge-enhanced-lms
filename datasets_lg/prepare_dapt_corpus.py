import json
import os

ROOT = "/home/lu.dong1/rag-embed-study-v4"
RAW_DIR = os.path.join(ROOT, "datasets_lg", "raw")
PROC_DIR = os.path.join(ROOT, "datasets_lg", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

domain_path = os.path.join(RAW_DIR, "domain_articles.jsonl")
dapt_txt_path = os.path.join(PROC_DIR, "dapt_corpus.txt")

min_words = 200

def main():
    texts = []
    
    with open(domain_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text"].strip()
            
            if len(text.split()) < min_words:
                continue
            
            text = "\n".join(p.strip() for p in text.split("\n") if p.strip())
            texts.append(text)

    print(f"valid articles: {len(texts)}")

    with open(dapt_txt_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")

    print(f"saved DAPT corpus to: {dapt_txt_path}")

if __name__ == "__main__":
    main()