import os
import re
import json
import time
import random
import hashlib
from typing import List, Dict, Set, Optional

import wikipediaapi
import requests
from tqdm import tqdm


ROOT = "/home/lu.dong1/rag-embed-study-v4"

DATASET_DIR = os.path.join(ROOT, "datasets_lg")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

OUTPUT_JSONL = os.path.join(RAW_DIR, "domain_articles.jsonl")
SEEN_TITLES_TXT = os.path.join(RAW_DIR, "seen_titles.txt")

CATEGORIES = [
    "Category:Cognitive biases",
    "Category:Heuristics",
    "Category:Decision-making",
]

TARGET_PER_CAT = 500
MAX_DEPTH = 3

# Network retry params + rate limiting
MAX_RETRIES = 5
BASE_BACKOFF = 2.0
JITTER = 0.5
SLEEP_CAT = 0.3
SLEEP_ART = 0.4

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="rag-embed-study-v4/1.0 (lu.dong1@northeastern.edu)"
)


CUT_HEADS = [
    "\n== References ==",
    "\n== Reference ==",
    "\n== Bibliography ==",
    "\n== External links ==",
    "\n== See also ==",
    "\n== Notes ==",
    "\n== Further reading ==",
]


def clean_wiki_text(text):
    """Clean Wikipedia article text by removing references and formulas."""
    if not text:
        return ""

    # Cut references section
    cut_pos = len(text)
    for h in CUT_HEADS:
        p = text.find(h)
        if p != -1 and p < cut_pos:
            cut_pos = p
    text = text[:cut_pos]

    # Remove math formulas
    text = re.sub(r"<math>.*?</math>", " ", text, flags=re.DOTALL)
    text = re.sub(r"\{\\displaystyle.*?\}", " ", text, flags=re.DOTALL)

    # Compress whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_category_page_safe(cat_name):
    """Safely get category page with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            page = wiki.page(cat_name)
            _ = page.exists()
            _ = page.categorymembers
            return page
        except Exception as e:
            wait = BASE_BACKOFF * (2 ** attempt)
            print(f"[WARN] Category '{cat_name}' failed, retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s")
            time.sleep(wait + random.uniform(0, JITTER))

    print(f"[ERROR] Category '{cat_name}' not accessible after {MAX_RETRIES} retries")
    return None


def get_article_page_safe(title):
    """Safely get article page with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            page = wiki.page(title)
            if not page.exists():
                return None
            _ = page.text
            return page
        except Exception as e:
            wait = BASE_BACKOFF * (2 ** attempt)
            print(f"[WARN] Article '{title}' failed, retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s")
            time.sleep(wait + random.uniform(0, JITTER))

    print(f"[ERROR] Article '{title}' failed after {MAX_RETRIES} retries")
    return None


def load_seen_titles(path):
    """Load previously seen titles from file."""
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                seen.add(t)
    print(f"[INFO] Loaded {len(seen)} seen titles from {path}")
    return seen


def save_seen_titles(path, seen):
    """Save seen titles to file."""
    with open(path, "w", encoding="utf-8") as f:
        for t in sorted(seen):
            f.write(t + "\n")
    print(f"[INFO] Saved {len(seen)} seen titles to {path}")


def collect_titles_bfs(cat_name, target=240, max_depth=3, seen_titles=None):
    """Collect article titles using BFS over categories."""
    if seen_titles is None:
        seen_titles = set()

    cat_page = get_category_page_safe(cat_name)
    if cat_page is None:
        return []

    titles = []
    visited_cats = set()
    queue = [(cat_page, 0)]

    while queue and len(titles) < target:
        page, depth = queue.pop(0)
        if page.title in visited_cats:
            continue
        visited_cats.add(page.title)

        try:
            members = list(page.categorymembers.items())
        except Exception as e:
            print(f"[WARN] Failed to access categorymembers for '{page.title}': {e}")
            continue

        for _, member in members:
            # Articles (namespace 0)
            if member.ns == 0:
                if (member.title not in titles) and (member.title not in seen_titles):
                    titles.append(member.title)
                    if len(titles) >= target:
                        break
            # Subcategories (namespace 14)
            elif member.ns == 14 and depth < max_depth:
                queue.append((member, depth + 1))

        time.sleep(SLEEP_CAT + random.uniform(0, JITTER))

    return titles


def fetch_articles(titles):
    """Fetch full text for article titles."""
    records = []
    for title in tqdm(titles, desc="Fetching articles"):
        page = get_article_page_safe(title)
        if page is None:
            continue

        raw_text = page.text or ""
        text = clean_wiki_text(raw_text)

        # Filter short articles
        if len(text.split()) < 200:
            continue

        rec = {
            "id": hashlib.md5(title.encode("utf-8")).hexdigest(),
            "title": page.title,
            "url": page.fullurl,
            "text": text,
        }
        records.append(rec)

        time.sleep(SLEEP_ART + random.uniform(0, JITTER))

    return records


def append_jsonl(path, records):
    """Append records to JSONL file."""
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] Appended {len(records)} records to {path}")


def main():
    random.seed(42)

    # Load seen titles
    seen_titles = load_seen_titles(SEEN_TITLES_TXT)
    all_new_titles = []

    # BFS for each category
    for cat in CATEGORIES:
        print(f"\n[INFO] Collecting titles for {cat}...")
        titles = collect_titles_bfs(
            cat_name=cat,
            target=TARGET_PER_CAT,
            max_depth=MAX_DEPTH,
            seen_titles=seen_titles,
        )
        print(f"[INFO] {cat}: collected {len(titles)} new titles")
        all_new_titles.extend(titles)
        seen_titles.update(titles)

        # Save seen titles after each category
        save_seen_titles(SEEN_TITLES_TXT, seen_titles)

    print(f"\n[INFO] Total new titles to fetch: {len(all_new_titles)}")
    random.shuffle(all_new_titles)

    # Fetch article content
    articles = fetch_articles(all_new_titles)
    print(f"[INFO] Fetched {len(articles)} usable articles (>= 200 words)")

    # Save to JSONL
    append_jsonl(OUTPUT_JSONL, articles)
    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()