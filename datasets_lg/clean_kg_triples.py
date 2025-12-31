import json
import re
from collections import Counter
from typing import Set, Tuple

INPUT_PATH = "processed/kg_triples.jsonl"
OUTPUT_PATH = "processed/kg_triples_cleaned.jsonl"

# Domain relevance keywords
CORE_KEYWORDS = {
    'cognitive bias', 'confirmation bias', 'anchoring bias', 'availability heuristic',
    'representativeness heuristic', 'framing effect', 'loss aversion', 'prospect theory',
    'dual process', 'behavioral economics', 'decision making', 'heuristic',
    'mental model', 'cognitive load', 'working memory'
}

EXTENDED_KEYWORDS = {
    'psychology', 'cognitive', 'behavior', 'behavioral', 'mental', 'reasoning', 
    'judgment', 'decision', 'bias', 'heuristic', 'perception', 'attention',
    'memory', 'learning', 'emotion', 'motivation', 'therapy', 'treatment'
}

CONTEXT_KEYWORDS = {
    'brain', 'mind', 'research', 'study', 'experiment', 'theory', 'model',
    'process', 'mechanism', 'effect', 'disorder', 'syndrome'
}

# High-value relations
HIGH_VALUE_RELATIONS = {
    'causes', 'leads to', 'results in', 'triggers', 'influences', 'affects',
    'contributes to', 'associated with', 'correlated with', 'predicts',
    'is a', 'type of', 'kind of', 'category of', 'includes', 'part of',
    'component of', 'characterized by', 'defined as', 'involves',
    'used for', 'applied to', 'treats', 'helps with', 'reduces', 'improves'
}

def calculate_domain_score(triple):
    """Calculate domain relevance score for a triple"""
    head, rel, tail = triple
    text = f"{head} {rel} {tail}".lower()
    
    score = 0
    
    # Core domain concepts (high weight)
    for keyword in CORE_KEYWORDS:
        if keyword in text:
            score += 3
    
    # Extended domain concepts (medium weight)
    for keyword in EXTENDED_KEYWORDS:
        if keyword in text:
            score += 1
    
    # Context keywords (low weight)
    for keyword in CONTEXT_KEYWORDS:
        if keyword in text:
            score += 0.3
    
    return score

def is_high_value_relation(relation):
    """Check if relation is high-value type"""
    rel_lower = relation.lower().strip()
    return any(hvr in rel_lower for hvr in HIGH_VALUE_RELATIONS)

def is_noise_entity(entity):
    """Check if entity is likely noise"""
    entity = entity.strip()
    
    if len(entity) <= 1:
        return True
    if re.match(r'^[^\w\s]+$', entity):
        return True
    if re.search(r'[<>=+\-*/\\^_{}[\]()]+', entity):
        return True
    if re.match(r'^\d+(\.\d+)?$', entity) or re.match(r'^\d{4}$', entity):
        return True
    if len(entity) <= 3 and entity.isupper():
        common_acronyms = {'IQ', 'EQ', 'CBT', 'DSM', 'ICD', 'fMRI', 'EEG', 'PET'}
        if entity not in common_acronyms:
            return True
    letters = sum(c.isalpha() for c in entity)
    if letters < len(entity) * 0.5:
        return True
    
    return False

def normalize_entity(entity):
    entity = re.sub(r'\s+', ' ', entity.strip())
    entity = re.sub(r'^(the|a|an)\s+', '', entity, flags=re.IGNORECASE)
    entity = re.sub(r'\s+(the|a|an)$', '', entity, flags=re.IGNORECASE)
    
    return entity

def normalize_relation(relation):
    relation = re.sub(r'\s+', ' ', relation.strip())
    relation = relation.lower()
    
    # standardize common relation variants
    relation_map = {
        'is a type of': 'is a',
        'is a kind of': 'is a',
        'is part of': 'part of',
        'is associated with': 'associated with',
        'can cause': 'causes',
        'may cause': 'causes',
        'leads to': 'causes',
        'results in': 'causes'
    }
    
    for variant, standard in relation_map.items():
        if variant in relation:
            relation = relation.replace(variant, standard)
    
    return relation

def should_keep_triple(triple):
    """Decide whether to keep a triple based on quality criteria"""
    head, rel, tail = triple
    
    # Basic noise filtering
    if is_noise_entity(head) or is_noise_entity(tail):
        return False
    
    # Normalize
    head = normalize_entity(head)
    tail = normalize_entity(tail)
    rel = normalize_relation(rel)
    
    # Skip if normalization made them too short
    if len(head) < 2 or len(tail) < 2 or len(rel) < 2:
        return False
    
    normalized_triple = (head, rel, tail)
    
    # Calculate domain relevance score
    domain_score = calculate_domain_score(normalized_triple)
    
    # High-value relations can use lower threshold, but still need domain relevance
    if is_high_value_relation(rel):
        return domain_score >= 0.3  # Lower threshold for high-value relations
    else:
        return domain_score >= 0.5  # Normal threshold

def clean_triples():
    """Main cleaning function"""
    
    print(f"[INFO] Reading from: {INPUT_PATH}")
    
    original_count = 0
    kept_count = 0
    seen_normalized = set()
    
    # Statistics tracking
    high_value_kept = 0
    regular_kept = 0
    noise_filtered = 0
    domain_filtered = 0
    duplicate_filtered = 0
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            original_count += 1
            
            try:
                record = json.loads(line.strip())
                head = record['head']
                rel = record['relation'] 
                tail = record['tail']
                
                triple = (head, rel, tail)
                
                # Check for basic noise
                if is_noise_entity(head) or is_noise_entity(tail):
                    noise_filtered += 1
                    continue
                
                # Normalize
                norm_head = normalize_entity(head)
                norm_rel = normalize_relation(rel)
                norm_tail = normalize_entity(tail)
                
                # Check normalized lengths
                if len(norm_head) < 2 or len(norm_tail) < 2 or len(norm_rel) < 2:
                    noise_filtered += 1
                    continue
                
                normalized_triple = (norm_head, norm_rel, norm_tail)
                
                # Domain score check
                domain_score = calculate_domain_score(normalized_triple)
                is_high_value = is_high_value_relation(norm_rel)
                
                # Apply thresholds
                threshold = 0.3 if is_high_value else 0.5
                if domain_score < threshold:
                    domain_filtered += 1
                    continue
                
                # Deduplication
                dedup_key = (norm_head.lower(), norm_rel.lower(), norm_tail.lower())
                if dedup_key in seen_normalized:
                    duplicate_filtered += 1
                    continue
                
                seen_normalized.add(dedup_key)
                
                # Write cleaned record
                cleaned_record = {
                    'triple_id': kept_count,
                    'article_id': record.get('article_id', ''),
                    'head': norm_head,
                    'relation': norm_rel,
                    'tail': norm_tail,
                }
                
                outfile.write(json.dumps(cleaned_record, ensure_ascii=False) + '\n')
                kept_count += 1
                
                # Track statistics
                if is_high_value:
                    high_value_kept += 1
                else:
                    regular_kept += 1
                
            except Exception as e:
                print(f"[WARN] Error processing line {original_count}: {e}")
                continue
    
    # Statistics
    reduction_rate = (original_count - kept_count) / original_count * 100
    
    print(f"\n[INFO] Cleaning completed:")
    print(f"  - Original triples: {original_count:,}")
    print(f"  - Final kept triples: {kept_count:,}")
    print(f"    * High-value relations: {high_value_kept:,}")
    print(f"    * Regular relations: {regular_kept:,}")
    print(f"  - Filtered out:")
    print(f"    * Noise entities: {noise_filtered:,}")
    print(f"    * Low domain score: {domain_filtered:,}")
    print(f"    * Duplicates: {duplicate_filtered:,}")
    print(f"  - Reduction rate: {reduction_rate:.1f}%")
    print(f"  - Output saved to: {OUTPUT_PATH}")
    
    # Sample results for inspection
    print(f"\n[INFO] Sample cleaned triples (first 10):")
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            record = json.loads(line)
            domain_score = calculate_domain_score((record['head'], record['relation'], record['tail']))
            is_high_val = is_high_value_relation(record['relation'])
            marker = " [HV]" if is_high_val else ""
            print(f"  {record['head']} --[{record['relation']}]--> {record['tail']} (score: {domain_score:.1f}){marker}")

if __name__ == "__main__":
    clean_triples()