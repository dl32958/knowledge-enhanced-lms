import os
import json
import random
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from kadapter_model import BertKAdapterForMaskedLM

"""
Stage 1: knowledge injection

Text Task + KG Task
BERT encoder frozen, only train adapter, kg_proj, ent_emb / rel_emb, MLM head (cls)
"""

BERT_DAPT_PATH = "../bert_dapt/bert_base_dapt/epoch_3"
PROC_DIR = "../datasets_lg/processed"
KADAPTER_CORPUS = f"{PROC_DIR}/kadapter_corpus.txt"
KG_TRIPLES_PATH = f"{PROC_DIR}/kg_triples.jsonl"
OUTPUT_DIR = "kadapter_stage1"
os.makedirs(OUTPUT_DIR, exist_ok=True)


BOTTLE_NECK = 256    # adapter bottleneck dimension

MAX_SEQ_LEN = 256
MLM_PROB = 0.15

NUM_EPOCHS = 5
BATCH_SIZE = 32

LR_ADAPTER = 5e-4
LAMBDA_MLM = 1.0     # Text MLM loss
LAMBDA_KG = 5.0     # KG-MLM loss

KG_BATCH_SIZE = 32
SEED = 42
set_seed(SEED)

torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# dataset for K-Adapter text corpus
class KAdapterTextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    texts.append(s)
        print(f"Loaded {len(texts)} lines from {path}")

        encodings = tokenizer(
            texts,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_special_tokens_mask=True,
        )
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


# load KG triples
def load_kg_triples(path: str):
    """
        triples_tensor: [N, 3] (h_id, r_id, t_id)
        ent2id: {entity_text -> entity_id}   str -> int
        rel2id: {rel_text -> entity_id}  str -> int
        id2ent: {id -> entity_text}   int -> str
        id2rel: {id -> rel_text}   int -> str
    """
    ent2id = {}
    rel2id = {}
    id2ent = {}
    id2rel = {}
    triples = []

    def get_ent_id(e):
        if e not in ent2id:
            idx = len(ent2id)
            ent2id[e] = idx
            id2ent[idx] = e
        return ent2id[e]  # int

    def get_rel_id(r):
        if r not in rel2id:
            idx = len(rel2id)
            rel2id[r] = idx
            id2rel[idx] = r
        return rel2id[r]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            h = rec["head"].strip()
            r = rec["relation"].strip()
            t = rec["tail"].strip()
            if not h or not r or not t:
                continue

            h_id = get_ent_id(h)
            r_id = get_rel_id(r)
            t_id = get_ent_id(t)
            triples.append((h_id, r_id, t_id))

    triples_tensor = torch.tensor(triples, dtype=torch.long)
    print(f"KG triples: {len(triples_tensor)}")
    print(f"#entities: {len(ent2id)}, #relations: {len(rel2id)}")

    # save mapping
    with open(os.path.join(OUTPUT_DIR, "ent2id.json"), "w", encoding="utf-8") as f:
        json.dump(ent2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "rel2id.json"), "w", encoding="utf-8") as f:
        json.dump(rel2id, f, ensure_ascii=False, indent=2)

    return triples_tensor, ent2id, rel2id, id2ent, id2rel


# ------------------ 自定义 Trainer ------------------

class KAdapterTrainer(Trainer):
    """Text MLM + KG-MLM"""
    def __init__(self, kg_triples, id2ent, id2rel, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.kg_triples = kg_triples     # [N, 3]
        self.id2ent = id2ent             # {id: entity_text}
        self.id2rel = id2rel             # {id: rel_text}
        self.tokenizer = tokenizer
        self.num_triples = kg_triples.size(0)

    def _build_kg_mlm_batch(self, device):
        """
        construct KG-MLM batch, <head> <relation> [MASK]
        """
        tokenizer = self.tokenizer
        mask_token = tokenizer.mask_token
        mask_token_id = tokenizer.mask_token_id

        # randomly sample KG triples
        idx = torch.randint(0, self.num_triples, (KG_BATCH_SIZE,))
        batch_triples = self.kg_triples[idx]  # [Bkg, 3]

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        head_ids_list = []
        rel_ids_list = []
        tail_ids_list = []
        mask_pos_list = []

        for (h_id, r_id, t_id) in batch_triples.tolist():
            head_text = self.id2ent[int(h_id)]
            tail_text = self.id2ent[int(t_id)]
            rel_text = self.id2rel[int(r_id)]

            # template: predict tail
            text = f"{head_text} {rel_text} {mask_token}"

            enc = tokenizer(
                text,
                max_length=MAX_SEQ_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            ids = enc["input_ids"][0]        # [L]
            attn = enc["attention_mask"][0]  # [L]

            # find [MASK] position
            mask_positions = (ids == mask_token_id).nonzero(as_tuple=True)[0]
            if mask_positions.numel() == 0:
                continue
            mask_idx = mask_positions[0].item()

            # gold tail -> single token
            gold_tokens = tokenizer.tokenize(tail_text)
            if len(gold_tokens) == 0:
                continue
            if len(gold_tokens) > 1:
                gold_token = gold_tokens[0]
            else:
                gold_token = gold_tokens[0]
            gold_id = tokenizer.convert_tokens_to_ids(gold_token)

            # only mask_idx has gold, others -100
            labels = torch.full_like(ids, -100)
            labels[mask_idx] = gold_id

            input_ids_list.append(ids)
            attention_mask_list.append(attn)
            labels_list.append(labels)
            head_ids_list.append(h_id)
            rel_ids_list.append(r_id)
            tail_ids_list.append(t_id)
            mask_pos_list.append(mask_idx)

        if len(input_ids_list) == 0:
            return None

        input_ids = torch.stack(input_ids_list).to(device)
        attention_mask = torch.stack(attention_mask_list).to(device)
        labels = torch.stack(labels_list).to(device)
        head_ids = torch.tensor(head_ids_list, dtype=torch.long, device=device)
        rel_ids = torch.tensor(rel_ids_list, dtype=torch.long, device=device)
        tail_ids = torch.tensor(tail_ids_list, dtype=torch.long, device=device)
        mask_positions = torch.tensor(mask_pos_list, dtype=torch.long, device=device)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "head_ids": head_ids,
            "rel_ids": rel_ids,
            "tail_ids": tail_ids,
            "mask_positions": mask_positions,
        }
        return batch

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        otal loss = Text MLM (kadapter_corpus) + KG-MLM (triples-based cloze)
        """
        # 1. text MLM loss（kadapter_corpus）
        outputs = model(**inputs)
        mlm_loss = outputs["loss"]

        device = next(model.parameters()).device

        # 2. KG-MLM loss
        kg_batch = self._build_kg_mlm_batch(device)
        if kg_batch is not None:
            kg_outputs = model(
                input_ids=kg_batch["input_ids"],
                attention_mask=kg_batch["attention_mask"],
                labels=kg_batch["labels"],
                head_ids=kg_batch["head_ids"],
                rel_ids=kg_batch["rel_ids"],
                tail_ids=kg_batch["tail_ids"],
                mask_positions=kg_batch["mask_positions"],
                use_kg=True,
            )
            kg_mlm_loss = kg_outputs["loss"]
            loss = LAMBDA_MLM * mlm_loss + LAMBDA_KG * kg_mlm_loss
        else:
            loss = LAMBDA_MLM * mlm_loss

        if return_outputs:
            return loss, outputs
        return loss


# ------------------ main ------------------

def main():
    print("Stage1 K-Adapter pretraining (Knowledge Injection)")
    print(f"Backbone: {BERT_DAPT_PATH}")
    print(f"Text corpus: {KADAPTER_CORPUS}")
    print(f"KG triples: {KG_TRIPLES_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_DAPT_PATH, use_fast=True)

    # Text data
    dataset = KAdapterTextDataset(KADAPTER_CORPUS, tokenizer, MAX_SEQ_LEN)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB
    )

    # KG data
    kg_triples, ent2id, rel2id, id2ent, id2rel = load_kg_triples(KG_TRIPLES_PATH)

    # Model
    model = BertKAdapterForMaskedLM(
        backbone_path=BERT_DAPT_PATH,
        bottleneck_dim=BOTTLE_NECK,
        num_entities=len(ent2id),
        num_relations=len(rel2id),
        freeze_bert=True,   # stage1 freeze BERT
    )

    # freeze BERT encoder, only train adapter, kg_proj, ent_emb, rel_emb, MLM head
    for name, p in model.named_parameters():
        if name.startswith("bert_mlm.bert."):
            p.requires_grad = False
        else:
            p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR_ADAPTER)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_steps=100,
        save_strategy="no",
        eval_strategy="no",
        fp16=torch.cuda.is_available(),
        prediction_loss_only=True,
        report_to=[],
    )

    trainer = KAdapterTrainer(
        kg_triples=kg_triples,
        id2ent=id2ent,
        id2rel=id2rel,
        tokenizer=tokenizer,
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )

    trainer.train()

    ckpt = os.path.join(OUTPUT_DIR, "stage1_kadapter.pt")
    torch.save(model.state_dict(), ckpt)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"[INFO] Stage1 finished. Saved → {ckpt}")


if __name__ == "__main__":
    main()
