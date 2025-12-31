import os
import json
import shutil
from typing import Dict

import torch
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
Stage 2: K-Adapter Fusion DAPT
only do MLM training on dapt_corpus.txt
for backbone+adapter domain adaptation
"""

BERT_DAPT_PATH = "../bert_dapt/bert_base_dapt/epoch_3"
STAGE1_DIR = "kadapter_stage1"
STAGE1_CKPT = f"{STAGE1_DIR}/stage1_kadapter.pt"
PROC_DIR = "../datasets_lg/processed"
DAPT_CORPUS = f"{PROC_DIR}/dapt_corpus.txt"
OUTPUT_DIR = "kadapter_stage2_fusion"
os.makedirs(OUTPUT_DIR, exist_ok=True)


BOTTLE_NECK = 256
MAX_SEQ_LEN = 256
MLM_PROB = 0.15

NUM_EPOCHS = 3
BATCH_SIZE = 32
GRAD_ACC = 2

LR_BERT = 1e-5
LR_ADAPTER = 5e-5

SEED = 42
set_seed(SEED)
assert torch.cuda.is_available(), "CUDA required!"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DAPTTextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        print(f"Loading DAPT corpus: {path}")
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    texts.append(s)
        print(f"Loaded {len(texts)} lines.")

        enc = tokenizer(
            texts,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_special_tokens_mask=True,
        )
        self.encodings = enc

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


def create_optimizer(model, lr_bert, lr_adapter):
    """Create optimizer with different learning rates for different parameter groups"""
    bert_params = []
    adapter_params = []

    for name, param in model.named_parameters():
        if name.startswith("bert_mlm.bert."):
            bert_params.append(param)
        else:
            adapter_params.append(param)

    print(f"#BERT params: {len(bert_params)}, LR={lr_bert}")
    print(f"#Adapter+KG+Head params: {len(adapter_params)}, LR={lr_adapter}")

    return torch.optim.AdamW(
        [
            {"params": bert_params, "lr": lr_bert},
            {"params": adapter_params, "lr": lr_adapter},
        ]
    )


def main():
    print("Stage2 Fusion DAPT running...")

    if not os.path.exists(STAGE1_CKPT):
        raise FileNotFoundError(f"Missing Stage1 checkpoint: {STAGE1_CKPT}")
    if not os.path.exists(DAPT_CORPUS):
        raise FileNotFoundError(f"Missing DAPT corpus: {DAPT_CORPUS}")

    tokenizer = AutoTokenizer.from_pretrained(STAGE1_DIR, use_fast=True)

    # load entity/relation mappings for consistency
    with open(os.path.join(STAGE1_DIR, "ent2id.json"), "r", encoding="utf-8") as f:
        ent2id = json.load(f)
    with open(os.path.join(STAGE1_DIR, "rel2id.json"), "r", encoding="utf-8") as f:
        rel2id = json.load(f)

    num_entities = len(ent2id)
    num_relations = len(rel2id)

    # Dataset
    dataset = DAPTTextDataset(DAPT_CORPUS, tokenizer, MAX_SEQ_LEN)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB
    )

    # same model structure as stage1
    model = BertKAdapterForMaskedLM(
        backbone_path=BERT_DAPT_PATH,
        bottleneck_dim=BOTTLE_NECK,
        num_entities=num_entities,
        num_relations=num_relations,
        freeze_bert=False,   # unfreezed
    )

    # load Stage1 weights
    print("Loading Stage1 checkpoint...")
    state_dict = torch.load(STAGE1_CKPT, map_location="cpu")
    model.load_state_dict(state_dict)
    print("Stage1 checkpoint loaded.")

    # Stage2: all parameters trainable, but backbone with smaller LR
    for p in model.parameters():
        p.requires_grad = True

    optimizer = create_optimizer(model, LR_BERT, LR_ADAPTER)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        logging_steps=50,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to=[],
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        tokenizer=tokenizer,
    )

    print("Training Stage2...")
    trainer.train()

    # Save final
    final_ckpt = os.path.join(OUTPUT_DIR, "stage2_kadapter_fusion.pt")
    torch.save(model.state_dict(), final_ckpt)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Copy ent2id / rel2id to Stage2 directory
    shutil.copy(os.path.join(STAGE1_DIR, "ent2id.json"), OUTPUT_DIR)
    shutil.copy(os.path.join(STAGE1_DIR, "rel2id.json"), OUTPUT_DIR)

    print(f"Stage2 complete! Saved to {final_ckpt}")


if __name__ == "__main__":
    main()
