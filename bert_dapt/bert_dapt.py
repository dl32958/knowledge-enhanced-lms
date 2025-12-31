import os
import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)


CORPUS_PATH = "../datasets_lg/processed/dapt_corpus.txt"
OUTPUT_DIR = "bert_base_dapt"

MODEL_NAME = "bert-base-cased"
MAX_SEQ_LEN = 256
MLM_PROB = 0.15

NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        # tokenize all texts into encodings
        encodings: Dict[str, List[List[int]]] = tokenizer(
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
        # returns dict[str, tensor]
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


# Save model and tokenizer at the end of each epoch
class EpochCheckpointCallback(TrainerCallback):
    def __init__(self, tokenizer, base_output_dir):
        self.tokenizer = tokenizer
        self.base_output_dir = base_output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is None:
            return control

        epoch_idx = int(round(state.epoch))
        save_dir = os.path.join(self.base_output_dir, f"epoch_{epoch_idx}")
        os.makedirs(save_dir, exist_ok=True)

        model = kwargs["model"]
        print(f"[INFO] Saving epoch {epoch_idx} checkpoint to: {save_dir}")
        model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        return control


def load_corpus(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            texts.append(s)

    print(f"Loaded {len(texts)} lines from corpus.")
    return texts



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)

    texts = load_corpus(CORPUS_PATH)

    random.seed(SEED)
    random.shuffle(texts)

    n_total = len(texts)
    n_eval = max(1, int(0.05 * n_total))       # Train/Eval, 95/5
    eval_texts = texts[:n_eval]
    train_texts = texts[n_eval:]

    print(f"Train samples: {len(train_texts)}, Eval samples: {len(eval_texts)}")

    # load tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    # build dataset
    train_dataset = TextDataset(train_texts, tokenizer, MAX_SEQ_LEN)
    eval_dataset  = TextDataset(eval_texts, tokenizer, MAX_SEQ_LEN)

    # MLM data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROB,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,

        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,

        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="no",

        prediction_loss_only=True,
        fp16=True,
    )

    epoch_cb = EpochCheckpointCallback(tokenizer=tokenizer, base_output_dir=OUTPUT_DIR)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[epoch_cb],
    )

    print("Start DAPT training ...")
    trainer.train()

    print("Saving final DAPT model to:", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training completed.")


if __name__ == "__main__":
    main()
