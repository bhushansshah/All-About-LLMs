# dataset_finetune.py
import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from config import Config, EnvironmentConfig

cfg = Config()
env_cfg = EnvironmentConfig()

TOKENIZER_DIR = os.path.join("tokenizer", "llama2")


def get_tokenizer():
    """Load or download the LLaMA-2 tokenizer."""
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    if os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer_config.json")):
        tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True, token=env_cfg.hf_token)
    else:
        print("[Tokenizer] Downloading meta-llama/Llama-2-7b-hf...")
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True, token=env_cfg.hf_token)
        tok.save_pretrained(TOKENIZER_DIR)

    # Ensure special tokens exist
    if tok.bos_token is None:
        tok.add_special_tokens({"bos_token": "<s>"})
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token": "</s>"})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.truncation_side = "right"
    return tok


def load_poem_dataset(data_dir, tokenizer, block_size=cfg.block_size):
    """Prepare dataset for instruction-based fine-tuning."""
    ds = load_from_disk(data_dir)

    def format_example(example):
        prompt = example["INSTRUCTION"].strip()
        answer = example["RESPONSE"].strip()
        text = f"Instruction: {prompt}\nResponse: {answer}"
        return {"text": text}

    ds = ds.map(format_example, remove_columns=ds["train"].column_names)
    ds = ds["train"].train_test_split(test_size=cfg.train_test_split, seed=42)

    def tokenize_fn(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=block_size - 2,  # space for <bos> and <eos>
            padding=False,
        )
        # Add BOS/EOS manually
        input_ids = [
            [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id] for ids in tokenized["input_ids"]
        ]
        return {"input_ids": input_ids}

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return ds


class DataCollatorForInstructionPoem:
    """
    Pads sequences to the same length and masks out instruction tokens (loss only on response).
    """

    def __init__(self, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __call__(self, batch):
        pad_id = self.tokenizer.pad_token_id
        max_len = min(
            max(len(b["input_ids"]) for b in batch),
            self.block_size
        )

        input_ids = []
        labels = []

        # Precompute the tokenized form of "Response:"
        resp_pattern = self.tokenizer.encode("Response:", add_special_tokens=False)

        for b in batch:
            ids = b["input_ids"][:max_len]
            pad_len = max_len - len(ids)
            padded = ids + [pad_id] * pad_len

            # ----- Find where "Response:" begins -----
            start = 0
            for i in range(len(padded) - len(resp_pattern)):
                if padded[i:i + len(resp_pattern)] == resp_pattern:
                    start = i
                    break

            # ----- Build labels -----
            label = padded.copy()

            # Mask everything before response
            for i in range(start):
                label[i] = -100

            # Shift labels by one for causal LM
            label[:-1] = label[1:]
            label[-1] = -100  # last token has no prediction target

            input_ids.append(padded)
            labels.append(label)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def get_dataloaders(ds, tokenizer, batch_size=cfg.batch_size):
    collator = DataCollatorForInstructionPoem(tokenizer, cfg.block_size)
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(ds["test"], batch_size=max(1, batch_size // 2), shuffle=False, collate_fn=collator)
    return train_loader, test_loader
