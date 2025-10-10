# dataset.py
import os
import torch
from functools import partial
from datasets import load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from configs import Config

cfg = Config()

# Directory where the tokenizer will be stored locally
TOKENIZER_DIR = os.path.join("tokenizer", "llama2")

def get_tokenizer():
    """
    Load the official LLaMA-2 tokenizer.
    If it’s already downloaded locally, load from disk.
    Otherwise, download from Hugging Face and save for future use.
    """
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    # If tokenizer already saved locally
    if os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer_config.json")):
        print(f"[Tokenizer] Loading LLaMA-2 tokenizer from local directory: {TOKENIZER_DIR}")
        tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
    else:
        print("[Tokenizer] Downloading 'meta-llama/Llama-2-7b-hf' tokenizer from Hugging Face...")
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
        print(f"[Tokenizer] Saving tokenizer to {TOKENIZER_DIR}")
        tok.save_pretrained(TOKENIZER_DIR)

    # Ensure EOS/PAD tokens are defined for LM training
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token": "</s>"})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return tok


def load_and_prepare_dataset(data_dir=cfg.data_dir, tokenizer=None, block_size=cfg.block_size):
    """
    Load the Wikipedia dataset (saved with `save_to_disk`) and tokenize + chunk it.
    """
    ds = load_from_disk(data_dir)
    if tokenizer is None:
        tokenizer = get_tokenizer()

    # Determine text column name
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]

    # Tokenize each example
    def tokenize_example(ex):
        out = tokenizer(ex[text_col], truncation=False)
        return {"input_ids": out["input_ids"]}

    ds = ds.map(tokenize_example, batched=False, num_proc=4, remove_columns=ds.column_names)

    # Concatenate and chunk into blocks
    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // block_size) * block_size
        result = {
            "input_ids": [concatenated[i: i + block_size] for i in range(0, total_len, block_size)]
        }
        return result

    ds = ds.map(group_texts, batched=True, batch_size=1000, num_proc=4)

    # Split into train/test sets
    ds = ds.train_test_split(test_size=cfg.train_test_split, seed=42)
    return ds, tokenizer


class DataCollatorForCausal:
    """
    Simple collator for causal language modeling:
    returns input_ids and labels (same sequence shifted by one internally by the model).
    """
    def __init__(self, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __call__(self, batch):
        input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def get_dataloaders(ds, tokenizer, batch_size=cfg.batch_size):
    train_ds = ds["train"]
    test_ds = ds["test"]
    collator = DataCollatorForCausal(tokenizer, cfg.block_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=max(1, batch_size // 2), shuffle=False,
        collate_fn=collator, num_workers=2
    )
    return train_loader, test_loader
