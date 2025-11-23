# dataset.py
import os
import itertools
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader

from datasets import load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer

from config import Config, EnvironmentConfig

cfg = Config()
env_cfg = EnvironmentConfig()

# where to persist the tokenizer locally
TOKENIZER_DIR = os.path.join("tokenizer", "llama")


# ---------------------------------------------------------
# Tokenizer loader / initializer
# ---------------------------------------------------------
def get_tokenizer(tokenizer_dir: str = TOKENIZER_DIR, hf_token: Optional[str] = None):
    """
    Load or download the LLaMA tokenizer. Ensures eos and pad tokens exist.
    Returns a HuggingFace tokenizer instance.
    """

    os.makedirs(tokenizer_dir, exist_ok=True)
    local_config = os.path.join(tokenizer_dir, "tokenizer_config.json")

    if os.path.exists(local_config):
        print(f"[Tokenizer] Loading tokenizer from local directory: {tokenizer_dir}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, use_auth_token=hf_token)
    else:
        print("[Tokenizer] Downloading 'meta-llama/Llama-2-7b-hf' tokenizer from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True, use_auth_token=hf_token)
        print(f"[Tokenizer] Saving tokenizer to {tokenizer_dir}")
        tokenizer.save_pretrained(tokenizer_dir)

    # Ensure EOS and PAD tokens exist and persist the tokenizer if changed
    changed = False
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        print("[Tokenizer] Added eos_token </s>")
        changed = True

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        print("[Tokenizer] Added pad_token <pad>")
        changed = True

    if changed:
        tokenizer.save_pretrained(tokenizer_dir)

    print("[Tokenizer] Ready.")
    return tokenizer


# ---------------------------------------------------------
# Core pipeline: split documents, tokenize, pack into blocks
# ---------------------------------------------------------
def _tokenize_batch(batch, tokenizer, text_col: str):
    """Helper for dataset.map: tokenizes a batch of examples (no truncation)."""
    # return tokenizers standard dict: input_ids, attention_mask (we drop attention_mask later)
    return tokenizer(batch[text_col], truncation=False, add_special_tokens=False, return_attention_mask=False)


def _group_texts(examples, block_size: int, eos_token_id: int):
    """
    Group tokenized examples into fixed-size blocks for pretraining.
    - inserts eos_token_id at the end of every document before concatenation.
    - flattens efficiently using itertools.chain.from_iterable
    """
    # Each entry in examples["input_ids"] is a list of token ids (one document)
    docs_with_eos = (doc + [eos_token_id] for doc in examples["input_ids"])  # generator
    concatenated = list(itertools.chain.from_iterable(docs_with_eos))  # O(n)

    if len(concatenated) < block_size:
        return {"input_ids": []}  # not enough tokens from this batch

    total_len = (len(concatenated) // block_size) * block_size
    chunks = [
        concatenated[i: i + block_size]
        for i in range(0, total_len, block_size)
    ]
    return {"input_ids": chunks}


def load_and_prepare_dataset(
    data_dir: str = cfg.data_dir,
    tokenizer: Optional[object] = None,
    block_size: int = cfg.block_size,
    test_fraction: float = cfg.train_test_split,
    num_proc: int = 4,
    seed: int = 42,
) -> Tuple[DatasetDict, object]:
    """
    Full pretraining dataset preparation:
      1. Load dataset saved with `datasets.save_to_disk`
      2. Shuffle (documents)
      3. Split train/test (documents are split here to avoid leakage)
      4. Tokenize each split (no truncation)
      5. Insert EOS between documents, concatenate and chunk into block_size samples
    Returns:
      - prepared datasets (DatasetDict with 'train' and 'test', where each item has input_ids:list[int] of length block_size)
      - tokenizer object
    """

    # ---------------------------
    # Load dataset and tokenizer
    # ---------------------------
    print(f"[Data] Loading dataset from: {data_dir}")
    ds = load_from_disk(data_dir)
    if tokenizer is None:
        tokenizer = get_tokenizer(hf_token=env_cfg.hf_token)

    # figure out text column
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]
    print(f"[Data] Using text column: {text_col}")

    # ---------------------------
    # Shuffle and split (document-level)
    # ---------------------------
    print("[Data] Shuffling dataset (documents) ...")
    ds = ds.shuffle(seed=seed)

    print(f"[Data] Splitting dataset into train/test with test_fraction={test_fraction}")
    split_ds = ds.train_test_split(test_size=test_fraction, seed=seed)

    # ---------------------------
    # Tokenize each split (no truncation)
    # ---------------------------
    print("[Data] Tokenizing train split ...")
    split_ds["train"] = split_ds["train"].map(
        lambda batch: _tokenize_batch(batch, tokenizer, text_col),
        batched=True,
        num_proc=num_proc,
        remove_columns=split_ds["train"].column_names,
    )

    print("[Data] Tokenizing test split ...")
    split_ds["test"] = split_ds["test"].map(
        lambda batch: _tokenize_batch(batch, tokenizer, text_col),
        batched=True,
        num_proc=max(1, num_proc // 2),
        remove_columns=split_ds["test"].column_names,
    )

    # ---------------------------
    # Group tokenized data into blocks
    # ---------------------------
    eos_id = tokenizer.eos_token_id
    print(f"[Data] Grouping tokenized train split into blocks of {block_size} tokens ...")
    split_ds["train"] = split_ds["train"].map(
        lambda examples: _group_texts(examples, block_size=block_size, eos_token_id=eos_id),
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=split_ds["train"].column_names,
    )

    print(f"[Data] Grouping tokenized test split into blocks of {block_size} tokens ...")
    split_ds["test"] = split_ds["test"].map(
        lambda examples: _group_texts(examples, block_size=block_size, eos_token_id=eos_id),
        batched=True,
        batch_size=1000,
        num_proc=max(1, num_proc // 2),
        remove_columns=split_ds["test"].column_names,
    )

    # After grouping, each split has column "input_ids", where each row is a block-length list
    # Optionally drop empty rows (some batches can return empty lists)
    def filter_empty(example):
        return len(example["input_ids"]) == block_size

    print("[Data] Filtering only full blocks ...")
    split_ds["train"] = split_ds["train"].filter(filter_empty)
    split_ds["test"] = split_ds["test"].filter(filter_empty)

    print("[Data] Dataset preparation complete.")
    return split_ds, tokenizer


# ---------------------------------------------------------
# Collator & dataloader builders
# ---------------------------------------------------------
class DataCollatorForCausalLM:
    """
    Collator that returns:
      - input_ids: LongTensor(batch, seq_len)
      - labels: LongTensor(batch, seq_len) (same as input_ids)
      - attention_mask: FloatTensor(batch, seq_len) (ones, since blocks are full)
    """
    def __init__(self, tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __call__(self, batch):
        # batch: list of {"input_ids": [...]}
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
        labels = input_ids.clone()  # causal LM uses labels = input_ids (model shifts internally)
        return {
            "input_ids": input_ids,
            "labels": labels
        }


def get_dataloaders(
    ds: DatasetDict,
    tokenizer,
    train_batch_size: int = cfg.batch_size,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 2,
    pin_memory: bool = True
):
    """
    Build PyTorch dataloaders from prepared datasets.
    Expects ds to be a DatasetDict with 'train' and 'test', each containing fixed-length input_ids lists.
    """
    if eval_batch_size is None:
        eval_batch_size = max(1, train_batch_size // 2)

    collator = DataCollatorForCausalLM(tokenizer, cfg.block_size)

    train_loader = DataLoader(
        ds["train"],
        batch_size=train_batch_size,
        shuffle=True,  # shuffling blocks is fine here
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    eval_loader = DataLoader(
        ds["test"],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory
    )

    return train_loader, eval_loader


# ---------------------------------------------------------
# Small debugging / validation utilities
# ---------------------------------------------------------
def validate_dataset_blocks(ds: Dataset, n_samples: int = 5):
    """
    Print some simple checks and examples from a Dataset of blocks (each row contains `input_ids` list of length block_size).
    """
    print("=== Dataset validation ===")
    print(f"Number of examples (blocks): {len(ds)}")
    if len(ds) == 0:
        print("WARNING: dataset is empty")
        return

    # sample a few blocks
    import random
    for i in random.sample(range(len(ds)), min(n_samples, len(ds))):
        block = ds[i]["input_ids"]
        print(f"- Block #{i}: length={len(block)} first_tokens={block[:8]} last_tokens={block[-8:]}")
    print("=== End validation ===\n")


# ---------------------------------------------------------
# Example usage (not executed on import)
# ---------------------------------------------------------
if __name__ == "__main__":
    # quick local test
    ds_prepared, tokenizer = load_and_prepare_dataset()
    print(ds_prepared)
    validate_dataset_blocks(ds_prepared["train"], n_samples=3)
    train_loader, eval_loader = get_dataloaders(ds_prepared, tokenizer)
    batch = next(iter(train_loader))
    print("Sample batch shapes:", {k: v.shape for k, v in batch.items()})
