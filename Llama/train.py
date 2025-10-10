# train.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from configs import Config
from dataset import load_and_prepare_dataset, get_dataloaders, get_tokenizer
from model import MiniLLaMA
from utils import save_checkpoint, evaluate
import os

cfg = Config()

def build_scheduler(optimizer, total_steps, warmup_steps, final_lr_ratio):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay from 1.0 -> final_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_ratio + (1.0 - final_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)

import math
def train():
    # load tokenizer and dataset
    tokenizer = get_tokenizer()
    ds, tokenizer = load_and_prepare_dataset(cfg.data_dir, tokenizer, cfg.block_size)
    train_loader, test_loader = get_dataloaders(ds, tokenizer, cfg.batch_size)

    # instantiate model
    vocab_size = tokenizer.vocab_size
    model = MiniLLaMA(vocab_size, d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers, block_size=cfg.block_size, rotary_dim=cfg.rotary_dim)
    model = model.to(cfg.device)

    # optimizer & scheduler
    optim = AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    total_steps = cfg.max_steps
    scheduler = build_scheduler(optim, total_steps, cfg.warmup_steps, cfg.final_lr_ratio)

    # training loop
    global_step = 0
    model.train()
    for epoch in range(999999):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch["input_ids"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)
            logits = model(input_ids)
            B, T, V = logits.shape
            loss = torch.nn.functional.cross_entropy(logits.view(-1, V), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            scheduler.step()
            global_step += 1

            if global_step % 10 == 0:
                print(f"[step {global_step}] loss = {loss.item():.4f} lr={optim.param_groups[0]['lr']:.6g}")

            if global_step % cfg.eval_every == 0:
                eval_metrics = evaluate(model, test_loader, cfg.device)
                print(f"[eval @ step {global_step}] loss={eval_metrics['loss']:.4f} ppl={eval_metrics['ppl']:.3f}")

            if global_step % cfg.save_every == 0:
                save_checkpoint(model, optim, scheduler, global_step, ckpt_dir=cfg.checkpoint_dir)

            if global_step >= cfg.max_steps:
                print("Reached max steps.")
                save_checkpoint(model, optim, scheduler, global_step, ckpt_dir=cfg.checkpoint_dir)
                return

if __name__ == "__main__":
    train()
