# train.py
import os
import math
import time
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import argparse
import wandb

from config import Config
from dataset import get_tokenizer, load_and_prepare_dataset, get_dataloaders
from model import MiniLLaMA
import random

cfg = Config()


# -----------------------
# Utilities
# -----------------------
def build_scheduler(optimizer, total_steps, warmup_steps, final_lr_ratio):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_ratio + (1.0 - final_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(state: dict, ckpt_dir: str, step: int):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt-step-{step}.pt")
    torch.save(state, path)
    print(f"[ckpt] saved {path}")
    # also save "latest" pointer
    latest = os.path.join(ckpt_dir, "latest.json")
    with open(latest, "w") as f:
        json.dump({"path": path, "step": step}, f)
    return path


def load_checkpoint_if_exists(path: str, model, optimizer=None, scheduler=None, device="cpu", reset_scheduler=False):
    if path is None:
        return 0
    if not os.path.isfile(path):
        print(f"[ckpt] Not found: {path}")
        return 0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
        if reset_scheduler:
            print("[ckpt] Resetting the learning rate to the initial value.")
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.lr
    if reset_scheduler:
        print("[ckpt] Resetting optimizer and scheduler states as requested.")
        scheduler = build_scheduler(optimizer, cfg.max_steps, cfg.warmup_steps, cfg.final_lr_ratio)
    elif scheduler is not None and "sched_state" in ckpt:
        scheduler.load_state_dict(ckpt["sched_state"])
    step = ckpt.get("global_step", 0)
    print(f"[ckpt] Loaded checkpoint from {path} (step={step})")
    return step


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=500, sample_prob=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    seen_batches = 0
    used_batches = 0

    # If user doesn't set sample_prob, estimate it so expected selected batches ≈ max_batches
    if sample_prob is None:
        sample_prob = max_batches / len(dataloader)

    for batch in dataloader:
        seen_batches += 1
        if random.random() > sample_prob:
            continue  # skip this batch

        # ---- evaluate this batch ----
        input_ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)

        logits = model(input_ids)   # (B, T, V)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        if attn is not None:
            shift_attn = attn[:, 1:]
            shift_labels = shift_labels.masked_fill(shift_attn == 0, -100)
            n_real = int(shift_attn.sum().item())
        else:
            n_real = shift_labels.numel()

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += n_real
        used_batches += 1

        if used_batches >= max_batches:
            break

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    model.train()
    return {
        "loss": avg_loss,
        "ppl": ppl,
        "batches": used_batches,
        "tokens": total_tokens,
    }


# -----------------------
# Main training function
# -----------------------
def train(resume_ckpt: str = None, wandb_id: str = None, reset_scheduler: bool = False):
    # device
    device = cfg.device
    print(f"[run] device = {device}")

    # -----------------------
    # tokenizer & dataset
    # -----------------------
    tokenizer = get_tokenizer()
    ds, tokenizer = load_and_prepare_dataset(cfg.data_dir, tokenizer, cfg.block_size)
    train_loader, eval_loader = get_dataloaders(ds, tokenizer, train_batch_size=cfg.batch_size)

    # -----------------------
    # model
    # -----------------------
    vocab_size = tokenizer.vocab_size
    model = MiniLLaMA(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        block_size=cfg.block_size,
        rotary_dim=cfg.rotary_dim
    ).to(device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params: {num_params:,}")

    # -----------------------
    # optimizer & scheduler (paper-like)
    # -----------------------
    optimizer = AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.max_steps, cfg.warmup_steps, cfg.final_lr_ratio)

    # optionally resume from checkpoint
    start_step = 0
    if resume_ckpt is not None:
        start_step = load_checkpoint_if_exists(resume_ckpt, model, optimizer, scheduler, device=device, reset_scheduler=reset_scheduler)

    # -----------------------
    # wandb init
    # -----------------------
    wandb_run = None
    if os.getenv("WANDB_API_KEY") is not None:
        if wandb_id:
            wandb_run = wandb.init(project="minillama-pretrain", config=vars(cfg), id=wandb_id, resume="must")
        else:
            wandb_run = wandb.init(project="minillama-pretrain", config=vars(cfg), reinit=True)
        print("[wandb] initialized")
    else:
        print("[wandb] WANDB_API_KEY not set, skipping wandb logging.")

    # -----------------------
    # training loop
    # -----------------------

    model.train()
    global_step = start_step
    running_loss = 0.0
    running_tokens = 0

    try:
        for epoch in range(999999):
            loop = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
            for batch in loop:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)    # (B, T)

                # Forward (with AMP)
                with torch.amp.autocast("mps", dtype=torch.bfloat16):
                    logits = model(input_ids)  # (B, T, V)

                    # causal shift: logits[:, :-1] should predict input_ids[:, 1:]
                    shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
                    shift_labels = input_ids[:, 1:].contiguous()    # (B, T-1)
                    
                    B, Tm, V = shift_logits.shape
                    loss = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100)

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()

                global_step += 1
                running_loss += loss.item() * (shift_labels != -100).sum().item()  # accumulate token-sum
                running_tokens += (B * Tm)

                # Logging to console & wandb
                if global_step % 10 == 0:
                    avg_token_loss = running_loss / max(1, running_tokens)
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"[step {global_step}] step_loss={loss.item():.6f} token_loss={avg_token_loss:.6f} lr={lr:.6g}")
                    if wandb_run:
                        wandb_run.log({"train/token_loss": avg_token_loss, "train/lr": lr, "global_step": global_step, 'train/step_loss': loss.item()})

                    running_loss = 0.0
                    running_tokens = 0

                # Periodic eval
                if global_step % cfg.eval_every == 0:
                    eval_stats = evaluate(model, eval_loader, device)
                    print(f"[eval] step={global_step} loss={eval_stats['loss']:.6f} ppl={eval_stats['ppl']:.3f}")
                    if wandb_run:
                        wandb_run.log({
                            "eval/loss": eval_stats["loss"],
                            "eval/ppl": eval_stats["ppl"],
                            "global_step": global_step
                        })

                # Periodic checkpoint
                if global_step % cfg.save_every == 0:
                    ckpt_state = {
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "sched_state": scheduler.state_dict(),
                        "global_step": global_step
                    }
                    save_checkpoint(ckpt_state, cfg.checkpoint_dir, global_step)

                # stop if reached max steps
                if global_step >= cfg.max_steps:
                    print("Reached max steps; finishing.")
                    ckpt_state = {
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "sched_state": scheduler.state_dict(),
                        "global_step": global_step
                    }
                    save_checkpoint(ckpt_state, cfg.checkpoint_dir, global_step)
                    return

    except KeyboardInterrupt:
        print("Interrupted — saving checkpoint.")
        ckpt_state = {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "global_step": global_step
        }
        save_checkpoint(ckpt_state, cfg.checkpoint_dir, global_step)
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    # optionally pass a resume checkpoint path via environment variable RESUME_CKPT
    # argparse can also be used here if desired

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to resume checkpoint")
    parser.add_argument("--wandb_id", type=str, default=None, help="Weights & Biases run ID to resume")
    parser.add_argument("--reset_scheduler", action="store_true", help="If set, do not load optimizer/scheduler state from checkpoint")
    args = parser.parse_args()
    train(args.resume_ckpt, args.wandb_id, args.reset_scheduler)