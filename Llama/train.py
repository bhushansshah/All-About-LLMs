# train.py
import os
import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from config import Config
from dataset import load_and_prepare_dataset, get_dataloaders, get_tokenizer
from model import MiniLLaMA
from utils import save_checkpoint, evaluate, load_checkpoint
from tqdm import tqdm

cfg = Config()

def build_scheduler(optimizer, total_steps, warmup_steps, final_lr_ratio):
    def lr_lambda(step):
        # step passed in is integer number of scheduler steps (0-based)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay from 1.0 -> final_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_ratio + (1.0 - final_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)

def train(resume_from: str = None):
    # Optional: speedups / determinism
    torch.backends.cudnn.benchmark = True

    # Load tokenizer and dataset
    tokenizer = get_tokenizer()
    ds, tokenizer = load_and_prepare_dataset(cfg.data_dir, tokenizer, cfg.block_size)
    train_loader, test_loader = get_dataloaders(ds, tokenizer, cfg.batch_size)

    # Instantiate model and move to device
    vocab_size = tokenizer.vocab_size
    model = MiniLLaMA(
        vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        block_size=cfg.block_size,
        rotary_dim=cfg.rotary_dim
    ).to(cfg.device)

    # ---- Print model architecture ----
    print("===== Model Architecture =====")
    print(model)

    # ---- Count trainable parameters ----
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    # Optimizer & scheduler
    optim = AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    total_steps = cfg.max_steps
    scheduler = build_scheduler(optim, total_steps, cfg.warmup_steps, cfg.final_lr_ratio)

    # Optionally resume
    start_step = 0
    if resume_from is not None and os.path.isfile(resume_from):
        ckpt_step = load_checkpoint(resume_from, model, optim, scheduler)  # implement load_checkpoint if you want resume
        start_step = ckpt_step

    # Training loop
    global_step = start_step
    model.train()

    # Optional: automatic mixed precision
    use_amp = getattr(cfg, "use_amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    training_losses = []
    try:
        for epoch in range(999999):
            loop = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
            for batch_idx, batch in enumerate(loop):
                optim.zero_grad()

                # Move input_ids to device
                input_ids = batch["input_ids"].to(cfg.device)  # (B, T)

                # ---- Build next-token labels ----
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = -100  # ignore last token

                # ---- Sanity checks ----
                assert labels.min() >= -100, f"Labels min {labels.min()} < -100"
                assert labels.max() < vocab_size, f"Labels max {labels.max()} >= vocab_size"

                if global_step % 100 == 0 and batch_idx == 0:
                    print("Sample input_ids:", input_ids[0, :10])
                    print("Sample labels   :", labels[0, :10])
                    print("Vocab size      :", vocab_size)

                # ---- Forward / backward with optional AMP ----
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(input_ids)  # (B, T, V)
                    B, T, V = logits.shape
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, V), labels.view(-1), ignore_index=-100
                    )

                # ---- Backward and gradient step ----
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
                scheduler.step()

                global_step += 1

                # ---- Logging ----
                if global_step % 10 == 0:
                    lr = optim.param_groups[0]['lr']
                    training_losses.append(loss.item())
                    print(f"[step {global_step}] loss = {loss.item():.4f} lr={lr:.6g}")

                # ---- Evaluation ----
                if global_step % cfg.eval_every == 0:
                    eval_metrics = evaluate(model, test_loader, cfg.device)
                    print(f"[eval @ step {global_step}] loss={eval_metrics['loss']:.4f} ppl={eval_metrics['ppl']:.3f}")

                # ---- Checkpoint ----
                if global_step % cfg.save_every == 0:
                    save_checkpoint(model, optim, scheduler, training_losses, global_step, ckpt_dir=cfg.checkpoint_dir)

                if global_step >= cfg.max_steps:
                    print("Reached max steps.")
                    save_checkpoint(model, optim, scheduler, training_losses, global_step, ckpt_dir=cfg.checkpoint_dir)
                    return
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        save_checkpoint(model, optim, scheduler, training_losses, global_step, ckpt_dir=cfg.checkpoint_dir)
    except Exception as e:
        print(f"Exception during training: {e!r}. Saving checkpoint at step {global_step}.")
        save_checkpoint(model, optim, scheduler, training_losses, global_step, ckpt_dir=cfg.checkpoint_dir)
        raise


if __name__ == "__main__":
    train()
