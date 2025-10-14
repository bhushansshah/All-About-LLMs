# finetune.py
import os
import torch
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from config import Config
from model import MiniLLaMA
from dataset_finetuning import get_tokenizer, load_poem_dataset, get_dataloaders
from utils import save_checkpoint, evaluate, load_checkpoint

cfg = Config()
cfg.warmup_steps = 700
cfg.batch_size = 8
cfg.lr = 3e-5
cfg.checkpoint_dir = "./finetune_checkpoints"

os.makedirs(cfg.checkpoint_dir, exist_ok=True)
def build_scheduler(optimizer, total_steps, warmup_steps, final_lr_ratio):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_ratio + (1.0 - final_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)

def finetune(checkpoint_path, data_dir):
    device = cfg.device
    tokenizer = get_tokenizer()
    ds = load_poem_dataset(data_dir, tokenizer, block_size=cfg.block_size)
    train_loader, test_loader = get_dataloaders(ds, tokenizer, batch_size=cfg.batch_size)

    vocab_size = tokenizer.vocab_size
    model = MiniLLaMA(vocab_size).to(device)

    # Load pretrained checkpoint
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"[Fine-tune] Loading pretrained weights from {checkpoint_path}")
        load_checkpoint(checkpoint_path, model, cfg)
    else:
        print("[Fine-tune] No checkpoint provided. Starting from scratch!")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.max_steps, cfg.warmup_steps, cfg.final_lr_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    global_step = 0
    training_losses = []
    try:
        for epoch in range(5):  # run for multiple passes over small poetry data
            model.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch}")
            for batch in loop:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    logits = model(input_ids)
                    B, T, V = logits.shape
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, V), labels.view(-1), ignore_index=-100
                    )

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                global_step += 1
                loop.set_postfix(loss=loss.item())

                if global_step % 10 == 0:
                    lr = optimizer.param_groups[0]['lr']
                    training_losses.append(loss.item())
                    print(f"[step {global_step}] loss = {loss.item():.4f} lr={lr:.6g}")

            eval_metrics = evaluate(model, test_loader, device)
            print(f"[Eval @ Epoch {epoch}] loss={eval_metrics['loss']:.4f} ppl={eval_metrics['ppl']:.3f}")
            save_checkpoint(model, optimizer, scheduler, training_losses, global_step, ckpt_dir=cfg.checkpoint_dir)
        save_checkpoint(model, optimizer, scheduler, training_losses, global_step, ckpt_dir=cfg.checkpoint_dir)
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        save_checkpoint(model, optimizer, scheduler, training_losses, global_step, ckpt_dir=cfg.checkpoint_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to pretrained checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to local poem dataset")
    args = parser.parse_args()

    finetune(args.checkpoint, args.data_dir)
