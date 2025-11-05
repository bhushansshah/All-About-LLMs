# utils.py
import torch
import os
from pathlib import Path
import math
from tqdm import tqdm

def save_checkpoint(model, optimizer, scheduler, training_losses, step, ckpt_dir="./checkpoints"):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    fn = os.path.join(ckpt_dir, f"ckpt-step-{step}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict() if scheduler is not None else None,
        "training_losses": training_losses,
        "step": step
    }, fn)
    print(f"[checkpoint] saved {fn}")

def load_checkpoint(path, model, config, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=config.device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and ckpt.get("opt_state"):
        optimizer.load_state_dict(ckpt["opt_state"])
    if scheduler and ckpt.get("sched_state"):
        scheduler.load_state_dict(ckpt["sched_state"])
    return ckpt.get("step", 0)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_f = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    loop = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        logits = model(input_ids)
        B, T, V = logits.shape
        loss = loss_f(logits.view(-1, V), labels.view(-1))
        total_loss += loss.item()
        total_tokens += (labels != -100).sum().item()
        loop.set_postfix(loss=total_loss / total_tokens if total_tokens > 0 else float('inf'))
    ppl = math.exp(total_loss / total_tokens)
    return {"loss": total_loss / total_tokens, "ppl": ppl}
