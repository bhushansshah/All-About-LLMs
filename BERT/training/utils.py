import torch
import os

def save_checkpoint(model, optimizer, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.optimizer.state_dict() if hasattr(optimizer, "optimizer") else optimizer.state_dict(),
        "step": step
    }, path)

def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    try:
        optimizer_state = ckpt.get("optimizer_state_dict", None)
        if optimizer_state:
            if hasattr(optimizer, "optimizer"):
                optimizer.optimizer.load_state_dict(optimizer_state)
            else:
                optimizer.load_state_dict(optimizer_state)
    except Exception as e:
        print("Failed to load optimizer state:", e)
    return ckpt.get("step", 0)
