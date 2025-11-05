import os
import re
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import Config
from dataset import get_tokenizer, load_and_prepare_dataset, get_dataloaders
from model import MiniLLaMA
from utils import evaluate

cfg = Config()


def extract_step(filename):
    """Extract step number from filename like ckpt-step-1000.pt."""
    m = re.search(r"ckpt-step-(\d+)\.pt", filename)
    return int(m.group(1)) if m else -1


@torch.no_grad()
def compute_validation_metrics(ckpt_dir, data_dir, device=cfg.device):
    """Evaluate all checkpoints and compute validation loss + perplexity."""
    tokenizer = get_tokenizer()
    ds, _ = load_and_prepare_dataset(data_dir, tokenizer, cfg.block_size)
    _, val_loader = get_dataloaders(ds, tokenizer, batch_size=cfg.batch_size)

    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt") and "ckpt-step" in f]
    ckpts = sorted(ckpts, key=extract_step)
    #take every alternate checkpoint
    ckpts = ckpts[::2]
    results = []
    for ckpt_file in tqdm(ckpts, desc="Evaluating checkpoints"):
        step = extract_step(ckpt_file)
        path = os.path.join(ckpt_dir, ckpt_file)

        vocab_size = tokenizer.vocab_size
        model = MiniLLaMA(
            vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            block_size=cfg.block_size,
            rotary_dim=cfg.rotary_dim,
        ).to(device)

        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval()

        metrics = evaluate(model, val_loader, device)
        results.append((step, metrics["loss"], metrics["ppl"]))
        print(f"[Step {step}] Val Loss={metrics['loss']:.4f}, PPL={metrics['ppl']:.2f}")

    return results


def plot_results(results, save_path=None):
    """Plot validation loss and perplexity vs training step."""
    steps, losses, ppls = zip(*results)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_loss = "tab:blue"
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Validation Loss", color=color_loss)
    ax1.plot(steps, losses, marker="o", color=color_loss, label="Validation Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Create second y-axis for perplexity
    ax2 = ax1.twinx()
    color_ppl = "tab:red"
    ax2.set_ylabel("Perplexity", color=color_ppl)
    ax2.plot(steps, ppls, marker="s", color=color_ppl, label="Perplexity")
    ax2.tick_params(axis="y", labelcolor=color_ppl)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Validation Loss and Perplexity vs Training Step")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[Plot saved] {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot validation loss and perplexity vs checkpoint step")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to directory with checkpoints")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save figure")
    args = parser.parse_args()

    results = compute_validation_metrics(args.ckpt_dir, args.data_dir)
    plot_results(results, save_path=args.save_path)
