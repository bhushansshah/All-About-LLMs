import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth(values, window=50):
    """Apply moving average smoothing to reduce noise."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")

def plot_training_curve(ckpt_path, smooth_window=50, save_path=None):
    """Load training losses from checkpoint and plot them."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "training_losses" not in ckpt:
        raise KeyError(f"'training_losses' not found in checkpoint {ckpt_path}")

    losses = ckpt["training_losses"]
    step_interval = 10  # since you saved every 10 steps
    steps = [i * step_interval for i in range(len(losses))]

    # Plot raw loss
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, label="Raw Loss", alpha=0.4, color="gray")

    # Plot smoothed loss
    smoothed = smooth(losses, smooth_window)
    smooth_steps = steps[:len(smoothed)]
    plt.plot(smooth_steps, smoothed, label=f"Smoothed (window={smooth_window})", color="blue")

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve ({os.path.basename(ckpt_path)})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training losses from a checkpoint")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file (.pt)")
    parser.add_argument("--smooth_window", type=int, default=50, help="Smoothing window size")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save the plot as an image")

    args = parser.parse_args()
    plot_training_curve(args.ckpt_path, smooth_window=args.smooth_window, save_path=args.save_path)
