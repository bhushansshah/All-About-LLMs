# generate.py
import argparse
import torch
from transformers import AutoTokenizer
from model import MiniLLaMA
from utils import load_checkpoint
from config import Config
import os
import torch.nn.functional as F

cfg = Config()

# ------------------------
# Sampling function
# ------------------------
@torch.no_grad()
def sample(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=None, device="cpu"):
    """
    Generate text autoregressively from the model.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Crop to block_size if context too long
        context = generated[:, -model.block_size:]

        # Forward pass
        logits = model(context)
        logits = logits[:, -1, :] / temperature  # (B, vocab_size)

        # Optionally restrict to top_k tokens
        if top_k is not None:
            top_values, _ = torch.topk(logits, top_k)
            min_allowed = top_values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_allowed, torch.full_like(logits, -float("Inf")), logits)

        probs = F.softmax(logits, dim=-1)

        # Multinomial sampling
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # Append new token
        generated = torch.cat((generated, next_token), dim=1)

    # Decode tokens to text
    output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    return output_text


# ------------------------
# Main script
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate text using MiniLLaMA checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="Path to tokenizer directory")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Optional top-k sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device: cuda or cpu")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)

    # Load model
    vocab_size = tokenizer.vocab_size
    model = MiniLLaMA(vocab_size, d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers, block_size=cfg.block_size, rotary_dim=cfg.rotary_dim)
    model = model.to(args.device)

    # Load checkpoint
    print(f"[Info] Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    # Generate
    print(f"[Prompt] {args.prompt}")
    output = sample(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )

    print("\n[Generated Text]")
    print("=" * 60)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()
