# generate.py
import argparse
import torch
from transformers import AutoTokenizer
from model import MiniLLaMA
from utils import load_checkpoint
from config import Config
import torch.nn.functional as F
import os

cfg = Config()

# ------------------------
# Filtering helpers
# ------------------------
def top_k_top_p_filtering(logits, top_k=None, top_p=None):
    """Filter logits using top-k and/or nucleus (top-p) sampling."""
    logits = logits.clone()

    # Top-k filtering
    if top_k is not None and top_k > 0:
        top_values, _ = torch.topk(logits, top_k)
        min_allowed = top_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_allowed, torch.full_like(logits, -float("Inf")), logits)

    # Top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        for b in range(logits.size(0)):
            indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
            logits[b, indices_to_remove] = -float("Inf")

    return logits


# ------------------------
# Sampling function
# ------------------------
@torch.no_grad()
def sample(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    top_p=None,
    device="cpu"
):
    """
    Generate text autoregressively until EOS token or max_new_tokens reached.
    """
    model.eval()

    # Add BOS token explicitly
    if tokenizer.bos_token:
        prompt = tokenizer.bos_token + prompt

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        context = generated[:, -model.block_size:]

        logits = model(context)
        logits = logits[:, -1, :] / temperature  # (B, vocab_size)

        # Apply top-k and/or top-p filtering
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        next_id = next_token.item()

        generated = torch.cat((generated, next_token), dim=1)

        if eos_token_id is not None and next_id == eos_token_id:
            break

    output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    return output_text


# ------------------------
# Main script
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate a poem using MiniLLaMA checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    print(f"[Tokenizer] Loaded from {args.tokenizer_dir}")

    # Load model
    vocab_size = tokenizer.vocab_size
    model = MiniLLaMA(
        vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        block_size=cfg.block_size,
        rotary_dim=cfg.rotary_dim,
    ).to(args.device)

    # Load checkpoint
    print(f"[Info] Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        print(f"[Warning] Missing keys: {missing}")
        print(f"[Warning] Unexpected keys: {unexpected}")
    else:
        print("[Info] Model weights loaded successfully.")

    # Generate
    print(f"\n[Prompt] {args.prompt}\n")
    output = sample(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )

    print("\n[Generated Poem]")
    print("=" * 60)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()
