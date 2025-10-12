# configs.py
from dataclasses import dataclass
from dotenv import load_dotenv
import os
load_dotenv()

@dataclass
class Config:
    # Data
    data_dir: str = "/Users/bhushanshah/Documents/All-About-LLMs/Llama/data/wikipedia_15percent"   # your ds.save_to_disk() path
    tokenizer_name_or_path: str = "/Users/bhushanshah/Documents/All-About-LLMs/Llama/tokenizer"  # example; replace with your BPE model
    block_size: int = 512   # context length
    train_test_split: float = 0.01  # fraction for test/validation

    # Model (mini LLaMA-ish)
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    ffn_multiplier: float = 8/3  # as in LLaMA they use 2/3 * 4d? (use 8/3 here)
    rotary_dim: int = 64  # head dim uses rotary on first rotary_dim dims

    # Training
    max_steps: int = 20000
    eval_every: int = 1000
    save_every: int = 1000
    batch_size: int = 4         # number of sequences per batch (increase if memory allows)
    tokens_per_batch: int = 4_096  # or compute batch as batch_size * block_size
    lr: float = 3e-4            # max LR (match small LLaMA example). Paper uses 3e-4 or 1.5e-4 depending on size. :contentReference[oaicite:2]{index=2}
    final_lr_ratio: float = 0.1  # final LR is 10% of max according to paper. :contentReference[oaicite:3]{index=3}
    weight_decay: float = 0.1   # from paper. :contentReference[oaicite:4]{index=4}
    betas: tuple = (0.9, 0.95)   # from paper. :contentReference[oaicite:5]{index=5}
    warmup_steps: int = 2000    # as in the paper. :contentReference[oaicite:6]{index=6}
    grad_clip: float = 1.0      # from paper. :contentReference[oaicite:7]{index=7}

    # Misc
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    checkpoint_dir: str = "./checkpoints"
    use_amp: bool = False  # use mixed precision training

class EnvironmentConfig:
    # Environment variables (like HF_TOKEN) can be loaded here if needed
    hf_token: str = os.getenv("HF_TOKEN")  # Set this to your Hugging Face token if required
