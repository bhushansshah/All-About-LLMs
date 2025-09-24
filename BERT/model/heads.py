import torch.nn as nn
import torch

class MLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(config["hidden_size"], config["hidden_size"]),
            nn.GELU(),
            nn.LayerNorm(config["hidden_size"], eps=1e-12)
        )
        # projection to vocab
        self.decoder = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        self.bias = nn.Parameter(torch.zeros(config["vocab_size"]))
        self.decoder.bias = self.bias

    def forward(self, sequence_output):
        x = self.transform(sequence_output)
        logits = self.decoder(x)
        return logits

class NSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config["hidden_size"], 2)

    def forward(self, pooled_output):
        return self.seq_relationship(pooled_output)
