import torch.nn as nn
from .attention import MultiHeadSelfAttention

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.attention_norm = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.ffn = nn.Sequential(
            nn.Linear(config["hidden_size"], config["intermediate_size"]),
            nn.GELU(),
            nn.Linear(config["intermediate_size"], config["hidden_size"])
        )
        self.ffn_norm = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, attention_mask=None):
        attn_out = self.attention(x, attention_mask=attention_mask)
        x = self.attention_norm(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))
        return x

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        for _ in range(config["num_hidden_layers"]):
            layers.append(BertLayer(config))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x
