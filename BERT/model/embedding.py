import torch
import torch.nn as nn

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"], padding_idx=0)
        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], config["hidden_size"])

        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        words = self.word_embeddings(input_ids)
        positions = self.position_embeddings(position_ids)
        token_types = self.token_type_embeddings(token_type_ids)
        x = words + positions + token_types
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x
