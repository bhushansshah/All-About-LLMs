import torch
import torch.nn as nn
from .embedding import BertEmbeddings
from .encoder import BertEncoder
from .heads import MLMHead, NSPHead

class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.pooler_act = nn.Tanh()

    def forward(self, input_ids, token_type_ids, attention_mask):
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output = self.encoder(embedding_output, attention_mask=attention_mask)
        # pooled output (take first token)
        pooled_output = self.pooler_act(self.pooler(encoder_output[:, 0]))
        return encoder_output, pooled_output

class BertForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.mlm = MLMHead(config)
        self.nsp = NSPHead(config)

    def forward(self, input_ids, token_type_ids, attention_mask):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        mlm_logits = self.mlm(sequence_output)
        nsp_logits = self.nsp(pooled_output)
        return mlm_logits, nsp_logits
