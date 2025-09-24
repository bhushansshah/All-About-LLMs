import json
import os
from tokenizer.tokenizer import SentencePieceTokenizer
from tokenizer.build_tokenizer import train_sentencepiece
from data.dataloader import create_wiki_dataloader
from model.bert import BertForPretraining
from training.pretraining import pretrain_loop

def main():
    with open("configs/config_bert_small.json") as f:
        config = json.load(f)

    # 1) Train tokenizer (if model not present)
    sp_model = "spm.model"
    if not os.path.exists(sp_model):
        print("Training SentencePiece tokenizer on 25% of Wikipedia...")
        train_sentencepiece(out_prefix="spm", vocab_size=config["vocab_size"], subset_split="train[:25%]")
    else:
        print("Found existing tokenizer:", sp_model)

    tokenizer = SentencePieceTokenizer(sp_model)

    # 2) Create dataloader using split=25%
    train_loader = create_wiki_dataloader(
        tokenizer=tokenizer,
        split="train[:25%]",
        batch_size=config["batch_size"],
        max_seq_length=config["max_seq_length"],
        mask_prob=config["mask_prob"],
        nsp_prob=config["nsp_prob"],
        shuffle=True
    )

    # 3) Create model
    model = BertForPretraining(config)

    # 4) Pretrain
    pretrain_loop(model, train_loader, config)

if __name__ == "__main__":
    main()
