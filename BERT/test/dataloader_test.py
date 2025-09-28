from tokenizer.tokenizer import SentencePieceTokenizer

from data.dataloader import create_wiki_dataloader  # import your function

def test_dataloader():
    # ✅ load a tokenizer (use your trained tokenizer instead of HF if available)
    tokenizer = SentencePieceTokenizer(model_path="./data/tokenizer/spm.model")

    # ✅ create small dataloader
    loader = create_wiki_dataloader(
        tokenizer=tokenizer,
        dataset_path="./data/wikipedia_15percent",
        shuffle=True
    )

    # ✅ take one batch
    batch = next(iter(loader))

    print("Keys in batch:", batch.keys())
    for k, v in batch.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        if k == "input_ids":
            print("Decoded examples:")
            for row in v[:2]:
                print(tokenizer.decode(row.tolist()))

    print("\nCheck mlm_labels (masked tokens):")
    for i in range(batch["mlm_labels"].size(0)):
        masked = batch["mlm_labels"][i]
        inp = batch["input_ids"][i]
        for j, lbl in enumerate(masked):
            if lbl.item() != -100:
                print(f"  Example {i}, position {j}: label={tokenizer.decode([lbl.item()])}, input={tokenizer.decode([inp[j].item()])}")

    print("\nNSP labels:", batch["nsp_label"].tolist())

if __name__ == "__main__":
    test_dataloader()
