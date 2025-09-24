import sentencepiece as spm
import os
from datasets import load_dataset
import tempfile

def train_sentencepiece(out_prefix="spm", vocab_size=30000, subset_split="train[:25%]"):
    # load 25% of en wiki as raw text
    ds = load_dataset("wikipedia", "20220301.en", split=subset_split)
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
    print("Writing raw text to", tmp.name)
    for i, ex in enumerate(ds):
        text = ex.get("text", "")
        if not text:
            continue
        tmp.write(text.replace("\n", " ") + "\n")
        if (i+1) % 10000 == 0:
            print("wrote", i+1, "articles")
    tmp.close()

    model_prefix = out_prefix
    spm.SentencePieceTrainer.Train(
        f"--input={tmp.name} --model_prefix={model_prefix} --vocab_size={vocab_size} "
        "--model_type=unigram --character_coverage=1.0 --pad_id=0 --unk_id=1 "
        "--bos_id=2 --eos_id=3"
    )
    print("Trained SentencePiece model:", model_prefix + ".model")
    os.unlink(tmp.name)
    return model_prefix + ".model", model_prefix + ".vocab"

if __name__ == "__main__":
    train_sentencepiece()
