import torch
from tokenizer.tokenizer import SentencePieceTokenizer
from model.bert import BertForPretraining  # your custom BERT model
from training.utils import load_checkpoint  # your checkpoint loader
import json
import random
def make_input_batch(tokenizer, sent_a, sent_b=None, max_seq_length=128, device="cpu", mask_tokens=True, num_masks=3, nsp_label=None):
    """
    Convert one or two sentences into a proper BERT input batch.
    Optionally apply random masking (up to num_masks tokens).
    """
    # Encode sentences
    a_ids = tokenizer.encode(sent_a)
    b_ids = tokenizer.encode(sent_b) if sent_b else []

    # Truncate if too long
    while len(a_ids) + len(b_ids) + 3 > max_seq_length:
        if len(a_ids) > len(b_ids):
            a_ids.pop()
        elif b_ids:
            b_ids.pop()
        else:
            a_ids.pop()

    # Build input_ids
    input_ids = [tokenizer.bos_token_id] + a_ids + [tokenizer.eos_token_id]
    token_type_ids = [0] * len(input_ids)

    if b_ids:
        input_ids += b_ids + [tokenizer.eos_token_id]
        token_type_ids += [1] * (len(b_ids) + 1)

    # Attention mask
    attention_mask = [1] * len(input_ids)

    # Pad
    pad_len = max_seq_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        token_type_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    # === Random masking for MLM test ===
    mlm_labels = [-100] * len(input_ids)  # -100 ignored in CE loss
    if mask_tokens:
        valid_positions = [i for i, tid in enumerate(input_ids)
                           if tid not in (tokenizer.pad_token_id,
                                          tokenizer.bos_token_id,
                                          tokenizer.eos_token_id)]
        random.shuffle(valid_positions)
        for pos in valid_positions[:num_masks]:
            mlm_labels[pos] = input_ids[pos]
            input_ids[pos] = tokenizer.unk_token_id  # we used UNK as [MASK] substitute

    # Convert to batch of size 1
    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "token_type_ids": torch.tensor([token_type_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=device),
        "mlm_labels": torch.tensor([mlm_labels], dtype=torch.long, device=device),
        "nsp_label": torch.tensor([nsp_label if nsp_label is not None else 0], dtype=torch.long, device=device)
    }
    return batch


# ---- Load model + tokenizer ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("config/config_small_bert.json") as f:
    config = json.load(f)
tokenizer = SentencePieceTokenizer("data/tokenizer/spm.model")
model = BertForPretraining(config)
checkpoint = "new_checkpoints/bert_step_80000.pt"
training_step = load_checkpoint(model, None, checkpoint, 'cpu')
model = model.to(device)
model.eval()

# ---- Test MLM ----
def test_mlm(sentence):
    # Mask a token

    batch = make_input_batch(tokenizer, sentence, max_seq_length=config["max_seq_length"], device=device)
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        mlm_logits, _ = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    mask_indices = (batch["input_ids"][0] == tokenizer.unk_token_id).nonzero(as_tuple=True)[0]
    print("Input sentence:", sentence)
    print("Masked input:", tokenizer.decode(batch["input_ids"][0].tolist()))
    print("Predictions for masked positions:")
    for mask_index in mask_indices:
        mask_index = mask_index.item()
        predicted_id = mlm_logits[0, mask_index].argmax(dim=-1).item()
        predicted_token = tokenizer.decode([predicted_id])
        print(f"Mask at position {mask_index}: predicted '{predicted_token}'")

# ---- Test NSP ----
def test_nsp(sent_a, sent_b):
    batch = make_input_batch(tokenizer, sent_a, sent_b=sent_b, max_seq_length=config["max_seq_length"], device=device)
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        _, nsp_logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    probs = torch.softmax(nsp_logits, dim=-1)
    print("Input A:", sent_a)
    print("Input B:", sent_b)
    print("IsNext prob:", probs[0][0].item())
    print("NotNext prob:", probs[0][1].item())

if __name__ == "__main__":
    print("=== Testing MLM ===")
    test_mlm("The capital of France is Paris and it is known as the city of lights.")
    test_mlm("The largest planet in our solar system is Jupiter and it has many moons.")
    test_mlm("The chemical symbol for water is H2O and it is essential for life on Earth.")
    test_mlm("Albert Einstein developed the theory of relativity in the early 20th century.")
    test_mlm("The Great Wall of China is one of the most famous landmarks in the world.")

    print("\n=== Testing NSP ===")
    test_nsp("The sky is blue.", "Grass is green.")
    test_nsp("The sky is blue.", "I love pizza.")
    test_nsp("The capital of Japan is Tokyo.", "Mount Fuji is the highest mountain in Japan.")
    test_nsp("Python is a popular programming language.", "It is widely used for data science.")
    test_nsp("Cats are small domesticated animals.", "The Eiffel Tower is located in Paris.")