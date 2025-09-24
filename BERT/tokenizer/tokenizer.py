import sentencepiece as spm

class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def encode(self, text, add_special_tokens=True, max_length=None):
        ids = self.sp.EncodeAsIds(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids):
        return self.sp.DecodeIds(ids)

    @property
    def vocab_size(self):
        return self.sp.GetPieceSize()
