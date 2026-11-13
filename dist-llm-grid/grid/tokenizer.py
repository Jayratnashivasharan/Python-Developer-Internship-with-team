"""Tokenizers. The byte tokenizer needs no files; HF tokenizers are optional."""
from __future__ import annotations


class ByteTokenizer:
    """256 byte tokens + <bos>=256, <eos>=257, <pad>=258 (matches the tiny demo model)."""

    bos_id, eos_id, pad_id = 256, 257, 258
    vocab_size = 259

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        return ([self.bos_id] if add_bos else []) + ids

    def decode_piece(self, token_id: int) -> bytes:
        return bytes([token_id]) if token_id < 256 else b""


class StreamDecoder:
    """Turns a token stream into text pieces, holding back incomplete UTF-8 sequences."""

    def __init__(self, tok):
        import codecs

        self.tok, self.ids = tok, []
        self.inc = codecs.getincrementaldecoder("utf-8")(errors="replace") if isinstance(tok, ByteTokenizer) else None

    def push(self, token_id: int) -> str:
        if self.inc is not None:
            return self.inc.decode(self.tok.decode_piece(token_id))
        prev = self.tok.decode(self.ids) if self.ids else ""
        self.ids.append(token_id)
        cur = self.tok.decode(self.ids)
        return cur[len(prev):] if not cur.endswith("\ufffd") else ""


class HFTokenizer:
    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer  # optional dependency

        self.t = AutoTokenizer.from_pretrained(name_or_path)
        self.bos_id = self.t.bos_token_id
        self.eos_id = self.t.eos_token_id
        self.vocab_size = len(self.t)

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        return self.t.encode(text, add_special_tokens=add_bos)

    def decode(self, ids: list[int]) -> str:
        return self.t.decode(ids, skip_special_tokens=True)


def load_tokenizer(spec: str):
    return ByteTokenizer() if spec == "byte" else HFTokenizer(spec)
