"""Tokenize Python source into a small fixed vocabulary of token ids.

The vocabulary is *structural*: keywords, operators and structural tokens are
distinct ids; identifiers, numbers and string literals collapse to generic ids
so the tokenizer works on any code with no learned vocabulary. Fine-grained
number / naming signals are carried by the AST feature vector instead.
"""
from __future__ import annotations

import io
import textwrap
import token
import tokenize

KEYWORDS = [
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
    "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield",
]

OPS = [
    "=", "+", "-", "*", "/", "//", "%", "**", "==", "!=", "<", ">", "<=",
    ">=", "(", ")", "[", "]", "{", "}", "^", "&", "|", "~", ":", ",", ".",
    "@", "+=", "-=", "*=", "/=", "->", ":", ";",
]

_SPECIALS = ["<pad>", "<unk>", "<num>", "<str>", "<name>", "<comment>",
             "<nline>", "<indent>", "<dedent>", "<op>", "<other>"]

PAD_ID = 0


def build_vocab() -> dict[str, int]:
    vocab: dict[str, int] = {}
    nxt = [0]

    def add(sym: str) -> int:
        if sym not in vocab:
            vocab[sym] = nxt[0]
            nxt[0] += 1
        return vocab[sym]

    for s in _SPECIALS:
        add(s)
    for kw in KEYWORDS:
        add(kw)
    for op in OPS:
        add(op)
    return vocab


VOCAB = build_vocab()
VOCAB_SIZE = len(VOCAB)
NAME_ID = VOCAB["<name>"]
NUM_ID = VOCAB["<num>"]
STR_ID = VOCAB["<str>"]
UNK_ID = VOCAB["<unk>"]

# Maximum token sequence fed to the network (train and scan agree on this).
# Longness is carried explicitly by the numeric features, so capping the
# sequence keeps training tractable without losing the signal.
MAX_TOKENS = 320


def encode(src: str, max_len: int = 768) -> list[int]:
    """Encode a Python source snippet into a list of token ids (<= max_len)."""
    if src is None:
        return [PAD_ID]
    dedented = textwrap.dedent(src)
    ids: list[int] = []
    try:
        toks = tokenize.generate_tokens(io.StringIO(dedented).readline)
        for t in toks:
            ttype, tstr = t.type, t.string
            if ttype == token.ENDMARKER:
                break
            if ttype in (tokenize.ENCODING,):
                continue
            if ttype == token.NAME:
                ids.append(VOCAB.get(tstr, NAME_ID))
            elif ttype == token.NUMBER:
                ids.append(NUM_ID)
            elif ttype == token.STRING:
                ids.append(STR_ID)
            elif ttype == tokenize.COMMENT:
                ids.append(VOCAB["<comment>"])
            elif ttype in (tokenize.NL, tokenize.NEWLINE):
                ids.append(VOCAB["<nline>"])
            elif ttype == tokenize.INDENT:
                ids.append(VOCAB["<indent>"])
            elif ttype == tokenize.DEDENT:
                ids.append(VOCAB["<dedent>"])
            elif ttype == token.OP:
                ids.append(VOCAB.get(tstr, VOCAB["<op>"]))
            else:
                ids.append(UNK_ID)
            if len(ids) >= max_len:
                break
    except Exception:
        # If tokenization fails for any reason fall back to a bare marker.
        return [PAD_ID, UNK_ID] if len(ids) == 0 else ids
    return ids[:max_len] if ids else [PAD_ID]
