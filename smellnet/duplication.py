"""Cross-function duplication detection.

By definition duplicated logic requires comparing more than one function, so
it cannot come out of the per-function neural head. This module fingerprints
normalized token streams (identifiers, numbers and strings collapsed) and
flags near-identical bodies via shingle Jaccard similarity.
"""
from __future__ import annotations

from .tokenizer import PAD_ID, VOCAB, encode


def normalized_tokens(src: str) -> list[str]:
    toks = []
    for i in encode(src):
        if i == PAD_ID:
            continue
        if i == VOCAB["<name>"]:
            toks.append("N")
        elif i == VOCAB["<num>"]:
            toks.append("#")
        elif i == VOCAB["<str>"]:
            toks.append("$")
        else:
            toks.append(_id_to_sym[i])
    return toks


def _shingles(tokens: list[str], k: int = 8) -> set:
    if len(tokens) < k:
        return {tuple(tokens)}
    return {tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def _jaccard(a: set, b: set) -> float:
    inter, union = len(a & b), len(a | b)
    return inter / union if union else 0.0


def find_duplicates(sources: list[str], threshold: float = 0.82):
    """Return [(i, j, similarity)] for functions whose bodies are near-duplicates."""
    sets = [_shingles(normalized_tokens(s)) for s in sources]
    n = len(sets)
    if n < 2:
        return []
    # bucket by (length, first 8 tokens) to avoid O(n^2) on big files
    buckets: dict[tuple, list[int]] = {}
    for i, s in enumerate(sets):
        key = None
        for cand in s:
            key = (len(cand), cand[:4])
            break
        if key is None:
            key = (0, ())
        buckets.setdefault(key, []).append(i)
    pairs: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for group in buckets.values():
        if len(group) < 2:
            continue
        for ai in range(len(group)):
            for bi in range(ai + 1, len(group)):
                a, b = group[ai], group[bi]
                if (a, b) in seen:
                    continue
                sim = _jaccard(sets[a], sets[b])
                if sim >= threshold:
                    seen.add((a, b))
                    pairs.append((a, b, sim))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs


_id_to_sym = {v: k for k, v in VOCAB.items()}
