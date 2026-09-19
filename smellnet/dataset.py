"""PyTorch dataset + collate for packed, padded token sequences."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import PAD_ID, MAX_TOKENS, encode


class SmellDataset(Dataset):
    def __init__(self, samples, mean: np.ndarray, std: np.ndarray,
                 max_len: int = MAX_TOKENS):
        self.data = []
        for src, vec, feats in samples:
            ids = encode(src, max_len=max_len)
            if not ids:
                ids = [PAD_ID]
            f = np.asarray(feats, dtype=np.float32)
            f = (f - mean) / (std + 1e-8)
            self.data.append((np.asarray(ids, dtype=np.int64),
                              f,
                              np.asarray(vec, dtype=np.float32)))
        self.lengths = np.array([len(d[0]) for d in self.data], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i):
        ids, feats, labels = self.data[i]
        return (torch.from_numpy(ids),
                torch.from_numpy(feats),
                torch.from_numpy(labels))


class BucketBatchSampler:
    """Yields batches of similar-length samples so padding stays cheap."""

    def __init__(self, lengths, batch_size: int, shuffle: bool = True,
                 seed: int = 0, shard_factor: int = 16):
        lengths = np.asarray(lengths)
        n = len(lengths)
        order = np.argsort(lengths, kind="stable")
        shard_size = max(batch_size, batch_size * shard_factor)
        rng = np.random.RandomState(seed)
        shards = [order[i:i + shard_size] for i in range(0, n, shard_size)]
        if shuffle:
            rng.shuffle(shards)
        self._plan = []
        for shard in shards:
            if shuffle:
                rng.shuffle(shard)
            for i in range(0, len(shard), batch_size):
                self._plan.append(shard[i:i + batch_size].tolist())

    def __iter__(self):
        return iter(self._plan)

    def __len__(self) -> int:
        return len(self._plan)


def collate(batch):
    ids, feats, labels = zip(*batch)
    max_len = max(len(x) for x in ids)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, x in enumerate(ids):
        padded[i, :len(x)] = x
        lengths[i] = len(x)
    return padded, lengths, torch.stack(list(feats)), torch.stack(list(labels))
