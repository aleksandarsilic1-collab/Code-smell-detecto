"""The neural network: a bidirectional LSTM over tokenized function bodies
with a parallel MLP over AST-derived numeric features, fused into a
multi-label classifier head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SmellNet(nn.Module):
    def __init__(self, vocab_size: int, n_features: int, n_classes: int,
                 emb: int = 64, hid: int = 80, feat_hid: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.n_classes = n_classes

        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)

        self.feat_net = nn.Sequential(
            nn.Linear(n_features, feat_hid),
            nn.ReLU(),
            nn.Linear(feat_hid, feat_hid),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hid * 4 + feat_hid, 160),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(160, n_classes),
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor,
                feats: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        e = self.emb(tokens)                                     # B,T,E
        packed = pack_padded_sequence(e, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)      # B,T,2H

        mask = (tokens != 0).unsqueeze(-1)
        out = out * mask
        h_max = out.max(dim=1).values                            # B,2H
        denom = mask.sum(dim=1).clamp(min=1)
        h_mean = out.sum(dim=1) / denom                          # B,2H
        h = torch.cat([h_max, h_mean], dim=-1)                   # B,4H

        f = self.feat_net(feats)                                 # B,feat_hid
        x = torch.cat([h, f], dim=-1)
        logits = self.head(self.dropout(x))
        return logits


def loss_fn(logits: torch.Tensor, targets: torch.Tensor):
    return F.binary_cross_entropy_with_logits(logits, targets)


def predict(model: SmellNet, tokens: torch.Tensor, lengths: torch.Tensor,
            feats: torch.Tensor):
    model.eval()
    with torch.no_grad():
        logits = model(tokens, lengths, feats)
        return torch.sigmoid(logits)
