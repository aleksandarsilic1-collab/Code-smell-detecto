"""Training script.

Bundles:
  * synthetic labeled samples from generate_data
  * optionally real Python functions from the stdlib, labelled by the same
    rule thresholds (and cross-function duplication) -> "distant supervision"
  * tokenizer + feature normalization
  * SmellNet training loop with per-class validation metrics
  * checkpoint written to models/smellnet.pt
"""
from __future__ import annotations

import argparse
import ast
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import LABELS
from .dataset import BucketBatchSampler, SmellDataset, collate
from . import duplication, features, generate_data
from .model import SmellNet, loss_fn
from .tokenizer import VOCAB, VOCAB_SIZE

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Real code (distant supervision)
# ---------------------------------------------------------------------------

def _collect_real_files(limit: int) -> list[Path]:
    import sysconfig
    root = Path(sysconfig.get_paths()["stdlib"])
    rng = random.Random(11)
    files = [p for p in root.rglob("*.py") if p.stat().st_size < 300_000]
    rng.shuffle(files)
    return files[:limit]


def _extract_file_functions(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    funcs = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        seg = ast.get_source_segment(text, node) or ast.unparse(node)
        feats_vals, _, _ = features.extract(seg)
        if feats_vals is None:
            continue
        funcs.append({
            "src": seg,
            "feats": feats_vals,
            "name": node.name,
            "lineno": getattr(node, "lineno", 0),
        })
    return funcs


def collect_real_samples(max_functions: int) -> list[tuple[str, list[int], dict]]:
    label_idx = {l: i for i, l in enumerate(LABELS)}
    samples = []
    func_counts = {"duplication": 0, "long_function": 0, "many_parameters": 0,
                   "deep_nesting": 0, "magic_numbers": 0, "unclear_naming": 0,
                   "complex_responsibilities": 0}
    for path in _collect_real_files(60):
        funcs = _extract_file_functions(path)
        if not funcs:
            continue
        dup_flags = [False] * len(funcs)
        for a, b, _ in duplication.find_duplicates([f["src"] for f in funcs]):
            dup_flags[a] = True
            dup_flags[b] = True
        for f, is_dup in zip(funcs, dup_flags):
            feats = dict(zip(features.FEATURES, f["feats"]))
            rules = generate_data._rules(feats)
            rules["duplication"] = is_dup
            vec = [0] * len(LABELS)
            for l, hit in rules.items():
                if hit and l in label_idx:
                    vec[label_idx[l]] = 1
                    func_counts[l] += 1
            samples.append((f["src"], vec, f["feats"]))
            if len(samples) >= max_functions:
                return samples, func_counts
        if len(samples) >= max_functions:
            break
    return samples, func_counts


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _evaluate(model, loader, n_classes: int):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for tokens, lengths, feats, labels in loader:
            p = torch.sigmoid(model(tokens, lengths, feats))
            preds.append(p)
            tgts.append(labels)
    P = torch.cat(preds)
    T = torch.cat(tgts)
    truth = (T > 0.5)
    pred_b = (P > 0.5)
    acc = (pred_b == truth).float().mean(dim=0)
    tp = (pred_b & truth).sum(dim=0).float()
    fp = (pred_b & ~truth).sum(dim=0).float()
    fn = (~pred_b & truth).sum(dim=0).float()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return acc, f1, prec, rec


def main(argv=None) -> None:
    import os
    torch.set_num_threads(os.cpu_count() or 4)

    ap = argparse.ArgumentParser(description="Train the smell classifier.")
    ap.add_argument("--per-family", type=int, default=1500)
    ap.add_argument("--real", action="store_true",
                    help="add stdlib functions labelled by rules (distant supervision)")
    ap.add_argument("--real-max", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(MODEL_DIR / "smellnet.pt"))
    ap.add_argument("--print-samples", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("[1/4] generating synthetic samples ...")
    syn = generate_data.build_dataset(per_family=args.per_family, seed=args.seed)
    print(f"      {len(syn)} synthetic samples")
    if args.print_samples:
        for src, vec, _ in syn[:args.print_samples]:
            print("----", [l for l, v in zip(LABELS, vec) if v])
            print(src)
            print()

    if args.real:
        print("[2/4] collecting real stdlib functions ...")
        real, counts = collect_real_samples(args.real_max)
        print(f"      {len(real)} real samples labelled {counts}")
        syn = syn + real
        random.Random(args.seed).shuffle(syn)

    label_summary = generate_data.family_summary(syn)
    print(f"      label distribution: {label_summary}")

    rng = random.Random(args.seed)
    rng.shuffle(syn)
    split = int(len(syn) * 0.9)
    train_s, val_s = syn[:split], syn[split:]

    means = np.asarray([s[2] for s in train_s], dtype=np.float32).mean(axis=0)
    stds = np.asarray([s[2] for s in train_s], dtype=np.float32).std(axis=0)

    train_ds = SmellDataset(train_s, means, stds)
    val_ds = SmellDataset(val_s, means, stds)
    train_batch = BucketBatchSampler(train_ds.lengths, args.batch,
                                     shuffle=True, seed=args.seed)
    val_batch = BucketBatchSampler(val_ds.lengths, args.batch,
                                   shuffle=False, seed=args.seed)
    train_dl = DataLoader(train_ds, batch_sampler=train_batch, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_sampler=val_batch, collate_fn=collate)

    print(f"[3/4] building model (vocab={VOCAB_SIZE}, features={features.FEATURES.__len__()})")
    model = SmellNet(vocab_size=VOCAB_SIZE, n_features=len(features.FEATURES),
                     n_classes=len(LABELS))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[4/4] training {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, nbatch = 0.0, 0
        for tokens, lengths, feats, labels in train_dl:
            opt.zero_grad()
            logits = model(tokens, lengths, feats)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nbatch += 1
        acc, f1, _, _ = _evaluate(model, val_dl, len(LABELS))
        avg = f1.mean().item()
        print(f"  epoch {epoch:2d} | loss {total_loss / nbatch:.4f} | "
              f"val acc {acc.mean().item():.3f} | macro-F1 {avg:.3f}")

    acc, f1, prec, rec = _evaluate(model, val_dl, len(LABELS))
    print("\nvalidation (threshold 0.5):")
    print(f"  {'label':<26}{'acc':>7}{'prec':>7}{'rec':>7}{'f1':>7}")
    for i, l in enumerate(LABELS):
        print(f"  {l:<26}{acc[i].item():>7.3f}{prec[i].item():>7.3f}"
              f"{rec[i].item():>7.3f}{f1[i].item():>7.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_config": {"vocab_size": VOCAB_SIZE,
                         "n_features": len(features.FEATURES),
                         "n_classes": len(LABELS)},
        "feature_mean": means,
        "feature_std": stds,
        "feature_names": features.FEATURES,
        "labels": LABELS,
        "vocab": VOCAB,
        "epochs": args.epochs,
        "val_f1": f1.detach().cpu().numpy().tolist(),
    }, out)
    print(f"\nsaved model -> {out}")


if __name__ == "__main__":
    main()
