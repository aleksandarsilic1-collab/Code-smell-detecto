"""Scan Python files with the trained model and assemble structured findings."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from . import duplication, features
from .model import SmellNet
from .tokenizer import MAX_TOKENS, VOCAB, encode

DEFAULT_THRESHOLD = 0.5

# features surfaced as human-readable evidence in reports
EVIDENCE_FEATURES = ["n_lines", "n_stmts", "n_branches", "max_nesting",
                     "n_params", "n_magic", "n_calls", "bad_param_frac",
                     "name_bad", "name_snake"]


class ScanModel:
    def __init__(self, model: SmellNet, ckpt: dict):
        self.model = model
        self.mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
        self.std = np.asarray(ckpt["feature_std"], dtype=np.float32)
        self.labels = list(ckpt["labels"])

    def predict(self, src: str):
        ids = encode(src, max_len=MAX_TOKENS)
        if not ids:
            ids = [0]
        feats = self._normalize(src)
        tokens = torch.tensor([ids], dtype=torch.long)
        lengths = torch.tensor([len(ids)], dtype=torch.long)
        f = torch.from_numpy(np.asarray([feats], dtype=np.float32))
        logits = self.model(tokens, lengths, f)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        return probs, feats

    def _normalize(self, src: str) -> np.ndarray:
        vals, _node, _magic = features.extract(src)
        if vals is None:
            vals = np.zeros(len(features.FEATURES), dtype=np.float32)
        vals = np.asarray(vals, dtype=np.float32)
        return (vals - self.mean) / (self.std + 1e-8)


def load_model(path: str) -> ScanModel:
    import os
    torch.set_num_threads(os.cpu_count() or 4)
    # Our own locally-produced checkpoints contain feature normalization
    # statistics (numpy arrays); allow them explicitly.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = SmellNet(vocab_size=cfg["vocab_size"],
                     n_features=cfg["n_features"],
                     n_classes=cfg["n_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if ckpt.get("vocab", VOCAB) != VOCAB:
        print("warning: checkpoint vocabulary differs from current tokenizer")
    return ScanModel(model, ckpt)


def extract_functions(src_text: str) -> list[dict]:
    """Return all function blocks in source order (outer + nested)."""
    try:
        tree = ast.parse(src_text)
    except (SyntaxError, ValueError):
        return []
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(src_text, node)
            if seg is None:
                try:
                    seg = ast.unparse(node)
                except Exception:
                    continue
            vals, _n, magic = features.extract(seg)
            if vals is None:
                continue
            feats = dict(zip(features.FEATURES, vals))
            funcs.append({
                "name": node.name,
                "lineno": getattr(node, "lineno", 0),
                "end": getattr(node, "end_lineno", None) or getattr(node, "lineno", 0),
                "src": seg,
                "feats": feats,
                "magic": [m for m in magic],
            })
    return funcs


def scan_text(src_text: str, model: ScanModel, threshold: float = DEFAULT_THRESHOLD,
              file: str = "<text>") -> dict:
    funcs = extract_functions(src_text)
    findings = []
    for fn in funcs:
        probs, _ = model.predict(fn["src"])
        smells = [{"label": lbl, "prob": round(float(probs[i]), 3)}
                  for i, lbl in enumerate(model.labels)
                  if probs[i] >= threshold]
        evidence = {fl: fn["feats"].get(fl, 0.0) for fl in EVIDENCE_FEATURES}
        findings.append({
            "file": file,
            "name": fn["name"],
            "lineno": fn["lineno"],
            "end": fn["end"],
            "smells": smells,
            "scores": {lbl: round(float(probs[i]), 3)
                       for i, lbl in enumerate(model.labels)},
            "evidence": evidence,
            "magic_numbers": fn["magic"],
        })

    # cross-function duplication within the file
    pairs = duplication.find_duplicates([f["src"] for f in funcs])
    dups = []
    for a, b, sim in pairs:
        dups.append({
            "file": file,
            "a": {"name": funcs[a]["name"], "lineno": funcs[a]["lineno"]},
            "b": {"name": funcs[b]["name"], "lineno": funcs[b]["lineno"]},
            "sim": round(float(sim), 3),
        })
    return {"file": file, "findings": findings, "duplicates": dups}


def scan_file(path: Path, model: ScanModel, threshold: float = DEFAULT_THRESHOLD) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"file": str(path), "findings": [], "duplicates": [], "error": "unreadable"}
    return scan_text(text, model, threshold, file=str(path))


def scan_path(path: str, model: ScanModel, threshold: float = DEFAULT_THRESHOLD) -> dict:
    p = Path(path)
    if p.is_file():
        files = [p]
    else:
        files = sorted(f for f in p.rglob("*.py") if f.is_file())
    results = []
    for f in files:
        results.append(scan_file(f, model, threshold))
    return aggregate(results)


def aggregate(results: list[dict]) -> dict:
    summary: dict[str, int] = {}
    total = 0
    for r in results:
        for fid in r["findings"]:
            total += len(fid["smells"])
            for s in fid["smells"]:
                summary[s["label"]] = summary.get(s["label"], 0) + 1
    files_with_smells = sum(1 for r in results if any(f["smells"] for f in r["findings"]))
    return {
        "files_scanned": len(results),
        "files_with_smells": files_with_smells,
        "total_smells": total,
        "summary": summary,
        "files": results,
    }
