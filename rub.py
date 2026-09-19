#!/usr/bin/env python3
"""SmellNet CLI.

Usage examples:
  python run.py train --per-family 1500 --real --epochs 8
  python run.py scan path/to/project --report text
  python run.py scan path/to/file.py --report html --out report.html
  python run.py scan path/to/project --report json --threshold 0.6
"""
from __future__ import annotations

import argparse
import sys


def _cmd_train(argv):
    from smellnet.train import main as train_main
    train_main(argv)


def _cmd_scan(argv):
    from smellnet.scan import load_model, scan_path
    from smellnet.report import write

    ap = argparse.ArgumentParser(prog="run.py scan",
                                 description="Scan Python code for smells.")
    ap.add_argument("path", help="file or directory of .py files")
    ap.add_argument("--model", default="models/smellnet.pt",
                    help="trained checkpoint (default models/smellnet.pt)")
    ap.add_argument("--report", choices=["text", "json", "html"],
                    default="text", help="report format")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="probability threshold per smell (default 0.5)")
    ap.add_argument("--out", default="",
                    help="write report to this file instead of stdout")
    args = ap.parse_args(argv)

    model = load_model(args.model)
    report = scan_path(args.path, model, threshold=args.threshold)
    write(args.out, report, args.report)


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        sys.exit(0)
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        _cmd_train(rest)
    elif cmd == "scan":
        _cmd_scan(rest)
    elif cmd in ("-h", "--help"):
        print(__doc__)
    else:
        print(f"unknown command: {cmd}\n{__doc__}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
