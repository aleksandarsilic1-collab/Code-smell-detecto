# SmellNet — a neural code-smell detector for Python

SmellNet analyzes Python source code and flags **maintainability smells** —
patterns that aren't bugs but hurt readability and changeability. Instead of a
hand-written rules file, it uses a **neural network** trained to recognise the
patterns, with an AST feature branch to ground the model in real structural
facts (line counts, nesting depth, parameter counts, ...).

## What it detects

| class                       | definition                                                        |
|-----------------------------|-------------------------------------------------------------------|
| `long_function`             | one function doing too much (typically >50 lines / many steps)    |
| `many_parameters`           | 6+ parameters; needs a config object / dataclass                  |
| `deep_nesting`              | control flow nested 5+ levels                                     |
| `magic_numbers`             | unexplained numeric literals                                          |
| `unclear_naming`            | generic or single-letter names for functions/parameters           |
| `complex_responsibilities`  | "god function" mixing unrelated jobs                              |
| `duplication`               | copy-pasted logic (cross-function dedup + in-function repeats)    |

Cross-function duplication cannot come out of a per-function classifier (a
single function never sees its siblings), so it is detected with a normalized
token-shingle similarity pass over each file's functions — the ground truth it
emits is also used to train a `duplication` head for in-function repeats.

## Architecture

```
                          function source
                        /                  \
          [tokenizer.py]                    [features.py]
      structural vocab, 320 max tokens      21 AST-derived numbers
                      |                               |
               embedding (64)                  feature MLP (64)
                      |                               |
           BiLSTM (80, bidir)                        |
                      |                               |
        max + mean pooling (4*80)              feat_hid (64)
                      \______________ merge ____________/
                                     |
                          head MLP (160 -> 7)
                                     |
                          sigmoid multi-label logits
```

* The tokenizer is **vocabulary-free**: keywords, operators and structural
  tokens are fixed ids; identifiers/numbers/strings collapse to generic ids so
  any code can be scored without a learned dictionary.
* Length, nesting and count signals are deliberately carried by the numeric
  branch: sequence length is capped at 320 tokens so training stays tractable.
* Output is multi-label (`sigmoid` + BCE loss): one function can legitimately
  smell in several ways at once.

## Training data

A neural net needs labelled examples. Three sources are mixed:

1. **Synthetic (primary)** — `smellnet/generate_data.py` generates randomized
   clean and smelly Python functions (identifiers, values, fields, and
   structure vary per sample), each labelled by construction.
2. **Real stdlib (distant supervision)** — a seeded sample of Python stdlib
   functions is labelled with the same rule thresholds the generator uses for
   snapping, plus real cross-function duplication labels from the
   duplication module (see `train.py` → `collect_real_samples`).
3. Labels are *snapped* to rule thresholds so a "clean" function can never
   accidentally exceed a structural limit.

Run the reproducibility-minded seeds produce the same training/validation
split (`--seed`, default 7).

## Installation

```bash
pip install -r requirements.txt   # torch (CPU fine), numpy

# or install the package itself (adds a `smellnet` command):
pip install .
```

## Usage

```bash
# train (writes models/smellnet.pt)
python run.py train --per-family 1500 --real --epochs 5 --seed 7

# scan a file or directory
python run.py scan path/to/source.py            # text report
python run.py scan path/to/project --report html --out report.html
python run.py scan path/to/project --report json --out report.json
python run.py scan path/to/project --threshold 0.6
```

### Programmatic

```python
from smellnet.scan import load_model, scan_path
model = load_model("models/smellnet.pt")
report = scan_path("src", model, threshold=0.5)
```

## Example

Scanning `examples/smelly_demo.py` with the trained model:

```
### examples\smelly_demo.py
  process_data2  @ line 5
    unclear_naming (0.96)
    evidence: 10 lines, 2 params, nesting 1, 4 magic numbers, 100% weak param names
    magic literals: [100, 3, 50, 5]
  do_stuff  @ line 17
    magic_numbers (0.84)
```

Reference validation metrics (threshold 0.5, macro-F1 0.97):

| label                     | acc   | prec  | rec   | f1    |
|---------------------------|-------|-------|-------|-------|
| long_function             | 0.999 | 1.000 | 0.994 | 0.997 |
| many_parameters           | 0.997 | 0.987 | 0.980 | 0.983 |
| deep_nesting              | 1.000 | 1.000 | 1.000 | 1.000 |
| magic_numbers             | 0.998 | 1.000 | 0.979 | 0.990 |
| unclear_naming            | 0.999 | 1.000 | 0.989 | 0.994 |
| complex_responsibilities  | 0.989 | 1.000 | 0.912 | 0.954 |
| duplication               | 0.955 | 0.924 | 0.775 | 0.843 |

## Project layout

```
run.py                    # CLI (train / scan)
smellnet/
  tokenizer.py            # structural token ids (no learned vocab)
  features.py             # 21 AST-derived numeric features
  generate_data.py        # synthetic labelled data generator
  model.py                # SmellNet (BiLSTM + feature MLP + multi-label head)
  dataset.py              # dataset + length-bucketing sampler
  duplication.py          # cross-function near-duplicate detection
  train.py                # training + distant supervision
  scan.py                 # inference over files/dirs
  report.py               # text / JSON / HTML reporters
examples/                 # demo code + sample reports
models/smellnet.pt        # trained checkpoint
```

## Limitations (be honest with yourself)

* Detection quality depends on the synthetic generator's closeness to real
  code; on arbitrary real-world styles you should re-train with your own
  seeded real-code sample on top.
* The classifier is threshold-calibrated on its validation distribution —
  borderline functions near thresholds can swap sides. Use the numeric
  evidence in the report to judge.
* Duplication detection is per-file and shingle-based: it finds near-identical
  function bodies, not semantic duplicates written differently (that is a much
  harder problem).
* Only Python is supported today.
