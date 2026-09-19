"""smellnet - a neural code-smell detector for Python.

Detects maintainability problems (code smells) in Python functions using a
trained neural network over tokenized source plus AST-derived numeric features.
"""

__version__ = "0.1.0"

LABELS = [
    "long_function",
    "many_parameters",
    "deep_nesting",
    "magic_numbers",
    "unclear_naming",
    "complex_responsibilities",
    "duplication",
]

N_LABELS = len(LABELS)
