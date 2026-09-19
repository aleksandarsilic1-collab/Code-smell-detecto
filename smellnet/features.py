"""AST-based numeric features for a single Python function.

These feed the feature branch of the neural net and double as evidence
for reports. All extraction is synchronous and dependency-free (stdlib ast).
"""
from __future__ import annotations

import ast
import re
from typing import Any

COMMON_NUMBERS = {0, 1, 2, -1, 0.0, 1.0, 2.0, -1.0}

BAD_NAMES = {
    "x", "y", "z", "d", "t", "tmp", "temp", "temp2", "temp1", "stuff",
    "thing", "things", "obj", "o", "foo", "bar", "baz", "data2", "data",
    "dd", "var", "v", "n", "m", "k", "ret", "res", "val", "value2", "q",
    "f", "func", "fn",
}

SNAKE_RE = re.compile(r"^[a-z_][a-z0-9_]*(?:_[a-z0-9_]+)*$")

FEATURES = [
    "n_lines", "n_stmts", "n_branches", "max_nesting", "n_params",
    "n_defaults", "n_varargs", "n_magic", "n_calls", "n_assigns",
    "n_returns", "n_raises", "n_nested_defs", "n_loops", "has_try",
    "has_while", "name_len", "name_snake", "name_bad", "bad_param_frac",
    "n_imports",
]

_NEST_CTX = (ast.If, ast.For, ast.While, ast.Try, ast.With,
             ast.ExceptHandler, ast.AsyncFor, ast.AsyncWith)


def _is_magic_number(node: ast.AST, parent: ast.AST) -> bool:
    if not isinstance(node, ast.Constant):
        return False
    if not isinstance(node.value, (int, float)):
        return False
    if isinstance(node.value, bool):
        return False
    if node.value in COMMON_NUMBERS or node.lineno is None:
        return False
    # Numbers that are assigned to an ALL_CAPS named constant are idioms, not smells.
    if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
        tgt = parent.targets[0]
        if isinstance(tgt, ast.Name) and tgt.id.isupper():
            return False
    return True


def _collect(node: ast.AST, stats: dict[str, Any], parent: ast.AST | None = None,
             depth: int = 0) -> None:
    t = type(node)
    if t is ast.FunctionDef or t is ast.AsyncFunctionDef:
        if parent is not None:
            stats["n_nested_defs"] += 1
        stats["max_nesting"] = max(stats["max_nesting"], depth)
        for a in node.args.args + node.args.kwonlyargs:
            stats["n_stmts"] += 1  # params counted as pseudo-statements
    if t in _NEST_CTX and parent is not None:
        stats["n_stmts"] += 1
        stats["max_nesting"] = max(stats["max_nesting"], depth)
        if t in (ast.For, ast.While, ast.AsyncFor):
            stats["n_loops"] += 1
        if t is ast.Try:
            stats["has_try"] = 1
        if t is ast.While:
            stats["has_while"] = 1
    if t is ast.If or t is ast.IfExp:
        stats["n_branches"] += 1
    if t is ast.If:
        stats["n_stmts"] += 1
    if t is ast.Call and parent is not None:
        stats["n_calls"] += 1
    if t is ast.Assign or t is ast.AnnAssign or t is ast.AugAssign:
        stats["n_assigns"] += 1
        stats["n_stmts"] += 1
    if t is ast.Return:
        stats["n_returns"] += 1
    if t is ast.Raise:
        stats["n_raises"] += 1
    if t is ast.Import or t is ast.ImportFrom:
        stats["n_imports"] += 1
    if isinstance(node, ast.Constant):
        if _is_magic_number(node, parent):
            num = node.value
            stats["magic_numbers"].append(num if isinstance(num, int) else round(num, 2))
        stats["n_stmts"] += 0
    n = depth
    if t in _NEST_CTX:
        # An `elif` (a single If stashed in the parent's orelse) is NOT a
        # deeper nesting level -- it is the same block at the same indentation.
        is_elif = (t is ast.If and parent is not None
                   and isinstance(parent, ast.If)
                   and any(c is node for c in parent.orelse))
        if not is_elif:
            n = depth + 1
    for child in ast.iter_child_nodes(node):
        _collect(child, stats, parent=node, depth=n)


def extract_from_function(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[float]:
    """Extract the feature vector for an already-parsed AST function node."""
    stats: dict[str, Any] = {
        "n_stmts": 0, "n_branches": 0, "max_nesting": 0, "n_params": 0,
        "n_defaults": 0, "n_varargs": 0, "n_magic": 0, "n_calls": 0,
        "n_assigns": 0, "n_returns": 0, "n_raises": 0, "n_nested_defs": 0,
        "n_loops": 0, "has_try": 0, "has_while": 0, "name_len": 0,
        "name_snake": 0, "name_bad": 0, "bad_param_frac": 0, "n_imports": 0,
        "magic_numbers": [], "n_lines": 0,
    }

    src_seg_lo = 0
    if hasattr(func, "lineno") and func.lineno is not None and hasattr(func, "end_lineno"):
        src_seg_lo = func.lineno
        stats["n_lines"] = int(func.end_lineno - func.lineno + 1)

    name = func.name
    stats["name_len"] = len(name)
    stats["name_snake"] = 1 if SNAKE_RE.match(name) else 0
    stats["name_bad"] = 1 if name in BAD_NAMES else 0

    args = func.args
    positional = list(args.posonlyargs) + list(args.args)
    stats["n_params"] = len(positional) + len(args.kwonlyargs) + (1 if args.vararg else 0) + (1 if args.kwarg else 0)
    stats["n_defaults"] = len(args.defaults)
    stats["n_varargs"] = (1 if args.vararg else 0) + (1 if args.kwarg else 0)

    bad = 0
    total = 0
    for a in positional + list(args.kwonlyargs):
        an = a.arg
        total += 1
        if an in ("self", "cls"):
            continue
        if an in BAD_NAMES:
            bad += 1
    stats["bad_param_frac"] = bad / total if total else 0.0

    _collect(func, stats, parent=None, depth=0)
    stats["n_magic"] = len(stats["magic_numbers"])
    stats["magic_numbers"] = stats["magic_numbers"][:16]

    return [float(stats[f]) for f in FEATURES]


def extract(src: str) -> tuple[list[float] | None, ast.AST | None, list[Any]]:
    """Extract features for the first function found in `src`.

    Returns (features, function_node, magic_numbers).
    """
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None, None, []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            feats = extract_from_function(node)
            stats = {f: v for f, v in zip(FEATURES, feats)}
            _m = _collect_magic(node)
            return feats, node, _m
    return None, None, []


def _collect_magic(func: ast.AST) -> list[Any]:
    magic: list[Any] = []
    for par, child in _walk_with_parent(func):
        if isinstance(child, ast.Constant) and isinstance(child.value, (int, float)) \
                and not isinstance(child.value, bool) and _is_magic_number(child, par):
            v = child.value
            magic.append(v if isinstance(v, int) else round(v, 2))
    return magic[:16]


def _walk_with_parent(root: ast.AST):
    stack = [(root, None)]
    while stack:
        node, parent = stack.pop()
        yield parent, node
        for c in reversed(list(ast.iter_child_nodes(node))):
            stack.append((c, node))
