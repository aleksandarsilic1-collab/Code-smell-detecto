"""Synthetic training-data generator.

Produces labeled Python function snippets for the smell classifier. The
generator is template-driven but randomized: identifiers, string literals,
values, field names and structural choices vary between samples so the model
learns *patterns* instead of memorizing fixed strings. Clean and smelly
families are produced in controlled proportions; labels are assigned by
construction and then snapped to the same rule thresholds the scanner uses,
keeping the dataset self-consistent.
"""
from __future__ import annotations

import ast
import random
from typing import Callable, Optional

from . import LABELS as _LABELS
from .features import FEATURES as _FEATURE_NAMES

# ---------------------------------------------------------------------------
# Template pickers
# ---------------------------------------------------------------------------

NOUNS = [
    "items", "records", "users", "orders", "products", "entries", "payloads",
    "config", "rows", "fields", "results", "matches", "tokens", "chunks",
    "values", "counts", "scores", "prices", "names", "paths", "events",
    "samples", "reports", "messages", "contacts", "invoices", "metrics",
    "slots", "queues", "profiles", "buckets",
]

VERBS = [
    "normalize", "validate", "process", "parse", "convert", "extract",
    "compute", "merge", "aggregate", "filter", "sort", "format", "build",
    "resolve", "precompute", "summarize", "sanitize", "collect", "describe",
    "transform", "deduplicate",
]

SUBJECTS = [
    "order", "user", "record", "row", "entry", "token", "metric", "event",
    "invoice", "profile", "message", "sample", "config", "report",
]

BAD_FUNC_NAMES = [
    "do_stuff", "handle_data", "process_data2", "x", "temp", "stuff",
    "make_it_work", "func", "f", "d", "thing", "foo_bar", "helper",
    "do_it", "get_thing",
]

BAD_VARS = ["x", "y", "z", "d", "t", "tmp", "stuff", "foo", "bar", "data2", "obj", "v", "q"]

GOOD_FUNC_NAMES = [f"{v}_{s}" for v in VERBS for s in SUBJECTS] + [
    "load_config", "save_state", "connect_db", "read_input", "write_output",
    "get_total", "find_by_id", "apply_discount", "run_pipeline",
    "handle_request", "make_response", "split_lines", "flatten_matrix",
]


def _good_name(rng: random.Random, used: set[str]) -> str:
    for _ in range(50):
        n = f"{rng.choice(VERBS)}_{rng.choice(SUBJECTS)}"
        if n not in used:
            used.add(n)
            return n
    n = f"process_{rng.choice(SUBJECTS)}_{rng.randint(10, 99)}"
    used.add(n)
    return n


def _var(rng: random.Random, used: set[str]) -> str:
    for _ in range(80):
        n = rng.choice(NOUNS)
        if n not in used:
            used.add(n)
            return n
    n = f"{rng.choice(NOUNS)}_{rng.randint(10, 99)}"
    used.add(n)
    return n


class CodeGen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used_vars: set[str] = set()
        self.used_params: set[str] = set()
        self.used_names: set[str] = set()

    def var(self) -> str:
        return _var(self.rng, self.used_vars)

    def param(self) -> str:
        return _var(self.rng, self.used_params)

    def name(self) -> str:
        return _good_name(self.rng, self.used_names)

    def p_int(self, lo: int, hi: int) -> int:
        return self.rng.randint(lo, hi)

    def p_granular(self):
        if self.rng.random() < 0.7:
            return self.p_int(3, 400)
        return round(self.rng.uniform(2.5, 250.5), 1)

    def pick(self, seq):
        return self.rng.choice(seq)

    def maybe(self, prob: float) -> bool:
        return self.rng.random() < prob

    def field(self) -> str:
        return self.rng.choice(FIELDS)


FIELDS = ["id", "name", "status", "total", "count", "value", "price", "type",
          "source", "target", "key", "label", "size", "score", "level", "sku"]


def _wrap(g: CodeGen, params: list[str], body: list[str], prelude: Optional[str] = None,
          name: Optional[str] = None) -> str:
    if name is None:
        name = g.name()
    sig = ", ".join(params)
    lines = [f"def {name}({sig}):"]
    if prelude:
        lines.append("    " + prelude)
    lines += ["    " + l for l in body]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared statement chunks (return function-body lines, self-contained)
# ---------------------------------------------------------------------------

def _chunk_sum(g: CodeGen, src: str) -> list[str]:
    total, v = g.var(), g.var()
    return [f"{total} = 0.0", f"for {v} in {src}:", f"    {total} += {v}"]


def _chunk_filter(g: CodeGen, src: str) -> list[str]:
    out, item = g.var(), g.var()
    return [
        f"{out} = []",
        f"for {item} in {src}:",
        f"    if {item} is None:",
        f"        continue",
        f"    {out}.append({item})",
    ]


def _chunk_group(g: CodeGen, src: str) -> list[str]:
    grouped, item, key = g.var(), g.var(), g.field()
    return [
        f"{grouped} = {{}}",
        f"for {item} in {src}:",
        f"    {grouped}.setdefault({item}.get('{key}'), []).append({item})",
    ]


def _chunk_score(g: CodeGen, src: str) -> list[str]:
    out, item = g.var(), g.var()
    return [
        f"{out} = []",
        f"for {item} in {src}:",
        f"    {out}.append((len(str({item})), {item}))",
        f"{out}.sort(key=lambda pair: pair[0], reverse=True)",
    ]


# ---------------------------------------------------------------------------
# Clean-function builders (8-28 lines, good names, <=5 params)
# ---------------------------------------------------------------------------

def _clean_sum(g: CodeGen) -> str:
    src, total, v = g.param(), g.var(), g.var()
    body = [
        f'"""Count non-null entries in {src}."""',
        f"{total} = 0",
        f"for {v} in {src}:",
        f"    if {v} is None:",
        f"        continue",
        f"    {total} += 1",
        f"return {total}",
    ]
    return _wrap(g, [src], body)


def _clean_parse(g: CodeGen) -> str:
    src, row, ident, key = g.param(), g.var(), g.var(), g.field()
    body = [
        f'"""Map {src} to validated dicts."""',
        f"{row} = []",
        f"for {g.var()} in {src}:",
        f"    {ident} = {g.var()}.get('{key}')",
        f"    if {ident} is None:",
        f"        continue",
        f"    {row}.append(dict(id={ident}, source={g.var()}))",
        f"return {row}",
    ]
    return _wrap(g, [src], body)


def _clean_ints(g: CodeGen) -> str:
    out, src, v = g.var(), g.param(), g.var()
    body = [
        f'"""Return integer forms of {src}, skipping junk."""',
        f"{out} = []",
        f"for {v} in ({src} or []):",
        f"    try:",
        f"        {out}.append(int({v}))",
        f"    except (TypeError, ValueError):",
        f"        continue",
        f"return {out}",
    ]
    return _wrap(g, [src], body)


def _clean_load(g: CodeGen) -> str:
    conf, path = g.var(), g.param()
    body = [
        f'"""Load a JSON {conf} file, raising on malformed input."""',
        f"try:",
        f"    with open({path}, 'r', encoding='utf-8') as handle:",
        f"        {conf} = json.load(handle)",
        f"except (OSError, json.JSONDecodeError) as exc:",
        f"    raise ValueError('cannot load {conf}') from exc",
        f"if not isinstance({conf}, dict):",
        f"    raise TypeError('{conf} must be a mapping')",
        f"return {conf}",
    ]
    return _wrap(g, [path], body)


def _clean_group(g: CodeGen) -> str:
    grouped, items, key = g.var(), g.param(), g.field()
    body = [
        f'"""Group {items} by \'{key}\' preserving original order."""',
        f"{grouped} = {{}}",
        f"for {g.var()} in {items}:",
        f"    {g.var()}.setdefault(" f"{g.var()}.get('{key}')" f", []).append({g.var()})",
        f"result = []",
        f"for {g.var()}, {g.var()} in sorted({grouped}.items()):",
        f"    result.append(dict(key={g.var()}, values={g.var()}))",
        f"return result",
    ]
    return _wrap(g, [items], body)


def _clean_flatten(g: CodeGen) -> str:
    batches, out, batch = g.param(), g.var(), g.var()
    body = [
        f'"""Flatten {batches} and drop repeated entries."""',
        f"{out} = []",
        f"for {batch} in {batches}:",
        f"    for {g.var()} in {batch}:",
        f"        if {g.var()} in {out}:",
        f"            continue",
        f"        {out}.append({g.var()})",
        f"return {out}",
    ]
    return _wrap(g, [batches], body)


def _clean_topk(g: CodeGen) -> str:
    items, keep, key = g.param(), g.p_int(2, 20), g.field()
    scored, entry = g.var(), g.var()
    body = [
        f'"""Return the top {keep} {items} by \'{key}\'."""',
        f"{scored} = []",
        f"for {entry} in {items}:",
        f"    val = {entry}.get('{key}')",
        f"    if val is None:",
        f"        continue",
        f"    {scored}.append((val, {entry}))",
        f"{scored}.sort(key=lambda pair: pair[0], reverse=True)",
        f"return [entry for _, entry in {scored}[:{keep}]]",
    ]
    return _wrap(g, [items, f"limit={keep}"], body)


def _clean_orchestrate(g: CodeGen) -> str:
    """Single-responsibility but longer: 30-45 lines, up to 5 params."""
    src, threshold = g.param(), g.p_int(3, 12)
    collated, dedup, flagged = g.var(), g.var(), g.var()
    body = [
        f'"""Pipeline: normalize {src}, deduplicate, and flag anomalies."""',
        f"{collated} = []",
        f"for {g.var()} in {src}:",
        f"    if {g.var()} is None:",
        f"        continue",
        f"    {collated}.append({g.var()})",
        f"# de-duplicate while keeping first occurrence",
        f"{dedup} = dict.fromkeys({collated})",
        f"{flagged} = []",
        f"for {g.var()}, {g.var()} in enumerate({dedup}):",
        f"    if len(str({g.var()})) > {threshold}:",
        f"        {flagged}.append({g.var()})",
        f"    else:",
        f"        {flagged}.append({g.var()})",
        f"{collated} = sorted({flagged}, key=str, reverse=False)",
        f"seen = set()",
        f"unique = []",
        f"for {g.var()} in {collated}:",
        f"    if {g.var()} not in seen:",
        f"        seen.add({g.var()})",
        f"        unique.append({g.var()})",
        f"counts = dict()",
        f"for {g.var()} in unique:",
        f"    counts[{g.var()}] = counts.get({g.var()}, 0) + 1",
        f"return dict(counts)",
    ]
    return _wrap(g, [src, f"limit={g.p_int(5, 30)}"], body)


_CLEAN_BUILDERS = [_clean_sum, _clean_parse, _clean_ints, _clean_load,
                   _clean_group, _clean_flatten, _clean_topk,
                   _clean_orchestrate]


# ---------------------------------------------------------------------------
# Smelly builders
# ---------------------------------------------------------------------------

def gen_long(g: CodeGen) -> str:
    """Very long function (>55 lines) doing many small things."""
    src = g.param()
    body = [f'"""Process the whole {src} pipeline in one go."""', f"acc = []"]
    chunks: list[Callable[[CodeGen, str], list[str]]] = [
        _chunk_sum, _chunk_filter, _chunk_group, _chunk_score,
    ]
    while len(body) < 58:
        step = g.pick(chunks)(g, src)
        body += ["# phase"] + step
        if g.maybe(0.4):
            body.append(f"acc.extend({step[0].split(' = ')[0]})" if " = " in step[0] else "acc.append(1)")
        if g.maybe(0.3):
            body.append(f"for {g.var()} in acc:")
            body.append(f"    {g.var()} = {g.var()} + 1")
    body.append("return acc")
    return _wrap(g, [src], body, prelude="")



def gen_many_params(g: CodeGen, n: Optional[int] = None) -> str:
    n = n or g.p_int(7, 12)
    params = []
    for _ in range(n):
        p = g.param()
        params.append(p)
    if g.maybe(0.5):
        params.append(f"limit={g.p_int(1, 99)}")
    first = params[0]
    body = [f'"""Apply {n} inputs at once."""',
            f"if {first} is None:",
            f"    return []",
            f"result = [x for x in {first} if x]",
            f"return result"]
    return _wrap(g, params, body)


def gen_nested(g: CodeGen) -> str:
    """Deeply nested control flow (max depth 6)."""
    src, out = g.param(), g.var()
    row, link, extra = g.var(), g.var(), g.var()
    order = [
        (1, f"# walk {src} recursively and collect leaves"),
        (1, f"for {row} in {src}:"),
        (2, f"if {row}.get('active'):"),
        (3, f"for {link} in {row}.get('links', []):"),
        (4, f"try:"),
        (5, f"if {link}:"),
        (6, f"while {link}:"),
        (7, f"{out}.append({link})"),
        (7, f"break"),
        (5, f"else:"),
        (6, f"{out}.append('')"),
        (4, f"except (KeyError, TypeError):"),
        (5, f"continue"),
        (3, f"for {extra} in {row}.get('children', []):"),
        (4, f"if {extra} is not None:"),
        (5, f"{out}.append({extra})"),
    ]
    body = ["    " * lvl + line for lvl, line in order]
    return _wrap(g, [src], body, prelude='')


def gen_magic(g: CodeGen) -> str:
    a = g.param()
    body = [
        f'"""Tune thresholds from observed behaviour."""',
        f"if {a} > {g.p_granular()}:",
        f"    result = {a} * {g.p_granular()}",
        f"else:",
        f"    result = {a} - {g.p_int(3, 40)}",
        f"if result < {g.p_int(2, 9)}:",
        f"    result = {g.p_granular()}",
        f"for {g.var()} in range({g.p_int(3, 30)}):",
        f"    result += {g.p_granular()}",
        f"return result",
    ]
    return _wrap(g, [a], body)


def gen_bad_names(g: CodeGen) -> str:
    a1, a2 = g.pick(BAD_VARS), g.pick(BAD_VARS)
    b = g.pick(BAD_VARS)
    name = g.pick(BAD_FUNC_NAMES)
    body = [
        f"# {name} routine - tune later",
        f"b = {{}}",
        f"for i in range(len({a1})):",
        f"    x = {a1}[i]",
        f"    if x in {a2}:",
        f"        y = {a2}[x]",
        f"        b[x] = y * 2",
        f"return b",
    ]
    body = [l.replace("b = {}", f"{b} = {{}}") for l in body]
    body = [l.replace("b[x]", f"{b}[x]") for l in body]
    return _wrap(g, [a1, a2], body, name=name)


def gen_god(g: CodeGen) -> str:
    """Large, god-object style function mixing unrelated responsibilities."""
    src, dest = g.param(), g.param()
    rows, total = g.var(), g.var()
    body = [
        f'"""End to end: read {src}, clean, aggregate, write {dest}."""',
        f"log(f'starting {src}')",
        f"{rows} = []",
        f"with open({src}, 'r', encoding='utf-8') as fp:",
        f"    for line in fp:",
        f"        if line.strip():",
        f"            {rows}.append(line.strip().split(','))",
        f"arg = {dest}.replace('.txt', '.csv')",
        f"# business rule A",
        f"{total} = 0",
        f"for {g.var()} in {rows}:",
        f"    {total} += len({g.var()})",
        f"# business rule B",
        f"for {g.var()} in range({g.p_int(2, 9)}):",
        f"    {total} *= {g.p_int(2, 5)}",
        f"# business rule C",
        f"report = [dict(row=r, size=len(r)) for r in {rows}]",
        f"report.sort(key=lambda r: r['size'], reverse=False)",
        f"# write everything in the same function",
        f"with open(arg, 'w', encoding='utf-8') as fp:",
        f"    fp.write(str(report))",
        f"log(f'wrote {{len(report)}} rows to {{arg}}')",
        f"return {total}",
    ]
    return _wrap(g, [src, dest], body)


def gen_in_function_dup(g: CodeGen) -> str:
    src = g.param()
    out, item = g.var(), g.var()
    block = [
        f"for {item} in {src}:",
        f"    if {item} is None:",
        f"        continue",
        f"    {out}.append({item})",
    ]
    body = []
    for _ in range(3):
        body.append(f"{out} = []")
        body += block
    body.append(f"return {out}")
    return _wrap(g, [src], body, prelude='"""Deduplicate rows from every source."""')


def gen_many_params_clean(g: CodeGen) -> str:
    """5 cohesive params with defaults -- boundary case, NOT smelly."""
    a, b, c = g.param(), g.param(), g.param()
    body = [
        f'"""Merge sources in order, honouring the key and flag."""',
        f"base = dict({b} or [])",
        f"for {g.var()} in ({a} or []):",
        f"    base[{g.var()}.get('id')] = {g.var()}",
        f"for {g.var()}, {g.var()} in ({c} or []):",
        f"    base.setdefault({g.var()}, {g.var()})",
        f"if not {g.var()}:",
        f"    return list(base.values())",
        f"keys = {g.var()}.split(',')",
        f"return [base[k] for k in keys if k in base]",
    ]
    return _wrap(g, [a, b, c, "keys=None", "flag=True"], body)


# ---------------------------------------------------------------------------
# Label assignment (constructed + rule-snap for consistency)
# ---------------------------------------------------------------------------

def _rules(f: dict):
    return {
        "long_function": f["n_lines"] >= 50,
        "many_parameters": f["n_params"] >= 6,
        "deep_nesting": f["max_nesting"] >= 5,
        "magic_numbers": f["n_magic"] >= 3,
        "unclear_naming": bool(f["name_bad"]) or f["bad_param_frac"] >= 0.4,
        "complex_responsibilities": f["n_stmts"] >= 40,
        "duplication": False,
    }


def snap_labels(feats: dict, constructed: set[str]) -> set[str]:
    labels = set(constructed)
    for label, hit in _rules(feats).items():
        if hit:
            labels.add(label)
    return labels


def generate_sample(rng: random.Random, family: str):
    g = CodeGen(rng)
    src: Optional[str] = None
    constructed: set[str] = set()

    if family == "clean":
        src = g.pick(_CLEAN_BUILDERS)(g)
    elif family == "long":
        src = gen_long(g)
        constructed.add("long_function")
    elif family == "many_params":
        src = gen_many_params(g)
        constructed.add("many_parameters")
    elif family == "nested":
        src = gen_nested(g)
        constructed.add("deep_nesting")
    elif family == "magic":
        src = gen_magic(g)
        constructed.add("magic_numbers")
    elif family == "bad_names":
        src = gen_bad_names(g)
        constructed.add("unclear_naming")
    elif family == "god":
        src = gen_god(g)
        constructed.add("complex_responsibilities")
    elif family == "dup":
        src = gen_in_function_dup(g)
        constructed.add("duplication")
    else:
        return None, set(), {}

    if src is None:
        return None, set(), {}

    try:
        ast.parse(src)
    except SyntaxError:
        return None, set(), {}

    val_feats, _node, _magic = _extract(src)
    if val_feats is None:
        return None, set(), {}
    feats = dict(zip(_FEATURE_NAMES, val_feats))
    labels = snap_labels(feats, constructed)
    return src, labels, feats


def build_dataset(per_family: int = 1500, seed: int = 7):
    """Return (src, label_vector, features_dict) samples, roughly balanced."""
    rng = random.Random(seed)
    families = ["clean"] * int(per_family * 1.2)
    for fam in ["long", "many_params", "nested", "magic", "bad_names", "god", "dup"]:
        families += [fam] * per_family
    rng.shuffle(families)

    label_idx = {l: i for i, l in enumerate(_LABELS)}
    samples = []
    for fam in families:
        src, labels, feats = generate_sample(rng, fam)
        if src is None:
            continue
        vec = [0] * len(_LABELS)
        for l in labels:
            if l in label_idx:
                vec[label_idx[l]] = 1
        samples.append((src, vec, [feats[f] for f in _FEATURE_NAMES]))
    return samples


from .features import extract as _extract  # noqa: E402


def family_summary(samples) -> dict[str, int]:
    counts = {l: 0 for l in _LABELS}
    for _, vec, _ in samples:
        for l, v in zip(_LABELS, vec):
            if v:
                counts[l] += 1
    return counts
