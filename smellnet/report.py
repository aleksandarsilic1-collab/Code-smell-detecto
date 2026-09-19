"""Render scan results as text, JSON or HTML."""
from __future__ import annotations

import json
import html as html_mod

SEVERITY = {"long_function": "#e74c3c", "many_parameters": "#e67e22",
            "deep_nesting": "#f39c12", "magic_numbers": "#e67e22",
            "unclear_naming": "#8e44ad", "complex_responsibilities": "#c0392b",
            "duplication": "#2980b9"}

SMELL_HINT = {
    "long_function": "single function >50 lines or juggling many steps",
    "many_parameters": "6+ parameters; consider a config object or dataclass",
    "deep_nesting": "control flow nested 5+ levels; extract guarded helpers",
    "magic_numbers": "unexplained numeric literals; use named constants",
    "unclear_naming": "generic / single-letter names; rename for intent",
    "complex_responsibilities": "god function doing several jobs; split it",
    "duplication": "copy-pasted logic; extract a shared helper",
}


def _evidence_text(ev: dict) -> str:
    parts = []
    if ev.get("n_lines"):
        parts.append(f"{int(ev['n_lines'])} lines")
    if ev.get("n_params"):
        parts.append(f"{int(ev['n_params'])} params")
    if ev.get("max_nesting"):
        parts.append(f"nesting {int(ev['max_nesting'])}")
    if ev.get("n_magic"):
        parts.append(f"{int(ev['n_magic'])} magic numbers")
    if ev.get("n_stmts"):
        parts.append(f"{int(ev['n_stmts'])} statements")
    if ev.get("bad_param_frac", 0):
        parts.append(f"{int(ev['bad_param_frac'] * 100)}% weak param names")
    return ", ".join(parts) if parts else "-"


def text_report(report: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(" SMELLNET - neural code sniff report")
    lines.append("=" * 72)
    s = report["summary"]
    lines.append(f"files scanned       : {report['files_scanned']}")
    lines.append(f"files with smells   : {report['files_with_smells']}")
    lines.append(f"total smells found  : {report['total_smells']}")
    if s:
        lines.append("by kind:")
        for lbl in sorted(s):
            lines.append(f"  - {lbl}: {s[lbl]}")
    lines.append("")

    for fr in report["files"]:
        findings = fr["findings"]
        if not findings or not any(f["smells"] for f in findings):
            continue
        lines.append(f"\n### {fr['file']}")
        for f in findings:
            if not f["smells"]:
                continue
            smells = ", ".join(f"{s['label']} ({s['prob']:.2f})"
                               for s in f["smells"])
            lines.append(f"  {f['name']}  @ line {f['lineno']}")
            lines.append(f"    {smells}")
            ev = _evidence_text(f["evidence"])
            if ev != "-":
                lines.append(f"    evidence: {ev}")
            if f["magic_numbers"]:
                lines.append(f"    magic literals: {f['magic_numbers'][:6]}")
        for d in fr["duplicates"]:
            lines.append(f"  DUPLICATE ({d['sim']:.2f}): {d['a']['name']}@"
                         f"{d['a']['lineno']} ~ {d['b']['name']}@{d['b']['lineno']}")

    lines.append("")
    lines.append("hints:")
    for lbl in sorted(s):
        lines.append(f"  {lbl}: {SMELL_HINT[lbl]}")
    return "\n".join(lines)


def json_report(report: dict) -> str:
    return json.dumps(report, indent=2)


def html_report(report: dict) -> str:
    def finding_rows(fr: dict) -> str:
        rows = []
        for f in fr["findings"]:
            if not f["smells"]:
                continue
            cells = f"<td>{html_mod.escape(f['name'])}</td><td>{f['lineno']}</td>"
            badge = []
            for s in f["smells"]:
                color = SEVERITY.get(s["label"], "#333")
                badge.append(
                    f"<span class='badge' style='background:{color}'>"
                    f"{s['label']} <b>{s['prob']:.2f}</b></span>")
            cells += f"<td>{''.join(badge)}</td>"
            ev = _evidence_text(f["evidence"])
            cells += f"<td>{html_mod.escape(ev)}</td>"
            rows.append(f"<tr>{cells}</tr>")
        for d in fr["duplicates"]:
            rows.append(
                f"<tr><td>-</td><td>-</td>"
                f"<td><span class='badge' style='background:{SEVERITY['duplication']}'>"
                f"duplication <b>{d['sim']:.2f}</b></span> "
                f"{html_mod.escape(d['a']['name'])} @{d['a']['lineno']} ~ "
                f"{html_mod.escape(d['b']['name'])} @{d['b']['lineno']}</td>"
                f"<td>near-identical bodies</td></tr>")
        return "\n".join(rows)

    files_html = []
    for fr in report["files"]:
        rows = finding_rows(fr)
        if not rows:
            continue
        files_html.append(
            f"<h3>{html_mod.escape(fr['file'])}</h3>\n"
            f"<table><thead><tr><th>function</th><th>line</th>"
            f"<th>smells</th><th>evidence</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")

    summary_items = "".join(
        f"<li><span class='badge' style='background:{SEVERITY.get(k, '#333')}'>"
        f"{k}</span> x {v}</li>" for k, v in sorted(report["summary"].items()))
    doc_body = "\n".join(files_html) if files_html else "<p>No smells detected.</p>"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>SmellNet report</title>
<style>
 body {{ font-family: ui-monospace, Menlo, Consolas, monospace; margin: 24px; color: #1d2433; }}
 h1 {{ border-bottom: 3px solid #1d2433; padding-bottom: 6px; }}
 h3 {{ margin-top: 28px; background:#eef1f6; padding:6px 10px; border-radius:4px; }}
 table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
 th {{ text-align: left; border-bottom: 2px solid #bbb; padding: 6px 10px; }}
 td {{ border-bottom: 1px solid #e0e4ec; padding: 8px 10px; vertical-align: top; }}
 .badge {{ display:inline-block; padding: 2px 8px; margin: 1px 2px 1px 0;
          color:#fff; border-radius: 10px; font-size: 12px; }}
 ul {{ list-style:none; padding: 0; }} li {{ display:inline-block; margin-right: 12px; }}
 .stats {{ color:#445; }}
</style></head><body>
<h1>&#128026; SmellNet &#8212; neural code sniff report</h1>
<p class="stats">files scanned: {report['files_scanned']} &middot;
files with smells: {report['files_with_smells']} &middot;
total smells: {report['total_smells']}</p>
<ul>{summary_items}</ul>
{doc_body}
</body></html>"""


def write(path: str, report: dict, fmt: str) -> None:
    fmt = fmt.lower()
    if fmt == "text":
        content = text_report(report)
        ext = ".txt"
    elif fmt == "json":
        content = json_report(report)
        ext = ".json"
    elif fmt == "html":
        content = html_report(report)
        ext = ".html"
    else:
        raise ValueError(f"unknown format {fmt}")
    if path:
        out = path if path != "-" else ""
    else:
        out = ""
    if out.endswith((".txt", ".json", ".html")):
        pass
    elif out:
        out = out + ext
    if out:
        import io
        with io.open(out, "w", encoding="utf-8") as fh:
            fh.write(content)
        print(f"report written -> {out}")
    else:
        print(content)
