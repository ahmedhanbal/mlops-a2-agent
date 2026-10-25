#!/usr/bin/env python3
"""Render GitHub Release notes from an audit report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

SEVERITIES = ("critical", "high", "medium", "low")
TOP_N = 10


def md(text: Any, limit: int = 160) -> str:
    """Make untrusted text safe for inline Markdown / table cells."""
    s = " ".join(str(text).split())
    if len(s) > limit:
        s = s[: limit - 1].rstrip() + "..."
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = s.replace("|", "\\|").replace("`", "'")
    s = re.sub(r"([\\*_\[\]~#])", r"\\\1", s)
    s = s.replace("@", "@​")
    return s


def safe_repo_url(url: str) -> str:
    return url if re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", url) else ""


def render(report: Dict[str, Any], run_url: str = "", agent_sha: str = "") -> str:
    violations: List[Dict[str, Any]] = report.get("violations", [])
    repo = str(report.get("repository", ""))
    link = safe_repo_url(repo.removesuffix("/").removesuffix(".git"))
    sev = Counter(v.get("severity") for v in violations)
    cat = Counter(v.get("category") for v in violations)

    lines = ["## DevOps audit report", ""]
    lines.append(f"**Repository:** [{md(repo, 200)}]({link})" if link else f"**Repository:** {md(repo, 200)}")
    if run_url:
        lines.append(f"**Workflow run:** {run_url}")
    if agent_sha:
        lines.append(f"**Agent version:** `{re.sub(r'[^0-9a-f]', '', agent_sha.lower())[:10]}`")
    lines += ["", f"**Violations found: {len(violations)}**", ""]

    if not violations:
        lines.append("No defensible DevOps violations were detected.")
    else:
        lines += ["| Severity | Count |", "|---|---|"]
        lines += [f"| {s} | {sev[s]} |" for s in SEVERITIES if sev[s]]
        lines += ["", "By category: " + ", ".join(f"{md(c)} ({n})" for c, n in sorted(cat.items(), key=lambda x: (-x[1], str(x[0]))))]
        lines += ["", f"### Top {min(TOP_N, len(violations))} findings", "",
                  "| Severity | Rule | Location | Description |", "|---|---|---|---|"]
        for v in violations[:TOP_N]:
            loc = v.get("file") or "repo-wide"
            if v.get("file") and v.get("line"):
                loc = f"{loc}:{v['line']}"
            lines.append(f"| {md(v.get('severity'))} | {md(v.get('rule_id'))} | {md(loc, 80)} | {md(v.get('description'))} |")
        if len(violations) > TOP_N:
            lines += ["", f"...and {len(violations) - TOP_N} more in the attached `report.json`."]

    lines += ["", "### Attached files", "",
              "- `report.json` - the full machine-readable report",
              "- `sample_report.json` - the same report, named as the assignment's sample output",
              "- `trace.jsonl` - redacted tool-call trace of the agent run", ""]
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report")
    p.add_argument("--output", required=True)
    p.add_argument("--run-url", default="")
    p.add_argument("--agent-sha", default="")
    args = p.parse_args(argv)
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        if not isinstance(report, dict) or not isinstance(report.get("violations"), list):
            raise ValueError("not an audit report")
    except (OSError, ValueError) as exc:
        print(f"error: cannot read report: {exc}", file=sys.stderr)
        return 2
    Path(args.output).write_text(render(report, args.run_url, args.agent_sha), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
