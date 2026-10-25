import json
import subprocess
import sys
from pathlib import Path

from scripts.release_notes import md, render, safe_repo_url

ROOT = Path(__file__).resolve().parent.parent


def v(sev="high", rule="CI-002", file="ci.yml", line=3, desc="Gate cannot fail", cat="github_actions_ci"):
    return {"rule_id": rule, "category": cat, "severity": sev, "file": file, "line": line,
            "description": desc, "evidence": "e", "recommendation": "r"}


def test_render_summary_counts_and_links():
    report = {"repository": "https://github.com/o/r", "violations": [v("critical"), v("high"), v("high", file=None, line=None)]}
    text = render(report, "https://github.com/me/agent/actions/runs/9", "ABCDEF1234567890zz")
    assert "[https://github.com/o/r](https://github.com/o/r)" in text
    assert "**Violations found: 3**" in text
    assert "| critical | 1 |" in text and "| high | 2 |" in text
    assert "ci.yml:3" in text and "repo-wide" in text
    assert "actions/runs/9" in text and "`abcdef1234`" in text
    assert "report.json" in text and "sample_report.json" in text


def test_render_empty_report():
    assert "No defensible DevOps violations" in render({"repository": "https://github.com/o/r", "violations": []})


def test_top_n_and_overflow_note():
    text = render({"repository": "https://github.com/o/r", "violations": [v(rule=f"CI-{i:03d}") for i in range(14)]})
    assert "Top 10 findings" in text and "4 more" in text


def test_untrusted_text_is_neutralised():
    nasty = "<img src=x onerror=1> @everyone @org/team | `code` [link](http://evil) *bold*"
    out = md(nasty)
    assert "<" not in out and "@​" in out and "@everyone" not in out
    assert "|" not in out.replace("\\|", "") and "`" not in out
    assert "[link]" not in out
    row = render({"repository": "https://github.com/o/r", "violations": [v(desc=nasty, file="a|b.py")]})
    assert "<img" not in row and "@everyone" not in row


def test_non_github_repository_is_not_linked():
    assert safe_repo_url("https://evil.example/x") == ""
    text = render({"repository": "javascript:alert(1)", "violations": []})
    assert "](" not in text.split("\n")[2]


def test_cli_roundtrip_and_bad_input(tmp_path):
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps({"repository": "https://github.com/o/r", "violations": [v()]}))
    out = tmp_path / "n.md"
    ok = subprocess.run([sys.executable, str(ROOT / "scripts" / "release_notes.py"), str(rep), "--output", str(out)])
    assert ok.returncode == 0 and "Violations found: 1" in out.read_text()
    rep.write_text("not json")
    bad = subprocess.run([sys.executable, str(ROOT / "scripts" / "release_notes.py"), str(rep), "--output", str(out)],
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "cannot read report" in bad.stderr
