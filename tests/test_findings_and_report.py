import json

import pytest

from conftest import FAKE_PASSWORD
from findings import FindingLedger, canonical_category
from report import ReportError, build_report, validate_report, write_report
from tools import RepoTools


@pytest.fixture()
def env(sample_repo):
    tools = RepoTools(sample_repo)
    return tools, FindingLedger(tools)


def finding(**over):
    base = {
        "category": "security", "severity": "critical", "rule_id": "SEC-001", "file": "config.py", "line": 4,
        "quote": 'DB_PASSWORD = "', "description": "Hardcoded database password",
        "evidence": "DB_PASSWORD is assigned a literal string", "recommendation": "Read it from the environment",
    }
    base.update(over)
    return base


def test_accepts_verified_finding_and_corrects_line(env):
    tools, ledger = env
    tools.call("read_file", {"path": "config.py"})
    res = ledger.add(finding(line=99, quote="DB_PASSWORD ="))
    assert res.ok, res.message
    v = ledger.violations()[0]
    assert v["file"] == "config.py" and v["line"] == 4 and v["rule_id"] == "SEC-001"


def test_rejects_uninspected_file(env):
    _, ledger = env
    res = ledger.add(finding())
    assert not res.ok and "not inspected" in res.message


def test_rejects_quote_not_in_file_and_missing_quote(env):
    tools, ledger = env
    tools.call("read_file", {"path": "config.py"})
    assert "not found verbatim" in ledger.add(finding(quote="SECRET_KEY = 'abc'")).message
    assert "'quote' is required" in ledger.add(finding(quote="")).message


def test_secret_never_stored_in_finding(env):
    tools, ledger = env
    tools.call("read_file", {"path": "config.py"})
    res = ledger.add(finding(evidence=f'DB_PASSWORD = "{FAKE_PASSWORD}" is committed'))
    assert res.ok
    assert FAKE_PASSWORD not in json.dumps(ledger.violations())


def test_absence_finding_with_nonexistent_file_nulls_file(env):
    _, ledger = env
    res = ledger.add({"category": "repository hygiene", "severity": "low", "rule_id": "REPO-001", "file": "LICENSE",
                      "line": 3, "description": "No license", "evidence": "list_files shows none",
                      "recommendation": "Add one"})
    assert res.ok
    v = ledger.violations()[0]
    assert v["file"] is None and v["line"] is None and "LICENSE" in v["evidence"]
    assert v["category"] == "repository_hygiene"


def test_invalid_severity_category_and_rule_id(env):
    tools, ledger = env
    assert not ledger.add(finding(severity="urgent")).ok
    assert not ledger.add(finding(category="vibes")).ok
    tools.call("read_file", {"path": "config.py"})
    assert ledger.add(finding(rule_id="weird")).ok
    assert ledger.violations()[0]["rule_id"].startswith("SEC-9")


def test_duplicates_rejected_and_retract_allows_replacement(env):
    tools, ledger = env
    tools.call("read_file", {"path": "config.py"})
    first = ledger.add(finding())
    assert first.ok
    dup = ledger.add(finding(line=5, quote='AWS_ACCESS_KEY_ID', description="Hardcoded database password!"))
    assert not dup.ok and "duplicate" in dup.message
    assert ledger.retract(first.finding_id, "redo").ok
    assert ledger.add(finding()).ok
    assert len(ledger.violations()) == 1


def test_violations_sorted_by_severity(env):
    tools, ledger = env
    tools.call("read_file", {"path": "config.py"})
    ledger.add(finding(severity="low", rule_id="CFG-001", category="configuration", quote="DEBUG = True",
                       description="Debug enabled"))
    ledger.add(finding())
    assert [v["severity"] for v in ledger.violations()] == ["critical", "low"]


def test_category_aliases():
    assert canonical_category("GitHub Actions / CI") == "github_actions_ci"
    assert canonical_category("Build and release") == "build_and_release"
    assert canonical_category("nonsense") is None


def _violation(**over):
    v = {"rule_id": "SEC-001", "category": "security", "severity": "high", "file": "a.py", "line": 1,
         "description": "d", "evidence": "e", "recommendation": "r"}
    v.update(over)
    return v


def test_validate_report_accepts_empty_and_valid():
    assert validate_report(build_report("https://github.com/o/r", [])) == []
    assert validate_report(build_report("https://github.com/o/r", [_violation(file=None, line=None)])) == []


@pytest.mark.parametrize("bad", [
    {"severity": "blocker"}, {"category": "x"}, {"line": 0}, {"line": "3"}, {"file": None, "line": 3},
    {"description": ""}, {"extra": 1},
])
def test_validate_report_rejects_bad_violation(bad):
    assert validate_report(build_report("u", [_violation(**bad)]))


def test_validate_report_rejects_missing_field():
    v = _violation()
    del v["recommendation"]
    assert validate_report(build_report("u", [v]))


def test_write_report_is_atomic_and_valid(tmp_path):
    out = tmp_path / "nested" / "report.json"
    write_report(out, build_report("https://github.com/o/r", [_violation()]))
    assert json.loads(out.read_text())["violations"][0]["rule_id"] == "SEC-001"
    assert [p.name for p in out.parent.iterdir()] == ["report.json"]
    with pytest.raises(ReportError):
        write_report(tmp_path / "bad.json", {"repository": "x", "violations": [{"nope": 1}]})
    assert not (tmp_path / "bad.json").exists()
