import json
import subprocess
import sys
from pathlib import Path

import pytest

import agent as cli
from conftest import FAKE_AWS_KEY, FAKE_PASSWORD, ScriptedLLM, act
from llm_client import LLMConfigError, build_llm, load_env_file
from report import validate_report
from workspace import CloneError, WorkspaceError, clone_repository, parse_github_url

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("url,slug", [
    ("https://github.com/owner/repo", "owner/repo"),
    ("https://github.com/owner/repo.git", "owner/repo"),
    ("https://github.com/owner/repo/", "owner/repo"),
    ("http://github.com/Owner-1/my.repo_x", "Owner-1/my.repo_x"),
    ("github.com/owner/repo", "owner/repo"),
])
def test_parse_github_url_valid(url, slug):
    assert parse_github_url(url).slug == slug


def test_parse_github_url_branch():
    ref = parse_github_url("https://github.com/o/r/tree/dev")
    assert ref.branch == "dev" and ref.clone_url == "https://github.com/o/r.git"


@pytest.mark.parametrize("url", [
    "", "   ", "not a url", "https://gitlab.com/o/r", "https://github.com/owner", "https://github.com/",
    "https://token@github.com/o/r", "https://user:pw@github.com/o/r", "ftp://github.com/o/r",
    "https://github.com/o/r/issues/1", "https://github.com/-bad/repo", "git@github.com:o/r.git",
])
def test_parse_github_url_invalid(url):
    with pytest.raises(WorkspaceError):
        parse_github_url(url)


def test_clone_local_repo_and_cleanup(sample_repo):
    ref = parse_github_url(str(sample_repo), allow_local=True)
    co = clone_repository(ref, depth=1, timeout=60)
    try:
        assert (co.root / "README.md").exists() and co.branch == "main" and len(co.head) == 40
    finally:
        co.cleanup()
    assert not co.tmp_dir.exists()


def test_clone_failure_is_reported(tmp_path):
    bogus = parse_github_url(str(tmp_path), allow_local=True)
    with pytest.raises(CloneError):
        clone_repository(bogus, timeout=30)


def run_cli(*args, env_extra=None):
    import os
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY") and not k.startswith("LLM_")}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(ROOT / "agent.py"), *args], capture_output=True, text=True,
                          env=env, cwd=ROOT / "tests", timeout=120)


def test_cli_rejects_invalid_url_nonzero():
    res = run_cli("--repo", "https://example.com/x", "--output", "out.json", env_extra={"ANTHROPIC_API_KEY": "k"})
    assert res.returncode == 2 and "error:" in res.stderr


def test_cli_without_credentials_exits_nonzero_with_help():
    res = run_cli("--repo", "https://github.com/o/r", "--output", "/tmp/never.json")
    assert res.returncode == 2 and "API_KEY" in res.stderr
    assert not Path("/tmp/never.json").exists()


def test_cli_requires_repo_argument():
    assert run_cli("--output", "x.json").returncode == 2


def test_build_llm_provider_selection(monkeypatch):
    for var in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY", "LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(LLMConfigError, match="DEEPSEEK_API_KEY"):
        build_llm()

    monkeypatch.setenv("LLM_API_KEY", "k")
    client = build_llm()
    assert type(client).__name__ == "OpenAIClient" and client.model == "deepseek-flash"
    assert str(client._client.base_url).startswith("https://api.deepseek.com")

    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert build_llm().model == "gpt-4o"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert type(build_llm()).__name__ == "AnthropicClient"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    assert build_llm().model == "deepseek-flash"

    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-pro")
    assert build_llm().model == "deepseek-v4-pro"
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(LLMConfigError):
        build_llm()


def test_env_file_loader_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\nFOO_TEST=bar\nexport BAZ_TEST="q u"\nKEEP_TEST=file\n')
    monkeypatch.setenv("KEEP_TEST", "real")
    monkeypatch.delenv("FOO_TEST", raising=False)
    monkeypatch.delenv("BAZ_TEST", raising=False)
    assert load_env_file(f)
    import os
    assert os.environ["FOO_TEST"] == "bar" and os.environ["BAZ_TEST"] == "q u" and os.environ["KEEP_TEST"] == "real"
    monkeypatch.delenv("FOO_TEST"); monkeypatch.delenv("BAZ_TEST")


def _args(repo, out, **kw):
    ns = cli.parse_args(["--repo", str(repo), "--output", str(out), "--allow-local", "--max-steps", str(kw.pop("steps", 10))])
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_end_to_end_with_fake_llm(sample_repo, tmp_path):
    finding = {
        "category": "security", "severity": "critical", "rule_id": "SEC-001", "file": "config.py", "line": 1,
        "quote": "DB_PASSWORD =", "description": "Hardcoded database password",
        "evidence": f"DB_PASSWORD is a literal ({FAKE_PASSWORD[:3]}...)", "recommendation": "Use environment variables",
    }
    llm = ScriptedLLM([
        act("orient", ("list_files", {}), ("git_status", {})),
        act("look at config + workflow", ("read_file", {"path": "config.py"}),
            ("parse_workflow", {"path": ".github/workflows/ci.yml"})),
        act("record", ("record_finding", finding)),
        act("more", ("search_code", {"pattern": "TODO"}), ("git_log", {})),
        act("finish", ("finish", {"summary": "done"})),
        act("finish after review", ("finish", {"summary": "done"})),
    ])
    out = tmp_path / "out" / "report.json"
    before = subprocess.run(["git", "status", "--porcelain"], cwd=sample_repo, capture_output=True, text=True).stdout
    rc = cli.run_audit(_args(sample_repo, out), llm_factory=lambda: llm)
    assert rc == 0
    report = json.loads(out.read_text())
    assert validate_report(report) == []
    assert len(report["violations"]) == 1 and report["violations"][0]["line"] == 4
    raw = out.read_text() + (tmp_path / "out" / "report.trace.jsonl").read_text()
    assert FAKE_PASSWORD not in raw and FAKE_AWS_KEY not in raw
    after = subprocess.run(["git", "status", "--porcelain"], cwd=sample_repo, capture_output=True, text=True).stdout
    assert before == after == ""
    trace = [json.loads(l) for l in (tmp_path / "out" / "report.trace.jsonl").read_text().splitlines()]
    assert any(e["type"] == "integrity" and e["unchanged"] for e in trace)


def test_empty_violations_is_valid_report(sample_repo, tmp_path):
    llm = ScriptedLLM([
        act("a", ("list_files", {}), ("git_status", {}), ("read_file", {"path": "README.md"}), ("git_log", {})),
        act("b", ("finish", {})), act("c", ("finish", {})),
    ])
    out = tmp_path / "r.json"
    assert cli.run_audit(_args(sample_repo, out, no_trace=True), llm_factory=lambda: llm) == 0
    assert json.loads(out.read_text()) == {"repository": str(sample_repo), "violations": []}
    assert not (tmp_path / "r.trace.jsonl").exists()


def test_llm_failure_exits_4_and_writes_no_report(sample_repo, tmp_path):
    from llm_client import LLMError

    class Down:
        model = "x"

        def complete(self, *a, **k):
            raise LLMError("service unavailable")

    out = tmp_path / "r.json"
    assert cli.run_audit(_args(sample_repo, out), llm_factory=lambda: Down()) == cli.EXIT_LLM
    assert not out.exists()


def test_clone_failure_exits_3(tmp_path):
    out = tmp_path / "r.json"
    args = _args(tmp_path, out)
    assert cli.run_audit(args, llm_factory=lambda: ScriptedLLM([])) == cli.EXIT_CLONE
    assert not out.exists()
