import os
import shutil

import pytest

from conftest import FAKE_AWS_KEY, FAKE_PASSWORD
from tools import RepoTools, TOOL_SPECS


@pytest.fixture()
def tools(sample_repo):
    return RepoTools(sample_repo)


def test_required_tools_registered():
    for name in ("list_files", "read_file", "search_code", "git_log"):
        assert name in TOOL_SPECS


def test_list_files_shows_tracked_and_summary(tools):
    res = tools.call("list_files")
    assert res.ok
    assert "Tracked files:" in res.text and "config.py" in res.text and ".github/workflows/ci.yml" in res.text
    sub = tools.call("list_files", {"path_prefix": ".github/"})
    assert "ci.yml" in sub.text and "config.py" not in sub.text


def test_read_file_numbers_lines_and_redacts(tools):
    res = tools.call("read_file", {"path": "config.py"})
    assert res.ok
    assert FAKE_AWS_KEY not in res.text and FAKE_PASSWORD not in res.text
    assert "[REDACTED" in res.text and "    3| " in res.text
    assert "config.py" in tools.seen_files


def test_read_file_window_and_paging_hint(tools, sample_repo):
    big = "\n".join(f"line {i}" for i in range(1, 801))
    (sample_repo / "big.txt").write_text(big)
    res = tools.call("read_file", {"path": "big.txt", "start_line": 10, "end_line": 12})
    assert "lines 10-12 of 800" in res.text and "   10| line 10" in res.text
    first = tools.call("read_file", {"path": "big.txt"})
    assert "start_line=301" in first.text


@pytest.mark.parametrize("bad", ["../etc/passwd", "/etc/passwd", "a/../../x", ".git/config", "C:/Windows/win.ini"])
def test_path_traversal_blocked(tools, bad):
    res = tools.call("read_file", {"path": bad})
    assert not res.ok and res.text.startswith("ERROR")


def test_symlink_escape_blocked(tools, sample_repo, tmp_path):
    secret = tmp_path / "outside.txt"
    secret.write_text("top secret")
    os.symlink(secret, sample_repo / "link.txt")
    res = tools.call("read_file", {"path": "link.txt"})
    assert not res.ok and "outside" in res.text


def test_missing_file_suggests_alternatives(tools):
    res = tools.call("read_file", {"path": "confg.py"})
    assert not res.ok and "config.py" in res.text


def test_binary_file_not_displayed(tools, sample_repo):
    (sample_repo / "blob.bin").write_bytes(b"\x00\x01\x02" * 100)
    res = tools.call("read_file", {"path": "blob.bin"})
    assert not res.ok and "binary" in res.text


def test_search_code_finds_and_redacts(tools):
    res = tools.call("search_code", {"pattern": r"password\s*=", "path_glob": "*.py"})
    assert res.ok and "config.py:4:" in res.text
    assert FAKE_PASSWORD not in res.text
    assert "config.py" in tools.seen_files


def test_search_code_no_match_and_bad_regex(tools):
    assert "no matches" in tools.call("search_code", {"pattern": "zzz_not_there"}).text
    bad = tools.call("search_code", {"pattern": "("})
    assert not bad.ok and "invalid regular expression" in bad.text


def test_search_fixed_string_and_context(tools):
    res = tools.call("search_code", {"pattern": "DEBUG = True", "fixed_string": True, "context_lines": 1})
    assert "config.py:2:" in res.text and "config.py-1-" in res.text


def test_git_log_and_status(tools):
    log = tools.call("git_log", {"max_count": 5})
    assert log.ok and "initial commit" in log.text and "docs: expand README" in log.text
    filt = tools.call("git_log", {"message_grep": "docs"})
    assert "initial commit" not in filt.text
    files = tools.call("git_log", {"show_files": True, "max_count": 1})
    assert "README.md" in files.text
    status = tools.call("git_status")
    assert "main" in status.text and "working tree: clean" in status.text


def test_tools_leave_repo_untouched(tools, sample_repo):
    import subprocess
    def snap():
        return subprocess.run(["git", "status", "--porcelain"], cwd=sample_repo, capture_output=True, text=True).stdout
    index_before = (sample_repo / ".git" / "index").stat().st_mtime_ns
    for name, args in [("list_files", {}), ("read_file", {"path": "app.py"}), ("search_code", {"pattern": "def"}),
                       ("git_log", {}), ("git_status", {}), ("parse_workflow", {"path": ".github/workflows/ci.yml"}),
                       ("run_ruff", {})]:
        tools.call(name, args)
    # check the index timestamp first: the test's own plain `git status` below may refresh it
    assert (sample_repo / ".git" / "index").stat().st_mtime_ns == index_before
    assert snap() == ""


def test_parse_workflow_signals(tools):
    res = tools.call("parse_workflow", {"path": ".github/workflows/ci.yml"})
    assert res.ok
    text = res.text
    assert "does NOT include the repository default branch 'main'" in text
    assert "write-all" in text
    assert "mutable branch ref" in text and "some-org/setup-thing@main" in text
    assert "continue-on-error" in text
    assert "masks failures" in text
    assert "github.event.pull_request.title" in text
    assert "commented-out" in text
    assert "test: test@" in text.replace("job ", "") or "test:" in text


def test_parse_workflow_invalid_yaml_and_non_workflow(tools, sample_repo):
    (sample_repo / "broken.yml").write_text("on: [push\njobs: {")
    assert "YAML PARSE ERROR" in tools.call("parse_workflow", {"path": "broken.yml"}).text
    assert "not a GitHub Actions workflow" in tools.call("parse_workflow", {"path": "README.md"}).text


def test_argument_validation(tools):
    assert not tools.call("read_file", {}).ok
    assert not tools.call("read_file", {"path": "a", "bogus": 1}).ok
    assert not tools.call("nope", {}).ok
    assert tools.call("list_files", {"max_results": "5"}).ok


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
def test_run_ruff_finds_undefined_name(tools):
    res = tools.call("run_ruff")
    assert res.ok and "app.py" in res.text and "undefined_name" in res.text
