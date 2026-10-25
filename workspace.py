"""URL validation and hardened shallow clone of the repository under audit."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from redaction import redact_text


class WorkspaceError(Exception):
    """Invalid input (bad URL, missing git, ...)."""


class CloneError(Exception):
    """The repository could not be fetched."""


_GITHUB_RE = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9_.-]*))/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?(?:/tree/(?P<branch>[^/\s]+))?/?$"
)


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str
    branch: Optional[str]
    clone_url: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass
class RepoCheckout:
    root: Path
    tmp_dir: Path
    ref: RepoRef
    head: str
    branch: str

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


def parse_github_url(url: str, allow_local: bool = False) -> RepoRef:
    """Validate and normalise a public GitHub repository URL."""
    if not isinstance(url, str) or not url.strip():
        raise WorkspaceError("Repository URL is empty.")
    url = url.strip()

    if allow_local:
        local = Path(url).expanduser()
        if local.is_dir():
            resolved = local.resolve()
            return RepoRef("local", resolved.name, None, resolved.as_uri())

    if "@" in url.split("://", 1)[-1].split("/", 1)[0]:
        raise WorkspaceError("Credentials in the URL are not supported; only public repositories are audited.")
    if not url.startswith(("https://", "http://")):
        if url.startswith("github.com/"):
            url = "https://" + url
        else:
            raise WorkspaceError(
                f"Unsupported repository URL {redact_text(url)!r}. "
                "Expected https://github.com/<owner>/<repo>."
            )
    url = re.sub(r"^http://", "https://", url)
    m = _GITHUB_RE.match(url)
    if not m:
        raise WorkspaceError(
            f"Not a valid GitHub repository URL: {redact_text(url)!r}. "
            "Expected https://github.com/<owner>/<repo>."
        )
    owner, repo, branch = m.group("owner"), m.group("repo"), m.group("branch")
    if repo in {".", ".."} or owner in {".", ".."}:
        raise WorkspaceError("Invalid repository path.")
    return RepoRef(owner, repo, branch, f"https://github.com/{owner}/{repo}.git")


def _git_env() -> dict:
    env = os.environ.copy()
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "echo",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    return env


def _run_git(args: list, cwd: Optional[Path], timeout: float) -> subprocess.CompletedProcess:
    base = [
        "git", "--no-pager",
        "-c", "core.hooksPath=/dev/null",
        "-c", "core.fsmonitor=false",
        "-c", "protocol.ext.allow=never",
        "-c", "filter.lfs.required=false",
        "-c", "filter.lfs.smudge=",
        "-c", "filter.lfs.process=",
    ]
    proc = subprocess.Popen(
        base + args,
        cwd=str(cwd) if cwd else None,
        env=_git_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.communicate()
        raise
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def _classify_clone_failure(stderr: str) -> str:
    low = stderr.lower()
    if "repository not found" in low or "not found" in low or "could not read username" in low \
            or "authentication failed" in low or "terminal prompts disabled" in low:
        return "Repository not found, or it is private. Only public GitHub repositories are supported."
    if "could not resolve host" in low or "unable to access" in low or "connection" in low \
            or "timed out" in low or "network is unreachable" in low:
        return "Network error while contacting GitHub: " + redact_text(stderr.strip().splitlines()[-1] if stderr.strip() else "unknown")
    if "remote branch" in low and "not found" in low:
        return "The requested branch does not exist in the repository."
    return "git clone failed: " + redact_text(stderr.strip()[-500:] or "unknown error")


def clone_repository(ref: RepoRef, depth: int = 200, timeout: float = 240.0) -> RepoCheckout:
    """Shallow-clone ``ref`` into a fresh temp dir and return the checkout."""
    if shutil.which("git") is None:
        raise WorkspaceError("git is not installed or not on PATH.")
    tmp_dir = Path(tempfile.mkdtemp(prefix="devops-audit-"))
    root = tmp_dir / "repo"
    args = ["clone", "--quiet", "--no-tags", "--single-branch", "--no-recurse-submodules"]
    if depth and depth > 0:
        args += ["--depth", str(int(depth))]
    if ref.branch:
        args += ["--branch", ref.branch]
    args += ["--", ref.clone_url, str(root)]
    if ref.owner == "local":
        args = ["-c", "protocol.file.allow=always"] + args
    try:
        result = _run_git(args, cwd=tmp_dir, timeout=timeout)
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise CloneError(f"Timed out after {int(timeout)}s while cloning {ref.slug}.")
    except OSError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise CloneError(f"Could not run git: {exc}")
    if result.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise CloneError(_classify_clone_failure(result.stderr))

    head, branch = snapshot_head(root)
    return RepoCheckout(root=root, tmp_dir=tmp_dir, ref=ref, head=head, branch=branch)


def snapshot_head(root: Path) -> Tuple[str, str]:
    head = _run_git(["rev-parse", "HEAD"], cwd=root, timeout=30).stdout.strip()
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root, timeout=30).stdout.strip()
    return head, branch


def working_tree_changes(root: Path) -> str:
    """Return ``git status --porcelain`` output ('' when the tree is clean)."""
    return _run_git(["status", "--porcelain"], cwd=root, timeout=60).stdout.strip()
