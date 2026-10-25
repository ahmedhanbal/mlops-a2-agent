"""Read-only inspection tools exposed to the auditing model."""

from __future__ import annotations

import difflib
import fnmatch
import os
import posixpath
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from redaction import redact_lines, redact_text

try:
    import regex as _regex
except ImportError:  # pragma: no cover - fallback when the optional dep is missing
    _regex = None

MAX_OUTPUT_CHARS = 14_000
MAX_FILE_BYTES = 3_000_000
MAX_LINE_CHARS = 500
DEFAULT_READ_LINES = 300
MAX_READ_LINES = 500
SEARCH_TIME_BUDGET_S = 25.0


class ToolError(Exception):
    """A tool could not complete; the message is shown to the LLM."""


@dataclass
class ToolResult:
    ok: bool
    text: str
    truncated: bool = False


@dataclass(frozen=True)
class ParamSpec:
    type: type
    description: str
    required: bool = False
    default: Any = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    summary: str
    params: Dict[str, ParamSpec] = field(default_factory=dict)


TOOL_SPECS: Dict[str, ToolSpec] = {
    "list_files": ToolSpec(
        "list_files",
        "List files tracked by git (path, size) plus a top-level summary. Use path_prefix to drill into a directory.",
        {
            "path_prefix": ParamSpec(str, "only list paths starting with this prefix, e.g. '.github/'", default=""),
            "max_results": ParamSpec(int, "max entries to print (default 400, max 1000)", default=400),
        },
    ),
    "read_file": ToolSpec(
        "read_file",
        "Read a text file with line numbers (secrets are redacted). Default window is 300 lines; page with start_line/end_line.",
        {
            "path": ParamSpec(str, "repo-relative file path", required=True),
            "start_line": ParamSpec(int, "1-based first line (default 1)", default=1),
            "end_line": ParamSpec(int, "1-based last line (default start+299, max window 500 lines)", default=None),
        },
    ),
    "search_code": ToolSpec(
        "search_code",
        "Regex search over tracked text files. Returns path:line: text (redacted). Use to hunt suspicious patterns across the repo.",
        {
            "pattern": ParamSpec(str, "regular expression (Python/PCRE-like syntax)", required=True),
            "path_glob": ParamSpec(str, "optional glob(s), comma separated, e.g. '*.py,*.yml' or 'src/**'", default=None),
            "ignore_case": ParamSpec(bool, "case-insensitive (default true)", default=True),
            "fixed_string": ParamSpec(bool, "treat pattern as a literal string (default false)", default=False),
            "max_results": ParamSpec(int, "max matches to print (default 60, max 200)", default=60),
            "context_lines": ParamSpec(int, "lines of context around each match, 0-3 (default 0)", default=0),
        },
    ),
    "git_log": ToolSpec(
        "git_log",
        "Show commit history (hash, date, author, subject). Optionally filter by path or commit-message text, or list changed files.",
        {
            "max_count": ParamSpec(int, "number of commits (default 25, max 100)", default=25),
            "path": ParamSpec(str, "only commits touching this path", default=None),
            "message_grep": ParamSpec(str, "only commits whose message matches this text", default=None),
            "show_files": ParamSpec(bool, "also list files changed per commit (default false)", default=False),
        },
    ),
    "git_status": ToolSpec(
        "git_status",
        "Repository facts: checked-out (default) branch, HEAD commit, working-tree cleanliness, shallow-clone flag, tracked file count.",
    ),
    "parse_workflow": ToolSpec(
        "parse_workflow",
        "Parse a GitHub Actions workflow YAML into a structured summary with line numbers (triggers, permissions, jobs, steps) "
        "plus mechanical 'signals' (facts such as mutable action refs, branch filters vs the default branch, continue-on-error, "
        "test/lint/build/deploy commands). Signals are observations, not verdicts. YAML comments are not shown; read_file shows them.",
        {"path": ParamSpec(str, "path of a workflow file, e.g. .github/workflows/ci.yml", required=True)},
    ),
    "run_ruff": ToolSpec(
        "run_ruff",
        "Static analysis only (nothing is executed): ruff check for Python syntax errors and undefined names (E9,F63,F7,F82). "
        "Reports 'unavailable' if ruff is not installed.",
        {"path": ParamSpec(str, "file or directory to check (default: whole repo)", default=None)},
    ),
}

INSPECTION_TOOLS = tuple(TOOL_SPECS)


def render_tool_docs() -> str:
    lines = []
    for spec in TOOL_SPECS.values():
        if spec.params:
            params = ", ".join(
                f"{n}{'' if p.required else '?'}: {p.type.__name__}" for n, p in spec.params.items()
            )
        else:
            params = ""
        lines.append(f"- {spec.name}({params}) - {spec.summary}")
        for n, p in spec.params.items():
            lines.append(f"    · {n}{' (required)' if p.required else ''}: {p.description}")
    return "\n".join(lines)


def _coerce(name: str, param: ParamSpec, value: Any) -> Any:
    if value is None:
        return param.default
    try:
        if param.type is bool:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y"}
            return bool(value)
        if param.type is int:
            if isinstance(value, bool):
                raise ValueError
            return int(value)
        if param.type is str:
            if not isinstance(value, (str, int, float)):
                raise ValueError
            return str(value)
    except (ValueError, TypeError):
        raise ToolError(f"argument '{name}' must be {param.type.__name__}, got {value!r}")
    return value


def validate_args(spec: ToolSpec, args: Any) -> Dict[str, Any]:
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolError(f"'args' for {spec.name} must be a JSON object")
    unknown = set(args) - set(spec.params)
    if unknown:
        raise ToolError(
            f"unknown argument(s) {sorted(unknown)} for {spec.name}; valid: {sorted(spec.params) or 'none'}"
        )
    clean: Dict[str, Any] = {}
    for name, param in spec.params.items():
        if name not in args or args[name] is None:
            if param.required:
                raise ToolError(f"missing required argument '{name}' for {spec.name}")
            clean[name] = param.default
        else:
            clean[name] = _coerce(name, param, args[name])
    return clean


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> Tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    cut = text[:limit].rsplit("\n", 1)[0]
    return cut + f"\n... [output truncated at {limit} chars; narrow the request]", True


def _short(value: Any, limit: int = 110) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


_TEST_CMD = re.compile(
    r"\b(pytest|py\.test|unittest|nosetests|tox|nox|npm\s+(?:run\s+)?test|yarn\s+test|pnpm\s+(?:run\s+)?test|"
    r"jest|vitest|mocha|go\s+test|cargo\s+test|mvn\b[^\n]*\btest|gradle\w*\s[^\n]*test|ctest|rspec|"
    r"dotnet\s+test|make\s+test|phpunit|bundle\s+exec\s+rake\s+test)\b", re.I)
_LINT_CMD = re.compile(
    r"\b(ruff|flake8|pylint|black|isort|mypy|pyright|eslint|prettier|golangci-lint|go\s+vet|clippy|rustfmt|"
    r"cargo\s+fmt|pre-commit|shellcheck|hadolint|yamllint|rubocop|checkstyle|spotless)\b", re.I)
_BUILD_CMD = re.compile(
    r"\b(docker\s+build|docker/build-push-action|python\s+-m\s+build|pip\s+wheel|npm\s+run\s+build|yarn\s+build|"
    r"pnpm\s+(?:run\s+)?build|go\s+build|cargo\s+build|mvn\b[^\n]*\b(package|install|verify)|gradle\w*\s[^\n]*build|"
    r"make\b|cmake|tsc\b|poetry\s+build|hatch\s+build)\b", re.I)
_DEPLOY_CMD = re.compile(
    r"\b(twine\s+upload|npm\s+publish|docker\s+push|kubectl\s+(?:apply|rollout)|terraform\s+apply|helm\s+(?:upgrade|install)|"
    r"gh\s+release|aws\s+s3\s+sync|aws\s+deploy|gcloud\s+(?:app|run)\s+deploy|az\s+webapp|flyctl\s+deploy|"
    r"pypa/gh-action-pypi-publish|publish|deploy|release)\b", re.I)
_UNTRUSTED_EXPR = re.compile(
    r"\$\{\{\s*(github\.event\.(?:issue|pull_request|comment|review|review_comment|head_commit|commits|pages|discussion|"
    r"workflow_run)[\w.\[\]*-]*|github\.head_ref|github\.event\.inputs\.[\w-]+)\s*\}\}")
_CURL_PIPE = re.compile(r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z)?sh\b", re.I)


class RepoTools:
    """Inspection tools bound to one cloned repository."""

    def __init__(self, root: Path, git_timeout: float = 45.0):
        self.root = Path(root).resolve()
        self.git_timeout = git_timeout
        self.seen_files: set = set()
        self._tracked: Optional[List[str]] = None
        self._redacted_cache: Dict[str, List[str]] = {}
        self._handlers: Dict[str, Callable[..., str]] = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_code": self.search_code,
            "git_log": self.git_log,
            "git_status": self.git_status,
            "parse_workflow": self.parse_workflow,
            "run_ruff": self.run_ruff,
        }

    def call(self, name: str, args: Any = None) -> ToolResult:
        spec = TOOL_SPECS.get(name)
        if spec is None:
            return ToolResult(False, f"Unknown tool '{name}'. Valid tools: {', '.join(TOOL_SPECS)}")
        try:
            clean = validate_args(spec, args)
            text = self._handlers[name](**clean)
        except ToolError as exc:
            return ToolResult(False, f"ERROR: {redact_text(str(exc))}")
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"ERROR: {name} timed out; try a narrower request.")
        except Exception as exc:
            return ToolResult(False, f"ERROR: internal failure in {name}: {type(exc).__name__}: {redact_text(str(exc))[:200]}")
        text = redact_text(text)
        text, truncated = _clip(text)
        return ToolResult(True, text, truncated)

    def _git(self, *args: str, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", "GIT_CONFIG_NOSYSTEM": "1", "LC_ALL": "C"})
        cmd = ["git", "--no-pager", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
               "-c", "core.quotepath=off", *args]
        try:
            return subprocess.run(cmd, cwd=self.root, env=env, capture_output=True, text=True,
                                  errors="replace", timeout=timeout or self.git_timeout)
        except FileNotFoundError:
            raise ToolError("git is not installed")

    def tracked_files(self) -> List[str]:
        if self._tracked is None:
            files: List[str] = []
            res = self._git("ls-files", "-z")
            if res.returncode == 0 and res.stdout:
                files = sorted(f for f in res.stdout.split("\0") if f)
            if not files:
                for dirpath, dirnames, filenames in os.walk(self.root):
                    dirnames[:] = [d for d in dirnames if d != ".git"]
                    for fn in filenames:
                        rel = os.path.relpath(os.path.join(dirpath, fn), self.root)
                        files.append(rel.replace(os.sep, "/"))
                files.sort()
            self._tracked = files
        return self._tracked

    def _norm_rel(self, rel: Any) -> str:
        if not isinstance(rel, str) or not rel.strip():
            raise ToolError("a non-empty path is required")
        rel = rel.strip().replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        if rel.startswith("/") or re.match(r"^[A-Za-z]:/", rel):
            raise ToolError("absolute paths are not allowed; use a repo-relative path")
        rel = posixpath.normpath(rel)
        parts = rel.split("/")
        if ".." in parts:
            raise ToolError("path escapes the repository")
        if parts[0] == ".git":
            raise ToolError("the .git directory is not inspectable; use git_log / git_status")
        return rel

    def _resolve_file(self, rel: Any) -> Tuple[str, Path]:
        rel = self._norm_rel(rel)
        path = self.root / rel
        real = Path(os.path.realpath(path))
        try:
            real.relative_to(self.root)
        except ValueError:
            raise ToolError("path resolves outside the repository (symlink escape blocked)")
        if not path.exists() and not path.is_symlink():
            hint = difflib.get_close_matches(rel, self.tracked_files(), n=3, cutoff=0.55)
            extra = f" Did you mean: {', '.join(hint)}?" if hint else " Use list_files to discover paths."
            raise ToolError(f"file not found: {rel}.{extra}")
        if not real.is_file():
            if real.is_dir():
                raise ToolError(f"{rel} is a directory; use list_files with path_prefix='{rel}/'")
            raise ToolError(f"{rel} is not a regular file")
        return rel, real

    @staticmethod
    def _is_binary(path: Path) -> bool:
        with open(path, "rb") as fh:
            return b"\0" in fh.read(8192)

    def _redacted_lines(self, rel: str, real: Path) -> List[str]:
        """Whole-file redaction (stateful for PEM blocks), cached per file."""
        cached = self._redacted_cache.get(rel)
        if cached is not None:
            return cached
        size = real.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ToolError(f"{rel} is too large to display ({size} bytes); use search_code to look inside it")
        if self._is_binary(real):
            raise ToolError(f"{rel} is a binary file ({size} bytes); not displayed")
        text = real.read_text(encoding="utf-8", errors="replace")
        lines = redact_lines(text.split("\n"))
        if lines and lines[-1] == "":
            lines.pop()
        if len(self._redacted_cache) > 40:
            self._redacted_cache.pop(next(iter(self._redacted_cache)))
        self._redacted_cache[rel] = lines
        return lines

    def redacted_file_lines(self, rel: str) -> List[str]:
        """Public helper used by the findings ledger for quote verification."""
        rel, real = self._resolve_file(rel)
        return self._redacted_lines(rel, real)

    def default_branch(self) -> str:
        res = self._git("symbolic-ref", "--short", "refs/remotes/origin/HEAD")
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().split("/", 1)[-1]
        res = self._git("rev-parse", "--abbrev-ref", "HEAD")
        return res.stdout.strip() or "unknown"

    def list_files(self, path_prefix: str = "", max_results: int = 400) -> str:
        prefix = (path_prefix or "").strip().replace("\\", "/")
        while prefix.startswith("./"):
            prefix = prefix[2:]
        if prefix.startswith("/") or ".." in prefix.split("/"):
            raise ToolError("path_prefix must be repo-relative")
        max_results = max(1, min(int(max_results or 400), 1000))
        all_files = self.tracked_files()
        files = [f for f in all_files if f.startswith(prefix)] if prefix else all_files
        out: List[str] = []
        out.append(f"Tracked files: {len(all_files)} total" + (f", {len(files)} under '{prefix}'" if prefix else ""))

        if not prefix:
            top = Counter(f.split("/", 1)[0] + ("/" if "/" in f else "") for f in all_files)
            summary = ", ".join(f"{k} ({v})" if k.endswith("/") else k for k, v in sorted(top.items())[:80])
            out.append("Top level: " + summary)
            ext = Counter((os.path.splitext(f)[1].lower() or "(none)") for f in all_files)
            out.append("By extension: " + ", ".join(f"{k}:{v}" for k, v in ext.most_common(15)))
        out.append("")
        for f in files[:max_results]:
            full = self.root / f
            try:
                if full.is_symlink():
                    out.append(f"{f}  -> symlink ({os.readlink(full)})")
                else:
                    out.append(f"{f}  ({full.stat().st_size} B)")
            except OSError:
                out.append(f"{f}  (unreadable)")
        if len(files) > max_results:
            out.append(f"... {len(files) - max_results} more not shown; use path_prefix to narrow")
        if not files:
            out.append("(no files match)")
        return "\n".join(out)

    def read_file(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        rel, real = self._resolve_file(path)
        lines = self._redacted_lines(rel, real)
        total = len(lines)
        start = max(1, int(start_line or 1))
        end = int(end_line) if end_line else start + DEFAULT_READ_LINES - 1
        end = min(end, start + MAX_READ_LINES - 1, total)
        self.seen_files.add(rel)
        if total == 0:
            return f"{rel}: (empty file)"
        if start > total:
            raise ToolError(f"start_line {start} is beyond the end of {rel} ({total} lines)")
        body = []
        for n in range(start, end + 1):
            line = lines[n - 1]
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS] + f"...[+{len(line) - MAX_LINE_CHARS} chars]"
            body.append(f"{n:>5}| {line}")
        footer = f"[{rel}: lines {start}-{end} of {total}]"
        if end < total:
            footer += f" - call again with start_line={end + 1} for more"
        return footer + "\n" + "\n".join(body)

    def search_code(self, pattern: str, path_glob: Optional[str] = None, ignore_case: bool = True,
                    fixed_string: bool = False, max_results: int = 60, context_lines: int = 0) -> str:
        if not pattern or len(pattern) > 400:
            raise ToolError("pattern must be 1-400 characters")
        max_results = max(1, min(int(max_results or 60), 200))
        context_lines = max(0, min(int(context_lines or 0), 3))
        expr = re.escape(pattern) if fixed_string else pattern
        flags = (_regex.IGNORECASE if ignore_case else 0) if _regex else (re.IGNORECASE if ignore_case else 0)
        try:
            compiled = (_regex or re).compile(expr, flags)
        except Exception as exc:
            raise ToolError(f"invalid regular expression: {exc}")
        globs = [g.strip() for g in (path_glob or "").split(",") if g.strip()]

        def selected(path: str) -> bool:
            if not globs:
                return True
            base = path.rsplit("/", 1)[-1]
            return any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(base, g) for g in globs)

        deadline = time.monotonic() + SEARCH_TIME_BUDGET_S
        total_matches = 0
        files_hit = 0
        printed: List[str] = []
        timed_out = False
        for rel in self.tracked_files():
            if not selected(rel):
                continue
            if time.monotonic() > deadline:
                timed_out = True
                break
            real = self.root / rel
            try:
                if real.is_symlink() or not real.is_file() or real.stat().st_size > MAX_FILE_BYTES:
                    continue
                if self._is_binary(real):
                    continue
                raw = real.read_text(encoding="utf-8", errors="replace").split("\n")
            except OSError:
                continue
            hits: List[int] = []
            for idx, line in enumerate(raw):
                probe = line[:2000]
                try:
                    found = compiled.search(probe, timeout=0.25) if _regex else compiled.search(probe)
                except TimeoutError:
                    continue
                if found:
                    hits.append(idx)
                    if len(hits) >= 12:
                        break
            if not hits:
                continue
            files_hit += 1
            red = self._redacted_lines(rel, real) if len(raw) <= 60000 else redact_lines(raw)
            self.seen_files.add(rel)
            for idx in hits:
                total_matches += 1
                if len(printed) >= max_results * (1 + 2 * context_lines):
                    continue
                lo, hi = max(0, idx - context_lines), min(len(red) - 1, idx + context_lines)
                for j in range(lo, hi + 1):
                    text = red[j] if j < len(red) else ""
                    if len(text) > MAX_LINE_CHARS:
                        text = text[:MAX_LINE_CHARS] + "..."
                    sep = ":" if j == idx else "-"
                    printed.append(f"{rel}{sep}{j + 1}{sep} {text}")
                if context_lines:
                    printed.append("--")
        head = f"{total_matches} match(es) in {files_hit} file(s) for /{pattern}/"
        if globs:
            head += f" (glob: {', '.join(globs)})"
        if total_matches == 0:
            return head + "\n(no matches)" + ("\n[search time budget exhausted]" if timed_out else "")
        notes = []
        if timed_out:
            notes.append("search time budget exhausted; results incomplete")
        notes.append("up to 12 matches per file are reported")
        return head + "\n" + "\n".join(printed) + "\n[" + "; ".join(notes) + "]"

    def git_log(self, max_count: int = 25, path: Optional[str] = None, message_grep: Optional[str] = None,
                show_files: bool = False) -> str:
        max_count = max(1, min(int(max_count or 25), 100))
        args = ["log", f"-n{max_count}", "--date=short", "--pretty=format:@@%h%x09%ad%x09%an%x09%s"]
        if message_grep:
            args += ["-i", "--grep=" + message_grep[:200]]
        if show_files:
            args.append("--name-status")
        if path:
            args += ["--", self._norm_rel(path)]
        res = self._git(*args)
        if res.returncode != 0:
            raise ToolError("git log failed: " + (res.stderr.strip()[:200] or "unknown error"))
        total = self._git("rev-list", "--count", "HEAD").stdout.strip() or "?"
        shallow = self._git("rev-parse", "--is-shallow-repository").stdout.strip() == "true"

        commits: List[str] = []
        files_for_commit = 0
        for line in res.stdout.splitlines():
            if line.startswith("@@"):
                h, d, a, s = (line[2:].split("\t", 3) + ["", "", "", ""])[:4]
                commits.append(f"{h} {d} {a}: {s}")
                files_for_commit = 0
            elif line.strip() and show_files:
                files_for_commit += 1
                if files_for_commit <= 12:
                    commits.append("      " + line.replace("\t", " "))
                elif files_for_commit == 13:
                    commits.append("      ...")
        n_shown = sum(1 for c in commits if not c.startswith("      "))
        header = f"{n_shown} commit(s) shown; {total} reachable in this clone"
        if shallow:
            header += " (shallow clone: older history is not available)"
        if not commits:
            return header + "\n(no commits match)"
        return header + "\n" + "\n".join(commits)

    def git_status(self) -> str:
        branch = self._git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or "(unknown)"
        head = self._git("log", "-1", "--date=short", "--pretty=format:%h %ad %s").stdout.strip()
        remote = self._git("remote", "get-url", "origin").stdout.strip()
        changes = self._git("status", "--porcelain").stdout.strip().splitlines()
        shallow = self._git("rev-parse", "--is-shallow-repository").stdout.strip() == "true"
        lines = [
            f"checked-out branch: {branch}  (this is the remote's default branch unless a branch was requested)",
            f"default branch name: {self.default_branch()}",
            f"HEAD: {head or '(none)'}",
            f"remote origin: {remote or '(none)'}",
            f"working tree: {'clean' if not changes else str(len(changes)) + ' change(s): ' + '; '.join(changes[:10])}",
            f"shallow clone: {'yes' if shallow else 'no'}",
            f"tracked files: {len(self.tracked_files())}",
        ]
        return "\n".join(lines)

    def parse_workflow(self, path: str) -> str:
        rel, real = self._resolve_file(path)
        if real.stat().st_size > 400_000:
            raise ToolError(f"{rel} is too large to parse as a workflow")
        text = real.read_text(encoding="utf-8", errors="replace")
        raw_lines = text.split("\n")
        self.seen_files.add(rel)
        try:
            root_node = yaml.compose(text, Loader=yaml.SafeLoader)
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
            return f"{rel}: YAML PARSE ERROR{where}: {getattr(exc, 'problem', str(exc))}. GitHub would reject this workflow."
        if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict):
            return f"{rel}: not a GitHub Actions workflow (no top-level 'jobs' mapping). Use read_file to inspect it."

        def nmap(node) -> Dict[Any, Tuple[Any, Any]]:
            if isinstance(node, yaml.MappingNode):
                return {k.value: (k, v) for k, v in node.value if isinstance(k, yaml.ScalarNode)}
            return {}

        top = nmap(root_node)
        on_key = "on" if "on" in data else (True if True in data else None)
        on_val = data.get(on_key) if on_key is not None else None
        on_line = top["on"][0].start_mark.line + 1 if "on" in top else None
        default_branch = self.default_branch()
        signals: List[str] = []
        out: List[str] = [f"Workflow file: {rel}", f"name: {data.get('name', '(unnamed)')}"]

        triggers: Dict[str, Any] = {}
        if isinstance(on_val, str):
            triggers = {on_val: None}
        elif isinstance(on_val, list):
            triggers = {str(t): None for t in on_val}
        elif isinstance(on_val, dict):
            triggers = {str(k): v for k, v in on_val.items()}
        trig_txt = []
        for name, cfg in triggers.items():
            if isinstance(cfg, dict) and cfg:
                trig_txt.append(f"{name}(" + ", ".join(f"{k}={_short(v, 80)}" for k, v in cfg.items()) + ")")
            else:
                trig_txt.append(name)
        out.append(f"triggers (line {on_line}): " + ("; ".join(trig_txt) if trig_txt else "NONE FOUND"))
        auto = {"push", "pull_request", "pull_request_target"} & set(triggers)
        if not auto:
            signals.append("No push/pull_request/pull_request_target trigger: this workflow does not run automatically on code changes "
                           f"(triggers: {', '.join(triggers) or 'none'}).")
        if "pull_request_target" in triggers:
            signals.append("Trigger pull_request_target is present (runs with base-repo token/secrets; check what code it checks out/executes).")
        for tname in ("push", "pull_request", "pull_request_target"):
            cfg = triggers.get(tname)
            if isinstance(cfg, dict):
                branches = cfg.get("branches")
                if isinstance(branches, str):
                    branches = [branches]
                if isinstance(branches, list) and branches:
                    pos = [str(b) for b in branches if not str(b).startswith("!")]
                    if pos and not any(fnmatch.fnmatch(default_branch, p) for p in pos):
                        signals.append(f"{tname} branch filter {pos} does NOT include the repository default branch '{default_branch}'.")
                ignore = cfg.get("branches-ignore")
                if isinstance(ignore, list) and any(fnmatch.fnmatch(default_branch, str(p)) for p in ignore):
                    signals.append(f"{tname} branches-ignore excludes the default branch '{default_branch}'.")
                if cfg.get("paths-ignore"):
                    signals.append(f"{tname} has paths-ignore={_short(cfg['paths-ignore'], 100)} (these changes skip the workflow).")
                if cfg.get("paths"):
                    signals.append(f"{tname} is restricted to paths={_short(cfg['paths'], 100)}.")

        perm_node = top.get("permissions")
        wf_perm = data.get("permissions")
        out.append(f"workflow permissions{f' (line {perm_node[0].start_mark.line + 1})' if perm_node else ''}: "
                   f"{_short(wf_perm, 150) if 'permissions' in data else 'not set'}")
        env_keys = list((data.get("env") or {}).keys()) if isinstance(data.get("env"), dict) else []
        if env_keys:
            out.append(f"workflow env keys: {', '.join(map(str, env_keys))} (values hidden)")

        def perm_signals(where: str, perm: Any, line: Optional[int]) -> None:
            ln = f"line {line}: " if line else ""
            if perm == "write-all":
                signals.append(f"{ln}{where} permissions: write-all.")
            elif isinstance(perm, dict):
                writes = [k for k, v in perm.items() if str(v).lower() == "write"]
                if writes:
                    signals.append(f"{ln}{where} grants write permission(s): {', '.join(map(str, writes))}.")

        perm_signals("workflow", wf_perm, perm_node[0].start_mark.line + 1 if perm_node else None)

        jobs_node = nmap(top["jobs"][1]) if "jobs" in top else {}
        out.append("jobs:")
        found_cmds = {"test": [], "lint": [], "build": [], "deploy": []}
        any_job_perm = False
        for job_id, job in data["jobs"].items():
            jk = jobs_node.get(job_id, (None, None))
            jline = jk[0].start_mark.line + 1 if jk[0] is not None else None
            if not isinstance(job, dict):
                out.append(f"  - {job_id} (line {jline}): <not a mapping>")
                continue
            head = [f"  - {job_id} (line {jline})"]
            for key in ("name", "runs-on", "needs", "if", "environment", "timeout-minutes", "continue-on-error", "uses", "strategy"):
                if key in job:
                    head.append(f"{key}={_short(job[key], 90)}")
            out.append(" ".join(head))
            if "permissions" in job:
                any_job_perm = True
                out.append(f"      job permissions: {_short(job['permissions'], 120)}")
                perm_signals(f"job '{job_id}'", job["permissions"], jline)
            if str(job.get("runs-on", "")).find("self-hosted") != -1:
                signals.append(f"line {jline}: job '{job_id}' runs on a self-hosted runner.")
            if str(job.get("if", "")).strip().lower() in {"false", "${{ false }}", "0"}:
                signals.append(f"line {jline}: job '{job_id}' has a constant-false condition and never runs.")
            if job.get("continue-on-error") in (True, "true"):
                signals.append(f"line {jline}: job '{job_id}' sets continue-on-error: true (its failure cannot fail the workflow).")
            if job.get("secrets") == "inherit":
                signals.append(f"line {jline}: job '{job_id}' passes 'secrets: inherit' to a reusable workflow.")
            steps = job.get("steps") if isinstance(job.get("steps"), list) else []
            step_nodes = []
            if jk[1] is not None:
                sn = nmap(jk[1]).get("steps")
                if sn is not None and isinstance(sn[1], yaml.SequenceNode):
                    step_nodes = sn[1].value
            for i, step in enumerate(steps):
                sline = step_nodes[i].start_mark.line + 1 if i < len(step_nodes) else None
                if not isinstance(step, dict):
                    continue
                bits = []
                if step.get("name"):
                    bits.append(f"name={_short(step['name'], 60)}")
                uses = step.get("uses")
                if uses:
                    bits.append(f"uses={uses}")
                    ref = str(uses).split("@", 1)[1] if "@" in str(uses) else None
                    repo_part = str(uses).split("@", 1)[0]
                    if not str(uses).startswith("./") and not str(uses).startswith("docker://"):
                        first_party = repo_part.startswith(("actions/", "github/"))
                        if ref is None:
                            signals.append(f"line {sline}: action '{uses}' has no version ref.")
                        elif not re.fullmatch(r"[0-9a-f]{40}", ref):
                            mutable = ref in {"main", "master", "latest", "develop", "dev", "head"}
                            if mutable:
                                signals.append(f"line {sline}: action '{uses}' is pinned to a mutable branch ref "
                                               f"({'first' if first_party else 'third'}-party).")
                if isinstance(step.get("with"), dict) and step["with"]:
                    bits.append("with={" + ", ".join(f"{k}: {_short(v, 50)}" for k, v in step["with"].items()) + "}")
                    ref_val = str(step["with"].get("ref", ""))
                    if "pull_request_target" in triggers and re.search(r"pull_request\.head|head_ref|head\.sha", ref_val):
                        signals.append(f"line {sline}: checkout of the PR head ref ({_short(ref_val, 70)}) in a workflow triggered by "
                                       "pull_request_target (untrusted code with privileged token).")
                run = step.get("run")
                if run is not None:
                    run_s = str(run).strip()
                    first = run_s.split("\n", 1)[0]
                    extra = f" (+{run_s.count(chr(10))} more line(s))" if "\n" in run_s else ""
                    bits.append(f"run={_short(first, 150)}{extra}")
                    for kind, rx in (("test", _TEST_CMD), ("lint", _LINT_CMD), ("build", _BUILD_CMD), ("deploy", _DEPLOY_CMD)):
                        if rx.search(run_s):
                            found_cmds[kind].append(f"{job_id}@line {sline}")
                    for m in _UNTRUSTED_EXPR.finditer(run_s):
                        signals.append(f"line {sline}: run script interpolates untrusted-capable expression '${{{{ {m.group(1)} }}}}' directly.")
                    if re.search(r"\|\|\s*(true|:|exit 0)\b", run_s) or re.search(r"^\s*set \+e\b", run_s, re.M):
                        signals.append(f"line {sline}: run step masks failures ('|| true' / 'set +e').")
                    if _CURL_PIPE.search(run_s):
                        signals.append(f"line {sline}: run step pipes a download into a shell (curl|bash pattern).")
                    if re.search(r"\$\{\{\s*secrets\.[A-Za-z0-9_]+\s*\}\}", run_s):
                        signals.append(f"line {sline}: secret expanded inline in a run script (can leak via logs/shell).")
                    if re.search(r"\becho\b[^\n]*\$\{\{\s*secrets\.", run_s):
                        signals.append(f"line {sline}: a secret is echoed to the log.")
                if "if" in step:
                    bits.append(f"if={_short(step['if'], 70)}")
                    if str(step["if"]).strip().lower() in {"false", "${{ false }}", "0"}:
                        signals.append(f"line {sline}: step has a constant-false condition and never runs.")
                if step.get("continue-on-error") in (True, "true"):
                    bits.append("continue-on-error=true")
                    signals.append(f"line {sline}: step sets continue-on-error: true (failure is ignored).")
                if step.get("env") and isinstance(step["env"], dict):
                    bits.append("env keys=" + ",".join(map(str, step["env"])))
                out.append(f"      step {i + 1} (line {sline}): " + " | ".join(bits))
        if "permissions" not in data and not any_job_perm:
            out.append("(no 'permissions' block at workflow or job level)")

        commented = [i + 1 for i, l in enumerate(raw_lines)
                     if re.match(r"^\s*#\s*(-\s+)?(run|uses|name|if|with)\s*:", l)]
        if commented:
            signals.append(f"{len(commented)} commented-out YAML line(s) that look like workflow steps, e.g. lines "
                           f"{', '.join(map(str, commented[:8]))} (read_file to see what was disabled).")

        out.append("")
        out.append("command facts: " + "; ".join(
            f"{k}: {', '.join(v[:6]) if v else 'none'}" for k, v in found_cmds.items()))
        out.append("signals (mechanical observations, not verdicts):")
        out.extend(f"  * {s}" for s in signals) if signals else out.append("  (none)")
        return "\n".join(out)

    def run_ruff(self, path: Optional[str] = None) -> str:
        target = "."
        if path:
            rel = self._norm_rel(path)
            if not (self.root / rel).exists():
                raise ToolError(f"path not found: {rel}")
            target = rel
        exe = shutil.which("ruff")
        cmd = [exe] if exe else [sys.executable, "-m", "ruff"]
        cmd += ["check", "--isolated", "--no-cache", "--select", "E9,F63,F7,F82",
                "--output-format", "concise", "--exit-zero", "--", target]
        try:
            res = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=90,
                                 env={**os.environ, "RUFF_NO_CACHE": "1"})
        except FileNotFoundError:
            return "ruff is unavailable in this environment; no result (nothing was inferred)."
        if "No module named ruff" in res.stderr:
            return "ruff is unavailable in this environment; no result (nothing was inferred)."
        lines = [l for l in res.stdout.splitlines() if l.strip() and not l.startswith("Found ")]
        if res.returncode not in (0, 1) and not lines:
            return f"ruff could not run (exit {res.returncode}): {res.stderr.strip()[:300]}"
        if not lines:
            return f"ruff (E9,F63,F7,F82) on '{target}': no syntax errors or undefined names found."
        return f"ruff (E9,F63,F7,F82) on '{target}': {len(lines)} diagnostic(s)\n" + "\n".join(lines[:80])
