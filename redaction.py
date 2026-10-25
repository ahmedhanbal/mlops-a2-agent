"""Secret redaction applied to everything leaving the tool layer."""

from __future__ import annotations

import re
from typing import Any, Iterable, List


_PLACEHOLDER_MARKERS = (
    "example", "placeholder", "your_", "your-", "yourkey", "<", ">", "${", "{{",
    "%(", "os.environ", "getenv", "process.env", "secrets.", "[redacted", "xxxx",
    "****", "changeme", "change_me", "change-me", "dummy", "replace", "insert_",
    "todo", "fixme", "env(", "env[", "sample",
)

_BENIGN_VALUES = {
    "none", "null", "nil", "true", "false", "bearer", "basic", "string", "required",
    "optional", "token", "secret", "password", "username", "undefined", "empty",
    "enabled", "disabled", "default", "auto",
}


def _looks_like_placeholder(value: str) -> bool:
    low = value.lower()
    if low in _BENIGN_VALUES:
        return True
    return any(marker in low for marker in _PLACEHOLDER_MARKERS)


def _is_code_reference(value: str) -> bool:
    """True for unquoted values that are expressions, not literals."""
    if re.search(r"[()\[\]{}$<>]", value):
        return True
    if re.fullmatch(r"[A-Za-z_]+(\.[A-Za-z_]+)+", value):
        return True
    if re.fullmatch(r"[a-z_]+", value) and "_" in value:
        return True
    return False


_TOKEN_PATTERNS = [
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[0-9A-Z]{16}\b")),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("slack_webhook", re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]{20,}")),
    ("discord_webhook", re.compile(r"https://discord(?:app)?\.com/api/webhooks/\d+/[A-Za-z0-9_-]{20,}")),
    ("stripe_key", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("llm_api_key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
]

_URL_CRED = re.compile(
    r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)(?P<user>[^/\s:@'\"]+):(?P<pw>[^/\s@'\"]+)@"
)

_KEYWORDS = (
    r"(?:api[_-]?key|apikey|secret|token|passwd|password|pwd|private[_-]?key|"
    r"access[_-]?key|auth[_-]?key|credential|client[_-]?secret)"
)
_ASSIGN = re.compile(
    r"(?P<key>[A-Za-z0-9_.-]*" + _KEYWORDS + r"[A-Za-z0-9_.-]*)"
    r"(?P<sep>[\"']?\s*(?::=|=>|==|:|=)\s*)"
    r"(?:\"(?P<dq>[^\"\n]{6,})\"|'(?P<sq>[^'\n]{6,})'|(?P<raw>[^\s\"'#,;]{8,}))",
    re.IGNORECASE,
)

_PEM_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")
_PEM_END = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")


def _tag(kind: str, value: str) -> str:
    return f"[REDACTED:{kind} len={len(value)}]"


def _redact_generic(match: "re.Match[str]") -> str:
    key, sep = match.group("key"), match.group("sep")
    if match.group("dq") is not None:
        value, quote = match.group("dq"), '"'
    elif match.group("sq") is not None:
        value, quote = match.group("sq"), "'"
    else:
        value, quote = match.group("raw"), ""

    if _looks_like_placeholder(value):
        return match.group(0)
    if quote:
        if " " in value or re.match(r"https?://[^@\s]*$", value):
            return match.group(0)
    elif _is_code_reference(value):
        return match.group(0)
    return f"{key}{sep}{quote}{_tag('secret', value)}{quote}"


def _redact_url_cred(match: "re.Match[str]") -> str:
    pw = match.group("pw")
    if _looks_like_placeholder(pw) or _is_code_reference(pw):
        return match.group(0)
    return f"{match.group('scheme')}{match.group('user')}:{_tag('url_password', pw)}@"


def _redact_line(line: str) -> str:
    if not line:
        return line
    for kind, pattern in _TOKEN_PATTERNS:
        line = pattern.sub(lambda m, k=kind: _tag(k, m.group(0)), line)
    line = _URL_CRED.sub(_redact_url_cred, line)
    line = _ASSIGN.sub(_redact_generic, line)
    return line


def redact_lines(lines: Iterable[str]) -> List[str]:
    """Redact a sequence of lines, preserving the line count exactly."""
    out: List[str] = []
    in_key = False
    for line in lines:
        if in_key:
            if _PEM_END.search(line):
                in_key = False
                out.append(line)
            else:
                out.append("[REDACTED:private_key_body]")
            continue
        begin = _PEM_BEGIN.search(line)
        if begin:
            rest = line[begin.end():]
            end = _PEM_END.search(rest)
            if end:
                out.append(line[: begin.end()] + "[REDACTED:private_key]" + rest[end.start():])
            else:
                in_key = True
                out.append(line[: begin.end()])
            continue
        out.append(_redact_line(line))
    return out


def redact_text(text: str) -> str:
    """Redact a (possibly multi-line) string; line count is preserved."""
    if not text:
        return text
    return "\n".join(redact_lines(text.split("\n")))


def contains_secret(text: str) -> bool:
    return redact_text(text) != text


def redact_obj(obj: Any) -> Any:
    """Recursively redact every string in a JSON-like structure."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {redact_text(str(k)): redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj
