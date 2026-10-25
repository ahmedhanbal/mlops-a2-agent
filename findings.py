"""Findings ledger: validation, quote verification and de-duplication."""

from __future__ import annotations

import difflib
import posixpath
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from redaction import redact_text
from tools import RepoTools, ToolError

SEVERITIES = ("critical", "high", "medium", "low")
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}

CATEGORIES = (
    "repository_hygiene",
    "dependency_management",
    "code_quality",
    "testing",
    "github_actions_ci",
    "security",
    "configuration",
    "collaboration",
    "build_and_release",
)

CATEGORY_PREFIX = {
    "repository_hygiene": "REPO",
    "dependency_management": "DEP",
    "code_quality": "QUAL",
    "testing": "TEST",
    "github_actions_ci": "CI",
    "security": "SEC",
    "configuration": "CFG",
    "collaboration": "COLLAB",
    "build_and_release": "BUILD",
}

_CATEGORY_ALIASES = {
    "repo_hygiene": "repository_hygiene", "hygiene": "repository_hygiene", "repository": "repository_hygiene",
    "dependencies": "dependency_management", "dependency": "dependency_management", "deps": "dependency_management",
    "quality": "code_quality", "lint": "code_quality", "linting": "code_quality",
    "tests": "testing", "test": "testing",
    "ci": "github_actions_ci", "ci_cd": "github_actions_ci", "cicd": "github_actions_ci",
    "github_actions": "github_actions_ci", "github_actions_ci": "github_actions_ci", "actions": "github_actions_ci",
    "github_actions_/_ci": "github_actions_ci", "workflows": "github_actions_ci",
    "sec": "security",
    "config": "configuration",
    "collab": "collaboration", "review": "collaboration",
    "build": "build_and_release", "release": "build_and_release", "build_release": "build_and_release",
    "build_and_release": "build_and_release", "deployment": "build_and_release",
}

_RULE_RE = re.compile(r"^[A-Z]{2,8}-\d{3}$")
MAX_DESCRIPTION = 300
MAX_EVIDENCE = 700
MAX_RECOMMENDATION = 700


def canonical_category(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    key = re.sub(r"[\s/\-]+", "_", value.strip().lower()).strip("_")
    key = re.sub(r"_+", "_", key)
    if key in CATEGORIES:
        return key
    return _CATEGORY_ALIASES.get(key)


@dataclass
class Finding:
    id: str
    rule_id: str
    category: str
    severity: str
    file: Optional[str]
    line: Optional[int]
    description: str
    evidence: str
    recommendation: str
    retracted: bool = False

    def as_violation(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "severity": self.severity,
            "file": self.file,
            "line": self.line,
            "description": self.description,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
        }


@dataclass
class LedgerResult:
    ok: bool
    message: str
    finding_id: Optional[str] = None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _clean_text(value: Any, limit: int) -> str:
    text = redact_text(" ".join(str(value).split()))
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


class FindingLedger:
    def __init__(self, tools: RepoTools):
        self.tools = tools
        self.findings: List[Finding] = []
        self._counter = 0
        self._auto_rule: Dict[str, int] = {}

    def active(self) -> List[Finding]:
        return [f for f in self.findings if not f.retracted]

    def summary_lines(self) -> List[str]:
        out = []
        for f in self.active():
            loc = f"{f.file}:{f.line}" if f.file and f.line else (f.file or "repo-wide")
            out.append(f"{f.id} {f.rule_id} {f.severity} {loc} - {f.description[:90]}")
        return out

    def retract(self, finding_id: Any, reason: Any = "") -> LedgerResult:
        for f in self.findings:
            if f.id == str(finding_id).strip() and not f.retracted:
                f.retracted = True
                return LedgerResult(True, f"Retracted {f.id} ({f.rule_id}).", f.id)
        return LedgerResult(False, f"No active finding with id '{finding_id}'. Active ids: "
                                   f"{', '.join(f.id for f in self.active()) or 'none'}.")

    def violations(self) -> List[Dict[str, Any]]:
        ordered = sorted(
            self.active(),
            key=lambda f: (SEVERITY_RANK[f.severity], f.category, f.file or "", f.line or 0, f.rule_id),
        )
        return [f.as_violation() for f in ordered]

    def add(self, raw: Any) -> LedgerResult:
        if not isinstance(raw, dict):
            return LedgerResult(False, "REJECTED: record_finding args must be a JSON object.")

        category = canonical_category(raw.get("category"))
        if category is None:
            return LedgerResult(False, f"REJECTED: category {raw.get('category')!r} is invalid. Use one of: {', '.join(CATEGORIES)}.")
        severity = str(raw.get("severity", "")).strip().lower()
        if severity not in SEVERITY_RANK:
            return LedgerResult(False, f"REJECTED: severity {raw.get('severity')!r} is invalid. Use one of: {', '.join(SEVERITIES)}.")
        for key in ("description", "evidence", "recommendation"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                return LedgerResult(False, f"REJECTED: '{key}' is required and must be a non-empty string.")

        file_val = raw.get("file")
        line_val = raw.get("line")
        quote = raw.get("quote")
        file_rel: Optional[str] = None
        line: Optional[int] = None
        evidence_suffix = ""

        if isinstance(file_val, str) and file_val.strip() and file_val.strip().lower() not in {"null", "none", "n/a"}:
            requested = file_val.strip()
            try:
                rel, _real = self.tools._resolve_file(requested)
            except ToolError as exc:
                msg = str(exc)
                if "not found" in msg or "directory" in msg or "not a regular file" in msg:
                    # Absence/repo-level finding: keep the claim, but file must be null.
                    normalized = posixpath.normpath(requested.replace("\\", "/")).lstrip("./")
                    evidence_suffix = f" (Expected path: {normalized}.)"
                    rel = None
                else:
                    return LedgerResult(False, f"REJECTED: invalid file path: {msg}")
            if rel is not None:
                if rel not in self.tools.seen_files:
                    return LedgerResult(False, f"REJECTED: you have not inspected {rel} yet. read_file or search_code it first, "
                                               "then record the finding with an exact quote.")
                lines = self.tools.redacted_file_lines(rel)
                if not isinstance(quote, str) or not quote.strip():
                    return LedgerResult(False, f"REJECTED: 'quote' is required when 'file' is set: copy a short exact snippet "
                                               f"(one line is best) from {rel} that shows the defect. If the defect is an absence "
                                               "with no relevant file, set file to null.")
                ok, found_lines = self._verify_quote(lines, quote)
                if not ok:
                    return LedgerResult(False, f"REJECTED: the quote was not found verbatim in {rel}. Re-read the file and copy the "
                                               "exact text (whitespace is ignored; secrets appear as [REDACTED:...]).")
                file_rel = rel
                line = self._pick_line(line_val, found_lines, len(lines))
        elif line_val not in (None, "", 0):
            line = None

        category_prefix = CATEGORY_PREFIX[category]
        rule_id = str(raw.get("rule_id", "")).strip().upper()
        if not _RULE_RE.match(rule_id):
            self._auto_rule[category_prefix] = self._auto_rule.get(category_prefix, 900) + 1
            rule_id = f"{category_prefix}-{self._auto_rule[category_prefix]:03d}"

        description = _clean_text(raw["description"], MAX_DESCRIPTION)
        evidence = _clean_text(raw["evidence"], MAX_EVIDENCE - len(evidence_suffix)) + evidence_suffix
        recommendation = _clean_text(raw["recommendation"], MAX_RECOMMENDATION)

        dup = self._find_duplicate(rule_id, category, file_rel, line, description)
        if dup is not None:
            return LedgerResult(False, f"REJECTED: duplicate of {dup.id} ({dup.rule_id}: {dup.description[:80]}). "
                                       "Group repeated instances of one root cause into a single finding; if you need to improve "
                                       f"it, retract_finding {dup.id} and record the better version.")

        self._counter += 1
        finding = Finding(f"F{self._counter}", rule_id, category, severity, file_rel, line,
                          description, evidence, recommendation)
        self.findings.append(finding)
        loc = f"{file_rel}:{line}" if file_rel and line else (file_rel or "repo-wide")
        note = ""
        if file_val and not file_rel:
            note = " (file set to null because that path does not exist; the expected path is kept in the evidence)"
        return LedgerResult(True, f"ACCEPTED {finding.id}: {rule_id} {severity} at {loc}{note}.", finding.id)

    @staticmethod
    def _verify_quote(lines: List[str], quote: str):
        q_lines = [_norm(l) for l in redact_text(quote).split("\n") if l.strip()]
        if not q_lines:
            return False, []
        norm_lines = [_norm(l) for l in lines]
        joined = " ".join(norm_lines)
        whole = " ".join(q_lines)
        first_hits = [i + 1 for i, nl in enumerate(norm_lines) if q_lines[0] in nl]
        if whole in joined:
            return True, first_hits or [1]
        if len(q_lines) > 1 and all(any(ql in nl for nl in norm_lines) for ql in q_lines):
            return True, first_hits or [1]
        return False, []

    @staticmethod
    def _pick_line(line_val: Any, candidates: List[int], total: int) -> Optional[int]:
        given: Optional[int] = None
        try:
            if line_val not in (None, ""):
                given = int(line_val)
        except (TypeError, ValueError):
            given = None
        if given is not None and 1 <= given <= total and (not candidates or given in candidates):
            return given
        if candidates:
            if given is not None:
                return min(candidates, key=lambda c: abs(c - given))
            return candidates[0]
        return given if given is not None and 1 <= given <= total else None

    def _find_duplicate(self, rule_id: str, category: str, file: Optional[str], line: Optional[int],
                        description: str) -> Optional[Finding]:
        for f in self.active():
            if f.rule_id == rule_id and f.file == file:
                return f
            if f.file == file and f.category == category:
                if file and line and f.line and abs(f.line - line) <= 2:
                    return f
                ratio = difflib.SequenceMatcher(None, _norm(f.description), _norm(description)).ratio()
                if ratio >= 0.82:
                    return f
            if file is None and f.file is None and f.category == category:
                ratio = difflib.SequenceMatcher(None, _norm(f.description), _norm(description)).ratio()
                if ratio >= 0.75:
                    return f
        return None
