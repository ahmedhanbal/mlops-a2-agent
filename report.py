"""Report construction, strict validation and atomic writing."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from findings import CATEGORIES, SEVERITIES

REQUIRED_FIELDS = (
    "rule_id", "category", "severity", "file", "line", "description", "evidence", "recommendation",
)


class ReportError(Exception):
    pass


def build_report(repository: str, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"repository": repository, "violations": violations}


def validate_report(report: Any) -> List[str]:
    """Return a list of problems (empty list means the report is valid)."""
    errors: List[str] = []
    if not isinstance(report, dict):
        return ["report must be a JSON object"]
    if set(report) != {"repository", "violations"}:
        errors.append(f"top-level keys must be exactly repository, violations; got {sorted(report)}")
    if not isinstance(report.get("repository"), str) or not report.get("repository"):
        errors.append("'repository' must be a non-empty string")
    violations = report.get("violations")
    if not isinstance(violations, list):
        errors.append("'violations' must be an array")
        return errors
    for i, v in enumerate(violations):
        where = f"violations[{i}]"
        if not isinstance(v, dict):
            errors.append(f"{where} must be an object")
            continue
        missing = [k for k in REQUIRED_FIELDS if k not in v]
        if missing:
            errors.append(f"{where} missing fields: {missing}")
        extra = set(v) - set(REQUIRED_FIELDS)
        if extra:
            errors.append(f"{where} has unexpected fields: {sorted(extra)}")
        if v.get("severity") not in SEVERITIES:
            errors.append(f"{where}.severity must be one of {SEVERITIES}")
        if v.get("category") not in CATEGORIES:
            errors.append(f"{where}.category must be one of {CATEGORIES}")
        for key in ("rule_id", "description", "evidence", "recommendation"):
            if not isinstance(v.get(key), str) or not v.get(key).strip():
                errors.append(f"{where}.{key} must be a non-empty string")
        if v.get("file") is not None and not isinstance(v.get("file"), str):
            errors.append(f"{where}.file must be a string or null")
        line = v.get("line")
        if line is not None and (not isinstance(line, int) or isinstance(line, bool) or line < 1):
            errors.append(f"{where}.line must be a positive integer or null")
        if v.get("file") is None and line is not None:
            errors.append(f"{where}.line must be null when file is null")
    return errors


def write_report(path: Path, report: Dict[str, Any]) -> None:
    """Validate then write atomically (never leaves a half-written report)."""
    problems = validate_report(report)
    if problems:
        raise ReportError("refusing to write an invalid report: " + "; ".join(problems[:5]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".report-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
