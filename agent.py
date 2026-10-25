#!/usr/bin/env python3
"""DevOps repository auditing agent CLI."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from agent_loop import AuditAgent, AuditFatal, Limits, TraceLog
from findings import FindingLedger
from llm_client import LLMError, build_llm, load_env_file
from redaction import redact_text
from report import ReportError, build_report, write_report
from tools import RepoTools
from workspace import (CloneError, WorkspaceError, clone_repository, parse_github_url,
                       snapshot_head, working_tree_changes)

EXIT_OK, EXIT_USAGE, EXIT_CLONE, EXIT_LLM, EXIT_INTERNAL, EXIT_INTERRUPT = 0, 2, 3, 4, 5, 130

log = logging.getLogger("audit")


class _RedactingFilter(logging.Filter):
    """Guarantee that no secret reaches stderr/log files."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(record.getMessage())
        record.args = ()
        return True


def _configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    handler.addFilter(_RedactingFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for noisy in ("httpx", "httpcore", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="agent.py",
        description="Autonomous DevOps auditing agent for public GitHub repositories.",
    )
    p.add_argument("--repo", required=True, help="public GitHub repository URL, e.g. https://github.com/owner/repo")
    p.add_argument("--output", default="report.json", help="where to write the JSON report (default: report.json)")
    p.add_argument("--provider", choices=["deepseek", "anthropic", "openai"], help="LLM provider (default: deepseek)")
    p.add_argument("--model", help="model name (default: LLM_MODEL env var or provider default)")
    p.add_argument("--max-steps", type=int, default=40, help="maximum LLM turns (default 40)")
    p.add_argument("--timeout", type=float, default=900.0, help="overall wall-clock budget in seconds (default 900)")
    p.add_argument("--clone-depth", type=int, default=200, help="git history depth to clone (default 200, 0 = full)")
    p.add_argument("--trace", help="write a JSONL tool-call trace here (default: <output>.trace.jsonl)")
    p.add_argument("--no-trace", action="store_true", help="do not write a trace file")
    p.add_argument("--allow-local", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging on stderr")
    args = p.parse_args(argv)
    if args.max_steps < 1 or args.max_steps > 200:
        p.error("--max-steps must be between 1 and 200")
    if args.timeout < 30:
        p.error("--timeout must be at least 30 seconds")
    return args


def _default_trace_path(output: Path) -> Path:
    return output.with_name(output.stem + ".trace.jsonl")


def run_audit(args: argparse.Namespace, llm_factory: Callable[[], Any] = None) -> int:
    """Execute one audit. ``llm_factory`` is injectable for tests."""
    output = Path(args.output)

    try:
        ref = parse_github_url(args.repo, allow_local=getattr(args, "allow_local", False))
    except WorkspaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and output.is_dir():
            raise OSError(f"{output} is a directory")
    except OSError as exc:
        print(f"error: cannot write to output path {output}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        llm = (llm_factory or (lambda: build_llm(args.provider, args.model)))()
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    trace = TraceLog(None if getattr(args, "no_trace", False) else Path(args.trace) if args.trace else _default_trace_path(output))
    checkout = None
    try:
        log.info("Cloning %s (depth %s) ...", ref.slug, args.clone_depth or "full")
        try:
            checkout = clone_repository(ref, depth=args.clone_depth, timeout=min(300.0, max(60.0, args.timeout / 2)))
        except (CloneError, WorkspaceError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            trace.event("fatal", stage="clone", error=str(exc))
            return EXIT_CLONE
        before = snapshot_head(checkout.root)
        trace.event("start", repository=args.repo, branch=checkout.branch, head=checkout.head,
                    max_steps=args.max_steps, model=getattr(llm, "model", "custom"))

        tools = RepoTools(checkout.root)
        ledger = FindingLedger(tools)
        agent = AuditAgent(llm, tools, ledger, args.repo.strip(), checkout.branch, checkout.head,
                           Limits(max_steps=args.max_steps, timeout_s=args.timeout), trace)
        try:
            outcome = agent.run()
        except AuditFatal as exc:
            print(f"error: audit aborted: {redact_text(str(exc))}", file=sys.stderr)
            return EXIT_LLM

        after = snapshot_head(checkout.root)
        dirty = working_tree_changes(checkout.root)
        unchanged = before == after and not dirty
        trace.event("integrity", unchanged=unchanged, dirty=dirty[:300])
        if not unchanged:
            log.warning("working tree of the clone changed during the audit (unexpected)")

        report = build_report(args.repo.strip(), ledger.violations())
        try:
            write_report(output, report)
        except (ReportError, OSError) as exc:
            print(f"error: could not write report: {exc}", file=sys.stderr)
            return EXIT_INTERNAL
        n = len(report["violations"])
        log.info("Audit %s after %d turn(s): %d violation(s) -> %s", outcome.finish_reason, outcome.steps, n, output)
        if outcome.finish_reason != "finished":
            log.warning("The audit stopped on its %s before the model finished; the report covers what was verified so far.",
                        outcome.finish_reason)
        print(f"Report written to {output} ({n} violation(s)).")
        return EXIT_OK
    finally:
        trace.close()
        if checkout is not None:
            checkout.cleanup()


def main(argv: Optional[list] = None) -> int:
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"):
        load_env_file(candidate)
    args = parse_args(argv)
    _configure_logging(args.verbose)

    def _on_term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_term)
    try:
        return run_audit(args)
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return EXIT_INTERRUPT
    except Exception as exc:
        log.debug("unhandled error", exc_info=True)
        print(f"error: unexpected internal failure: {type(exc).__name__}: {redact_text(str(exc))[:300]}", file=sys.stderr)
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())
