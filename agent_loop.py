"""Agent loop: the model picks tools, the loop validates, dispatches and feeds results back."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from findings import FindingLedger
from llm_client import LLMError, LLMResponse
from prompts import (FINAL_TURNS_NOTICE, FINISH_TOO_EARLY, REVIEW_PROMPT, build_system_prompt,
                     build_task_message)
from redaction import redact_obj, redact_text
from tools import INSPECTION_TOOLS, RepoTools

log = logging.getLogger("audit.loop")

AGENT_ACTIONS = ("record_finding", "retract_finding", "finish")


class AuditFatal(Exception):
    """The audit cannot continue (LLM unusable / protocol broken)."""


@dataclass
class Limits:
    max_steps: int = 40
    timeout_s: float = 900.0
    max_actions_per_turn: int = 4
    context_chars: int = 160_000
    max_tokens: int = 4096
    min_inspections_before_finish: int = 4
    max_protocol_errors: int = 4


@dataclass
class Observation:
    id: str
    tool: str
    args: Dict[str, Any]
    ok: bool
    text: str
    elided: bool = False

    def render(self) -> str:
        args = json.dumps(redact_obj(self.args), ensure_ascii=True, sort_keys=True).replace("'", "\\u0027")
        head = f"<tool_result id=\"{self.id}\" tool=\"{self.tool}\" args='{args}' ok=\"{str(self.ok).lower()}\""
        if self.elided:
            return head + " elided=\"true\">[output elided to save context; call the tool again if you need it]</tool_result>"
        body = self.text.replace("</tool_result", "<\\/tool_result")
        return f"{head}>\n{body}\n</tool_result>"


@dataclass
class Turn:
    step: int
    assistant_text: str
    observations: List[Observation] = field(default_factory=list)
    notices: List[str] = field(default_factory=list)


@dataclass
class AuditOutcome:
    finish_reason: str
    steps: int
    summary: str = ""
    tool_counts: Dict[str, int] = field(default_factory=dict)


class TraceLog:
    """JSONL trace of every LLM turn, tool call and finding (redacted)."""

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self._fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "w", encoding="utf-8")

    def event(self, kind: str, **data: Any) -> None:
        if not self._fh:
            return
        record = {"ts": round(time.time(), 3), "type": kind, **data}
        self._fh.write(json.dumps(redact_obj(record), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Find the first JSON object in ``text`` (tolerates fences and chatter)."""
    decoder = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            obj, _ = decoder.raw_decode(text[pos:])
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        pos = text.find("{", pos + 1)
    return None


def _coerce_response(resp: Any) -> LLMResponse:
    if isinstance(resp, LLMResponse):
        return resp
    return LLMResponse(str(resp))


class AuditAgent:
    def __init__(self, llm: Any, tools: RepoTools, ledger: FindingLedger, repo_url: str, branch: str, head: str,
                 limits: Optional[Limits] = None, trace: Optional[TraceLog] = None):
        self.llm = llm
        self.tools = tools
        self.ledger = ledger
        self.repo_url = repo_url
        self.branch = branch
        self.head = head
        self.limits = limits or Limits()
        self.trace = trace or TraceLog(None)
        self.system = build_system_prompt()
        self.turns: List[Turn] = []
        self.tool_counts: Counter = Counter()
        self._calls: Dict[str, Observation] = {}
        self._reviewed = False
        self._finish_refusals = 0
        self._protocol_errors = 0
        self._finished = False
        self._summary = ""

    def run(self) -> AuditOutcome:
        started = time.monotonic()
        deadline = started + self.limits.timeout_s
        first_message = build_task_message(self.repo_url, self.branch, self.head,
                                           len(self.tools.tracked_files()), self.limits.max_steps)
        reason = "step_budget"
        step = 0
        for step in range(1, self.limits.max_steps + 1):
            if time.monotonic() > deadline:
                reason = "time_budget"
                step -= 1
                break
            messages = self._build_messages(first_message, step)
            try:
                resp = _coerce_response(self.llm.complete(self.system, messages, self.limits.max_tokens))
            except LLMError as exc:
                self.trace.event("fatal", step=step, error=str(exc))
                raise AuditFatal(str(exc)) from exc
            text = resp.text or ""
            self.trace.event("llm_response", step=step, text=text[:6000],
                             input_tokens=resp.input_tokens, output_tokens=resp.output_tokens)

            turn = Turn(step=step, assistant_text=text)
            self.turns.append(turn)
            parsed = extract_json_object(text)
            actions = self._extract_actions(parsed)
            if actions is None:
                self._protocol_errors += 1
                turn.notices.append(
                    "PROTOCOL ERROR: reply with exactly one JSON object of the form "
                    '{"thought": "...", "actions": [{"tool": "<name>", "args": {...}}]}. No prose, no code fences.')
                self.trace.event("protocol_error", step=step, raw=text[:500])
                if self._protocol_errors >= self.limits.max_protocol_errors:
                    raise AuditFatal(f"The model failed to follow the JSON action protocol {self._protocol_errors} times.")
                continue

            thought = str(parsed.get("thought", "")).strip() if parsed else ""
            self.trace.event("interpretation", step=step, thought=thought[:1500])
            if len(actions) > self.limits.max_actions_per_turn:
                turn.notices.append(f"Only the first {self.limits.max_actions_per_turn} actions were executed; "
                                    f"the other {len(actions) - self.limits.max_actions_per_turn} were ignored.")
                actions = actions[: self.limits.max_actions_per_turn]

            for idx, action in enumerate(actions, 1):
                self._execute(step, idx, action, turn)
                if self._finished:
                    break
            if self._finished:
                reason = "finished"
                break
        else:
            reason = "step_budget"
        outcome = AuditOutcome(reason, step, self._summary, dict(self.tool_counts))
        self.trace.event("done", finish_reason=reason, steps=step, findings=len(self.ledger.active()),
                         tool_counts=dict(self.tool_counts), summary=self._summary[:1000])
        return outcome

    @staticmethod
    def _extract_actions(parsed: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        if not parsed:
            return None
        actions = parsed.get("actions")
        if actions is None and "tool" in parsed:
            actions = [{"tool": parsed["tool"], "args": parsed.get("args", {})}]
        if isinstance(actions, dict):
            actions = [actions]
        if not isinstance(actions, list) or not actions:
            return None
        return [a for a in actions if isinstance(a, dict)] or None

    def _execute(self, step: int, idx: int, action: Dict[str, Any], turn: Turn) -> None:
        name = str(action.get("tool") or action.get("name") or "").strip()
        args = action.get("args", action.get("arguments", {}))
        obs_id = f"{step}.{idx}"

        if name in INSPECTION_TOOLS:
            self._run_inspection(obs_id, name, args, turn)
        elif name == "record_finding":
            result = self.ledger.add(args)
            self.trace.event("finding", step=step, accepted=result.ok, message=result.message,
                             finding_id=result.finding_id, args=args if isinstance(args, dict) else str(args)[:300])
            turn.observations.append(Observation(obs_id, name, {"category": (args or {}).get("category") if isinstance(args, dict) else None,
                                                                "file": (args or {}).get("file") if isinstance(args, dict) else None},
                                                 result.ok, result.message))
        elif name == "retract_finding":
            fid = args.get("id") if isinstance(args, dict) else args
            reason = args.get("reason", "") if isinstance(args, dict) else ""
            result = self.ledger.retract(fid, reason)
            self.trace.event("retract", step=step, ok=result.ok, message=result.message, reason=str(reason)[:300])
            turn.observations.append(Observation(obs_id, name, {"id": fid}, result.ok, result.message))
        elif name == "finish":
            self._handle_finish(obs_id, args, turn)
        else:
            valid = ", ".join(list(INSPECTION_TOOLS) + list(AGENT_ACTIONS))
            turn.observations.append(Observation(obs_id, name or "?", {}, False,
                                                 f"Unknown tool '{name}'. Valid tools/actions: {valid}"))

    def _run_inspection(self, obs_id: str, name: str, args: Any, turn: Turn) -> None:
        key = name + json.dumps(args, sort_keys=True, default=str)
        previous = self._calls.get(key)
        if previous is not None and not previous.elided:
            turn.observations.append(Observation(
                obs_id, name, args if isinstance(args, dict) else {}, True,
                f"Duplicate of call {previous.id}: the result is already in your context. Choose a different action."))
            return
        t0 = time.monotonic()
        result = self.tools.call(name, args)
        dur = int((time.monotonic() - t0) * 1000)
        self.tool_counts[name] += 1
        obs = Observation(obs_id, name, args if isinstance(args, dict) else {}, result.ok, result.text)
        self._calls[key] = obs
        turn.observations.append(obs)
        self.trace.event("tool_call", step=int(obs_id.split(".")[0]), id=obs_id, tool=name,
                         args=args, ok=result.ok, duration_ms=dur, output_chars=len(result.text),
                         truncated=result.truncated, output_preview=result.text[:400])
        log.info("step %s: %s %s -> %s (%d chars)", obs_id, name, json.dumps(redact_obj(args))[:120],
                 "ok" if result.ok else "error", len(result.text))

    def _handle_finish(self, obs_id: str, args: Any, turn: Turn) -> None:
        inspections = sum(self.tool_counts.values())
        if inspections < self.limits.min_inspections_before_finish and self._finish_refusals < 2:
            self._finish_refusals += 1
            turn.observations.append(Observation(obs_id, "finish", {}, False, FINISH_TOO_EARLY.format(n=inspections)))
            return
        if not self._reviewed:
            self._reviewed = True
            usage = ", ".join(f"{k}x{v}" for k, v in sorted(self.tool_counts.items())) or "none"
            findings = "\n".join(self.ledger.summary_lines()) or "(none recorded)"
            turn.observations.append(Observation(obs_id, "finish", {}, False, REVIEW_PROMPT.format(usage=usage, findings=findings)))
            self.trace.event("review_requested", findings=len(self.ledger.active()))
            return
        self._finished = True
        self._summary = redact_text(str(args.get("summary", "")) if isinstance(args, dict) else str(args))[:1000]
        turn.observations.append(Observation(obs_id, "finish", {}, True, "Audit finished."))

    def _status_block(self, step: int) -> str:
        lines = [f"[status] turn {step}/{self.limits.max_steps}; findings recorded: {len(self.ledger.active())}; "
                 f"files inspected: {len(self.tools.seen_files)}"]
        summary = self.ledger.summary_lines()
        if summary:
            lines.append("Recorded findings (do not re-record these):")
            lines.extend("  " + s for s in summary)
        left = self.limits.max_steps - step
        if left <= 2:
            lines.append(FINAL_TURNS_NOTICE.format(left=left + 1))
        return "\n".join(lines)

    def _build_messages(self, first_message: str, step: int) -> List[Dict[str, str]]:
        def size() -> int:
            total = len(first_message)
            for t in self.turns:
                total += len(t.assistant_text)
                total += sum(len(o.text) + 120 if not o.elided else 160 for o in t.observations)
            return total

        if size() > self.limits.context_chars:
            keep_from = max(0, len(self.turns) - 3)
            for t in self.turns[:keep_from]:
                for o in sorted(t.observations, key=lambda o: -len(o.text)):
                    if o.tool in ("record_finding", "retract_finding", "finish") or o.elided or len(o.text) < 400:
                        continue
                    o.elided = True
                    if size() <= self.limits.context_chars:
                        break
                if size() <= self.limits.context_chars:
                    break

        messages: List[Dict[str, str]] = [{"role": "user", "content": first_message}]
        for t in self.turns:
            messages.append({"role": "assistant", "content": t.assistant_text or "{}"})
            parts = [o.render() for o in t.observations] + t.notices
            is_last = t is self.turns[-1]
            if is_last:
                parts.append(self._status_block(step))
            messages.append({"role": "user", "content": "\n".join(parts) if parts else "(no observations)"})
        if messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": self._status_block(step)})
        return messages
