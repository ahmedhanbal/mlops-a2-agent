# MLOPS - Repository Auditing Agent

An agent that audits a **public GitHub repository** for DevOps best-practice violations and writes an
evidence-based JSON report.

```
python agent.py --repo https://github.com/<owner>/<repo> --output report.json
```

## Install

Requires Python 3.10+ and `git` on `PATH`.

```bash
git clone <this repository> && cd <this repository>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then fill in ONE provider key
```

## Run

```bash
python agent.py --repo https://github.com/example/demo-project --output report.json
```

Options: `--provider deepseek|anthropic|openai`, `--model`, `--max-steps 40` (LLM turns), `--timeout 900` (seconds), `--clone-depth 200`,
`--trace PATH` / `--no-trace`, `-v`. It runs unattended; there are no interactive prompts.

Outputs:
* `report.json` - the report (exactly `{"repository": ..., "violations": [...]}`; `"violations": []` when nothing defensible is found).
* `report.trace.jsonl` - a redacted JSONL trace of every LLM turn (including the model's interpretation of each
  observation), tool call, finding decision and the final integrity check, so the genuine agent behaviour can be inspected.

**Exit codes:** `0` completed · `2` invalid input / missing configuration · `3` clone or network failure ·
`4` LLM failure · `5` internal error · `130` interrupted. On any non-zero exit no report is written and a message goes to stderr.

### Report format

```json
{
  "repository": "https://github.com/example/demo-project",
  "violations": [
    {
      "rule_id": "SEC-001",
      "category": "security",
      "severity": "critical",
      "file": "config.py",
      "line": 17,
      "description": "Hardcoded API credential",
      "evidence": "Credential assigned directly in source code",
      "recommendation": "Use a secrets manager or environment variable"
    }
  ]
}
```

* `severity`: `critical | high | medium | low`.
* `category`: `repository_hygiene | dependency_management | code_quality | testing | github_actions_ci | security | configuration | collaboration | build_and_release`.
* `file`/`line` are JSON `null` when not applicable (e.g. "no tests exist"). A finding about an absent file keeps the
  expected path in `evidence`. Secret values are never written; they appear only as `[REDACTED:...]`.


**Tools** (`tools.py`)

| Tool | Purpose |
|---|---|
| `list_files(path_prefix, max_results)` | tracked files with sizes, top-level and extension summary |
| `read_file(path, start_line, end_line)` | numbered, redacted, paged file contents |
| `search_code(pattern, path_glob, ...)` | regex search (ReDoS-guarded) with optional context |
| `git_log(max_count, path, message_grep, show_files)` | history, shallow-clone aware |
| `git_status()` | default branch, HEAD, cleanliness, shallow flag |
| `parse_workflow(path)` | GitHub Actions YAML -> structured summary with line numbers and mechanical *signals* (branch filter vs default branch, `continue-on-error`, `\|\| true`, mutable action refs, untrusted `${{ }}` in `run:`, test/lint/build/deploy commands, commented-out steps). Signals are facts, the model decides if they are defects |
| `run_ruff(path)` | optional static analysis for syntax errors/undefined names; reports "unavailable" instead of guessing |

Agent actions: `record_finding`, `retract_finding`, `finish`.

**Precision controls**
* A finding about a file is rejected unless the agent actually read/searched that file, and its `quote` must appear
  verbatim in the file. A verified quote also corrects wrong line numbers. Rejections go back to the model to fix or drop.
* Duplicates (same rule+file, nearby lines, near-identical text) are rejected; one finding per root cause.
* The first `finish` triggers a mandatory precision/coverage review (retract weak findings, inspect unexamined areas).
  `finish` is refused before a minimum amount of real inspection.
* The prompt encodes the evidence rule: absent optional files/tools are not defects when equivalents exist, and remote
  settings such as branch protection are never inferred from files.


## Tests

```bash
pytest          # no network and no API key needed: uses a scripted fake LLM and local fixture repositories
ruff check .
```

## Architecture

```
agent.py          CLI, exit codes, orchestration, integrity check, report writing
agent_loop.py     the agent loop: LLM turn -> parse JSON actions -> dispatch -> feed observations back
tools.py          read-only inspection tools + argument validation (the tool registry)
findings.py       findings ledger: evidence gates, line correction, de-duplication
prompts.py        system prompt, review prompt
llm_client.py     provider SDK wrappers (Anthropic / OpenAI-compatible), retries, .env loader
workspace.py      URL validation and hardened shallow clone into a temp dir
redaction.py      secret redaction applied to everything that leaves the tool layer
report.py         strict schema validation and atomic report writing
```


