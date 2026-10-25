"""Shared fixtures: a small git repository with deliberately planted defects."""

import json
import subprocess
from pathlib import Path

import pytest

# A fake credential assembled at runtime so no secret-looking literal sits in this repo.
FAKE_AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"
FAKE_PASSWORD = "Sup3r" + "S3cretPassw0rd"

FILES = {
    "README.md": "# Demo\n\nA demo service.\n",
    ".gitignore": "__pycache__/\n*.pyc\n",
    "requirements.txt": "flask\nrequests>=2\n",
    "config.py": (
        "import os\n"
        "DEBUG = True\n"
        f'AWS_ACCESS_KEY_ID = "{FAKE_AWS_KEY}"\n'
        f'DB_PASSWORD = "{FAKE_PASSWORD}"\n'
        'API_TOKEN = os.environ["API_TOKEN"]\n'
        'DATA_DIR = "C:\\\\Users\\\\alice\\\\data"\n'
    ),
    "app.py": "from config import DEBUG\n\n\ndef main():\n    print(undefined_name)\n",
    "tests/test_app.py": "def test_ok():\n    assert True\n",
    ".DS_Store": "junk",
    ".github/workflows/ci.yml": (
        "name: CI\n"
        "on:\n"
        "  push:\n"
        "    branches: [master]\n"
        "permissions: write-all\n"
        "jobs:\n"
        "  test:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: some-org/setup-thing@main\n"
        "      - name: Run tests\n"
        "        run: pytest || true\n"
        "        continue-on-error: true\n"
        "      - name: Greet\n"
        "        run: echo \"${{ github.event.pull_request.title }}\"\n"
        "# - name: Lint\n"
        "#   run: ruff check .\n"
    ),
}


def _git(path: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=path, check=True, capture_output=True, text=True,
    )


@pytest.fixture()
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "sample"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    for rel, content in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial commit")
    (root / "README.md").write_text("# Demo\n\nA demo service.\n\nMore.\n")
    _git(root, "commit", "-q", "-am", "docs: expand README")
    return root


class ScriptedLLM:
    """Fake LLM. ``script`` items are JSON strings or callables(messages)->str."""

    model = "scripted"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def complete(self, system, messages, max_tokens):
        self.calls.append(messages)
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of responses")
        item = self.script.pop(0)
        return item(messages) if callable(item) else item


def act(thought, *actions):
    return json.dumps({"thought": thought, "actions": [{"tool": t, "args": a} for t, a in actions]})
