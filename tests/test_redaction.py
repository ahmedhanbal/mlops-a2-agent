from redaction import contains_secret, redact_lines, redact_obj, redact_text
from conftest import FAKE_AWS_KEY, FAKE_PASSWORD


def test_provider_tokens_are_redacted():
    out = redact_text(f"key = {FAKE_AWS_KEY}")
    assert FAKE_AWS_KEY not in out and "[REDACTED:aws_access_key" in out
    gh = "ghp_" + "a" * 36
    assert gh not in redact_text(f"token: {gh}")


def test_assignments_redacted_but_keys_visible():
    out = redact_text(f'DB_PASSWORD = "{FAKE_PASSWORD}"')
    assert FAKE_PASSWORD not in out
    assert "DB_PASSWORD" in out and "len=" in out


def test_placeholders_and_references_survive():
    for line in [
        'API_TOKEN = os.environ["API_TOKEN"]',
        "token: ${{ secrets.NPM_TOKEN }}",
        'password = "your_password_here"',
        "PASSWORD=${DB_PASSWORD}",
        "password = get_password()",
        'api_key = settings.API_KEY',
        'secret_url = "https://example.com/path"',
    ]:
        assert redact_text(line) == line, line


def test_url_credentials_redacted():
    out = redact_text("postgres://admin:hunter2hunter2@db.internal:5432/app")
    assert "hunter2hunter2" not in out and "admin" in out


def test_private_key_block_redacted_and_line_count_preserved():
    lines = ["a", "-----BEGIN RSA PRIVATE KEY-----", "MIIEabc", "def456", "-----END RSA PRIVATE KEY-----", "b"]
    out = redact_lines(lines)
    assert len(out) == len(lines)
    assert "MIIEabc" not in "\n".join(out)
    assert out[0] == "a" and out[-1] == "b"


def test_idempotent_and_contains_secret():
    text = f'x = "{FAKE_PASSWORD}"\npassword = "{FAKE_PASSWORD}"'
    once = redact_text(text)
    assert redact_text(once) == once
    assert contains_secret(text) and not contains_secret(once)


def test_redact_obj_recurses():
    obj = {"a": [f"token = '{FAKE_PASSWORD}'"], "b": {"c": 1}}
    assert FAKE_PASSWORD not in str(redact_obj(obj))
