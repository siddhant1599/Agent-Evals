"""The .env loader: parsing, placeholder handling, precedence, and never leaking a value."""

from __future__ import annotations

import os

import pytest
from agent_evals.env import EnvFileError, load_env_file

SECRET = "FAKE_EXAMPLE_not_a_real_token"


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("FACTAGENT_A", "FACTAGENT_B", "FACTAGENT_C", "FACTAGENT_EMPTY"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _write(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


class TestParsing:
    def test_sets_keys_and_skips_comments_and_blank_lines(self, tmp_path, clean_env):
        path = _write(tmp_path, "# comment\n\nFACTAGENT_A=1\nexport FACTAGENT_B=two\n")
        assert load_env_file(path) == ("FACTAGENT_A", "FACTAGENT_B")
        assert os.environ["FACTAGENT_A"] == "1"
        assert os.environ["FACTAGENT_B"] == "two"

    def test_strips_matching_quotes_only(self, tmp_path, clean_env):
        load_env_file(_write(tmp_path, "FACTAGENT_A=\"x y\"\nFACTAGENT_B='z'\nFACTAGENT_C=\"q'\n"))
        assert os.environ["FACTAGENT_A"] == "x y"
        assert os.environ["FACTAGENT_B"] == "z"
        assert os.environ["FACTAGENT_C"] == "\"q'"

    def test_value_may_contain_equals_signs(self, tmp_path, clean_env):
        load_env_file(_write(tmp_path, "FACTAGENT_A=http://h:1/?a=b\n"))
        assert os.environ["FACTAGENT_A"] == "http://h:1/?a=b"


class TestPlaceholdersAndPrecedence:
    def test_empty_placeholder_leaves_variable_unset(self, tmp_path, clean_env):
        assert load_env_file(_write(tmp_path, "FACTAGENT_EMPTY=\n")) == ()
        assert "FACTAGENT_EMPTY" not in os.environ

    def test_existing_environment_wins_by_default(self, tmp_path, clean_env):
        clean_env.setenv("FACTAGENT_A", "from-shell")
        assert load_env_file(_write(tmp_path, "FACTAGENT_A=from-file\n")) == ()
        assert os.environ["FACTAGENT_A"] == "from-shell"

    def test_override_replaces_existing_value(self, tmp_path, clean_env):
        clean_env.setenv("FACTAGENT_A", "from-shell")
        assert load_env_file(_write(tmp_path, "FACTAGENT_A=from-file\n"), override=True) == (
            "FACTAGENT_A",
        )
        assert os.environ["FACTAGENT_A"] == "from-file"

    def test_missing_file_sets_nothing(self, tmp_path, clean_env):
        assert load_env_file(tmp_path / "absent.env") == ()


class TestNeverLeaksValues:
    def test_malformed_line_error_names_line_not_value(self, tmp_path, clean_env):
        with pytest.raises(EnvFileError, match=r"\.env:2") as excinfo:
            load_env_file(_write(tmp_path, f"FACTAGENT_A=1\n{SECRET}\n"))
        assert SECRET not in str(excinfo.value)

    def test_invalid_key_error_does_not_echo_the_value(self, tmp_path, clean_env):
        with pytest.raises(EnvFileError, match="invalid variable name") as excinfo:
            load_env_file(_write(tmp_path, f"BAD-KEY={SECRET}\n"))
        assert SECRET not in str(excinfo.value)

    def test_log_output_contains_names_but_not_values(self, tmp_path, clean_env, caplog):
        caplog.set_level("INFO", logger="agent_evals")
        load_env_file(_write(tmp_path, f"FACTAGENT_A={SECRET}\n"))
        assert "FACTAGENT_A" in caplog.text
        assert SECRET not in caplog.text
