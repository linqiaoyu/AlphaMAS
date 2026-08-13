"""Shared pytest fixtures that prevent CI hangs when API keys are absent."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from scripts.pytest_interpreter_guard import is_known_bad_pytest_interpreter


def pytest_sessionstart(session):
    """Fail fast on the interpreter known to hang during test collection.

    Python 3.14 currently stalls while importing the eager Agent/Graph package
    surface used by the M0 contract tests. The formal experiment still uses
    Python 3.12, but this test-only protection does not block 3.10--3.13.
    """
    if is_known_bad_pytest_interpreter(sys.version_info):
        pytest.exit(
            "Python 3.14 is known to stall while collecting the eager "
            "tradingagents Agent/Graph imports. Use a supported Python "
            "3.10--3.13 interpreter; formal experiments require Python 3.12. "
            "Run `uv sync --frozen --extra dev --python 3.12` for the formal suite.",
            returncode=4,
        )


def pytest_configure(config):
    for marker in ("unit", "integration", "smoke"):
        config.addinivalue_line("markers", f"{marker}: {marker}-level tests")


_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_CN_API_KEY",
    "ZHIPU_API_KEY",
    "ZHIPU_CN_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
)


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    for env_var in _API_KEY_ENV_VARS:
        # `or` not a .get default: an env var present but empty (e.g. a key left
        # blank in a .env copied from .env.example) must still get the placeholder.
        monkeypatch.setenv(env_var, os.environ.get(env_var) or "placeholder")


@pytest.fixture(autouse=True)
def _isolate_config():
    """Reset the global dataflows config before and after each test.

    ``set_config`` merges (it never clears keys absent from the override), so a
    test that sets e.g. ``tool_vendors`` would otherwise leak into later tests
    and make routing behavior order-dependent. Replace the global outright so
    every test starts from a clean DEFAULT_CONFIG.
    """
    import copy

    import tradingagents.dataflows.config as config_module
    import tradingagents.default_config as default_config

    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture()
def mock_llm_client():
    client = MagicMock()
    client.get_llm.return_value = MagicMock()
    with patch(
        "tradingagents.llm_clients.factory.create_llm_client",
        return_value=client,
    ):
        yield client
