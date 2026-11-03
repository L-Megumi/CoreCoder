"""Keep offline unit tests independent of a developer's live provider configuration."""

import pytest


@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch):
    for name in (
        "CORECODER_MODEL", "CORECODER_PROVIDER", "CORECODER_MAX_TOKENS",
        "CORECODER_TEMPERATURE", "CORECODER_MAX_CONTEXT", "CORECODER_API_KEY",
        "CORECODER_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("corecoder.config._load_dotenv", lambda: None)
