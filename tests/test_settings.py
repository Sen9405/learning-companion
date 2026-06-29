"""Tests for production hardening settings."""

from __future__ import annotations

from learning_companion.settings import Settings


def test_settings_defaults_are_safe():
    settings = Settings.from_env({})

    assert settings.deepseek_model == "deepseek-v4-flash"
    assert settings.deepseek_base_url == "https://api.deepseek.com/v1"
    assert settings.enable_prompt_cache is False
    assert settings.max_cost_usd_per_run > 0
    assert settings.max_llm_calls_per_run > 0


def test_settings_reads_environment_overrides(tmp_path):
    cache_db = tmp_path / "cache.sqlite3"
    ledger_db = tmp_path / "ledger.sqlite3"

    settings = Settings.from_env({
        "DEEPSEEK_MODEL": "deepseek-v4-flash",
        "DEEPSEEK_BASE_URL": "https://example.test/v1",
        "LC_ENABLE_PROMPT_CACHE": "true",
        "LC_PROMPT_CACHE_DB": str(cache_db),
        "LC_RUN_LEDGER_DB": str(ledger_db),
        "LC_MAX_COST_USD_PER_RUN": "0.123",
        "LC_MAX_LLM_CALLS_PER_RUN": "7",
    })

    assert settings.deepseek_model == "deepseek-v4-flash"
    assert settings.deepseek_base_url == "https://example.test/v1"
    assert settings.enable_prompt_cache is True
    assert settings.prompt_cache_db == cache_db
    assert settings.run_ledger_db == ledger_db
    assert settings.max_cost_usd_per_run == 0.123
    assert settings.max_llm_calls_per_run == 7
