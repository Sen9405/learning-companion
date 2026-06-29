"""Production hardening settings for Learning Companion."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def _default_state_dir() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "learning-companion"


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _as_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    """Runtime settings controlled by environment variables."""

    deepseek_model: str
    deepseek_base_url: str
    enable_prompt_cache: bool
    prompt_cache_db: Path
    run_ledger_db: Path
    prompt_cache_ttl_days: int
    max_cost_usd_per_run: float
    max_prompt_tokens_per_run: int
    max_completion_tokens_per_run: int
    max_llm_calls_per_run: int
    router_enabled: bool
    router_simple_model: str
    router_standard_model: str
    router_complex_model: str
    router_simple_max_chars: int
    router_complex_min_chars: int
    router_simple_max_tokens: int
    router_standard_max_tokens: int
    router_complex_max_tokens: int
    alert_cost_usd: float
    alert_error_count: int
    alert_avg_latency_seconds: float
    alert_min_cache_hit_rate: float

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from an environment-like mapping."""
        source = os.environ if env is None else env
        state_dir = Path(source.get("LC_STATE_DIR", str(_default_state_dir()))).expanduser()

        return cls(
            deepseek_model=source.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            deepseek_base_url=source.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            enable_prompt_cache=_as_bool(source.get("LC_ENABLE_PROMPT_CACHE"), default=False),
            prompt_cache_db=Path(source.get("LC_PROMPT_CACHE_DB", str(state_dir / "prompt-cache.sqlite3"))).expanduser(),
            run_ledger_db=Path(source.get("LC_RUN_LEDGER_DB", str(state_dir / "run-ledger.sqlite3"))).expanduser(),
            prompt_cache_ttl_days=_as_int(source.get("LC_PROMPT_CACHE_TTL_DAYS"), 30),
            max_cost_usd_per_run=_as_float(source.get("LC_MAX_COST_USD_PER_RUN"), 0.05),
            max_prompt_tokens_per_run=_as_int(source.get("LC_MAX_PROMPT_TOKENS_PER_RUN"), 200_000),
            max_completion_tokens_per_run=_as_int(source.get("LC_MAX_COMPLETION_TOKENS_PER_RUN"), 50_000),
            max_llm_calls_per_run=_as_int(source.get("LC_MAX_LLM_CALLS_PER_RUN"), 30),
            router_enabled=_as_bool(source.get("LC_ROUTER_ENABLED"), default=True),
            router_simple_model=source.get("LC_ROUTER_SIMPLE_MODEL", "deepseek-v4-flash"),
            router_standard_model=source.get("LC_ROUTER_STANDARD_MODEL", "deepseek-v4-flash"),
            router_complex_model=source.get("LC_ROUTER_COMPLEX_MODEL", "deepseek-v4-flash"),
            router_simple_max_chars=_as_int(source.get("LC_ROUTER_SIMPLE_MAX_CHARS"), 2_000),
            router_complex_min_chars=_as_int(source.get("LC_ROUTER_COMPLEX_MIN_CHARS"), 12_000),
            router_simple_max_tokens=_as_int(source.get("LC_ROUTER_SIMPLE_MAX_TOKENS"), 2_048),
            router_standard_max_tokens=_as_int(source.get("LC_ROUTER_STANDARD_MAX_TOKENS"), 4_096),
            router_complex_max_tokens=_as_int(source.get("LC_ROUTER_COMPLEX_MAX_TOKENS"), 8_192),
            alert_cost_usd=_as_float(source.get("LC_ALERT_COST_USD"), 0.03),
            alert_error_count=_as_int(source.get("LC_ALERT_ERROR_COUNT"), 0),
            alert_avg_latency_seconds=_as_float(source.get("LC_ALERT_AVG_LATENCY_SECONDS"), 30.0),
            alert_min_cache_hit_rate=_as_float(source.get("LC_ALERT_MIN_CACHE_HIT_RATE"), 0.0),
        )


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return cached runtime settings."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.from_env()
    return _SETTINGS


def reset_settings_cache() -> None:
    """Reset cached settings; useful for tests after env changes."""
    global _SETTINGS
    _SETTINGS = None
