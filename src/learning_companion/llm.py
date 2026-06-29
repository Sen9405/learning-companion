"""LLM client — DeepSeek API wrapper with cost tracking and production guards."""

from __future__ import annotations

import os
import time
from typing import Optional

from learning_companion.ledger import LlmCallRecord, RunLedger
from learning_companion.prompt_cache import PromptCache, make_cache_key
from learning_companion.routing import ModelRouter
from learning_companion.security import redact_secrets
from learning_companion.settings import get_settings

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

DEEPSEEK_API_KEY = ""
DEEPSEEK_MODEL = "deepseek-v4-flash"

# Цены DeepSeek V4 Flash (US $ за 1M токенов)
DEEPSEEK_INPUT_PRICE_PER_1M = 0.14
DEEPSEEK_OUTPUT_PRICE_PER_1M = 0.28

# Загружаем API ключи из .env файлов
for env_file in [
    os.path.expanduser("~/.hermes/.env"),
    os.path.expanduser("~/.env"),
]:
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k == "DEEPSEEK_API_KEY" and not DEEPSEEK_API_KEY:
                        DEEPSEEK_API_KEY = v
                    elif k in ("OPENAI_API_KEY", "LLM_API_KEY"):
                        os.environ.setdefault(k, v)

_client: Optional[OpenAI] = None

# Счётчики стоимости (глобальные на процесс)
_total_prompt_tokens = 0
_total_completion_tokens = 0
_total_cost = 0.0
_total_llm_calls = 0


class BudgetExceededError(RuntimeError):
    """Raised when a run exceeds configured hard budget limits."""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if OpenAI is None:
            raise ImportError("openai package not installed. pip install openai")
        settings = get_settings()
        api_key = DEEPSEEK_API_KEY or os.environ.get("OPENAI_API_KEY") or ""
        if not api_key:
            raise ValueError(
                "No API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY env var."
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=settings.deepseek_base_url,
        )
    return _client


def _get_model() -> str:
    configured = get_settings().deepseek_model
    return configured or DEEPSEEK_MODEL


def _calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000) * DEEPSEEK_INPUT_PRICE_PER_1M + (completion_tokens / 1_000_000) * DEEPSEEK_OUTPUT_PRICE_PER_1M


def _check_budget(*, next_cost: float = 0.0, next_prompt_tokens: int = 0, next_completion_tokens: int = 0) -> None:
    settings = get_settings()
    projected_cost = _total_cost + next_cost
    projected_prompt_tokens = _total_prompt_tokens + next_prompt_tokens
    projected_completion_tokens = _total_completion_tokens + next_completion_tokens

    if projected_cost > settings.max_cost_usd_per_run:
        raise BudgetExceededError(
            f"Run exceeded cost budget: ${projected_cost:.6f} > ${settings.max_cost_usd_per_run:.6f}"
        )
    if projected_prompt_tokens > settings.max_prompt_tokens_per_run:
        raise BudgetExceededError(
            f"Run exceeded prompt token budget: {projected_prompt_tokens} > {settings.max_prompt_tokens_per_run}"
        )
    if projected_completion_tokens > settings.max_completion_tokens_per_run:
        raise BudgetExceededError(
            f"Run exceeded completion token budget: {projected_completion_tokens} > {settings.max_completion_tokens_per_run}"
        )
    if _total_llm_calls >= settings.max_llm_calls_per_run:
        raise BudgetExceededError(
            f"Run exceeded LLM call budget: {_total_llm_calls + 1} > {settings.max_llm_calls_per_run}"
        )


def _record_ledger(
    *,
    run_id: str,
    stage: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    latency: float,
    cache_hit: bool,
    error: str | None = None,
) -> None:
    settings = get_settings()
    ledger = RunLedger(settings.run_ledger_db)
    ledger.record_llm_call(
        LlmCallRecord(
            run_id=run_id,
            stage=stage,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            latency=latency,
            cache_hit=cache_hit,
            error=redact_secrets(error) if error else None,
        )
    )


def llm_call(
    system: str,
    messages: list[dict],
    response_model: type | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    *,
    stage: str = "unknown",
    run_id: str = "default",
    cache: bool | None = None,
) -> tuple[str, dict]:
    """Call DeepSeek, return (text, meta).

    meta contains: prompt_tokens, completion_tokens, cost, latency, cache_hit.
    """
    global _total_prompt_tokens, _total_completion_tokens, _total_cost, _total_llm_calls
    settings = get_settings()
    route = None
    if settings.router_enabled:
        route = ModelRouter(settings).route(
            stage=stage,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        model = route.model
        max_tokens = route.max_tokens
        temperature = route.temperature
    else:
        model = _get_model()
    response_format = {"type": "json_object"} if response_model else None
    should_cache = settings.enable_prompt_cache if cache is None else cache
    if route is not None and cache is None:
        should_cache = settings.enable_prompt_cache or route.cache
    should_cache = bool(should_cache and temperature == 0)

    _check_budget()

    cache_key = None
    if should_cache:
        cache_key = make_cache_key(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        prompt_cache = PromptCache(settings.prompt_cache_db, ttl_days=settings.prompt_cache_ttl_days)
        cached = prompt_cache.get(cache_key)
        if cached is not None:
            _total_llm_calls += 1
            meta = dict(cached.meta)
            original_cost = meta.get("cost", 0.0)
            meta.update({
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost": 0.0,
                "latency": 0.0,
                "cache_hit": True,
                "original_cost": original_cost,
            })
            _record_ledger(
                run_id=run_id,
                stage=stage,
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0,
                latency=0.0,
                cache_hit=True,
            )
            return cached.text, meta

    client = _get_client()
    t0 = time.time()
    full = [{"role": "system", "content": system}] + messages  # type: ignore
    kwargs = dict(model=model, messages=full, max_tokens=max_tokens, temperature=temperature)
    if response_format:
        kwargs["response_format"] = response_format

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        latency = time.time() - t0
        _record_ledger(
            run_id=run_id,
            stage=stage,
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            latency=latency,
            cache_hit=False,
            error=redact_secrets(exc),
        )
        raise

    latency = time.time() - t0
    usage = resp.usage
    pt = usage.prompt_tokens if usage else 0
    ct = usage.completion_tokens if usage else 0
    cost = _calculate_cost(pt, ct)

    _check_budget(next_cost=cost, next_prompt_tokens=pt, next_completion_tokens=ct)

    _total_llm_calls += 1
    _total_prompt_tokens += pt
    _total_completion_tokens += ct
    _total_cost += cost

    text = redact_secrets(resp.choices[0].message.content or "")
    meta = {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cost": round(cost, 6),
        "latency": round(latency, 2),
        "cache_hit": False,
    }

    if should_cache and cache_key:
        safe_meta = {key: redact_secrets(value) if isinstance(value, str) else value for key, value in meta.items()}
        PromptCache(settings.prompt_cache_db, ttl_days=settings.prompt_cache_ttl_days).set(cache_key, text=text, meta=safe_meta)

    _record_ledger(
        run_id=run_id,
        stage=stage,
        model=model,
        prompt_tokens=pt,
        completion_tokens=ct,
        cost=round(cost, 6),
        latency=round(latency, 2),
        cache_hit=False,
    )
    return text, meta


def get_cost() -> dict:
    return {
        "prompt_tokens": _total_prompt_tokens,
        "completion_tokens": _total_completion_tokens,
        "total_tokens": _total_prompt_tokens + _total_completion_tokens,
        "cost": round(_total_cost, 6),
        "llm_calls": _total_llm_calls,
    }


def reset_counters() -> None:
    global _total_prompt_tokens, _total_completion_tokens, _total_cost, _total_llm_calls
    _total_prompt_tokens = 0
    _total_completion_tokens = 0
    _total_cost = 0.0
    _total_llm_calls = 0
