"""LLM client — DeepSeek API wrapper with cost tracking."""

from __future__ import annotations

import os
import time
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

DEEPSEEK_API_KEY = ""
DEEPSEEK_MODEL = "deepseek-chat"

# Цены DeepSeek-chat (US $ за 1M токенов)
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


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if OpenAI is None:
            raise ImportError("openai package not installed. pip install openai")
        api_key = DEEPSEEK_API_KEY or os.environ.get("OPENAI_API_KEY") or ""
        if not api_key:
            raise ValueError(
                "No API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY env var."
            )
        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
    return _client


def llm_call(
    system: str,
    messages: list[dict],
    response_model: type | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> tuple[str, dict]:
    """Call DeepSeek, return (text, meta).

    meta contains: prompt_tokens, completion_tokens, cost, latency.
    """
    global _total_prompt_tokens, _total_completion_tokens, _total_cost
    client = _get_client()
    t0 = time.time()
    full = [{"role": "system", "content": system}] + messages  # type: ignore
    kwargs = dict(model=DEEPSEEK_MODEL, messages=full, max_tokens=max_tokens, temperature=temperature)
    if response_model:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    latency = time.time() - t0
    usage = resp.usage
    pt = usage.prompt_tokens if usage else 0
    ct = usage.completion_tokens if usage else 0
    cost = (pt / 1_000_000) * DEEPSEEK_INPUT_PRICE_PER_1M + (ct / 1_000_000) * DEEPSEEK_OUTPUT_PRICE_PER_1M

    _total_prompt_tokens += pt
    _total_completion_tokens += ct
    _total_cost += cost

    text = resp.choices[0].message.content or ""
    meta = {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cost": round(cost, 6),
        "latency": round(latency, 2),
    }
    return text, meta


def get_cost() -> dict:
    return {
        "prompt_tokens": _total_prompt_tokens,
        "completion_tokens": _total_completion_tokens,
        "total_tokens": _total_prompt_tokens + _total_completion_tokens,
        "cost": round(_total_cost, 6),
    }


def reset_counters() -> None:
    global _total_prompt_tokens, _total_completion_tokens, _total_cost
    _total_prompt_tokens = 0
    _total_completion_tokens = 0
    _total_cost = 0.0
