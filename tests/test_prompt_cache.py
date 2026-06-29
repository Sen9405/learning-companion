"""Tests for SQLite prompt cache."""

from __future__ import annotations

from learning_companion.prompt_cache import PromptCache, make_cache_key


def test_make_cache_key_is_stable_and_changes_on_temperature():
    messages = [{"role": "user", "content": "hello"}]

    key1 = make_cache_key(
        model="deepseek-v4-flash",
        system="system",
        messages=messages,
        max_tokens=100,
        temperature=0,
        response_format=None,
    )
    key2 = make_cache_key(
        model="deepseek-v4-flash",
        system="system",
        messages=messages,
        max_tokens=100,
        temperature=0,
        response_format=None,
    )
    key3 = make_cache_key(
        model="deepseek-v4-flash",
        system="system",
        messages=messages,
        max_tokens=100,
        temperature=0.7,
        response_format=None,
    )

    assert key1 == key2
    assert key1 != key3
    assert len(key1) == 64


def test_prompt_cache_round_trip(tmp_path):
    cache = PromptCache(tmp_path / "prompt-cache.sqlite3")
    key = "abc123"

    assert cache.get(key) is None

    cache.set(key, text="cached response", meta={"cost": 0.001})
    cached = cache.get(key)

    assert cached is not None
    assert cached.text == "cached response"
    assert cached.meta["cost"] == 0.001


def test_prompt_cache_ttl_expires_immediately(tmp_path):
    cache = PromptCache(tmp_path / "prompt-cache.sqlite3", ttl_days=0)

    cache.set("expired", text="old", meta={})

    assert cache.get("expired") is None
