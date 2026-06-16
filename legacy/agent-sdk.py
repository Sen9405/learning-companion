#!/usr/bin/env python3
"""
Phase 1 — Agent loop через OpenAI Agents SDK
"""

import os
import json
from datetime import datetime
from agents import Agent, Runner, function_tool


@function_tool
def get_current_time():
    """Получить текущее время и дату."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@function_tool
def web_search(query: str):
    """Поиск информации в интернете."""
    from ddgs import DDGS
    results = list(DDGS().text(query, max_results=2,
                               headers={"User-Agent": "Mozilla/5.0"}))
    if not results:
        return "Ничего не найдено."
    text_parts = []
    for i, r in enumerate(results):
        text_parts.append(f"{i+1}. {r['title']}")
        text_parts.append(f"   {r['body'][:120]}")
        text_parts.append(f"   {r['href']}")
    return "\n".join(text_parts)


@function_tool
def read_local_file(path: str, max_lines: int = 20):
    """Прочитать содержимое текстового файла."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return f"Файл не найден: {path}"
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    total = len(lines)
    content = "".join(lines[:max_lines])
    result = f"Файл: {path} ({total} строк, показано {min(max_lines, total)})\n\n{content}"
    if total > max_lines:
        result += f"\n... (+{total - max_lines} строк)"
    return result[:500]


# ──────── Настройки ────────
os.environ["OPENAI_API_KEY"] = "sk-0ef79154289240c2a2743c50f44b625a"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

# ──────── Создаём агента ────────
agent = Agent(
    name="Помощник",
    instructions="Ты — полезный AI-агент. Отвечай на русском, кратко и по делу.",
    tools=[get_current_time, web_search, read_local_file],
)

# ──────── Запуск ────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "Который час?"
    print(f"🔍 Запрос: {query}")
    result = Runner.run_sync(agent, query)
    print(f"\n=== ОТВЕТ ===")
    print(result.final_output)
