#!/usr/bin/env python3
"""
Phase 1 — Мой первый сырой agent loop
"""

import os
import json
import sys
from datetime import datetime
from openai import OpenAI

# ──────── Настройки ────────
API_KEY = "sk-0ef79154289240c2a2743c50f44b625a"
MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"
MAX_TURNS = 7

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ──────── Описание инструментов для модели ────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Получить текущее время и дату.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Поиск информации в интернете.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос (3-6 слов)",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_local_file",
            "description": "Прочитать содержимое текстового файла.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Путь к файлу",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Сколько строк прочитать (по умолч. 20)",
                    },
                },
                "required": ["path"],
            },
        },
    },
]

# ──────── Выполнение инструментов ────────
def execute_tool(name, args):
    """Выполняет инструмент и возвращает текст-результат для модели"""

    # ─── 1. Узнать время ───
    if name == "get_current_time":
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ─── 2. Поиск в интернете ───
    elif name == "web_search":
        try:
            from ddgs import DDGS
            results = list(DDGS().text(
                args["query"], max_results=2,
                headers={"User-Agent": "Mozilla/5.0"}
            ))
        except Exception:
            from duckduckgo_search import DDGS
            results = list(DDGS().text(
                args["query"], max_results=2
            ))
        if not results:
            return "Ничего не найдено."
        text_parts = []
        for i, r in enumerate(results):
            text_parts.append(f"{i+1}. {r['title']}")
            text_parts.append(f"   {r['body'][:120]}")
            text_parts.append(f"   {r['href']}")
        return "\n".join(text_parts)

    # ─── 3. Чтение файла ───
    elif name == "read_local_file":
        path = os.path.expanduser(args["path"])
        if not os.path.isfile(path):
            return f"Файл не найден: {path}"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        max_lines = args.get("max_lines", 20)
        total = len(lines)
        content = "".join(lines[:max_lines])
        result = f"Файл: {path} ({total} строк, показано {min(max_lines, total)})\n\n{content}"
        if total > max_lines:
            result += f"\n... (+{total - max_lines} строк)"
        return result[:500]

    else:
        return f"Неизвестный инструмент: {name}"

# ──────── Главный цикл ────────
def agent_loop(user_input):
    """Основной цикл: модель → инструмент → модель → ... → ответ"""

    # Собираем начальные сообщения
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — полезный AI-агент. У тебя есть инструменты:\n"
                "1) get_current_time — узнать время\n"
                "2) web_search — поиск в интернете\n"
                "3) read_local_file — чтение файлов\n\n"
                "Если данных достаточно — сразу отвечай.\n"
                "Отвечай на русском, кратко и по делу."
            ),
        },
        {"role": "user", "content": user_input},
    ]

    # Цикл: максимум MAX_TURNS шагов
    for turn in range(1, MAX_TURNS + 1):
        print(f"\n--- Шаг {turn} ---")

        # 1. Отправляем всё модели
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        # 2. Получаем ответ
        msg = response.choices[0].message

        # 3. Если модель просто ответила текстом — готово!
        if not msg.tool_calls:
            return msg.content or ""

        # 4. Если модель хочет вызвать инструмент — печатаем и выполняем
        print(f"🧠 Модель вызывает инструмент!")

        # Добавляем ответ модели в историю
        messages.append(msg)

        # 5. Проходимся по всем вызовам инструментов
        for tc in msg.tool_calls:
            func_name = tc.function.name
            func_args = json.loads(tc.function.arguments)
            print(f"🔧 {func_name}({func_args})")

            # Выполняем инструмент
            try:
                result = execute_tool(func_name, func_args)
            except Exception as e:
                result = f"Ошибка при вызове {func_name}: {e}"
            print(f"📦 {result[:100]}...")

            # Добавляем результат в историю
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result[:500],
            })

    # Если дошли до сюда — кончились шаги
    return "Достигнут лимит шагов. Попробуй уточнить запрос."

# ──────── Запуск ────────
if __name__ == "__main__":
    # Берём всё, что написали после python3 agent-raw.py
    query = " ".join(sys.argv[1:])

    # Если ничего не написали — спросим время
    if not query:
        query = "Который час?"

    print(f"🔍 Запрос: {query}")

    # Запускаем агента
    answer = agent_loop(query)

    print(f"\n=== ОТВЕТ ===")
    print(answer)
