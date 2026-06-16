#!/usr/bin/env python3
"""
Learning Companion v2 — Phase 2 Multi-Agent Learning Note Generator

Архитектура (LangGraph граф):
  1. Planner — определяет, какой контент и откуда брать
  2. Fetcher — 3 сабагента (YouTube, Web, PDF)
  3. Analyst — извлекает концепты, связи, вопросы, глоссарий
  4. Writer — сохраняет в Obsidian + возвращает для Telegram

Phase 2 фичи:
  - LangGraph граф с TypedDict состоянием
  - PostgresSaver persistence (между сессиями)
  - Human-in-the-loop (interrupt) на анализе и сохранении
  - Phoenix/OpenTelemetry tracing с семантическими span'ами и inferred cost
  - Telegram-delivered результаты

Использование:
  python3 learning_companion_v2.py run --url "https://youtu.be/..."
  python3 learning_companion_v2.py run --url "https://arxiv.org/..."
  python3 learning_companion_v2.py run --text "some text to analyze"
  python3 learning_companion_v2.py resume <run_id> resume-analyst|reject-analyst|save-writer|skip-save
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import date
from typing import Any, Dict, List, Literal, Optional, TypedDict
from uuid import uuid4

# ---------------------------------------------------------------------------
# DeepSeek / LLM helpers
# ---------------------------------------------------------------------------

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
            base_url="https://api.deepseek.com",
        )
    return _client


# Глобальные счётчики токенов за прогон
_input_tokens = 0
_output_tokens = 0


def llm_call(system: str, user: str, max_tokens: int = 4096) -> str:
    """Call DeepSeek with a system + user prompt, return text."""
    global _input_tokens, _output_tokens
    client = _get_client()
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    usage = getattr(resp, "usage", None)
    if usage:
        _input_tokens += usage.prompt_tokens or 0
        _output_tokens += usage.completion_tokens or 0
    return resp.choices[0].message.content or ""


def get_cost() -> Dict[str, Any]:
    """Возвращает стоимость прогона на основе глобальных счётчиков."""
    input_cost = (_input_tokens / 1_000_000) * DEEPSEEK_INPUT_PRICE_PER_1M
    output_cost = (_output_tokens / 1_000_000) * DEEPSEEK_OUTPUT_PRICE_PER_1M
    return {
        "input_tokens": _input_tokens,
        "output_tokens": _output_tokens,
        "total_tokens": _input_tokens + _output_tokens,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(input_cost + output_cost, 6),
    }


def reset_counters():
    """Сбрасывает счётчики токенов для нового прогона."""
    global _input_tokens, _output_tokens
    _input_tokens = 0
    _output_tokens = 0


# ---------------------------------------------------------------------------
# Telegram отправка
# ---------------------------------------------------------------------------

_TELEGRAM_BOT_TOKEN = ""
_TELEGRAM_CHAT_ID = "144288459"  # Telegram ID Ди

# Загружаем TELEGRAM_BOT_TOKEN из .env
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
                    if k == "TELEGRAM_BOT_TOKEN" and not _TELEGRAM_BOT_TOKEN:
                        _TELEGRAM_BOT_TOKEN = v


def send_telegram(text: str, title: str = "") -> bool:
    """Отправляет сообщение в Telegram через Bot API (прямой curl).

    Отправляет ровно один раз.
    """
    if not _TELEGRAM_BOT_TOKEN:
        print("⚠️  TELEGRAM_BOT_TOKEN не найден. Сообщение не отправлено.")
        return False

    import subprocess
    import json

    payload = json.dumps({
        "chat_id": _TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    })

    try:
        r = subprocess.run(
            ["curl", "-s", "-X", "POST",
             f"https://api.telegram.org/bot{_TELEGRAM_BOT_TOKEN}/sendMessage",
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True, text=True, timeout=15,
        )
        result = json.loads(r.stdout)
        if result.get("ok"):
            return True
        print(f"⚠️  Telegram error: {result.get('description', 'unknown')}")
        return False
    except Exception as e:
        print(f"⚠️  Не удалось отправить в Telegram: {e}")
        return False


# Максимум частей для отправки анализа через Telegram
MAX_TELEGRAM_PARTS = 8
TELEGRAM_PART_SIZE = 3900  # чуть меньше лимита для запаса


def send_telegram_long(text: str, header: str = "") -> None:
    """Отправляет длинный текст в Telegram частями (до 8 частей по ~3900 символов).

    Первая часть начинается с header (если есть), остальные — продолжение.
    """
    if not _TELEGRAM_BOT_TOKEN:
        print("⚠️  TELEGRAM_BOT_TOKEN не найден. Сообщение не отправлено.")
        return

    # Экранируем Obsidian-стиль ссылок [[ ]] до отправки,
    # чтобы Markdown не ломался на каждой части
    text = _escape_telegram_md(text)
    header = _escape_telegram_md(header)

    parts = []
    if header:
        remaining = TELEGRAM_PART_SIZE - len(header) - 2
        if len(text) <= remaining:
            parts.append(header + "\n\n" + text)
            text = ""  # !!! обнуляем, чтобы while не добавил дубль
        else:
            parts.append(header + "\n\n" + text[:remaining])
            text = text[remaining:]
    else:
        if not text:
            return

    while text and len(parts) < MAX_TELEGRAM_PARTS:
        chunk = text[:TELEGRAM_PART_SIZE]
        parts.append(chunk)
        text = text[TELEGRAM_PART_SIZE:]

    if text:
        # Последняя часть — обрезаем с пометкой
        parts[-1] = parts[-1][:TELEGRAM_PART_SIZE - 100]
        parts[-1] += f"\n\n*... (продолжение обрезано, всего {len(text)} символов не поместилось)*"

    for i, part in enumerate(parts):
        success = send_telegram(part)


def notify_telegram(state: 'LearningState') -> None:
    """Формирует и отправляет обзор в Telegram (без вопросов)."""
    analysis = state.get("analysis", "") or ""
    url = state.get("input_url", "")
    cost = get_cost()

    source_emoji = "🎬" if state.get("content_source") == "youtube" else "📄"

    # Убираем секцию с вопросами из анализа для отправки в Telegram
    clean_analysis = _strip_questions_section(analysis)

    # Формат заголовка
    title_line = ""
    for line in analysis.split("\n"):
        if line.startswith("# ") and not line.startswith("## "):
            title_line = line[2:].strip()[:100]
            break

    header = f"{source_emoji} **{title_line or 'Анализ контента'}**"

    if url:
        header += f"\n🔗 {url}"

    header += f"\n━━━━━━━━━━━━━━━━━━━━\n"

    # Добавляем стоимость и подсказку про вопросы в конец
    footer = (
        f"\n━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 ${cost['total_cost']:.6f} | {cost['total_tokens']:,} токенов\n"
        f"{'📁 Obsidian сохранён' if state.get('saved') else '📝 Только просмотр'}\n"
        f"🆔 `{state.get('run_id', '')[:8]}...`"
    )

    text_to_send = clean_analysis + footer
    send_telegram_long(text_to_send, header=header)


def _escape_telegram_md(text: str) -> str:
    """Экранирует символы, ломающие Telegram Markdown.

    Telegram Markdown (v1) не переваривает:
    - [[ и ]] (Obsidian-стиль ссылки)
    - _ внутри слов
    - Некорректные [ ] без URL
    """
    import re
    # Заменяем [[ на « и ]] на » — безопасно для Telegram
    text = text.replace("[[", "«").replace("]]", "»")
    return text


def _strip_questions_section(analysis: str) -> str:
    """Удаляет секцию вопросов для самопроверки из анализа.

    DeepSeek генерирует вопросы с ответами в секции '## ❓ Вопросы для самопроверки'.
    В Telegram мы шлём только обзор без вопросов, вопросы — отдельно по команде check.
    """
    if "## ❓ Вопросы для самопроверки" not in analysis:
        return analysis

    parts = analysis.split("## ❓ Вопросы для самопроверки", 1)
    before = parts[0].rstrip()
    after = parts[1] if len(parts) > 1 else ""

    # Находим следующую секцию после вопросов (если есть)
    next_section = ""
    for section_marker in ["## 📖 Глоссарий", "## 📎 Источник", "## 🔗 Связи"]:
        if section_marker in after:
            next_section = section_marker
            break

    if next_section:
        # Обрезаем до следующей секции
        return before + "\n\n" + after[after.index(next_section):]
    else:
        # Вопросы были последней секцией — просто убираем
        return before


# ---------------------------------------------------------------------------
# Last note tracker
# ---------------------------------------------------------------------------

LAST_NOTE_FILE = os.path.expanduser(
    "~/.hermes/cron/output/learning_companion_last.json"
)


def save_last_note(run_id: str, title: str, path: str, url: str = "",
                    analysis_preview: str = "") -> None:
    """Сохраняет информацию о последней обработанной заметке."""
    import json
    data = {
        "run_id": run_id,
        "title": title,
        "path": path,
        "url": url,
        "analysis_preview": analysis_preview[:200] if analysis_preview else "",
        "timestamp": date.today().isoformat(),
    }
    os.makedirs(os.path.dirname(LAST_NOTE_FILE), exist_ok=True)
    with open(LAST_NOTE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_last_note() -> dict:
    """Возвращает информацию о последней заметке или пустой словарь."""
    import json
    if not os.path.exists(LAST_NOTE_FILE):
        return {}
    try:
        with open(LAST_NOTE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception):
        return {}


# ---------------------------------------------------------------------------
# Postgres connection URI
# ---------------------------------------------------------------------------

POSTGRES_URI = os.environ.get(
    "LC_DATABASE_URL",
    "postgresql://sen:learning_companion_pass@localhost:5432/learning_companion",
)


# ---------------------------------------------------------------------------
# Long-Term Memory (Postgres + JSON cache)
# ---------------------------------------------------------------------------

LTM_CACHE_FILE = os.path.expanduser(
    "~/.hermes/cron/output/ltm_cache.json"
)


class LongTermMemory:
    """Long-term memory для учебных заметок.

    Хранит заметки в Postgres (таблица notes) и кэширует в JSON
    для быстрой загрузки в контекст агента.
    """

    def __init__(self, db_url: str = POSTGRES_URI):
        self._db_url = db_url

    # ── Postgres ──────────────────────────────────────────

    def _connect(self):
        import psycopg
        return psycopg.connect(self._db_url, autocommit=True)

    def add_note(self, note_id: str, title: str, source: str = "",
                 url: str = "", key_concepts: list = None,
                 summary: str = "") -> bool:
        """Добавляет заметку в LTM."""
        import json
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO notes (id, title, source, url, key_concepts, summary)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                       last_reviewed = CURRENT_DATE,
                       key_concepts = EXCLUDED.key_concepts,
                       summary = EXCLUDED.summary""",
                (note_id, title, source, url,
                 json.dumps(key_concepts or [], ensure_ascii=False),
                 summary),
            )
            conn.commit()
            cur.close()
            conn.close()
            self._refresh_cache()
            return True
        except Exception as e:
            print(f"⚠️  LTM add_note error: {e}")
            return False

    def get_note(self, note_id: str) -> dict:
        """Возвращает заметку по ID."""
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT * FROM notes WHERE id = %s", (note_id,))
            row = cur.fetchone()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            cur.close()
            conn.close()
            if row:
                return self._row_to_dict(row, columns)
            return {}
        except Exception as e:
            print(f"⚠️  LTM get_note error: {e}")
            return {}

    def search_notes(self, query: str, limit: int = 10) -> list:
        """Поиск заметок по названию, концептам или саммари (ILIKE)."""
        try:
            conn = self._connect()
            cur = conn.cursor()
            pattern = f"%{query}%"
            cur.execute(
                """SELECT * FROM notes
                   WHERE title ILIKE %s
                      OR key_concepts ILIKE %s
                      OR summary ILIKE %s
                   ORDER BY last_reviewed DESC
                   LIMIT %s""",
                (pattern, pattern, pattern, limit),
            )
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            cur.close()
            conn.close()
            return [self._row_to_dict(r, columns) for r in rows]
        except Exception as e:
            print(f"⚠️  LTM search_notes error: {e}")
            return []

    def get_all_concepts(self) -> list:
        """Возвращает список всех уникальных концептов из всех заметок."""
        import json
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT key_concepts FROM notes")
            all_concepts = set()
            for (concepts_json,) in cur.fetchall():
                try:
                    concepts = json.loads(concepts_json) if concepts_json else []
                    for c in concepts:
                        all_concepts.add(c)
                except (json.JSONDecodeError, TypeError):
                    pass
            cur.close()
            conn.close()
            return sorted(all_concepts)
        except Exception as e:
            print(f"⚠️  LTM get_all_concepts error: {e}")
            return []

    def get_recent_notes(self, limit: int = 5) -> list:
        """Последние заметки (по created_at)."""
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM notes ORDER BY created_at DESC, last_reviewed DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            cur.close()
            conn.close()
            return [self._row_to_dict(r, columns) for r in rows]
        except Exception as e:
            print(f"⚠️  LTM get_recent_notes error: {e}")
            return []

    def update_mastery(self, note_id: str, delta: int = 10) -> bool:
        """Увеличивает mastery_level заметки."""
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                """UPDATE notes SET
                       mastery_level = LEAST(mastery_level + %s, 100),
                       last_reviewed = CURRENT_DATE
                   WHERE id = %s""",
                (delta, note_id),
            )
            conn.commit()
            cur.close()
            conn.close()
            self._refresh_cache()
            return True
        except Exception as e:
            print(f"⚠️  LTM update_mastery error: {e}")
            return False

    def mark_questions_asked(self, note_id: str) -> bool:
        """Отмечает, что по заметке были сгенерированы вопросы."""
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                "UPDATE notes SET questions_asked = TRUE WHERE id = %s",
                (note_id,),
            )
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"⚠️  LTM mark_questions_asked error: {e}")
            return False

    def _row_to_dict(self, row, columns: list) -> dict:
        """Конвертирует строку Postgres в словарь."""
        import json
        d = dict(zip(columns, row))
        # Десериализуем key_concepts из JSON строки
        if isinstance(d.get("key_concepts"), str):
            try:
                d["key_concepts"] = json.loads(d["key_concepts"])
            except (json.JSONDecodeError, TypeError):
                d["key_concepts"] = []
        # Даты в строки
        for key in ("created_at", "last_reviewed"):
            if key in d and d[key] is not None:
                d[key] = str(d[key])
        return d

    # ── JSON Cache ─────────────────────────────────────────

    def _refresh_cache(self):
        """Обновляет JSON-кэш всех заметок для быстрой загрузки."""
        import json
        try:
            notes = self.get_recent_notes(limit=50)
            concepts = self.get_all_concepts()
            cache = {
                "notes_count": len(notes),
                "total_concepts": len(concepts),
                "notes": notes,
                "all_concepts": concepts,
                "last_updated": date.today().isoformat(),
            }
            os.makedirs(os.path.dirname(LTM_CACHE_FILE), exist_ok=True)
            with open(LTM_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  LTM cache refresh error: {e}")

    def load_cache(self) -> dict:
        """Загружает JSON-кэш."""
        import json
        if not os.path.exists(LTM_CACHE_FILE):
            return {"notes": [], "all_concepts": [], "notes_count": 0}
        try:
            with open(LTM_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"notes": [], "all_concepts": [], "notes_count": 0}

    def format_context(self, limit: int = 3) -> str:
        """Форматирует LTM для вставки в промпт (контекст агента)."""
        cache = self.load_cache()
        notes = cache.get("notes", [])[:limit]
        concepts = cache.get("all_concepts", [])

        if not notes:
            return ""

        lines = ["📚 **Твоя база знаний (Long-Term Memory):**", ""]

        for n in notes:
            title = n.get("title", "Untitled")
            source = n.get("source", "")
            mastery = n.get("mastery_level", 0)
            kc = n.get("key_concepts", [])
            mastery_bar = "█" * (mastery // 10) + "░" * (10 - mastery // 10)
            s = f"• **{title}** ({source}) [{mastery_bar} {mastery}%]"
            if kc:
                s += f"\n  Концепты: {', '.join(kc[:5])}"
            lines.append(s)

        if concepts:
            lines.append("")
            lines.append(f"📌 Всего концептов в базе: {len(concepts)}")
            lines.append(f"   {', '.join(concepts[:10])}")

        return "\n".join(lines)


# Глобальный экземпляр LTM
_ltm = None


def get_ltm() -> LongTermMemory:
    global _ltm
    if _ltm is None:
        _ltm = LongTermMemory()
    return _ltm


# ---------------------------------------------------------------------------
# Question Generator (Check mode)
# ---------------------------------------------------------------------------

QUESTIONS_SYSTEM = """Ты — эксперт по проверке знаний. Твоя задача: на основе учебной заметки составить список вопросов для проверки усвоения материала.

Правила составления вопросов:
1. Вопросы должны проверять **понимание**, а не поверхностное запоминание
2. Вопросы на русском языке
3. Каждый вопрос — отдельный пункт от 1.
4. Не пиши ответы — только вопросы
5. 6-10 вопросов разного уровня сложности:
   - Простые: "Какое определение у концепта X?"
   - Средние: "Как связаны концепты A и B?"
   - Сложные: "Как бы ты применил концепт X в ситуации Y?"
6. Вопросы должны строго соответствовать содержанию заметки — не выдумывай то, чего нет в материале
7. Если предоставлена Long-Term Memory (предыдущие заметки пользователя), включи 1-2 вопроса про **связь текущего материала с ранее изученным**

Формат ответа — только список вопросов, без лишнего текста:

1. **Вопрос?** — простой
2. **Вопрос?** — средний
3. **Вопрос?** — сложный
...

Не добавляй "Ответ:", "Правильно:" или другие подсказки."""

QUESTIONS_LTM_SYSTEM = """Ты — эксперт по проверке знаний. Твоя задача: на основе учебной заметки составить список вопросов для проверки усвоения материала.

Ниже ты получишь:
1. **Текущая заметка** — материал, который пользователь только что изучил
2. **Long-Term Memory** — список его предыдущих учебных заметок (названия, концепты, уровень освоения)

Правила составления вопросов:
1. Вопросы должны проверять **понимание**, а не поверхностное запоминание
2. Вопросы на русском языке
3. Каждый вопрос — отдельный пункт от 1.
4. Не пиши ответы — только вопросы
5. 6-10 вопросов разного уровня сложности:
   - Простые: "Какое определение у концепта X?" (по текущей заметке)
   - Средние: "Как связаны концепты A и B?" (внутри текущей заметки)
   - Сложные: "Как бы ты применил концепт X в ситуации Y?"
6. **Обязательно включи 1-2 вопроса про связь текущего материала с ранее изученным** (из Long-Term Memory)
7. Вопросы должны строго соответствовать содержанию — не выдумывай то, чего нет
8. Названия концептов из LTM используй в формате [[Концепт]]

Формат ответа — только список вопросов:

1. **Вопрос?** — простой
2. **Вопрос?** — средний
3. **Вопрос?** — сложный
..."""


def generate_questions(note_path: str, ltm_context: str = "") -> str:
    """Генерирует вопросы по заметке из Obsidian."""
    if not os.path.exists(note_path):
        return ""

    try:
        with open(note_path, "r", encoding="utf-8") as f:
            content = f.read()

        if len(content) < 100:
            return ""

        if ltm_context:
            prompt = f"**Твоя база знаний (Long-Term Memory):**\n{ltm_context}\n\n---\n\n**Текущая заметка для проверки:**\n{content}"
            return llm_call(QUESTIONS_LTM_SYSTEM, prompt, max_tokens=2048)
        else:
            return llm_call(QUESTIONS_SYSTEM, content, max_tokens=2048)
    except Exception as e:
        print(f"⚠️  Ошибка генерации вопросов: {e}")
        return ""

try:
    from langgraph.graph import END, StateGraph
    from langgraph.types import Command
    from langgraph.errors import GraphInterrupt
except ImportError:
    print("ERROR: langgraph not installed. Run: pip install langgraph")
    sys.exit(1)

# PostgresSaver — чекпоинтер для LangGraph

_checkpointer_instance = None


def get_checkpointer():
    """Возвращает PostgresSaver с sync connection."""
    global _checkpointer_instance
    if _checkpointer_instance is None:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            import psycopg

            conn = psycopg.connect(POSTGRES_URI, autocommit=True)
            _checkpointer_instance = PostgresSaver(conn)
        except Exception as e:
            print(f"⚠️  PostgresSaver не удался ({e}). Использую MemorySaver.")
            from langgraph.checkpoint.memory import MemorySaver
            _checkpointer_instance = MemorySaver()
    return _checkpointer_instance


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class LearningState(TypedDict):
    """Состояние графа Learning Companion."""

    input_url: str
    input_text: str
    run_id: str
    plan: str
    raw_content: str
    content_source: str
    fetch_errors: List[str]
    analysis: str
    obsidian_path: str
    saved: bool
    human_approved_analysis: bool
    human_approved_save: bool
    error: str


def make_initial_state(url: str = "", text: str = "") -> LearningState:
    return LearningState(
        input_url=url,
        input_text=text,
        run_id=str(uuid4()),
        plan="",
        raw_content="",
        content_source="",
        fetch_errors=[],
        analysis="",
        obsidian_path="",
        saved=False,
        human_approved_analysis=False,
        human_approved_save=False,
        error="",
    )


# ---------------------------------------------------------------------------
# OpenTelemetry / Phoenix
# ---------------------------------------------------------------------------

_tracer = None


def _get_tracer():
    global _tracer
    if _tracer is None:
        try:
            from opentelemetry import trace
            _tracer = trace.get_tracer("learning-companion")
        except ImportError:
            _tracer = None
    return _tracer


def span_wrapper(node_name: str, project_name: str = "Learning Companion"):
    """Декоратор для создания OTel span вокруг узла графа."""
    tracer = _get_tracer()
    if tracer is None:
        def decorator(func):
            return func
        return decorator

    def decorator(func):
        def wrapper(state: LearningState) -> LearningState:
            with tracer.start_as_current_span(node_name) as span:
                span.set_attribute("learning_companion.node", node_name)
                span.set_attribute("learning_companion.run_id", state.get("run_id", ""))
                span.set_attribute("learning_companion.source", state.get("content_source", ""))
                span.set_attribute("project.name", project_name)

                if node_name == "planner":
                    span.set_attribute("learning_companion.input_url", state.get("input_url", ""))
                elif node_name == "fetcher":
                    span.set_attribute("learning_companion.raw_content_length", len(state.get("raw_content", "")))
                elif node_name == "writer":
                    span.set_attribute("learning_companion.saved", state.get("saved", False))
                    span.set_attribute("learning_companion.obsidian_path", state.get("obsidian_path", ""))

                result = func(state)

                if node_name == "analyst":
                    span.set_attribute("learning_companion.analysis_length", len(result.get("analysis", "")))
                    span.set_attribute("learning_companion.input_tokens", _input_tokens)
                    span.set_attribute("learning_companion.output_tokens", _output_tokens)
                    cost = get_cost()
                    span.set_attribute("learning_companion.total_cost", cost["total_cost"])
                    # OpenInference семантические атрибуты для Phoenix
                    try:
                        from openinference.semconv.trace import SpanAttributes
                        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, _input_tokens)
                        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, _output_tokens)
                        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, _input_tokens + _output_tokens)
                        span.set_attribute(SpanAttributes.LLM_COST_TOTAL, cost["total_cost"])
                        span.set_attribute(SpanAttributes.LLM_COST_PROMPT, cost["input_cost"])
                        span.set_attribute(SpanAttributes.LLM_COST_COMPLETION, cost["output_cost"])
                        span.set_attribute(SpanAttributes.LLM_PROVIDER, "deepseek")
                        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, "deepseek-chat")
                    except ImportError:
                        pass

                if result.get("error"):
                    try:
                        from opentelemetry.sdk.trace import Status, StatusCode
                        span.set_status(Status(StatusCode.ERROR, result["error"]))
                    except ImportError:
                        pass
                    span.set_attribute("learning_companion.error", result["error"])

            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Node 1: Planner
# ---------------------------------------------------------------------------


def planner_node(state: LearningState) -> LearningState:
    """Определяет, какой контент нужно получить и откуда."""
    url = state["input_url"]
    text = state["input_text"]

    if url:
        if "youtube.com" in url or "youtu.be" in url:
            plan = f"Получить транскрипт YouTube видео. URL: {url}"
            source = "youtube"
        elif "arxiv.org" in url:
            plan = f"Скачать и прочитать PDF с arXiv. URL: {url}"
            source = "pdf"
        else:
            plan = f"Загрузить содержимое веб-страницы. URL: {url}"
            source = "web"
    elif text:
        plan = "Использовать переданный текст напрямую."
        source = "direct"
    else:
        return {**state, "error": "Нет ни URL, ни текста для анализа."}

    return {
        **state,
        "plan": plan,
        "content_source": source,
    }


# ---------------------------------------------------------------------------
# Node 2: Fetcher
# ---------------------------------------------------------------------------

YOUTUBE_SCRIPT = os.path.expanduser(
    "~/.hermes/skills/media/youtube-content/scripts/fetch_transcript.py"
)


def fetcher_node(state: LearningState) -> LearningState:
    """Загружает контент в зависимости от источника."""
    source = state["content_source"]
    url = state["input_url"]
    text = state["input_text"]

    if source == "youtube":
        return _fetch_youtube(state, url)
    elif source == "web":
        return _fetch_web(state, url)
    elif source == "pdf":
        return _fetch_pdf(state, url)
    elif source == "direct":
        return {**state, "raw_content": text}
    else:
        return {**state, "error": f"Неизвестный источник: {source}"}


def _fetch_youtube(state: LearningState, url: str) -> LearningState:
    import subprocess

    try:
        r = subprocess.run(
            ["curl", "-s", "--connect-timeout", "5", "-o", "/dev/null",
             "-w", "%{http_code}", "https://www.youtube.com"],
            capture_output=True, text=True, timeout=10,
        )
        if r.stdout.strip() != "200":
            return {**state, "error": "YouTube недоступен с этого сервера. Нужен VPN."}
    except Exception as e:
        return {**state, "error": f"Не удалось проверить доступность YouTube: {e}"}

    for lang in [["ru", "en"], ["en"], None]:
        try:
            cmd = ["python3", YOUTUBE_SCRIPT, url, "--text-only", "--timestamps"]
            if lang:
                cmd += ["--language", ",".join(lang)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout.strip():
                content = r.stdout.strip()
                if content.startswith("{"):
                    err_data = json.loads(content)
                    if "error" in err_data:
                        continue
                return {**state, "raw_content": content}
        except Exception:
            continue

    return {**state, "error": "Не удалось получить транскрипт YouTube видео."}


def _fetch_web(state: LearningState, url: str) -> LearningState:
    import subprocess, re

    try:
        r = subprocess.run(
            ["curl", "-sL", "--max-time", "15", url],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0 and r.stdout.strip():
            text = r.stdout.strip()
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return {**state, "raw_content": text[:100_000]}
        return {**state, "error": f"Веб-страница не загрузилась (код: {r.returncode})"}
    except Exception as e:
        return {**state, "error": f"Ошибка загрузки веб-страницы: {e}"}


def _fetch_pdf(state: LearningState, url: str) -> LearningState:
    import subprocess

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = tmp.name
    try:
        r = subprocess.run(
            ["curl", "-sL", "--max-time", "30", "-o", pdf_path, url],
            capture_output=True, text=True, timeout=35,
        )
        if r.returncode != 0:
            return {**state, "error": "PDF не загрузился"}

        try:
            import fitz
            doc = fitz.open(pdf_path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
        except ImportError:
            r2 = subprocess.run(
                ["pdftotext", pdf_path, "-"],
                capture_output=True, text=True, timeout=15,
            )
            text = r2.stdout or ""

        return {**state, "raw_content": text[:100_000]}
    except Exception as e:
        return {**state, "error": f"Ошибка загрузки PDF: {e}"}
    finally:
        try:
            os.unlink(pdf_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Условия (edges) графа
# ---------------------------------------------------------------------------


def has_content(state: LearningState) -> Literal["analyze", "error"]:
    if state.get("raw_content") and not state.get("error"):
        return "analyze"
    return "error"


def has_analysis(state: LearningState) -> Literal["check_analysis", "error"]:
    if state.get("analysis") and not state.get("error"):
        return "check_analysis"
    return "error"


# ---------------------------------------------------------------------------
# Node 3: Analyst
# ---------------------------------------------------------------------------

ANALYST_SYSTEM = """Ты — эксперт по анализу образовательного контента.
Твоя задача: из транскрипта видео или текста статьи создать структурированную учебную заметку.

Формат заметки:
# [Название]

▶️ **Видео: [Название]** (если видео)
📺 Ссылка: [YouTube](url)

## 📌 Концепты
- **Концепт** — определение и описание, связи с другими концептами

## 🔗 Связи
- [[Концепт A]] ↔ [[Концепт B]] — описание связи

## ❓ Вопросы для самопроверки
1. **Вопрос?**
   → **Ответ:** ...

## 📖 Глоссарий
- **Термин** — краткое определение

## 📎 Источник
- Ссылка: [оригинал](url)
- Дата обработки: {date}

Правила:
- Все концепты, вопросы, глоссарий — на русском языке
- Используй только информацию из предоставленного текста
- Не выдумывай факты
- Будь максимально содержательным и структурированным
"""


def analyst_node(state: LearningState) -> LearningState:
    raw = state["raw_content"]
    url = state["input_url"]
    today = date.today().isoformat()

    if not raw or len(raw) < 50:
        return {**state, "error": "Слишком мало контента для анализа."}

    MAX_CHARS = 40_000
    if len(raw) > MAX_CHARS:
        raw = raw[:MAX_CHARS] + "\n\n[...пропущен средний фрагмент...]\n\n" + raw[-10_000:]

    system = ANALYST_SYSTEM.format(date=today)
    user = f"Вот контент для анализа:\n\n{raw}\n\nИсточник: {url if url else 'прямой текст'}"

    try:
        analysis = llm_call(system, user, max_tokens=8192)
    except Exception as e:
        return {**state, "error": f"Ошибка при анализе: {e}"}

    return {**state, "analysis": analysis}


def analyst_with_hitl(state: LearningState) -> LearningState:
    """Анализ контента (LLM вызов). Без interrupt."""
    state = analyst_node(state)

    if state.get("error"):
        return state

    cost = get_cost()
    print(f"\n{'='*60}")
    print(f"📝 Анализ готов! (run_id: {state['run_id']})")
    print(f"{'-'*60}")
    if not state.get("human_approved_analysis"):
        print(state["analysis"][:600])
        if len(state["analysis"]) > 600:
            print(f"... (всего {len(state['analysis'])} символов)")
        print(f"{'-'*60}")
        print(f"💰 Стоимость: ${cost['total_cost']:.6f} ({cost['total_tokens']:,} токенов)")
    print(f"{'='*60}\n")

    return state


def approve_analysis(state: LearningState) -> LearningState:
    """Human-in-the-loop: подтверждение анализа (без LLM вызова)."""
    from langgraph.types import interrupt

    cost = get_cost()

    print(f"\n{'='*60}")
    print(f"⚡ HUMAN-IN-THE-LOOP: Подтвердите анализ? (run_id: {state['run_id']})")
    print(f"   Стоимость: ${cost['total_cost']:.6f} ({cost['total_tokens']:,} токенов)")
    print(f"{'='*60}")
    print(f"   ➡️  resume-analyst {state['run_id']}  — принять и сохранить")
    print(f"   ➡️  reject-analyst {state['run_id']}  — отклонить")
    print(f"{'='*60}\n")

    interrupt(value={"node": "approve_analysis", "msg": "Подтвердите анализ"})

    # После resume помечаем, что анализ одобрен
    state = {**state, "human_approved_analysis": True}
    print(f"📨 Анализ подтверждён пользователем. Переход к сохранению.")
    return state


# ---------------------------------------------------------------------------
# Node 4: Writer
# ---------------------------------------------------------------------------

OBSIDIAN_VAULT = os.path.expanduser("~/Obsidian/Learning")


def writer_node(state: LearningState) -> LearningState:
    analysis = state["analysis"]

    if not analysis:
        return {**state, "error": "Нет анализа для сохранения."}

    title = "Untitled"
    for line in analysis.split("\n"):
        if line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()
            break

    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    safe_title = safe_title.strip().replace(" ", " ")[:100]
    filename = f"{safe_title}.md"

    articles_dir = os.path.join(OBSIDIAN_VAULT, "Articles")
    os.makedirs(articles_dir, exist_ok=True)

    filepath = os.path.join(articles_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(analysis)
        return {**state, "obsidian_path": filepath, "saved": True}
    except Exception as e:
        return {**state, "error": f"Ошибка сохранения: {e}"}


def writer_with_hitl(state: LearningState) -> LearningState:
    """Решает, нужно ли сохранять (без interrupt)."""
    print(f"\n{'='*60}")
    print(f"💾 Сохранить заметку в Obsidian?")
    print(f"Путь: ~/Obsidian/Learning/Articles/")
    print(f"Размер: {len(state.get('analysis', ''))} символов")
    cost = get_cost()
    print(f"💰 Стоимость: ${cost['total_cost']:.6f}")
    print(f"{'='*60}")
    print(f"   ➡️  save-writer — сохранить")
    print(f"   ➡️  skip-save  — пропустить")
    print(f"{'='*60}\n")
    return state


def approve_save(state: LearningState) -> LearningState:
    """Human-in-the-loop: подтверждение сохранения (без LLM вызова)."""
    from langgraph.types import interrupt

    interrupt(value={"node": "approve_save", "msg": "Сохранить заметку?"})

    # После resume решаем: save или skip
    state = {**state, "human_approved_save": True}

    if not state.get("error"):
        state = writer_node(state)

    if state.get("saved"):
        print(f"✅ Сохранено: {state['obsidian_path']}")
        # Сохраняем как последнюю заметку
        title = ""
        for line in (state.get("analysis", "") or "").split("\n"):
            if line.startswith("# ") and not line.startswith("## "):
                title = line[2:].strip()
                break
        save_last_note(
            run_id=state.get("run_id", ""),
            title=title,
            path=state["obsidian_path"],
            url=state.get("input_url", ""),
            analysis_preview=state.get("analysis", ""),
        )
        # Добавляем в Long-Term Memory
        ltm = get_ltm()
        ltm_note_id = state.get("run_id", "") or str(uuid4())
        ltm.add_note(
            note_id=ltm_note_id,
            title=title,
            source=state.get("content_source", ""),
            url=state.get("input_url", ""),
            key_concepts=state.get("extracted_concepts", []),
            summary=state.get("analysis", "")[:300],
        )
        # Отправляем результат в Telegram
        state_for_tg = {**state}
        notify_telegram(state_for_tg)
    else:
        print(f"⏭️  Сохранение пропущено.")

    return state


# ---------------------------------------------------------------------------
# Сборка графа
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Строит LangGraph граф Learning Companion с OTel span'ами для каждого узла."""
    from langgraph.graph import StateGraph

    graph = StateGraph(LearningState)

    # Узлы с декоратором span_wrapper для семантических span'ов
    graph.add_node("planner", span_wrapper("planner")(planner_node))
    graph.add_node("fetcher", span_wrapper("fetcher")(fetcher_node))
    graph.add_node("analyst", span_wrapper("analyst")(analyst_with_hitl))
    graph.add_node("approve", span_wrapper("approve")(approve_analysis))
    graph.add_node("writer", span_wrapper("writer")(writer_with_hitl))
    graph.add_node("approve_save", span_wrapper("approve_save")(approve_save))
    graph.add_node("error_handler", lambda s: s)  # терминальный узел ошибки

    # Старт
    graph.set_entry_point("planner")

    # Рёбра
    graph.add_edge("planner", "fetcher")
    graph.add_conditional_edges(
        "fetcher",
        has_content,
        {"analyze": "analyst", "error": "error_handler"},
    )
    graph.add_edge("analyst", "approve")
    graph.add_conditional_edges(
        "approve",
        has_analysis,
        {"check_analysis": "writer", "error": "error_handler"},
    )
    graph.add_edge("writer", "approve_save")
    graph.add_edge("approve_save", END)
    graph.add_edge("error_handler", END)

    return graph


# ---------------------------------------------------------------------------
# Compile с PostgresSaver
# ---------------------------------------------------------------------------


def compile_agent():
    """Компилирует граф с PostgresSaver (или MemorySaver как fallback)."""
    graph = build_graph()
    checkpointer = get_checkpointer()
    if hasattr(checkpointer, 'setup'):
        try:
            checkpointer.setup()
        except Exception:
            pass
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Phoenix Tracing
# ---------------------------------------------------------------------------


def setup_tracing():
    """Настраивает Phoenix/OpenTelemetry трейсинг."""
    try:
        os.environ["PHOENIX_PROJECT_NAME"] = "Learning Companion"

        from openinference.instrumentation.openai import OpenAIInstrumentor
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({
            "service.name": "Learning Companion",
            "openinference.project.name": "Learning Companion",
            "project.name": "Learning Companion",
        })

        span_exporter = OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)

        OpenAIInstrumentor().instrument()

        print(f"🕵️ Phoenix tracing включён. Проект: 'Learning Companion'")
        return True

    except ImportError as e:
        print(f"⚠️  Phoenix/OpenTelemetry не установлены. Трейсинг отключён: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Не удалось настроить трейсинг: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_agent(url: str = "", text: str = "", enable_tracing: bool = False):
    """Запускает агента."""
    reset_counters()

    if enable_tracing:
        setup_tracing()

    initial = make_initial_state(url=url, text=text)

    if initial.get("error"):
        print(f"❌ {initial['error']}")
        return None

    print(f"\n{'='*60}")
    print(f"🧠 Learning Companion v2 — Phase 2 (PostgresSaver)")
    print(f"Run ID: {initial['run_id']}")
    print(f"URL: {url or 'прямой текст'}")
    print(f"{'='*60}\n")

    agent = compile_agent()
    thread_config = {"configurable": {"thread_id": initial["run_id"]}}

    try:
        result = agent.invoke(initial, config=thread_config)

        # Проверяем, не остановился ли граф на interrupt
        if isinstance(result, dict) and result.get("__interrupt__"):
            print(f"\n{'='*60}")
            print(f"⏸️  Граф на паузе (interrupt). Resume через:")
            print(f"   python3 learning_companion_v2.py resume {initial['run_id']} <command>")
            print(f"{'='*60}\n")
            return {"run_id": initial["run_id"], "status": "interrupted"}

        if result.get("error"):
            print(f"❌ Ошибка: {result['error']}")
            return None

    except GraphInterrupt:  # fallback для старых версий
        # Old-style interrupt (fallback)
        print(f"\n{'='*60}")
        print(f"⏸️  Граф на паузе (interrupt). Resume через:")
        print(f"   python3 learning_companion_v2.py resume {initial['run_id']} <command>")
        print(f"{'='*60}\n")
        return {"run_id": initial["run_id"], "status": "interrupted"}

    print(f"\n{'='*60}")
    print(f"✅ Готово!")
    print(f"Run ID: {result.get('run_id', 'N/A')}")

    if result.get("saved"):
        print(f"📁 Сохранено: {result.get('obsidian_path', 'N/A')}")

    cost = get_cost()
    print(f"\n💰 Статистика:")
    print(f"   Входные токены: {cost['input_tokens']:,}")
    print(f"   Выходные токены: {cost['output_tokens']:,}")
    print(f"   Всего токенов: {cost['total_tokens']:,}")
    print(f"   Стоимость: ${cost['total_cost']:.6f}")

    print(f"\n📋 Заметка ({len(result.get('analysis', ''))} символов):")
    print("-" * 40)
    print(result.get("analysis", ""))

    return result


def resume_agent(run_id: str, command: str) -> bool:
    """Продолжает выполнение графа после interrupt."""
    agent = compile_agent()
    thread_config = {"configurable": {"thread_id": run_id}}

    state = agent.get_state(thread_config)
    if not state:
        print(f"❌ Состояние для run_id '{run_id}' не найдено.")
        return False

    next_node = state.next
    if not next_node:
        print(f"❌ Граф уже завершён для run_id '{run_id}'.")
        return False

    print(f"📌 Текущий узел: {next_node}")

    if command == "resume-analyst":
        # Подтверждаем анализ через Command(resume=...)
        result = agent.invoke(Command(resume="approved"), config=thread_config)
        if isinstance(result, dict) and result.get("__interrupt__"):
            print(f"✅ Анализ подтверждён. Граф на паузе в writer.")
            return True
        if result.get("error"):
            print(f"❌ Ошибка: {result['error']}")
            return False
        print(f"✅ Граф завершён.")
        return True

    elif command == "reject-analyst":
        print(f"❌ Анализ отклонён. Граф завершён.")
        return True
    elif command == "save-writer":
        # Сохраняем через Command(resume=...)
        result = agent.invoke(Command(resume="save"), config=thread_config)
        if result.get("error"):
            print(f"❌ Ошибка: {result['error']}")
            return False
        if result.get("saved"):
            print(f"✅ Сохранено: {result.get('obsidian_path', 'N/A')}")
        print(f"\n📋 Заметка:")
        print("-" * 40)
        print(result.get("analysis", ""))
        return True

    elif command == "skip-save":
        # Пропускаем через Command(resume=...)
        result = agent.invoke(Command(resume="skip"), config=thread_config)
        if result.get("error"):
            print(f"❌ Ошибка: {result['error']}")
            return False
        print(f"⏭️  Сохранение пропущено. Граф завершён.")
        return True

    else:
        print(f"❌ Неизвестная команда: {command}")
        return False


def run_check(note_path: str = "") -> bool:
    """Запускает генерацию вопросов по заметке."""
    reset_counters()

    if not note_path:
        last = get_last_note()
        note_path = last.get("path", "")
        if not note_path or not os.path.exists(note_path):
            print("❌ Нет последней заметки. Укажите путь к файлу вручную.")
            return False
        print(f"📄 Использую последнюю заметку: {last.get('title', 'Неизвестно')}")

    print(f"\n{'='*60}")
    print(f"🧠 Генерация вопросов для проверки знаний")
    print(f"{'='*60}")
    print(f"📄 {note_path}")
    print(f"{'-'*60}\n")

    # Загружаем LTM контекст для связывания с предыдущими заметками
    ltm = get_ltm()
    ltm_context = ltm.format_context(limit=3) or ""
    if ltm_context:
        print(f"📚 LTM: {ltm.load_cache().get('notes_count', 0)} заметок, "
              f"{ltm.load_cache().get('total_concepts', 0)} концептов")

    questions = generate_questions(note_path, ltm_context=ltm_context)
    if not questions:
        print("❌ Не удалось сгенерировать вопросы.")
        return False

    cost = get_cost()
    print(f"\n{'='*60}")
    print(f"❓ Вопросы готовы! ({cost['total_tokens']:,} токенов, ${cost['total_cost']:.6f})")
    print(f"{'='*60}\n")
    print(questions)

    # Отправляем вопросы в Telegram
    header = "📝 **Проверка знаний**\n━━━━━━━━━━━━━━━━━━━━\nОтветь на вопросы, и я проверю твои ответы! ✍️"
    footer = f"\n\n━━━━━━━━━━━━━━━━━━━━\n💬 Ответь мне прямо сюда — я проверю!\n💰 {cost['total_tokens']:,} токенов | ${cost['total_cost']:.6f}"

    # Сохраняем вопросы в last_note для проверки ответов
    last = get_last_note()
    last["questions"] = questions
    import json
    os.makedirs(os.path.dirname(LAST_NOTE_FILE), exist_ok=True)
    with open(LAST_NOTE_FILE, "w", encoding="utf-8") as f:
        json.dump(last, f, ensure_ascii=False, indent=2)

    # Отмечаем в LTM, что вопросы заданы + повышаем мастерство
    ltm_note_id = last.get("run_id", "")
    if ltm_note_id:
        ltm.mark_questions_asked(ltm_note_id)
        ltm.update_mastery(ltm_note_id, delta=10)

    send_telegram_long(questions + footer, header=header)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Learning Companion v2 — Phase 2 Multi-Agent"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Режим работы")

    run_parser = subparsers.add_parser("run", help="Запустить анализ")
    run_parser.add_argument("--url", "-u", default="", help="URL для анализа")
    run_parser.add_argument("--text", "-t", default="", help="Текст для анализа")
    run_parser.add_argument("--trace", action="store_true", help="Включить Phoenix tracing")

    resume_parser = subparsers.add_parser("resume", help="Продолжить после HITL")
    resume_parser.add_argument("run_id", help="ID прогона")
    resume_parser.add_argument(
        "command",
        choices=["resume-analyst", "reject-analyst", "save-writer", "skip-save"],
        help="Команда для interrupt",
    )

    check_parser = subparsers.add_parser("check", help="Сгенерировать вопросы для проверки знаний")
    check_parser.add_argument("note", nargs="?", default="",
                              help="Путь к заметке в Obsidian (опционально, по умолчанию последняя)")

    args = parser.parse_args()

    if args.mode == "run":
        if not args.url and not args.text:
            print("⚠️  Укажите --url или --text")
            sys.exit(1)
        run_agent(url=args.url, text=args.text, enable_tracing=args.trace)

    elif args.mode == "resume":
        resume_agent(args.run_id, args.command)

    elif args.mode == "check":
        run_check(note_path=args.note)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
