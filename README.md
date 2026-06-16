# AI Engineer Agent Roadmap 2026 — Learning Companion

> Phase 2 проект: LangGraph-агент для анализа контента (YouTube, статьи, PDF), создания учебных заметок в Obsidian,
> проверки знаний через вопросы, с Long-Term Memory на PostgreSQL и Telegram-интеграцией.

## Состав проекта

```
📁 Learning Companion/
├── skill.json               — конфиг навыка Hermes для запуска агента
📄 learning_companion_v2.py  — главный агент (LangGraph граф, LTM, Telegram, HITL) [1552 строки, 61 КБ]
📄 telegram_sender.py        — утилита отправки сообщений в Telegram
📄 agent-raw.py              — Phase 1: сырой agent loop на DeepSeek API
📄 agent-litellm.py          — Phase 1: agent loop через LiteLLM
📄 agent-sdk.py              — Phase 1: agent loop через OpenAI SDK
📄 gen_phase0_docx.py        — генератор DOCX для Phase 0
📄 make_docx.py              — вспомогательный генератор DOCX
📄 MY_ROADMAP.md             — персонализированный учебный план
📄 AGENTS.md                 — контекст для Hermes
📄 README.md                 — этот файл
```

## Архитектура агента

**Граф (5 узлов, LangGraph + PostgresSaver):**

```
planner → fetcher → analyst → [approve ⏸️ HITL] → writer → [approve_save ⏸️ HITL] → END
```

**Ключевые компоненты:**
- **Long-Term Memory** — PostgreSQL таблица `notes` + JSON кэш на файле
- **Human-in-the-Loop** — две точки паузы: подтверждение анализа и подтверждение сохранения
- **Telegram** — отправка обзора (без вопросов) и вопросов для проверки знаний
- **Obsidian** — сохранение заметок в `/home/sen/Obsidian/Learning/Articles/`

## Использование

```bash
# Запустить анализ
python3 learning_companion_v2.py run --url "https://youtu.be/..."

# Продолжить после HITL
python3 learning_companion_v2.py resume <run_id> resume-analyst
python3 learning_companion_v2.py resume <run_id> save-writer

# Проверка знаний
python3 learning_companion_v2.py check
```

## Зависимости

- Python 3.12+
- `langgraph`, `langchain`, `httpx`, `openai`, `psycopg2-binary`, `beautifulsoup4`, `yt-dlp`
- PostgreSQL (БД `learning_companion`)
- DeepSeek API (ключ в `~/.hermes/.env`)

## Хронология

- **v1** — сырой agent-loop с DeepSeek, фаза 1
- **v2** — LangGraph граф + PostgresSaver + HITL + LTM + Telegram + Obsidian

## Бекапы

Последний бекап: `learning_companion_v2.backup.py` (1552 строки, 61 КБ).
