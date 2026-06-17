# Learning Companion

[![CI](https://github.com/Sen9405/learning-companion/actions/workflows/ci.yml/badge.svg)](https://github.com/Sen9405/learning-companion/actions/workflows/ci.yml)

A LangGraph-powered agent that analyzes content from YouTube, articles, and PDFs — generates structured learning notes in Obsidian, tests your knowledge with questions, and keeps long-term memory of what you've learned.

**Stack:** Python 3.12+, LangGraph, PostgreSQL, DeepSeek API, Telegram API, Obsidian

---

## Architecture

The agent runs as a **5-node LangGraph** with Human-in-the-Loop checkpoints:

```
planner → fetcher → analyst → [approve ⏸️] → writer → [approve_save ⏸️] → END
```

**Core components:**
- **Long-Term Memory** — PostgreSQL `notes` table + JSON cache
- **Human-in-the-Loop** — two pause points: review analysis before saving, confirm before writing
- **Telegram** — sends summaries (without questions) and knowledge-check questions
- **Obsidian** — saves notes to `~/Obsidian/Learning/Articles/`

---

## Usage

```bash
# Analyze a URL
learning-companion run --url "https://youtu.be/..."

# Continue after HITL pause
learning-companion resume <run_id> resume-analyst
learning-companion resume <run_id> save-writer

# Knowledge check
learning-companion check
```

---

## CLI Commands

| Command | Description |
|---|---|
| `run` | Start a new analysis (supports `--url`, `--text`, `--title`, `--language`) |
| `resume` | Continue an interrupted run (`resume-analyst` or `save-writer`) |
| `check` | Review recent learning notes |

---

## Dependencies

**Base:**
- `langgraph`, `openai`, `httpx`, `beautifulsoup4`

**Optional extras:**
- `youtube` — YouTube transcript support (`yt-dlp`)
- `pdf` — PDF extraction (`pymupdf`, `marker-pdf`)
- `postgres` — PostgreSQL persistence (`psycopg2-binary`, `langgraph-checkpoint-postgres`)
- `tracing` — OpenTelemetry tracing via Arize Phoenix
- `all` — everything above

Install with extras:
```bash
pip install -e ".[all]"
```

---

## Configuration

Create `~/.hermes/.env` or `~/.env`:

```env
DEEPSEEK_API_KEY=your_deepseek_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## Testing

```bash
# All tests
python -m pytest tests/ -v

# Lint
ruff check src/ tests/
```

The CI pipeline runs ruff linting and pytest across two matrix configurations: all extras and base only.

---

## Project Structure

```
src/learning_companion/
├── __init__.py         — package metadata
├── cli.py              — CLI entry point (argparse)
├── llm.py              — DeepSeek client with cost tracking
├── telegram.py         — Telegram send helpers
├── memory.py           — Long-Term Memory (Postgres/SQLite)
├── utils.py            — utilities (save/load state)
└── graph/
    ├── __init__.py     — LearningState TypedDict
    ├── nodes.py        — planner/fetcher/analyst/writer nodes
    ├── builder.py      — build_graph() + compile_agent()
    └── tracing.py      — OpenTelemetry tracing setup
tests/                  — 66 tests across all modules
legacy/                 — original monolith scripts
```
