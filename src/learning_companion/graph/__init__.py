"""Graph state definitions for Learning Companion."""

from __future__ import annotations

from typing import TypedDict


class LearningState(TypedDict):
    """Состояние графа Learning Companion."""
    # Входные данные
    url: str
    text: str
    title: str

    # Результаты этапов
    content: str  # сырой контент после fetcher
    analysis: str  # анализ после analyst
    note: str  # заметка после writer
    questions_list: str  # вопросы для проверки
    telegram_note: str  # заметка без вопросов (для отправки)

    # Мета-информация
    run_id: str
    source_type: str  # youtube, web, pdf, text
    stage: str  # текущий этап для resume
    approved: bool  # HITL approval
    error: str
    language: str  # ru / en
    trace_id: str  # Phoenix trace ID

    # Cost tracking
    prompt_tokens: int
    completion_tokens: int
    cost: float

    # Счётчики
    notes_checked: int  # сколько существующих заметок проверено


def make_initial_state(
    url: str = "",
    text: str = "",
    title: str = "",
    language: str = "ru",
    run_id: str = "",
) -> LearningState:
    return {
        "url": url,
        "text": text,
        "title": title,
        "content": "",
        "analysis": "",
        "note": "",
        "questions_list": "",
        "telegram_note": "",
        "run_id": run_id,
        "source_type": "text",
        "stage": "planner",
        "approved": False,
        "error": "",
        "language": language,
        "trace_id": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
        "notes_checked": 0,
    }
