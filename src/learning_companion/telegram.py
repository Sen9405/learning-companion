"""Telegram messaging — send notes, questions, and long messages."""

from __future__ import annotations

import os
import re
from typing import Any


def send_telegram(
    message: str,
    token: str | None = None,
    chat_id: str | None = None,
    parse_mode: str = "MarkdownV2",
) -> dict[str, Any] | None:
    """Send a message to Telegram.

    Falls back to reading TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env
    or from ~/.hermes/.env.
    """
    import httpx

    if not token:
        token = os.environ.get("TELEGRAM_BOT_TOKEN") or _read_dotenv_key("TELEGRAM_BOT_TOKEN")
    if not chat_id:
        chat_id = os.environ.get("TELEGRAM_CHAT_ID") or _read_dotenv_key("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[send_telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set, skipping")
        return None

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": parse_mode}

    try:
        resp = httpx.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[send_telegram] Error: {e}")
        # fallback: отправить без разметки
        if parse_mode:
            payload.pop("parse_mode", None)
            try:
                resp = httpx.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as e2:
                print(f"[send_telegram] Fallback also failed: {e2}")
        return None


def send_telegram_pdf(
    pdf_path: str,
    token: str | None = None,
    chat_id: str | None = None,
) -> dict[str, Any] | None:
    """Send a PDF file to Telegram as a document."""
    import httpx

    if not token:
        token = os.environ.get("TELEGRAM_BOT_TOKEN") or _read_dotenv_key("TELEGRAM_BOT_TOKEN")
    if not chat_id:
        chat_id = os.environ.get("TELEGRAM_CHAT_ID") or _read_dotenv_key("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[send_telegram_pdf] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set, skipping")
        return None

    if not os.path.exists(pdf_path):
        print(f"[send_telegram_pdf] File not found: {pdf_path}")
        return None

    url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        with open(pdf_path, "rb") as f:
            resp = httpx.post(url, data={"chat_id": chat_id}, files={"document": f}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[send_telegram_pdf] Error: {e}")
        return None


def send_telegram_long(
    message: str,
    token: str | None = None,
    chat_id: str | None = None,
    max_len: int = 4096,
) -> None:
    """Send a long message, splitting into chunks if needed."""
    if not message:
        return
    if len(message) <= max_len:
        send_telegram(message, token, chat_id)
        return

    # Split by paragraphs
    parts = []
    for para in message.split("\n\n"):
        if parts and len(parts[-1]) + len(para) + 2 < max_len:
            parts[-1] += "\n\n" + para
        else:
            parts.append(para)

    for part in parts:
        send_telegram(part, token, chat_id)


def notify_telegram(
    message: str,
    run_id: str = "",
    stage: str = "",
    token: str | None = None,
    chat_id: str | None = None,
) -> dict[str, Any] | None:
    """Send a short status notification about agent progress."""
    prefix = "🤖 Learning Companion"
    if run_id:
        prefix += f" [{run_id[:8]}]"
    if stage:
        prefix += f" — {stage}"
    return send_telegram(f"{prefix}\n{message}", token, chat_id)


def _escape_telegram_md(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special = r"_*[]()~`>#+-=|{}.!"
    result = []
    for ch in text:
        if ch in special:
            result.append("\\" + ch)
        else:
            result.append(ch)
    return "".join(result)


def _strip_questions_section(text: str) -> str:
    """Удаляет секцию 'Вопросы для проверки знаний' из текста."""
    # Ищем заголовок "Вопросы для проверки знаний" или аналогичные
    patterns = [
        r"(?s)\n#{1,3}\s*Вопросы?\s*(для\s*проверки\s*знаний|для\s*закрепления|по\s*теме).*",
        r"(?s)\n#{1,3}\s*Questions?\s*(for\s*review|for\s*check|to\s*check).*",
        r"(?s)\n-{2,}\s*(?:Вопросы?|Questions?).*",
        r"(?s)\n\*\*Вопросы?\s*(?:для\s*проверки|для\s*закрепления|по\s*теме).*",
    ]
    result = text
    for pat in patterns:
        result = re.sub(pat, "", result, count=1, flags=re.DOTALL)
    # Также удаляем последние строки, если они пустые после удаления
    result = result.rstrip()
    return result


def _read_dotenv_key(key: str) -> str | None:
    """Read a single key from ~/.hermes/.env or ~/.env."""
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
                        if k.strip().strip('"').strip("'") == key:
                            val = v.strip().strip('"').strip("'")
                            if val:
                                return val
    return None
