"""
Telegram sender — отправка сообщений в Telegram через Bot API.

Используется learning_companion_v2.py и другими скриптами.
"""

import subprocess
import json


def send_telegram(text: str, bot_token: str, chat_id: str) -> bool:
    """Отправляет одно сообщение в Telegram через Bot API.

    Args:
        text: Текст сообщения (до 4096 символов)
        bot_token: Токен Telegram бота
        chat_id: ID чата для отправки

    Returns:
        True если успешно, False если ошибка
    """
    if not bot_token or not chat_id:
        print("⚠️  TELEGRAM_BOT_TOKEN или CHAT_ID не заданы.")
        return False

    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    })

    try:
        r = subprocess.run(
            ["curl", "-s", "-X", "POST",
             f"https://api.telegram.org/bot{bot_token}/sendMessage",
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True, text=True, timeout=15,
        )
        result = json.loads(r.stdout)
        if result.get("ok"):
            return True
        else:
            print(f"⚠️  Telegram error: {result.get('description', 'unknown')}")
            return False
    except Exception as e:
        print(f"⚠️  Не удалось отправить в Telegram: {e}")
        return False
