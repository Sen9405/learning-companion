"""Telegram bot for Learning Companion.

Two modes:
  1. Send a URL/text -> get note + questions (read-only, as before).
  2. Interactive quiz: answer questions one by one, get LLM evaluation + score.

Usage:
    learning-companion-bot           # run with COMPANION_BOT_TOKEN from env / .env

Environment:
    COMPANION_BOT_TOKEN  — Telegram bot token
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import telebot
    from telebot import types
except ImportError:
    print("pyTelegramBotAPI not installed. Run: pip install pyTelegramBotAPI")
    sys.exit(1)

# Reuse LLM client from the LC package for answer evaluation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    from learning_companion.llm import llm_call
except ImportError:
    llm_call = None  # type: ignore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
VENV_PYTHON = str(SCRIPT_DIR / ".venv" / "bin" / "python")
PROJECT_DIR = str(SCRIPT_DIR)

ALLOWED_IDS: list[int] = []
"""If non-empty, only these Telegram user IDs can use the bot."""

MAX_TEXT_LENGTH = 30000
TELEGRAM_MSG_LIMIT = 4096

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("learning-companion-bot")


# ---------------------------------------------------------------------------
# Quiz session storage (in-memory, not persisted across restarts)
# ---------------------------------------------------------------------------

_quiz_lock = threading.Lock()
_quiz_sessions: dict[int, dict[str, Any]] = {}
_saved_questions: dict[int, list[dict]] = {}
"""chat_id -> last quiz questions list (preserved after quiz ends for retry)."""


def _get_quiz(chat_id: int) -> dict[str, Any] | None:
    with _quiz_lock:
        return _quiz_sessions.get(chat_id)


def _set_quiz(chat_id: int, session: dict[str, Any]) -> None:
    session["_chat_id"] = chat_id
    with _quiz_lock:
        _quiz_sessions[chat_id] = session


def _del_quiz(chat_id: int) -> None:
    with _quiz_lock:
        _quiz_sessions.pop(chat_id, None)


def _save_questions(chat_id: int, questions: list[dict]) -> None:
    with _quiz_lock:
        _saved_questions[chat_id] = questions


def _get_saved_questions(chat_id: int) -> list[dict] | None:
    with _quiz_lock:
        return _saved_questions.get(chat_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_or_dotenv(key: str) -> str | None:
    val = os.environ.get(key)
    if val:
        return val
    for env_file in [
        Path.home() / ".hermes" / ".env",
        Path.home() / ".env",
    ]:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip().strip("\"'") == key:
                        v_clean = v.strip().strip("\"'")
                        if v_clean:
                            return v_clean
    return None


def _is_allowed(user_id: int) -> bool:
    if not ALLOWED_IDS:
        return True
    return user_id in ALLOWED_IDS


def _is_url(text: str) -> bool:
    return bool(re.match(r"https?://\S+", text.strip()))


def _split_long_message(text: str, max_len: int = TELEGRAM_MSG_LIMIT) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    for para in text.split("\n\n"):
        if parts and len(parts[-1]) + len(para) + 2 < max_len:
            parts[-1] += "\n\n" + para
        else:
            parts.append(para)
    return parts


def _run_lc_agent(url: str = "", text: str = "") -> dict[str, Any]:
    """Run learning-companion agent in bot mode and return structured result."""
    env = os.environ.copy()
    env["COMPANION_BOT_MODE"] = "1"

    cmd = [
        VENV_PYTHON,
        "-m",
        "learning_companion",
        "run",
        "--no-hitl",
    ]
    if url:
        cmd.extend(["--url", url])
    if text:
        cmd.extend(["--text", text])

    logger.info(f"Running: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=PROJECT_DIR,
        env=env,
    )
    elapsed = time.time() - start
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    logger.info(f"LC agent finished in {elapsed:.1f}s (exit={result.returncode})")

    # Try to parse JSON from the last line of stdout
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                data["elapsed"] = elapsed
                data["exit_code"] = result.returncode
                return data
            except json.JSONDecodeError:
                continue

    return {
        "title": "Error",
        "note": "",
        "questions": "",
        "analysis": "",
        "run_id": "",
        "cost": 0.0,
        "tokens_in": 0,
        "tokens_out": 0,
        "error": stderr[:1500] or stdout[:1500],
        "elapsed": elapsed,
        "exit_code": result.returncode,
    }


def _parse_questions_for_quiz(questions_text: str) -> list[dict]:
    """Parse formatted questions text into list of {'q': str, 'a': str} for quiz.

    Extracts question stems only; answers are stripped for the quiz UI
    but kept in the dict for reference.
    """
    items: list[dict] = []
    # Split by numbered items: 1.  2.  etc (with or without **)
    parts = re.split(r"\n(?=\d+\.\s)", questions_text)
    for part in parts:
        part = part.strip()
        if (
            not part
            or part.startswith("---")
            or part.startswith("###")
            or part.startswith("**")
        ):
            continue
        # Must start with a number
        if not re.match(r"\d+\.", part):
            continue

        lines = part.split("\n")
        q_lines: list[str] = []
        a_lines: list[str] = []
        in_answer = False
        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith(">")
                or stripped.startswith("||")
                or stripped.startswith("<")
            ):
                in_answer = True
                a_lines.append(stripped)
            elif in_answer and stripped:
                a_lines.append(stripped)
            else:
                q_lines.append(line)

        question = "\n".join(q_lines).strip()
        answer = "\n".join(a_lines).strip()
        if question:
            items.append({"q": question, "a": answer})
    return items


def _evaluate_answer(
    question: str, user_answer: str, context: str = ""
) -> dict[str, Any]:
    """Use LLM to evaluate the user's answer.

    Returns {"correct": bool, "feedback": str, "expected": str}
    """
    if llm_call is None:
        return {"correct": False, "feedback": "⚠️ LLM не доступен.", "expected": ""}

    system = (
        "Ты — ассистент-экзаменатор. Оцени ответ пользователя на вопрос.\n\n"
        '1. Если ответ верный и полный -> {"correct": true, "feedback": "..."}\n'
        '2. Если ответ частично верный -> {"correct": false, "feedback": "...верно, но..."}\n'
        '3. Если ответ неверный -> {"correct": false, "feedback": "...", "expected": "правильный ответ..."}\n\n'
        "Будь конкретным. Если ответ неполный — допиши недостающее. "
        "Если неверный — объясни почему и дай правильный ответ. "
        "Ответь строго в JSON."
    )

    prompt = f"Вопрос: {question}\n\nОтвет пользователя: {user_answer}"
    if context:
        prompt += f"\n\nКонтекст (из материала): {context[:2000]}"

    try:
        resp, _ = llm_call(
            system,
            [{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
            stage="bot.answer_eval",
        )

        try:
            data = json.loads(resp)
            return {
                "correct": data.get("correct", False),
                "feedback": data.get("feedback", ""),
                "expected": data.get("expected", ""),
            }
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", resp, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {
                    "correct": data.get("correct", False),
                    "feedback": data.get("feedback", ""),
                    "expected": data.get("expected", ""),
                }
    except Exception as e:
        logger.warning(f"LLM eval failed: {e}")

    return {"correct": False, "feedback": "⚠️ Не удалось оценить ответ.", "expected": ""}


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

START_QUIZ_CB = "start_quiz"
QUIZ_SKIP_CB = "quiz_skip"
QUIZ_END_CB = "quiz_end"


def run_bot(token: str) -> None:
    """Start the Learning Companion Telegram bot."""
    bot = telebot.TeleBot(token, threaded=False)

    # -------------------------------------------------------------------
    # /start /help
    # -------------------------------------------------------------------
    @bot.message_handler(commands=["start", "help"])
    def cmd_help(message: types.Message) -> None:
        if not _is_allowed(message.from_user.id):
            bot.reply_to(message, "❌ У вас нет доступа к этому боту.")
            return
        bot.reply_to(
            message,
            "🤖 **Learning Companion Bot**\n\n"
            "Пришли мне ссылку на YouTube, статью или текст — "
            "я проанализирую и создам учебную заметку.\n\n"
            "**Команды:**\n"
            "/start, /help — это сообщение\n"
            "/status — статус бота\n"
            "/cancel — отменить текущую викторину\n\n"
            "**Режимы:**\n"
            "• Просто пришли ссылку/текст — получу конспект + вопросы\n"
            "• После анализа нажми 🎯 **Проверить знания** — отвечай на вопросы по одному\n\n"
            "**Примеры:**\n"
            "• `https://youtu.be/...`\n"
            "• `https://habr.com/...`\n"
            "• Просто текст для анализа\n\n"
            "⚠️ Анализ может занять 1-2 минуты.",
            parse_mode="Markdown",
        )

    # -------------------------------------------------------------------
    # /status
    # -------------------------------------------------------------------
    @bot.message_handler(commands=["status"])
    def cmd_status(message: types.Message) -> None:
        if not _is_allowed(message.from_user.id):
            return
        bot.reply_to(
            message,
            "✅ **Learning Companion Bot** активен\n"
            f"Версия: 3.1.0\n"
            f"VENV: `{VENV_PYTHON}`\n"
            f"Режим: анализ + интерактивная проверка знаний",
            parse_mode="Markdown",
        )

    # -------------------------------------------------------------------
    # /cancel
    # -------------------------------------------------------------------
    @bot.message_handler(commands=["cancel"])
    def cmd_cancel(message: types.Message) -> None:
        if not _is_allowed(message.from_user.id):
            return
        chat_id = message.chat.id
        session = _get_quiz(chat_id)
        if session:
            stats = session.get("stats", {})
            correct = stats.get("correct", 0)
            wrong = stats.get("wrong", 0)
            total = correct + wrong
            _del_quiz(chat_id)
            if total > 0:
                pct = round(correct / total * 100) if total else 0
                bot.send_message(
                    chat_id,
                    f"❌ Викторина прервана.\n\n📊 Прогресс: {correct}/{total} ({pct}%)",
                )
            else:
                bot.send_message(chat_id, "❌ Викторина отменена.")
        else:
            bot.reply_to(message, "Нет активной викторины.")

    # -------------------------------------------------------------------
    # Main text handler -> analyze + show questions + quiz offer
    # -------------------------------------------------------------------
    @bot.message_handler(
        func=lambda m: _get_quiz(m.chat.id) is None, content_types=["text"]
    )
    def handle_analyze(message: types.Message) -> None:
        """Handle URL/text input when no quiz is active."""
        user_id = message.from_user.id
        chat_id = message.chat.id
        if not _is_allowed(user_id):
            return

        raw = message.text.strip()
        if not raw or raw.startswith("/"):
            return

        url = raw if _is_url(raw) else ""
        text = "" if url else raw[:MAX_TEXT_LENGTH]

        msg = bot.reply_to(message, "⏳ Анализирую... Это займёт 1-2 минуты.")

        try:
            result = _run_lc_agent(url=url, text=text)
            elapsed = result.get("elapsed", 0)
            run_id = result.get("run_id", "")
            cost = result.get("cost", 0)
            note = result.get("note", "")
            questions = result.get("questions", "")
            title = result.get("title", "Без названия")
            error = result.get("error", "")

            try:
                bot.delete_message(chat_id, msg.message_id)
            except Exception:
                pass

            if error:
                bot.send_message(
                    chat_id,
                    f"❌ **Ошибка при анализе**\n\n```\n{error[:1500]}\n```",
                    parse_mode="Markdown",
                )
                return

            # Header
            cost_str = f"${cost:.4f}" if cost else "N/A"
            bot.send_message(
                chat_id,
                f"📚 **{title}**\n⏱ {elapsed:.0f}с | 💰 {cost_str} | 🪪 `{run_id[:8]}`",
                parse_mode="Markdown",
            )

            # Note (strip questions section)
            if note:
                clean_note = note
                for pat in [r"(?s)\n#{1,3}\s*Вопрос.*", r"(?s)\n#{1,3}\s*Question.*"]:
                    clean_note = re.sub(pat, "", clean_note, count=1)
                clean_note = clean_note.strip()
                _send_long(bot, chat_id, clean_note, parse_mode="Markdown")

            # Questions
            if questions:
                q_list = _parse_questions_for_quiz(questions)
                if q_list:
                    quiz_sesh = {
                        "questions": q_list,
                        "current_idx": 0,
                        "stats": {"correct": 0, "wrong": 0, "details": []},
                        "note_context": (note or questions)[:3000],
                    }
                    _set_quiz(chat_id, quiz_sesh)

                _send_long(
                    bot,
                    chat_id,
                    f"📝 **Вопросы для проверки**\n\n{questions}",
                    parse_mode="Markdown",
                )

                if q_list:
                    markup = types.InlineKeyboardMarkup()
                    markup.add(
                        types.InlineKeyboardButton(
                            "🎯 Проверить знания", callback_data=START_QUIZ_CB
                        )
                    )
                    bot.send_message(
                        chat_id,
                        "Готов проверить себя? Отвечай на вопросы по одному — я оценю каждый ответ.",
                        reply_markup=markup,
                    )

        except subprocess.TimeoutExpired:
            try:
                bot.edit_message_text(
                    "⏰ **Таймаут** — анализ занял больше 10 минут.",
                    chat_id=chat_id,
                    message_id=msg.message_id,
                    parse_mode="Markdown",
                )
            except Exception:
                pass
        except Exception as e:
            logger.exception("Error processing message")
            try:
                bot.edit_message_text(
                    f"❌ **Ошибка:** `{e!s}`",
                    chat_id=chat_id,
                    message_id=msg.message_id,
                    parse_mode="Markdown",
                )
            except Exception:
                pass

    # -------------------------------------------------------------------
    # Quiz answer handler (text during active quiz)
    # -------------------------------------------------------------------
    @bot.message_handler(
        func=lambda m: _get_quiz(m.chat.id) is not None and not m.text.startswith("/"),
        content_types=["text"],
    )
    def handle_quiz_answer(message: types.Message) -> None:
        """Handle user's answer during an active quiz."""
        chat_id = message.chat.id
        session = _get_quiz(chat_id)
        if not session:
            return

        idx = session["current_idx"]
        questions = session["questions"]
        if idx < 0 or idx >= len(questions):
            _del_quiz(chat_id)
            bot.send_message(
                chat_id, "Викторина уже завершена. Отправь новый материал для анализа."
            )
            return

        q = questions[idx]
        user_answer = message.text.strip()
        if not user_answer:
            bot.reply_to(message, "Напиши ответ текстом или используй кнопки.")
            return

        thinking_msg = bot.reply_to(message, "🤔 Проверяю ответ...")

        try:
            result = _evaluate_answer(
                q["q"], user_answer, context=session.get("note_context", "")
            )
            correct = result.get("correct", False)
            feedback = result.get("feedback", "✅ Ответ принят.")
            expected = result.get("expected", "")

            try:
                bot.delete_message(chat_id, thinking_msg.message_id)
            except Exception:
                pass

            if correct:
                bot.send_message(
                    chat_id, f"✅ **Верно!**\n\n{feedback}", parse_mode="Markdown"
                )
            else:
                msg_text = f"❌ **Не совсем.**\n\n{feedback}"
                if expected:
                    msg_text += f"\n\n💡 **Правильный ответ:**\n{expected[:500]}"
                bot.send_message(chat_id, msg_text, parse_mode="Markdown")

            stats = session["stats"]
            if correct:
                stats["correct"] += 1
            else:
                stats["wrong"] += 1
            stats["details"].append({"correct": correct, "feedback": feedback})

            session["current_idx"] = idx + 1
            _set_quiz(chat_id, session)

            # Next question or results
            if session["current_idx"] >= len(questions):
                _show_quiz_results(bot, chat_id, session)
            else:
                _send_quiz_question(bot, chat_id, session)

        except Exception:
            logger.exception("Error evaluating answer")
            try:
                bot.delete_message(chat_id, thinking_msg.message_id)
            except Exception:
                pass
            bot.send_message(
                chat_id, "⚠️ Ошибка при проверке. Попробуй ещё раз или введи /cancel."
            )

    # -------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------
    @bot.callback_query_handler(func=lambda c: c.data == START_QUIZ_CB)
    def callback_start_quiz(call: types.CallbackQuery) -> None:
        chat_id = call.message.chat.id
        session = _get_quiz(chat_id)

        # If session was deleted (quiz completed), restore from saved questions
        if not session:
            saved = _get_saved_questions(chat_id)
            if saved:
                session = {
                    "questions": saved,
                    "current_idx": 0,
                    "stats": {"correct": 0, "wrong": 0, "details": []},
                    "note_context": "",
                }
                _set_quiz(chat_id, session)
            else:
                bot.answer_callback_query(
                    call.id, "❌ Нет сохранённых вопросов. Отправь новый материал."
                )
                return

        bot.answer_callback_query(call.id, "Поехали! 🚀")
        try:
            bot.delete_message(chat_id, call.message.message_id)
        except Exception:
            pass
        session["current_idx"] = 0
        session["stats"] = {"correct": 0, "wrong": 0, "details": []}
        _set_quiz(chat_id, session)
        _send_quiz_question(bot, chat_id, session)

    @bot.callback_query_handler(func=lambda c: c.data == QUIZ_SKIP_CB)
    def callback_skip(call: types.CallbackQuery) -> None:
        bot.answer_callback_query(call.id, "Пропускаем 👀")
        chat_id = call.message.chat.id
        session = _get_quiz(chat_id)
        if not session:
            return
        idx = session["current_idx"]
        questions = session["questions"]
        stats = session["stats"]
        stats["wrong"] += 1
        stats["details"].append({"correct": False, "feedback": "Пропущен"})
        session["current_idx"] = idx + 1
        _set_quiz(chat_id, session)
        try:
            bot.delete_message(chat_id, call.message.message_id)
        except Exception:
            pass
        if session["current_idx"] >= len(questions):
            _show_quiz_results(bot, chat_id, session)
        else:
            _send_quiz_question(bot, chat_id, session)

    @bot.callback_query_handler(func=lambda c: c.data == QUIZ_END_CB)
    def callback_end(call: types.CallbackQuery) -> None:
        bot.answer_callback_query(call.id, "Завершаем 📊")
        chat_id = call.message.chat.id
        session = _get_quiz(chat_id)
        if not session:
            return
        try:
            bot.delete_message(chat_id, call.message.message_id)
        except Exception:
            pass
        _show_quiz_results(bot, chat_id, session)

    # -------------------------------------------------------------------
    # Quiz helpers
    # -------------------------------------------------------------------
    def _send_quiz_question(
        bot_obj: telebot.TeleBot, chat_id: int, session: dict[str, Any]
    ) -> None:
        idx = session["current_idx"]
        questions = session["questions"]
        if idx < 0 or idx >= len(questions):
            _show_quiz_results(bot_obj, chat_id, session)
            return

        q = questions[idx]
        total = len(questions)
        stats = session["stats"]
        correct = stats.get("correct", 0)
        wrong = stats.get("wrong", 0)

        markup = types.InlineKeyboardMarkup(row_width=2)
        markup.add(
            types.InlineKeyboardButton("⏭ Пропустить", callback_data=QUIZ_SKIP_CB),
            types.InlineKeyboardButton("❌ Завершить", callback_data=QUIZ_END_CB),
        )

        bot_obj.send_message(
            chat_id,
            f"📝 **Вопрос {idx + 1}/{total}**\n\n{q['q']}\n\n_Напиши свой ответ в чат._\nПрогресс: ✅ {correct} | ❌ {wrong}",
            parse_mode="Markdown",
            reply_markup=markup,
        )

    def _show_quiz_results(
        bot_obj: telebot.TeleBot, chat_id: int, session: dict[str, Any]
    ) -> None:
        stats = session["stats"]
        correct = stats.get("correct", 0)
        wrong = stats.get("wrong", 0)
        total = correct + wrong
        all_q = len(session["questions"])
        pct = round(correct / total * 100) if total else 0

        bar_len = 10
        filled = round(pct / 100 * bar_len)
        bar = "🟩" * filled + "⬜" * (bar_len - filled)

        msg = f"📊 **Результаты проверки знаний**\n\n{bar}\n\n✅ **Верно:** {correct}/{all_q}\n❌ **Неверно/пропущено:** {wrong}/{all_q}\n**Точность:** {pct}%\n\n"

        if pct >= 80:
            msg += "🏆 **Отлично!** Ты хорошо усвоил материал!\n"
        elif pct >= 50:
            msg += "👍 **Неплохо!** Есть куда расти — повтори слабые места.\n"
        else:
            msg += "📖 **Стоит повторить.** Перечитай конспект и попробуй снова.\n"

        wrong_details = [d for d in stats.get("details", []) if not d.get("correct")]
        if wrong_details:
            msg += "\n**Нужно доработать:**\n"
            for i, d in enumerate(wrong_details, 1):
                fb = d.get("feedback", "")
                if fb:
                    fb_short = fb[:200] + "…" if len(fb) > 200 else fb
                    msg += f"{i}. {fb_short}\n"

        markup = types.InlineKeyboardMarkup()
        markup.add(
            types.InlineKeyboardButton("🔄 Пройти заново", callback_data=START_QUIZ_CB)
        )
        bot_obj.send_message(chat_id, msg, parse_mode="Markdown", reply_markup=markup)

        # Save questions for retry, then remove active session
        _save_questions(chat_id, session["questions"])
        _del_quiz(chat_id)

    def _send_long(
        bot_obj: telebot.TeleBot, chat_id: int, text: str, parse_mode: str | None = None
    ) -> None:
        for part in _split_long_message(text):
            try:
                bot_obj.send_message(chat_id, part, parse_mode=parse_mode)
            except Exception as e:
                logger.warning(f"Send failed with {parse_mode}, retrying plain: {e}")
                try:
                    bot_obj.send_message(chat_id, part)
                except Exception as e2:
                    logger.error(f"Fallback send also failed: {e2}")

    # -------------------------------------------------------------------
    # Start polling
    # -------------------------------------------------------------------
    logger.info(f"Starting bot... (allowed_users: {ALLOWED_IDS or 'all'})")
    logger.info(f"VENV python: {VENV_PYTHON}")

    while True:
        try:
            bot.infinity_polling(timeout=30, long_polling_timeout=30)
        except Exception as e:
            logger.error(f"Polling error: {e}")
            time.sleep(5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Learning Companion Telegram Bot")
    parser.add_argument(
        "--token", default="", help="Telegram bot token (or COMPANION_BOT_TOKEN env)"
    )
    parser.add_argument(
        "--allowed",
        default="",
        help="Comma-separated Telegram user IDs allowed to use the bot",
    )
    args = parser.parse_args()

    token = args.token or _env_or_dotenv("COMPANION_BOT_TOKEN")
    if not token:
        print(
            "Error: Bot token required.\n  Set COMPANION_BOT_TOKEN in env or ~/.hermes/.env\n  Or pass --token TOKEN"
        )
        sys.exit(1)

    if args.allowed:
        global ALLOWED_IDS
        ALLOWED_IDS = [int(x.strip()) for x in args.allowed.split(",") if x.strip()]

    run_bot(token)


if __name__ == "__main__":
    main()
