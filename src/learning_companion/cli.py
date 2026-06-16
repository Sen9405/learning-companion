"""CLI entry point — run, resume, check commands."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any
from uuid import uuid4

from learning_companion import __version__
from learning_companion.graph import LearningState, make_initial_state
from learning_companion.graph.builder import compile_agent
from learning_companion.graph.tracing import setup_tracing
from learning_companion.llm import get_cost, llm_call, reset_counters
from learning_companion.memory import get_ltm
from learning_companion.telegram import (
    notify_telegram,
    send_telegram,
    send_telegram_long,
    _strip_questions_section,
)


def _detect_source_type(url: str, text: str) -> str:
    if url:
        if "youtube" in url.lower() or "youtu.be" in url.lower():
            return "youtube"
        if url.lower().endswith(".pdf"):
            return "pdf"
        return "web"
    return "text"


def _get_checkpointer() -> Any:
    """Get PostgresSaver checkpointer, or None if unavailable."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        pg_dsn = os.environ.get("PG_DSN", "")
        if not pg_dsn:
            return None

        checkpointer = PostgresSaver.from_conn_string(pg_dsn)
        # Initialize tables if needed
        try:
            checkpointer.setup()
        except Exception:
            pass
        return checkpointer
    except ImportError:
        return None
    except Exception:
        return None


def run_agent(
    url: str = "",
    text: str = "",
    title: str = "",
    language: str = "",
    enable_tracing: bool = False,
    skip_hitl: bool = False,
) -> dict[str, Any]:
    """Run the full Learning Companion pipeline."""
    run_id = uuid4().hex[:12]
    reset_counters()

    if enable_tracing:
        setup_tracing()

    # Initial state
    initial = make_initial_state(url=url, text=text, title=title, language=language, run_id=run_id)
    initial["source_type"] = _detect_source_type(url, text)

    notify_telegram(
        f"📥 Starting analysis...\n"
        f"Source: {initial['source_type']}\n"
        f"URL: {url or 'N/A'}\n"
        f"Title: {title or 'N/A'}",
        run_id=run_id, stage="planning",
    )

    # Compile agent
    checkpointer = _get_checkpointer()
    agent = compile_agent(checkpointer=checkpointer)

    # Run
    config = {"configurable": {"thread_id": run_id}}
    result = agent.invoke(initial, config)

    state = result if isinstance(result, dict) else {}

    # Get final state
    final_note = state.get("note", "")
    questions = state.get("questions_list", "")
    analysis = state.get("analysis", "")

    # Send results
    cost_info = get_cost()
    cost_str = (
        f"💰 Cost: ${cost_info['cost']:.4f} "
        f"({cost_info['prompt_tokens']}+{cost_info['completion_tokens']} tokens)"
    )

    notify_telegram(f"✅ Analysis complete!\n{cost_str}", run_id=run_id, stage="done")

    # Send summary to Telegram
    summary = (
        f"📚 **Learning Companion — Результат**\n\n"
        f"**Источник:** {url or 'text input'}\n"
        f"**Run ID:** `{run_id}`\n\n"
    )

    if final_note:
        summary += f"**Заметка:** {len(final_note)} символов\n"
        # Send the full note without questions
        note_to_send = _strip_questions_section(final_note)
        send_telegram_long(note_to_send)
    else:
        summary += "*Заметка не создана*\n"

    if analysis:
        summary += f"**Анализ:** {len(analysis)} символов\n"

    summary += f"\n{cost_str}"
    send_telegram(summary)

    # Если есть вопросы — отправляем отдельно
    if questions:
        send_telegram(
            f"📝 **Вопросы для проверки знаний**\n\n{questions}",
        )

    # Save to Obsidian
    _save_to_obsidian(state)

    return {
        "run_id": run_id,
        "state": state,
        "cost": cost_info,
    }


def _save_to_obsidian(state: LearningState) -> None:
    """Save note to Obsidian vault."""
    note = state.get("note", "")
    title = state.get("title", "Learning Note")
    url = state.get("url", "")

    if not note:
        return

    obsidian_dir = os.path.expanduser("~/Obsidian/Learning/Articles")
    os.makedirs(obsidian_dir, exist_ok=True)

    # Sanitize filename
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:80]
    filepath = os.path.join(obsidian_dir, f"{safe_title}.md")

    with open(filepath, "w") as f:
        f.write(f"# {title}\n\n")
        if url:
            f.write(f"Source: {url}\n\n")
        f.write(note)

    print(f"[Obsidian] Saved: {filepath}")


def resume_agent(
    run_id: str,
    action: str = "resume-analyst",
    skip_hitl: bool = False,
) -> dict[str, Any]:
    """Resume agent from HITL breakpoint.

    Actions: resume-analyst, reject-analyst, save-writer, skip-save
    """
    reset_counters()

    checkpointer = _get_checkpointer()
    if not checkpointer:
        return {"error": "No checkpointer available (PostgresSaver required for resume)"}

    agent = compile_agent(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": run_id}}

    # Get current state
    try:
        state = agent.get_state(config)
    except Exception as e:
        return {"error": f"Cannot get state for run {run_id}: {e}"}

    if action == "resume-analyst":
        notify_telegram("Continuing with analysis...", run_id=run_id, stage="analyst")
        result = agent.invoke(None, config)
        return {"run_id": run_id, "state": result if isinstance(result, dict) else {}}

    elif action == "reject-analyst":
        notify_telegram("Analysis rejected by user.", run_id=run_id, stage="rejected")
        return {"run_id": run_id, "status": "rejected"}

    elif action == "save-writer":
        notify_telegram("Saving note...", run_id=run_id, stage="writer")
        result = agent.invoke(None, config)

        state = result if isinstance(result, dict) else {}
        final_note = state.get("note", "")
        questions = state.get("questions_list", "")

        # Send results
        cost_info = get_cost()
        if final_note:
            note_to_send = _strip_questions_section(final_note)
            send_telegram_long(note_to_send)

        if questions:
            send_telegram(f"📝 **Вопросы для проверки знаний**\n\n{questions}")

        cost_str = (
            f"💰 Cost: ${cost_info['cost']:.4f} "
            f"({cost_info['prompt_tokens']}+{cost_info['completion_tokens']} tokens)"
        )
        notify_telegram(f"✅ Note saved!\n{cost_str}", run_id=run_id, stage="done")

        _save_to_obsidian(state)

        return {"run_id": run_id, "state": state, "cost": cost_info}

    elif action == "skip-save":
        notify_telegram("Save skipped.", run_id=run_id, stage="skipped")
        return {"run_id": run_id, "status": "skipped"}

    else:
        return {"error": f"Unknown action: {action}"}


def run_check(limit: int = 5) -> None:
    """Generate review questions from LTM notes."""
    reset_counters()
    ltm = get_ltm()
    notes = ltm.get_recent_notes(limit)

    if not notes:
        msg = "📝 **Нет заметок для проверки**\n\nСначала добавь хотя бы одну заметку через `run`."
        print(msg)
        send_telegram(msg)
        return

    all_questions: list[str] = []
    for note in notes:
        system = (
            "Ты — ассистент проверки знаний. Сгенерируй 3 вопроса по теме "
            "для проверки понимания. Вопросы должны проверять понимание, а не факты."
        )
        prompt = (
            f"Title: {note.get('title', 'Untitled')}\n"
            f"Summary: {note.get('summary', '')[:2000]}"
        )
        resp, _ = llm_call(system, [{"role": "user", "content": prompt}])
        all_questions.append(f"### {note.get('title', 'Untitled')}\n{resp}")

    if all_questions:
        full = "📝 **Проверка знаний**\n\n" + "\n\n".join(all_questions)
        send_telegram_long(full)

    cost_info = get_cost()
    notify_telegram(
        f"✅ Questions generated for {len(notes)} notes.\n"
        f"💰 Cost: ${cost_info['cost']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Learning Companion v{__version__} — Multi-Agent Learning Note Generator",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # Run
    run_p = sub.add_parser("run", help="Run full analysis pipeline")
    run_p.add_argument("--url", default="", help="URL to analyze (YouTube, web, PDF)")
    run_p.add_argument("--text", default="", help="Direct text input")
    run_p.add_argument("--title", default="", help="Note title")
    run_p.add_argument("--language", default="", choices=["ru", "en", ""], help="Force language")
    run_p.add_argument("--trace", action="store_true", help="Enable Phoenix tracing")
    run_p.add_argument("--no-hitl", action="store_true", help="Skip human-in-the-loop pauses")

    # Resume
    res_p = sub.add_parser("resume", help="Resume from HITL breakpoint")
    res_p.add_argument("run_id", help="Run ID to resume")
    res_p.add_argument("action", nargs="?", default="resume-analyst",
                       choices=["resume-analyst", "reject-analyst", "save-writer", "skip-save"],
                       help="Action to take")

    # Check
    check_p = sub.add_parser("check", help="Generate review questions from LTM")
    check_p.add_argument("--limit", type=int, default=5, help="Number of recent notes to use")

    args = parser.parse_args()

    if args.command == "run":
        if not args.url and not args.text:
            parser.error("Either --url or --text is required")
        result = run_agent(
            url=args.url,
            text=args.text,
            title=args.title,
            language=args.language,
            enable_tracing=args.trace,
            skip_hitl=args.no_hitl,
        )
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
        print(f"\nDone. Run ID: {result.get('run_id', '?')}")
        cost = result.get("cost", {})
        print(f"Tokens: {cost.get('prompt_tokens', 0)} in / {cost.get('completion_tokens', 0)} out")
        print(f"Cost: ${cost.get('cost', 0):.4f}")

    elif args.command == "resume":
        result = resume_agent(args.run_id, args.action)
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
        print(f"Resumed. Run ID: {args.run_id}")
        cost = result.get("cost", {})
        if cost:
            print(f"Cost: ${cost.get('cost', 0):.4f}")

    elif args.command == "check":
        run_check(limit=args.limit)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
