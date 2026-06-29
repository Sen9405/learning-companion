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
from learning_companion.hardening import build_hardening_report
from learning_companion.ledger import RunLedger
from learning_companion.llm import get_cost, llm_call, reset_counters
from learning_companion.memory import get_ltm
from learning_companion.settings import get_settings
from learning_companion.telegram import (
    notify_telegram,
    send_telegram,
    send_telegram_long,
    send_telegram_pdf,
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
        from psycopg import Connection

        pg_dsn = os.environ.get("PG_DSN", "")
        if not pg_dsn:
            # Try Unix socket (peer auth) first
            for socket_dir in ["/var/run/postgresql", "/run/postgresql"]:
                if os.path.exists(socket_dir):
                    dsn = f"postgresql://sen@/learning_companion?host={socket_dir}"
                    try:
                        conn = Connection.connect(dsn)
                        cp = PostgresSaver(conn=conn)
                        cp.setup()
                        return cp
                    except Exception:
                        continue
            return None

        conn = Connection.connect(pg_dsn)
        checkpointer = PostgresSaver(conn=conn)
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
    if skip_hitl:
        checkpointer = None
    agent = compile_agent(checkpointer=checkpointer)

    # Run
    config = {"configurable": {"thread_id": run_id}}
    result = agent.invoke(initial, config)
    # Get final state
    state = result if isinstance(result, dict) else {}

    # Get final state
    final_note = state.get("note", "")
    questions = state.get("questions_list", "")
    analysis = state.get("analysis", "")
    title = state.get("title", "Learning Note")

    cost_info = get_cost()
    cost_str = (
        f"💰 Cost: ${cost_info['cost']:.4f} "
        f"({cost_info['prompt_tokens']}+{cost_info['completion_tokens']} tokens)"
    )

    # Bot mode: print structured result to stdout, skip Telegram
    bot_mode = os.environ.get("COMPANION_BOT_MODE", "") == "1"
    if bot_mode:
        import json
        result_data = {
            "title": title,
            "url": url,
            "note": final_note,
            "questions": questions,
            "analysis": analysis,
            "run_id": run_id,
            "cost": cost_info["cost"],
            "tokens_in": cost_info["prompt_tokens"],
            "tokens_out": cost_info["completion_tokens"],
        }
        print(json.dumps(result_data, ensure_ascii=False))
        return {"run_id": run_id, "state": state, "cost": cost_info}

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

    # Save and send PDF
    pdf_path = _save_to_pdf(note=final_note, title=state.get("title", "Learning Note"))
    if pdf_path:
        send_telegram_pdf(pdf_path)

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
        # Добавляем вопросы в Obsidian-файл, если есть
        questions = state.get("questions_list", "")
        if questions:
            # Если вопросы уже в note — не дублируем
            if questions not in note:
                f.write(f"\n\n{questions}\n")

    print(f"[Obsidian] Saved: {filepath}")


def _save_to_pdf(note: str, title: str, pdf_dir: str = "/tmp") -> str | None:
    """Save note as PDF using fpdf2."""
    if not note:
        return None

    try:
        from fpdf import FPDF

        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:60]
        pdf_path = os.path.join(pdf_dir, f"{safe_title}.pdf")

        pdf = FPDF()
        pdf.add_font("DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", uni=True)
        pdf.add_font("DejaVu", "B", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", uni=True)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        pdf.set_font("DejaVu", "B", 16)
        pdf.multi_cell(180, 8, title)

        pdf.ln(4)
        pdf.set_font("DejaVu", "", 11)

        for line in note.split("\n"):
            line = line.strip()
            if not line:
                pdf.ln(4)
            elif line.startswith("# ") or line.startswith("## "):
                is_h1 = line.startswith("# ")
                pdf.ln(2)
                pdf.set_font("DejaVu", "B", 14 if is_h1 else 12)
                pdf.multi_cell(180, 7, line.lstrip("# ").strip())
                pdf.set_font("DejaVu", "", 11)
            elif line.startswith("### "):
                pdf.ln(1)
                pdf.set_font("DejaVu", "B", 11)
                pdf.multi_cell(180, 7, line.lstrip("#").strip())
                pdf.set_font("DejaVu", "", 11)
            elif line.startswith("- ") or line.startswith("* "):
                pdf.set_x(pdf.l_margin + 5)
                pdf.multi_cell(175, 6, "• " + line[2:])
            elif len(line) > 1 and line[0].isdigit() and ". " in line[:4]:
                pdf.set_x(pdf.l_margin + 5)
                pdf.multi_cell(175, 6, line)
            elif line.startswith("> "):
                pdf.set_fill_color(245, 245, 245)
                pdf.set_x(pdf.l_margin + 5)
                pdf.multi_cell(175, 6, line[2:], fill=True)
            else:
                pdf.multi_cell(180, 6, line)

        pdf.output(pdf_path)
        print(f"[PDF] Saved: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
        return pdf_path
    except Exception as e:
        print(f"[PDF] Error: {e}")
        return None


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

        # Save and send PDF
        pdf_path = _save_to_pdf(note=final_note, title=state.get("title", "Learning Note"))
        if pdf_path:
            send_telegram_pdf(pdf_path)

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
        resp, _ = llm_call(system, [{"role": "user", "content": prompt}], stage="check.questions")
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

    # Eval
    eval_p = sub.add_parser("eval", help="Run eval on golden dataset")
    eval_p.add_argument("--golden", default="tests/golden.json",
                        help="Path to golden dataset JSON")
    eval_p.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum average score to pass")

    # Inspect benchmark
    ib_p = sub.add_parser("inspect-benchmark", help="Run Inspect AI benchmark on golden dataset")
    ib_p.add_argument("--golden", default="tests/golden.json",
                      help="Path to golden dataset JSON")

    # Report
    report_p = sub.add_parser("report", help="Show LLM cost/cache ledger report")
    report_p.add_argument("--run-id", default="", help="Optional run ID to scope the report")
    report_p.add_argument("--limit", type=int, default=10, help="Number of recent calls to show")

    # Hardening report (Sprint 3)
    hrd_p = sub.add_parser("hardening-report", help="Run final hardening verification and print report")
    hrd_p.add_argument("--run-id", default="", help="Optional run ID to scope alerts")

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

    elif args.command == "eval":
        from learning_companion.eval import run_eval
        report = run_eval(args.golden, args.threshold)
        if report["overall_avg_score"] < args.threshold:
            sys.exit(1)

    elif args.command == "inspect-benchmark":
        try:
            from learning_companion.inspect_benchmark import run_inspect_benchmark
            run_inspect_benchmark(args.golden)
        except ImportError as e:
            print("Error: inspect-ai not installed. Run: pip install inspect-ai")
            print(f"  Details: {e}")
            sys.exit(1)

    elif args.command == "report":
        run_report(run_id=args.run_id or None, limit=args.limit)

    elif args.command == "hardening-report":
        run_hardening_report(run_id=args.run_id or None)

    else:
        parser.print_help()


def run_report(run_id: str | None = None, limit: int = 10) -> None:
    """Print a production cost/cache report from the persistent ledger."""
    settings = get_settings()
    ledger = RunLedger(settings.run_ledger_db)
    summary = ledger.summary(run_id=run_id)
    recent = ledger.recent_calls(limit=limit)

    title = "LLM Ledger Report"
    if run_id:
        title += f" — run_id={run_id}"
    print(title)
    print("=" * len(title))
    print(f"DB: {settings.run_ledger_db}")
    print(f"Total calls: {summary['total_calls']}")
    print(f"Prompt tokens: {summary['prompt_tokens']}")
    print(f"Completion tokens: {summary['completion_tokens']}")
    print(f"Total cost: ${summary['total_cost']:.6f}")
    cache_rate = (summary['cache_hits'] / summary['total_calls'] * 100) if summary['total_calls'] else 0
    print(f"Cache hits: {summary['cache_hits']} ({cache_rate:.1f}%)")
    print(f"Errors: {summary['errors']}")
    print(f"Average latency: {summary['avg_latency']:.3f}s")

    rows = [row for row in recent if not run_id or row["run_id"] == run_id]
    if not rows:
        return

    print("\nRecent calls:")
    for row in rows:
        marker = "cache" if row["cache_hit"] else "api"
        print(
            f"- #{row['id']} {row['run_id']} {row['stage']} [{marker}] "
            f"tokens={row['prompt_tokens']}/{row['completion_tokens']} "
            f"cost=${row['cost']:.6f} latency={row['latency']:.2f}s"
        )


def run_hardening_report(run_id: str | None = None) -> None:
    """Print a full hardening verification report (Sprint 3)."""
    settings = get_settings()
    from learning_companion.ledger import RunLedger
    ledger = RunLedger(settings.run_ledger_db)
    report = build_hardening_report(settings=settings, ledger=ledger, run_id=run_id)
    print("=== Hardening Report ===")
    if run_id:
        print(f"Run ID: {run_id}")
    print(f"Verdict: {report['verdict'].upper()}")
    print()
    checks_pass = sum(1 for c in report["checks"] if c["status"] == "pass")
    checks_fail = sum(1 for c in report["checks"] if c["status"] == "fail")
    print(f"Checks: {checks_pass} pass / {checks_fail} fail")
    for check in report["checks"]:
        icon = "✅" if check["status"] == "pass" else "❌"
        print(f"  {icon} {check['name']}: {check['detail']}")
    alerts = report.get("alerts", [])
    if alerts:
        print(f"\nAlerts ({len(alerts)}):")
        for alert in alerts:
            icon = "🔴" if alert["severity"] == "critical" else "🟡"
            print(f"  {icon} [{alert['code']}] {alert['message']}")
    else:
        print("\nAlerts: ✅ none")
    print(f"\nSummary: {report['summary']['total_calls']} calls, "
          f"${report['summary']['total_cost']:.6f}")


if __name__ == "__main__":
    main()
