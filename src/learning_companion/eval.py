"""Eval — LLM-as-judge прогон по golden dataset.

Usage:
    python -m learning_companion eval --golden tests/golden.json
    python -m learning_companion eval --golden tests/golden.json --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from learning_companion.graph import make_initial_state
from learning_companion.graph.builder import compile_agent
from learning_companion.graph.nodes import (
    analyst_node,
    fetcher_node,
    has_content,
    planner_node,
    writer_node,
)
from learning_companion.llm import llm_call, reset_counters

# ---------------------------------------------------------------------------
# LLM-as-judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """Ты — судья качества учебных заметок, созданных AI-агентом.
Оцени заметку по 5 критериям, каждый от 0.0 до 1.0:

1. completeness (полнота) — все ли ключевые темы исходного материала освещены?
2. structure (структура) — есть ли заголовки, разделы, логичная организация?
3. accuracy (точность) — нет ли фактических ошибок или противоречий?
4. practical_value (практическая ценность) — есть ли конкретные применимые выводы?
5. language (язык) — соответствует ли язык исходному материалу?

Учитывай ожидаемый язык (ru или en) — если заметка не на том языке, ставь 0.

Ответь ТОЛЬКО в формате JSON (без пояснений):
{{"scores": {{"completeness": 0.0, "structure": 0.0, "accuracy": 0.0, "practical_value": 0.0, "language": 0.0}}, "total": 0.0, "weaknesses": ["..."]}}"""


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------


def load_golden(path: str) -> list[dict]:
    """Загружает golden dataset из JSON."""
    with open(path) as f:
        return json.load(f)


def build_agent():
    """Собирает агента (без checkpointer — eval без HITL)."""
    return compile_agent(checkpointer=None)


def run_single(agent: Any, example: dict) -> dict[str, Any]:
    """Прогоняет один пример через агента (без HITL)."""
    inp = example["input"]
    initial = make_initial_state(
        url="",
        text=inp.get("text", ""),
        title=inp.get("title", ""),
        language=inp.get("language", "ru"),
        run_id=f"eval-{example['id']}",
    )
    initial["source_type"] = "text"

    config = {"configurable": {"thread_id": f"eval-{example['id']}"}}
    result = agent.invoke(initial, config)
    state = result if isinstance(result, dict) else {}
    return state


def judge_note(note: str, language: str, input_text: str) -> dict:
    """Оценивает заметку через LLM-as-judge."""
    prompt = (
        f"Original material (first 1000 chars):\n{input_text[:1000]}\n\n"
        f"Expected language: {language}\n\n"
        f"Generated note:\n{note[:3000]}\n"
    )
    resp, _ = llm_call(JUDGE_SYSTEM, [{"role": "user", "content": prompt}],
                        response_model=None, max_tokens=1024, temperature=0.0)
    try:
        data = json.loads(resp)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", resp, re.DOTALL)
        data = json.loads(match.group()) if match else {"scores": {}, "total": 0.0, "weaknesses": ["parse error"]}
    return data


def check_basic(state: dict, expected: dict) -> dict:
    """Базовые проверки без LLM."""
    note = state.get("note", "")
    issues = []
    passed = 0
    total = 0

    # Длина заметки
    min_len = expected.get("min_note_length", 0)
    total += 1
    if len(note) >= min_len:
        passed += 1
    else:
        issues.append(f"note too short: {len(note)} < {min_len}")

    # Наличие концептов
    if expected.get("must_have_concepts"):
        total += 1
        concepts = state.get("concepts", [])
        if concepts:
            passed += 1
        else:
            issues.append("no concepts extracted")

    # Наличие вопросов
    if expected.get("must_have_questions"):
        total += 1
        questions = state.get("questions", [])
        if questions:
            passed += 1
        else:
            issues.append("no questions extracted")

    # Проверка языка
    exp_lang = expected.get("language", "ru")
    if exp_lang == "en":
        total += 1
        import re
        has_cyrillic = bool(re.search(r"[а-яА-Я]", note))
        if not has_cyrillic:
            passed += 1
        else:
            issues.append("note has cyrillic but expected English")

    score = passed / total if total > 0 else 0
    return {"basic_score": score, "basic_passed": passed, "basic_total": total, "issues": issues}


def run_eval(golden_path: str, threshold: float = 0.5) -> dict:
    """Запускает полный eval по golden dataset."""
    golden = load_golden(golden_path)
    agent = build_agent()

    results = []
    total_scores = []
    total_basic = []

    for example in golden:
        reset_counters()
        print(f"\n{'='*60}")
        print(f"[{example['id']}] {example['name']}")

        state = run_single(agent, example)
        note = state.get("note", "")

        if not note:
            print(f"  ❌ No note generated")
            results.append({**example, "status": "fail", "error": "no note", "score": 0.0})
            continue

        print(f"  Note: {len(note)} chars")

        # Basic checks
        basic = check_basic(state, example["expected"])
        total_basic.append(basic["basic_score"])
        print(f"  Basic score: {basic['basic_score']:.2f} ({basic['basic_passed']}/{basic['basic_total']})")
        for issue in basic["issues"]:
            print(f"    ⚠ {issue}")

        # LLM-as-judge (если есть заметка)
        judge_result = judge_note(
            note,
            language=example["input"].get("language", "ru"),
            input_text=example["input"].get("text", ""),
        )
        judge_score = judge_result.get("total", 0.0)
        total_scores.append(judge_score)

        print(f"  Judge score: {judge_score:.2f}")
        for w in judge_result.get("weaknesses", []):
            print(f"    ⚠ {w}")

        results.append({
            **example,
            "status": "pass" if judge_score >= threshold else "fail",
            "basic_score": basic["basic_score"],
            "judge_score": judge_score,
            "judge_details": judge_result.get("scores", {}),
        })

    # Итоги
    avg_judge = sum(total_scores) / len(total_scores) if total_scores else 0.0
    avg_basic = sum(total_basic) / len(total_basic) if total_basic else 0.0
    passed_count = sum(1 for r in results if r["status"] == "pass")

    report = {
        "total": len(golden),
        "passed": passed_count,
        "failed": len(golden) - passed_count,
        "avg_basic_score": round(avg_basic, 3),
        "avg_judge_score": round(avg_judge, 3),
        "threshold": threshold,
        "results": results,
    }

    # Сохраняем отчёт
    report_path = Path("eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"📊 EVAL RESULTS")
    print(f"  Passed: {passed_count}/{len(golden)}")
    print(f"  Avg basic score: {avg_basic:.3f}")
    print(f"  Avg judge score: {avg_judge:.3f}")
    print(f"  Threshold: {threshold}")
    print(f"  Overall: {'✅ PASS' if avg_judge >= threshold else '❌ FAIL'}")
    print(f"  Report saved: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Learning Companion Eval")
    parser.add_argument("--golden", default="tests/golden.json",
                        help="Path to golden dataset JSON")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum average score to pass (0.0-1.0)")
    args = parser.parse_args()

    report = run_eval(args.golden, args.threshold)

    if report["avg_judge_score"] < args.threshold:
        sys.exit(1)


if __name__ == "__main__":
    main()
