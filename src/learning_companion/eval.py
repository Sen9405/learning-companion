"""Eval — 4 типа eval'ов для Learning Companion.

Типы:
  1. single-turn: базовые проверки (длина, концепты, вопросы, язык)
  2. trajectory: оценка полной траектории агента (план → поиск → анализ → запись)
  3. llm-as-judge: оценка заметки через LLM по 5 критериям
  4. end-state: проверка финального состояния графа

Usage:
    learning-companion eval --golden tests/golden.json
    learning-companion eval --golden tests/golden.json --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from learning_companion.graph import make_initial_state
from learning_companion.graph.builder import compile_agent
from learning_companion.llm import llm_call, reset_counters

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """Ты — судья качества учебных заметок, созданных AI-агентом.
Оцени заметку по 5 критериям, каждый от 0.0 до 1.0:

1. completeness (полнота) — все ли ключевые темы исходного материала освещены?
2. structure (структура) — есть ли заголовки, разделы, логичная организация?
3. accuracy (точность) — нет ли фактических ошибок или противоречий?
4. practical_value (практическая ценность) — есть ли конкретные применимые выводы?
5. language (язык) — соответствует ли язык исходному материалу?

Учитывай ожидаемый язык (ru или en) — если заметка не на том языке, ставь 0.

ВАЖНО: Каждый score от 0.0 до 1.0 (дробь). total = среднее арифметическое 5 scores, тоже от 0.0 до 1.0.

Ответь ТОЛЬКО в формате JSON (без пояснений):
{"scores": {"completeness": 0.0, "structure": 0.0, "accuracy": 0.0, "practical_value": 0.0, "language": 0.0}, "total": 0.0, "weaknesses": ["..."]}"""

TRAJECTORY_SYSTEM = """Ты оцениваешь качество работы AI-агента по созданию учебной заметки.
Агент прошёл 4 этапа: planning (план), fetching (поиск информации),
analysis (анализ), writing (написание заметки).
Оцени траекторию по 4 критериям, каждый от 0.0 до 1.0:

1. plan_quality (качество плана) — соответствует ли план задаче?
2. analysis_depth (глубина анализа) — выделены ли ключевые концепты и связи?
3. progression (прогрессия) — логично ли развивается мысль от плана к заметке?
4. no_regression (без регрессии) — не потерялась ли информация между этапами?

Ответь ТОЛЬКО в формате JSON:
{"scores": {"plan_quality": 0.0, "analysis_depth": 0.0, "progression": 0.0, "no_regression": 0.0}, "total": 0.0, "weaknesses": ["..."]}"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_golden(path: str) -> list[dict]:
    """Загружает golden dataset из JSON."""
    with open(path) as f:
        return json.load(f)


def build_agent():
    """Собирает агента (без checkpointer — eval без HITL)."""
    return compile_agent(checkpointer=None)


def run_single(agent: Any, example: dict) -> dict[str, Any]:
    """Прогоняет один пример через агента и возвращает полное состояние."""
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
    return result if isinstance(result, dict) else {}


def _parse_json(text: str) -> dict:
    """Парсит JSON из ответа LLM, с fallback на regex."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"scores": {}, "total": 0.0, "weaknesses": ["parse error"]}


# ---------------------------------------------------------------------------
# Eval type 1: single-turn (basic checks)
# ---------------------------------------------------------------------------


def eval_single_turn(state: dict, expected: dict) -> dict:
    """Базовые проверки одного прогона: длина, концепты, вопросы, язык.

    Это single-turn eval — оцениваем только финальный output.
    """
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
        has_cyrillic = bool(re.search(r"[а-яА-Я]", note))
        if not has_cyrillic:
            passed += 1
        else:
            issues.append("note has cyrillic but expected English")

    score = passed / total if total > 0 else 0
    return {"score": score, "passed": passed, "total": total, "issues": issues}


# ---------------------------------------------------------------------------
# Eval type 2: trajectory
# ---------------------------------------------------------------------------


def eval_trajectory(state: dict, input_text: str) -> dict:
    """Оценивает полную траекторию: план → анализ → заметка.

    Сравнивает промежуточные состояния и проверяет логику progression.
    """
    plan = state.get("content", "")
    analysis = state.get("analysis", "")
    note = state.get("note", "")
    issues = []
    passed = 0
    total = 4  # 4 checks: plan exists, analysis exists, note exists, progression

    # 1. Есть ли план (контент получен)
    has_content = len(plan) >= 50
    if has_content:
        passed += 1
    else:
        issues.append("no content fetched or too short")

    # 2. Есть ли анализ
    has_analysis = len(analysis) >= 100
    if has_analysis:
        passed += 1
    else:
        issues.append("analysis too short or missing")

    # 3. Есть ли заметка
    has_note = len(note) >= 100
    if has_note:
        passed += 1
    else:
        issues.append("note too short or missing")

    # 4. Progression: анализ не короче контента (разумная пропорция)
    if has_content and has_analysis:
        if len(analysis) >= len(plan) * 0.3:
            passed += 1
        else:
            issues.append(f"analysis too short relative to content: {len(analysis)} vs {len(plan)}")
    elif has_content:
        issues.append("cannot check progression — no analysis")
    else:
        total -= 1

    score = passed / total if total > 0 else 0
    return {"score": score, "passed": passed, "total": total, "issues": issues}


# ---------------------------------------------------------------------------
# Eval type 3: LLM-as-judge
# ---------------------------------------------------------------------------


def eval_llm_judge(note: str, language: str, input_text: str) -> dict:
    """Оценивает качество заметки через LLM по 5 критериям."""
    if not note:
        return {"score": 0.0, "details": {}, "weaknesses": ["empty note"]}

    prompt = (
        f"Original material (first 1000 chars):\n{input_text[:1000]}\n\n"
        f"Expected language: {language}\n\n"
        f"Generated note:\n{note[:3000]}\n"
    )
    resp, _ = llm_call(
        JUDGE_SYSTEM,
        [{"role": "user", "content": prompt}],
        response_model=None,
        max_tokens=1024,
        temperature=0.0,
        stage="eval.judge",
        cache=True,
    )
    data = _parse_json(resp)
    return {
        "score": data.get("total", 0.0),
        "details": data.get("scores", {}),
        "weaknesses": data.get("weaknesses", []),
    }


# ---------------------------------------------------------------------------
# Eval type 4: end-state
# ---------------------------------------------------------------------------


def eval_end_state(state: dict) -> dict:
    """Проверяет финальное состояние графа — корректность структуры.

    end-state eval не оценивает качество, а проверяет, что граф
    завершился в валидном состоянии.
    """
    issues = []
    passed = 0
    total = 5

    # 1. Финальный stage
    stage = state.get("stage", "")
    total += 0  # already counted
    if stage in ("done", "writer"):
        passed += 1
    else:
        issues.append(f"unexpected final stage: {stage}")

    # 2. Есть run_id
    if state.get("run_id"):
        passed += 1
    else:
        issues.append("no run_id in final state")
        total -= 1

    # 3. Ни одно поле не пустое если должно быть (note)
    if state.get("note"):
        passed += 1
    else:
        issues.append("note field empty in final state")

    # 4. Нет ошибок в state (extra fields check)
    required = {"url", "text", "title", "content", "analysis", "note", "concepts",
                "questions", "run_id", "source_type", "stage", "language"}
    missing = required - set(state.keys())
    if not missing:
        passed += 1
    else:
        issues.append(f"missing state fields: {missing}")

    # 5. source_type установлен
    if state.get("source_type"):
        passed += 1
    else:
        issues.append("source_type not set")

    score = passed / total if total > 0 else 0
    return {"score": score, "passed": passed, "total": total, "issues": issues}


# ---------------------------------------------------------------------------
# Full eval runner
# ---------------------------------------------------------------------------


def run_eval(golden_path: str, threshold: float = 0.5) -> dict:
    """Запускает все 4 типа eval'ов по golden dataset."""
    golden = load_golden(golden_path)
    agent = build_agent()

    results = []
    tracker = {
        "single_turn": {"scores": [], "passed": 0, "count": 0},
        "trajectory": {"scores": [], "passed": 0, "count": 0},
        "llm_judge": {"scores": [], "passed": 0, "count": 0},
        "end_state": {"scores": [], "passed": 0, "count": 0},
    }

    for example in golden:
        reset_counters()
        print(f"\n{'='*60}")
        print(f"[{example['id']}] {example['name']} ({example.get('difficulty', '?')})")

        state = run_single(agent, example)
        note = state.get("note", "")

        example_results: dict[str, Any] = {
            "id": example["id"],
            "name": example["name"],
            "difficulty": example.get("difficulty", "unknown"),
            "input": example["input"],
            "expected": example["expected"],
        }

        # --- Single-turn ---
        st = eval_single_turn(state, example["expected"])
        example_results["single_turn"] = st
        tracker["single_turn"]["scores"].append(st["score"])
        tracker["single_turn"]["count"] += 1
        if st["score"] >= 0.5:
            tracker["single_turn"]["passed"] += 1
        print(f"  Single-turn: {st['score']:.2f} ({st['passed']}/{st['total']}) "
              f"{'⚠ ' + ' | '.join(st['issues']) if st['issues'] else '✅'}")

        # --- Trajectory ---
        tr = eval_trajectory(state, example["input"].get("text", ""))
        example_results["trajectory"] = tr
        tracker["trajectory"]["scores"].append(tr["score"])
        tracker["trajectory"]["count"] += 1
        if tr["score"] >= 0.5:
            tracker["trajectory"]["passed"] += 1
        print(f"  Trajectory:  {tr['score']:.2f} ({tr['passed']}/{tr['total']}) "
              f"{'⚠ ' + ' | '.join(tr['issues']) if tr['issues'] else '✅'}")

        # --- LLM-as-judge ---
        if note:
            lj = eval_llm_judge(
                note,
                language=example["input"].get("language", "ru"),
                input_text=example["input"].get("text", ""),
            )
        else:
            lj = {"score": 0.0, "details": {}, "weaknesses": ["no note generated"]}
        example_results["llm_judge"] = lj
        tracker["llm_judge"]["scores"].append(lj["score"])
        tracker["llm_judge"]["count"] += 1
        if lj["score"] >= threshold:
            tracker["llm_judge"]["passed"] += 1
        print(f"  LLM-judge:   {lj['score']:.2f} "
              f"{'⚠ ' + ' | '.join(lj['weaknesses']) if lj.get('weaknesses') else '✅'}")

        # --- End-state ---
        es = eval_end_state(state)
        example_results["end_state"] = es
        tracker["end_state"]["scores"].append(es["score"])
        tracker["end_state"]["count"] += 1
        if es["score"] >= 0.5:
            tracker["end_state"]["passed"] += 1
        print(f"  End-state:   {es['score']:.2f} ({es['passed']}/{es['total']}) "
              f"{'⚠ ' + ' | '.join(es['issues']) if es['issues'] else '✅'}")

        example_results["overall_pass"] = (
            st["score"] >= 0.5
            and tr["score"] >= 0.5
            and lj["score"] >= threshold
            and es["score"] >= 0.5
        )
        results.append(example_results)

    # --- Итоги ---
    def _summarize_eval(name: str, data: dict) -> dict:
        scores = data["scores"]
        avg = sum(scores) / len(scores) if scores else 0.0
        return {
            "count": data["count"],
            "passed": data["passed"],
            "passed_pct": round(data["passed"] / data["count"] * 100, 1) if data["count"] else 0,
            "avg_score": round(avg, 3),
        }

    eval_summary = {k: _summarize_eval(k, v) for k, v in tracker.items()}
    all_passed = sum(1 for r in results if r["overall_pass"])
    overall_avg = (
        sum(eval_summary[k]["avg_score"] for k in eval_summary) / len(eval_summary)
    )

    # Разбивка по difficulty
    by_difficulty: dict[str, dict] = {}
    for r in results:
        d = r.get("difficulty", "unknown")
        if d not in by_difficulty:
            by_difficulty[d] = {"count": 0, "overall_passed": 0}
        by_difficulty[d]["count"] += 1
        if r.get("overall_pass"):
            by_difficulty[d]["overall_passed"] += 1

    diff_report = {}
    for d, info in by_difficulty.items():
        diff_report[d] = {
            "count": info["count"],
            "overall_passed": info["overall_passed"],
            "overall_passed_pct": round(info["overall_passed"] / info["count"] * 100, 1)
            if info["count"] else 0,
        }

    report: dict[str, Any] = {
        "total": len(golden),
        "all_passed": all_passed,
        "all_failed": len(golden) - all_passed,
        "overall_avg_score": round(overall_avg, 3),
        "threshold": threshold,
        "eval_types": eval_summary,
        "by_difficulty": diff_report,
        "results": results,
    }

    # Сохраняем
    report_path = Path("eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Вывод
    print(f"\n{'='*60}")
    print("📊 EVAL RESULTS — 4 eval types")
    for k, v in eval_summary.items():
        print(f"  {k:15s}: {v['passed']}/{v['count']} passed ({v['passed_pct']}%), "
              f"avg={v['avg_score']:.2f}")
    print(f"  {'─'*50}")
    if diff_report:
        print("  By difficulty:")
        for d_name in ["basic", "intermediate", "advanced"]:
            if d_name in diff_report:
                d = diff_report[d_name]
                print(f"    {d_name:15s}: {d['overall_passed']}/{d['count']} "
                      f"({d['overall_passed_pct']}%)")
    print(f"  {'─'*50}")
    print(f"  Overall:      {all_passed}/{len(golden)} all-4-pass ({overall_avg:.2f} avg)")
    print(f"  Threshold:    {threshold}")
    print(f"  {'✅ PASS' if overall_avg >= threshold else '❌ FAIL'}")
    print(f"  Report:       {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Learning Companion Eval")
    parser.add_argument("--golden", default="tests/golden.json",
                        help="Path to golden dataset JSON")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum average score to pass (0.0-1.0)")
    args = parser.parse_args()

    report = run_eval(args.golden, args.threshold)
    if report["overall_avg_score"] < args.threshold:
        sys.exit(1)


if __name__ == "__main__":
    main()
