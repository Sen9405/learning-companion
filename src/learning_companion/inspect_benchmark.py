"""Inspect AI benchmark для Learning Companion.

Оборачивает golden dataset в Inspect Task:
- Dataset: 36 примеров с текстом и expected-полями
- Solver: запускает Learning Companion агента
- Scorer: оценивает заметку по single-turn + LLM-as-judge

Usage:
    learning-companion inspect-benchmark
    learning-companion inspect-benchmark --golden tests/golden.json
    inspect view  # просмотр логов
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Inspect импорты
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (
    Score,
    scorer,
    Target,
)
from inspect_ai.solver import (
    Solver,
    TaskState,
    solver,
)

from learning_companion.eval import (
    build_agent,
    eval_single_turn,
    run_single,
)
from learning_companion.llm import reset_counters


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _golden_to_samples(path: str) -> list[Sample]:
    """Конвертирует golden.json в Inspect Samples."""
    with open(path) as f:
        data = json.load(f)

    samples = []
    for item in data:
        inp = item["input"]
        expected = item["expected"]

        # Храним всё в metadata чтобы solver мог прочитать
        samples.append(
            Sample(
                input=inp.get("text", ""),
                target=inp.get("language", "ru"),  # target = язык
                id=item["id"],
                metadata={
                    "title": inp.get("title", ""),
                    "language": inp.get("language", "ru"),
                    "expected": expected,
                    "difficulty": item.get("difficulty", "unknown"),
                    "name": item.get("name", ""),
                },
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@solver
def learning_companion_solver() -> Solver:
    """Solver: запускает Learning Companion агента на каждом sample."""

    async def solve(state: TaskState, generate: Any) -> TaskState:
        metadata = state.metadata
        text = state.input_text
        language = metadata.get("language", "ru")

        # Собираем агента
        agent = build_agent()

        # Создаём пример в формате golden.json
        example = {
            "id": state.sample_id or "unknown",
            "name": metadata.get("name", ""),
            "difficulty": metadata.get("difficulty", "unknown"),
            "input": {
                "text": text,
                "title": metadata.get("title", ""),
                "language": language,
            },
            "expected": metadata.get("expected", {}),
        }

        reset_counters()
        result_state = run_single(agent, example)

        # Сохраняем полное состояние для scorer
        state.metadata["agent_state"] = result_state
        state.metadata["note"] = result_state.get("note", "")
        state.metadata["concepts"] = result_state.get("concepts", [])
        state.metadata["questions"] = result_state.get("questions", [])

        # LLM-as-judge оценка
        from learning_companion.eval import eval_llm_judge

        note = result_state.get("note", "")
        judge = eval_llm_judge(note, language, text) if note else {"score": 0.0, "details": {}, "weaknesses": ["no note"]}
        state.metadata["judge_score"] = judge["score"]
        state.metadata["judge_details"] = judge["details"]
        state.metadata["judge_weaknesses"] = judge.get("weaknesses", [])

        # Single-turn
        st = eval_single_turn(result_state, metadata.get("expected", {}))
        state.metadata["single_turn_score"] = st["score"]
        state.metadata["single_turn_issues"] = st["issues"]

        # Output — финальная заметка (для логирования)
        state.output.completion = note[:2000] if note else "No note generated"

        return state

    return solve


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@scorer(metrics=[])
def learning_companion_scorer():
    """Scorer: оценивает результат работы агента."""

    async def score(state: TaskState, target: Target) -> Score:
        judge_score = state.metadata.get("judge_score", 0.0)
        st_score = state.metadata.get("single_turn_score", 0.0)
        language = target.text

        # Проходной балл — 0.5 по каждому типу
        judge_pass = judge_score >= 0.5
        st_pass = st_score >= 0.5

        overall = "pass" if (judge_pass and st_pass) else "fail"

        explanation = (
            f"Single-turn: {st_score:.2f} {'✅' if st_pass else '❌'} "
            f"({' | '.join(state.metadata.get('single_turn_issues', [])) if state.metadata.get('single_turn_issues') else 'ok'})\n"
            f"Judge score: {judge_score:.2f} {'✅' if judge_pass else '❌'}\n"
            f"Language: {language}\n"
            f"Weaknesses: {', '.join(state.metadata.get('judge_weaknesses', [])) or 'none'}\n"
            f"Note length: {len(state.metadata.get('note', ''))} chars"
        )

        return Score(
            value=overall,
            answer=state.output.completion[:500] if state.output.completion else "",
            explanation=explanation,
        )

    return score


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@task
def learning_companion_benchmark(golden: str = "tests/golden.json") -> Task:
    """Learning Companion benchmark task for Inspect."""
    dataset = _golden_to_samples(golden)
    return Task(
        dataset=dataset,
        solver=[learning_companion_solver()],
        scorer=learning_companion_scorer(),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_inspect_benchmark(golden_path: str = "tests/golden.json") -> dict:
    """Запускает Inspect benchmark и возвращает результаты."""
    abs_path = str(Path(golden_path).resolve())

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logs = eval(
        learning_companion_benchmark(golden=abs_path),
        model="openai/deepseek-chat",
        model_base_url="https://api.deepseek.com",
        log_dir=str(log_dir),
    )

    results = []
    for log_entry in logs:
        if log_entry.samples:
            for sample in log_entry.samples:
                if sample.scores:
                    # Use scores dict (new API) — get the main scorer result
                    scores_list = list(sample.scores.values()) if hasattr(sample.scores, "values") else []
                    main_score = scores_list[0] if scores_list else None
                    results.append({
                        "sample_id": sample.id,
                        "score": main_score.value if main_score and hasattr(main_score, "value") else "unknown",
                        "judge_score": None,  # metadata available in sample.store
                        "explanation": str(getattr(main_score, "explanation", "")),
                    })

    total = len(results)
    passed = sum(1 for r in results if r.get("score") == "pass")
    failed = total - passed

    report = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total * 100, 1) if total else 0,
        "results": results,
    }

    print("\n📊 INSPECT BENCHMARK RESULTS")
    print(f"  Total:  {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Rate:   {report['pass_rate']}%")
    print(f"  Logs:   {log_dir}/")

    return report


if __name__ == "__main__":
    run_inspect_benchmark()
