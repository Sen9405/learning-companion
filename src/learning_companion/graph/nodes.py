"""Graph nodes — planner, fetcher, analyst, writer."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from learning_companion.graph import LearningState
from learning_companion.llm import llm_call
from learning_companion.memory import get_ltm


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """Ты — ассистент-планировщик. Определи тип источника и спланируй,
как извлечь знания. Ответь только JSON:
{{
  "source_type": "youtube|web|pdf|text",
  "title_hint": "предполагаемый заголовок",
  "approach": "что нужно извлечь"
}}"""

FETCHER_SYSTEM = """Ты — ассистент-извлекатель. Твоя задача — извлечь полное содержание из переданного текста.
Сохрани все ключевые идеи, факты, примеры и детали. Не сокращай."""

ANALYST_SYSTEM = """Ты — ассистент-аналитик. Проанализируй контент и выдели:

1. **Ключевые концепты** (5-10 терминов с пояснениями)
2. **Связи между концептами** (как они связаны)
3. **Практические выводы** (как применить)
4. **Вопросы для проверки знаний** (5-7 вопросов с ответами)
5. **Глоссарий** (ключевые термины)

Ответь в формате JSON:
{{
  "analysis": "полный анализ текстом",
  "concepts": ["concept1", "concept2", ...],
  "questions": [
    {{"q": "вопрос?", "a": "ответ"}}
  ],
  "glossary": [
    {{"term": "термин", "definition": "определение"}}
  ]
}}"""

WRITER_SYSTEM = """Ты — ассистент-писатель. Создай учебную заметку в формате Markdown на основе анализа.

Структура:
1. Заголовок (название темы)
2. Введение (2-3 предложения о чём материал)
3. Ключевые концепты (список с пояснениями)
4. Основное содержание (подробное, со структурированными разделами)
5. Связи и контекст (как это связано с другими темами)
6. Практические выводы
7. Вопросы для проверки знаний (5-7 вопросов с ответами под спойлерами)

Формат: Markdown, язык: {language}.
Используй ''' для inline кода, ### для подзаголовков.
Вопросы отдели разделителем ---

Верни только заметку, без лишнего текста."""

CHECK_SYSTEM = """Ты — ассистент проверки знаний.
Твоя задача — сгенерировать 5 вопросов на основе заметки для проверки понимания.

Верни только вопросы с краткими ответами под спойлерами в формате Markdown."""
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_language(text: str) -> str:
    """Определяет язык текста (ru/en)."""
    cyrillic = len(re.findall(r"[а-яА-Я]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    return "ru" if cyrillic > latin else "en"


def has_content(state: LearningState) -> bool:
    """Conditional edge: есть ли контент после fetcher."""
    return bool(state.get("content", "").strip())


def has_analysis(state: LearningState) -> bool:
    """Conditional edge: есть ли анализ после analyst."""
    return bool(state.get("analysis", "").strip())


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def planner_node(state: LearningState) -> dict[str, Any]:
    """Определяет источник и планирует подход."""
    url = state.get("url", "")
    text = state.get("text", "")

    # Определяем тип источника
    if url:
        if "youtube" in url.lower() or "youtu.be" in url.lower():
            source_type = "youtube"
        elif url.lower().endswith(".pdf"):
            source_type = "pdf"
        else:
            source_type = "web"
    else:
        source_type = "text"

    language = state.get("language", "") or _detect_language(text or url)

    # Пробуем LLM для уточнения
    try:
        system = PLANNER_SYSTEM + f"\nLanguage: {language}"
        prompt = f"Source: {url or 'direct text'}\nText preview: {text[:500] if text else 'N/A'}"
        resp, meta = llm_call(system, [{"role": "user", "content": prompt}])
        plan = json.loads(resp)
        title = plan.get("title_hint", state.get("title", "")) or "Learning Note"
        source_type = plan.get("source_type", source_type)
    except Exception:
        title = state.get("title", "") or "Learning Note"

    return {
        "source_type": source_type,
        "title": title,
        "stage": "fetcher",
        "language": language,
    }


def fetcher_node(state: LearningState) -> dict[str, Any]:
    """Извлекает контент из источника."""
    url = state.get("url", "")
    text = state.get("text", "")
    source_type = state.get("source_type", "text")

    content = ""

    if source_type == "web" and url:
        content = _fetch_web(url)
    elif source_type == "pdf" and url:
        content = _fetch_pdf(url)
    elif source_type == "youtube" and url:
        content = _fetch_youtube(url)
    elif text:
        content = text

    title = state.get("title", "")
    if not title and content:
        # Извлекаем первую строку как заголовок
        lines = content.strip().split("\n")
        if lines:
            title = lines[0][:100]

    return {
        "content": content,
        "title": title,
        "stage": "analyst",
    }


def _fetch_youtube(url: str) -> str:
    """Извлекает транскрипт YouTube видео."""
    try:
        result = _exec_yt_dlp(url)
        if result:
            return result
    except Exception:
        pass
    return f"[YouTube transcript fetch failed for {url}]"


def _exec_yt_dlp(url: str) -> str:
    """Выполняет yt-dlp для извлечения субтитров."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        outpath = f.name

    try:
        subprocess.run(
            ["yt-dlp", "--write-auto-subs", "--sub-langs", "ru,en",
             "--skip-download", "--print", "title",
             "-o", outpath, url],
            capture_output=True, text=True, timeout=60,
        )
        # Получаем субтитры
        result = subprocess.run(
            ["yt-dlp", "--write-auto-subs", "--sub-langs", "ru,en",
             "--skip-download", "--print", "subtitle",
             "-o", outpath, url],
            capture_output=True, text=True, timeout=60,
        )
        return result.stdout or "[No subtitles found]"
    except subprocess.TimeoutExpired:
        return "[YouTube fetch timed out]"
    except Exception as e:
        return f"[YouTube fetch error: {e}]"
    finally:
        try:
            os.unlink(outpath)
        except OSError:
            pass


def _fetch_web(url: str) -> str:
    """Извлекает текст с веб-страницы."""
    try:
        import httpx
        from bs4 import BeautifulSoup
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Удаляем не-контент элементы
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Ограничиваем длину
        return text[:15000]
    except Exception as e:
        return f"[Web fetch error: {e}]"


def _fetch_pdf(url: str) -> str:
    """Извлекает текст из PDF."""
    try:
        import httpx
        import tempfile
        resp = httpx.get(url, timeout=60, follow_redirects=True)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(resp.content)
            pdf_path = f.name

        try:
            # Попытка через PyMuPDF
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text[:15000]
        except ImportError:
            pass

        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            return text[:15000]
        except ImportError:
            pass

        try:
            from pdfminer.high_level import extract_text
            text = extract_text(pdf_path)
            return text[:15000]
        except ImportError:
            pass

        return f"[PDF downloaded to {pdf_path}, but no PDF parser available]"

    except Exception as e:
        return f"[PDF fetch error: {e}]"


def analyst_node(state: LearningState) -> dict[str, Any]:
    """Анализирует контент и извлекает концепты."""
    content = state.get("content", "")
    language = state.get("language", "ru")

    system = ANALYST_SYSTEM
    prompt = f"Language: {language}\n\nContent:\n{content[:12000]}"

    resp, meta = llm_call(system, [{"role": "user", "content": prompt}],
                          response_model=None, max_tokens=4096)

    # Парсим JSON из ответа
    try:
        data = json.loads(resp)
    except json.JSONDecodeError:
        # Пробуем найти JSON в ответе
        match = re.search(r"\{.*\}", resp, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = {"analysis": resp, "concepts": [], "questions": [], "glossary": []}
        else:
            data = {"analysis": resp, "concepts": [], "questions": [], "glossary": []}

    analysis = data.get("analysis", resp)
    concepts = data.get("concepts", [])
    questions = data.get("questions", [])
    glossary = data.get("glossary", [])

    # Сохраняем в LTM
    try:
        ltm = get_ltm()
        ltm.add_note(
            title=state.get("title", "Learning Note"),
            url=state.get("url", ""),
            summary=analysis[:500],
            concepts=concepts,
            questions=questions,
            glossary=glossary,
        )
    except Exception:
        pass

    # Создаём вопросы для проверки
    questions_text = _format_questions(questions, language)

    return {
        "analysis": analysis,
        "questions_list": questions_text,
        "stage": "writer",
    }


def _format_questions(questions: list[dict], language: str) -> str:
    """Форматирует вопросы в Markdown."""
    if not questions:
        return ""
    lines = ["\n---\n### Вопросы для проверки знаний\n"]
    for i, q in enumerate(questions, 1):
        question = q.get("q", q.get("question", f"Вопрос {i}"))
        answer = q.get("a", q.get("answer", ""))
        lines.append(f"{i}. {question}")
        if answer:
            lines.append(f"   ||{answer}||")
    return "\n".join(lines)


def writer_node(state: LearningState) -> dict[str, Any]:
    """Создаёт учебную заметку."""
    analysis = state.get("analysis", "")
    language = state.get("language", "ru")
    url = state.get("url", "")

    system = WRITER_SYSTEM.format(language=language)

    # Подготовка контекста с LTM
    ltm_context = ""
    try:
        ltm = get_ltm()
        ltm_context = ltm.format_context()
    except Exception:
        pass

    ltm_part = f"\n\n### Long-Term Memory Context\n{ltm_context}\n" if ltm_context else ""

    prompt = (
        f"URL: {url or 'N/A'}\n"
        f"Title: {state.get('title', 'Learning Note')}\n"
        f"Language: {language}\n"
        f"Analysis: {analysis[:10000]}"
        f"{ltm_part}"
    )

    resp, meta = llm_call(system, [{"role": "user", "content": prompt}],
                          max_tokens=4096, temperature=0.5)

    # Отделяем вопросы от основной заметки
    note = resp
    questions_section = ""

    # Ищем разделитель
    for sep in ["\n---\n", "\n---\r\n", "---"]:
        if sep in note:
            parts = note.split(sep, 1)
            note = parts[0].strip()
            questions_section = parts[1].strip()
            break

    # Если вопросы не отделены — ищем секцию вопросов
    if not questions_section:
        patterns = [
            r"(?s)(Вопросы?\s*(?:для\s*проверки|для\s*закрепления).*)",
            r"(?s)(Questions?\s*(?:for\s*review|for\s*check).*)",
        ]
        for pat in patterns:
            match = re.search(pat, note)
            if match:
                questions_section = match.group(1)
                note = note[: match.start()].strip()
                break

    return {
        "note": note,
        "questions_list": questions_section or state.get("questions_list", ""),
        "stage": "done",
    }
