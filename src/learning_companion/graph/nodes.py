"""Graph nodes — planner, fetcher, analyst, writer."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from learning_companion.graph import LearningState
from learning_companion.llm import llm_call
from learning_companion.memory import get_ltm
from learning_companion.security import RunContext, run_sandboxed, wrap_untrusted_document

_VENV_PYTHON = str(Path(__file__).resolve().parent.parent.parent.parent / ".venv" / "bin" / "python")


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

1. **Ключевые концепты** (5-10 терминов с определениями)
2. **Связи между концептами** (как они связаны)
3. **Практические выводы** (как применить)
4. **Вопросы для проверки знаний** (7-10 вопросов с ответами)
5. **Глоссарий** (ключевые термины с краткими определениями)

ВАЖНО: поле "analysis" должно быть ПОДРОБНЫМ (2000-5000 символов).
ВАЖНО: все вопросы и ответы ДОЛЖНЫ быть на том же языке, что и контент.

Ответь в формате JSON:
{{
  "analysis": "подробный анализ текстом на 2000-5000 символов",
  "concepts": [
    {{"term": "название", "definition": "определение"}}
  ],
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
7. Вопросы для проверки знаний (5-7 вопросов для самопроверки)

Формат: Markdown, язык: {language}.
ВАЖНО: вся заметка, включая вопросы, ДОЛЖНА быть на языке {language}.
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

    language = state.get("language", "")
    if not language:
        # Для YouTube не можем определить язык по URL — откладываем до контента
        if source_type == "youtube":
            language = ""
        else:
            language = _detect_language(text or url)

    # Пробуем LLM для уточнения (но не перезаписываем title, если он уже есть)
    try:
        system = PLANNER_SYSTEM + f"\nLanguage: {language}"
        prompt = f"Source: {url or 'direct text'}\nText preview: {text[:500] if text else 'N/A'}"
        resp, meta = llm_call(system, [{"role": "user", "content": prompt}], stage="planner", run_id=state.get("run_id", "default"))
        plan = json.loads(resp)
        source_type = plan.get("source_type", source_type)
        # title берём из переданного state, и только если пуст — из LLM
        title = state.get("title", "") or plan.get("title_hint", "") or "Learning Note"
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
    ctx = RunContext.create(state.get("run_id", "default"))

    youtube_title = ""
    if source_type == "web" and url:
        content = _fetch_web(url)
    elif source_type == "pdf" and url:
        content = _fetch_pdf(url, ctx=ctx)
    elif source_type == "youtube" and url:
        content = _fetch_youtube(url, ctx=ctx)
        youtube_title = _fetch_youtube_title(url, ctx=ctx)
    elif text:
        content = text

    # Если контент — сообщение об ошибке (начинается с [), сбрасываем
    if content and content.startswith("["):
        content = ""

    if source_type in {"web", "pdf", "youtube"} and content and not content.startswith("["):
        content = wrap_untrusted_document(content, source=source_type)

    title = state.get("title", "")
    # Для YouTube всегда перезаписываем заголовок из yt-dlp
    if source_type == "youtube" and youtube_title:
        title = youtube_title
    elif not title:
        if content:
            # Извлекаем первую строку как заголовок
            lines = content.strip().split("\n")
            if lines:
                title = lines[0][:100]

    return {
        "content": content,
        "title": title,
        "stage": "analyst",
    }


def _fetch_youtube(url: str, *, ctx: RunContext | None = None) -> str:
    """Извлекает транскрипт YouTube видео — сначала субтитры, затем ASR-запасной вариант."""
    # Попытка 1: субтитры
    try:
        result = _exec_yt_dlp(url, ctx=ctx)
        if result and not result.startswith("["):
            return result
    except Exception:
        pass
    # Попытка 2: ASR (Whisper) — если субтитров нет
    try:
        result = _transcribe_youtube_audio(url, ctx=ctx)
        if result and not result.startswith("["):
            return result
    except Exception:
        pass
    return f"[YouTube transcript fetch failed for {url}]"


def _transcribe_youtube_audio(url: str, *, ctx: RunContext | None = None) -> str:
    """Скачивает аудиодорожку YouTube видео и распознаёт речь через Whisper.

    Используется как fallback, когда субтитры отсутствуют.
    """
    ctx = ctx or RunContext.create("default")
    audio_dir = ctx.safe_path("tmp", "youtube_audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_template = str(audio_dir / "audio")

    try:
        # Скачиваем аудио в opus (наименьший размер, хорошее качество)
        result = run_sandboxed(
            [_VENV_PYTHON, "-m", "yt_dlp", "-f", "bestaudio[ext=webm]/bestaudio",
             "--extract-audio", "--audio-format", "opus",
             "-o", audio_template, url],
            ctx,
            cwd=ctx.work_dir,
            timeout=300,
        )
        if result.returncode != 0 and not result.stdout:
            return f"[Audio download failed: {result.stderr[:200]}]"

        # Ищем скачанный аудиофайл
        import glob
        audio_files = sorted(glob.glob(str(audio_dir / "*")))
        if not audio_files:
            return "[No audio file downloaded]"

        audio_path = audio_files[0]
        import os
        if os.path.getsize(audio_path) < 1024:
            return "[Downloaded audio too small]"

        # Транскрибируем через Whisper
        try:
            import whisper
        except ImportError:
            return "[Whisper not available — install openai-whisper]"

        model = whisper.load_model("base")
        result = model.transcribe(
            audio_path,
            language=None,  # автоопределение
            task="transcribe",
            verbose=False,
        )
        text = (result.get("text") or "").strip()
        if len(text) < 20:
            return "[Transcription too short or empty]"
        return text[:150000]

    except Exception as e:
        return f"[YouTube audio transcription error: {e}]"
    finally:
        # Чистим временные файлы
        import shutil
        try:
            shutil.rmtree(audio_dir)
        except Exception:
            pass


def _fetch_youtube_title(url: str, *, ctx: RunContext | None = None) -> str:
    """Извлекает только заголовок YouTube видео."""
    try:
        ctx = ctx or RunContext.create("default")
        result = run_sandboxed(
            [_VENV_PYTHON, "-m", "yt_dlp", "--skip-download", "--print", "title", url],
            ctx,
            cwd=ctx.work_dir,
            timeout=30,
        )
        title = result.stdout.strip()
        if title and not title.startswith("WARNING"):
            return title
    except Exception:
        pass
    return ""


def _exec_yt_dlp(url: str, *, ctx: RunContext | None = None) -> str:
    """Выполняет yt-dlp для извлечения субтитров.

    Возвращает только текст транскрипта (без заголовка).
    """
    import re
    import subprocess

    ctx = ctx or RunContext.create("default")
    tmpdir = ctx.safe_path("tmp", "youtube_subs")
    tmpdir.mkdir(parents=True, exist_ok=True)
    outtemplate = str(tmpdir / "subs")

    try:
        # Скачиваем субтитры в .vtt файл
        result = run_sandboxed(
            [_VENV_PYTHON, "-m", "yt_dlp", "--write-auto-subs", "--sub-langs", "ru,en",
             "--skip-download", "--convert-subs", "srt",
             "-o", outtemplate, url],
            ctx,
            cwd=ctx.work_dir,
            timeout=120,
        )

        if result.returncode != 0:
            # yt-dlp может выдавать warning (старая версия) но всё равно скачать файл
            # Проверяем по наличию файлов, а не по returncode
            pass

        # Ищем скачанный файл субтитров
        vtt_files = sorted([
            f for f in os.listdir(tmpdir)
            if f.endswith(".srt") or f.endswith(".vtt")
        ])

        if not vtt_files:
            return "[No subtitle files found]"

        # Берём первый (русский если есть, иначе английский)
        sub_path = ctx.safe_path("tmp", "youtube_subs", vtt_files[0])

        with open(sub_path, encoding="utf-8") as f:
            raw = f.read()

        # Очищаем от временных меток и HTML-тегов VTT/SRT
        # Убираем строки с таймкодами
        lines = []
        for line in raw.split("\n"):
            line = line.strip()
            # Пропускаем пустые, таймкоды, номера строк, WEBVTT заголовок
            if not line:
                continue
            if "-->" in line:
                continue
            if re.match(r"^\d+$", line):
                continue
            if line.startswith("WEBVTT"):
                continue
            if line.startswith("Kind:") or line.startswith("Language:"):
                continue
            # Убираем HTML-теги вроде <c> </c>
            line = re.sub(r"<[^>]+>", "", line)
            lines.append(line)

        text = "\n".join(lines)
        if len(text) < 20:
            return "[Subtitles too short or empty]"

        return text[:150000]

    except subprocess.TimeoutExpired:
        return "[YouTube fetch timed out]"
    except Exception as e:
        return f"[YouTube fetch error: {e}]"
    finally:
        # Чистим временные файлы
        import shutil
        try:
            shutil.rmtree(tmpdir)
        except Exception:
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
        return text[:150000]
    except Exception as e:
        return f"[Web fetch error: {e}]"


MAX_FETCH_CHARS = 150000


def _fetch_pdf(url: str, *, ctx: RunContext | None = None) -> str:
    """Извлекает текст из PDF."""
    try:
        import httpx
        ctx = ctx or RunContext.create("default")
        resp = httpx.get(url, timeout=60, follow_redirects=True)
        resp.raise_for_status()

        pdf_path = ctx.safe_path("input", "source.pdf")
        with open(pdf_path, "wb") as f:
            f.write(resp.content)

        try:
            # Попытка через PyMuPDF
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text[:MAX_FETCH_CHARS]
        except ImportError:
            pass

        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            return text[:MAX_FETCH_CHARS]
        except ImportError:
            pass

        try:
            from pdfminer.high_level import extract_text
            text = extract_text(pdf_path)
            return text[:MAX_FETCH_CHARS]
        except ImportError:
            pass

        return f"[PDF downloaded to {pdf_path}, but no PDF parser available]"

    except Exception as e:
        return f"[PDF fetch error: {e}]"


# Константы для chunking
CHUNK_SIZE = 10000       # размер одного чанка в символах
CHUNK_OVERLAP = 500      # перекрытие между чанками
CHUNK_SUMMARIZE_SYSTEM = """Ты — ассистент-анализатор фрагмента. Проанализируй этот фрагмент текста и выдели:

1. Основные темы и идеи (2-5 пунктов)
2. Ключевые термины с определениями
3. Практические выводы

Ответь кратко (300-500 символов), только факты по существу."""

CHUNK_MERGE_SYSTEM = """Ты — ассистент-синтезатор. У тебя есть анализ нескольких фрагментов одного материала.
Объедини их в целостный анализ по следующей структуре:

1. **Ключевые концепты** (5-10 терминов с определениями)
2. **Связи между концептами** (как они связаны)
3. **Практические выводы** (как применить)
4. **Вопросы для проверки знаний** (7-10 вопросов с ответами)
5. **Глоссарий** (ключевые термины с краткими определениями)

ВАЖНО: поле "analysis" должно быть ПОДРОБНЫМ (2000-5000 символов).
ВАЖНО: все вопросы и ответы ДОЛЖНЫ быть на том же языке, что и контент.

Ответь в формате JSON:
{{
  "analysis": "подробный анализ текстом на 2000-5000 символов",
  "concepts": [
    {{"term": "название", "definition": "определение"}}
  ],
  "questions": [
    {{"q": "вопрос?", "a": "ответ"}}
  ],
  "glossary": [
    {{"term": "термин", "definition": "определение"}}
  ]
}}"""


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Разбивает текст на перекрывающиеся чанки по границам предложений."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Ищем границу предложения (точка + пробел + заглавная)
            for sep in [". ", "! ", "? ", ".\n", "\n\n"]:
                boundary = text.rfind(sep, start + chunk_size // 2, end)
                if boundary != -1:
                    end = boundary + len(sep)
                    break
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks


def analyst_node(state: LearningState) -> dict[str, Any]:
    """Анализирует контент (с поддержкой chunking для длинных текстов)."""
    content = state.get("content", "")
    language = state.get("language", "")
    
    # Если контент пустой — возвращаем ошибку
    if not content:
        error_msg = "❌ Не удалось извлечь содержимое для анализа. Для YouTube-видео необходимы субтитры."
        return {
            "analysis": error_msg,
            "concepts": [],
            "questions": [],
            "glossary": [],
            "questions_list": "",
            "stage": "writer",
            "error": error_msg,
        }
    
    # Определяем язык по контенту (не по URL — для YouTube)
    if not language and content:
        language = _detect_language(content)
    language = language or "ru"

    # Если контент короткий — прямой анализ (старое поведение)
    if len(content) <= CHUNK_SIZE:
        system = ANALYST_SYSTEM
        prompt = f"Language: {language}\n\nContent:\n{content}"
        resp, meta = llm_call(system, [{"role": "user", "content": prompt}],
                               response_model=None, max_tokens=8192,
                               stage="analyst.direct", run_id=state.get("run_id", "default"))
        data = _parse_analyst_response(resp)
    else:
        # Длинный контент — разбиваем на чанки, анализируем каждый, собираем
        chunks = _chunk_text(content)
        partials = []
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            prompt = f"Language: {language}\n\nFragment {i+1}/{total_chunks}:\n{chunk}"
            resp, _ = llm_call(CHUNK_SUMMARIZE_SYSTEM, [{"role": "user", "content": prompt}],
                                response_model=None, max_tokens=2048,
                                stage="analyst.chunk_summary", run_id=state.get("run_id", "default"))
            partials.append(f"--- Fragment {i+1}/{total_chunks} ---\n{resp}")

        merge_prompt = f"Language: {language}\n\nPartial analyses:\n\n{chr(10).join(partials)}"
        resp, meta = llm_call(CHUNK_MERGE_SYSTEM, [{"role": "user", "content": merge_prompt}],
                               response_model=None, max_tokens=8192,
                               stage="analyst.merge", run_id=state.get("run_id", "default"))
        data = _parse_analyst_response(resp)

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

    # Создаём вопросы для проверки (с fallback, если не сгенерировались)
    if not questions and analysis:
        questions = _generate_questions(analysis, language)
    questions_text = _format_questions(questions, language)

    return {
        "analysis": analysis,
        "concepts": concepts,
        "questions": questions,
        "glossary": glossary,
        "questions_list": questions_text,
        "stage": "writer",
    }


def _parse_analyst_response(resp: str) -> dict:
    """Парсит JSON из ответа analyst, с fallback на raw text."""
    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", resp, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"analysis": resp, "concepts": [], "questions": [], "glossary": []}


def _format_questions(questions: list[dict], language: str) -> str:
    """Форматирует вопросы в Markdown (только вопросы, без ответов).

    Ответы остаются в questions (raw list) для LTM, но в questions_list
    их нет — пользователь получает только вопросы для самопроверки.
    """
    if not questions:
        return ""
    lines = ["\n---\n### Вопросы для проверки знаний\n"]
    for i, q in enumerate(questions, 1):
        question = q.get("q", q.get("question", f"Вопрос {i}"))
        lines.append(f"{i}. {question}")
    return "\n".join(lines)


def _generate_questions(analysis: str, language: str) -> list[dict]:
    """Fallback: генерирует вопросы по анализу, если основная модель не вернула."""
    try:
        lang_hint = "Russian" if language == "ru" else "English"
        system = (
            f"Ты — ассистент генерации вопросов для проверки знаний. "
            f"На основе анализа сгенерируй 5 вопросов с ответами. "
            f"Все вопросы и ответы ДОЛЖНЫ быть на {lang_hint} языке.\n"
            "Ответь строго в JSON: [{\"q\": \"вопрос?\", \"a\": \"ответ\"}]"
        )
        prompt = f"Analysis (language={language}):\n{analysis[:3000]}"
        resp, meta = llm_call(system, [{"role": "user", "content": prompt}],
                              response_model=None, max_tokens=2048, temperature=0.7,
                              stage="questions.fallback")
        data = json.loads(resp)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


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
                          max_tokens=4096, temperature=0.5,
                          stage="writer", run_id=state.get("run_id", "default"))

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
        "questions_list": state.get("questions_list", ""),
        "stage": "done",
    }
