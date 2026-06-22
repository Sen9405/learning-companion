# 🧭 Мой путь AI Agent Engineer — 2026

*Персонализировано с https://github.com/codejunkie99/agent-roadmap-2026 — 15 мая 2026*

---

## 👤 Профиль

**Уровень:** Начинающий (vibe coding Telegram ботов, простые MVP)
**Время:** ~15 часов/неделю
**Стек:** Python + DeepSeek / open-weights модели
**Цель:** Найти работу + запустить продукт
**Общая длительность:** ~26 недель

> Расчёт: 17 недель (канон) × 1.5 (15 ч/нед) ≈ 25.5 → округлил до 26

---

## 📅 План по фазам

### 🟢 Phase 0: Foundations — 3 недели

> NORMAL — ты начинаешь, поэтому фундамент без спешки.

**Что добавил для тебя:**
- **Перед стартом** — [HuggingFace LLM Course](https://huggingface.co/learn/llm-course) (токенизация, трансформеры — база)
- [Open Models have crossed a threshold](https://blog.langchain.com/open-models-have-crossed-a-threshold/) — open-source догоняет закрытые модели

**Основные материалы:**
- [Building Effective Agents (Anthropic)](https://anthropic.com/research/building-effective-agents) — 5 паттернов, workflow vs agent
- [Effective context engineering (Anthropic)](https://anthropic.com/engineering/effective-context-engineering-for-ai-agents) — прочитать дважды
- [Context Engineering for Agents (LangChain)](https://blog.langchain.com/context-engineering-for-agents/) — Write, Select, Compress, Isolate
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) — код к каждому паттерну

**🎯 Проект:** Написать 2-страничный документ своими словами: workflow vs agent, augmented LLM, 4 контекстных примитива, orchestrator-worker, что такое harness.

✅ **Чекпоинт:** Можешь объяснить без терминов фреймворков, что такое агент.

---

### 🟡 Phase 1: Первый простой агент — 4 недели

> NORMAL + время на привыкание к API.

**Что важно для тебя:**
- Используешь DeepSeek API (OpenAI-совместимый — tool use работает)
- Принципы те же: agent loop, tool_use, парсинг ответа модели

**Материалы:**
- [Tutorial: Build a tool-using agent (Anthropic docs)](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — концепции универсальны
- [Building agents with the Claude Agent SDK](https://anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- DeepSeek API docs — как устроен tool use

**🎯 Проект:** Написать tool-using агента дважды:
1. «Сырой» цикл ~100 строк: модель → парсинг tool_use → выполнение → результат
2. С использованием LiteLLM или готового SDK

✅ **Чекпоинт:** Агент, который может ходить в интернет/файлы и возвращать ответ.

---

### 🟡 Phase 2: Deep Agent (исследователь) — 5 недель

> NORMAL. Самая объёмная фаза.

**Адаптация под open-weights:**
- LangGraph 1.0 работает с любыми моделями — используем с DeepSeek
- Вместо LangSmith → [Phoenix (Arize)](https://phoenix.arize.com) — open-source, бесплатно
- Deep Agents middleware от LangChain — тоже работает с любыми моделями

**Материалы:**
- [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [LangChain Academy: Intro to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph) — бесплатно
- [Deep Agents by LangChain](https://github.com/langchain-ai/deepagents) — reference open-source harness
- [Multi-agent research system (Anthropic)](https://anthropic.com/engineering/multi-agent-research-system) — orchestrator-worker

**🎯 Проект:** «Исследователь-аналитик»
- Агент получает вопрос → пишет план → запускает 3 саб-агента поиска → отчёт
- PostgresSaver (persistence) + human-in-the-loop на дорогих операциях
- Phoenix trace для одного полного прогона

✅ **Чекпоинт:** Работающий deep agent с durability и sub-agents.

---

### 🟠 Phase 3: Строим свой harness — 4 недели

> SPEEDRUN — для трудоустройства достаточно понимания архитектуры.
> Для продукта — Deep Agents как база + модификации.

**Что меняем:**
- Не пишешь 1500 строк с нуля — форкаешь Deep Agents и добавляешь свой модуль
- Фокус на понимании, а не на копировании

**Материалы:**
- [The Anatomy of an Agent Harness (LangChain)](https://blog.langchain.com/the-anatomy-of-an-agent-harness/)
- [Improving Deep Agents with harness engineering](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)
- [deepagents source](https://github.com/langchain-ai/deepagents) — читать исходники

**🎯 Проект:** Форкнуть Deep Agents + свой модуль (система промптов, хук pre_tool/post_tool) + post-mortem на 500+ слов.

✅ **Чекпоинт:** Post-mortem с сравнением форка и Claude Agent SDK.

---

### 🔴 Phase 4: Eval и CI — 5 недель

> NORMAL + утяжелено (для поиска работы и продукта это ключ).
> Quality — #1 барьер в индустрии. За evals платят деньги.

**Материалы:**
- [Demystifying evals for AI agents (Anthropic)](https://anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [Agent Evaluation Readiness Checklist (LangChain)](https://blog.langchain.com/agent-evaluation-readiness-checklist/)
- [Evaluating Deep Agents: Our Learnings](https://blog.langchain.com/evaluating-deep-agents-our-learnings/)
- [Inspect AI (UK AISI)](https://inspect.aisi.org.uk) — benchmark-grade evals

**🎯 Проект:**
- Golden dataset 30–50 вопросов, 3 уровня сложности
- 4 типа eval'ов: single-turn, trajectory, LLM-as-judge, end-state
- CI-гейт в GitHub Actions
- Запуск benchmark через Inspect (GAIA Level 1)
- **Бонус:** опубликовать проект публично (GitHub + статья) — для портфолио

✅ **Чекпоинт:** `make eval` выдаёт CI pass/fail + Inspect лог.

---

### 🔴 Phase 5: Production Hardening — бессрочно

> DEEP — для продукта самая важная фаза.

**Адаптация под open-weights:**
- Весь Phase 5 + отдельный фокус на cost-discipline
- Sandboxing: E2B (бесплатный tier) или Modal
- Credential broker — ключи API не попадают в контекст

**🎯 Deliverables:**
- Prompt caching + routing по сложности
- Model-routing с cost-per-task бюджетами
- Sandbox для кода
- Durable execution (Inngest / Temporal / LangGraph PostgresSaver)
- Trace sampling + drift alerts

✅ **Чекпоинт:** Агент, который переживает контакт с реальными пользователями.

---

## 📚 Ресурсы для твоего профиля

### Для начала (Phase 0)
| Ресурс | Зачем |
|--------|-------|
| [HuggingFace LLM Course](https://huggingface.co/learn/llm-course) | База: токенизация, трансформеры |
| [Building Effective Agents](https://anthropic.com/research/building-effective-agents) | Прочитать первым |
| [Effective Context Engineering](https://anthropic.com/engineering/effective-context-engineering-for-ai-agents) | Самый важный текст |
| [Open Models Have Crossed a Threshold](https://blog.langchain.com/open-models-have-crossed-a-threshold/) | Для open-weights пользователей |
| [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) | Код к паттернам |

### Бесплатные модели
- **DeepSeek** (Deep Think / Chat) — лучший balance quality/cost
- **GLM-5** / **MiniMax M2.7** — догоняют закрытые frontier
- **LiteLLM** — прослойка для переключения провайдеров

### На будущее
- **YouTube:** Andrej Karpathy, AI Engineer канал
- **Блоги:** Anthropic Engineering Blog, LangChain Blog, Hamel Husain
- **Рассылка:** Latent Space
- **Комьюнити:** LangChain Discord, HuggingFace Discord

---

## 📦 Чеклист прогресса

- [x] **Phase 0 (нед 1–3):** Прочитаны 4 статьи + написан итоговый документ
- [ ] **Phase 1 (нед 4–7):** Tool-using агент (сырой + на SDK)
- [ ] **Phase 2 (нед 8–12):** Deep Agent «исследователь» + Phoenix trace
- [x] **Phase 3 (нед 13–16):** Структура + пакет + тесты + CI + eval gate ✅
- [ ] **Phase 4 (нед 17–21):** Golden dataset + CI eval gate + Inspect benchmark
- [ ] **Phase 5 (нед 22+):** Production hardening

---

## 🎯 Следующий шаг (прямо сегодня)

👉 **Прочитать** [Building Effective Agents by Anthropic](https://anthropic.com/research/building-effective-agents) — 15 минут чтения, с которых начинается весь путь.

После прочтения — напиши в Telegram «прочитал», и я скажу, что делать дальше.
