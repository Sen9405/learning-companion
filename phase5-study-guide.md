# Phase 5 Study Guide — Production Hardening для AI-агентов

Цель этого материала — объяснить термины Phase 5 простым языком и дать понятный порядок изучения перед тем, как мы начнём писать код.

Phase 5 — это переход от “агент работает у меня локально” к “агентом можно пользоваться регулярно, он не сжигает бюджет, не теряет состояние, не светит секреты и его поведение можно объяснить по логам”.

---

## 1. Главная идея Phase 5

В предыдущих фазах мы учились строить агента:

- получать материал;
- анализировать;
- писать заметку;
- сохранять в память;
- проверять качество через eval;
- запускать CI.

Теперь вопрос другой:

> Что будет, если этим агентом пользоваться каждый день, на реальных материалах, с реальными деньгами, API-ключами, ошибками сети и длинными задачами?

Production hardening — это набор инженерных практик, которые делают агента:

1. **Дешёвым** — не делает лишние LLM-вызовы, использует cache, считает стоимость.
2. **Наблюдаемым** — видно, что произошло внутри запуска.
3. **Безопасным** — секреты не попадают в контекст, опасные действия изолированы.
4. **Устойчивым** — если процесс умер, задачу можно восстановить.
5. **Управляемым** — есть лимиты, бюджеты, алерты, отчёты.

---

## 2. Мини-карта терминов

| Термин | Простое объяснение |
|---|---|
| Prompt caching | Кэширование ответа модели на одинаковый промпт, чтобы не платить повторно |
| Cost ledger | Журнал расходов: какой вызов модели сколько стоил |
| Budget limit | Жёсткий лимит стоимости/токенов на один запуск |
| Observability | Видимость внутренней работы агента: логи, метрики, трейсы |
| Trace | Полная цепочка выполнения одного запуска |
| Span | Один шаг внутри trace: planner, fetcher, analyst, writer |
| Sandbox | Изолированная среда, где агент может выполнять рискованные действия |
| Credential broker | Прослойка, которая выдаёт доступ к секретам без передачи самих ключей в модель |
| Durable execution | Возможность продолжить задачу после сбоя процесса |
| Model routing | Выбор модели по сложности/цене задачи |
| Drift | Деградация качества или изменение поведения со временем |
| Alert | Уведомление, когда что-то пошло не так |

---

## 3. Prompt caching

### Что это

Prompt caching — это идея: если мы уже отправляли модели тот же самый запрос и получили ответ, можно сохранить результат и при повторном запуске взять ответ из cache.

Пример:

```text
System prompt: "Ты — судья качества учебных заметок..."
User prompt: "Оцени заметку X по материалу Y"
Model: deepseek-v4-flash
Temperature: 0
```

Если всё это повторилось один в один, второй раз модель можно не вызывать.

### Почему это важно

LLM-вызовы стоят денег и времени. В eval/benchmark мы часто повторяем одинаковые примеры. Без кэша каждый прогон снова платный. С кэшем повторный прогон может быть почти бесплатным.

### Где cache особенно полезен

1. **Eval** — golden dataset не меняется, judge prompt часто одинаковый.
2. **Inspect benchmark** — повторные прогоны одних и тех же samples.
3. **Chunk summaries** — если источник тот же, чанки те же.
4. **Planner** — одинаковый URL/текст даёт одинаковый план.
5. **Question generation** — вопросы по одной и той же заметке можно не генерировать заново.

### Что должно входить в cache key

Cache key — это отпечаток запроса. Он должен учитывать всё, что влияет на ответ:

- model;
- system prompt;
- user/assistant messages;
- max_tokens;
- temperature;
- response_format;
- версию промпта, если мы её добавим.

Пример логики:

```python
cache_key = sha256(json.dumps({
    "model": model,
    "system": system,
    "messages": messages,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "response_format": response_format,
}, sort_keys=True).encode()).hexdigest()
```

### Что нельзя кэшировать бездумно

Кэш опасен, если вход выглядит одинаково, но смысл меняется из-за скрытого контекста.

Например:

- запрос зависит от текущей даты;
- запрос зависит от внешнего состояния сайта;
- промпт просит “используй последние новости”;
- ответ должен быть случайным/творческим;
- temperature высокая.

Поэтому для начала лучше кэшировать только deterministic-вызовы:

```text
temperature == 0
```

А потом расширять.

### В нашем проекте

У нас есть единая точка LLM-вызовов:

```text
src/learning_companion/llm.py
```

Значит prompt caching логично встроить именно туда:

```python
llm_call(..., cache=True, stage="eval.judge", run_id=run_id)
```

---

## 4. Cost ledger

### Что это

Cost ledger — это журнал всех LLM-вызовов.

Не просто “весь запуск стоил $0.003”, а детально:

| run_id | stage | model | prompt_tokens | completion_tokens | cost | latency | cache_hit |
|---|---|---|---:|---:|---:|---:|---:|
| abc123 | planner | deepseek-v4-flash | 500 | 120 | 0.0001 | 1.2s | false |
| abc123 | analyst.chunk_summary | deepseek-v4-flash | 9000 | 600 | 0.0014 | 5.8s | false |
| abc123 | writer | deepseek-v4-flash | 4000 | 2500 | 0.0013 | 8.1s | true |

### Зачем это нужно

Без ledger мы не знаем:

- какая стадия самая дорогая;
- какой материал внезапно стал дорогим;
- сколько экономит prompt cache;
- где растёт latency;
- были ли ошибки модели;
- какой run нужно оптимизировать.

### Почему обычного print недостаточно

Print исчезает после запуска. Ledger сохраняется в SQLite, значит можно построить отчёт:

```bash
learning-companion report --limit 20
```

И увидеть:

```text
Total calls: 84
Total cost: $0.0182
Cache hit rate: 37%
Most expensive stage: analyst.chunk_summary
Recent errors: 0
```

### В нашем проекте

Сейчас `llm.py` уже считает стоимость глобально в памяти процесса:

```python
_total_prompt_tokens
_total_completion_tokens
_total_cost
```

Но после завершения процесса история пропадает. Поэтому добавляем SQLite ledger.

---

## 5. Budget limits

### Что это

Budget limit — это ограничение, после которого агент обязан остановиться.

Примеры:

```text
max_cost_per_run = $0.05
max_prompt_tokens_per_run = 200_000
max_completion_tokens_per_run = 50_000
max_llm_calls_per_run = 30
```

### Почему это важно

Агент может случайно уйти в дорогой цикл:

- слишком длинный YouTube transcript;
- слишком много чанков;
- ошибка парсинга → повторный fallback;
- eval запускается не на smoke dataset, а на полном;
- модель возвращает мусор → retry → retry → retry.

Budget limit делает систему fail-safe:

> Лучше остановиться с понятной ошибкой, чем молча потратить деньги.

### Как это выглядит для пользователя

Вместо бесконечного запуска:

```text
BudgetExceededError: run abc123 exceeded max cost $0.05.
Current cost: $0.052.
Most expensive stage: analyst.chunk_summary.
```

### В нашем проекте

Budget check нужно делать в `llm_call()`:

1. Перед вызовом — проверить текущий расход.
2. После вызова — обновить счётчики.
3. Если лимит превышен — остановить run понятной ошибкой.

---

## 6. Observability

### Что это

Observability — это способность ответить на вопрос:

> Что именно произошло внутри системы?

Для обычного кода достаточно логов. Для агента нужны ещё:

- LLM-вызовы;
- tool calls;
- токены;
- latency;
- стоимость;
- ошибки парсинга;
- cache hits/misses;
- переходы графа;
- итоговое состояние.

### Три уровня observability

#### 1. Logs

События текстом:

```text
[planner] started
[planner] completed in 1.2s
[analyst] chunk 3/8 failed JSON parse
```

#### 2. Metrics

Числа:

```text
llm_calls_total = 12
run_cost_usd = 0.0042
cache_hit_rate = 0.31
avg_latency_seconds = 3.8
```

#### 3. Traces

Дерево выполнения одного run:

```text
run abc123
├── planner span
├── fetcher span
├── analyst span
│   ├── chunk 1 span
│   ├── chunk 2 span
│   └── merge span
└── writer span
```

### Что такое trace и span

**Trace** — вся история одного запуска.

**Span** — отдельный шаг внутри запуска.

Например:

```text
Trace: run_id=abc123
Span 1: planner
Span 2: fetcher
Span 3: analyst.chunk_summary
Span 4: analyst.merge
Span 5: writer
```

### Phoenix / OpenTelemetry

OpenTelemetry — стандарт для описания traces/metrics/logs.

Phoenix от Arize — open-source UI, где можно смотреть LLM traces.

У нас уже есть файл:

```text
src/learning_companion/graph/tracing.py
```

Но пока это только базовый setup. В Phase 5 надо добавить более полезные span attributes:

- stage;
- model;
- prompt_tokens;
- completion_tokens;
- cost;
- cache_hit;
- error;
- run_id.

---

## 7. Sandbox

### Что это

Sandbox — изолированная среда, где агент может выполнять потенциально опасные действия.

Например, агенту нужно запустить код. Если он запускает его прямо на основной машине, код может:

- удалить файлы;
- прочитать секреты;
- открыть сеть;
- повесить процесс;
- использовать слишком много CPU/RAM;
- записать мусор в проект.

Sandbox ограничивает ущерб.

### Типы sandbox

| Тип | Пример | Плюсы | Минусы |
|---|---|---|---|
| Локальная временная папка | `/tmp/run-abc123` | просто | слабая изоляция |
| Docker container | контейнер на run | хорошая изоляция | нужно настраивать Docker |
| Отдельный worker | отдельный процесс/пользователь | гибко | сложнее инфраструктура |
| Cloud sandbox | E2B / Modal | мощно | внешний сервис, стоимость |

### Для нашего проекта

Learning Companion сейчас в основном:

- читает URL;
- вызывает yt-dlp;
- парсит PDF/web;
- пишет в Obsidian;
- отправляет Telegram;
- вызывает LLM.

Он не является полноценным coding-agent, поэтому sandbox — не первый шаг. Но он всё равно нужен для:

- `yt-dlp`;
- PDF parsing;
- будущих tool calls;
- обработки непроверенных файлов.

Поэтому sandbox логично делать во втором спринте, после ledger/cache/budget.

---

## 8. Secrets и credential broker

### Что такое secret

Secret — это любой ключ или токен:

- `DEEPSEEK_API_KEY`;
- Telegram bot token;
- GitHub token;
- database password;
- OAuth token.

### Проблема

Если секрет попал в prompt или лог, он может:

- сохраниться в истории;
- попасть в trace;
- попасть в LLM provider;
- быть случайно отправлен пользователю;
- попасть в GitHub.

### Плохой подход

```python
prompt = f"Use this API key: {DEEPSEEK_API_KEY}"
```

Или:

```python
print(os.environ)
```

### Хороший подход

Модель не должна видеть ключ. Она должна сказать:

```text
Нужно вызвать Telegram sendMessage.
```

А инструмент сам берёт токен из защищённого места и выполняет действие.

### Credential broker

Credential broker — это прослойка между агентом и секретами.

Агент говорит:

```text
send telegram message to chat X
```

Broker:

1. проверяет, можно ли это действие;
2. берёт токен;
3. вызывает API;
4. возвращает только безопасный результат.

Модель не получает сам токен.

### Для нашего проекта

Сейчас ключи читаются из `.env`. Это нормально для локального проекта, но production-hardening требует:

- не печатать ключи;
- не сохранять их в ledger/cache;
- не отправлять их в LLM;
- централизовать чтение настроек;
- добавить redaction для логов.

---

## 9. Durable execution

### Что это

Durable execution — это способность продолжить задачу после сбоя.

Пример:

1. Агент скачал transcript.
2. Проанализировал 5 из 10 чанков.
3. Процесс умер.
4. После restart агент продолжает с 6-го чанка, а не начинает заново.

### Почему это важно

Agent runs могут быть долгими:

- длинные видео;
- большие PDF;
- много шагов;
- HITL approval;
- сетевые ошибки.

Если всё хранится только в памяти процесса, любой сбой всё теряет.

### Как это связано с LangGraph

LangGraph поддерживает checkpointing. У нас уже есть PostgresSaver:

```text
PostgresSaver + thread_id/run_id
```

Это уже частичная durability.

Но production-level durability шире:

- очередь задач;
- статусы run’ов;
- retry policy;
- idempotency;
- resume после crash;
- отчёт о failed runs.

### Для нашего проекта

В Sprint 1 мы не строим Temporal/Inngest. Но ledger — первый шаг: он даст историю вызовов и ошибок. Потом можно будет сделать run table:

```text
runs(id, status, started_at, finished_at, cost, error)
```

---

## 10. Model routing

### Что это

Model routing — выбор модели под задачу.

Например:

| Задача | Модель |
|---|---|
| простой planner | дешёвая быстрая модель |
| сложный analysis | более сильная модель |
| eval judge | deterministic модель с temperature=0 |
| extraction | дешёвая модель или вообще без LLM |

### Важно для нас

Ты явно предпочитаешь **DeepSeek V4 Flash** и не хочешь дорогую pro-модель. Поэтому в нашем случае routing не значит “давай всё гонять на дорогой модели”.

Для нас routing скорее такой:

- можно ли вообще не вызывать LLM;
- можно ли взять из prompt cache;
- можно ли использовать shorter prompt;
- можно ли chunking сделать меньше;
- можно ли planner пропустить, если source_type очевиден;
- можно ли использовать deterministic extraction вместо LLM.

То есть routing по стоимости — это не только выбор модели, но и выбор стратегии.

---

## 11. Drift и alerts

### Что такое drift

Drift — это когда система со временем начинает вести себя иначе или хуже.

Примеры:

- модель стала давать более короткие заметки;
- вопросы снова начали содержать ответы;
- JSON стал чаще ломаться;
- latency выросла;
- стоимость одного run выросла в 3 раза;
- качество eval упало.

### Что такое alert

Alert — автоматическое уведомление:

```text
Средняя стоимость run за сутки выросла на 80%.
```

или:

```text
Eval score упал ниже threshold.
```

### Для нашего проекта

Alerts можно строить на основе:

- eval results;
- ledger;
- cache hit rate;
- failures;
- Telegram delivery errors.

Но alerts — не первый шаг. Сначала нужны данные. Данные даст ledger.

---

## 12. Что почитать

Ниже — короткий список, без перегруза.

### Обязательно

1. **DeepSeek Context Caching**
   - https://api-docs.deepseek.com/guides/kv_cache
   - Зачем: понять, как DeepSeek сам умеет кэшировать context/prefix и почему повторяющиеся промпты могут быть дешевле/быстрее.

2. **LangGraph Persistence**
   - https://docs.langchain.com/oss/python/langgraph/persistence
   - Зачем: понять checkpointing, thread_id, resume, durability.

3. **Arize: LLM Observability for AI Agents and Applications**
   - https://arize.com/blog/llm-observability-for-ai-agents-and-applications/
   - Зачем: понять traces, spans, token/cost/latency monitoring.

4. **AgentGateway budget limits**
   - https://agentgateway.dev/docs/kubernetes/main/llm/budget-limits/
   - Зачем: увидеть, как production-системы задают лимиты расходов.

### Дополнительно

5. **Infisical Agent Vault**
   - https://github.com/Infisical/agent-vault
   - Зачем: пример идеи credential proxy/vault для AI agents.

6. **AI Agent Sandbox Architecture**
   - https://pub.towardsai.net/ai-agent-sandbox-architecture-how-to-let-agents-run-code-without-letting-them-run-everything-63a9293c35fb
   - Зачем: обзор идеи sandbox для выполнения кода агентами.

---

## 13. Как это связано с нашим первым спринтом

Мы не будем сразу читать всё подряд. Нам достаточно понять 4 базовые вещи:

1. **Prompt caching** — как не платить дважды за одинаковое.
2. **Cost ledger** — как видеть, куда ушли деньги.
3. **Budget limits** — как остановить runaway-задачу.
4. **Observability** — как понимать, что произошло внутри run.

После этого код Sprint 1 будет понятен.

---

## 14. Практическая аналогия

Представь Learning Companion как маленькую производственную линию.

Сейчас он умеет делать продукт: учебную заметку.

Но production hardening добавляет приборную панель и предохранители:

| Без Phase 5 | С Phase 5 |
|---|---|
| “Вроде сработало” | “Вот trace каждого шага” |
| “Сколько стоило? примерно...” | “Вот точная стоимость по stage” |
| “Повторный eval снова платный” | “Повторный eval взялся из cache” |
| “Если зациклится — плохо” | “Budget limit остановит запуск” |
| “Секреты просто в env” | “Секреты не попадают в prompt/log/cache” |
| “Если процесс умер — начинай заново” | “Можно восстановиться с checkpoint” |

---

## 15. Что нужно понять перед кодом

Перед реализацией Sprint 1 достаточно ответить на вопросы:

1. Что такое prompt cache key?
2. Почему нельзя кэшировать все LLM-вызовы подряд?
3. Чем cost ledger отличается от обычного print/log?
4. Почему budget limit должен быть hard stop?
5. Что такое trace и span?
6. Почему secrets нельзя передавать модели?
7. Почему routing — это не только выбор модели, но и выбор стратегии?

Если эти вопросы понятны — можно писать код.

---

## 16. Мини-глоссарий

**Prompt** — текстовая инструкция модели.

**System prompt** — главная инструкция, задающая роль и правила поведения модели.

**User message** — конкретный запрос или данные для обработки.

**Token** — единица текста, по которой считается стоимость и лимит контекста.

**Prompt tokens** — токены входа: system prompt + messages + контекст.

**Completion tokens** — токены ответа модели.

**Latency** — сколько времени занял вызов.

**Cache hit** — ответ найден в cache, модель не вызывалась.

**Cache miss** — ответа в cache нет, нужно вызвать модель.

**TTL** — срок жизни записи в cache.

**Ledger** — журнал событий/расходов.

**Run** — один полный запуск агента.

**Stage** — логический этап run: planner, fetcher, analyst, writer.

**Trace** — дерево всех действий одного run.

**Span** — один узел trace, например `analyst.merge`.

**Budget** — лимит денег/токенов/времени.

**Sandbox** — изолированная среда для опасных операций.

**Secret** — API key, токен или пароль.

**Credential broker** — прослойка, которая выполняет действия с секретами, не отдавая секреты модели.

**Durability** — способность пережить сбой и продолжить выполнение.

**Routing** — выбор модели или стратегии под конкретную задачу.

**Drift** — ухудшение или изменение поведения системы со временем.

**Alert** — автоматическое уведомление о проблеме.

---

## 17. Рекомендуемый порядок изучения

### День 1

Прочитать этот файл целиком.

Цель: понять словарь Phase 5.

### День 2

Прочитать:

- DeepSeek Context Caching;
- LangGraph Persistence.

Цель: понять cache и durability.

### День 3

Прочитать:

- Arize LLM Observability;
- AgentGateway budget limits.

Цель: понять observability и cost controls.

### После этого

Переходим к Sprint 1 implementation:

1. `settings.py`
2. `prompt_cache.py`
3. `ledger.py`
4. budget enforcement в `llm.py`
5. `learning-companion report`

---

## 18. Главное резюме

Phase 5 — это не про “добавить ещё фич”.

Phase 5 — это про то, чтобы агент стал похож на production-систему:

- измеряемую;
- ограниченную по расходам;
- устойчивую к сбоям;
- безопасную по секретам;
- понятную в отладке;
- пригодную для регулярного использования.

Первый практический шаг — **prompt caching + cost ledger + budgets**, потому что без них мы не видим экономику и поведение агента.
