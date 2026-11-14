# Argus / Spectra Lab — единый план разработки

Implementation update 2026-11-12: the new [isolated experiment workflow](isolated-experiments.md) now provides synthetic preparation IDs, exact membership, saved inputs/models, validation-only candidates and explicit final evaluation. Descriptions below of shared root outputs and mandatory test access apply to legacy scripts; CERT preparation and Spectra remain future work.

Версия: 2.0. Обновлено: 27 октября 2026 года. Основание аудита: исходники Argus, HEAD 08d6576.

Статус: утверждённое направление будущей разработки. Spectra Lab и перечисленные ниже команды пока не реализованы. Этот документ определяет состав продукта и изменения кода; [roadmap](roadmap.md) — очередность работ после SSD, [experimental protocol](experimental-protocol.md) — правила научной оценки.

Уточнение 12.11.2026: техническая приёмка CERT завершена; зафиксирована
[label policy v1](cert-r4.2-label-policy.md). User-group split для CERT означает
неделимые группы аккаунтов, связанных официальным инцидентом, включая сценарий
№3. Индекс меток проверен; распределение групп, user-day preparation и обучение
ещё не реализованы.

## 1. Одно приложение и одно название лаборатории

**Argus** — всё приложение: данные, признаки, обучение, оценка и встроенная **Spectra Lab**. Репозиторий, backend, frontend и установка общие.

В Spectra четыре пользовательских раздела:

| Раздел | Задача |
|---|---|
| Обзор | Выбрать данные/эксперимент, посмотреть метрики и список примеров |
| Подбор моделей | Ограниченно исследовать конфигурации, сравнить качество, время и память |
| Разбор событий | Проверить источник, вход модели, окно, нормальный ориентир и расхождения решений |
| Проверка сценариев | Изменить разрешённое исходное поле в копии и повторно применить зафиксированную модель |

EventLens — прежнее имя функций «Разбора событий». Отдельный модуль, сервер, продукт или пункт меню EventLens не создаётся. Названия Adaptive, Edge и Causal также не используются как самостоятельные подсистемы.

Карта поведения — последующий способ просмотра внутри «Обзора». Отдельный Streamlit-dashboard и универсальный AutoML-продукт не разрабатываются.

Визуальная спецификация: [docs/design.md](design.md). Минималистичный dark navy, нейтральные controls и логотип, приглушённо-синее выделение. Красный только для ошибок/аномалий, без красной навигации и обычных CTA. Старые цветные концепты архивные.

## 2. Что взято из рассмотренных идей

| Источник идеи | Включаемая функция | Ограничение |
|---|---|---|
| Spectra Lab | Общая лаборатория исследования результатов | Одна оболочка внутри Argus |
| EventLens | Источник, точное окно, признаки, временная линия | Раздел «Разбор событий» |
| Causal | Допустимые изменения и реакция модели | Без обещаний причинности в реальном мире |
| Adaptive | Бюджетированный выбор конфигураций | Random search, затем TPE; фиксированные search space и validation |
| Edge | Скорость, время обучения, память и компромисс качества/затрат | CPU baseline; новые runtime/quantization только по измеренной потребности |
| Live, Wave | Не входят в согласованный объём | Нет системного агента, Wi-Fi sensing или нового оборудования |

Два исследовательских процесса: разбор зафиксированной модели и подбор конфигураций. В первом меняется вход, во втором — обучаемая конфигурация. Сценарные примеры не возвращаются автоматически в обучение или в метрики исходного test.

## 3. Как работает система целиком

~~~text
Источники → принятие snapshot → split → train-fitted признаки
                                      ↓
                    фиксированные baselines / подбор конфигураций
                                      ↓
                          заморозка model + state + threshold
                                      ↓
                             финальная оценка на test
                                      ↓
                         Spectra: обзор и разбор примера
                                      ↓
                 нормальный ориентир → копия с изменением
                                      ↓
                      пересчёт признаков → та же модель
                                      ↓
                         сравнение результатов и экспорт
~~~

Первые разбор и сценарии реализуются на фиксированной synthetic LSTM. «Подбор моделей» подключается позже. Пользовательский маршрут позволяет подобрать модель перед разбором, но реализация начинается с более узкого воспроизводимого пути.

## 4. Текущий код и разрывы

| Участок | Сейчас | Что изменить |
|---|---|---|
| src/data_paths.py | ARGUS_DATA_ROOT, fail-fast при отсутствующем external root | Разрешение новых artifact/cache roots; не создавать скрытые fallback-копии |
| src/datasets/loading.py | Chunked CSV с source file/group/record index | Стабильные event IDs, нормализованные источники и переносимая identity |
| src/preprocessing.py | Synthetic split, CSV/NPZ, общий feature state | Immutable preparation, IDs до sort/split, сохранение зависимостей |
| src/features.py | Causal context, train-fitted state, окна без membership IDs | Общий порядок, exact members, schema/dtype и повторяемый recompute |
| src/train_baseline.py | RF/IF, собственный endpoint cohort, test в каждом запуске, нет сохранённых RF/IF weights | Общий cohort, train/validate-only mode, per-run models и predictions |
| src/train_lstm_autoencoder.py | Строит окна заново; CPU; общий checkpoint; test после каждого запуска | Actual input mapping, reloadable per-run checkpoint, отделение финальной оценки |
| src/evaluate.py | Сверка user/time/label | Snapshot/item identity, target, split и explicit comparison scope |
| src/experiment_tracking.py | Каталоги метрик/ошибок, но root outputs перезаписываются | Полные immutable artifacts, group/study IDs и complete marker |
| src/utils.py | ML imports и synthetic path resolution при импорте | Отделить лёгкое чтение manifests от ML runtime/storage availability |
| src/explain_anomalies.py | Rule-based сигналы по общим RF/IF outputs | Run-bound фактические сигналы с rule provenance |
| run_all.sh | Полный synthetic workflow, включая test | Сохранить reference путь; добавить отдельный безопасный candidate workflow |

Существующие RF/IF оценивают endpoint feature row, LSTM — окно, а target соответствует endpoint. «Один пример» не означает одинаковый численный input для трёх моделей.

LSTM window в reference config — 5 событий, rolling history — 10. Пять members не являются всей историей, влияющей на признаки. Строгий normal reference pool также отличается от текущего train filter, который использует endpoint label.

## 5. После покупки SSD: входной рубеж

Подробные команды и порядок — в [roadmap](roadmap.md#resume-after-ssd) и [storage policy](data-storage.md).

1. Проверить фактический mount, создать выбранный data root на подключённом диске, определить бюджет по manifest выбранных файлов.
2. Сначала получить только CERT r4.2 и соответствующие answers; сохранить README, license, sizes и SHA-256.
3. Проверить schema, ключи/даты разметки и source locators. Не запускать весь набор публичных данных одновременно.
4. Стабилизировать artifacts и разделение candidate/final workflow на малом synthetic fixture.
5. Реализовать первый CERT user-day baseline через общий контракт и заморозить эксперимент. Разработка generic contracts и data acceptance может идти последовательно небольшими изменениями.
6. Сделать synthetic сценарный прототип и единый интерфейс; CERT сначала доступен для просмотра, what-if только после поддержки его агрегации.
7. Добавить ограниченный подбор конфигураций на стабильных сценариях.

CICIDS2017 и CSE-CIC-IDS2018 сохраняются в дипломном roadmap отдельными data milestones, но не блокируют первую лабораторию. Если объём ВКР уточняется, изменяется заранее объявленный scope; неподготовленные сценарии не объявляются оценёнными.

## 6. Архитектура и стек

Один FastAPI/Uvicorn backend раздаёт собранный React/TypeScript/Vite frontend. Parquet + DuckDB используются для событий и запросов; JSON — для manifests. SQLite допускается для состояния studies/jobs/notes при реальной потребности.

~~~text
src/
  artifact_contracts/       # identity, manifests, schemas, publication, validation
  datasets/                # существующие loaders/contracts + accepted adapters
  inference/               # frozen model loading, score/reconstruct/encode
  research/
    references.py          # train reference pools и nearest neighbors
    scenarios.py           # raw patch policy, lineage, frozen recompute
    reconstruction.py      # разложение MSE
    selection.py           # study/search orchestration без test
    benchmarks.py          # одинаковые условия времени и памяти
    projections.py         # PCA/UMAP, следующая очередь
  spectra/
    app.py                 # один backend
    config.py
    repositories/
    services/
      investigation.py     # раздел «Разбор событий»
      comparison.py
      jobs.py
      exports.py
    api/
    static/
  serve_spectra.py
  validate_spectra_artifacts.py
frontend/spectra/
  src/features/
    overview/
    model-selection/
    investigation/
    scenarios/
tests/fixtures/spectra/
~~~

Это целевая схема: каталоги создавать по мере задач, без переноса всего src ради структуры. Общие contracts/inference/research не зависят от frontend или FastAPI. В viewer torch не обязателен; inference объявляется capability.

Первый runtime — CPU, поскольку текущий LSTM явно выбирает CPU. MPS/Core ML/MLX и квантование не входят в обязательный результат. Версии Python/Node/зависимостей фиксируются после smoke-install, frontend поставляется без необходимости Node у пользователя.

## 7. Контракты и единицы анализа

### 7.1. Идентичность

| Поле | Назначение |
|---|---|
| dataset_snapshot_id | Состав/содержимое raw и правила чтения/нормализации |
| event_id | Логическая исходная запись до sorting/filtering |
| preparation_id | Split, feature schema/state, код и параметры подготовки |
| analysis_item_id + unit_type | Объект оценки: synthetic event, CERT user-day, CIC flow |
| sequence_set_id / sequence_id | Правила окон и ordered members |
| cohort_id | Объекты, split и определение цели |
| run_id / experiment_id | Одна модель и группа сопоставимых runs |
| study_id / trial_id | План поиска и одна candidate configuration |
| reference_set_id / representation_id | Кандидаты нормы и рецепт представления |
| scenario_id / evaluation_id | Версия raw patch и результат frozen inference |
| projection_id | Версия обзорной карты, позднее |

На synthetic item соответствует одному event. На CERT user-day item связан с множеством raw events; нужны item_members и при необходимости sequence→item→event. Нельзя назвать суточную оценку предсказанием одной произвольной записи.

Snapshot hash включает relative source IDs, sizes, SHA-256, parser/adapter/time policies. Абсолютный mount хранится локально, перенос тех же файлов не меняет identity. CSV logical record index не равен физическому номеру строки при multiline полях и не зависит от chunk size.

Один tie ordering применяется в features, splits и windows; labels не участвуют в IDs/ordering. Неизвестная timezone сохраняется неизвестной. Изменение timezone policy — новая подготовка, а не молчаливое добавление Z.

### 7.2. Минимальные сохраняемые артефакты

- Dataset manifest, events и отдельные labels; raw остаётся неизменным.
- Preparation manifest, split membership, feature schema/state и transformed feature rows.
- Analysis items, item_members для агрегатов, sequence members с positions и endpoint item.
- Cohort membership, target definition и provenance.
- Полные validation/test predictions конкретного run и decision rule.
- Model artifact + architecture/input schema/class order/versions/hashes.
- Experiment group manifest с expected/completed/skipped models.
- Для поиска — study config, trial configs/status/costs и validation results; у candidate нет test metrics.
- Для сценария — base identities, patch, policy, dependency context, input hashes, feature diff и frozen results.
- Для reference — train pool membership, normal/exclusion policy, representation и distances.

RF score — predict_proba с сохранённым class order; IF — отрицательное score_samples; LSTM — средняя squared reconstruction error по времени/признакам окна. Порог, оператор, направление и calibration provenance хранятся явно. Нормализация рангов не превращает scores в общую вероятность атаки.

### 7.3. Размещение и жизненный цикл

~~~text
data-root/<scenario>/
  raw/
  interim/spectra/<snapshot-id>/             # normalized events, labels
  processed/spectra/<preparation-id>/        # items, features, state, members
    models/<run-id>/
    predictions/<run-id>/
    research/references/
    research/scenarios/
  manifests/spectra/

results/experiments/<scenario>/<run-id>/      # небольшие метрики и manifests
results/studies/<study-id>/                  # лёгкий study index
~~~

Крупные models/predictions/context располагаются на data root. Сейчас results всё ещё привязаны к репозиторию: их изолируемый root нужно реализовать отдельно, новой поддерживаемой env-переменной пока нет.

Запись идёт в staging на целевом томе; после проверки создаются final manifest и complete marker. Atomic rename не делает общей транзакцией два разных тома. Завершённые artifacts не перезаписываются. Состояния incomplete/failed/skipped/legacy/unsupported видимы; availability SSD учитывается отдельно.

Модели хранятся один раз на run, shared events/features переиспользуются. Bundle с неизвестными weights не становится доверенным из-за checksum. Произвольные pickle/joblib из внешнего пакета не загружаются.

## 8. Разбор событий и проверка сценариев

### 8.1. Разбор

Карточка показывает исходные записи, численные входы модели, exact window и дополнительный контекст. Member window отличается от future context; source values — от standardized features.

Labels источника, prediction и мнение исследователя — разные поля. При missing score отображается «Нет оценки», при отсутствии label не вычисляется confusion category. Разные snapshots/targets не соединяются; intersection metrics имеют собственный scope.

Rule-based сигналы, MSE contributions, различия с нормальным примером и эффект raw patch подписываются раздельно. Ни один из них автоматически не доказывает причину инцидента.

### 8.2. Нормальный ориентир

Первый exact kNN baseline использует flatten общих LSTM feature windows одной длины и схемы с train-fitted scaling. Вторая representation — encoder hidden state конкретного checkpoint, если сравнение оправдано.

Reference pool формируется только из разрешённых train окон. Для synthetic требовать normal labels у всех members и проверенного rolling dependency context. Это более строгая политика, чем endpoint target исходного detector. В данных без labels писать «предсказано нормальным», не «подтверждённая норма».

Исключать self, дубли и пересечение source events; учитывать почти одинаковую историю при оценке. Metric/k/context filters выбираются на development/validation и замораживаются. Отсутствие достаточно похожего нормального примера — допустимый ответ. Поиск выполняется в representation space, не по экранным координатам UMAP.

### 8.3. Первый raw patch

Начать с bytes_sent endpoint synthetic окна, затем раннего member. Время, actor, category, ordering и membership фиксированы. Policy ограничивает поля, единицы и допустимые значения. Schema-valid не означает реалистичное нормальное поведение.

~~~text
Base run + исходная история
 → проверка patch
 → отдельная копия raw
 → тот же feature builder с frozen train state
 → зависимые признаки и model input
 → тот же checkpoint + threshold
 → отдельный scenario result
~~~

В baseline пересчитывать всю entity history в одном split, затем оптимизировать по зависимостям. Одного окна LSTM недостаточно для rolling features. В текущем коде prior_log_bytes_sent_sum — сумма логарифмов объёмов; воспроизводить точную формулу.

Изменение endpoint не попадает в его собственный prior sum; изменение early member может изменить признаки следующих members. No-op должен воспроизводить base features/X/score/decision с заранее объявленным числовым допуском. Сравнивать оптимизированный recompute с независимым full recompute oracle.

State, weights, threshold, raw, base labels и predictions неизменны. Новая ground truth сценария неизвестна, если отдельный механизм не определяет её. Изменённый пример не участвует в F1 исходного test.

CERT what-if включать только после raw→user-day reaggregation и определения доступных полей. Нельзя добавлять bytes_sent в источник, где такого наблюдения нет. Просмотр CERT может работать без сценарной capability.

### 8.4. Ошибка реконструкции

R[t,f] = (X[t,f] − reconstruction[t,f])². Среднее R совпадает с сохранённым score. Вклад по времени/feature groups показывает плохое восстановление; response на изменение проверяется отдельно. Не называть ручной patch минимальным контрфактуальным объяснением без оптимизации и проверки минимальности.

## 9. Подбор моделей: ограниченный поиск и измерения

Это второй процесс внутри Spectra, после корректного разбора фиксированных моделей.

### 9.1. Первоначальный объём

- Одна стабильная dataset/preparation и одна model family на study.
- Небольшой search space в config до открытия финальных результатов.
- Сначала seeded random search, затем TPE как сравниваемая стратегия.
- Для LSTM сначала hidden size и learning rate; длина окна/feature set фиксированы. RF/IF подключать после baseline runner.
- Бюджет wall-clock и maximum trials; CPU, thread limits и benchmark cohort объявлены.
- Trial status complete/pruned/failed/interrupted; причины и израсходованный бюджет сохраняются. Failed trials не исчезают из стоимости поиска.
- Candidate обучается/калибруется на train/validation без чтения test.
- Выбранные configurations/seeds фиксируются до final evaluation.

Это конкретная автоматизация эксперимента, не автономный ИИ-исследователь, active learning или универсальный поиск любых архитектур.

### 9.2. Цель, ограничения и Парето

Для первого study сохранить protocol v1: validation F1, tie-breakers по recall/FPR. Одновременно показывать PR-AUC, support/prevalence, время и память. Ограничения FPR/latency/RAM задаются до study; если они меняют selection rule, требуется новая именованная версия протокола.

Pareto-обзор сравнивает validation quality и measured inference cost на одном cohort. Batch throughput, single-item latency, training time и feature recompute time — разные измерения. Peak RSS отдельного process не равен memory profiler одного участка.

Условия фиксированы: hardware, device, batch size для benchmark, warm-up, repeats, threads, schema и cohort. Изменение train batch size не меняет молча inference benchmark batch. Длину окна в первом поиске не варьировать; позднее сравнивать configurations на заранее заданной общей популяции, явно учитывая потерю coverage.

Нельзя выбирать победителя по красивому test Pareto-графику и затем называть этот test независимым.

### 9.3. Сравнение стратегий

Одинаковый search domain, compute budget, правила early stopping/pruning и preprocessing reuse. Повторить search с несколькими search seeds, отдельно от model seeds. Если pruning различается, считать это другой стратегией и учитывать экономию бюджета.

Частые обращения к одному validation могут привести к overtuning. До финального study объявить дополнительную проверку устойчивости на допустимых temporal/group folds внутри development data, сохраняя естественный split сценария. Малую validation не расширять за счёт test.

UI показывает trials, параметры, качество/затраты и сохранённую выбранную configuration. Выбор «зафиксировать модель» создаёт immutable selection record. Последующие what-if сценарии относятся к ней и не меняют study ranking.

## 10. Порядок реализации и приёмка

Оценки относятся к одному разработчику с готовой средой. Это ориентиры активной работы после доступа к данным, не календарные обещания; ожидание загрузки, разметочных уточнений и участников оценивается отдельно.

| Этап | Работа | Приёмочный результат | Код |
|---|---|---|---|
| S0: SSD и принятие CERT | Mount, manifest/budget, r4.2+answers, schema/label audit | Принятый или явно отклонённый snapshot; большие данные вне Git | data_paths, check_data_storage, datasets |
| S1: Контракты и изоляция | IDs, item/sequence membership, state/schema, trial/final separation, isolated roots | Synthetic regression; два runs независимы; trial не читает test | features, preprocessing, tracking, training, evaluate |
| S2: Первый публичный baseline | CERT raw→user-day→group split→train-fitted features→оценка | Воспроизводимый user-day run с event mapping; ограничения cold start записаны | datasets + scenario preparation |
| S3: Frozen inference и Python-сценарий | Выполнен по [трём согласованным пунктам](s3-frozen-inference.md): общий API, диагностика, [raw-сценарии/CLI](s3-synthetic-scenarios.md) | RF/IF/HGB и обе LSTM score policies: no-op, bytes patch и full-split oracle; frozen sources сохранены | inference + research/scenarios |
| S4: Разбор и нормальные ориентиры | API/карточка, timeline, exact kNN, residuals, scope/coverage | Исходник→model input→reference проверяемы; CERT read-only capabilities | spectra + research/references |
| S5: Единый пользовательский маршрут | Обзор, разбор, проверка сценариев; history/jobs | Пользователь выполняет полный маршрут без правки файлов | один frontend/spectra |
| S6: Подбор моделей | Bounded random search, затем TPE, измерения и Pareto view | Study воспроизводим; только validation; winner заморожен | research/selection + benchmarks |
| S7: Поставка и исследование | Offline bundle/view, resource limits, tests, эксперимент статьи | Устанавливаемая лаборатория и подтверждённые результаты | exports/jobs/tests/docs |
| S8: Следующая очередь | CICIDS2017, затем CSE transfer по data roadmap; карта после основного маршрута | Каждый источник/метод имеет отдельные contracts и evidence | отдельные adapters/projections |

Ориентир S1: 5–9 дней; S2: 5–10 после принятия schema; S3: 3–5; S4–S5: 6–10; S6: 4–7; S7: 4–7. Сроки уточняются после S1/S2, общий резерв 20–30%; старые сроки отдельного EventLens больше не применяются.

Разбор synthetic и Python-прототип S3 допустимо развивать до окончания S2, если ожидание данных не даёт прогресса. Подбор S6 не обходится без изоляции S1 и стабильного baseline. Не ждать всех трёх публичных датасетов для первого работающего UI.

### Небольшие пакеты изменений

1. SP-01: identities, immutable preparation и exact membership.
2. SP-02: train/validate-only runner и отдельная final evaluation.
3. SP-03: per-run models/predictions, reload, group/status и results isolation.
4. SP-04: CERT item mapping, user-day features, group baseline и cold-start tests.
5. SP-05: synthetic no-op/patch/recompute.
6. SP-06: normal references, residuals, backend разбор.
7. SP-07: единая оболочка и проверка сценариев.
8. SP-08: bounded study runner, random/TPE comparison и resource measurements.
9. SP-09: offline export/import, robustness, research evidence и документация.

Это разбиение для будущей разработки, а не указание сейчас создавать коммиты, PR или выполнять training.

## 11. API, capabilities и локальный runtime

Один API /api/v1 предоставляет experiments/runs, examples, timelines, references, scenarios/evaluations, studies/trials/selections, jobs и exports.

Пример будущих маршрутов:
- GET /experiments, /runs/{id}/capabilities, /runs/{id}/examples.
- GET /runs/{id}/examples/{item_id}, /runs/{id}/sequences/{sequence_id}.
- GET /runs/{id}/examples/{item_id}/references и /comparisons.
- POST /scenarios/preview, /scenarios, /scenarios/{id}/evaluations.
- POST /studies/preview, /studies; GET /studies/{id}/trials.
- POST /studies/{id}/selections и отдельная явная final-evaluation job.
- GET /jobs/{id}; POST /jobs/{id}/cancel.
- POST /exports/preview, /exports; GET завершённого пакета по ID.

Все пути выше с префиксом /api/v1; это проект API, не готовые endpoints. Ни браузер, ни manifests извне не передают произвольный SQL/файловый путь. Cursor связан с parent IDs, filters и sort key; missing и truncated явно подписаны.

Один local server на loopback, bounded worker jobs, thread-local DuckDB cursor, ограниченные model cache/RSS/temp budgets. Viewer не импортирует heavyweight ML при старте и может показать недоступный SSD. Session/Origin/Host checks, escaping log values, allowed roots и cancellation обязательны в реализуемых путях.

Состояния unsupported_model, insufficient_context, invalid_patch, no_reference_match, invalid_cursor, resource_limit и incomplete_run не скрываются за «нормальным» результатом. После restart незавершённая задача получает interrupted status.

View bundle содержит достаточные events/features/members/predictions/references/готовые scenario results для автономного просмотра. Продолжение inference требует отдельно зарегистрированных доверенных models/state/history. Не обещать replay по одному screenshot или view bundle.

## 12. Проверки и правила научного эксперимента

| Область | Обязательная проверка |
|---|---|
| Identity/source | Chunk invariance, tied timestamps, root relocation, multiline CSV, changed raw |
| Inputs | No provenance in X, train-only state, exact membership/positions, schema/dtype |
| CERT | User-day→source mapping, no-user-overlap, cold-start без test-fitted personal profiles |
| Persistence | Reload score/decision, RF class order, IF sign, eval mode |
| Trial isolation | Нет чтения test в candidate; independent output roots; no stale model reuse |
| Search | Fixed domain/budget/cohort, failed-cost accounting, search/model seeds, immutable selection |
| Scenario | No-op, frozen state, no base mutations, invalid fields, unknown new label |
| Dependencies | Endpoint не входит в prior sum; early edit меняет нужные successors; full recompute oracle |
| Reference | Train-only strict normal pool, self/overlap exclusion, explicit no match |
| Comparison | Same target/snapshot/item population, scope/coverage и missing |
| UI | Raw vs normalized, input vs context, facts vs contributions vs response |
| Runtime/export | Missing SSD, limits, cancel/restart, round trip bundle, HTML/path/SQL boundaries |

Использовать существующие unittest и meaningful новые fixtures, API tests и browser маршрут. Для изменения только документации не выполнять training и не пересобирать reference metrics.

Основная статья: оценка реакции моделей на согласованные изменения событий и проверка полезности анализа. Методы: residual contributions, normal reference differences, raw patch с пересчётом. Метрики: consistency, допустимость, decision changes, размер изменения, latency и устойчивость.

Дополнительный эксперимент подбора: random vs TPE при одном compute budget, качество/стоимость и устойчивость выбора. Подключение Optuna само по себе не научная новизна. Не объединять две разные гипотезы в одно неразличимое утверждение.

Пилот UI: подготовленный CSV/notebook, простой разбор и полный маршрут с references/scenarios; одинаковые данные/обучение, balanced order и заранее известные ответы. Участник — независимая единица анализа; overlap windows/эпизоды группируются при статистике.

Параметры и cases выбираются на development/validation. После интерактивного изучения test его нельзя использовать как нетронутый holdout для новых настроек. Сценарий объясняет модель, не доказывает причинность мира.

## 13. Что отложено

Карта: одна point unit/representation, PCA baseline, UMAP train fit и frozen transform, фиксированные coordinates при model coloring. Neighbor search не по 2D coordinates. Кластеры, если понадобятся, оцениваются отдельно.

После доказанной потребности: дополнительные raw patches, CERT reaggregation what-if, SHAP, дополнительные runtime/quantization, portable replay. Live collection, Wi-Fi sensing, causal discovery, arbitrary architecture AutoML и отдельные репозитории не входят в обязательный scope.

## 14. Готовность и документы

- [ ] Приняты данные первого публичного сценария и сохранены provenance/label rules.
- [ ] Один продукт Argus, одна Spectra Lab и четыре согласованных раздела.
- [ ] Разбор, references и synthetic what-if воспроизводимы от сохранённых входов.
- [ ] Candidate studies не читают test и не меняют чужие runs.
- [ ] Выбранная модель зафиксирована до final test; сценарии сохраняют её state/threshold.
- [ ] Одинаковые inputs/cohorts и benchmark conditions подтверждены.
- [ ] CERT поддерживает user-day mapping и честные ограничения capabilities.
- [ ] Viewer/bundles/offline и отсутствие SSD обработаны.
- [ ] Существующие regression и новые meaningful tests проходят.
- [ ] Статья и demo используют реальные измерения и явно названные ограничения.

[Roadmap](roadmap.md) задаёт последовательность, [protocol](experimental-protocol.md) — инварианты, [features](features.md) — schema/formulas, [storage](data-storage.md) — SSD и артефакты, [datasets](datasets.md) — принятие источников, [design](design.md) — интерфейс. Исторические макеты от 17.10.2026 архивные.

Этот план заменяет версию 1.0 от 17.10.2026 и отдельные наброски EventLens. Исходный код и опубликованный synthetic reference при редактировании документации не меняются.
