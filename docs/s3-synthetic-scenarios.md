# S3, пункт 3: проверяемые raw-сценарии и единый CLI

Последний пункт [согласованного S3](s3-frozen-inference.md). Вклад — проверка
сценарного пересчёта и сохранности результата. Рабочие CERT-модели, веса и
пороги остаются замороженными; исследовательские fits и выбор threshold не
входят в этот пункт. Малые модели functional fixtures проверяют программный
контракт, их метрики не являются результатом улучшения детектора.

## Контракт изменения

[Протокол v1](../configs/s3_synthetic_scenarios_v1.json) объявлен до запуска.
[Реализация](../src/synthetic_scenarios.py) принимает sealed isolated synthetic
preparation и связанную с ней модель из общего inference API. Поддерживается
одно raw-поле — `bytes_sent`: endpoint или первый, ранний member того же окна.
Операции: `noop`, `double`, `zero`; `set --value` принимает конечное
неотрицательное число. Boolean, строки, отрицательные значения, NaN/Inf и
остальные поля отклоняются. Ноль допустим.

Окно выбирается по `(user_id, timestamp, event_id)` детерминированно, без labels
и scores. У его первого member должно быть не меньше `rolling_window_events`
предыдущих событий в выбранном split; после endpoint должен быть следующий
event. В обеих редактируемых позициях исходный объём положительный. Можно
передать явный eligible `--item-id`; проверки истории сохраняются. При
недостаточной истории возвращается ошибка, окно не дополняется чужими событиями.

Каждый вариант получает собственный `scenario_id`. Сохраняются исходные event
IDs как ссылки на original snapshot и logical CSV record; принадлежность к
варианту задаётся scenario ID. Actor, timestamps, порядок, состав окон и
численные threshold не меняются. Patch изменяет одно значение одного события.

True labels и anomaly families отделяются до вызова builder. Внутренний
`label=0` удовлетворяет legacy sort/schema и не задаёт ground truth варианта.
Исходная разметка записывается только в `base_label_provenance.csv`.
`variant_ground_truth="unknown"`, `include_in_benchmark_metrics=false`, в том
числе для no-op. Сценарий хранит ответы модели, а не precision/recall варианта.

## Пересчёт и численные гарантии

`load_scenario_source` проверяет seals, исходный raw SHA256, preparation/state
identity, config/contract agreement и соответствие event IDs, actor, time,
порядка и split. Raw IDs воспроизводятся до chronological split прежним
способом. Состояние train не переобучается и не адаптируется.

1. Исходный split целиком пересчитывается `features.apply_feature_state`.
   Его features/inputs/scores сравниваются с original preparation, решения —
   точно. Эта проверка предшествует оценке изменённого варианта.
2. Для patch пересчитываются **все события затронутого пользователя в этом
   split**, включая прошлую историю и последующие зависимые события. Результат
   подставляется на прежние позиции в исходный prepared split.
3. Отдельный вызов прежнего builder пересчитывает **весь изменённый split**.
   Он не получает результат оптимизированного пересчёта. Features, X, ordered
   IDs/membership, scores и решения должны совпасть с ним точно.
4. Features/inputs/scores/решения остальных пользователей проверяются на точную
   неизменность. Все inference passes сохраняют прежние веса, score policy,
   dtype, CPU и batch size 128 у LSTM.

Важна граница прежнего isolated producer: builder возвращает float32, затем
`preprocessing.py` пишет/читает CSV, `experiment.prepare` копирует split через
CSV, и candidate снова читает его как float64. Сценарий воспроизводит обе
writer/reader стадии; прямое расширение float32 до float64 не эквивалентно
этой границе. LSTM получает прежнее преобразование в float32.

Заранее объявленные no-op tolerances: float32 `rtol=1e-6, atol=1e-7`, float64
`rtol=1e-12, atol=1e-12`. Они применяются к raw recompute, а не к exact saved
NPZ replay; решения всегда сравниваются точно. Shapes/dtypes должны совпадать,
числа должны быть конечными. После просмотра результата допуски не расширяются.

Независимые formula tests проверяют `log1p(bytes_sent)`, отношение к
`bytes_received` (при нулевом знаменателе — прежний fallback 0), shifted
rolling sum, границу его окна, отсутствие собственного вклада в prior sum,
влияние изменения прошлого на будущее и изоляцию пользователей. Проверка
oracle намеренно ломает пересчёт пользователя и требует явного отказа.

## Единый CLI

[src/argus.py](../src/argus.py) устанавливает CPU thread settings=1 до импорта
численных библиотек. Root должен существовать; каждая команда создаёт новый
UUID-каталог. Запись внутрь source artifacts запрещена. Исходные файлы
повторно проверяются перед завершением; failed/interrupted output не получает
completion manifest.

```bash
# Модель, frozen policies, hashes и поддерживаемые возможности.
python3 src/argus.py --root "$ARTIFACT_ROOT" inspect --run "$CANDIDATE"

# Exact replay существующего validation cohort.
python3 src/argus.py --root "$ARTIFACT_ROOT" replay --run "$CANDIDATE"

# Для operating bundle профиль всегда указан явно, даже legacy_f1.
python3 src/argus.py --root "$ARTIFACT_ROOT" replay \
  --run "$OPERATING_BUNDLE" --profile fp1_per_1000 \
  --source "$SAVED_ASSESSMENT" --role assessment

# Просмотр prepared cohort без создания новой оценки.
python3 src/argus.py --root "$ARTIFACT_ROOT" inspect \
  --run "$CANDIDATE" --source "$SAVED_ASSESSMENT" --role assessment

# Согласованный полный протокол пункта 2; исходные access guards сохранены.
python3 src/argus.py --root "$ARTIFACT_ROOT" diagnose \
  --model-variants "$MODEL_VARIANTS_SERIES" \
  --operating-points "$OPERATING_POINT_SERIES"

# Отдельный raw-вариант, по умолчанию validation.
python3 src/argus.py --root "$ARTIFACT_ROOT" scenario \
  --run "$SYNTHETIC_CANDIDATE" --preparation "$SYNTHETIC_PREPARATION" \
  --position earlier --operation double
```

Переменные обозначают конкретные frozen artifact paths; поиска latest/best
нет. `--expected-manifest` и `--expected-source-manifest` позволяют проверить
ожидаемый SHA256 completion manifest. `inspect --source` показывает shape,
dtype, feature order и IDs prepared cohort.

[frozen_runtime.py](../src/frozen_runtime.py) связывает assessment с точным
candidate/preparation manifest; проверяет reporting labels отдельно от batch,
reference scores/decisions и численный threshold. Legacy validation CSV
читается round-trip с восстановлением исходного score dtype; saved NPZ
сравнивается точно. Candidate validation membership дополнительно строится
прежним builder. Replay сохраняет свои metrics/reporting labels в отдельном
результате, не меняя исходную оценку.

Replay принимает уже сохранённые validation/calibration/assessment inputs.
Он не создаёт новую CERT final evaluation. CERT поддерживает inspect/replay
prepared inputs; raw what-if отклоняется до появления отдельного raw→user-day
reaggregation contract. `diagnose` вызывает существующий runner в отдельном
process, сохраняя его запрет real-test access и записи в source tree.

## Сохранённый результат

Каждый `synthetic_scenario_*` содержит:

- `protocol.json`, `patch.json`, `lineage.json`, `summary.json`, `report.md`;
- `raw_base.csv`, `raw_variant.csv`, `prepared_base.csv`, `prepared_variant.csv`
  без ground-truth columns; `base_label_provenance.csv` отдельно;
- `base`, `base_raw_recomputed`, `variant`, `oracle` inputs NPZ с X/IDs/members
  и prediction NPZ с scores/decisions, без labels;
- `changed_features.csv`, `responses.csv` со всеми endpoint ответами;
- `runtime_source/`, code hashes, environment, source manifest/file hashes,
  preservation check, status и completion manifest.

Сценарный ответ показывает чувствительность замороженной модели к заданному
raw patch. Он не доказывает причинность реального поведения или правильность
новой метки. Удвоение объёма не обязано менять бинарное решение каждой модели.

## Приёмка

Functional protocol: 1 200 fabricated events, восемь пользователей, seed42,
три дерева RF/IF, три HGB iterations, одна LSTM epoch с hidden4. Пять вариантов
score: RF, IF, HGB, LSTM window и LSTM endpoint; шесть patches на каждый.
Это проверка маршрута, отдельно от исследовательских результатов CERT.

Выполняются focused сценарные проверки, полный existing suite, compileall,
shell syntax и whitespace/link checks. Legacy `run_all.sh` и reference builder
запускаются в отдельной временной копии; исходные публичные reference metrics
и SSD hashes проверяются на неизменность. Snapshot, отдельный incremental
patch и отчёт сохраняются локально без общего commit прежних изменений.

Пункт 3 выполнен. В сохранённой демонстрации прошли **30/30** cases; все
**10 no-op** воспроизвели features/inputs/scores точно, решения не изменились.
Во всех случаях full-split oracle совпал точно. Изменения объёма затронули
11 tabular endpoints или 15 LSTM windows; бинарное решение менялось только
в трёх fixture cases. Отсутствие смены решения при изменившихся inputs —
допустимый ответ замороженной модели.

Полный набор **104 tests** прошёл. После дополнительной проверки отказа при
ошибочной передаче CERT final/preparation directory повторно прошли все
девять focused tests. Эта проверка запрещает открывать закрытые test members
ещё до full checksum verification. Все восемь реальных CLI replays на прежнем
CERT fold1 (пять model/score вариантов и три operating profiles HGB) совпали
точно на **65 579 endpoints** каждый; это повторение известного cohort,
не восемь независимых оценок качества.

Legacy synthetic RF/IF/LSTM metrics JSON после полного pipeline/reference
builder совпали с первоначальными точно. Все **1 321** проверяемых SSD
manifest/weight/state/threshold/config/contract hashes сохранены, как и
прежние source/config/reference файлы. Изменены только новые модули/тесты/
протокол и документация. Отдельный patch проверен на начальном snapshot;
общий commit/push не создавался.

S3 завершён: есть общий frozen inference, диагностика и проверяемый сценарный
маршрут. Следующий implementation stage — S4, разбор событий и train-only
normal references. CERT primary HGB остаётся на precision52,23%, recall59,91%,
F1 0,5580, FP354/FN259; сценарный код не изменил показатели детекции.
