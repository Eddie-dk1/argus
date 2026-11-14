# S3, пункт 2: exact replay и диагностика сохранённого качества

Пункт 2 [согласованного плана](s3-frozen-inference.md), 14 ноября 2026 года.
Вклад в качество — проверка сохранности и диагностика. Новых research fits,
поиска, выбора threshold или новой финальной CERT-оценки нет.

## Зафиксированный объём

[Конфигурация диагностики](../configs/s3_frozen_diagnostics_v1.json) связывает
два прежних artifact manifests и объявляет методы анализа до запуска:

- все 60 кандидатов model-variants v5: четыре feature policies, пять методов,
  три outer folds, seed42;
- exact API replay девяти operating bundles seed42: три ранее объявленных
  варианта × три folds, все три сохранённых профиля;
- диагностика всех 45 operating bundles v6: пять model seeds, те же три folds,
  все profiles, в том числе ineligible calibration и outer budget violations.

Итого — 69 API replay моделей, 87 replay decision profiles; диагностика 105
сохранённых candidates/bundles и 195 model/profile/fold/seed случаев, для
каждого calibration и assessment. Совпадающие controls и повторяющиеся HGB
scores не являются новыми независимыми наблюдениями.

## Воспроизведение

[Runner](../src/run_frozen_diagnostics.py) читает frozen artifacts через
`load_frozen_run` и вызывает `score`/`predict` на полных сохранённых assessment
inputs. Все сопоставления NPZ чисел и решений точные: прежние dtype, CPU
threads=1, inference batch128 для LSTM. Допуски не применяются и не расширяются.

Проверяются seal/lineage и source summary; подготовленные calibration/assessment
records сопоставляются с original-primary-train reporting metadata и прежними
fold assignments. Численные inputs, feature order, IDs и ordered member IDs
повторно строятся прежним builder и сравниваются целиком. Reporting labels
проверяются отдельно и в `PreparedBatch` не входят. Candidate, preparation и
assessment связываются manifest/state hashes, numeric threshold и named policy.

Legacy calibration имеет CSV-view scores, поэтому parser использует round-trip
чтение с восстановлением исходного score dtype. Operating calibration и все
outer scores имеют exact NPZ. Все сохранённые calibration/assessment метрики
проверяются по прежнему metric implementation; threshold selection не вызывается.
Параллельно diagnostic representation явно отмечает неопределённые метрики.

Основной CERT test закрыт audit hook. Co-located `account_days.csv` читается
только по reporting columns, с фильтрацией original primary train до joins;
test-прогнозов/метрик не создаётся. Весь исходный processed artifact tree
защищён от записи, отчёт сохраняется вне него. Runtime fit/selection функции
в процессе заменены запретами; исходный runtime/training code не меняется.

## Диагностические определения

[Расчёты](../src/frozen_diagnostics.py) и [читаемый отчёт](../src/frozen_diagnostic_report.py)
не выбирают конфигурацию или threshold.

- Для каждого role/fold/profile: precision, recall, F1, PR-AUC, FP/FN, confusion,
  class support/prevalence, аккаунты и incident-connected groups.
- Для operating profiles отдельно проверяются calibration feasibility, outer
  integer FP budget и minimum recall. Legacy F1 не получает новый budget задним
  числом; вместо checks для него хранится `null`.
- Score distributions отдельно для background/observable positives: count,
  mean и min/p25/median/p75/p90/p99/max, внутри role/fold/model.
- FP concentration: все аккаунты с FP, background denominator и rate каждого,
  доля FP в top1/top3/top10. IDs используются только для reporting.
- Доступная история: число прежних active days в `[date-30,date)` того же
  аккаунта, buckets `0`, `1–4`, `5–9`, `10+`, без labels и scores в определении.
  Не смешивается с `history_missing`, `cold_start`, `file_usb_profile_missing`:
  global-only policy может иметь fallback при достаточной наблюдаемой истории.
- Saved binary flags восстанавливаются через frozen train mean/scale;
  заранее заданный предел округления inverse transform — `atol=1e-5`.
  Это проверка binary reporting flags, а не допуск для replay scores/inputs.
- Отсутствующий feature имеет `available=false`, причину и `null` buckets.
  Отсутствующая class support/предсказанные positives дают `null` с причиной
  для неопределённой метрики; legacy численное значение остаётся отдельно для
  проверки прежнего результата. Обязательные пропущенные artifacts/folds
  прерывают запуск, а не скрыто исключаются.
- Scenario membership допускает пересечения. Recall/support, аккаунты, группы
  и incidents сохраняются по каждому сценарию; support по сценариям не следует
  автоматически суммировать как непересекающиеся классы.

Pooled OOF объединяет только решения на endpoints, присутствующих ровно один
раз в каждом model/seed/profile cohort. Pooled cross-fold PR/ROC AUC и score
distributions остаются `null`: score scales разных fit не объединяются.
Среднее отдельных fold PR-AUC допустимо только как явно названное описательное
macro mean. Seed variation при fixed fold и fold variation при fixed seed
хранятся раздельно; sample SD не называется CI.

## Запуск и артефакты

```bash
PYTHONPATH=src python3 src/run_frozen_diagnostics.py \
  --model-variants "$MODEL_VARIANTS_SERIES" \
  --operating-points "$OPERATING_POINT_SERIES" \
  --root results/experiments \
  --config configs/s3_frozen_diagnostics_v1.json
```

Обе source variables — конкретные сохранённые серии из plan config, без поиска
latest/best. Report root должен существовать и находиться вне read-only source
artifact tree. Новый UUID output содержит:

- `summary.json`: `argus.frozen_diagnostics.v1`, все 195 records / 390 role reports,
  scenario/feature/score/training policies, оценочный source/evidence level,
  coverage, pooled decisions, отдельные variance summaries и ограничения;
- `cohort_metrics.csv`: полная плоская таблица calibration/outer;
- `report.md`, `calibration_outer.png`: читаемые сравнения и график основного
  HGB seed42 профиля;
- `quality_hypotheses.json`: факты, возможные объяснения и очередь проверок;
- `replays/`: exact повторные scores/decisions/IDs/membership и связанные source
  input hashes; `entries/`, `contexts/`: детальные reporting records;
- `plan.json`, `environment.json`, `source/`, `source_hashes.json`,
  `access_guard.json`, status и completion manifest.

Ни одна будущая quality-гипотеза здесь не запускается. Перенос калибровки,
ложные тревоги на новых аккаунтах и слабые сценарии требуют собственных
протоколов с данными, контролями, objective, bounded budget и seeds. Новое
independent подтверждение требует принятия действительно независимых данных.
CERT raw what-if требует отдельного контракта; единый runtime CLI теперь
реализован в [пункте 3](s3-synthetic-scenarios.md).

## Полученный результат

Пункт 2 завершён. Все 69 полных API replays / 87 decision profiles совпали
точно; все 195 model/profile/fold/seed случаев представлены обеими roles.
Сохранённые calibration/outer metrics и 65 pooled decision cohorts совпали
с прежними сериями. Новое независимое подтверждение качества не заявляется.

Основная рабочая точка HGB по-прежнему не выполняет одновременно FP1/1000
и recall≥50% ни в одном из трёх unique outer folds. Пять model seeds дают
совпадающие scores, поэтому 0/15 — повторение трёх outcomes, а не 15
независимых проверок. Calibration выполняет оба условия в двух folds;
в fold2 recall только 16,48% уже на calibration.

| HGB rolling, seed42, primary profile | Fold1 | Fold2 | Fold3 |
|---|---:|---:|---:|
| Calibration FP/1000 | 0,610 | 0,958 | 0,995 |
| Outer FP/1000 | 1,086 | 0,122 | 4,206 |
| Calibration recall | 89,61% | 16,48% | 61,96% |
| Outer recall | 66,22% | 48,87% | 65,02% |
| Outer одновременно budget и recall | нет | нет | нет |

Pooled primary HGB остаётся P52,23% / R59,91% / F1 0,558039,
TP387/FP354/FN259 и 1,805 FP/1000. Legacy F1-control остаётся отдельным
контролем: P59,34% / R60,99% / F1 0,601527, FP270/FN252. Другой профиль
после диагностики не назначался основным.

Новые диагностические наблюдения, reference HGB seed42 / primary:

- **353 из 354 FP** возникли при минимум **10 прежних active days за 30 дней**;
  у этих endpoints также нет history/file-USB fallback. Единственный другой
  FP находится в bucket 1–4 и использует fallback. Это факты распределения
  ошибок, не доказательство их причины.
- FP затронули **42 аккаунта**; top1/top3/top10 дают **18,64% / 43,22% / 81,64%**
  FP. Account IDs не превращены в признаки, blacklist или индивидуальные пороги.
- Scenario3: **6/32**, recall18,75%, девять аккаунтов, **одна независимая
  connected group**. Legacy-control находит4/32. Эти результаты не устанавливают
  хороший перенос и не усиливаются повторением model seeds.
- Калибровка и outer ведут себя неодинаково по folds: например, HGB fold2
  PR-AUC0,1470 на calibration и0,8292 на outer. Это наблюдаемый разрыв между
  разными группами, а не установленный механизм и не общая PR-AUC двух cohorts.

Очередь гипотез: представительность групповой калибровки; признаки активности/
контекста для концентрированных FP, включая аккаунты с накопленной историей;
новые независимо проверенные группы слабых сценариев. Контрольные модели,
evidence links и требования отдельного bounded protocol сохранены в JSON.
Ни одна гипотеза не запускалась.

Полный набор **95 tests** прошёл; после устранения арифметического roundoff
sample SD для строго одинаковых чисел повторно прошли восемь focused tests.
Сами per-run/saved/pooled metrics и replay files не менялись; identical values
имеют точный mean и SD0, а не численный шум порядка1e-17. Повторная генерация
читаемого отчёта и provenance этой правки записаны отдельно.

Compileall, CLI help, shell syntax, Markdown/whitespace и AST checks пройдены.
Полный legacy synthetic regression и reference builder выполнены во временной
копии; RF/IF/LSTM metrics JSON совпали точно. Все **1 321** исходных SSD
manifest/weight/state/threshold/config/contract hashes сохранены. Прежние
source/config/reference файлы, включая `inference.py`, не менялись.
Initial snapshot и отдельный применимый incremental patch сохранены локально;
общий commit/push с предыдущими незакоммиченными изменениями не создавался.
