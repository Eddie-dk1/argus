# CERT: причинная история v3 — пункт 1 нового этапа

14 ноября 2026 года. **Реализация и development-подготовка завершены.**
Это техническая проверка входов, не новый результат detection. Модели CERT
в пункте 1 не обучались, эпохи/пороги не выбирались. Последующее
[парное сравнение через group-CV](cert-r4.2-group-cv-v3.md) завершено отдельно
в пункте 2 и не меняет границ этой подготовки.

## Зачем требуется изменение

В [завершённой первой серии](cert-r4.2-final-evaluation.md) global-only RF
имеет mean recall 16,5%; IF/LSTM также недостаточны для практической детекции.
У v1 персональный контекст доступен только обучающим аккаунтам, у v2 его нет
ни у кого. Гипотеза v3: восстановить персональный контекст по одинаковому
правилу у всех аккаунтов. Улучшение качества пока не измерено и не обещается.

Политика `cert_frozen_global_causal_all_v3` реализована в
[`cert_experiments.py`](../src/datasets/cert_experiments.py). Отдельные
[настройки модели](../configs/cert_r4_2_causal_history_v3.json) отличаются от
global-only v2 только версией и history policy. [План подготовки](../configs/cert_r4_2_history_development_v3.json)
связывает по SHA-256 обе конфигурации и прежнюю дневную подготовку.
Десять признаков истории и 61 базовый признак остаются прежними.

## Правило доступности данных

Для каждого наблюдавшегося активного account-day:

1. Удалить из rolling-профиля дни раньше `date−30`.
2. Рассчитать признаки по наблюдениям **строго раньше текущей даты**.
3. Только затем добавить текущие объёмы и vocabulary аккаунта в его историю.

Это одинаковый алгоритм для train, validation и development assessment.
ID аккаунта — ключ состояния, не числовой вход модели. Split, labels, group,
incident/scenario и строки answer key не управляют обновлением истории.
Положительные дни тоже остаются в доступном прошлом; очистка по истинным
меткам создала бы недоступное в работе преимущество.

- Суммы событий используют `[date−7,date)` и `[date−30,date)`.
- Персональные median/IQR рассчитываются по `log1p(total_events)` при наличии
  хотя бы пяти предыдущих активных дней в 30-дневном окне. До этого — frozen
  global train fallback. Фиксированные IQR floor `1e-6` и clip `[−20,20]`
  сохранены, новые параметры не подбирались.
- Персональный vocabulary включает все предыдущие наблюдавшиеся дни этого
  аккаунта, даже если они уже вышли за 30-дневное окно. На первом дне действует
  frozen global train vocabulary. `cold_start` отмечает первый наблюдавшийся
  день; `history_missing` — недостаток rolling-дней для robust profile.
- Пропуски дат не заполняются фиктивными днями. Calendar gap сохраняется.
  Target остаётся official observable membership последнего дня окна из пяти
  активных дней; первые четыре дня не входят в сравниваемые endpoints.

Global median/IQR/vocabulary и scaler fit используют только train; внутри CV
они должны заново рассчитываться только на inner fit. Персональные обновления
held-out — детерминированная обработка уже наблюдавшихся входов, без fit на
полной held-out истории и без использования меток.

**Это offline-перенос между аккаунтами, не временной прогноз.** Global state
и scaler могут использовать весь разрешённый train, включая train-даты позже
конкретного оцениваемого дня. Поэтому тест неизменности при изменении будущего
проверяется при **зафиксированных** global state, scaler и весах. Он не означает
инвариантность при переобучении global state на изменённом train. Оценка
доступна в конце активного дня, а не до его событий.

Каждый вызов `transform` заново проигрывает предоставленную историю. Это пока
не сервис с сохранением состояния между запросами; передача только текущего
дня без предыдущих наблюдений означает cold start.

## Изолированная development-подготовка

Команда [`prepare_cert_development.py`](../src/prepare_cert_development.py)
создаёт новый UUID-каталог, сохраняет plan/source/environment snapshots,
фильтрует прежние train/validation и готовит обе policy. Scope —
`primary_development`; в результатах нет `test.csv` или `assessment.csv`.
Candidate работает с ними как с train/validation, а `final_evaluate` явно
отклоняет этот scope. Настоящая group-CV требует отдельных inner preparations
и assessment; эта двухчастная подготовка сама по себе не является CV.

Дневной source содержит все три исходных split в одном CSV/context-файле:
проверка manifest и чтение этого источника затрагивают общий файл. Test-строки
исключаются **до построения признаков**, не входят в fitting, численные
матрицы, новые predictions или метрики. Нельзя описывать этот запуск как
«ни один байт, связанный с test, не читался». Старые prepared test, final scores
и checkpoint не используются для выбора нового варианта.

`history_dependencies.jsonl.gz` сохраняет точные rolling7/30 item IDs,
исключающую границу даты и предыдущий активный item. Цепочка предыдущих items
позволяет восстановить зависимости expanding vocabulary без квадратичного
списка всех предыдущих дней. Отдельный аудит повторно проверяет цепочку и
календарные memberships для каждой строки.

Воспроизведение, где переменные указывают на существующие каталоги SSD:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 src/prepare_cert_development.py --root "$ARTIFACT_ROOT" \
  --days "$CERT_DAYS" --plan configs/cert_r4_2_history_development_v3.json
```

Исходные raw, assignments, дневная подготовка, frozen конфигурации v1/v2 и
результаты final не перезаписываются. Старые policy сохраняют прежнее поведение.

## Результат подготовки

На SSD создана серия `processed/cert_history_development_356c71d43306477eb71dc1015df13c97`:

SHA-256 её `complete.json`:
`fe415dd6cdb0d288788071bffd9bdfb50daa16fb9ffc1d71cf8f1bcad2599b05`.

| Популяция | Активные дни | Endpoints | Positive endpoints | Первые дни без personal vocabulary | Дни с personal robust profile |
|---|---:|---:|---:|---:|---:|
| Train | 199 165 | 196 765 | 646 | 600 | 196 165 |
| Validation | 65 054 | 64 254 | 180 | 200 | 64 054 |

Аудит проверил **264 219 строк**. В обеих policy совпадают ordered IDs,
labels, 61 базовый вход, его scaling, global vocabulary/reference и endpoint
coverage. Все 71 числовой вход конечны. Отличается только политика истории и
её преобразованные колонки. Это не означает совпадения distributions train и
validation или гарантию качества модели.

Сохранённые подготовки:

- v2: `cert_preparation_731f0260c616440c83e13682909b738e`;
- v3: `cert_preparation_2299489b440748778e894138e4701b6b`.

## Проверки и ограничения

Все **56 тестов** прошли, из них семь новых проверяют:

- инвариантность истории к роли split при фиксированном global state;
- неизменность предыдущих features и scores сохранённой RF при изменении
  будущих наблюдений и vocabulary;
- независимость feature state и матриц от labels/атрибуции;
- отсутствие влияния inner calibration/assessment на fitted state и train;
- точные границы 7/30 дней, current-day exclusion, gaps, cold start и
  vocabulary после длинного перерыва;
- preparation без model fit и работу всех трёх моделей на искусственном
  development fixture без test, включая отказ final evaluation;
- отказ при изменённом source/config hash или втором изменяемом факторе.

Дополнительно предыдущая реализация из локальной копии сравнивалась с новой:
v1/v2 сохраняют точные global state и history matrices на искусственном наборе.
На реальных данных все новые manifests независимо сверены; у v2 feature state
и train/validation CSV побайтно совпадают с сохранённой global-only подготовкой
первой серии. Train CSV v3 побайтно совпадает с прежним v1 train: правило
доступного прошлого для train не менялось. Хеши всех **30** прежних checkpoints
и manifest завершённого final остались прежними.
Полный `run_all.sh` и `build_reference_snapshot.py` выполнены в изолированной
временной копии; RF/IF/LSTM reference-метрики воспроизведены точно.
Исторические release reference-файлы репозитория не обновлялись.

Накопление истории может постепенно нормализовать длительный инцидент. Это
реальная проверяемая слабость адаптации, а не причина удалять положительные дни
из прошлого. Сравнение adaptive/frozen поведения — отдельный фактор, позднее
проверенный в [пункте 3](cert-r4.2-file-usb-v4.md). Признаки file/USB,
endpoint-only score и новая регуляризация в границах этого пункта 1 не добавлены.

## Переход к пункту 2

[Отдельная bounded-серия](cert-r4.2-group-cv-v3.md) из 18 кандидатов завершена
в пункте 2. `assess_development` теперь поддерживает RF, IF и LSTM. Состояние
переподготовлено в каждом inner fit, epoch/threshold выбраны только на inner
calibration, сохранённые модели оценены на outer assessment. Основной objective
остался прежним F1 с прежними tie-breakers; опубликованы также PR-AUC,
precision, recall, FPR, TP/FP/FN, support и coverage. Получено умеренное улучшение
RF с большим числом FP; улучшения IF/LSTM в целом не подтверждены.

Имеющиеся folds уже использовались для development-диагностики. Их повторное
использование не создаёт нетронутый assessment для новых confirmatory claims.
Пять model seeds также не заменяют независимые группы. Сценарий 3 по-прежнему
имеет слишком мало независимых групп для сильного вывода о переносе.

Правила train-only fitting и отдельного уровня оценки согласуются с
[рекомендациями sklearn по leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage),
[group-CV](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
и [nested selection](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
Это основания протокола, не доказательство, что v3 повысит метрики Argus.
Использованный CERT test остаётся использованным; независимое подтверждение
потребует отдельно принятых данных с проверкой дубликатов и связанных сущностей.
