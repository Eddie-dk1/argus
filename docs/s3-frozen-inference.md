# S3: общий inference, диагностика и проверяемые сценарии

Согласованный план на 14 ноября 2026 года — три пункта последовательно,
по одному за запрос. Он заменяет предварительное разбиение на четыре задачи
в [оценке предыдущего этапа](cert-r4.2-stage-assessment.md).

1. **Общие правила и inference — выполнено.** Постоянные требования всех этапов в
   [AGENTS.md](../AGENTS.md), CONTRIBUTING и экспериментальном протоколе;
   единая загрузка RF/IF/HGB/LSTM из frozen artifacts.
2. **Воспроизведение и диагностика — выполнено.** [Отчёт пункта 2](s3-frozen-diagnostics.md): все 60 результатов серии model-variants
   и все профили девяти operating bundles seed42/трёх folds; диагностика всех
   folds/seeds пунктов 4–5. Сравнить calibration/outer, FP-бюджет и recall,
   score distributions, аккаунты/независимые группы, cold-start и сценарии.
   Разделить факты, возможные объяснения и будущие quality-гипотезы.
3. **Synthetic scenarios и CLI — выполнено.** [Протокол и результаты](s3-synthetic-scenarios.md).
   `bytes_sent`: endpoint, ранний member, no-op,
   удвоение и ноль; пересчёт всей истории пользователя через frozen builder,
   проверка полным split oracle; сохранение lineage/patch/inputs/scores.
   CLI `inspect`, `replay`, `diagnose`, `scenario` и полный regression.

Исследовательские обучения и подбор параметров не входят в S3. Пороги по
диагностике не меняются. Уже использованные development folds и CERT test
не становятся независимыми данными. HGB сохраняет роль исследовательского
кандидата; отрицательный результат рабочего FP/recall-бюджета остаётся в силе.
Raw what-if CERT закрыт до отдельного контракта reaggregation.

## Пункт 1: контракт

Реализация — [src/inference.py](../src/inference.py). Загрузчик принимает
локальный доверенный sealed isolated candidate или operating-point bundle.
Нужен `complete.json` с `files`, полный статус и все исходные файлы с прежними
checksums. Проверяются schema/state/config agreement, feature/score policies,
семейство и размерность модели, конечный численный threshold и lineage.
Legacy `results/experiments/.../inference/` с другим форматом manifest этим API
не поддерживается: он не содержит полного isolated preparation contract.

```python
import numpy as np
from inference import PreparedBatch, load_frozen_run

# Candidate: единственный сохранённый threshold, профиль legacy_f1.
run = load_frozen_run(candidate_path, expected_manifest_sha256=manifest_sha256)
# Operating bundle: profile обязателен; автоматического выбора нет.
# run = load_frozen_run(bundle_path, profile="fp1_per_1000")

with np.load(saved_inputs_path, allow_pickle=False) as arrays:
    batch = PreparedBatch(
        X=arrays["X"],
        feature_columns=run.spec.feature_columns,
        item_ids=arrays["item_ids"],
        member_ids=arrays["member_ids"],
        preparation_id=run.spec.preparation_id,
        preparation_manifest_sha256=run.spec.preparation_manifest_sha256,
        feature_state_sha256=run.spec.feature_state_sha256,
    )
scores = run.score(batch)
decisions = run.predict(batch)
# Только LSTM:
# reconstruction = run.reconstruct(batch)
```

Пример предполагает, что вызывающая сторона проверила seal/lineage источника
saved inputs. Нельзя просто присвоить чужим inputs hashes из `run.spec` и
считать provenance проверенным. Batch хранит заявление производителя; runtime
проверяет его совпадение и структуру, но численные данные сами по себе не
доказывают происхождение, хронологию или membership. Пункт 2 отдельно проверяет
их точное совпадение с сохранёнными assessment artifacts.

Batch не содержит labels, threshold override или preprocessing fit. Числа
передаются без перестановки или изменения dtype. Tabular inputs — двумерные
float32/float64; LSTM — float32 `[items, time, features]` с frozen длиной окна.
В каждом batch нужны уникальные непустые Unicode item IDs, ordered member IDs
с endpoint на последней позиции, без повторов внутри окна, и hashes preparation
и feature state. Пересечение members разных окон допустимо. Пустой batch
сохраняет shape и dtype результатов. Владелец массивов не меняет их во время
вызова inference.

`batch_from_frame(...)` использует прежний `build_sequences` на одном уже
подготовленном split: порядок `user_id, timestamp, event_id`, табличные строки
на тех же endpoints. Labels отбрасываются до builder; внутренние нулевые
placeholders нужны только его legacy return schema. Helper не преобразует raw,
не проверяет seal split и не заменяет проверку provenance вызывающей стороной.

| Модель | Score policy | Направление | Решение |
|---|---|---|---|
| RF / HGB | `positive_class_probability_v1` | выше = больше аномальность | `score >= threshold` |
| IF | `negative_score_samples_v1` | выше = больше аномальность | `score >= threshold` |
| LSTM legacy | `window_mean_mse_v1` | выше = больше аномальность | `score >= threshold` |
| LSTM endpoint | `endpoint_mean_mse_v2` | выше = больше аномальность | `score >= threshold` |

Эти значения определяются точными историческими форматами; явные неизвестные
или несовместимые declarations отклоняются. `reconstruct` возвращает всё окно
при обоих LSTM score policies. CPU и inference batch size 128 совпадают с
прежними scorers; batch size обучения не подменяет inference batch size.
Версии зависимостей и CPU thread settings задаются прежним окружением replay.
Табличный путь не импортирует PyTorch. Metadata содержит scenario, scope,
нормализованный training policy, feature/score policies, profile, threshold и
исходные hashes. Объект spec неизменяем; metadata возвращается отдельной копией.

Bundle можно перенести целиком: исходные preparation/source paths не открываются
при загрузке. Hashes защищают от случайного изменения, а не подписывают
доверенность артефакта. Joblib остаётся форматом только доверенных локальных
моделей. Отсутствующие/повреждённые файлы не восстанавливаются автоматически.

## Приёмка и граница результата

Пункт 1 проверяет сохранность результата и функциональность, не улучшение
точности. Тесты покрывают все пять model/score вариантов, реальные bundles
существующего producer, все профили, inclusive threshold ties, reconstruction,
label-free builder, повреждение artifacts, неизвестные policies, несовместимые
batch, перенос bundle и табличный inference при запрещённом импорте PyTorch.

Исходный snapshot, логи, hashes, JSON-отчёт и incremental patch сохраняются
отдельно в игнорируемом `results/experiments/`. Patch применяется к snapshot
этого пункта; он не включает прежние незакоммиченные изменения. Общий commit
или push не создаётся. Данные/веса с SSD и опубликованный synthetic reference
остаются исходными. Полная диагностика выполнена в [пункте 2](s3-frozen-diagnostics.md).
Raw scenarios/CLI пункта 3 требуют следующего запроса.

Для будущего no-op raw recompute заранее сохраняются допуски: float32
`rtol=1e-6, atol=1e-7`; float64 `rtol=1e-12, atol=1e-12`, решения точные.
Изменённый пример имеет неизвестный ground truth; base label — provenance.

## Результат пункта 1 — 2026-11-14

- Исходные 77 tests прошли до изменений; полный набор после реализации —
  **87 tests**, включая 10 новых проверок. После расширения CERT negative cases
  и запрета импорта PyTorch также для CERT bundle все 10 проверок повторно прошли.
- Через новый API загружены **60 candidates** model-variants и **45 operating
  bundles** всех сохранённых folds/seeds. Это проверка совместимости формата.
- Calibration smoke: первые 128 inputs каждого из 20 model/score/history
  сочетаний и девять profiles трёх bundles seed42/fold1. Scores и решения
  совпали; legacy CSV прочитан с round-trip parser и исходным score dtype,
  operating NPZ сравнивался точно. Labels не передавались в inference.
  Полный assessment replay и диагностика остаются пунктом 2. Попыток доступа
  к assessment/test в этом smoke не было; отдельный audit hook закрывал их.
- Полный `run_all.sh` и `build_reference_snapshot.py` выполнены в отдельной
  временной копии. JSON метрик RF/IF/LSTM точно совпали с прежним reference;
  сгенерированный snapshot отличается только временем создания и environment.
- **1 321 исходный SSD файл** (веса, state, thresholds, contracts/configs и
  complete manifests) сохранил SHA-256. Прежний runtime/training/feature code
  и все исходные reference artifacts совпали с начальным snapshot.
- Compileall, shell syntax, Markdown links, AST-проверка отсутствия fit/
  threshold selection в inference и whitespace checks пройдены.

Вклад в качество — **проверка сохранности результата**. Новых research fits,
калибровки или CERT test-оценки нет; functional fixture/regression обучения
используют прежние протоколы и не объявляются улучшением качества.
