# S4 — разбор ошибок и первые улучшения моделей

Согласованный план, 14 ноября 2026 года. **Статус: запланирован; S4 ещё не
выполнялся.** [S3 завершён](s3-synthetic-scenarios.md). Этот документ сохраняет
утверждённые границы следующей работы; перечисленные experiments/CLI additions
пока не являются реализованными возможностями.

## Цель и границы

Приоритет — качество данных, признаков и моделей. S4 включает диагностику и
два последовательных ограниченных цикла улучшения. UI остаётся S5; более
широкий поиск/сравнение стратегий — S6. CERT raw what-if требует отдельного
reaggregation contract.

Контроль CERT: HGB rolling / primary FP1, precision52,23%, recall59,91%,
F1 0,558039, TP387/FP354/FN259 на196 765 primary-train development endpoints,
646 observable positives. Цель — FP≤1/1000 background account-days и recall≥50%.
Legacy F1-control сохраняется отдельно: precision59,34%, recall60,99%,
F1 0,601527, FP270/FN252. Профили не переназначаются по assessment.

Работа делится на семь пунктов, по одному за запрос. Перед каждым — snapshot,
после — отчёт, проверки и отдельный patch. Общий commit прежних изменений не
создаётся. Данные, weights, полные predictions и локальные review остаются вне Git.

## 1. Протокол и контрольные результаты

- Создать `configs/cert_s4_quality_v1.json` с hypotheses, источниками/hashes,
  populations, матрицей, seeds, budgets и selection rules до новых fits.
- Заморозить bindings существующих HGB/RF/IF/LSTM, признаков, inputs, thresholds
  и метрик. Сохранить RF rolling/frozen и обе LSTM score policies как controls.
- Новые эксперименты используют original primary train. Primary validation и
  уже использованный CERT test исключаются.
- Приёмка: controls воспроизводятся, sources сохранены, весь budget однозначен.

## 2. Разбор FP, FN и расхождений моделей

- Реализовать маршрут source events→account-day→history→features→exact model
  input→score/threshold/decision; отдельно model window и дополнительный context.
- Не смешивать source label, prediction и исследовательский комментарий.
- На каждом outer fold выбрать до5 FP с наибольшим score,5 FN с наименьшим,
  5 TP равномерно по упорядоченным scores и5 TN ближе всего к threshold.
  Сначала брать разные incident-connected groups, затем детерминированно
  заполнять остаток; ties — item ID. Добавить все32 positives scenario3,
  deduplicate: максимум92 подробных случая.
- Полные metrics считать на всей популяции, отдельно от diagnostic sample.
  Сопоставлять сохранённые модели на exact одинаковых items.
- Проверить концентрацию FP, sufficient-history cases, различия calibration/
  assessment и признаки найденных/пропущенных аномалий.
- Приёмка: JSON/читаемый отчёт, exact inputs/membership, проверяемые source links;
  факты, возможные объяснения и hypotheses разделены.

## 3. Нормальные ориентиры и residuals

- Reference pool — только permitted train соответствующей preparation;
  background у всех window members и персональных historical dependencies.
  CERT background означает отсутствие observable positive по текущей policy.
- Exact search по flattened standardized windows длины5, только внутри одной
  schema/preparation. RMS distance; максимум5 результатов. Исключить self,
  точные numeric duplicates, overlap source events с query и overlap members
  между возвращаемыми references.
- Similarity cutoff:95-й percentile межгрупповых train nearest distances на
  максимум256 детерминированных anchors. При недостаточной поддержке — явное
  отсутствие надёжного reference.
- LSTM: reconstruction, squared residuals по времени/группам features;
  aggregation соответствует window/endpoint score. Для HGB/RF показываются
  различия feature values с references, без заявления о причине инцидента.
- Приёмка: отсутствие train/assessment leakage, exact search oracle, корректный
  no-match, score/residual agreement.

## 4. Признаки контекста и механизм сравнения

Отдельный feature variant HGB добавляет четыре end-of-day отношения:

1. `file_off_hours_events / file_events`;
2. `device_off_hours_events / device_events`;
3. `email_external_message_count / email_events`;
4. `email_external_recipient_count / (email_internal_recipient_count + email_external_recipient_count)`.

Нулевой знаменатель даёт0. Scaling fit-only; прежние common features совпадают
точно. Идентификаторы/labels не становятся inputs.

Реализовать fit→calibration→inner selection→outer assessment, immutable
selections, failure/cost accounting и frozen operating profiles для LSTM.
Preprocessing reuse допустим только при identical fit population/feature policy.
CLI additions: `investigate`, `references`, `quality-study`; versioned JSON,
source/model/state hashes, IDs, policies и evidence scope. Поведение S3 сохраняется.

Приёмка — formula/schema/leakage tests на малых fabricated fixtures до research fits.

## 5. Цикл A: HGB

| ID | Вариант |
|---|---|
| H0 | Существующий HGB rolling |
| H1 | H0 с L2=10 вместо1 |
| H2 | H0 с L2=100 |
| H3 | H0 с min_samples_leaf=100 вместо20 |
| H4 | H0 + четыре context ratios |
| H5 | H1 + четыре context ratios |

Остальные параметры: learning_rate0,05, max_iter200, depth4,15 leaves,
class_weight=balanced, early_stopping=false. Внутреннее сравнение — seed42;
конфигурация выбирается отдельно в каждом outer fold по inner selection.
Итоговые seeds:7/21/42/84/168. Exact прежние fits переиспользуются по hashes.

Budget: максимум54 inner fits +15 final fits =69 новых обучений; до4 часов
вычислительного запуска. Сохранять отрицательные эффекты и совпавшие scores.

## 6. Цикл B: RF и LSTM

**RF:** frozen file/USB policy; baseline против min_samples_leaf20, остальные
параметры сохранены. Inner selection и пять final seeds как у HGB.
Максимум18 inner +15 final =33 новых обучения.

**LSTM:** endpoint hidden32 против hidden16; length5, Adam0,001, batch128,
max10epochs/patience5, full-window training MSE и прежний F1 checkpoint rule.
Seeds7/21/42; прежние hidden32/seed42 reuse. Максимум6 недостающих controls
+9 новых variants =15 обучений. Старый window score остаётся comparison.

После freeze checkpoint — общий FP1 profile; legacy F1/FP2 отдельно.
Budget: максимум48 новых fits при согласованном потолке51; до8 часов запуска.
Матрица циклаB фиксируется до просмотра новых outer results циклаA.

## 7. Итог и следующий quality cycle

- Paired таблицы control→variant→delta на одинаковых endpoints: P/R/F1,
  within-fold PR-AUC, TP/FP/FN/TN, FP/1000, support/accounts/groups/scenarios.
- Раздельные seed/fold variation; train/feature/inference time и peak RSS.
- Повторный разбор fixed cases: исправленные и новые ошибки. Итог каждого
  варианта — улучшение, компромисс, отсутствие эффекта или incomplete.
- Следующий bounded quality protocol S6 основан на результатах; S5 UI не
  является условием начала quality experiments.

## Общий протокол и приёмка

Сохранить3 outer folds. Внутри train+calibration каждого —5 incident-group
parts, seeds343/344/345. Три rotations: selection часть i, calibration следующая,
fit остальные3. Предварительно проверено: все9 combinations имеют оба класса.
Все fitted state/model используют только fit.

Calibration: maximum recall под integer FP1 budget, ties lower FP/higher
threshold. Minimum recall infeasible сохраняется без fallback. Inner ranking:
число parts с обоими условиями; число parts в budget; lower max FP/1000;
higher mean recall; higher mean PR-AUC; fixed variant order. Outer не выбирает
параметры, threshold или дополнительный trial.

Матрица — максимум117 новых research fits при общем потолке120; дополнительные
попытки не планируются. Failed/interrupted входят в budget. До12 часов
вычислений суммарно, один CPU worker/thread1, без GPU; минимум30GiB свободно.
Исчерпание лимита означает явный incomplete без самовольного расширения.

Приёмка: provenance/group/causality/reference/residual tests, полный suite,
compileall/shell/whitespace, synthetic regression во временной копии и
сохранность historical artifacts. Оба operational условия должны выполняться
во всех3 outer folds; частичный успех и tradeoffs отображаются отдельно.

Не рассчитывать pooled cross-fold PR-AUC. Model seeds не создают независимых
данных; scenario3 имеет одну независимую group. S4 использует уже исследованные
development данные и не является независимым подтверждением. Для него нужен
отдельно принятый compatible source с entity/duplicate/episode audit.
