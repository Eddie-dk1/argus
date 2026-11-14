# Roadmap — Argus / Spectra Lab

Updated: 2026-11-14. This is the single chronological roadmap. The [Spectra implementation plan](spectra-lab-implementation-plan.md) defines product scope and code changes; the [experimental protocol](experimental-protocol.md) defines evaluation rules.

## Current state

S0–S3 are complete within their documented scope: accepted CERT data/labels,
isolated preparation/candidate/final artifacts, five controlled quality
increments and the common frozen runtime. The [S3 reports](s3-frozen-inference.md)
cover exact saved-result replay, quality diagnosis and verified synthetic raw
scenarios. CERT raw what-if remains a separate reaggregation task.

The immediate next work is the [seven-part S4 quality plan](s4-model-quality-plan.md),
**approved but not started**. Error investigation and normal references lead
into bounded HGB context/regularization experiments and an RF/LSTM comparison.
Model quality takes priority over the S5 interface. The declared matrix has at
most 117 new fits, ceiling 120 and up to 12 hours of experiment execution.

Current primary HGB rolling development result: precision 52.23%, recall 59.91%,
F1 0.558039, TP387/FP354/FN259. The target FP≤1/1,000 and recall≥50% fails
jointly on all three outer folds. The separate legacy F1-control remains
P59.34%/R60.99%/F1 0.601527, FP270/FN252. Mean within-fold PR-AUC is 0.705642;
rankings are not pooled across folds. HGB seeds have identical scores.

All new quality results remain reused development evidence. The CERT test was
consumed on 2026-11-13; improved-model confirmation needs separately accepted
independent data. Network datasets remain separate milestones. See the
[stage assessment](cert-r4.2-stage-assessment.md) and the historical log below.

Completed in argus-v0.1.0 / argus-v0.2.0:
- repository foundation, deterministic synthetic generation and causal features;
- RF, IF and LSTM-AE on an aligned synthetic endpoint cohort;
- validation-selected thresholds/epoch, metrics, timing and compact reference;
- public-dataset research, schema fixtures, inspection and external-storage checks.

Still pending: the S4 improvement cycles, independent quality confirmation,
unified Spectra UI, CERT raw reaggregation and automated configuration studies. New isolated candidates
preserve their own inputs/models/predictions; legacy root outputs can still be
overwritten by legacy commands.

Argus is one application. Spectra Lab is its only research workspace: «Обзор», «Подбор моделей», «Разбор событий», «Проверка сценариев». EventLens is a previous name, not another module.

The [current design](design.md) is minimalist dark navy with neutral branding,
controls and selected states; red is reserved for actual problems/anomalies.
Archived red-heavy concepts are not implementation requirements.

<a id="resume-after-ssd"></a>
## Точка продолжения — S4 и качество моделей

Текущий порядок — [семь пунктов S4](s4-model-quality-plan.md), затем S5 UI и
более широкий S6 quality/search cycle. S4 выполняется по одному пункту за
запрос; ни один из новых research fits ещё не запускался.

### Исходная последовательность после SSD — историческая справка

Пункты1–4 ниже уже выполнены для CERT; frozen/synthetic часть пункта5 закрыта
в S3. Текст сохраняет исходный порядок, а текущие изменения S4 определены
отдельным согласованным планом. UI, расширенный поиск и поставка ещё впереди.

### 1. Подключить и проверить хранилище

Выбрать реальный существующий mount и data root по [storage policy](data-storage.md). Бюджет рассчитать по manifest выбранных файлов, включая распаковку, features, models, runs и temp. Не скачивать всю коллекцию.

Существующая команда проверки: python3 src/check_data_storage.py. Перед скачиванием передать рассчитанный --required-gib; каталоги сценария создавать только для выбранного источника.

### 2. Принять первый публичный источник

Начать с CERT r4.2 и соответствующих answers. Сохранить bundled README, license, SHA-256, schema и label mapping. Проверить соответствие предварительных contracts. Существующая команда инспекции: python3 src/inspect_dataset.py --scenario cert_r4_2.

CERT — публичная синтетическая коллекция, а не production telemetry. Не скачивать CICIDS2017/CSE одновременно с первым источником.

### 3. Подготовить код к независимым экспериментам

Начать с small synthetic fixtures:
- IDs до сортировки, item/source mapping, exact sequence membership;
- immutable feature state/schema, per-run models и full predictions;
- separate train/validate-only candidate и explicit final test;
- isolated data/results roots, experiment/study manifests и complete status.

Главные точки: src/features.py, src/preprocessing.py, src/experiment_tracking.py, training scripts и src/evaluate.py. Полная карта изменений — [разделы 4 и 10 плана](spectra-lab-implementation-plan.md#4-текущий-код-и-разрывы).

### 4. Получить первый CERT baseline

Реализовать user-day aggregation с item→raw events, frozen user-group split, train-fitted state и cold-start flags. Сначала табличные baselines, sequence модель только после точного определения входа/цели/coverage.

Group split и temporal-with-user-overlap — разные эксперименты. При новых пользователях нельзя fit personal profiles на validation/test. Заморозить configuration до final evaluation.

### 5. Сделать первый рабочий маршрут Spectra

На фиксированной synthetic LSTM проверить reload/no-op и bytes_sent patch с пересчётом зависимостей. Затем разбор исходных записей/входов, normal references, проверка сценариев и один UI.

CERT сначала получает просмотр и сравнение доступных результатов. CERT what-if требует отдельной source-field policy и повторной user-day aggregation; synthetic bytes_sent не переносится в CERT автоматически.

### 6. Добавить «Подбор моделей»

После стабильных artifacts и fixed-model workflow: bounded random search, затем TPE, validation F1 и одинаковые условия измерений времени/памяти. Fixed search space/budget/cohort; candidate не читает test. Выбор замораживается до final evaluation.

### 7. Проверить и передать результат

Offline view bundle, installation guide, meaningful tests, timings и эксперимент статьи. Карта/UMAP добавляется после основного маршрута, а не заменяет его.

## Data and release milestones

These remain intended milestones, not completed release claims. Internal Spectra work packages do not create or renumber releases automatically.

### argus-v0.3.0 — CERT r4.2

Accepted manifest/labels, user-group split, user-day source mapping, baselines, justified sequence representation, multi-seed evidence and compact release snapshot. Shared artifact foundations may land incrementally before this release.

### argus-v0.4.0 — CICIDS2017

Validate publisher CSVs, quality/corrections, frozen capture-day split, flow semantics and attack-family errors. Missing host/time fields disable unsupported context functions.

### argus-v0.5.0 — CSE-CIC-IDS2018 and transfer

Independent accepted source and within-dataset run; compatible feature transfer from CICIDS2017 with frozen model/threshold. Report failed transfer as evidence, not hidden retuning.

### argus-v0.6.0 — selected models and research evidence

Bounded configuration studies, quality/resource comparison, multi-seed selection, scenario sensitivity analysis and supported explanations. One additional model or method is added only when it answers a declared question.

### argus-v1.0.0 — diploma integration

One Spectra Lab UI, reproducible selected scenarios, article evidence, final tables, limitations and demo. The former separate Streamlit demonstration is superseded.

## Minimum and extensions

The first delivered laboratory requires synthetic correctness tests, one accepted public behavioral scenario, investigation/scenario workflow and bounded model selection. The network scenarios remain separate diploma milestones; they are not a prerequisite for the first UI. Any reduction of final diploma dataset scope must be declared explicitly before final experiments.

No new products for Live, Wave, causal discovery, general AutoML or Apple-specific runtime optimization. Hardware acceleration requires a measured need.

## Release gate

A milestone is complete only when code, contracts, data/config, tests, actual metrics, docs and release notes agree. Verify regression/contract tests, syntax, a reproducible reference command and diff whitespace; keep datasets, weights, private paths and large outputs out of Git.

Historical release notes and reference artifacts are not rewritten to describe future work.

### Artifact increment — 2026-11-12

Training scripts now preserve a run-local `inference/` bundle containing model
weights (including RF/IF), full test predictions, feature state and an inference
contract with feature order/threshold. `complete.json` records file hashes only
after successful serialization. Run IDs include a random suffix. This is the
first storage increment, not completion of S1: legacy root outputs still update,
concurrent training is unsupported, exact input/member IDs, immutable preparation,
external result roots and train/validate-only execution remain to implement.
Tests cover independent bundles, sklearn reload, overwrite rejection and failed
serialization. No CERT training or reference-metric regeneration was performed.

### Independent experiment foundation — completed 2026-11-12

The new [isolated workflow](isolated-experiments.md) completes the synthetic
foundation: versioned preparation/event IDs, exact ordered sequence membership,
actual model inputs, frozen state/schema, separate validation-only candidates
and explicit frozen final evaluations, independent artifact roots, checksums,
status and exact cohort comparison. All three model families are supported.
Legacy reference commands remain compatibility-only and are not study runners.
Next active stage: CERT connected-group allocation and user-day preparation,
then its first tabular baseline. This does not mark Spectra jobs, CERT adapters,
search, or UI as implemented.

### CERT preparation — three increments (2026-11-13)

1. **Connected-account assignment:** deterministic allocator, exclusive-output CLI,
   provenance hashes and invariant tests implemented and run on the accepted SSD
   snapshot: 600/200/200 accounts, no group overlap. Scenario 3 has only two
   independent groups: train/test coverage only; validation support is absent
   and explicitly recorded in the protocol. Assignment saved on SSD.
2. **Raw → active account-days — completed:** [daily preparation v1](cert-r4.2-day-preparation.md)
   ran on all 32,770,222 events, producing 330,452 active days, 59 numeric
   aggregates, exact source membership and 986 positive days. Raw hashes,
   all 7,323 observable rows, source-qualified unique IDs and frozen assignments
   checked. Artifacts are stored on SSD; no model training performed.
3. **Experiment integration — completed:** [CERT preparation v1](cert-r4.2-experiments.md)
   freezes train-only global state/scaling, causal train histories and explicit
   held-out cold-start. All model families share five-active-day endpoint cohorts;
   CERT LSTM trains only on all-background windows. Full preparation and a small
   IF train/validation smoke candidate are saved on SSD; real test not evaluated.

All three preparation increments are complete. The next stage is controlled CERT model experiments;
model investigation, Spectra UI/jobs, bounded search, network datasets and final
research deliverables remain subsequent milestones.

### Controlled CERT comparison — completed 2026-11-13

The [fixed benchmark plan](cert-r4.2-benchmark.md) divides the remaining S2
experiment work into four sequential increments: RF/IF validation baselines,
LSTM validation baseline, five-seed robustness, frozen final evaluation/report.
Increment 1 is complete: both 200-tree baselines ran with seed 42 on the same
64,254 validation endpoints, with saved inputs/models, reload and checksum
checks, and separate fit/inference measurements. Validation F1: RF 0.250304,
IF 0.012413. Increment 2 is also complete: LSTM seed 42, 195,756 background
windows, selected epoch 1, validation F1 0.008174, no real test access.
[Quality diagnostics](cert-r4.2-lstm-baseline.md) identified history-availability
shift. [Increment 3 is complete](cert-r4.2-robustness.md): both policies across
five model seeds, nested-calibration group-CV for LSTM, immutable candidate
selection records and unchanged real test. Global-only mean validation F1:
RF 0.257842, IF 0.014216, LSTM 0.020345; precision/FPR improved for RF/LSTM
with lower recall. Dropout/weight decay were not added. Increment 4 is now
complete: 30 frozen final evaluations, exact cohort/threshold checks, scenario
errors and timings. Global-only mean test F1: RF 0.186320, IF 0.009694,
LSTM 0.012889. RF mean recall is only 16.5%; LSTM misses almost all scenario-2
positive days. All four experimental increments are complete, with 49 passing
tests; these findings do not mark release v0.3.0 or production readiness.
This work does not start S3–S8 or the automated configuration search of S6.

### CERT quality improvements — increment 1 completed 2026-11-14

[Causal history v3](cert-r4.2-causal-history.md) restores the same personal
past-observation access in every split, with frozen train global fallback and
train-fitted scaling. The isolated preparation excludes test before feature
building, preserves the same 71 inputs/endpoint population and records exact
history dependencies. All 56 tests pass. No new CERT model fitting, threshold
selection or test scoring occurred. At that point four quality increments
remained; this is not a production-readiness claim or completion of the whole
improvement stage.

### CERT quality improvements — increment 2 completed 2026-11-14

[Paired group-CV v2/v3](cert-r4.2-group-cv-v3.md) completes 18 candidates across
three grouped outer folds, RF/IF/LSTM and seed 42. Every profile/vocabulary/scaler
is refit on inner fit; thresholds/epochs use inner calibration only. RF mean
outer F1 improves 0.226609 → 0.234597 and PR-AUC 0.231314 → 0.274137, with more FP.
IF worsens; LSTM recall does not improve. All folds and negative results remain
reported. All 61 tests pass. Original primary validation/test were not scored;
existing development folds are reused and do not become independent holdouts.
No source-specific feature or architecture change was included in this
comparison. Increment 3 is recorded separately below.

### CERT quality improvements — increment 3 completed 2026-11-14

[Generic file/USB deviations](cert-r4.2-file-usb-v4.md) add three past-only
log-volume deviations and a fallback flag to the same 71 common features.
Six preparations and every numerical reference dependency are audited before
fitting. Rolling versus frozen-first-five personal references use identical
labels/groups/endpoints; positive history is never cleaned with labels.
All 27 assessments complete (18 new fits, nine immutable v3 controls).
RF frozen mean outer F1/PR-AUC improve to 0.291851/0.304638; pooled precision
16.56% → 22.05%, recall 38.54% → 39.63%, FP 1,255 → 905. Rolling RF has more
recall and FP, including a fold F1 regression. IF gains are costly in FP and
LSTM detection remains weak. All 67 tests and isolated synthetic regression
pass; 48 older checkpoints and all prerequisite manifests remain unchanged.
Primary validation/test are not scored. At completion of increment 3,
increments 4–5 remained pending; no new architecture, score, threshold objective
or independent-data claim was included in that feature comparison.

### CERT quality improvements — increment 4 completed 2026-11-14

[Endpoint score and fixed HGB](cert-r4.2-model-variants-v5.md) complete two
independent hypotheses. All 12 existing preparations and 36 frozen controls
are reused; 12 LSTM candidates change only reconstruction score/inner epoch
selection, 12 HGB candidates use one fixed supervised configuration. Existing
labels/groups/cohort and reconstruction optimization loss remain unchanged.
HGB beats RF F1/PR-AUC in all 12 paired policy/fold cases; rolling HGB gives
mean F1 0.621700, PR-AUC 0.705642, pooled TP394/FP270/FN252. Variation remains
large, scenario 3 is weak, and LSTM endpoint recall is worse despite fewer FP.
HGB is a justified primary supervised candidate; LSTM is retained as a baseline.
All 72 tests, isolated synthetic regression and exact reload checks for all
60 assessments pass; 66 older checkpoints and frozen-final inference guards
remain intact. No primary validation/test scoring or independent claim is
added. At completion of increment 4, increment 5 remained pending; it is
recorded separately below.

### CERT quality improvements — increment 5 completed 2026-11-14

[Operating point and five seeds](cert-r4.2-operating-point-v6.md) freeze HGB
rolling, matched RF rolling and prior RF frozen. All 45 jobs/135 decision
profiles complete: 36 new fits and nine unchanged seed42 checkpoints. Primary
recall maximization under calibration FP1/1,000 with recall≥50% does not
transfer reliably to outer accounts. HGB primary pooled precision 52.23%,
recall 59.91%, F1 0.558039, FP354 (actual1.805/1,000); no outer fold satisfies
both conditions. All HGB seed score arrays are exactly equal, while RF varies;
no best seed or sensitivity profile is promoted. Independent accepted CERT
data are unavailable, so no new test scoring/confirmatory claim is made.

All 77 tests and isolated synthetic regression pass. Independent replay of
45 models/135 profiles reproduces exact inputs/members/scores/decisions and
the same calibration thresholds via a separate ROC implementation. All 97
previous checkpoints and 331 prior manifest/weight hashes are unchanged.
The five quality increments are complete within their declared development
scope; this does not complete a deployment, release or independent validation.
The [assessment](cert-r4.2-stage-assessment.md) records S0–S2 progress and the
remaining S3–S8 work. Its preliminary four-task suggestion is superseded by
the [agreed three-part S3 plan](s3-frozen-inference.md): permanent rules/common
frozen inference, saved-result diagnosis, then verified synthetic scenarios/CLI.
Parts 1–2 implement the shared API and complete saved-result replay/diagnosis.
[Part 3](s3-synthetic-scenarios.md) now completes verified synthetic scenarios
and `argus.py inspect/replay/diagnose/scenario`; S3 is complete. The next
implementation stage is the [expanded S4](s4-model-quality-plan.md): event investigation,
train-only normal references and two bounded quality-improvement cycles.
Quality hypotheses from diagnosis require their own bounded protocols in the
later improvement cycle. Independent improved-CERT confirmation remains open.
