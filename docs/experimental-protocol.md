# Experimental Protocol — v1 Baseline and Planned Spectra Rules

Implementation update 2026-11-14: the [isolated experiment workflow](isolated-experiments.md) provides synthetic/CERT preparation IDs, exact membership, saved inputs/models, validation-only candidates and explicit final evaluation. [The first frozen CERT final](cert-r4.2-final-evaluation.md) is complete; its test is used. [Causal history v3](cert-r4.2-causal-history.md), [paired RF/IF/LSTM group-CV](cert-r4.2-group-cv-v3.md), [file/USB ablation](cert-r4.2-file-usb-v4.md) and [endpoint/HGB comparison](cert-r4.2-model-variants-v5.md) are complete on development data. Descriptions below of shared root outputs and mandatory test access apply to legacy scripts; Spectra remains future work.

Status: fixed for the `argus-v0.2.0` synthetic baseline and the initial public
dataset implementations.

Integration notes updated 2026-10-27. Core v1 reference metrics/selection rules
are unchanged. The Spectra extension below governs future workflows; it does
not claim that candidate, search or scenario modes are implemented today.

This protocol defines how Argus creates splits, selects configurations and
thresholds, evaluates models, and records artifacts. Deviations are allowed
only when a scenario makes a rule impossible; the deviation and its rationale
must be committed before inspecting final test metrics.

## Research questions

Argus evaluates four questions:

1. How do supervised tabular, unsupervised tabular, and sequence-based anomaly
   detection methods compare under the same scenario split?
2. Which anomaly families and behavioral patterns cause false positives and
   false negatives?
3. How stable are model rankings across seeds and across datasets?
4. Which explanations are supported by measured features rather than dataset
   identifiers or capture shortcuts?

The goal is a defensible comparison, not a requirement that the neural model
must rank first.

## Split invariants

Every record receives exactly one immutable split label: `train`, `validation`,
or `test`.

- preprocessing state is fitted on `train` only;
- hyperparameters, early stopping, and decision thresholds use `validation`;
- `test` is evaluated only after a run configuration is frozen;
- no oversampling, undersampling, aggregation, scaling, encoding, or feature
  selection is fitted before the split;
- exact duplicates and source-derived groups must not cross split boundaries;
- sequence windows cannot cross a boundary;
- multiple events from one entity are kept together when the scenario uses an
  entity-group split;
- timestamps, row numbers, source file names, capture days, user IDs, IPs, and
  answer-key fields are provenance unless explicitly approved as features;
- every split report records size, prevalence, class support, time range,
  entity overlap, source files, and anomaly-family distribution.

If a split has no positive or no negative examples, ROC-AUC and PR-AUC are
reported as undefined. The split must not be silently reshuffled after seeing
test results.

## Scenario split policies

### Synthetic

Primary split: chronological `60/20/20` over sorted events.

- first 60%: train;
- next 20%: validation;
- final 20%: test.

Encoders and scalers are fitted on the train period. Sequence windows are built
separately inside each period and per user. The generator must be deterministic
for a fixed seed and distribute each configured anomaly family across the time
horizon; the split validation fails otherwise.

This replaces random re-splitting in individual training scripts.

For cross-model comparison, a sequence target is the label of its endpoint
event. Tabular models are evaluated on the same endpoint rows (the first
`sequence_length - 1` events of each user/split are excluded from the comparison
cohort). A separate any-anomaly-in-window target would be a different task and
must not be mixed into the main table.

### CERT r4.2

CERT amendment 1.1 (2026-11-12): primary split unit is a connected group of
accounts linked by an official incident; unrelated users form singleton groups.
This preserves no-user-overlap and prevents one cross-account incident from
crossing splits. Target approximately `60/20/20` users with seed 42, balancing
positive-account and scenario support without breaking connected groups.
The allocation algorithm and assignment manifest must be frozen before training.
All events and user-days for an account remain in one split, ordered in time.

The fixed target is official observable membership aggregated by logged account
and calendar day, not blanket malicious-user/interval labeling. Exact rules,
negative-label limitations, scenario 3 handling and checked group counts are in
[CERT label policy v1](cert-r4.2-label-policy.md). Synthetic v1 is unchanged.

A secondary temporal experiment may split each eligible user's history into
past/present/future periods, but it must be labeled `temporal-with-user-overlap`
and cannot replace the no-user-overlap primary result.

CERT history amendment v3 (2026-11-14): under the separately named
`cert_frozen_global_causal_all_v3` policy, personal state may consume past
observations of an unseen account, emitting the current day's features before
adding that day. No label or attribution controls the update. Global reference
and scaler fit remain train-only (inner-fit-only in CV). This is offline account
transfer with causal personal history, not temporal forecasting. Existing v1/v2
policies and frozen results remain unchanged. `primary_development` preparation
contains original train/validation only and cannot be used by final evaluation.

CERT file/USB amendment v4 (2026-11-14): the separately declared
[step-3 ablation](cert-r4.2-file-usb-v4.md) adds three generic log-volume
deviations and a profile-fallback flag to v3. It compares a causal 30-day
personal reference against a frozen reference from the first five observed
active days. Neither reference removes days using true labels. Frozen early
profiles are not backfilled into earlier rows; global fallback/scaling remain
inner-fit-only. Nine completed v3 candidates are reused as immutable controls;
the same development folds do not provide independent confirmation. Previous
observable-positive history may stratify a report after all inference, but
cannot enter features, history filtering or selection.

CERT model/score amendment v5 (2026-11-14): [increment 4](cert-r4.2-model-variants-v5.md)
separately compares endpoint-only LSTM reconstruction error and one fixed
HistGradientBoostingClassifier. The LSTM optimization loss remains full-window
MSE; epoch/threshold selection uses only inner calibration with its named score.
HGB fits the same labeled inner-fit days as RF, with fixed iteration budget and
no automatic random validation split. It has more label information in model
fit than IF/LSTM and is not an equal-supervision replacement claim. All four
existing feature policies, groups, targets and endpoints are preserved; no
test or outer-calibrated threshold is introduced. Defaults and frozen legacy
scorers remain unchanged; new policies use explicit configuration dispatch.

CERT operating-point amendment v6 (2026-11-14): [increment 5](cert-r4.2-operating-point-v6.md)
freezes HGB rolling, matched RF rolling and the previous strongest RF frozen
before five-seed repetition on the same development folds. The named primary
objective maximizes calibration recall under at most one FP per 1,000 background
active account-days, with a minimum recall of 50%. A two-FP sensitivity profile
is declared separately and cannot replace the primary profile after scoring.
All unique calibration-score decision boundaries plus reject-all form the
finite threshold grid; tied scores cannot be split. Integer FP budgets use
floor, with lower FP and higher threshold as recall tie-breakers. Infeasible
recall is recorded without relaxing the budget. Existing F1-selected thresholds
remain immutable controls, not silently redefined defaults. New operating
bundles are sealed before outer access. Outer budget violations do not trigger
selection; model seeds do not create independent groups or holdouts. No new
test scoring is performed because independent accepted CERT data are unavailable.

### CICIDS2017

Primary split unit: capture day.

- train: Monday, Tuesday, and Wednesday;
- validation: Thursday;
- test: Friday.

This deliberately evaluates temporal and attack-family shift. Monday provides
benign traffic while Tuesday and Wednesday add supervised attack examples. A
secondary stratified comparison may be reported, but must preserve flow groups
and must be clearly labeled `stratified-comparability`.

### CSE-CIC-IDS2018

Primary split unit: capture day/source file. After the publisher manifest is
accepted, ordered capture days are assigned approximately `60/20/20` without
splitting a day. Exact dates are frozen in the scenario config before model
tuning.

Two evaluations are required:

1. within-dataset training/validation/testing under the frozen daily split;
2. transfer evaluation using the compatible CICIDS2017 feature subset and a
   frozen CICIDS2017 model/threshold where technically possible.

## Training roles

### Supervised models

`Random Forest` and the later selected boosting baseline train on labeled train
records. Hyperparameters and score thresholds are selected on validation.

### Unsupervised and reconstruction models

`Isolation Forest` and `LSTM-Autoencoder` fit model parameters on normal train
records only. Labeled validation data may calibrate a decision threshold; this
is reported as supervised threshold calibration, not as a fully unsupervised
experiment.

An optional label-free threshold can be reported separately when it is derived
only from normal train/validation score quantiles.

## Configuration selection

Development runs use primary seed `42`. Final selected configurations are
repeated with seeds:

```text
7, 21, 42, 84, 168
```

The final table reports mean, standard deviation, minimum, and maximum for
stochastic metrics. Deterministic models may produce identical values; all
runs are still recorded.

Hyperparameter search must be bounded and declared in a committed config. The
selection objective is validation F1 for the anomaly class, with tie-breakers:

1. higher recall;
2. lower false-positive rate;
3. simpler or faster configuration;
4. lower threshold only when all preceding values tie.

The test split is not used to choose a model, epoch, feature set, seed, or
threshold.

## Threshold selection

For every model that produces a continuous score:

1. generate candidate thresholds from validation-score quantiles or a declared
   finite grid;
2. calculate validation metrics at every candidate;
3. apply the fixed selection objective and tie-breakers;
4. freeze the selected numeric threshold;
5. apply it once to test scores.

The candidate grid, selected threshold, validation metrics, and tie-break
outcome are stored. A default `0.5` classifier threshold is allowed only as a
separately identified reference result.

## Required metrics

Per model and split, record:

- anomaly precision;
- anomaly recall;
- anomaly F1;
- PR-AUC / average precision;
- ROC-AUC;
- false-positive rate: `FP / (FP + TN)`;
- confusion matrix and class support;
- anomaly prevalence;
- inference wall time and records per second.

Accuracy may appear as a secondary descriptive value, but never as the primary
selection metric. Per-family recall and false-negative counts are required when
multiclass anomaly labels are available.

Inference timing uses a completed warm-up followed by at least five measured
runs on the same prepared test matrix. Reports include hardware, software
versions, batch size, and whether GPU acceleration was used. Training time is
recorded separately.

## Leakage and quality checks

Every run must pass or explicitly waive:

- split labels are exhaustive and mutually exclusive;
- train-fitted encoders/scalers contain no validation/test fit calls;
- no sequence spans a split boundary;
- duplicate keys do not cross splits;
- target/answer-key columns are absent from features;
- identifier and source-proxy columns are absent from default feature sets;
- all model features are numeric and finite after preprocessing;
- train/validation/test class distributions are reported;
- data-manifest hash and split-config hash match the run metadata.

## Artifact contract

Each run writes to:

```text
results/experiments/<scenario>/<run_id>/
├── resolved_config.json
├── environment.json
├── data_manifest.json
├── split_report.json
├── feature_manifest.json
├── threshold_search.csv
├── metrics_validation.json
├── metrics_test.json
├── timing.json
├── error_analysis.csv
└── plots/
```

Large predictions, model weights, and intermediate arrays remain ignored.
Release snapshots may contain resolved configs, metrics, manifests, compact
error examples, and plots.

`run_id` has the form:

```text
YYYYMMDDTHHMMSSZ_<model>_<8-character-config-hash>
```

Run metadata includes:

- UTC timestamp;
- Git commit and dirty-worktree flag;
- scenario and dataset variant;
- raw manifest SHA-256;
- resolved configuration and its SHA-256;
- Python and dependency versions;
- seed and device;
- split policy and feature-set name.

## Planned Spectra: Selection and Scenario Workflows

Scope lives in the [implementation plan](spectra-lab-implementation-plan.md).
EventLens is the former name of Event Investigation inside one Spectra Lab.

### Candidate studies and final evaluation

- Freeze preparation, target, cohort, search space, objective, budgets, seeds
  and benchmark conditions before trials. Retain v1 validation F1/tie-breakers
  initially; selection-changing FPR/resource constraints require a named revision.
- Candidates read train/validation only: no test loading, scoring or metrics.
  Current complete training scripts require this separation before search.
- Isolate trial inputs/models/outputs and record complete/pruned/failed/interrupted
  states, including consumed cost. Failed trials count against the budget.
- Select/freeze configurations before explicit final test evaluation. Never
  choose a winner from a final-test Pareto plot and call test untouched.
- Start with random search, then compare TPE on the same domain, compute budget,
  stopping/pruning rules, data and objectives. Repeat search seeds independently
  of model seeds; do not assume TPE wins.
- Freeze sequence length/feature set initially. Later length comparisons need
  a predeclared common population and explicit coverage differences.
- Repeated validation use can overfit selection. Any robustness folds remain
  within development data and preserve chronological/entity-group split rules.

### Quality and resource measurements

Compare quality on identical validation items. Search inference benchmarks use
a fixed non-test cohort and declared hardware/device, batch, threads, warm-up
and repeats. Report training, feature preparation, batch throughput, single-item
latency and process peak RSS separately. Small timing differences within
measurement variation are not evidence of superiority.

### Reference examples and raw scenarios

- Reference pools use permitted train objects only, with explicit normal and
  context policies. A normal endpoint alone does not establish a normal window.
- Exclude self, duplicates and overlapping members; no-match is valid.
  Similarity is computed outside 2D plot coordinates.
- A scenario binds a frozen model/state/schema/threshold and base object to a
  versioned raw patch. Recompute affected dependencies through the same builder,
  without fit or silent schema changes.
- No-op reproduces inputs/scores/decisions within declared tolerances. Validate
  optimized recompute against an independent full-history oracle.
- New ground truth is unknown unless a separately declared labeling mechanism
  determines it; base labels remain provenance only.
- Scenario results and notes do not alter base data, labels, predictions,
  study ranking or final-test metrics.
- Distinguish facts, residual contributions and response to a patch. These
  outputs do not establish real-world causality.
- CERT user-day changes require source-field support and raw-event reaggregation;
  missing history or unsupported fields cause explicit refusal.

### Statistical independence and UI studies

Predeclare cases and metrics. Group overlapping windows by entity/episode for
uncertainty estimates; repeated seeds do not create independent datasets. UI
studies use comparable information, equivalent tasks, balanced order and
participant-level analysis, not independent observations per click. Describe
pilot participants accurately rather than calling students SOC professionals.

## Reporting rules

- Never describe synthetic metrics as production performance.
- Name the exact original/corrected dataset variant.
- Report undefined metrics as `null`/`N/A`, not zero.
- Include negative findings and limitations.
- Investigate near-perfect scores for duplicate leakage, identifiers, capture
  shortcuts, or label artifacts.
- Compare models only when their test records and target definition match.
- Freeze a release only after its documented reproduction command succeeds from
  a clean checkout with the required data manifest.

### CERT integration implementation update — 2026-11-13

All three CERT preparation increments are now implemented; see the
[exact transformation/history/window protocol](cert-r4.2-experiments.md).
The new preparation has 71 inputs fitted/transformed with train-only state.
Personal rolling/novelty history is causal within train; held-out users use
frozen global fallbacks with explicit cold-start flags. Calendar gaps are
retained, and windows span five active days. CERT LSTM uses all-background
training windows; synthetic retains its historical endpoint-normal policy.
Full CERT preparation and a small IF validation-only smoke candidate completed;
real test evaluation and research model comparisons remain future work.

### CERT robustness amendment — 2026-11-13

The [history-control protocol](cert-r4.2-robustness.md) adds one explicitly named
global-only policy alongside the unchanged v1 control. Both use five fixed
model seeds. Three outer incident-group folds within primary train additionally
check LSTM transfer; each outer fit has a separate inner group calibration set.
Feature state and weights fit only inner train, checkpoint/threshold select on
inner calibration, and frozen assessment uses outer fold only. Primary
validation and real test do not enter these development folds. This is a
bounded control experiment; no automatic parameter search, test-driven tuning
or best-seed selection is introduced. CV cannot repair scenario 3's limited
independent-group support.

### CERT final exposure — 2026-11-13

The [frozen final protocol](cert-r4.2-final-evaluation.md) completed all 30
predeclared candidates on the aligned real CERT test. Numeric thresholds,
epochs, weights and feature states were unchanged; source/dependency and
artifact bindings were verified before test access. Current test is now used.
It remains valid evidence for these frozen configurations and reproducibility
checks. New configurations require separate development validation and an
independent holdout for a new confirmatory claim; old-test re-evaluation must
be labelled exploratory. No new model is selected by its test ranking.
