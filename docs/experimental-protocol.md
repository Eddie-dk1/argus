# Experimental Protocol v1

Status: fixed for the `argus-v0.2.0` synthetic baseline and the initial public
dataset implementations.

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

Primary split unit: user.

Users are assigned `60/20/20` with a fixed seeded group allocation stratified
at user level by whether the official answer key marks any malicious activity.
All events and user-days for a user remain in one split. Events remain ordered
inside each user.

A secondary temporal experiment may split each eligible user's history into
past/present/future periods, but it must be labeled `temporal-with-user-overlap`
and cannot replace the no-user-overlap primary result.

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
