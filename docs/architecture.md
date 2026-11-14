# Architecture

Implementation update 2026-11-14: the [isolated experiment workflow](isolated-experiments.md) provides synthetic/CERT preparations, exact membership, saved inputs/models, validation-only candidates and explicit final evaluation. CERT frozen final and quality increments 1–4 are complete; [endpoint/HGB variants](cert-r4.2-model-variants-v5.md) are explicit optional extensions. Descriptions below of shared root outputs and mandatory test access apply to legacy scripts; dated progress notes retain their original stage scope. Spectra remains future work.

Status: implemented synthetic research pipeline plus public-dataset preparation
contracts for `argus-v0.2.0`.

Target architecture updated 2026-10-27. The inventory below describes current
code; the planned Spectra section describes future changes.

## Data Flow

```text
Config + data root
       |
       +--> synthetic generator ------------+
       |                                    |
       +--> public raw files -> inspection -+
                                            v
                                  immutable scenario raw data
                                            |
                                            v
                                  split before fitted state
                                            |
                         +------------------+------------------+
                         |                                     |
                         v                                     v
                 tabular features                       sequence features
                         |                                     |
                +--------+--------+                            |
                |                 |                            |
                v                 v                            v
          Random Forest    Isolation Forest             LSTM Autoencoder
                |                 |                            |
                +-----------------+-------------+--------------+
                                                v
                                  validation-only calibration
                                                |
                                                v
                                 identical endpoint test cohort
                                                |
                                                v
                           metrics + timing + errors + explanations
```

## Implemented Boundaries

### Configuration and paths

- `configs/synthetic_v0.2.json` is the resolved default experiment input.
- `src/data_paths.py` maps a scenario to
  `raw/interim/processed/manifests` under `ARGUS_DATA_ROOT`.
- `src/check_data_storage.py` validates mount existence and free space before a
  future acquisition.

### Ingestion and inspection

- `src/datasets/contracts.py` defines canonical minimum schemas and aliases.
- `src/datasets/inspection.py` inventories CSVs, samples quality conditions,
  and writes a manifest without loading a full public dataset.
- `src/datasets/loading.py` provides normalized, provenance-preserving chunked
  CSV iteration.
- `src/inspect_dataset.py` exposes inspection as a scenario-aware CLI.

The CERT core schema has been checked against bundled documentation and headers.
`src/datasets/cert_labels.py` implements the accepted answer index, exact-row
membership checks, positive account-days and incident-connected account groups;
actual split allocation and full user-day preparation are still pending.
CICIDS contracts intentionally define only a stable minimum; the full
feature list is discovered and frozen from actual publisher files.

### Synthetic preparation

- `src/generate_data.py` creates deterministic user profiles, work patterns,
  and four temporally distributed anomaly families.
- `src/preprocessing.py` assigns one chronological split, validates it, fits
  feature state on train, and builds each split's sequences independently.
- `src/features.py` owns causal rolling features, normal-profile deviations,
  categorical unknown handling, and endpoint sequence labels.

### Training and evaluation

- `src/train_baseline.py` fits Random Forest and Isolation Forest, calibrates
  thresholds on validation, and exports importance, errors, timing, and plots.
- `src/train_lstm_autoencoder.py` fits normal train sequences, chooses its epoch
  and reconstruction threshold on validation, and evaluates frozen state on
  test.
- `src/evaluate.py` checks sorted user/time/label cohorts. Stable item identity
  is required in the planned contract.
- `src/experiment_tracking.py` writes separate metrics/metadata directories.
  Full root predictions and the LSTM checkpoint can still be overwritten;
  immutable model/input/prediction artifacts are planned.
- `src/explain_anomalies.py` combines model flags with train-normal profile
  deviations; it does not infer an explanation from the true target.

## Stored State and Trust Boundaries

| Location | Purpose | Git policy |
| --- | --- | --- |
| `data/<scenario>/raw` | immutable local publisher/generated input | ignored |
| `data/<scenario>/interim` | rebuildable normalized data | ignored |
| `data/<scenario>/processed` | features, splits, sequence arrays | ignored |
| `data/<scenario>/manifests` | local inspection/acquisition reports | ignored by data-root policy |
| `results/experiments` | full timestamped run contracts | ignored |
| `results/baseline_reference` | compact public release evidence | tracked |

Raw data and target/answer-key fields are outside the model-feature trust
boundary. A preprocessing state may learn only from train. Validation can
select a declared configuration/threshold. Test can only measure a frozen
configuration.

## Current Extension Points

Each public scenario should add, in this order:

1. accepted publisher manifest and checksums;
2. frozen schema/quality report;
3. scenario split implementation;
4. train-fitted feature builder matching `docs/features.md`;
5. tabular baselines on the common cohort;
6. sequence model only when order and entity identity are meaningful;
7. error analysis and release snapshot.

Public scenarios must not be forced into one lossy universal table. They share
experiment interfaces and artifact contracts, not necessarily raw columns or
feature semantics.

## Planned Integrated Workspace: Spectra Lab

One Argus application, one FastAPI backend and one React/TypeScript frontend
provide Overview, Model Selection, Event Investigation and Scenario Checks.
EventLens is a former name, not another service/frontend/install. Scope and
module layout live in the [implementation plan](spectra-lab-implementation-plan.md);
order lives in the [roadmap](roadmap.md#resume-after-ssd), visual rules in
[design](design.md).

```text
train/validation -> bounded configuration study -> immutable selection
                                                -> explicit final test

frozen run -> investigation -> valid raw patch -> frozen feature recompute
                                             -> same model and threshold
```

Current complete training scripts evaluate test on every invocation; they need
a train/validate-only candidate mode before search. Trial outputs must be
isolated, with no stale models or root CSV reuse.

Shared contracts carry snapshot, preparation, analysis-item, sequence, cohort,
run and study identities. Synthetic items are events; CERT user-days require
item-to-source-event mapping. Membership must match the actual model input,
including length overrides. Large weights/predictions use the data root;
current ARGUS_DATA_ROOT does not relocate RESULTS_DIR.

Research services reuse core features and inference, without refitting train
state or reimplementing formulas in the frontend. Scenarios have separate
lineage and unknown new labels; source records and base metrics remain fixed.
Synthetic bytes_sent replay is first; CERT requires supported source fields
and raw-to-user-day reaggregation.

Viewer startup must not resolve an absent external volume or import torch.
Inference/search are capabilities of this same application, backed by bounded
jobs and registered artifacts. The viewer can display unavailable sources.

## Deliberately Out of Scope

- real-time event ingestion;
- SIEM/LDAP/firewall integrations;
- incident and user management;
- production authentication, authorization, and audit;
- availability and distributed scaling;
- general AutoML, causal discovery, Wi-Fi sensing and separate product repos;
- a second dashboard or map-first replacement for investigation.

See the [experimental protocol](experimental-protocol.md), [feature map](features.md),
and [storage policy](data-storage.md) for normative rules.

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

### CERT robustness implementation — 2026-11-13

`datasets.cert_validation` creates incident-group development folds from primary
train only, with separate inner fit/calibration and outer assessment roles.
`datasets.cert_experiments.prepare_cert_frame` fits each fold's own feature
state; `assess_development` loads a frozen LSTM/threshold without refitting.
The optional global-only history policy is separately versioned from v1.
`run_cert_robustness.py` executes the fixed two-policy, five-seed plan in
isolated CPU workers and freezes candidate/hash/threshold records. This bounded
runner is not a generalized search service or Spectra job manager. Real final
test evaluation remains a separate command and stage.

The [CERT frozen final driver](cert-r4.2-final-evaluation.md),
`run_cert_final.py`, has completed all 30 test evaluations. It publishes a
predeclared plan, source checks and test-exposure ledger, blocks training and
threshold-selection calls, and saves exact outputs, timings and scenario
diagnostics. Answer attributions are post-inference reporting data only.
`datasets.cert_reporting` deduplicates observable-day/scenario associations and
preserves independent-group support; neither module changes model inputs.

### CERT score/model extensions — 2026-11-14

[Quality increment 4](cert-r4.2-model-variants-v5.md) adds explicit configured
scoring and a supervised histogram-boosting candidate. `experiment.inputs`
and the frozen `experiment.score` retain their original AST/behavior.
`configured_score` delegates old settings to that scorer and dispatches the
named endpoint-only LSTM policy or positive-class HGB probability. The LSTM
module, full-window optimization loss and all feature builders are unchanged.
`assess_development` reloads these new candidates with their saved score policy;
the existing final scope guard still rejects development-only preparations.

`run_cert_model_variants.py` binds prior preparations/controls, snapshots all
code/configs, declares 24 jobs before fitting, and runs isolated CPU workers
with test/assessment read guards. HGB fits the same labeled inner-fit days as
RF; `early_stopping=False` preserves connected-group boundaries. This extends
the isolated research API, without replacing the legacy three-model synthetic
workflow or creating a search service. LSTM remains an available baseline;
HGB is the strongest supervised candidate in this development comparison.

### CERT operating-point extension — 2026-11-14

`cert_operating_point.py` provides a named calibration-only FP-budget selector;
the original F1 selector and model training code remain unchanged.
`run_cert_operating_point.py` fixes three tabular configurations and five seeds,
reuses nine seed42 checkpoints, and freezes every threshold profile before
outer assessment. An operating candidate owns copied model/state/config,
exact calibration arrays, all threshold boundaries, profile eligibility and
source-manifest lineage. Assessment reloads one frozen model and applies its
numeric thresholds without fit or calibration. Test reads remain blocked.

These are development-only research bundles, not a deployed detector or a
general inference service. A failed minimum recall is explicit; even passing
calibration does not guarantee the declared FPR on held-out accounts. All
seeds/folds/profiles are retained. The missing independent CERT source is
recorded explicitly rather than reusing the consumed test as confirmation.
See the [protocol and results](cert-r4.2-operating-point-v6.md) and
[remaining implementation stages](cert-r4.2-stage-assessment.md).

### Shared frozen inference — S3 part 1

`inference.py` loads sealed isolated candidates and operating bundles through
`load_frozen_run`. `PreparedBatch` separates numeric inputs and ordered IDs/window
membership from reporting labels, binding preparation and feature-state hashes.
RF/IF/HGB reuse the existing scorer without importing PyTorch; LSTM lazily
loads CPU weights and exposes full-window reconstruction with either named
score policy. An operating bundle requires an explicit profile. Schema, policy,
status, lineage and checksum checks precede scoring; no fit or threshold search
runs. The [S3 contract](s3-frozen-inference.md) documents producer provenance
responsibilities and the remaining diagnosis/scenario/CLI work.
