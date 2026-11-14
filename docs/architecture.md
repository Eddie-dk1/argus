# Architecture

Updated 2026-11-14. The [isolated workflow](isolated-experiments.md) implements
synthetic/CERT preparation, stable identities, saved inputs/models,
validation-only candidates and explicit frozen evaluation. CERT final and all
five quality increments are complete within their declared scope.
[S3](s3-frozen-inference.md) is complete: common inference, exact replay,
diagnosis and verified synthetic scenarios through `src/argus.py`.

The [next S4](s4-model-quality-plan.md) prioritizes error investigation and two
bounded model-improvement cycles. Spectra UI and general study automation remain
planned. Historical release notes below retain their original stage scope.

## Legacy synthetic Data Flow

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

The CERT core schema is checked against bundled documentation and headers.
The adapters implement the accepted answer index, exact-row membership,
positive account-days, incident-connected split and full daily preparation.
Frozen global/causal/file-USB feature policies and sequence membership are
implemented and evaluated on the declared development cohorts.
CICIDS contracts intentionally define only a stable minimum; the full
feature list is discovered and frozen from actual publisher files.

### Synthetic preparation

- `src/generate_data.py` creates deterministic user profiles, work patterns,
  and four temporally distributed anomaly families.
- `src/preprocessing.py` assigns one chronological split, validates it, fits
  feature state on train, and builds each split's sequences independently.
- `src/features.py` owns causal rolling features, normal-profile deviations,
  categorical unknown handling, and endpoint sequence labels.

### Isolated training, evaluation and runtime

- `src/experiment.py` and the CERT runners own per-preparation candidates,
  checkpoints, exact arrays, separate calibration/assessment and lineage.
- `src/inference.py` provides frozen RF/IF/HGB/LSTM score/predict and LSTM
  reconstruction. It performs no training or threshold selection.
- `src/argus.py` exposes inspect/replay/diagnose/scenario with isolated outputs;
  raw scenarios currently support synthetic bytes_sent only.

### Legacy reference commands

- `src/train_baseline.py` fits Random Forest and Isolation Forest, calibrates
  thresholds on validation, and exports importance, errors, timing, and plots.
- `src/train_lstm_autoencoder.py` fits normal train sequences, chooses its epoch
  and reconstruction threshold on validation, and evaluates frozen state on
  test.
- `src/evaluate.py` checks sorted user/time/label cohorts for this legacy route.
  Stable IDs and ordered membership are enforced in the isolated workflow.
- `src/experiment_tracking.py` writes run metadata and legacy inference bundles.
  Shared root outputs can still be overwritten by legacy commands; isolated
  candidates/evaluations preserve their own immutable artifacts.
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

Legacy complete training scripts evaluate test on every invocation. The
implemented isolated candidate mode supplies the train/validate-only boundary
for future studies. General search, jobs and the frontend are future work;
trials must preserve this existing isolation.

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
responsibilities; [part 3](s3-synthetic-scenarios.md) implements scenarios/CLI.

### Frozen replay and diagnosis — S3 part 2

`run_frozen_diagnostics.py` binds the two declared research series, source
preparations, model/state hashes and exact saved inputs. It replays all 60
model-variants candidates and all profiles of nine seed42 operating bundles
through the common API, checking complete inputs, feature/endpoint ordering,
window membership, scores and decisions exactly. `frozen_diagnostics.py`
separates reporting labels from inference, accounts/groups/scenarios, history
availability from policy fallback, seed variation from fold variation, and
undefined metrics from legacy numeric values. `frozen_diagnostic_report.py`
produces the readable report/figure and a quality-hypothesis queue without
selecting thresholds or models. The versioned JSON includes all 45 operating
bundles and every saved fold/seed/profile; cross-fold score rankings remain
undefined. See [the protocol and results](s3-frozen-diagnostics.md).

### Verified synthetic scenarios and CLI — S3 part 3

`argus.py` exposes `inspect`, `replay`, `diagnose`, `scenario`, setting original
CPU threads before numeric imports. `frozen_runtime.py` binds saved inputs and
reference predictions to sealed models/preparations; outputs are new UUID
artifacts with source/code hashes and preservation checks. `diagnose` delegates
to the established runner with its original access guards.

`synthetic_scenarios.py` patches only raw `bytes_sent`, recomputes the entire
affected user history using frozen train state and the original builder/CSV
boundary, and compares features, inputs, ordered membership and inference with
a separate full-split recompute. No-op uses predeclared dtype tolerances and
exact decisions; other users remain unchanged. Base labels are provenance,
variant ground truth is unknown, and variants never enter benchmark metrics.
CERT supports inspection/replay of saved prepared inputs; raw reaggregation
remains a separate contract. See [the scenario protocol](s3-synthetic-scenarios.md).
