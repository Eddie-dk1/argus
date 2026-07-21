# Architecture

Status: implemented synthetic research pipeline plus public-dataset preparation
contracts for `argus-v0.2.0`.

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

The CERT schema remains provisional until checked against documentation bundled
with r4.2. CICIDS contracts intentionally define only a stable minimum; the full
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
- `src/evaluate.py` refuses to compare models unless evaluation identities and
  targets match exactly.
- `src/experiment_tracking.py` writes one immutable run directory per model and
  configuration.
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

## Deliberately Out of Scope

- real-time event ingestion;
- SIEM/LDAP/firewall integrations;
- incident and user management;
- production authentication, authorization, and audit;
- availability and distributed scaling;
- a dashboard before the research artifacts stabilize.

See the [experimental protocol](experimental-protocol.md), [feature map](features.md),
and [storage policy](data-storage.md) for normative rules.
