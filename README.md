# Argus

Argus studies whether changes in a person's daily computer activity can identify insider-threat events at a low false-alarm rate. The project evaluates account-day predictions on CERT r4.2 and provides a deterministic synthetic benchmark for comparing anomaly detectors.

## CERT r4.2 results

The S6 experiment tested two changes to histogram gradient boosting (HGB): comparing activity with the account's own recent history and weighting training days by group and class. Model selection used inner group splits. All three decision thresholds were fixed before outer assessment.

| Primary operating point | Precision | Recall | F1 | Mean fold PR-AUC | False alarms |
|---|---:|---:|---:|---:|---:|
| Historical HGB control | 52.23% | 59.91% | 0.5580 | 0.7056 | 354 |
| S6 selected models | 82.47% | 85.91% | 0.8415 | 0.8985 | 118 |

On 196,765 account-days, S6 detected **555 of 646 positive days**, up from 387. Two of three folds met the joint target of at least 50% recall and at most one false alarm per 1,000 background days. The third recorded 72 false alarms against a budget of 65. Scenario 3 remains difficult, with 8 of 32 positive days detected in a single incident-connected group.

These figures come from three previously used development folds with model seed 42. Independent confirmation remains open. The original CERT test was used on 2026-11-13. [Experiment report](docs/s6-factorial-experiment.md) · [Results for all 39 jobs](docs/s6-factorial-results.json)

## Project status

S3 delivered frozen inference and replay. S4 completed the baseline investigations and bounded model comparisons. S6.3 completed the personal-history and group-weight experiment. **Next is S6.4**, a paired analysis of which errors changed, how false alarms are distributed across accounts, and how much the gains vary across groups.

The synthetic benchmark covers random forest, isolation forest and an LSTM autoencoder. Spectra Lab UI and evaluation on CICIDS2017 and CSE-CIC-IDS2018 remain planned. The published v0.2.0 release documents the historical synthetic reference.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Datasets and model weights are not included. Public sources require their own
acceptance manifests and local storage; see [dataset contracts](docs/datasets.md)
and [storage policy](docs/data-storage.md). `ARGUS_DATA_ROOT` must point to an
existing directory when configured.

## Isolated experiments

For new experiments, choose an **existing** artifact root. Every command
creates a separate UUID directory with checksummed inputs, state and results.

```bash
python3 src/experiment.py --root "$ARTIFACT_ROOT" prepare \
  --config configs/synthetic_v0.2.json
python3 src/experiment.py --root "$ARTIFACT_ROOT" candidate \
  --preparation "$PREPARATION" --model random_forest
# After configuration/threshold selection is frozen:
python3 src/experiment.py --root "$ARTIFACT_ROOT" final \
  --preparation "$PREPARATION" --candidate "$CANDIDATE"
```

Candidate reads train/validation; final evaluation is explicit. CERT development
preparations use separate assessment cohorts and prohibit real-test access.
Read the [isolated workflow](docs/isolated-experiments.md) and
[experimental protocol](docs/experimental-protocol.md) before launching research.

## Frozen runtime and scenario CLI

```bash
python3 src/argus.py --root "$ARTIFACT_ROOT" inspect --run "$CANDIDATE"
python3 src/argus.py --root "$ARTIFACT_ROOT" replay --run "$CANDIDATE"
python3 src/argus.py --root "$ARTIFACT_ROOT" replay \
  --run "$OPERATING_BUNDLE" --profile fp1_per_1000 \
  --source "$SAVED_ASSESSMENT" --role assessment
python3 src/argus.py --root "$ARTIFACT_ROOT" scenario \
  --run "$SYNTHETIC_CANDIDATE" --preparation "$SYNTHETIC_PREPARATION" \
  --position earlier --operation double
python3 src/argus.py --root "$ARTIFACT_ROOT" diagnose \
  --model-variants "$MODEL_VARIANTS_SERIES" \
  --operating-points "$OPERATING_POINT_SERIES"
```

Variables identify specific sealed artifacts. Operating bundles require an
explicit profile. Frozen inference preserves weights, feature order, window
membership, score policy, dtype and numeric thresholds; it performs no fit or
threshold selection.

Synthetic scenarios support raw `bytes_sent` at an endpoint or earlier window
member: no-op, double, zero or a finite nonnegative value. Full affected-user
history is recomputed through frozen state and checked against a separate
full-split oracle. Variant ground truth is unknown; variants never enter the
original benchmark metrics. CERT currently supports saved-input inspection
and replay; raw what-if needs a separate reaggregation contract.

See [inference](docs/s3-frozen-inference.md),
[diagnosis](docs/s3-frozen-diagnostics.md) and
[scenario protocol](docs/s3-synthetic-scenarios.md).

## Historical synthetic reference

The complete legacy reproduction command is:

```bash
./run_all.sh
python3 src/build_reference_snapshot.py
```

It trains/calibrates models, evaluates test and writes shared outputs. Use a
separate project copy for regression; new candidate studies use the isolated
workflow above.

The v0.2 reference has 30,000 events and 5,600 common test endpoints, including
361 anomalies, with a chronological 60/20/20 split and seed 42.

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 1.000 | 0.889 | 0.941 | 1.000 |
| Isolation Forest | 0.393 | 0.496 | 0.439 | 0.403 |
| LSTM Autoencoder | 0.443 | 0.399 | 0.420 | 0.351 |

These synthetic families are strongly separable, especially for RF. The table
is an engineering reference and must not be compared with CERT as a shared
quality scale. The compact [reference snapshot](results/baseline_reference/)
remains unchanged.

## Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
bash -n run_all.sh
git diff --check
```

S3 acceptance passed 104 tests, 30 synthetic scenario checks and exact saved
CERT replay. Full legacy regression in a temporary copy preserved RF/IF/LSTM
metrics. Source data, checkpoints, thresholds and state remain authoritative.

## Project scope and documentation

Argus is a research prototype. Spectra Lab is the planned single research
workspace; EventLens is its historical investigation name. There is no shipped
web UI, live collector or production SIEM integration.

- [Permanent repository rules](AGENTS.md) and [contributing](CONTRIBUTING.md)
- [Architecture](docs/architecture.md), [roadmap](docs/roadmap.md)
- [Next S4 quality plan](docs/s4-model-quality-plan.md)
- [Dataset research](docs/datasets.md), [feature contracts](docs/features.md)
- [Spectra implementation plan](docs/spectra-lab-implementation-plan.md)
- [Design requirements](docs/design.md), [literature](docs/literature-review.md)

Git contains code, tests, configs, protocols and the existing compact synthetic
reference. Downloaded data, weights, full predictions, experiment directories,
local reports, presentations and design previews stay outside version control.

Argus code is licensed under [MIT](LICENSE). Dataset terms are separate.
