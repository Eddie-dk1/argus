# Argus

Argus is an academic research project for anomaly detection in user behavior
and network activity. The current priority is **model quality, reliable
comparisons and reproducible evidence**.

## Current status

Updated **2026-11-14**. The published `argus-v0.2.0` release is the historical
synthetic reference. Current development includes CERT research and the
completed S3 frozen runtime; a new release is not implied by these changes.

| Area | State |
|---|---|
| Synthetic benchmark | Deterministic generation, causal features, aligned RF/IF/LSTM evaluation |
| CERT r4.2 | Accepted local snapshot, audited observable labels, incident-group split, daily preparation, frozen final and five controlled quality increments |
| S3 | Complete: common inference, exact replay, quality diagnosis and verified synthetic raw scenarios |
| S4 | Agreed next: error investigation and two bounded model-improvement cycles; not started |
| Spectra Lab UI | Planned after the research foundations; model quality has priority |
| CICIDS2017 / CSE-CIC-IDS2018 | Contracts and acquisition research; not yet accepted or evaluated |

The [seven-part S4 plan](docs/s4-model-quality-plan.md) covers HGB context
features/regularization, RF and LSTM alternatives, and saved controls. Its
matrix has at most 117 new research fits, an overall ceiling of 120 and up to 12
hours of experiment execution. These experiments have **not been run**.

See the [roadmap](docs/roadmap.md) and
[stage assessment](docs/cert-r4.2-stage-assessment.md).

## Current CERT quality

HGB rolling is the research candidate. RF, IF and both LSTM score policies
remain reproducible comparisons. The table uses seed 42 and pooled decisions
from three reused development group folds: **196,765 endpoints, 646 observable
positives**. Thresholds come from separate inner calibration.

| HGB profile | Precision | Recall | F1 | TP | FP | FN | FP/1,000 background days |
|---|---:|---:|---:|---:|---:|---:|---:|
| Primary FP1/1,000 | 52.23% | 59.91% | 0.5580 | 387 | 354 | 259 | 1.805 |
| Unchanged legacy F1 control | 59.34% | 60.99% | 0.6015 | 394 | 270 | 252 | 1.377 |

Mean within-fold PR-AUC is 0.7056; cross-fold score rankings are not pooled.
The primary requirement, **FP≤1/1,000 and recall≥50% together**, fails on all
three unique outer folds. All five HGB seeds produce identical scores;
repetition does not add independent evidence. Scenario 3 recall is 6/32 with
only one independent connected group.

The original CERT test was used on 2026-11-13. Improved HGB results are reused
**development evidence**, not independent confirmation or production IDS
performance. The legacy F1 profile has not been promoted after examining outer
results. Details: [model variants](docs/cert-r4.2-model-variants-v5.md),
[operating points](docs/cert-r4.2-operating-point-v6.md),
[S3 diagnosis](docs/s3-frozen-diagnostics.md).

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
