# Argus

Argus is an academic research prototype for anomaly detection in network
activity and user behavior. It provides a reproducible synthetic benchmark that
compares a supervised Random Forest, an unsupervised Isolation Forest, and a
sequence-based LSTM Autoencoder on one exactly aligned evaluation cohort.

Argus is not a production IDS/UEBA platform. The current release is a research
foundation: deterministic data generation, leakage-aware preprocessing,
validation-only threshold selection, experiment manifests, metrics, plots, and
short anomaly explanations.

## Release Status

`argus-v0.2.0` stabilizes the synthetic pipeline and prepares the repository for
public data without downloading the 200+ GiB dataset collection.

- Synthetic scenario: implemented, tested, and reproducible from a committed
  config.
- CERT r4.2: accepted external snapshot, frozen label/group policy, daily
  preparation and isolated experiment inputs implemented; no dataset is bundled.
  [First RF/IF validation baselines](docs/cert-r4.2-benchmark.md) completed on
  2026-11-13. [LSTM seed 42 and quality diagnostics](docs/cert-r4.2-lstm-baseline.md)
  are also complete. [Five-seed robustness and grouped CV](docs/cert-r4.2-robustness.md)
  compare the original and global-only history policies; 30 candidates are frozen.
  [Frozen final evaluation](docs/cert-r4.2-final-evaluation.md) is complete:
  mean test F1 for global-only RF/IF/LSTM is 0.1863/0.0097/0.0129.
  Current test is now used; new confirmatory improvement claims require an
  independent holdout. All models still need quality work, especially recall.
  [Causal history v3](docs/cert-r4.2-causal-history.md) is implemented and audited.
  [Paired group-CV](docs/cert-r4.2-group-cv-v3.md) is complete: mean outer RF F1
  0.2266 → 0.2346 with more false alarms; IF worsens and LSTM gains are limited.
  This is development evidence on reused folds, not independent confirmation.
  [File/USB deviations](docs/cert-r4.2-file-usb-v4.md) complete quality increment 3:
  RF with a frozen early profile reaches mean outer F1 0.2919; pooled FP
  1,255 → 905 and TP 249 → 256 versus v3, still missing 390/646 positives.
  IF/LSTM quality remains weak; no universal policy replacement is claimed.
  [Endpoint score and fixed histogram boosting](docs/cert-r4.2-model-variants-v5.md)
  complete increment 4: HGB rolling mean outer F1/PR-AUC 0.6217/0.7056;
  pooled precision 59.34%, recall 60.99%, TP394/FP270/FN252. Endpoint-only
  scoring does not resolve LSTM quality. HGB is a supervised development
  candidate; that increment used one seed on repeated development folds.
  [Operating points and five-seed stability](docs/cert-r4.2-operating-point-v6.md)
  complete increment 5: 36 new fits, nine frozen controls and 135 decision
  profiles. The primary budget of 1 FP/1,000 background days with recall≥50%
  is not met on outer accounts; HGB gives pooled FP354, recall59.91%,
  F1 0.5580 versus its unchanged F1 control FP270/F1 0.6015. All HGB seed scores
  are identical; repeated seeds do not resolve group-transfer variation.
  All five quality increments are complete within development scope;
  independent confirmation remains unavailable. See the
  [readiness assessment and remaining stages](docs/cert-r4.2-stage-assessment.md).
- CICIDS2017 and CSE-CIC-IDS2018: researched and represented by storage,
  schema-inspection and feature contracts; neither is claimed as evaluated.

## Development Direction — Planned

Research integrity is the project-owner priority: improve through justified
development experiments, retain negative results, and do not tune on test.
See [contribution rules](CONTRIBUTING.md#before-changing-an-experiment) and the
[concrete next experiment plan](docs/cert-r4.2-improvement-plan.md).

Argus will have one integrated research workspace, **Spectra Lab**: Overview,
Model Selection, Event Investigation, and Scenario Checks. EventLens is the
former investigation name, not another product, service or install.

One workflow investigates a frozen model through source records, exact inputs,
normal references and valid raw-data changes followed by repeat inference.
The other runs bounded configuration studies and compares validation quality,
training/inference cost and memory. Candidate studies must not read test;
final evaluation is separate, after selection is frozen.

These features are not implemented in v0.2. The agreed scope has no separate
Streamlit dashboard or live macOS/Wi-Fi collector. The interface will use
minimalist dark navy, neutral controls, and red for errors/anomaly indicators:
see the [design specification](docs/design.md).

Resume with the [post-SSD sequence](docs/roadmap.md#resume-after-ssd) and the
[implementation plan](docs/spectra-lab-implementation-plan.md). Small synthetic
fixtures and contract work do not require the external SSD.

## Quick Start

Create an isolated Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Run the complete reference pipeline:

```bash
./run_all.sh
```

Run a faster tabular-only check:

```bash
./run_all.sh --skip-autoencoder
```

Reuse an existing generated raw file:

```bash
./run_all.sh --skip-generate
```

The default experiment is fully described by
[`configs/synthetic_v0.2.json`](configs/synthetic_v0.2.json). Command-line model
overrides are intended for declared experiments, but the test split must not be
used to choose them.

## Reference Result

The v0.2 reference run uses 30,000 synthetic events, a chronological `60/20/20`
split, seed `42`, and a common test cohort of 5,600 sequence endpoints. That
cohort contains 361 anomalous events (6.45%). Thresholds and LSTM epoch are
selected on validation only.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 1.000 | 0.889 | 0.941 | 1.000 | 1.000 | 0.000 |
| Isolation Forest | 0.393 | 0.496 | 0.439 | 0.403 | 0.886 | 0.053 |
| LSTM Autoencoder | 0.443 | 0.399 | 0.420 | 0.351 | 0.831 | 0.035 |

These numbers are a controlled engineering result, not evidence of production
performance. In particular, the near-perfect Random Forest ranking indicates
that the generated anomaly families remain strongly separable. Multi-seed
stability and public-dataset validation are later research stages.

The compact release snapshot is in
[`results/baseline_reference/`](results/baseline_reference/), including config,
data hash, split/feature manifests, validation/test metrics, timing, and feature
importance.

## What the Pipeline Does

```text
committed config
      |
      v
deterministic synthetic events
      |
      v
chronological split before fitting features
      |
      +---------------------------+
      |                           |
      v                           v
tabular endpoint rows       per-user sequences
      |                           |
      v                           v
Random/Isolation Forest     LSTM Autoencoder
      |                           |
      +-------------+-------------+
                    v
validation threshold -> one common test cohort
                    v
metrics, timing, errors, plots, explanations
```

The generator models four anomaly families:

- `brute_force_login`;
- `unusual_night_activity`;
- `data_exfiltration`;
- `unusual_ip_change`.

Features are causal: rolling values use earlier events only. Encoders, normal
profiles, and scalers are fitted on train only. Raw user/IP identifiers and
labels are provenance, not default model inputs.

## External Dataset Storage

Synthetic data defaults to the ignored repository `data/` directory. Public
datasets should be kept on an external SSD by setting an existing mount path:

```bash
export ARGUS_DATA_ROOT="/Volumes/ArgusData/argus-data"
python3 src/check_data_storage.py \
  --scenario cert_r4_2 \
  --create-scenario-directories
```

Verify that the example mount exists before creating its data root. Before
downloading, also pass `--required-gib` with a budget calculated from the
selected source manifest. `ARGUS_DATA_ROOT` currently relocates datasets, not
the repository's `results/` directory.

If `ARGUS_DATA_ROOT` is set but the volume is missing, Argus fails instead of
silently creating a large internal-drive copy. The exact storage and capacity
procedure is in the [data storage policy](docs/data-storage.md).

After files are acquired from their publishers, inspect a raw directory before
writing preprocessing code:

```bash
python3 src/inspect_dataset.py --scenario cert_r4_2
python3 src/inspect_dataset.py --scenario cicids2017
python3 src/inspect_dataset.py --scenario cse_cic_ids2018
```

The CERT core schema has been reconciled with the bundled r4.2 README and actual
headers. Full data preparation follows the accepted observable account-day
label policy; model evidence and its limitations are linked in Release Status.

## Reproducibility Rules

- Split raw records before fitting preprocessing state.
- Use validation for features, hyperparameters, early stopping, and thresholds.
- Evaluate test only after the configuration is frozen.
- Compare models on identical evaluation objects and targets. The v0.2 check
  uses user/time/label; stable event/item IDs are part of the planned contract.
- Report precision, recall, F1, PR-AUC, ROC-AUC, FPR, support, prevalence, and
  timing.
- Keep data hashes, resolved config, environment, split report, feature
  manifest, threshold search, errors, and plots per run.
- Treat synthetic metrics and explanations as limited evidence.

The complete rules are in the [experimental protocol](docs/experimental-protocol.md).
Old test-informed autoencoder sweep scripts were removed in v0.2; future search
spaces must be bounded in a committed config and selected on validation only.

## Tests

Run the unit and contract checks:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
bash -n run_all.sh
```

The tests cover generator determinism, split invariants, target leakage,
unknown categories, endpoint sequence labels, storage failure behavior, schema
aliases, malformed input, and chunked CSV loading.

## Project Layout

```text
argus/
├── configs/                 # versioned experiment configurations
├── docs/                    # protocol, datasets, storage, features, literature
├── results/
│   ├── baseline_reference/  # compact v0.2 reproducibility snapshot
│   └── plots/               # selected reference plots
├── src/
│   ├── datasets/            # scenario contracts, inspection, chunked loading
│   ├── data_paths.py        # ARGUS_DATA_ROOT resolution
│   ├── generate_data.py
│   ├── preprocessing.py
│   ├── train_baseline.py
│   ├── train_lstm_autoencoder.py
│   ├── evaluate.py
│   └── experiment_tracking.py
├── tests/                   # unit tests and tiny synthetic schema fixtures
├── run_all.sh
└── requirements.txt
```

Generated datasets, weights, full predictions, and timestamped experiment runs
are ignored. They can be rebuilt from code, config, and publisher data.

## Documentation

- [Architecture](docs/architecture.md)
- [Spectra Lab implementation plan](docs/spectra-lab-implementation-plan.md)
- [Interface and design specification](docs/design.md)
- [Dataset research](docs/datasets.md)
- [Data storage policy](docs/data-storage.md)
- [Experimental protocol](docs/experimental-protocol.md)
- [Scenario feature map](docs/features.md)
- [Working literature review](docs/literature-review.md)
- [Roadmap](docs/roadmap.md)
- [Contributing](CONTRIBUTING.md)

## License

Argus is licensed under the [MIT License](LICENSE).

## Independent experiments

For new runs use [the isolated experiment workflow](docs/isolated-experiments.md).
It separates preparation, validation-only candidates and explicit final test,
with per-run weights, exact inputs, IDs and checksummed artifacts on a selected
storage root. The Quick Start above remains the legacy v0.2 reference workflow.

### CERT: active account-day preparation

`src/prepare_cert_days.py` streams the five accepted r4.2 activity logs into
raw daily aggregates, exact event membership and positive evidence using
the frozen account split. It verifies raw hashes, observable rows, ID uniqueness
and population counts. See [commands, formulas and artifact contract](docs/cert-r4.2-day-preparation.md).
Daily aggregation is increment 2. [Increment 3](docs/cert-r4.2-experiments.md)
now connects CERT through `experiment.py prepare-cert`: frozen train transforms,
causal train histories, held-out cold-start, shared endpoint cohorts and the
existing candidate/final workflow. The daily aggregation command itself does
not train models.

[Causal history v3 development preparation](docs/cert-r4.2-causal-history.md)
adds the same past-observation update rule in every role. Its separate command
prepares only original train/validation, with no model fit or new test scoring.
[Paired grouped comparisons](docs/cert-r4.2-group-cv-v3.md) for all three models
are complete, including negative results. [The separate file/USB ablation](docs/cert-r4.2-file-usb-v4.md)
is now complete: six new audited preparations, 18 new fits and nine reused
v3 controls. Optional rolling/frozen policies add four inputs while preserving
all common features, labels and endpoints. No test is scored; score/architecture
variants were compared separately in [quality increment 4](docs/cert-r4.2-model-variants-v5.md):
12 endpoint-score LSTMs and 12 HGB candidates reuse all four existing feature
policies/preparations. HGB fits the same labeled inner-fit days as RF;
its automatic random early stopping is disabled. The legacy models and
synthetic reference remain available. [Operating-point work](docs/cert-r4.2-operating-point-v6.md)
is also complete with an explicit negative result for the primary FP/recall
budget. No new accepted independent CERT holdout or deployment threshold is
claimed. [S3 part 1](docs/s3-frozen-inference.md) adds the common read-only
`load_frozen_run` API for RF/IF/HGB and both LSTM scores, explicit operating
profiles and label-free prepared batches. [Part 2](docs/s3-frozen-diagnostics.md)
completes exact assessment replay and diagnosis across all saved folds/seeds,
preserving the failed primary FP/recall result. [Part 3](docs/s3-synthetic-scenarios.md)
completes synthetic `bytes_sent` scenarios with full-split recompute verification
and the unified `src/argus.py` CLI: `inspect`, `replay`, `diagnose`, `scenario`.
S3 is complete; detection metrics remain unchanged. See the
[remaining stages](docs/cert-r4.2-stage-assessment.md).
