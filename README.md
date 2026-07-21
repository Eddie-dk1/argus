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
- CERT r4.2, CICIDS2017, CSE-CIC-IDS2018: researched and represented by storage,
  schema-inspection, and feature contracts; no real data is bundled or claimed
  as evaluated.
- Next active-data milestone: CERT r4.2 after suitable external storage is
  available.

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
  --required-gib 100 \
  --create-scenario-directories
```

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

The CERT contract is explicitly provisional until it is reconciled with the
README bundled in the chosen r4.2 release.

## Reproducibility Rules

- Split raw records before fitting preprocessing state.
- Use validation for features, hyperparameters, early stopping, and thresholds.
- Evaluate test only after the configuration is frozen.
- Compare models only on rows with identical `user_id`, `timestamp`, and target.
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
- [Dataset research](docs/datasets.md)
- [Data storage policy](docs/data-storage.md)
- [Experimental protocol](docs/experimental-protocol.md)
- [Scenario feature map](docs/features.md)
- [Working literature review](docs/literature-review.md)
- [Roadmap](docs/roadmap.md)
- [Contributing](CONTRIBUTING.md)

## License

Argus is licensed under the [MIT License](LICENSE).
