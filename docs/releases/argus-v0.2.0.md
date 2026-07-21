# Argus v0.2.0 — Reproducible Synthetic Baseline

This release turns the original synthetic prototype into a reproducible
research baseline and completes the work that can be done safely before the
200+ GiB public dataset collection is placed on external storage.

## Highlights

- Reworked deterministic generator with user profiles, temporal behavior, and
  four anomaly families distributed across every chronological split.
- One immutable `60/20/20` split created before fitted preprocessing.
- Causal rolling/profile features; raw labels, IDs, IP encodings, timestamps,
  and source shortcuts are excluded from default model inputs.
- Random Forest, Isolation Forest, and LSTM Autoencoder evaluated on an exact
  common cohort of sequence endpoints.
- Validation-only model/epoch/threshold selection and one frozen test report.
- FPR, PR-AUC, support, prevalence, timing, per-run configs/hashes, feature/split
  manifests, threshold searches, and error-analysis artifacts.
- Official-source notes for CERT r4.2, CICIDS2017, and CSE-CIC-IDS2018.
- Safe `ARGUS_DATA_ROOT` policy, external-volume failure behavior, free-space
  preflight, schema inspection, normalized chunked loading, and tiny fixtures.
- Fixed experimental protocol, scenario feature map, literature decision
  matrix, updated architecture, roadmap, README, and contribution checks.
- Removed old autoencoder sweep scripts that selected configurations using test
  metrics.

## Reference Configuration and Result

- Config: `configs/synthetic_v0.2.json`
- Seed: `42`
- Generated events: `30,000`
- Event split: `18,000 / 6,000 / 6,000`
- Common test cohort: `5,600` endpoint events
- Test anomalies: `361` (`6.45%`)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 1.000 | 0.889 | 0.941 | 1.000 | 1.000 | 0.000 |
| Isolation Forest | 0.393 | 0.496 | 0.439 | 0.403 | 0.886 | 0.053 |
| LSTM Autoencoder | 0.443 | 0.399 | 0.420 | 0.351 | 0.831 | 0.035 |

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
./run_all.sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The compact source config, data hash, environment versions, manifests, metrics,
timing, and feature importance are stored in `results/baseline_reference/`.

## Public Dataset Status

No public dataset is included, downloaded, preprocessed, or evaluated by this
release. CERT r4.2 is the first planned acquisition after external storage is
available. CICIDS2017 follows, then CSE-CIC-IDS2018 for transfer evaluation.

## Known Limitations

- All reported model metrics are synthetic and do not establish production or
  public-benchmark performance.
- The near-perfect Random Forest ranking reflects strong generator separability
  and must not be generalized to real activity.
- The release freezes one development seed; the protocol requires five seeds
  for later final selected public-data configurations.
- Public-dataset schemas can only be finalized after validating the actual files
  and their bundled documentation.
- Explanations are model/context summaries, not causal findings.
