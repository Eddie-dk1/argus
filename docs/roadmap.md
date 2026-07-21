# Roadmap

Argus is developed as a staged academic prototype. A stage is complete only
when its data/config, code, tests, metrics, limitations, and release notes agree.

## Completed

### argus-v0.1.0 — public foundation

- project rebranding and honest academic positioning;
- repository cleanup, MIT license, contribution guide;
- initial architecture, dataset notes, roadmap, and reference artifacts.

### argus-v0.2.0 — reproducible synthetic baseline

- deterministic, more realistic synthetic generator;
- single chronological split before fitted preprocessing;
- causal scenario-specific features and identifier/target exclusions;
- Random Forest, Isolation Forest, and LSTM Autoencoder compared on the exact
  same endpoint cohort;
- validation-only epoch and threshold selection;
- metrics including PR-AUC, FPR, support, prevalence, and timing;
- structured run artifacts, hashes, manifests, error analysis, and reference
  snapshot;
- official-source research for CERT r4.2, CICIDS2017, and CSE-CIC-IDS2018;
- external SSD policy, capacity preflight, schema inspector, chunked loader, and
  tiny contract fixtures;
- feature map, experimental protocol, and working literature review.

## Prepared, but Intentionally Not Claimed as Implemented

The repository has public-dataset contracts but contains no real public data,
preprocessed public features, trained public-data models, or public-data model
metrics. Those claims begin only after the files are acquired and their bundled
documentation is validated.

## Next Sequential Milestones

### argus-v0.3.0 — CERT r4.2 behavioral scenario

1. Acquire the selected official r4.2 release to external storage and freeze its
   file/checksum manifest.
2. Reconcile the provisional contract with the bundled README and answer keys.
3. Implement user-group `60/20/20` splits without user overlap.
4. Build user-day and ordered behavioral features without label history.
5. Train/evaluate tabular baselines, then adapt the sequence model.
6. Repeat the selected configuration across seeds `7, 21, 42, 84, 168`.
7. Analyze false positives/negatives and publish a compact release snapshot.

### argus-v0.4.0 — CICIDS2017 network scenario

- accept and validate the publisher CSV manifest;
- implement the fixed capture-day split;
- document any corrections prompted by the troubleshooting literature;
- build flow-specific features and compare the model families fairly;
- report attack-family recall and capture-day shift.

### argus-v0.5.0 — CSE-CIC-IDS2018 and transfer

- freeze source-day splits before tuning;
- reproduce an independent within-dataset result;
- test the compatible CICIDS2017 feature subset and frozen threshold;
- document what does and does not transfer.

### argus-v0.6.0 — final modeling and explainability

- add one justified boosting baseline after the data scenarios stabilize;
- use bounded, committed, validation-only hyperparameter searches;
- select final configurations using multi-seed evidence;
- add SHAP for the selected tabular model and reconstruction-based temporal
  explanations for the sequence model;
- perform per-family and cross-dataset error analysis.

### argus-v1.0.0 — diploma integration

- freeze final scenarios, metrics, tables, and limitations;
- build a limited Streamlit demonstration over prepared artifacts;
- prepare the reproducibility guide and defense demo;
- publish the final selected release without datasets or private artifacts.

## Release Gate

Every release must pass:

- unit and schema-contract tests;
- shell/Python syntax checks;
- documented reference command from a clean environment;
- `git diff --check` and review of ignored/untracked files;
- confirmation that no dataset, personal path, secret, or model weight is
  staged;
- release notes stating supported scenarios and known limitations.
