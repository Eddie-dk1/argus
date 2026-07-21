# Contributing

Argus is an academic research prototype. Contributions should strengthen a
reproducible research question without adding private data, local artifacts, or
unsupported performance claims.

## Before Changing an Experiment

1. State the scenario, target, split unit, evaluation cohort, and validation
   objective.
2. Add the bounded search space to a committed config before opening test
   results.
3. Fit encoders, scalers, profiles, sampling, and feature selection on train
   only.
4. Use validation for configuration, epoch, and threshold selection.
5. Preserve target, identifier, source-file, and answer-key fields as provenance
   unless the feature map explicitly permits them.

Read [`docs/experimental-protocol.md`](docs/experimental-protocol.md) before
changing splits, features, thresholds, or metrics.

## Verification

For ordinary changes, run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
bash -n run_all.sh
git diff --check
```

For changes to generation, preprocessing, features, training, or evaluation,
also run:

```bash
./run_all.sh
python3 src/build_reference_snapshot.py
```

Update reference artifacts only when the result change is intended and explain
the change in the release notes or commit message.

## Data and Artifact Rules

Do not commit:

- downloaded datasets or extracted copies;
- model weights, full predictions, or intermediate arrays;
- personal mount paths, `.env`, secrets, reports, or local previews;
- timestamped experiment directories;
- metrics from incomparable cohorts.

Tiny fabricated fixtures are allowed when they test schemas and contain no
publisher records. Compact configs, manifests, metrics, and selected plots may
be committed as an explicit release reference.

## Documentation and Claims

- Link dataset facts to publisher or owner documentation.
- Cite primary papers for technical decisions.
- Name exact dataset versions and corrections.
- State class support and prevalence with imbalanced metrics.
- Treat synthetic and benchmark performance as limited evidence.
- Record negative findings and undefined metrics honestly.

Keep changes small enough to review and update README/docs whenever a command,
contract, or supported scenario changes.
