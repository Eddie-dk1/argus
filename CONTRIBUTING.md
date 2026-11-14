# Contributing

Argus is an academic research prototype. Contributions should strengthen a
reproducible research question without adding private data, local artifacts, or
unsupported performance claims.

Argus has one planned Spectra Lab workspace. Use the
[implementation plan](docs/spectra-lab-implementation-plan.md),
[roadmap](docs/roadmap.md) and [design rules](docs/design.md).
EventLens is the historical name for Event Investigation, not a separate service.

The permanent [repository rules](AGENTS.md) apply to every Argus stage, dataset,
model, interface and thesis experiment, including S3–S8. Name each stage's
quality contribution: a new experiment, diagnosis, or preservation check.
An API/UI change alone is not an improvement in detection quality.

Current priority: the [seven-part S4 quality plan](docs/s4-model-quality-plan.md).
It is approved but not executed. Its bounded experiments precede the Spectra
UI work; implementation still follows one separately verified part per request.

## Before Changing an Experiment

Project-owner priority: improve detection quality through justified, testable
changes. Honest evaluation takes precedence over a larger metric. Preserve all
declared seeds and negative results; report precision, recall and false alarms
alongside F1. Do not improve a score by relabeling examples, changing the split,
dropping difficult cases, exposing target/provenance to the model, or choosing
parameters on test. Add dropout, regularization, features or model complexity
only for a stated hypothesis and evaluate their effect on development data.
Once a final test has been examined, record that exposure; later model changes
need separate development validation and an independent holdout for a new
confirmatory claim. The same used test may only provide exploratory evidence.
The current CERT r4.2 test was consumed by the
[frozen final evaluation on 2026-11-13](docs/cert-r4.2-final-evaluation.md).

1. State the hypothesis, control, scenario, target, split unit, evaluation
   cohort, objective, experiment budget and seeds before changing training,
   features, architecture or calibration.
2. Add the bounded search space to a versioned config before opening test
   results. Retain failed/interrupted trials within the declared budget.
3. Fit encoders, scalers, profiles, sampling, and feature selection on train
   only. A named causal observation policy may update personal context from
   previously observed held-out inputs after emitting their features, without
   labels; it must not fit on complete held-out history. See the
   [CERT v3 amendment](docs/cert-r4.2-causal-history.md#правило-доступности-данных).
4. Use validation for configuration, epoch, and threshold selection.
5. Preserve target, identifier, source-file, and answer-key fields as provenance
   unless the feature map explicitly permits them.

Report PR-AUC and class support as well as precision/recall/F1/false alarms;
accuracy is secondary. Retain regressions, weak scenarios and all seeds.
Source-backed data corrections require a named version. Reused development
folds and the consumed test are explicit evidence limitations; independent
confirmation requires independent data. Separate seed variation from fold
variation, and never pool cross-fold score rankings into PR-AUC.

Use the common model interface while selecting winners separately for CERT,
CICIDS2017 and CSE-CIC-IDS2018, with separate weights and feature contracts.
HGB remains the CERT research candidate; RF/IF/LSTM remain comparisons. Change
the active model only after a predeclared quality/stability/cost comparison.
Sequence models require justified event ordering and exact window membership.

The [three-part S3 plan](docs/s3-frozen-inference.md) proceeds one part per
request, without new research fits or calibration. Tiny generated fixture
models test functionality only. Snapshot the existing worktree before each
part and save a report and independently applicable incremental patch; do not
combine previous uncommitted work into a single commit.

Read [`docs/experimental-protocol.md`](docs/experimental-protocol.md) before
changing splits, features, thresholds, or metrics.

## Verification

For documentation-only changes, check Markdown links, anchors, consistency and
`git diff --check`; do not run training or regenerate reference artifacts.

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
Run synthetic regression in a separate temporary project copy when verifying
frozen inference so the existing reference and research artifacts stay intact.

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
