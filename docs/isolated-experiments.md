# Independent experiments — synthetic foundation

Implemented 2026-11-12; current status updated 2026-11-14. This is the supported
path for new isolated synthetic/CERT experiments. `run_all.sh` and the original training
CLIs remain the legacy reference workflow; they still update shared root
outputs and evaluate test. Do not use them for candidate searches.

## Commands

Create an artifact directory on an existing mounted disk (or a temporary local
folder for tiny tests). The CLI requires this root to exist and never creates
a missing mount. Each command creates a new UUID directory; no latest-file
lookup or root predictions are used.

```bash
python3 src/experiment.py --root "$ARTIFACT_ROOT" prepare --config configs/synthetic_v0.2.json
python3 src/experiment.py --root "$ARTIFACT_ROOT" candidate --preparation "$PREPARATION" --model random_forest
python3 src/experiment.py --root "$ARTIFACT_ROOT" candidate --preparation "$PREPARATION" --model isolation_forest
python3 src/experiment.py --root "$ARTIFACT_ROOT" candidate --preparation "$PREPARATION" --model lstm_autoencoder
# Only after choosing/fixing configuration using validation:
python3 src/experiment.py --root "$ARTIFACT_ROOT" final --preparation "$PREPARATION" --candidate "$CANDIDATE"
python3 src/experiment.py --root "$ARTIFACT_ROOT" compare "$EVALUATION_RF" "$EVALUATION_IF" "$EVALUATION_LSTM"
```

Variables above stand for paths printed by the corresponding previous command.
Candidate accepts an optional `--config` for model settings/seeds; generator,
split and feature configuration must match preparation. The initial isolated
protocol uses validation F1, recall and FPR, with validation percentile threshold
search. LSTM length is fixed by preparation. Its training filter remains
endpoint-normal, matching v0.2 semantics; it is not strict all-members-normal.
No automatic study or winner selection is implemented at this stage.

Named CERT extensions (2026-11-14): `candidate --model hist_gradient_boosting`
accepts `baseline.hist_gradient_boosting` settings through `--config`; it fits
all labeled train days like RF and requires `early_stopping=false` to avoid
an internal random validation split. LSTM accepts the explicit
`lstm_autoencoder.score_policy="endpoint_mean_mse_v2"`; the default remains
the original window-mean score. Its optimization loss and architecture stay
the same; epoch/threshold selection uses the configured score on validation.
Both extensions reuse preparations with identical generator/split/features.
The [bounded CERT comparison](cert-r4.2-model-variants-v5.md) runs all four
existing feature policies with separate inner calibration and immutable
outer assessments. This does not replace the synthetic three-model reference
or authorize a new independent claim on the used CERT test.

For LSTM, optional `candidate --diagnostics` records end-of-epoch reconstruction
MSE on fit windows and on strictly all-background validation windows, alongside
endpoint background/positive MSE. `history.json` also records online training
loss, epoch/selection durations, selected epoch and stopping reason. Extra
scoring preserves the CPU RNG so diagnostics do not change shuffled training
order; a three-epoch regression test checks identical weights/predictions.
Diagnostic labels only define reporting cohorts; they do not enter model inputs
or change checkpoint/threshold selection. Additional passes cost time and are
reported separately. `progress.json` describes the latest completed epoch,
which can differ from the selected checkpoint. No test records are accessed.

The [CERT robustness runner](cert-r4.2-robustness.md) additionally separates
inner calibration from outer development assessment and repeats both declared
history policies across five model seeds. CV preparations contain
`assessment.csv` instead of the real `test.csv`; the candidate does not read
that assessment file. The outer evaluator applies the saved checkpoint and
threshold without selection or fitting. Every fold refits its own preprocessing
state. New tabular training IDs use Unicode NPZ arrays readable without pickle;
legacy local model bundles remain an explicitly trusted artifact format.

## Artifact guarantees

- Preparation owns generated raw, split records, frozen feature state and schema.
  Event IDs combine raw SHA-256 and the original CSV record position, assigned
  before sorting. Relocating the root preserves IDs. Changed raw changes IDs.
- Equal timestamps use event ID as a deterministic tie-breaker. This is a new
  ordering contract; old published synthetic metrics are not regenerated or
  silently claimed identical. Raw record index is a logical CSV record, not
  physical line number. For public data, source-qualified IDs are still adapter
  work.
- Sequences save ordered `member_ids` and endpoint `item_ids`. Models compare
  identical endpoint targets, although tabular rows and LSTM windows differ.
- Candidate opens only train/validation and their metadata. It does not open,
  hash, load or score test records. Reading the preparation completion manifest
  binds an opaque test checksum but exposes no test records or metrics.
- Candidate stores actual training and validation numerical arrays, ordered
  columns, IDs/membership, train/validation records, weights, threshold,
  feature state, config, environment, validation metrics and threshold search.
- Final evaluation verifies the frozen candidate and preparation identity,
  reloads weights, applies the saved threshold without fitting, and saves a
  separate evaluation with test inputs, IDs, scores, decisions and metrics.
- Compare validates checksums, preparation/snapshot/target and exact item/label
  populations. It never merges stale shared CSV outputs.
  Cross-preparation comparisons require explicit `compare --allow-different-preparations`;
  snapshot, target and every item/label must still match. This supports declared
  feature-policy controls and does not permit different cohorts or labels.
- `complete.json` binds artifact hashes; incomplete or changed inputs are
  rejected. These are application-level immutable snapshots, not filesystem
  write protection or signatures against malicious tampering. Load only locally
  trusted model artifacts (joblib is not an untrusted interchange format).
- Status is running, complete, failed, or interrupted (KeyboardInterrupt).
  Hard process kills can leave running without a completion marker: such a run
  is incomplete and rejected, not resumable. Automatic recovery belongs to the
  later job manager.
- UUID roots isolate concurrent command outputs. Thread/process resource budgets
  and scheduling are later runtime work. Preparation never uses repository data
  or result outputs; compatibility imports may create empty legacy directories.

The preparation config must be frozen before final evaluation. Repeated explicit
final calls are permitted for reproducibility checks; examining test and retuning
invalidates its use as an untouched holdout. A command boundary cannot prevent
researcher misuse of the returned results.

## Validation

Tiny generated data (600 events), all three models, one LSTM epoch. Tests remove
the prepared test file while candidates run, then restore it for final evaluation;
check reloaded scores/decisions, exact membership, tied timestamp determinism,
changed-input rejection and incompatible-cohort rejection. These checks validate
workflow contracts, not model quality. CERT training and baseline reference
artifacts are untouched.

## CERT final update — 2026-11-13

The [frozen CERT final](cert-r4.2-final-evaluation.md) evaluated all 30 declared
candidates without fitting/recalibration. Final artifacts now include exact
NPZ scores/decisions alongside CSV, timing and environment metadata. The final
driver checks frozen inference source, dependency versions, all checkpoint and
threshold bindings, and records first test access. Current CERT test is used;
later model changes require an independent holdout for a confirmatory claim.

## CERT operating-point extension — 2026-11-14

The [step-5 runner](cert-r4.2-operating-point-v6.md) reuses existing
development preparations and a predeclared tabular shortlist across five
model seeds. It creates a separate `cert_operating_candidate_*` bundle with
copied weights/config/state, exact calibration inputs/scores, a complete
threshold-boundary table and `operating_points.json`. The legacy F1 candidate
and its thresholds remain unchanged. Profile selection uses calibration only;
assessment opens after the new bundle is complete. Infeasible recall and
outer budget violations are recorded without fallback or re-selection.

`cert_operating_assessment_*` stores exact outer inputs/scores/decisions for
every profile, metrics, timing and frozen-candidate hashes. The series saves
all declared jobs, frozen shortlist, code/config/environment snapshots and
all seed/fold results. Pooled decisions remain separate per seed; ranking
metrics remain within folds. These bundles preserve the development-only
scope and do not implement a new real-test command, deployment or general
Spectra job/search service.
