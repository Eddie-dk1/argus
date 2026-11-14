# Data Storage Policy

Argus keeps code and small reference artifacts in Git while storing all raw,
interim, and processed datasets outside Git. The storage root is selected with
the `ARGUS_DATA_ROOT` environment variable.

Integration update: 2026-10-27. The [roadmap](roadmap.md#resume-after-ssd) gives
the post-SSD order. Planned Spectra paths below are not yet current behavior.

## Safe default

If `ARGUS_DATA_ROOT` is not set, Argus uses the ignored `data/` directory in
the repository. This default is suitable for the small synthetic scenario.

For public datasets on macOS, first confirm the real mounted volume in Finder
and inspect it. Replace the example name with the actual SSD mount:

```bash
df -h "/Volumes/ArgusData"
```

Only after confirming this is the mounted SSD, create its data root and export
the path. Never create a missing volume mountpoint instead of connecting it:

```bash
mkdir -p "/Volumes/ArgusData/argus-data"
export ARGUS_DATA_ROOT="/Volumes/ArgusData/argus-data"
```

The configured root must already exist. If the SSD is missing or unmounted,
Argus stops with an error instead of creating a similarly named directory and
writing large files to the internal drive.

The volume name `ArgusData` is only an example. Do not add `.env` or a personal
mount path to Git.

## Directory layout

```text
$ARGUS_DATA_ROOT/
├── synthetic/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── manifests/
├── cert_r4_2/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── manifests/
├── cicids2017/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── manifests/
└── cse_cic_ids2018/
    ├── raw/
    ├── interim/
    ├── processed/
    └── manifests/
```

- `raw`: immutable publisher files and bundled documentation;
- `interim`: normalized or joined rebuildable data;
- `processed`: model-ready features, splits, and sequences;
- `manifests`: acquisition listings, hashes, schemas, and quality reports.

Raw files must never be edited in place. Every interim or processed output must
be reproducible from raw inputs plus a committed configuration.

## Preflight check

Validate the configured filesystem before a download:

```bash
python3 src/check_data_storage.py
python3 src/check_data_storage.py \
  --scenario cse_cic_ids2018 \
  --required-gib 250 \
  --create-scenario-directories
```

`--required-gib` is intentionally supplied by the operator after inspecting the
remote manifest. The command returns exit code `2` when free space is
insufficient.

## Capacity procedure

Before downloading:

1. obtain a publisher-side file listing;
2. sum only the selected files;
3. estimate extracted and normalized copies;
4. include processed features, sequence arrays, models, and plots;
5. retain a safety margin;
6. run `check_data_storage.py` with the resulting requirement.

Do not use the capacity of the advertised archive alone as the SSD requirement.
CSV expansion, preprocessing, and temporary files can require substantially
more free space.

## First acquisition after SSD

Start with CERT r4.2 and matching answers, not the complete CERT collection or
CSE bucket. Retain the bundled README, inventory, license and SHA-256 before
schema/label acceptance. Inspect the chosen raw directory using:

```bash
python3 src/inspect_dataset.py --scenario cert_r4_2
```

Storage checks do not download/accept data. A successful inspection does not
mean user-day preprocessing or model training is implemented.

## Current results and planned Spectra artifacts

ARGUS_DATA_ROOT currently moves scenario data only. RESULTS_DIR in src/utils.py
still points into the repository; root predictions/checkpoints can be overwritten.
Result-root isolation must be implemented before studies. An undocumented
results environment variable does not change current behavior.

Planned large artifacts use interim/spectra for snapshot events/labels and
processed/spectra for immutable features/state, memberships, models, predictions,
references and scenarios. Source/preparation manifests use manifests/spectra.
Lightweight run/study metadata references logical roots and relative paths.

Budget caches, query spill, exports and scenario copies separately. Include
peak staging/publication use, not just final sizes. Do not duplicate a dataset
per trial. Write staging files on the target volume, validate and publish the
complete manifest last; cross-volume references require independent checks.

Missing SSD stops writes and becomes an unavailable-source state in the planned
viewer, without a hidden large copy on the internal drive.

## Backup and deletion policy

- Publisher data can be re-downloaded; manifests and checksums are the critical
  local records.
- Raw inputs are immutable and deleted only deliberately after confirming that
  they can be reacquired.
- Interim and processed outputs are caches and may be rebuilt.
- Experiment configs, metrics, lightweight plots, and schema reports belong in
  versioned releases when they are safe and small.
- Personal paths, datasets, model weights, and large prediction files never
  belong in GitHub releases.
