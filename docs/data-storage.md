# Data Storage Policy

Argus keeps code and small reference artifacts in Git while storing all raw,
interim, and processed datasets outside Git. The storage root is selected with
the `ARGUS_DATA_ROOT` environment variable.

## Safe default

If `ARGUS_DATA_ROOT` is not set, Argus uses the ignored `data/` directory in
the repository. This default is suitable for the small synthetic scenario.

For public datasets on macOS, create a directory on the mounted SSD and export
its absolute path:

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
