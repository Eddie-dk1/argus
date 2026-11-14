# Dataset Research and Acquisition Plan

Last verified: 2026-07-21.

Project status updated 2026-11-14. CERT acceptance, label/group policy,
daily/sequence preparation, frozen final and five quality increments are
complete. [S3](s3-frozen-inference.md) is complete; [S4](s4-model-quality-plan.md)
is planned. Network-dataset publisher facts retain the earlier review date.

Argus does not store public cybersecurity datasets in Git. Public data must be
downloaded to local ignored storage, while the repository keeps only source
code, schemas, acquisition notes, checksums/manifests, and small synthetic test
fixtures.

This document separates facts stated by dataset publishers from project
decisions. File names and checksums must be verified against the publisher's
manifest or bundled README after acquisition. CERT r4.2 is now on external
storage; this does not imply network-dataset acquisition or model evaluation.

## Project Decisions

The planned diploma sources use the smallest useful published form of each
dataset, in this acquisition order:

1. `CERT r4.2` activity logs and its answer key for the behavioral scenario;
2. `CICIDS2017 MachineLearningCSV` flow files for the first network scenario;
3. the `Processed Traffic Data for ML Algorithms` CSV prefix from
   `CSE-CIC-IDS2018` for the second network scenario.

Raw PCAPs, host images, and the complete CERT release collection are optional
follow-up sources. They are not required for the first Argus model comparison.
This keeps acquisition and preprocessing feasible without weakening the stated
tabular/sequence research scope.

CERT r4.2 and matching answers were accepted first. Acquisition of the whole
collection remains unnecessary. The laboratory uses synthetic correctness
fixtures and the accepted CERT baseline; network sources remain subsequent
milestones with their own acceptance gates.

Every adapter declares its analysis unit/capabilities. Synthetic events, CERT
user-days and flows are not interchangeable. Preserve unit-to-source mappings
and enable raw scenarios only when the affected units can be recomputed from
supported fields. Viewing results does not imply scenario replay support.

See the [post-SSD roadmap](roadmap.md#resume-after-ssd) and
[implementation plan](spectra-lab-implementation-plan.md).

## CERT Insider Threat Test Dataset

### Publisher facts

- Publisher: Carnegie Mellon University Software Engineering Institute (SEI).
- Persistent identifier: `10.1184/R1/12841247.v1`.
- Official landing page: [SEI Insider Threat Test Dataset][cert-sei].
- Repository record: [CMU KiltHub][cert-kilthub].
- License shown by KiltHub: `CC BY 4.0`.
- The KiltHub record reports `87.23 GB` for downloading the complete collection
  of releases.
- The data is synthetic. It contains both background activity and synthetic
  malicious actors; it must not be described as production enterprise data.
- Releases are organized by data-generator version. Later releases generally
  include a superset of generator functionality, but are not interchangeable
  experimental samples.
- The official `answers.tar.bz2` contains malicious scenarios and user
  identifiers. Each release also contains its own README.

### Argus scope

Argus uses the accepted `r4.2` snapshot and matching answers. Activity inputs
are logon/logoff, device, file, HTTP and email. Acceptance reconciled their
schemas with the bundled README; organizational/psychometric metadata remain
provenance/population information rather than default model inputs.

The primary experimental entity is an active account-day. Labels follow the
fixed observable-membership policy, with incident user and logged account kept
separate. LSTM inputs are five active account-days with exact ordered membership;
calendar gaps are retained through history features. This is account transfer,
not a raw-event-sequence or temporal forecasting claim.

### Acquisition checklist

- [x] Select only `r4.2` and `answers.tar.bz2` in KiltHub instead of using
  `Download all`.
- [x] Save KiltHub metadata and the bundled `r4.2` README with the local raw
data.
- [x] Record file names, byte sizes, and SHA-256 hashes in a local manifest.
- [ ] Confirm that the license and required attribution have not changed.
- [x] Validate the answer-key user identifiers and scenario dates before
  generating labels.

### Known limitations and controls

- Synthetic organizational behavior is useful for controlled evaluation but
  cannot establish real-world prevalence or operational false-positive rates.
- Malicious activity is extremely sparse. Accuracy is therefore not a primary
  metric.
- Multiple events from the same user are correlated. Random row splitting
  would leak user and temporal patterns; Argus will use time-aware splits and
  report user overlap explicitly.
- Organizational attributes and activity categories can encode scenario
  shortcuts. Identifier-like columns are excluded from model features unless a
  documented ablation justifies them.

## CICIDS2017

### Publisher facts

- Publisher: Canadian Institute for Cybersecurity (CIC), University of New
  Brunswick.
- Official page: [CICIDS2017][cicids2017].
- Capture period: Monday 2017-07-03 through Friday 2017-07-07.
- Monday contains benign activity; attacks occur on the remaining days.
- Published sources include PCAPs, generated labeled flows, and
  `MachineLearningCSV.zip` flow-level CSV files extracted with CICFlowMeter.
- The official page reports raw daily sizes of `11.0 + 11 + 13 + 7.8 + 8.3 GB`
  (`51.1 GB` total across the five listed days).
- The traffic includes common protocols and multiple attack families including
  brute force, DoS/DDoS, Heartbleed, web attacks, infiltration, botnet, and
  port scanning.
- The publisher asks users to cite the dataset paper by Sharafaldin, Habibi
  Lashkari, and Ghorbani.

### Argus scope

The first implementation will download only `MachineLearningCSV.zip`. Raw PCAPs
will be added only if Argus later evaluates a new feature extractor. The loader
will preserve source day/file metadata before concatenation.

Primary experiments will include:

- binary detection: `BENIGN` versus any attack;
- per-attack-family error analysis;
- a chronological/day-based split as the main estimate;
- an explicitly labeled stratified split only as a secondary comparability
  experiment, never as the sole result.

### Acquisition checklist

- [ ] Download `MachineLearningCSV.zip` from the official CIC page directly to
  the external data root.
- [ ] Record archive and extracted-file sizes and SHA-256 hashes.
- [ ] Preserve the original daily CSV files as immutable raw inputs.
- [ ] Normalize whitespace in column names and labels without overwriting raw
  data.
- [ ] Count duplicate rows, missing values, positive/negative infinity, constant
  columns, repeated headers, and class distribution per source file.

### Known limitations and controls

The peer-reviewed CICIDS2017 troubleshooting study found problems in attack
simulation, flow construction, feature extraction, labeling, and benchmarking;
the authors report that more than 20% of original traffic traces required
reconstruction or relabeling in their corrected version. Argus will therefore:

- treat very high scores as a reason for leakage/shortcut investigation;
- report results on the original published CSVs with that version named;
- retain source day, timestamp, endpoint, and label provenance;
- run duplicate and non-finite-value checks before splitting;
- avoid using direct identifiers or source-day proxies as ordinary features;
- document any corrected/relabelled derivative as a distinct dataset variant.

See [Engelen, Rimmer, and Joosen (2021)][cicids-troubleshooting] and its
[extended documentation][cicids-troubleshooting-extended].

## CSE-CIC-IDS2018

### Publisher facts

- Publishers: Communications Security Establishment (CSE) and CIC/UNB.
- Official page: [CSE-CIC-IDS2018 at UNB][csecic2018].
- Distribution: public S3 bucket `s3://cse-cic-ids2018/` in
  `ca-central-1`; an AWS account is not required.
- AWS Registry record: [A Realistic Cyber Defense Dataset][csecic2018-aws].
- The testbed contains five departments, 420 machines, and 30 servers, with 50
  machines in the attacking infrastructure.
- Publisher materials describe seven attack scenarios and more than 80
  CICFlowMeter-V3 traffic features.
- The bucket contains daily raw network/host logs and processed ML CSV data.
- The license permits redistribution and mirroring, but use or redistribution
  must cite the dataset and link to its AWS page.

### Argus scope

Argus will acquire only the processed ML CSV prefix during the first network
experiment. The complete bucket must not be synchronized accidentally.

After the SSD is mounted, inspect the remote manifest before any download:

```bash
aws s3 ls --no-sign-request --region ca-central-1 \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \
  --recursive --human-readable --summarize
```

Download only the inspected prefix:

```bash
aws s3 sync --no-sign-request --region ca-central-1 \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \
  "$ARGUS_DATA_ROOT/cse_cic_ids2018/raw/processed_traffic/"
```

### Acquisition checklist

- [ ] Save the remote `aws s3 ls` listing before downloading.
- [ ] Confirm free space for the selected objects, extracted/normalized copies,
  caches, and experiment artifacts.
- [ ] Download only the processed prefix first.
- [ ] Record per-object sizes, ETags, and local SHA-256 hashes.
- [ ] Preserve daily source files and add a source-file column during ingestion.
- [ ] Detect repeated header rows, duplicates, missing/non-finite values, label
  variants, and schema differences between days.

### Known limitations and controls

- CSE-CIC-IDS2018 also uses CICFlowMeter-V3, so the flow construction and
  feature-extraction concerns documented for CICIDS2017 must be checked rather
  than assumed absent.
- Attack types are tied to capture dates and infrastructure. Random row splits
  can leak those patterns and overestimate transferability.
- The primary evaluation will preserve chronology and compare feature/threshold
  transfer from CICIDS2017 rather than tuning independently on the final test
  split.

## Local Storage Budget

Before acquisition, calculate capacity from a manifest rather than adding only
the advertised archive sizes:

```text
required free space = selected downloads
                    + extracted immutable raw files
                    + normalized/interim files
                    + processed features/sequences
                    + model and evaluation artifacts
                    + safety margin
```

The project will not keep redundant raw copies unless they are needed to verify
checksums or reproduce extraction. Interim and processed data must be
rebuildable and may be removed independently from immutable raw inputs.

## Dataset Acceptance Gate

A scenario is ready for modeling only when all of the following are recorded:

- official source, license, citation, and dataset variant;
- immutable raw manifest with sizes and hashes;
- schema report and source-file provenance;
- label mapping and class distribution;
- missing, non-finite, duplicate, and repeated-header report;
- split policy and leakage analysis;
- preprocessing configuration and generated artifact manifest.

[cert-sei]: https://sei.cmu.edu/library/insider-threat-test-dataset/
[cert-kilthub]: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
[cicids2017]: https://www.unb.ca/cic/datasets/ids-2017.html
[csecic2018]: https://www.unb.ca/cic/datasets/ids-2018.html
[csecic2018-aws]: https://registry.opendata.aws/cse-cic-ids2018/
[cicids-troubleshooting]: https://doi.org/10.1109/SPW53761.2021.00009
[cicids-troubleshooting-extended]: https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html
