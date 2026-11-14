# Scenario Feature Map

Last reviewed: 2026-07-21.

Spectra integration rules added 2026-10-27. Publisher statements retain their
original review date; this amendment concerns future development contracts.

Argus uses scenario-specific feature sets. It does not force CERT behavioral
events and CIC network flows into one artificial universal table. Shared names
below express comparable concepts, not identical measurements.

Feature status:

- **required**: part of the first scenario implementation;
- **candidate**: evaluated through a declared ablation after the required set;
- **provenance only**: retained for joins, splitting, and analysis but excluded
  from default model inputs;
- **forbidden**: target leakage or post-event information.

All train-derived profiles, vocabularies, scalers, and baselines are fitted on
the train split only.

## Shared feature principles

### Tabular models

Tabular models receive stable event or window aggregates:

- volume and frequency;
- duration and timing;
- diversity/count features;
- ratios and changes from an entity's train-period baseline;
- low-cardinality semantic categories;
- missingness indicators where missing values carry operational meaning.

Raw identifiers are not encoded as arbitrary integers. Low-cardinality
categories use a documented one-hot representation; high-cardinality values
are transformed into counts, novelty indicators, or semantic groups.

### Sequence models

Sequence models receive a smaller feature set to avoid sparse identifiers and
unstable category vocabularies:

- relative time or time-of-day encoding;
- log-scaled volumes/durations;
- event/flow category;
- status and direction;
- novelty and behavioral deviation;
- compact rolling counts.

Every sequence is ordered by timestamp inside an entity and split. Padding and
masking policy must be stored in the feature manifest.

### Transformations

- heavy-tailed non-negative volumes: `log1p` before scaling;
- hour: sine/cosine cyclical encoding rather than a linear integer;
- ratios: protected division plus an explicit zero-denominator rule;
- infinities: converted to missing before imputation and counted in the quality
  report;
- continuous features: scaler fitted on train only;
- unknown categories: explicit unknown bucket, never a new fit on validation or
  test.

## Synthetic scenario

### Provenance and targets

| Field | Status | Use |
|---|---|---|
| `timestamp` | provenance only | chronology, windows, split report |
| `user_id` | provenance only | entity grouping and explanations |
| `source_ip`, `destination_ip` | provenance only | novelty/diversity derivation |
| `label` | target | binary anomaly target |
| `anomaly_type` | target metadata | per-family error analysis |

### Required tabular features for `argus-v0.2.0`

| Group | Features |
|---|---|
| Traffic | `log_bytes_sent`, `log_bytes_received`, `bytes_ratio`, `log_session_duration` |
| Time | `hour_sin`, `hour_cos`, `is_weekend`, `is_off_hours` |
| Authentication | `failed_logins`, `is_login`, `is_login_failure` |
| Reach | `distinct_destinations`, `source_ip_novel_for_user`, `destination_ip_novel_for_user` |
| Privilege | `privileged_action` |
| Window | prior-event count, failed-login sum, bytes sent/received sum, unique source/destination count, IP-change count |
| Semantic category | one-hot `event_type`, `login_status`, `protocol`, `location`, `dst_service` |
| Stable profile | one-hot `department`, `user_role`, `device_type` |

The window length and whether it is event-count- or time-based are resolved in
the committed scenario config. Only earlier events inside the same split may
contribute to a row.

### Required sequence features

- `log_bytes_sent`;
- `log_bytes_received`;
- `log_session_duration`;
- `hour_sin`, `hour_cos`, `is_weekend`, `is_off_hours`;
- `failed_logins`, `distinct_destinations`, `privileged_action`;
- event/login/protocol semantic codes with an unknown bucket;
- source/destination novelty;
- compact prior-window event and failed-login counts.

Department, role, device, user, and raw IP codes are excluded from the default
sequence set. They can be examined only in a named ablation.

## CERT r4.2 behavioral scenario

The official collection contains versioned synthetic activity data and an
answer key. Peer-reviewed work on r4.2 describes separate logon, device, file,
email, HTTP, and LDAP/organizational inputs and frequently models a `user-day`.
The core schema was reconciled with bundled r4.2 documentation on 2026-11-12.
The target and active account-day population are fixed by
[label policy v1](cert-r4.2-label-policy.md). Complete-day aggregates are scored
at the end of the day. Answer membership and incident/group metadata are never
model features.

### Required user-day tabular groups

| Source | Required derived features |
|---|---|
| Logon | logon/logoff counts, failed/abnormal status if present, first/last activity, off-hours count, distinct PCs, new-PC indicator |
| Device | connect/disconnect counts, off-hours use, duration if derivable, new-device/PC indicator |
| File | operation count, distinct files, extension groups, removable-media operations, off-hours operations, volume if present |
| HTTP | request count, distinct domains/URLs, upload/download indicators if present, uncategorized/new-domain rate, off-hours activity |
| Email | sent/received counts, internal/external recipients, recipient diversity, attachment count, size aggregates, off-hours activity |
| Organization | department/role group and employment state for analysis; default-model inclusion requires an ablation |
| Cross-source | total events, active-source count, source proportions, first/last event, inter-event-time statistics |

### Behavioral deviation features

Global references fit train only. Personal references use past observations
under an explicitly named policy: the original policy adapts train accounts
only; [v3](cert-r4.2-causal-history.md) permits the same emit-before-update rule
in every role, without labels. Candidate context families include:

- deviation from the user's median and robust spread;
- rolling 7-day and 30-day counts/sums when enough history exists;
- change from department/peer-group baseline;
- new PC, domain, recipient, removable-device, or after-hours pattern;
- consecutive off-hours-active days;
- activity-source transition frequencies.

Cold-start and insufficient-history flags accompany every deviation value.

[Step 3 file/USB ablation](cert-r4.2-file-usb-v4.md) implements two optional
75-column preparations over v3's unchanged 71 inputs. Three added deviations
use `log1p(file_events)`, `log1p(connect_count)` and
`log1p(device_paired_seconds)`, plus a fallback indicator. References are either
past active days in `[date-30,date)` with at least five observations, or the
first five observed active days frozen thereafter. Fixed IQR floor 1 and clip
20 apply to these new deviations only; fallback/scaler fit inner train only.
Positive days stay in history. These are explicit candidate policies, not a
change to existing frozen configurations or a guarantee of better detection.
File-event counts do not establish copying to USB; paired device duration
covers matched connections within one PC/day, not complete device occupancy.

### Required sequence representation

Two sequence granularities will be compared without mixing them:

1. ordered activity codes inside a user-day with relative time and off-hours
   state;
2. sequences of daily aggregate vectors across consecutive user-days.

Activity codes must retain semantic category and action, not arbitrary row IDs.

### Forbidden CERT features

- answer-key identity or scenario code;
- current or previous ground-truth malicious-day indicator;
- any field created after the event using knowledge of the incident;
- user identifier encoded directly as a predictive number;
- a global behavioral baseline fitted using validation/test history.

Ground-truth history is deliberately forbidden even though some published
experiments use it: it would not be available for previously undetected
behavior and the cited authors acknowledge this operational limitation.

## CICIDS2017 network scenario

CIC publishes more than 80 CICFlowMeter features. Argus will preserve the
publisher column names in raw data and map accepted columns into the groups
below after schema inspection.

### Required flow-level tabular groups

| Group | Examples/concepts |
|---|---|
| Flow size | duration, forward/backward packet totals, forward/backward byte totals |
| Packet length | min/max/mean/std and directional packet-length statistics |
| Rate | bytes per second, packets per second, forward/backward packet rates |
| Inter-arrival | flow, forward, and backward IAT totals and summary statistics |
| TCP behavior | flag counts, initial window bytes, header-length aggregates |
| Activity | active and idle min/max/mean/std |
| Direction/bulk | down/up ratio, bulk and subflow counts/bytes where quality checks pass |
| Service | protocol plus destination-port semantic group |

All rate and ratio columns are recomputed or consistency-checked when the raw
components are present. Constant, duplicate, malformed, or non-finite columns
are rejected and listed in the feature manifest.

### Candidate context features

- flows per source/destination in a causal time window;
- destination-port diversity;
- new-destination and new-service indicators;
- failed/short connection burst counts;
- byte and packet deviation from source train baseline;
- inbound/outbound direction relative to the documented victim network.

### Sequence entity

CSV row order alone is not a valid sequence. Network sequences require a
declared entity such as source host, destination host, or bidirectional 5-tuple,
ordered by parsed timestamp. Argus will start with source-host windows and
compare a 5-tuple representation only if coverage is sufficient.

### Provenance-only or candidate identifier fields

- Flow ID;
- source/destination IP;
- exact timestamp and capture day;
- exact source port;
- source CSV name;
- attack schedule/time range.

Destination port is converted to a service group in the required set. Exact
destination port is candidate-only because it can become a scenario shortcut.

## CSE-CIC-IDS2018 network scenario

The required groups match CICIDS2017 where column semantics are compatible.
This common subset is the `cic_flow_core_v1` transfer feature set.

Additional CSE-CIC-IDS2018 columns are allowed only in a separate
`cse_extended_v1` feature set. The comparison must therefore include:

- within-dataset result with `cic_flow_core_v1`;
- within-dataset result with `cse_extended_v1`, if justified;
- frozen CICIDS2017-to-CSE transfer on `cic_flow_core_v1`.

Daily file name, attack date, AWS object key, victim identity, and known attack
schedule are provenance only.

## Explainability mapping

Features shown in an explanation must map to an analyst-readable reason:

| Feature pattern | Explanation family |
|---|---|
| repeated login failures | authentication burst |
| off-hours activity plus unusual volume | temporal behavioral deviation |
| new IP/PC/device/domain/recipient | entity novelty |
| high outbound bytes or ratio | possible exfiltration-like volume |
| many destinations/ports in a short window | scanning/reconnaissance-like behavior |
| abnormal duration/rate/IAT/flags | unusual network-flow dynamics |

An explanation states what the model observed; it must not assert that an
attack occurred merely because a feature is large.

## Feature acceptance gate

A feature set can be used in a final run only when its manifest records:

- source columns and formulas;
- output dtype and missing-value rule;
- fitting split and fitted-state hash;
- identifier/leakage classification;
- tabular/sequence applicability;
- stability and finite-value checks;
- feature-set name and version;
- ablation result for candidate additions.

## Planned Spectra Integration

Event Investigation and Scenario Checks share the same core formulas/state.

- Preserve source IDs before aggregation/sorting and store analysis units plus
  exact item/event membership used by each model.
- Separate raw values, transformed features, numerical inputs and additional
  timeline context. Record order, dtype, units and transformation rules.
- LSTM membership is not the whole feature-dependency graph: rolling history
  and frozen train profiles may extend beyond the displayed window.
- Start synthetic patches with bytes_sent. Recompute logs, ratios, affected
  histories and scaling from a separate raw copy. Never change only a scaled
  column while leaving dependent features inconsistent.
- The current prior_log_bytes_sent_sum sums log1p volumes; it is not log1p
  of the raw sum. Reproduce the implemented formula.
- Require no-op and full-history oracle checks; scenario execution never refits
  profiles, encoders or scalers.
- With primary no-user-overlap CERT splits, held-out users have no personal
  train profile. Use frozen group/global fallbacks and cold-start flags, never
  a personal baseline fitted on their full held-out histories. Causal online
  history would require a separate declared policy.
- CERT items are user-days mapped to source events. Raw changes require
  reaggregation; synthetic bytes_sent is not a universal input field.
- Reference normality/exclusions are distinct from detector targets. Predicted
  normal examples are not ground-truth-confirmed normal examples.
- Freeze features and window length in the initial study so comparisons use
  one common evaluation population.

See the [implementation plan](spectra-lab-implementation-plan.md) and
[experimental protocol](experimental-protocol.md).

## Sources

- [SEI Insider Threat Test Dataset](https://sei.cmu.edu/library/insider-threat-test-dataset/)
- [Optimising Insider Threat Prediction: Exploring BiLSTM Networks and Sequential Features](https://doi.org/10.1007/s41019-024-00260-z)
- [CICIDS2017 official description](https://www.unb.ca/cic/datasets/ids-2017.html)
- [CSE-CIC-IDS2018 official description](https://www.unb.ca/cic/datasets/ids-2018.html)
- [Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study](https://doi.org/10.1109/SPW53761.2021.00009)

## Implemented CERT daily aggregates — 2026-11-13

[Daily preparation v1](cert-r4.2-day-preparation.md) implements source counts,
fixed off-hours counts, timing/gaps, PC diversity, logon/device actions,
same-day device-pair durations with unmatched-event counts, file extension
groups, HTTP domain/URL diversity and email size/attachments/recipient aggregates.
Its exact feature allowlist is `datasets.cert_days.FEATURES`; the earlier
required-feature tables describe the broader target, not all implemented fields.
Historical novelty/deviations and train-fitted transforms belong to increment 3.
Missing publisher fields (failed logon, HTTP upload/download, file bytes,
received-email action) are not inferred as if observed. Device durations are
limited to same-day pairs. Psychometric values and organizational attributes
are not model inputs in this version.

### CERT integration implementation update — 2026-11-13

All three CERT preparation increments are now implemented; see the
[exact transformation/history/window protocol](cert-r4.2-experiments.md).
The new preparation has 71 inputs fitted/transformed with train-only state.
Personal rolling/novelty history is causal within train; held-out users use
frozen global fallbacks with explicit cold-start flags. Calendar gaps are
retained, and windows span five active days. CERT LSTM uses all-background
training windows; synthetic retains its historical endpoint-normal policy.
Full CERT preparation and a small IF validation-only smoke candidate completed;
real test evaluation and research model comparisons remain future work.

### History control variant — 2026-11-13

The separately named `cert_frozen_global_only_v2` applies the same frozen train
reference to every split. Personal rolling counts remain zero and personal
history availability flags one everywhere (all five standardize to zero).
Global robust volume deviation and novelty use train-only batch-fitted state;
calendar observation gaps remain causal. No target, window length or base
feature changes. Original v1 artifacts remain the control. The
[bounded comparison and nested-calibration group-CV](cert-r4.2-robustness.md)
refit all transformations separately in each development fit partition.
