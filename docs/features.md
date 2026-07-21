# Scenario Feature Map

Last reviewed: 2026-07-21.

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
Argus will verify exact columns against the bundled r4.2 README before accepting
the schema.

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

For each user-day, calculate from earlier train history only:

- deviation from the user's median and robust spread;
- rolling 7-day and 30-day counts/sums when enough history exists;
- change from department/peer-group baseline;
- new PC, domain, recipient, removable-device, or after-hours pattern;
- consecutive off-hours-active days;
- activity-source transition frequencies.

Cold-start and insufficient-history flags accompany every deviation value.

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

## Sources

- [SEI Insider Threat Test Dataset](https://sei.cmu.edu/library/insider-threat-test-dataset/)
- [Optimising Insider Threat Prediction: Exploring BiLSTM Networks and Sequential Features](https://doi.org/10.1007/s41019-024-00260-z)
- [CICIDS2017 official description](https://www.unb.ca/cic/datasets/ids-2017.html)
- [CSE-CIC-IDS2018 official description](https://www.unb.ca/cic/datasets/ids-2018.html)
- [Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study](https://doi.org/10.1109/SPW53761.2021.00009)
