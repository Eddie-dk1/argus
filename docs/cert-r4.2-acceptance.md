# CERT r4.2 — acquisition and technical audit

> Исторический протокол и результаты соответствующего этапа. Текущий статус —
> [оценка проекта](cert-r4.2-stage-assessment.md); следующий согласованный этап —
> [S4: качество моделей](s4-model-quality-plan.md), ещё не начат. Формулировки
> «следующий»/«пока» в датированных разделах относятся к моменту того этапа.

Audited: 2026-11-12. This records source inspection, **not model results**.

## Scope and result

The official r4.2 archive and answers were downloaded to external storage.
Archive sizes, publisher MD5 and SHA-256 matched. All five activity files were
scanned in chunks; exact duplicate IDs were checked by source using bounded
external sorting on the SSD. No raw files were edited or published.

| Source | Data rows, excluding header | Matched official observables |
| --- | ---: | ---: |
| logon | 854,859 | 198 / 198 |
| device | 405,380 | 2,785 / 2,785 |
| file | 445,581 | 10 / 10 |
| http | 28,434,423 | 3,860 / 3,860 |
| email | 2,629,979 | 470 / 470 |
| Total | 32,770,222 | 7,323 / 7,323 |

No missing required cells, invalid timestamps, repeated headers, duplicate IDs
within a source, unknown activity users relative to the LDAP union/psychometric
population, or invalid/negative email size/attachment counts were found.
Empty email cc/bcc cells are optional fields, not corrupt records. No automatic
repair or dropping of records was performed. Near-duplicate content was not
assessed. Observation dates span 2010-01-02 through 2011-05-17.

The master key contains 70 r4.2 incident users: 30 / 30 / 10 in scenarios 1 / 2 / 3.
All incident observables match complete raw rows, not just IDs; 50 observable
rows have a different actor from the master incident user. All observable dates
fall within the associated master intervals. These are **observables**, not
automatically approved positive event or user-day labels.

## Resolved contract differences

- Publisher file.csv has no activity column: a row represents a removable-media copy.
- Actual email header uses attachments; README describes attachment_count.
- macOS AppleDouble files beginning with `._` are excluded from CSV discovery.
- Event IDs are source-qualified; cross-source collisions are permitted by README.
- Answer detail files have variable-width, source-prefixed rows and no common CSV schema.

## Gates still open before training

1. Reconcile publisher-card CC BY 4.0 with bundled ExactData license.txt before
   any public data redistribution; keep raw and record-level derivatives off GitHub.
2. Label policy fixed on 2026-11-12: [observable account-day v1](cert-r4.2-label-policy.md).
   The checked index contains 986 positive account-days across 72 accounts.
   Cross-account observables remain positive benchmark evidence, attributed to
   the logged account. Full user-day preparation and split allocation remain open.
3. Implement isolated result roots and immutable run/preparation identities.
4. Implement user-day preparation with source mapping, the protocol's user-group
   split, train-only fitted state, and explicit cold-start handling.

## Reproduction and artifacts

With the SSD mounted and ARGUS_DATA_ROOT set to its existing argus-data directory:

```bash
python3 src/audit_cert.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Expected layout under `$ARGUS_DATA_ROOT/cert_r4_2`:

- raw/archives: publisher archives;
- raw/r4.2/r4.2: activity logs, LDAP, psychometric data, README and license;
- raw/answers/answers: official master key and detail files;
- manifests/acquisition_manifest.json and raw_file_manifest.json: sizes/hashes;
- manifests/schema_inspection.json: sampled schema inspection;
- manifests/cert_acceptance_audit.json: complete activity audit and answer matching.

Temporary sorting files use interim/ and are deleted after each successful check.
Do not train the synthetic pipeline directly on these raw CSVs.

## Label-policy follow-up, 2026-11-12

All 50 cross-account rows belong to scenario 3 (20 logon, 30 email).
The publisher describes credential theft and use of a supervisor's account;
these rows must not be discarded as harmless context solely because actors differ.
Incident-linked accounts form 988 singleton groups and two connected groups of
3 and 9 accounts. The next split implementation must keep each group intact.
The policy is frozen; features, complete day labels and model results are not yet built.
