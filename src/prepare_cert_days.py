"""Stream an accepted CERT snapshot into end-of-day features and exact membership."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import date
import gzip
import hashlib
import heapq
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

from datasets.cert_days import Day, FEATURES, FEATURE_SET, HEADERS, SOURCES
from datasets.cert_labels import POLICY_ID, build_label_index
from datasets.cert_split import allocate_groups


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


class HashedReader(io.RawIOBase):
    def __init__(self, path):
        super().__init__()
        self.stream = path.open('rb', buffering=0)
        self.hash = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.stream.readinto(buffer)
        if count:
            self.hash.update(memoryview(buffer)[:count])
        return count

    def close(self):
        self.stream.close()
        super().close()


def gzip_text(stack, path):
    # Empty filename and fixed mtime make membership bytes reproducible.
    binary = stack.enter_context(path.open('wb'))
    compressed = stack.enter_context(gzip.GzipFile(filename='', mode='wb',
                                                  fileobj=binary, compresslevel=1, mtime=0))
    return stack.enter_context(io.TextIOWrapper(compressed, encoding='utf-8', newline=''))


def source_rows(path, source, number, expected_hash, expected_count, index, stats, ids):
    hashed = HashedReader(path)
    previous = None
    dates = {}
    count = 0
    with io.TextIOWrapper(io.BufferedReader(hashed, 1024 * 1024), encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != HEADERS[source]:
            raise ValueError(f'{source}: publisher header mismatch')
        for count, raw in enumerate(reader, 1):
            if len(raw) != len(HEADERS[source]) or not all(raw[:4]):
                raise ValueError(f'{source}: malformed row {count}')
            event_id, timestamp, account = raw[:3]
            if '\n' in event_id or '\r' in event_id:
                raise ValueError('Newline in event ID')
            if account not in index.account_groups:
                raise ValueError(f'{source}: unknown account')
            if (len(timestamp) != 19 or timestamp[2] != '/' or timestamp[5] != '/'
                    or timestamp[10] != ' ' or timestamp[13] != ':' or timestamp[16] != ':'):
                raise ValueError('Invalid publisher timestamp')
            token = timestamp[:10]
            if token not in dates:
                dates[token] = date(int(token[6:]), int(token[:2]), int(token[3:5])).isoformat()
            day = dates[token]
            h, m, s = int(timestamp[11:13]), int(timestamp[14:16]), int(timestamp[17:19])
            if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
                raise ValueError('Invalid clock time')
            second = h * 3600 + m * 60 + s
            key = (day, second)
            if previous is not None and key < previous:
                raise ValueError(f'{source}: source is not chronological at record {count}')
            previous = key
            row = tuple(raw)
            observable = (source, event_id) in index.rows
            if observable:
                index.event_label(source, row)  # Full row match, including quoted content.
            ids.write(event_id + '\n')
            yield day, second, number, count, row, int(observable)
    actual = hashed.hash.hexdigest()
    if actual != expected_hash:
        raise ValueError(f'{source}: raw SHA-256 mismatch')
    if count != expected_count:
        raise ValueError(f'{source}: audit row count mismatch')
    stats[source] = {'rows': count, 'sha256': actual}


def check_duplicate_ids(path, work):
    ordered = path.with_suffix('.sorted')
    subprocess.run(['sort', '-S', '128M', '--parallel=2', '-T', str(work),
                    '-o', str(ordered), str(path)], env={**os.environ, 'LC_ALL': 'C'}, check=True)
    previous = None
    with ordered.open() as stream:
        for line in stream:
            if line == previous:
                raise ValueError(f'Duplicate source-qualified event ID: {path.stem}')
            previous = line
    ordered.unlink()
    path.unlink()


def prepare(source_dir, answers, split_path, raw_manifest_path, audit_path, output_root):
    source_dir, output_root = Path(source_dir), Path(output_root)
    if not output_root.is_dir():
        raise ValueError('Output root must already exist; no missing-volume fallback')
    if shutil.disk_usage(output_root).free < 10 * 2**30:
        raise ValueError('Preparation requires 10 GiB free for membership and external ID sorting')
    split_bytes = Path(split_path).read_bytes()
    split = json.loads(split_bytes)
    raw_bytes = Path(raw_manifest_path).read_bytes()
    raw_manifest = json.loads(raw_bytes)
    audit_bytes = Path(audit_path).read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get('blocking_data_issues') != {} or not audit.get('schema_valid'):
        raise ValueError('Acceptance audit is not clean')
    expected = {}
    for name in (*SOURCES, 'psychometric'):
        matches = [f for f in raw_manifest['files'] if f['path'] == f'raw/r4.2/r4.2/{name}.csv']
        if len(matches) != 1:
            raise ValueError('Missing/ambiguous raw manifest entry')
        expected[name] = matches[0]
        if (source_dir / f'{name}.csv').stat().st_size != expected[name]['size_bytes']:
            raise ValueError(f'{name}: raw size differs from accepted snapshot')
    if digest(source_dir / 'psychometric.csv') != expected['psychometric']['sha256']:
        raise ValueError('Account population file changed')
    with (source_dir / 'psychometric.csv').open(newline='') as stream:
        users = [r['user_id'] for r in csv.DictReader(stream)]
    if not users or any(not u for u in users) or len(set(users)) != len(users):
        raise ValueError('Invalid account population')
    index = build_label_index(Path(answers), accounts=set(users))
    recomputed = allocate_groups(index, tuple(split['fractions'][s] for s in ('train', 'validation', 'test')), split['seed'])
    if any(split.get(k) != v for k, v in recomputed.items()):
        raise ValueError('Frozen split does not match answers/population/algorithm')
    policy_path = Path(__file__).resolve().parents[1] / 'configs/cert_r4_2_labels_v1.json'
    if digest(policy_path) != split['policy_sha256']:
        raise ValueError('Frozen split policy hash mismatch')
    population_hash = hashlib.sha256(json.dumps(sorted(users), separators=(',', ':')).encode()).hexdigest()
    if population_hash != split['account_population_sha256']:
        raise ValueError('Frozen split population hash mismatch')
    snapshot_id = hashlib.sha256(json.dumps({k: v['sha256'] for k, v in expected.items()},
                                           sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    output = output_root / ('cert_days_' + uuid.uuid4().hex)
    output.mkdir()
    print(json.dumps({'output': str(output), 'status': 'running'}), flush=True)
    save(output / 'status.json', {'status': 'running'})
    try:
        (output / 'account_split.json').write_bytes(split_bytes)
        (output / 'raw_manifest.json').write_bytes(raw_bytes)
        (output / 'acceptance_audit.json').write_bytes(audit_bytes)
        (output / 'label_policy.json').write_bytes(policy_path.read_bytes())
        implementation = [Path(__file__), *[Path(__file__).parent / 'datasets' / name for name in
                                           ('cert_days.py', 'cert_labels.py', 'cert_split.py')]]
        save(output / 'provenance.json', {
            'python': sys.version, 'platform': platform.platform(),
            'implementation_sha256': {p.name: digest(p) for p in implementation},
            'answer_sha256': index.input_hashes, 'split_sha256': hashlib.sha256(split_bytes).hexdigest(),
            'raw_manifest_sha256': hashlib.sha256(raw_bytes).hexdigest(),
            'audit_sha256': hashlib.sha256(audit_bytes).hexdigest(),
        })
        save(output / 'contract.json', {
            'feature_set': FEATURE_SET, 'policy_id': POLICY_ID, 'snapshot_id': snapshot_id,
            'feature_columns': FEATURES, 'fitted_state': None, 'prediction_available': 'end_of_day',
            'day_boundary': 'publisher_naive_calendar_date', 'population': 'active_account_days_only',
            'item_id': 'snapshot_id:account_id:YYYY-MM-DD', 'day_key': 'account_id:YYYY-MM-DD',
            'membership_key': ['snapshot_id (contract)', 'source (filename)', 'event_id'],
            'raw_record_index': '1-based CSV data record, excluding header; not physical line number',
            'event_order': ['calendar_date', 'second', 'source_order', 'raw_record_index'],
            'source_order': list(SOURCES), 'off_hours': 'second < 07:00 or second >= 19:00; weekends separate',
            'device_duration': 'same-account/PC, same-day Connect→Disconnect only; newest Connect replaces unmatched one',
            'email_recipient_count': 'unique lowercased to+cc+bcc per message; internal domain exactly dtaa.com',
            'missing_source': 'zero counts, fractions, diversity, sizes and durations',
            'single_event_gap': 'zero count/mean/std/max; population standard deviation for observed gaps',
            'provenance_only': ['item_id', 'day_key', 'account_id', 'date', 'split', 'group', 'label',
                                'context', 'membership', 'positive_evidence'],
            'not_implemented': ['historical novelty/profiles', 'scaling', 'sequence construction', 'model integration'],
        })
        stats, matched = {}, set()
        summaries = {s: Counter() for s in ('train', 'validation', 'test')}
        active_accounts = {s: set() for s in summaries}
        active_days = 0
        positive_days = set()
        total = 0
        work = output / '_work'
        work.mkdir()
        with ExitStack() as stack:
            daily_stream = stack.enter_context((output / 'account_days.csv').open('w', newline=''))
            columns = ['item_id', 'day_key', 'account_id', 'date', 'split', 'group', 'label', *FEATURES]
            daily = csv.DictWriter(daily_stream, fieldnames=columns)
            daily.writeheader()
            context = gzip_text(stack, output / 'day_context.jsonl.gz')
            evidence = stack.enter_context((output / 'positive_evidence.jsonl').open('w'))
            membership = {}
            generators = []
            for number, source in enumerate(SOURCES):
                writer = csv.writer(gzip_text(stack, output / f'{source}_membership.csv.gz'))
                writer.writerow(['day_key', 'event_id', 'raw_record_index', 'second', 'label'])
                membership[source] = writer
                ids = stack.enter_context((work / f'{source}.ids').open('w'))
                generator = source_rows(source_dir / f'{source}.csv', source, number,
                                        expected[source]['sha256'], audit['sources'][source]['rows'], index, stats, ids)
                stack.callback(generator.close)
                generators.append(generator)
            current_day = None
            days = {}

            def flush():
                nonlocal active_days
                for account, accumulator in sorted(days.items()):
                    assignment = split['accounts'][account]
                    part = assignment['split']
                    day_key = f'{account}:{current_day}'
                    features = accumulator.features(current_day)
                    if not all(math.isfinite(v) and v >= 0 for v in features.values()):
                        raise ValueError('Non-finite or negative aggregate')
                    daily.writerow(dict(item_id=f'{snapshot_id}:{day_key}', day_key=day_key,
                                        account_id=account, date=current_day, split=part,
                                        group=assignment['group'], label=accumulator.label, **features))
                    context.write(json.dumps({'day_key': day_key, **accumulator.context()}, sort_keys=True) + '\n')
                    summaries[part]['active_account_days'] += 1
                    summaries[part]['positive_account_days'] += accumulator.label
                    summaries[part]['events'] += features['total_events']
                    active_accounts[part].add(account)
                    active_days += 1
                    if accumulator.label:
                        positive_days.add((account, current_day))

            for day, second, source_number, ordinal, row, label in heapq.merge(*generators):
                if current_day != day:
                    if current_day is not None:
                        flush()
                    current_day, days = day, {}
                source, account = SOURCES[source_number], row[2]
                if account not in days:
                    days[account] = Day()
                days[account].add(source, row, second, label)
                day_key = f'{account}:{day}'
                membership[source].writerow([day_key, row[0], ordinal, second, label])
                if label:
                    key = (source, row[0])
                    matched.add(key)
                    evidence.write(json.dumps({'day_key': day_key, 'source': source, 'event_id': row[0],
                                               'attributions': [vars(a) for a in index.attributions[key]]}, sort_keys=True) + '\n')
                total += 1
                if total % 1_000_000 == 0:
                    print(json.dumps({'events': total, 'through_date': day}), flush=True)
            if current_day is not None:
                flush()
        if matched != set(index.rows) or positive_days != set(index.positive_days()):
            raise ValueError('Missing official observable events/days')
        if total != audit['total_events']:
            raise ValueError('Total event count differs from acceptance audit')
        for source in SOURCES:
            print(json.dumps({'checking_unique_ids': source}), flush=True)
            check_duplicate_ids(work / f'{source}.ids', work)
        work.rmdir()
        for part, summary in summaries.items():
            summary['active_accounts'] = len(active_accounts[part])
            summary['background_account_days'] = summary['active_account_days'] - summary['positive_account_days']
            summary['positive_day_fraction'] = summary['positive_account_days'] / max(summary['active_account_days'], 1)
        report = {'snapshot_id': snapshot_id, 'sources': stats, 'total_events': total,
                  'active_account_days': active_days, 'positive_account_days': len(positive_days),
                  'matched_observables': len(matched), 'splits': summaries,
                  'duplicate_source_ids': 0, 'unknown_accounts': 0, 'feature_count': len(FEATURES),
                  'checks': ['raw SHA-256', 'full observable rows', 'all observables matched',
                             'source-qualified unique IDs', 'frozen group assignments', 'chronological sources',
                             'finite nonnegative aggregates', 'audit event counts', 'positive day set equality']}
        save(output / 'quality_report.json', report)
        save(output / 'status.json', {'status': 'complete'})
        save(output / 'complete.json', {'files': {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}})
        print(json.dumps({'output': str(output), 'report': report}), flush=True)
        return output
    except BaseException as exc:
        save(output / 'status.json', {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                     'error': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('source-dir', 'answers', 'split', 'raw-manifest', 'audit', 'output-root'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_dir, args.answers, args.split, args.raw_manifest, args.audit, args.output_root)


if __name__ == '__main__':
    main()
