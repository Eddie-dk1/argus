import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from datasets.cert_days import Day, FEATURES, HEADERS, SOURCES
from datasets.cert_labels import build_label_index
from datasets.cert_split import allocate_groups
from prepare_cert_days import digest, prepare, save


class CertDayTests(unittest.TestCase):
    def test_daily_features_and_device_missingness(self):
        d = Day()
        d.add('device', ('c', '', 'U', 'P', 'Connect'), 6 * 3600, 0)
        d.add('device', ('c2', '', 'U', 'P', 'Connect'), 7 * 3600, 0)
        d.add('device', ('d', '', 'U', 'P', 'Disconnect'), 8 * 3600, 0)
        d.add('device', ('d2', '', 'U', 'P', 'Disconnect'), 9 * 3600, 0)
        d.add('file', ('f', '', 'U', 'P', 'Example.PDF', ''), 10 * 3600, 1)
        d.add('http', ('h', '', 'U', 'P', 'https://Example.com/a', ''), 11 * 3600, 0)
        d.add('http', ('h2', '', 'U', 'P', 'http://example.com/b', ''), 12 * 3600, 0)
        d.add('email', ('e', '', 'U', 'P', 'A@dtaa.com;B@other.com', 'a@dtaa.com', '', 'u@dtaa.com', '42', '2', ''), 19 * 3600, 0)
        f = d.features('2010-01-02')
        self.assertEqual(set(f), set(FEATURES))
        self.assertEqual(f['total_events'], 8)
        self.assertEqual(f['device_matched_pairs'], 1)
        self.assertEqual(f['device_paired_seconds'], 3600)
        self.assertEqual(f['device_unmatched_connects'], 1)
        self.assertEqual(f['device_unmatched_disconnects'], 1)
        self.assertEqual(f['off_hours_events'], 2)
        self.assertEqual(f['http_distinct_domains'], 1)
        self.assertEqual(f['http_distinct_urls'], 2)
        self.assertEqual(f['file_extension_pdf_count'], 1)
        self.assertEqual(f['email_internal_recipient_count'], 1)
        self.assertEqual(f['email_external_recipient_count'], 1)
        self.assertEqual(f['email_distinct_recipients'], 2)
        self.assertEqual(f['email_size_bytes_mean'], 42)
        self.assertAlmostEqual(f['gap_mean_seconds'], 13 * 3600 / 7)
        self.assertAlmostEqual(sum(f[f'{s}_fraction'] for s in SOURCES), 1)
        self.assertEqual(d.label, 1)

    def fixture(self, root):
        source, answers, output = [root / n for n in ('raw', 'answers', 'out')]
        for p in (source, answers, output):
            p.mkdir()
        def write(path, rows):
            with path.open('w', newline='') as f:
                csv.writer(f).writerows(rows)
        users = ['U1', 'U2', 'U3', 'U4']
        write(source / 'psychometric.csv', [['user_id'], *[[u] for u in users]])
        logons = [(f'L{i}', '01/02/2010 08:00:00', u, 'PC', 'Logon') for i, u in enumerate(users)]
        master = [['dataset', 'scenario', 'details', 'user', 'start', 'end']]
        for i, row in enumerate(logons[:3]):
            master.append(['4.2', '1', f'{i}.csv', row[2], '01/02/2010 00:00:00', '01/02/2010 23:59:59'])
            write(answers / f'{i}.csv', [['logon', *row]])
        write(answers / 'insiders.csv', master)
        # Same ID across sources is legitimate; quoted multiline content is one CSV record.
        rows = {'logon': logons,
                'device': [('D', '01/02/2010 09:00:00', 'U4', 'PC', 'Connect')],
                'file': [('L0', '01/02/2010 10:00:00', 'U4', 'PC', 'x.doc', 'hello\nworld')],
                'http': [('H', '01/03/2010 10:00:00', 'U4', 'PC', 'https://example.com/x', '')],
                'email': [('E', '01/03/2010 11:00:00', 'U4', 'PC', 'a@dtaa.com', '', '', 'u@dtaa.com', '10', '0', '')]}
        for s in SOURCES:
            write(source / f'{s}.csv', [HEADERS[s], *rows[s]])
        index = build_label_index(answers, accounts=set(users))
        split = allocate_groups(index)
        split['policy_sha256'] = digest(Path(__file__).resolve().parents[1] / 'configs/cert_r4_2_labels_v1.json')
        split['account_population_sha256'] = hashlib.sha256(json.dumps(users, separators=(',', ':')).encode()).hexdigest()
        save(root / 'split.json', split)
        save(root / 'raw.json', {'files': [dict(path=f'raw/r4.2/r4.2/{s}.csv', sha256=digest(source / f'{s}.csv'),
                                               size_bytes=(source / f'{s}.csv').stat().st_size)
                                          for s in (*SOURCES, 'psychometric')]})
        save(root / 'audit.json', {'blocking_data_issues': {}, 'schema_valid': True,
                                  'sources': {s: {'rows': len(rows[s])} for s in SOURCES},
                                  'total_events': sum(map(len, rows.values()))})
        return source, answers, root / 'split.json', root / 'raw.json', root / 'audit.json', output

    def test_full_preparation_membership_labels_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            result = prepare(*args)
            with (result / 'account_days.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5)  # No fabricated inactive days.
            self.assertEqual(sum(int(r['label']) for r in rows), 3)
            self.assertEqual(sum(int(r['total_events']) for r in rows), 8)
            day_by_key = {r['day_key']: r for r in rows}
            members = []
            for s in SOURCES:
                with gzip.open(result / f'{s}_membership.csv.gz', 'rt') as f:
                    members.extend(dict(r, source=s) for r in csv.DictReader(f))
            for r in rows:
                linked = [m for m in members if m['day_key'] == r['day_key']]
                self.assertEqual(len(linked), int(r['total_events']))
                self.assertEqual(max(int(m['label']) for m in linked), int(r['label']))
            self.assertEqual(day_by_key['U4:2010-01-02']['device_unmatched_connects'], '1')
            again = prepare(*args)
            for name in ('account_days.csv', 'day_context.jsonl.gz', 'positive_evidence.jsonl', 'contract.json',
                         *[f'{s}_membership.csv.gz' for s in SOURCES]):
                self.assertEqual(digest(result / name), digest(again / name))
            for name, sha in json.loads((result / 'complete.json').read_text())['files'].items():
                self.assertEqual(digest(result / name), sha)

    def test_failed_preparation_has_no_complete_marker(self):
        for mode in ('hash', 'unknown', 'unordered', 'duplicate', 'missing_observable', 'row_mismatch'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                args = self.fixture(Path(tmp))
                path = args[0] / 'logon.csv'
                with path.open(newline='') as f:
                    rows = list(csv.reader(f))
                if mode == 'unknown':
                    rows[-1][2] = 'UX'
                elif mode == 'unordered':
                    rows[-1][1] = '01/01/2010 08:00:00'
                elif mode == 'duplicate':
                    rows.append(rows[-1])
                elif mode == 'missing_observable':
                    rows[1][0] = 'XX'
                elif mode == 'row_mismatch':
                    rows[1][-1] = 'Logoff'
                else:
                    rows[-1][-1] = 'Logoff'  # Same byte length, wrong SHA.
                with path.open('w', newline='') as f:
                    csv.writer(f).writerows(rows)
                raw = json.loads(args[3].read_text())
                entry = next(e for e in raw['files'] if e['path'].endswith('/logon.csv'))
                entry['size_bytes'] = path.stat().st_size
                if mode != 'hash':
                    entry['sha256'] = digest(path)
                save(args[3], raw)
                audit = json.loads(args[4].read_text())
                audit['sources']['logon']['rows'] = len(rows) - 1
                audit['total_events'] = sum(s['rows'] for s in audit['sources'].values())
                save(args[4], audit)
                with self.assertRaises(ValueError):
                    prepare(*args)
                for directory in args[-1].iterdir():
                    self.assertFalse((directory / 'complete.json').exists())
                    self.assertEqual(json.loads((directory / 'status.json').read_text())['status'], 'failed')
