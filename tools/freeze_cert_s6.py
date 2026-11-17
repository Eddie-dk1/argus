"""Create S6.2 feature cache and seal a single empty S6.3 ledger. No model fit."""
import argparse
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from cert_s6_worker import COLUMNS, RAW, apply_personal, fit_args, fit_personal, matrix
from experiment_tracking import file_sha256, payload_sha256
from verify_cert_s6_fit import PROJECT, PROTOCOL, checked_files, collect, read, require


def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def role_cache(folder, role, daily, prep, saved_path, binding, state):
    columns = read(prep / 'contract.json')['tabular_columns']
    table = pd.read_csv(prep / (role + '.csv'), usecols=['event_id', 'user_id', 'timestamp', *columns])
    rows = daily.loc[table.event_id].reset_index(drop=True)
    require(rows.account_id.tolist() == table.user_id.tolist() and rows.date.tolist() == table.timestamp.tolist(),
            'Daily provenance mismatch')
    if role == 'train':
        state, extra, starts = fit_personal(rows)
        save(folder / 'state.json', state)
    else:
        extra, starts = apply_personal(rows, state)
    base = table[columns].to_numpy()
    missing = columns.index('file_usb_profile_missing')
    # Scalers may differ by rounding. Equality of missing-history indicators is exact.
    require(np.array_equal(base[:, missing] > 0, extra[:, -3] > 0), 'Profile support differs')
    ids = rows.item_id.to_numpy(dtype='S')
    np.savez_compressed(folder / (role + '_history.npz'), item_ids=ids, start7=starts[7], start30=starts[30])
    with np.load(saved_path, allow_pickle=False) as z:
        if role == 'train':
            at = np.arange(len(rows))
            saved = {key: z[key] for key in z.files}
            require(np.array_equal(saved['y'], rows.label), 'Fit labels differ')
            for v in ('Q0', 'Q1', 'Q2', 'Q3'):
                x, _, _ = fit_args(v, saved, rows.group, extra, binding)
                require(np.array_equal(base, x[:, :75]), 'Common fit X differs')
        else:
            at = np.flatnonzero(rows.groupby('account_id').cumcount().to_numpy() >= 4)
            require(np.array_equal(z['member_ids'].astype('S'), ids[at[:, None] + np.arange(-4, 1)]),
                    'Endpoint members differ')
            require(np.array_equal(matrix('Q3', base[at], extra[at])[:, :75], z['X']), 'Endpoint X differs')
        require(np.array_equal(z['item_ids'].astype('S'), ids[at]), 'Ordered IDs differ')
    np.save(folder / (role + '_personal.npy'), extra[at], allow_pickle=False)
    return state, dict(days=len(rows), matrix_rows=len(at), input_sha256=file_sha256(saved_path))


def freeze(processed, study, out):
    protocol = read(PROTOCOL)
    ref = protocol['fit_population']['reference']
    require(file_sha256(PROJECT / ref['path']) == ref['sha256'], 'Fit reference changed')
    bindings = read(PROJECT / ref['path'])['controls']
    require(collect(study, processed, protocol) == bindings, 'S6.1 controls differ')
    require(tuple(protocol['feature_contract']['raw_metrics']) == RAW, 'Raw metric contract differs')
    source = protocol['sources']['daily']
    daily_root = processed / source['id']
    checked_files(daily_root, source['manifest_sha256'], ['account_days.csv'])
    fields = ['item_id', 'account_id', 'date', 'group', 'label', *RAW]
    # Restrict source rows before converting any feature values or labels.
    blocks = (part.loc[part.split.eq('train'), fields] for part in
              pd.read_csv(daily_root / 'account_days.csv', usecols=['split', *fields], chunksize=32000))
    days = pd.concat(blocks).set_index('item_id', drop=False)
    require(days.index.is_unique and len(days) == protocol['population']['active_days'], 'Primary train population differs')
    old = read(study / 'cycles/A/summary.json')['results']
    old_final = read(PROJECT / protocol['sources']['control_bindings']['path'])['runs']
    out.mkdir(parents=True, exist_ok=False)
    records, jobs = {}, []
    for key, binding in bindings.items():
        folder = out / key
        folder.mkdir()
        inner = '_inner' in key
        if inner:
            j, = [j for j in old if j['job']['job_id'] == key]
            prep, candidate, assessment = [Path(j[k]) for k in ('preparation', 'candidate', 'assessment')]
            assessment_hash = j['assessment_manifest_sha256']
        else:
            j, = [j for j in old_final if j['job_id'] == key and j['series'] == 'operating_points']
            prep, candidate = processed / j['preparation'], processed / j['trained']['path']
            assessment = processed / j['assessment']['path']
            assessment_hash = j['assessment']['manifest_sha256']
        checked_files(prep, binding['preparation_sha256'], [r + '.csv' for r in ('train','validation','assessment')])
        checked_files(assessment, assessment_hash, ['assessment_inputs.npz'])
        state = None
        roles = {}
        for role, directory in [('train', candidate), ('validation', candidate), ('assessment', assessment)]:
            state, roles[role] = role_cache(folder, role, days, prep, directory / (role + '_inputs.npz'),
                                            binding, state)
        records[key] = dict(roles=roles, files={p.name: file_sha256(p) for p in sorted(folder.iterdir())},
                            fallback_sha256=payload_sha256([state['median'], state['spread']]),
                            scaler_sha256=payload_sha256([state['mean'], state['std']]))
        variants = ('Q0','Q1','Q2','Q3') if inner else ('inner_winner',)
        for v in variants:
            jobs.append(dict(id=key + ':' + v, control=key, variant=v, seed=42,
                             action='exact_reuse' if v == 'Q0' else 'selected_reuse_or_fit' if not inner else 'fit'))
        print(key, roles['train']['days'], flush=True)
    sources = ['src/cert_s6_worker.py','src/cert_operating_point.py','src/utils.py','src/experiment_tracking.py',
               'src/datasets/cert_file_usb.py','tools/freeze_cert_s6.py','tools/verify_cert_s6_fit.py',
               str(PROTOCOL.relative_to(PROJECT)), ref['path'], protocol['fit_population']['h0_config']['path']]
    frozen = dict(schema='argus.cert_s6_runtime.v2', protocol=protocol['protocol_id'], columns=COLUMNS,
                  source_hashes={s:file_sha256(PROJECT / s) for s in sources},
                  python=platform.python_version(), packages={p:version(p) for p in
                      ('numpy','pandas','scipy','scikit-learn','threadpoolctl','joblib')},
                  budget=protocol['budget'], acceptance=protocol['assessment_acceptance'],
                  roles=protocol['roles'], models=protocol['models'], variants=protocol['variants'],
                  controls=records, jobs=jobs, personal_states=len(records))
    require(len(jobs) == 39 and len(records) == 12, 'Incomplete freeze')
    save(out / 'runtime.json', frozen)
    save(out / 'ledger.json', dict(runtime_sha256=file_sha256(out / 'runtime.json'),
         status='awaiting_S6.3', research_attempts=0, jobs={j['id']:'pending' for j in jobs}))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ('processed', 'study', 'out'):
        cli.add_argument('--' + arg, required=True, type=Path)
    args = cli.parse_args()
    require(shutil.disk_usage(args.out.parent).free >= 30 * 1024**3, 'Insufficient work disk')
    freeze(args.processed, args.study, args.out)
