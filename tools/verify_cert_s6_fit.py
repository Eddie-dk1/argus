"""Verify S6.1 against immutable S4 fit inputs without loading a model."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_tracking import file_sha256, payload_sha256

PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT / 'configs/cert_s6_personal_context_v2.json'


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_files(directory, digest, names):
    manifest = directory / 'complete.json'
    require(file_sha256(manifest) == digest, f'Manifest mismatch: {directory.name}')
    files = read(manifest)['files']
    for name in names:
        require(file_sha256(directory / name) == files[name], f'File mismatch: {directory.name}/{name}')
    return files


def weight_report(labels, groups):
    rows = pd.DataFrame({'class': labels, 'group': groups})
    cells = rows.groupby(['class', 'group'], sort=True).size()
    class_groups = cells.groupby(level='class').size()
    require(set(class_groups.index) == {0, 1}, 'Fit requires both classes')
    cell_size = rows.groupby(['class', 'group'])['group'].transform('size')
    weights = len(rows) / (2 * rows['class'].map(class_groups) * cell_size)
    require(np.isfinite(weights).all() and weights.gt(0).all(), 'Invalid row weights')
    detail = {}
    for label in (0, 1):
        selected = rows['class'].eq(label)
        w = weights[selected]
        mass = w.groupby(rows.loc[selected, 'group']).sum()
        np.testing.assert_allclose(mass, len(rows) / (2 * class_groups[label]), rtol=1e-12, atol=0)
        np.testing.assert_allclose(w.sum(), len(rows) / 2, rtol=1e-12, atol=0)
        detail[str(label)] = dict(days=len(w), groups=len(mass), mass=float(len(rows) / 2),
                                  minimum=round(float(w.min()), 9), maximum=round(float(w.max()), 9),
                                  effective_days=round(float(w.sum() ** 2 / (w ** 2).sum()), 6))
    return detail, payload_sha256(weights.tolist())


def fit_record(candidate, prep, frozen, digests, roles, accounts, control, protocol):
    cp, pp, fp = digests
    pf = checked_files(prep, pp, ['contract.json', 'train.csv', 'development.json', 'feature_state.json'])
    cf = checked_files(candidate, cp, ['config.json', 'model.joblib', 'train_inputs.npz', 'feature_state.json',
                                      'contract.json', 'selection.json', 'validation_inputs.npz'])
    ff = checked_files(frozen, fp, ['config.json', 'model.joblib', 'lineage.json', 'feature_state.json',
                                   'operating_points.json', 'validation_inputs.npz', 'contract.json'])
    lineage = read(frozen / 'lineage.json')
    require(lineage['source_candidate_manifest_sha256'] == cp and lineage['preparation_manifest_sha256'] == pp
            and lineage['no_refit'] and lineage['calibration_only'], 'Frozen candidate lineage mismatch')
    require(all(cf[k] == ff[k] for k in ('config.json', 'model.joblib', 'feature_state.json',
                                          'contract.json', 'validation_inputs.npz')), 'Frozen copy mismatch')
    selection = read(candidate / 'selection.json')
    require(selection['preparation_manifest_sha256'] == pp and not selection['test_accessed']
            and selection['training_policy'] == 'supervised_all_inner_fit_days'
            and selection['model'] == protocol['models']['family'], 'Candidate selection mismatch')
    require(read(candidate / 'config.json') == control, 'H0 configuration mismatch')
    require(all(cf[k] == pf[k] for k in ('contract.json', 'feature_state.json')), 'Preparation mismatch')
    dev = read(prep / 'development.json')
    group_roles = {accounts[a]['group']: role for a, role in roles.items()}
    require(all(accounts[a]['split'] == 'train' and group_roles[accounts[a]['group']] == role
                for a, role in roles.items()), 'Cross-role group or outside primary train')
    require(dev['account_roles'] == roles and dev['group_roles'] == group_roles, 'Preparation roles differ')
    frame = pd.read_csv(prep / 'train.csv')
    contract = read(prep / 'contract.json')
    features = contract['tabular_columns']
    require(contract['history_policy'] == control['features']['history_policy']
            and contract['daily_manifest_sha256'] == protocol['sources']['daily']['manifest_sha256']
            and contract['scope'] == 'development_group_cv' and contract['sequence_length'] == 5
            and len(set(features)) == len(features) == 75, 'Invalid H0 feature contract')
    require(set(frame.user_id) == {a for a, role in roles.items() if role == 'train'}, 'Fit account set differs')
    ids = frame.event_id
    require(ids.is_unique and not frame.duplicated(['user_id', 'timestamp']).any()
            and ids.equals(protocol['sources']['daily']['snapshot_id'] + ':' + frame.user_id + ':' + frame.timestamp)
            and frame.sort_values(['user_id', 'timestamp']).index.equals(frame.index), 'Invalid account-day order')
    with np.load(candidate / 'train_inputs.npz', allow_pickle=False) as saved:
        require(np.array_equal(saved['X'], frame[features].to_numpy())
                and np.array_equal(saved['item_ids'], ids.to_numpy(str))
                and np.array_equal(saved['y'], frame.label.to_numpy()), 'H0 fit arrays differ from preparation')
    groups = frame.user_id.map({a: accounts[a]['group'] for a in roles})
    detail, weights = weight_report(frame.label, groups)
    endpoints = frame.groupby('user_id').cumcount().ge(4)
    return dict(candidate=candidate.name, preparation=prep.name, frozen=frozen.name,
                candidate_sha256=cp, preparation_sha256=pp, frozen_sha256=fp,
                config_sha256=cf['config.json'], model_sha256=cf['model.joblib'],
                state_sha256=pf['feature_state.json'], train_inputs_sha256=cf['train_inputs.npz'],
                fit_ids_sha256=payload_sha256(ids.tolist()), labels_sha256=payload_sha256(frame.label.tolist()),
                groups_sha256=payload_sha256(groups.tolist()), weights_sha256=weights,
                tabular_order_sha256=payload_sha256(features), accounts=frame.user_id.nunique(),
                groups=groups.nunique(), fit_days=len(frame), fit_endpoints=int(endpoints.sum()),
                endpoint_positives=int(frame.label[endpoints].sum()), by_class=detail)


def collect(study, processed, protocol):
    plan = study / 'plan'
    source = protocol['sources']['daily']
    daily = processed / source['id']
    checked_files(daily, source['manifest_sha256'], ['account_split.json'])
    accounts = read(daily / 'account_split.json')['accounts']
    source = protocol['sources']['control_bindings']
    path = PROJECT / source['path']
    require(file_sha256(path) == source['sha256'], 'S4 bindings changed')
    binding = read(path)
    source = protocol['fit_population']['h0_config']
    path = PROJECT / source['path']
    require(file_sha256(path) == source['sha256'], 'H0 config changed')
    control = read(path)
    cycle = study / 'cycles/A'
    checked_files(cycle, protocol['fit_population']['cycle_a_manifest_sha256'], ['summary.json'])
    results = read(cycle / 'summary.json')['results']
    records = {}
    for fold in (1, 2, 3):
        files = {}
        for role, directory in [('outer', 'folds'), ('inner', 'inner')]:
            path = plan / directory / f'fold_{fold}.json'
            require(file_sha256(path) == protocol['roles'][role + '_roles_sha256'][f'fold_{fold}'],
                    f'{role} role hash mismatch, fold {fold}')
            files[role] = read(path)
        outer = files['outer']['account_roles']
        for rot in files['inner']['rotations']:
            key = f"H{fold}_H0_inner{rot['rotation']}"
            j, = [x for x in results if x['job']['job_id'] == key]
            require(j['status'] == 'complete' and j['job']['seed'] == 42, 'Invalid inner control')
            paths = []
            for name in ('candidate', 'preparation', 'frozen'):
                original = Path(j[name])
                # Sealed S4 records carry the original absolute study path.
                base = next(p for p in original.parents if p.name == study.name)
                paths.append(study / original.relative_to(base))
            roles = {a: {'fit': 'train', 'calibration': 'validation', 'selection': 'assessment'}[role]
                     for a, role in rot['account_roles'].items()}
            require(set(roles) == {a for a, role in outer.items() if role != 'assessment'}, 'Inner pool differs')
            r = fit_record(*paths, [j[k + '_manifest_sha256'] for k in ('candidate', 'preparation', 'frozen')],
                           roles, accounts, control, protocol)
            support = rot['support']['fit']
            require((r['fit_days'], r['fit_endpoints'], r['by_class']['1']['days'], r['endpoint_positives']) ==
                    (support['days'], support['endpoints'], support['positive_days'], support['positive_endpoints']),
                    f'Support differs for {key}')
            records[key] = r
        key = f'hgb_rolling_seed42_fold{fold}'
        j, = [x for x in binding['runs'] if x['job_id'] == key and x['series'] == 'operating_points']
        c, p, f = j['trained'], binding['preparations'][j['preparation']], j['operating_bundle']
        records[key] = fit_record(processed / c['path'], processed / j['preparation'], processed / f['path'],
                                  [e['manifest_sha256'] for e in (c, p, f)], outer, accounts, control, protocol)
    require(len(records) == 12, 'Expected nine inner and three final controls')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--processed', type=Path, required=True)
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    parent = protocol['amends']
    require(file_sha256(PROJECT / parent['path']) == parent['sha256'], 'Original protocol changed')
    previous = read(PROJECT / parent['path'])
    weight_id = 'group_class_equal_mass_all_fit_days_v2'
    choices = [(v['id'], v['class_weight'], v['sample_weight']) for v in protocol['variants']]
    require(choices == [('Q0', 'balanced', None), ('Q1', 'balanced', None),
                        ('Q2', None, weight_id), ('Q3', None, weight_id)], 'Variant weight policy differs')
    require(protocol['training_weight']['id'] == weight_id
            and protocol['fit_population']['variants'] == [v[0] for v in choices], 'Fit contract differs')
    for key in ('population', 'roles', 'budget', 'feature_contract', 'assessment_acceptance'):
        require(protocol[key] == previous[key], f'Unexpected change to {key}')
    require(protocol['inner_selection']['lexicographic_order'] == previous['inner_selection']['lexicographic_order'],
            'Inner selection order differs')
    binding = protocol['fit_population']['reference']
    path = PROJECT / binding['path']
    require(file_sha256(path) == binding['sha256'], 'S6.1 reference changed')
    actual = collect(args.study, args.processed, protocol)
    require(actual == read(path)['controls'], 'S6.1 verification differs from reference')
    print('Verified 9 inner and 3 final H0 fit populations, weights and control bindings.')


if __name__ == '__main__':
    main()
