"""Readable and machine-readable conclusions from fixed S3 diagnostics."""
from __future__ import annotations

import numpy as np

from frozen_diagnostics import statistics
from utils import save_json


def conclusions(summary):
    operating = [r for r in summary['records'] if r['series'] == 'operating_points']
    outcomes = []
    for method in sorted({r['method'] for r in operating}):
        for profile in ('fp1_per_1000', 'fp2_per_1000'):
            rows = [r for r in operating if (r['method'], r['profile']) == (method, profile)]
            outcomes.append(dict(method=method, profile=profile, cases=len(rows),
                calibration_both=sum(r['operating_checks']['calibration']['both_met'] for r in rows),
                assessment_budget=sum(r['operating_checks']['assessment']['budget_met'] for r in rows),
                assessment_recall=sum(r['operating_checks']['assessment']['minimum_recall_met'] for r in rows),
                assessment_both=sum(r['operating_checks']['assessment']['both_met'] for r in rows),
                independent_observations=False))
    primary_rows = [r for r in operating if (r['method'], r['seed'], r['profile']) == ('hgb_rolling', 42, 'fp1_per_1000')]
    primary_pooled = next(r for r in summary['pooled_oof'] if
        (r['series'], r['method'], r['seed'], r['profile']) == ('operating_points', 'hgb_rolling', 42, 'fp1_per_1000'))
    fp_with_history = sum(sum(r['assessment']['history']['observed_prior30_days']['buckets'][bucket]['metrics']['fp']
                             for bucket in ('5-9', '10+')) for r in primary_rows)
    fp_with_ten_days = sum(r['assessment']['history']['observed_prior30_days']['buckets']['10+']['metrics']['fp']
                           for r in primary_rows)
    ids = [r['id'] for r in primary_rows]
    facts = dict(operating_conditions=outcomes,
        hgb_primary_reference=dict(pooled_metrics=primary_pooled['metrics'], scenarios=primary_pooled['scenarios'],
            fp_concentration=primary_pooled['false_alarm_concentration'],
            fp_with_at_least_five_prior30_days=fp_with_history,
            fp_with_at_least_five_prior30_days_fraction=fp_with_history / primary_pooled['metrics']['fp'] if primary_pooled['metrics']['fp'] else None,
            fp_with_at_least_ten_prior30_days=fp_with_ten_days,
            source_record_ids=ids),
        score_identity=summary['score_identity_across_model_seeds'])
    explanations = [
        dict(id='E1', statement='Различие распределений и состава аккаунтов calibration/outer может объяснять ограничение переноса FP/recall.',
             status='возможное объяснение; влияние activity mix, prevalence, признаков и ограничений разметки отдельно не установлено'),
        dict(id='E2', statement='Редкая активность или особенности контекста могут быть связаны с концентрацией FP на отдельных аккаунтах.',
             status='только наблюдаемая связь; IDs остаются provenance и не используются как признаки, blacklist или новые пороги'),
        dict(id='E3', statement='Слабый recall сценария 3 может быть связан с несоответствием признаков цели и ограниченной вариативностью групп.',
             status='эти причины не разделены текущими folds; одна connected group не подтверждает общий перенос')]
    hypotheses = [
        dict(id='Q1', priority=1, topic='calibration transfer', hypothesis='Более представительная групповая калибровка может улучшить перенос заранее заданного FP/recall objective.',
            control='нынешний frozen HGB rolling, fp1_per_1000; legacy_f1 сохраняется отдельным контролем',
            evidence=ids, future_objective='FP<=1/1000 and recall>=0.5, with precision/F1/PR-AUC and all folds/seeds reported',
            required_protocol=['declare permitted development data and independent confirmation source',
                'freeze calibration grouping/representativeness alternative, trials, seeds and cost before any fit',
                'preserve infeasible cases and never select on outer outcomes']),
        dict(id='Q2', priority=2, topic='false alarms on previously unseen accounts',
            hypothesis='Признаки активности/контекста или обоснованная регуляризация могут уменьшить концентрацию ложных тревог, в том числе при достаточной истории.',
            control='нынешние HGB inputs/model и все объявленные группы; IDs остаются provenance', evidence=ids,
            future_objective='lower FP concentration/rate while preserving declared recall, precision/F1 and resource limits',
            required_protocol=['define activity/history strata without scores or labels',
                'declare one feature/model factor, identical comparison endpoints, budget and seeds',
                'report cold/fallback and adequately observed accounts separately; no account blacklists']),
        dict(id='Q3', priority=3, topic='weak scenarios and independent group support',
            hypothesis='Новые независимо проверенные incident groups и обоснованные признаки могут улучшить измеримый перенос сценария 3.',
            control='нынешний frozen результат сценария 3; RF/IF/LSTM сохраняются сравнениями', evidence=ids,
            future_objective='scenario recall with support/groups, plus total FP/recall tradeoff',
            required_protocol=['accept new compatible data and audit entity/duplicate/episode independence',
                'freeze feature hypothesis, controls, fit/assessment roles, trial budget and seeds',
                'do not call repeated seeds or a reshuffle of the existing group independent confirmation'])]
    return dict(facts=facts, possible_explanations=explanations, future_quality_hypotheses=hypotheses,
                hypotheses_executed=False, new_protocol_required_before_future_experiments=True)


def _number(value, percent=False):
    if value is None:
        return 'N/A'
    return f'{100 * value:.2f}%' if percent else f'{value:.6g}'


def _table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
        *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def write_report(summary, output):
    derived = conclusions(summary)
    summary['diagnostic_conclusions'] = derived
    save_json(derived, output / 'quality_hypotheses.json')
    c = summary['coverage']
    lines = ['# S3, пункт 2: воспроизведение и диагностика', '',
        'Вклад в качество — проверка сохранности и диагностика. Новых исследовательских обучений, '
        'калибровки, выбора профиля по assessment и CERT test-scoring нет. '
        'Это reused development evidence; независимого подтверждения качества нет.', '',
        '## Полнота и exact replay', '',
        f"Диагностика: {c['source_models']} сохранённых candidates/bundles, {c['decision_profiles']} профилей решений, "
        f"{c['role_reports']} отдельных calibration/assessment reports. Exact replay через общий API: "
        f"{c['replay_models']} моделей и {c['replay_decision_profiles']} профилей (60 candidates и 9 operating bundles seed42).", '',
        'NPZ inputs/scores/decisions сравниваются точно, с исходными dtype, CPU threads=1 и LSTM batch=128. '
        'Полностью проверены порядок features, endpoints, labels для reporting и ordered window membership против '
        'прежнего builder на sealed preparation. Labels не передаются в runtime. '
        'Legacy calibration CSV читается round-trip parser с восстановлением исходного score dtype; '
        'outer scores берутся из exact NPZ. Остальные operating seeds диагностируются по sealed saved scores, '
        'без заявления о новом API replay для них.', '',
        'RF/HGB обучались на всех размеченных inner-fit rows, IF — на background rows, '
        'LSTM — на all-background windows. Это разные режимы supervision; таблица не доказывает '
        'превосходство любой supervised модели над всеми нейросетями.', '',
        f"Популяция: {summary['population']['accounts']} аккаунтов, {summary['population']['independent_groups']} "
        f"incident-connected групп, {summary['population']['endpoints']} endpoints, "
        f"{summary['population']['positive_endpoints']} observable positives. Seeds и повторяющиеся профили эту поддержку не увеличивают.", '',
        '## Наблюдаемые факты', '',
        '### Ограничения рабочих профилей', '',
        'Число случаев = fixed folds × model seeds; случаи не независимы. Calibration feasibility '
        'и outer перенос проверяются отдельно. Основной профиль остаётся fp1_per_1000; sensitivity не назначается вместо него.', '',
        _table(['Модель', 'Профиль', 'Cal оба / cases', 'Outer budget', 'Outer recall', 'Outer оба'], [
            [r['method'], r['profile'], f"{r['calibration_both']}/{r['cases']}", r['assessment_budget'],
             r['assessment_recall'], r['assessment_both']] for r in derived['facts']['operating_conditions']]), '',
        '### HGB rolling, заранее выбранный seed42', '',
        _table(['Fold', 'Профиль', 'Threshold', 'Cal P/R/F1', 'Outer P/R/F1', 'Cal FP/1000', 'Outer FP/1000', 'Outer PR-AUC'], [
            [r['fold'], r['profile'], _number(r['threshold']),
             '/'.join(_number(r['calibration']['metrics'][k], True) for k in ('precision', 'recall', 'f1_score')),
             '/'.join(_number(r['assessment']['metrics'][k], True) for k in ('precision', 'recall', 'f1_score')),
             _number(r['calibration']['metrics']['fp_per_1000_background']),
             _number(r['assessment']['metrics']['fp_per_1000_background']), _number(r['assessment']['metrics']['pr_auc'])]
            for r in summary['records'] if (r['series'], r['method'], r['seed']) == ('operating_points', 'hgb_rolling', 42)]), '',
        '### Все model-variants: pooled decisions seed42', '',
        'Precision/recall/F1 и FP/FN объединяют только решения на непересекающихся endpoints. '
        'PR-AUC ниже — описательное среднее отдельных fold PR-AUC; общего cross-fold ranking нет.', '',
        _table(['Feature policy', 'Метод', 'P', 'R', 'F1', 'Mean fold PR-AUC', 'FP', 'FN', 'Scenario3 TP/support; groups'], [
            [p['policy'], p['method'], _number(p['metrics']['precision'], True), _number(p['metrics']['recall'], True),
             _number(p['metrics']['f1_score']), _number(statistics([r['assessment']['metrics']['pr_auc'] for r in summary['records']
                if (r['series'], r['method'], r['policy']) == ('model_variants', p['method'], p['policy'])])['mean']),
             p['metrics']['fp'], p['metrics']['fn'],
             f"{p['scenarios']['3']['true_positive_days']}/{p['scenarios']['3']['positive_days']}; {p['scenarios']['3']['independent_groups']}"]
            for p in summary['pooled_oof'] if p['series'] == 'model_variants']), '',
        '### Operating profiles: model-seed variation отдельно', '',
        'Среднее и sample SD ниже — по пяти отдельным pooled decision metrics при тех же folds. '
        'Это не CI и не результат объединения seeds в одну выборку.', '']
    aggregates = []
    pooled = [p for p in summary['pooled_oof'] if p['series'] == 'operating_points']
    for method in sorted({r['method'] for r in pooled}):
        for profile in ('legacy_f1', 'fp1_per_1000', 'fp2_per_1000'):
            parts = [r for r in pooled if (r['method'], r['profile']) == (method, profile)]
            f1, fp = statistics([r['metrics']['f1_score'] for r in parts]), statistics([r['metrics']['fp'] for r in parts])
            aggregates.append([method, profile, _number(statistics([r['metrics']['precision'] for r in parts])['mean'], True),
                _number(statistics([r['metrics']['recall'] for r in parts])['mean'], True),
                f"{_number(f1['mean'])} ± {_number(f1['sample_std'])}",
                f"{_number(fp['mean'])} [{_number(fp['min'])}, {_number(fp['max'])}]"])
    lines.extend([_table(['Модель', 'Профиль', 'Mean P', 'Mean R', 'Mean F1 ± seed SD', 'FP mean [min,max]'], aggregates), '',
        '### Fold variation отдельно от model seeds', '',
        _table(['Модель', 'Fold', 'Seeds', 'Exact equal scores', 'Unique score arrays'], [
            [r['method'], r['fold'], ','.join(map(str, r['seeds'])), r['all_seed_scores_exactly_equal'], r['unique_score_arrays']]
            for r in summary['score_identity_across_model_seeds']]), '',
        'Полные fixed-seed/fold и fixed-fold/seed statistics находятся в summary.json → variation. '
        'Совпадение HGB scores не даёт новых независимых наблюдений и не устраняет различия между аккаунтами/folds.', '',
        '### False alarms, история и слабые сценарии', ''])
    ref = derived['facts']['hgb_primary_reference']
    metrics = ref['pooled_metrics']
    lines.extend([
        f"Основной HGB seed42: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, "
        f"precision={_number(metrics['precision'], True)}, recall={_number(metrics['recall'], True)}, "
        f"F1={_number(metrics['f1_score'])}, FP/1000={_number(metrics['fp_per_1000_background'])}.", '',
        f"False alarms на аккаунтах с минимум десятью доступными prior30 днями: "
        f"{ref['fp_with_at_least_ten_prior30_days']}/{metrics['fp']}. "
        'Это описывает наличие истории, а не доказывает механизм ошибки. Все outer аккаунты отсутствовали в inner fit.', '',
        f"Аккаунтов с FP: {ref['fp_concentration']['accounts_with_fp']}; доля FP в top1/top3/top10: "
        + '/'.join(_number(ref['fp_concentration']['shares'][str(n)], True) for n in (1, 3, 10)) + '.', '',
        _table(['Scenario', 'Support', 'TP', 'FN', 'Recall', 'Accounts', 'Independent groups'], [
            [s, r['positive_days'], r['true_positive_days'], r['false_negative_days'], _number(r['recall'], True),
             r['accounts'], r['independent_groups']] for s, r in ref['scenarios'].items()]), '',
        'Support по сценариям может пересекаться; scenario 3 не становится набором независимых случаев от смены seed. '
        'Полные per-role/per-fold score distributions отдельно для background/positives, account FP denominators, '
        'история 0/1–4/5–9/10+ prior30 дней и saved history_missing/cold_start/file_usb_profile_missing '
        'присутствуют в каждой записи summary.json. Policy-forced fallback и фактический дефицит истории разделены.', '',
        '## Возможные объяснения', '',
        *[f"- {r['id']}: {r['statement']} {r['status']}." for r in derived['possible_explanations']], '',
        '## Очередь будущих quality-гипотез', '',
        *[f"- {r['id']} (приоритет {r['priority']}): {r['hypothesis']} Контроль: {r['control']}. "
          'Перед запуском требуется собственный bounded protocol: данные, objective, факторы, бюджет и seeds.'
          for r in derived['future_quality_hypotheses']], '',
        'Гипотезы не запускались. Пороги и активные модели по этой диагностике не заменялись.', '',
        '## Файлы и ограничения', '',
        '- summary.json — все 195 model/profile/fold/seed records и 390 role reports; versioned Spectra schema.',
        '- cohort_metrics.csv — плоская таблица всех calibration/outer metrics, support, accounts/groups и checks.',
        '- quality_hypotheses.json — отдельно факты, возможные объяснения и будущие проверки.',
        '- replays/ — exact повторные scores/decisions, IDs и membership; input origins связаны hashes в summary.',
        '- entries/, contexts/ — отдельные проверяемые записи и reporting-only history strata.',
        '- access_guard.json — закрытый test и защита исходного artifact tree от записи.', '',
        *['- ' + text for text in summary['limitations']], '',
        'Undefined metrics сохраняются как null с причиной; legacy численные значения дополнительно '
        'сохранены для exact сравнения. Отсутствующие обязательные artifacts/folds прерывают задачу, '
        'а не исключаются из отчёта. Raw what-if и общий inspect/replay/diagnose/scenario CLI остаются пунктом 3.', ''])
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    write_figure(summary, output)


def write_figure(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = sorted((r for r in summary['records'] if (r['series'], r['method'], r['seed'], r['profile']) ==
        ('operating_points', 'hgb_rolling', 42, 'fp1_per_1000')), key=lambda r: r['fold'])
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    for axis, key, factor, reference, title in (
        (axes[0], 'fp_per_1000_background', 1, 1, 'False positives / 1,000 background days'),
        (axes[1], 'recall', 100, 50, 'Recall (%)')):
        for role, offset, color in (('calibration', -.18, '#8fa6bc'), ('assessment', .18, '#274c70')):
            axis.bar(x + offset, [factor * r[role]['metrics'][key] for r in rows], width=.34, label=role, color=color)
        axis.axhline(reference, color='#b24a43', linestyle='--', linewidth=1.2, label='declared condition')
        axis.set_xticks(x, [f"Fold {r['fold']}" for r in rows])
        axis.set_title(title, fontsize=10)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(fontsize=8)
    fig.suptitle('Frozen HGB rolling, seed42, primary profile — reused development', fontsize=11)
    fig.savefig(output / 'calibration_outer.png', dpi=180)
    plt.close(fig)
