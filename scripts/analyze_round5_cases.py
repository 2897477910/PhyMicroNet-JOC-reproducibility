"""Transparent analysis-only illustrative case selection.

Only new Round5 case/cost artifacts are written. No fitting or upstream Uni-pKa
generation occurs. Formal cost measurements must run after all training ends.
"""


from __future__ import annotations


import argparse


from datetime import datetime, timezone


import hashlib


import importlib.metadata


import itertools


import json


import os


from pathlib import Path


import platform


import time


import numpy as np


import pandas as pd


from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[1]


OUT = ROOT / 'runs/round5_cases'


CACHE = ROOT / 'data/processed/round3_unipka'


DEFAULT_PREDICTIONS = ROOT / 'runs/round3_graph/combined/predictions.csv'


GROUPS = ('improved', 'near_unchanged', 'worsened')


RULES = dict(contrast='abs(prior-y) - abs(parent_ionization-y)',
    improved='delta <= -0.10', near_unchanged='abs(delta) <= 0.02', worsened='delta >= 0.10',
    omitted_middle='0.02 < abs(delta) < 0.10',
    identity='Only normalized identities with one source record, avoiding ambiguous replicate labels in a six-panel illustration',
    FN='Full-distribution net-zero-charge probability; not the locally uncharged-atom fraction',
    strata={'low': 'FN < 0.1', 'middle': '0.1 <= FN < 0.9', 'high': 'FN >= 0.9'},
    selection='Within each category/FN stratum choose the record nearest the category median absolute delta, ties by raw_row. Choose two distinct strata per category; require all three FN strata across the six cases. Minimize total median distance divided by each category IQR; ties by ordered raw_row tuple.',
    interpretation='Outcome-selected illustrations, not a held-out statistical test, experimental microstates, or mechanistic proof',
    final_round5='R3 choices are provisional. Reapply the same rules after all declared Round5 seeds finish; preserve both versions and their prediction hashes.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def is_complete(manifest):
    return manifest.get('complete') is True or manifest.get('status') == 'complete'


def select_cases(args):
    folder = OUT / args.stage
    folder.mkdir(parents=True, exist_ok=True)
    predicted = pd.read_csv(args.predictions)
    required = {'raw_row', 'parent_ionization', 'prior'}
    if not required.issubset(predicted) or predicted.raw_row.duplicated().any():
        raise ValueError('Expected one prediction row per raw_row with prior and parent_ionization columns')
    completion = None
    if args.stage == 'r5_final':
        if args.completion_manifest is None:
            raise ValueError('Final selection requires an explicit completed-run manifest; provisional R3 is still available')
        completion = json.loads(args.completion_manifest.read_text(encoding='utf-8'))
        if not is_complete(completion) or completion.get('seeds') != [11, 29, 47]:
            raise ValueError('Final manifest must explicitly declare completion and all three seeds [11,29,47]')
        if completion.get('artifact_hashes', {}).get('predictions.csv') != sha(args.predictions):
            raise ValueError('Prediction hash must match the completed combined-run manifest')
        for mode in ('prior', 'parent_ionization'):
            columns = [f'{mode}__seed{s}' for s in completion['seeds']]
            np.testing.assert_allclose(predicted[mode], predicted[columns].mean(axis=1), rtol=0, atol=1e-12)
    source = pd.read_csv(ROOT / 'data/processed/logd/molecules.csv')
    np.testing.assert_array_equal(np.sort(predicted.raw_row), source.raw_row)
    modes = ('parent_only', 'parent_fn', 'parent_ionization', 'dominant', 'uniform', 'prior')
    prediction_columns = [c for c in predicted if c in modes or any(c.startswith(m+'__seed') for m in modes)]
    f = source.merge(predicted[['raw_row', *prediction_columns]], on='raw_row', validate='one_to_one')
    if not np.isfinite(f[['y', 'prior', 'parent_ionization']]).all().all():
        raise ValueError('Nonfinite predictions/targets must not be silently excluded by case categories')
    with np.load(CACHE / 'ionization_features.npz') as ion:
        np.testing.assert_array_equal(ion['raw_row'], source.raw_row)
        assert list(ion['feature_names'])[0] == 'net_zero_fraction'
        f['FN_predicted_full'] = ion['Z'][:, 0]
        f['full_mean_charge_predicted'] = ion['Z'][:, 5]
        f['ionization_available'] = ion['available']
    states = [json.loads(line) for line in (CACHE / 'states.jsonl').read_text().splitlines()]
    multiplicity = f.compound_id.value_counts()
    f['normalized_identity_record_count'] = f.compound_id.map(multiplicity)
    f['absolute_error_parent_ionization'] = abs(f.parent_ionization-f.y)
    f['absolute_error_prior'] = abs(f.prior-f.y)
    f['delta_absolute_error'] = f.absolute_error_prior-f.absolute_error_parent_ionization
    f['absolute_delta'] = abs(f.delta_absolute_error)
    f['FN_stratum'] = np.select([f.FN_predicted_full < .1, f.FN_predicted_full < .9], ['low', 'middle'], default='high')
    f['category'] = np.select([f.delta_absolute_error <= -.10, abs(f.delta_absolute_error) <= .02,
                               f.delta_absolute_error >= .10], GROUPS, default='outside_prespecified_categories')
    pool = f[f.normalized_identity_record_count.eq(1) & f.ionization_available & f.category.isin(GROUPS)].copy()
    if not np.isfinite(pool[['y', 'prior', 'parent_ionization', 'FN_predicted_full']]).all().all():
        raise ValueError('Nonfinite eligible input; do not silently discard a case')
    options, counts = [], []
    for group in GROUPS:
        subset = pool[pool.category.eq(group)].copy()
        median = float(subset.absolute_delta.median())
        iqr = float(subset.absolute_delta.quantile(.75)-subset.absolute_delta.quantile(.25))
        subset['category_median_absolute_delta'] = median
        subset['distance_to_category_median'] = abs(subset.absolute_delta-median)
        subset['scaled_median_distance'] = subset.distance_to_category_median/max(iqr, 1e-12)
        subset['within_category_absolute_delta_percentile'] = subset.absolute_delta.rank(method='average', pct=True)
        nominees = subset.sort_values(['distance_to_category_median', 'raw_row']).groupby('FN_stratum', sort=True).head(1)
        if len(nominees) < 2:
            raise ValueError(f'Fewer than two populated FN strata in {group}; no rule-changing fallback is allowed')
        options.append(list(itertools.combinations(nominees.to_dict('records'), 2)))
        counts.extend(dict(category=group, FN_stratum=s, eligible_records=int(len(g)), category_median_absolute_delta=median, category_IQR=iqr)
                      for s, g in subset.groupby('FN_stratum'))
    possibilities = []
    for choice in itertools.product(*options):
        rows = [row for pair in choice for row in pair]
        if len({row['FN_stratum'] for row in rows}) == 3:
            possibilities.append((sum(row['scaled_median_distance'] for row in rows), tuple(int(row['raw_row']) for row in rows), rows))
    if not possibilities:
        raise ValueError('Cannot cover three FN strata under the frozen rule')
    chosen = min(possibilities, key=lambda x: (x[0], x[1]))[2]
    result, state_rows = [], []
    prefix = dict(improved='I', near_unchanged='N', worsened='W')
    for group in GROUPS:
        for number, row in enumerate((r for r in chosen if r['category'] == group), 1):
            raw = int(row['raw_row']); state = states[raw]
            assert state['raw_row'] == raw and state['smiles'][0] == row['parent_smiles']
            case_id = prefix[group] + str(number)
            row.update(case_id=case_id, actual_logD=float(row['y']),
                signed_error_parent_ionization=float(row['parent_ionization']-row['y']),
                signed_error_prior=float(row['prior']-row['y']),
                retained_predicted_mass=state['retained_mass_fraction'], n_selected_states=len(state['smiles']))
            result.append(row)
            for index, (smiles, weight) in enumerate(zip(state['smiles'], state['prior'])):
                state_rows.append(dict(case_id=case_id, raw_row=raw, state_index=index, parent=index==0,
                    smiles=smiles, formal_charge=Chem.GetFormalCharge(Chem.MolFromSmiles(smiles)),
                    conditional_model_prior=weight, upstream_probability=weight*state['retained_mass_fraction']))
    selected = pd.DataFrame(result)
    selected.to_csv(folder / 'selected_cases.csv', index=False)
    selected_states = pd.DataFrame(state_rows)
    selected_states.to_csv(folder / 'selected_case_states.csv', index=False)
    pd.DataFrame(counts).to_csv(folder / 'selection_pool_counts.csv', index=False)
    pool[['raw_row','compound_id','category','FN_stratum','delta_absolute_error','FN_predicted_full']].to_csv(folder/'eligible_case_pool.csv',index=False)
    manifest = dict(stage=args.stage, provisional=args.stage=='r3_preselection', rules=RULES,
        source_records=len(f), single_record_normalized_compounds=int(f.normalized_identity_record_count.eq(1).sum()),
        eligible_case_records=len(pool), selected_raw_rows=selected.raw_row.astype(int).tolist(),
        selected_compound_ids=selected.compound_id.tolist(), selected_case_count=len(selected),
        predictions_sha256=sha(args.predictions), source_sha256=sha(ROOT/'data/processed/logd/molecules.csv'),
        states_sha256=sha(CACHE/'states.jsonl'), features_sha256=sha(CACHE/'ionization_features.npz'),
        code_sha256=sha(__file__), completed_round5_manifest=completion,
        outputs='Analysis-only case and state CSVs; no depictions generated')
    write_json(folder/'selection_manifest.json',manifest)
    return manifest

