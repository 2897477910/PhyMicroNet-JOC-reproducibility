"""Frozen, exploratory prior-confidence diagnostics; saved data only, no fitting."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'runs/round6/prior_diagnostics'
FEATURES = ['full_entropy', 'dominant_occupancy', 'effective_state_number',
            'full_state_count', 'parent_is_dominant', 'neutral_fraction',
            'selected_retained_mass', 'external_nn_az']
LABELS = ['Entropy (nats)', 'Top occupancy', 'Effective states', 'Full state count',
          'Parent dominant', 'Net-zero fraction', 'Retained mass', 'NN to AZ']
COHORTS = ['az_logd', 'az_solubility', 'asap_external']
NAMES = {'az_logd': 'AZ logD', 'az_solubility': 'AZ solubility', 'asap_external': 'ASAP external'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def boolean(series):
    if series.dtype == bool:
        return series
    value = series.astype(str).str.lower()
    assert value.isin(['true', 'false', '1', '0']).all()
    return value.isin(['true', '1'])


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def check_inputs(plan, path):
    assert sha(path) == path.with_suffix('.sha256').read_text().split()[0]
    for name, digest in plan['input_hashes'].items():
        assert sha(ROOT / name) == digest, name
    return len(plan['input_hashes'])


def make_records(cohort, spec):
    predictions = pd.read_csv(ROOT / spec['predictions'])
    completion = read(ROOT / spec['completion'])
    assert completion['artifact_hashes']['predictions.csv'] == sha(ROOT / spec['predictions'])
    for model in ('prior', 'uniform'):
        columns = [f'{model}__seed{s}' for s in (11, 29, 47)]
        assert np.allclose(predictions[columns].mean(axis=1), predictions[model], rtol=0, atol=1e-12)
    cache = ROOT / spec['cache']
    source = pd.read_csv(cache / 'molecules.csv').set_index('raw_row', verify_integrity=True)
    full = pd.read_csv(ROOT / spec['raw'] / 'ionization_features.csv').set_index('raw_row', verify_integrity=True)
    audit = pd.read_csv(cache / 'state_selection_audit.csv').set_index('raw_row', verify_integrity=True)
    states = {s['raw_row']: s for s in map(json.loads, (cache / 'states.jsonl').read_text().splitlines())}
    assert predictions.raw_row.is_unique
    rows = []
    for p in predictions.itertuples():
        f, a, s = full.loc[p.raw_row], audit.loc[p.raw_row], states[p.raw_row]
        src = source.loc[p.raw_row]
        assert p.compound_id == f.compound_id == src.compound_id
        assert p.parent_smiles == f.parent_smiles == src.parent_smiles == s['parent_smiles']
        assert abs(float(p.y) - float(src.y)) < 1e-12
        assert s['smiles'][0] == s['parent_smiles']
        assert f.input_key == a.input_key == s['input_key']
        assert abs(s['retained_mass_fraction'] - a.retained_mass_fraction) < 1e-12
        available = bool(s['ionization_available'])
        assert available == bool(f.basic_quality_pass)
        parent_dominant = float(int(np.argmax(s['prior'])) == 0) if available else np.nan
        full_parent = float(canonical(f.dominant_smiles) == canonical(p.parent_smiles)) if available else np.nan
        row = dict(cohort=cohort, raw_row=int(p.raw_row), source_id=p.source_id,
                   compound_id=p.compound_id, parent_smiles=p.parent_smiles, input_key=f.input_key,
                   y=p.y, prior=p.prior, uniform=p.uniform,
                   delta_weight=abs(p.prior-p.y)-abs(p.uniform-p.y),
                   available=available, conformer_fallback=bool(s['conformer_fallback']),
                   full_entropy=f.entropy, dominant_occupancy=f.dominant_occupancy,
                   effective_state_number=np.exp(f.entropy), full_state_count=f.n_states,
                   parent_is_dominant=parent_dominant, full_parent_is_dominant=full_parent,
                   neutral_fraction=f.net_zero_fraction,
                   selected_retained_mass=np.round(np.clip(s['retained_mass_fraction'], 0, 1), 12),
                   selected_retained_mass_raw=s['retained_mass_fraction'],
                   selected_state_count=len(s['smiles']), full_mean_charge=f.mean_charge,
                   external_nn_az=np.nan, component_t60='',
                   source_split=getattr(p, 'source_split', ''), uncertain_stereo=getattr(p, 'uncertain_stereo', False))
        if not available:
            for name in FEATURES[:-1]:
                row[name] = np.nan
        rows.append(row)
    records = pd.DataFrame(rows)
    if cohort == 'asap_external':
        neighbor = pd.read_csv(ROOT/'runs/round5_external_domain/external_neighbors.csv').set_index('raw_row', verify_integrity=True)
        for i, r in records.iterrows():
            n = neighbor.loc[r.raw_row]
            assert n.compound_id == r.compound_id and n.parent_smiles == r.parent_smiles
            records.loc[i, 'external_nn_az'] = n.max_tanimoto_full_AZ
            records.loc[i, 'component_t60'] = n.candidate_component_t60
    assert np.isfinite(records[['y', 'prior', 'uniform', 'delta_weight']]).all().all()
    return records


def compound_table(records):
    rows = []
    numeric = FEATURES + ['full_parent_is_dominant', 'selected_state_count', 'full_mean_charge', 'delta_weight']
    for (cohort, cid), frame in records.groupby(['cohort', 'compound_id'], sort=True):
        row = dict(cohort=cohort, compound_id=cid, n_records=len(frame),
                   raw_rows=';'.join(map(str, frame.raw_row)), source_ids=';'.join(frame.source_id.astype(str)),
                   n_unique_parents=frame.parent_smiles.nunique(), available_fraction=float(frame.available.mean()),
                   conformer_fallback_fraction=float(frame.conformer_fallback.mean()))
        for name in numeric:
            row[name] = float(frame[name].mean())
            row[name + '_record_range'] = float(frame[name].max()-frame[name].min())
        row['parent_dominant_category'] = ('missing' if np.isnan(row['parent_is_dominant']) else
            'no' if row['parent_is_dominant'] == 0 else 'yes' if row['parent_is_dominant'] == 1 else 'mixed')
        row['component_t60'] = frame.component_t60.iloc[0]
        assert frame.component_t60.nunique(dropna=False) == 1
        rows.append(row)
    result = pd.DataFrame(rows)
    assert result.groupby('cohort').n_records.sum().to_dict() == records.groupby('cohort').size().to_dict()
    return result


def rank_metadata(values):
    _, inverse = np.unique(values, return_inverse=True)
    order = np.argsort(values, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(values[order]))+1]
    return order, starts, inverse


def weighted_midrank(weights, meta):
    order, starts, inverse = meta
    totals = np.add.reduceat(weights[:, order], starts, axis=1)
    # Omit the common +0.5; weighted correlations are invariant to this offset.
    ranks = np.cumsum(totals, axis=1) - totals/2
    return ranks[:, inverse]


def weighted_spearman(weights, x, y):
    """Exact Spearman of expanded resamples, including ties, without expansion."""
    results = []
    xm, ym = rank_metadata(x), rank_metadata(y)
    for start in range(0, len(weights), 64):
        w = weights[start:start+64].astype(float)
        n = w.sum(axis=1)
        xr = weighted_midrank(w, xm) - n[:, None]/2
        yr = weighted_midrank(w, ym) - n[:, None]/2
        numerator = (w*xr*yr).sum(axis=1)
        denominator = np.sqrt((w*xr*xr).sum(axis=1)*(w*yr*yr).sum(axis=1))
        values = np.full(len(w), np.nan)
        np.divide(numerator, denominator, out=values, where=denominator > 0)
        results.append(values)
    return np.concatenate(results)


def self_test():
    rng = np.random.default_rng(33)
    x = np.array([0, 0, 1, 2, 2, 3.0]); y = np.array([1, -1, 3, 3, 2, 0.0])
    weights = rng.integers(0, 4, size=(100, len(x)))
    actual = weighted_spearman(weights, x, y)
    expected = np.array([spearmanr(np.repeat(x,w),np.repeat(y,w)).statistic
        if np.unique(x[w>0]).size > 1 and np.unique(y[w>0]).size > 1 else np.nan for w in weights])
    assert np.allclose(actual, expected, atol=1e-12, equal_nan=True)
    maximum_difference = float(np.nanmax(abs(actual-expected)))
    assert np.isnan(weighted_spearman(np.ones((2, 6)), np.ones(6), y)).all()
    # Cluster copies must repeat all member compounds and retain variable totals.
    membership = np.array([0,0,0,1,2,2]); counts = np.array([[2,0,1],[0,2,1]])
    w = counts[:, membership]
    expected = [spearmanr(np.repeat(x,a), np.repeat(y,a)).statistic for a in w]
    assert np.allclose(weighted_spearman(w,x,y),expected,atol=1e-12)
    return dict(weighted_spearman_vs_scipy_max_abs=maximum_difference,
                synthetic_tied_and_cluster_draws_passed=True)


def bootstrap_weights(frame, unit, draws, seed):
    n = len(frame); rng = np.random.default_rng(seed)
    if unit == 'compound':
        return rng.multinomial(n, np.full(n, 1/n), size=draws).astype(np.uint16), n
    groups, ids = np.unique(frame.component_t60, return_inverse=True)
    assert all(groups != '')
    counts = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=draws)
    return counts[:, ids].astype(np.uint16), len(groups)


def interval(values):
    values = np.asarray(values); finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan, np.nan, 0
    return *np.quantile(finite, [.025,.975]), len(finite)


def definitions(compounds):
    defs = {}
    for cohort in COHORTS:
        defs[cohort] = {}
        for feature in FEATURES:
            if feature == 'parent_is_dominant':
                defs[cohort][feature] = dict(kind='binary_with_mixed', categories=['no','yes','mixed'])
                continue
            ref = 'az_logd' if cohort == 'asap_external' and feature != 'external_nn_az' else cohort
            x = compounds.loc[compounds.cohort.eq(ref), feature].dropna().to_numpy()
            if len(x) == 0:
                defs[cohort][feature] = dict(kind='unavailable', reference=ref)
                continue
            requested = np.quantile(x,[.25,.5,.75],method='linear')
            cuts = [] if np.unique(x).size == 1 else np.unique(requested).tolist()
            defs[cohort][feature] = dict(kind='quartile',reference=ref,reference_compounds=len(x),
                requested_quantiles=requested.tolist(),cuts=cuts,n_bins=len(cuts)+1,
                tied_cutpoints_collapsed=3-len(cuts),reference_unique_values=int(np.unique(x).size),
                tails='negative and positive infinity',closure='right')
    return defs


def bin_masks(frame, feature, definition):
    x = frame[feature].to_numpy(); valid = np.isfinite(x)
    if definition['kind'] == 'unavailable':
        return []
    if definition['kind'] == 'binary_with_mixed':
        return [(i, name, valid & frame.parent_dominant_category.eq(name).to_numpy(), np.nan, np.nan)
                for i,name in enumerate(definition['categories'])]
    cuts = definition['cuts']; index = np.searchsorted(cuts, x, side='left')
    edges = [-np.inf, *cuts, np.inf]
    return [(i,f'({edges[i]:.17g}, {edges[i+1]:.17g}]',valid & (index==i),edges[i],edges[i+1])
            for i in range(len(edges)-1)]


def associations(compounds, plan, defs, out):
    correlations, bins, bootstrap_saved = [], [], {}
    draws, seed = plan['bootstrap']['draws'], plan['bootstrap']['seed']
    for ci,cohort in enumerate(COHORTS):
        frame = compounds.loc[compounds.cohort.eq(cohort)].reset_index(drop=True)
        units = ['compound','component_t60'] if cohort == 'asap_external' else ['compound']
        for ui,unit in enumerate(units):
            weights, n_units = bootstrap_weights(frame,unit,draws,seed+100*ci+ui)
            y = frame.delta_weight.to_numpy()
            for feature in FEATURES:
                x = frame[feature].to_numpy(); valid = np.isfinite(x) & np.isfinite(y)
                row = dict(cohort=cohort,feature=feature,resampling=unit,n_compounds=len(frame),
                    n_finite=int(valid.sum()),n_missing=int((~valid).sum()),n_resampling_units=n_units,
                    unique_feature_values=int(np.unique(x[valid]).size),bootstrap_draws=draws)
                if valid.sum() < 3 or np.unique(x[valid]).size < 2:
                    row.update(rho=np.nan,ci_low=np.nan,ci_high=np.nan,valid_bootstrap_draws=0,
                               status='unavailable_or_constant')
                else:
                    rho = spearmanr(x[valid],y[valid]).statistic
                    boot = weighted_spearman(weights[:,valid],x[valid],y[valid])
                    low,high,nvalid = interval(boot)
                    row.update(rho=float(rho),ci_low=low,ci_high=high,valid_bootstrap_draws=nvalid,
                               status='ok' if nvalid>=.95*draws else 'few_nonconstant_resamples')
                    bootstrap_saved[f'{cohort}__{unit}__{feature}'] = boot
                correlations.append(row)
                for index,label,mask,left,right in bin_masks(frame,feature,defs[cohort][feature]):
                    n = int(mask.sum()); w = weights[:,mask].astype(float)
                    denom = w.sum(axis=1); estimates = np.full(draws,np.nan)
                    np.divide(w@y[mask],denom,out=estimates,where=denom>0)
                    low,high,nvalid=interval(estimates)
                    bins.append(dict(cohort=cohort,feature=feature,resampling=unit,bin_index=index,
                        bin_label=label,left_edge=left,right_edge=right,n_compounds=n,
                        n_records=int(frame.loc[mask,'n_records'].sum()),
                        n_components=int(frame.loc[mask,'component_t60'].nunique()) if cohort=='asap_external' else np.nan,
                        feature_mean=float(x[mask].mean()) if n else np.nan,
                        feature_median=float(np.median(x[mask])) if n else np.nan,
                        delta_mean=float(y[mask].mean()) if n else np.nan,
                        delta_median=float(np.median(y[mask])) if n else np.nan,
                        ci_low=low,ci_high=high,valid_bootstrap_draws=nvalid,
                        status='ok' if n and nvalid>=.95*draws else 'empty_or_sparse_bin'))
            print(json.dumps(dict(completed_cohort=cohort,unit=unit,compounds=len(frame),units=n_units)),flush=True)
    np.savez_compressed(out/'bootstrap_correlations.npz',**bootstrap_saved)
    return pd.DataFrame(correlations),pd.DataFrame(bins)


def flank_diagnostics(records, plan, out):
    output=[]; validations=[]
    for cohort,spec in plan['cohorts'].items():
        frame=records.loc[records.cohort.eq(cohort)]
        allph=pd.read_csv(ROOT/spec['raw']/'ionization_features_all_ph.csv')
        allph=allph.set_index(['raw_row','pH'],verify_integrity=True)
        d=pd.read_csv(ROOT/spec['raw']/'microstate_distributions.csv',usecols=['input_key','microstate_smi','pH','normalized_occupancy'])
        sums=d.groupby(['input_key','pH']).normalized_occupancy.sum()
        assert np.allclose(sums,1,atol=1e-10,rtol=0)
        assert np.isfinite(d.normalized_occupancy).all() and d.normalized_occupancy.ge(0).all()
        wide=d.pivot_table(index=['input_key','microstate_smi'],columns='pH',values='normalized_occupancy',aggfunc='sum',fill_value=0)
        assert {6.4,7.4,8.4}.issubset(wide.columns)
        tv={ph:(wide[ph]-wide[7.4]).abs().groupby(level=0).sum()/2 for ph in [6.4,8.4]}
        for r in frame.itertuples():
            base=allph.loc[(r.raw_row,7.4)]
            for ph in [6.4,8.4]:
                other=allph.loc[(r.raw_row,ph)]
                assert base.input_key==other.input_key==r.input_key
                output.append(dict(cohort=cohort,raw_row=r.raw_row,compound_id=r.compound_id,pH=ph,
                    dominant_changed=float(canonical(base.dominant_smiles)!=canonical(other.dominant_smiles)),
                    delta_mean_charge=other.mean_charge-base.mean_charge,
                    delta_neutral_fraction=other.net_zero_fraction-base.net_zero_fraction,
                    delta_entropy=other.entropy-base.entropy,total_variation=tv[ph].loc[r.input_key]))
        validations.append(dict(cohort=cohort,distribution_groups=len(sums),max_normalization_error=float(abs(sums-1).max())))
    record=pd.DataFrame(output)
    features=['dominant_changed','delta_mean_charge','delta_neutral_fraction','delta_entropy','total_variation']
    compound=record.groupby(['cohort','compound_id','pH'],as_index=False)[features].mean()
    summary=[]
    for (cohort,ph),g in compound.groupby(['cohort','pH']):
        for name in features:
            v=g[name]
            summary.append(dict(cohort=cohort,pH=ph,feature=name,n_compounds=len(g),mean=v.mean(),
                                median=v.median(),q25=v.quantile(.25),q75=v.quantile(.75),minimum=v.min(),maximum=v.max()))
    record.to_csv(out/'ph_flanks_per_record.csv',index=False)
    compound.to_csv(out/'ph_flanks_per_compound.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'ph_flanks_summary.csv',index=False)
    return validations






def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--plan',type=Path,default=DEFAULT_OUT/'plan.json')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args(); out=args.out.resolve(); planpath=args.plan.resolve()
    plan=read(planpath); n_inputs=check_inputs(plan,planpath); tests=self_test()
    if args.validate_only:
        if (out/'READY.json').exists():
            ready=read(out/'READY.json')
            assert ready['complete'] and ready['script_sha256']==sha(__file__)
            for name,digest in ready['artifact_hashes'].items(): assert sha(out/name)==digest,name
        print(json.dumps(dict(inputs_verified=n_inputs,**tests)),flush=True)
        return
    assert not (out/'READY.json').exists(), 'Completed output protected; use --out with a new directory'
    out.mkdir(parents=True,exist_ok=True)
    if out/'plan.json' != planpath:
        (out/'plan.json').write_bytes(planpath.read_bytes())
        (out/'plan.sha256').write_bytes(planpath.with_suffix('.sha256').read_bytes())
    started=time.perf_counter()
    records=pd.concat([make_records(c,s) for c,s in plan['cohorts'].items()],ignore_index=True)
    compounds=compound_table(records)
    records.to_csv(out/'per_record.csv',index=False); compounds.to_csv(out/'per_compound.csv',index=False)
    defs=definitions(compounds); save(out/'bin_definitions.json',defs)
    correlations,bins=associations(compounds,plan,defs,out)
    correlations.to_csv(out/'correlations.csv',index=False); bins.to_csv(out/'binned_summaries.csv',index=False)
    ph=flank_diagnostics(records,plan,out)
    validation=dict(inputs_verified=n_inputs,**tests,ph_distributions=ph,
        record_counts=records.groupby('cohort').size().to_dict(),compound_counts=compounds.groupby('cohort').size().to_dict(),
        finite_features=compounds.groupby('cohort')[FEATURES].count().to_dict(),no_residual_filter=True,
        completed_without_model_fit_or_inference=True)
    save(out/'validation.json',validation)
    import scipy
    save(out/'READY.json',dict(complete=True,status='complete',created_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started,plan_sha256=sha(out/'plan.json'),script_sha256=sha(__file__),
        versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        artifact_hashes={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='READY.json'},validation=validation))
    print(json.dumps(dict(complete=True,output=str(out),seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':
    main()
