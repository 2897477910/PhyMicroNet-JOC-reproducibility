"""Structure-only applicability-domain description for the fixed ASAP cohort.

No endpoint column is parsed. The reference is the complete AZ logD source
chemical space, not the training subset of any individual fold model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/round5_external.json'
OUT = ROOT / 'runs/round5_external_domain'
REPORT = ROOT / 'reports/ROUND5_EXTERNAL_DOMAIN.md'
COMMON_COLUMNS = ['raw_row', 'source_id', 'parent_smiles', 'compound_id', 'scaffold']
EXTERNAL_COLUMNS = COMMON_COLUMNS + ['source_split', 'uncertain_stereo']
DESCRIPTORS = ['molecular_weight', 'formal_charge', 'TPSA']
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)
THRESHOLDS = (0.6, 0.7)


def sha(path):
    # Byte-level provenance only: endpoint values are never parsed or accessed.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_structures(path, external=False):
    columns = EXTERNAL_COLUMNS if external else COMMON_COLUMNS
    frame = pd.read_csv(path, usecols=columns, keep_default_na=False).sort_values('raw_row').reset_index(drop=True)
    if frame.raw_row.duplicated().any() or frame.parent_smiles.eq('').any() or frame.compound_id.eq('').any():
        raise ValueError('Missing structure/identity or duplicate raw_row')
    # The AZ pipeline serializes the empty Murcko scaffold as this sentinel.
    frame['scaffold'] = frame.scaffold.replace({'ACYCLIC': ''})
    if external:
        flags = frame.uncertain_stereo.astype(str).str.lower()
        if not flags.isin(['true', 'false', '0', '1']).all():
            raise ValueError('Invalid source stereochemistry uncertainty flag')
        frame['uncertain_stereo'] = flags.isin(['true', '1'])
    return frame


def featurize(frame):
    fingerprints, rows = [], []
    for row in frame.itertuples():
        mol = Chem.MolFromSmiles(row.parent_smiles)
        if mol is None or not mol.GetNumAtoms():
            raise ValueError(f'Invalid parent structure at raw_row {row.raw_row}')
        fingerprints.append(FPGEN.GetFingerprint(mol))
        rows.append((Descriptors.MolWt(mol), Chem.GetFormalCharge(mol), Descriptors.TPSA(mol)))
    result = frame.copy()
    result[DESCRIPTORS] = np.asarray(rows)
    result['formal_charge'] = result.formal_charge.astype(int)
    return result, fingerprints


def nearest_neighbors(external, external_fp, reference, reference_fp):
    result = external.copy()
    similarities, indices = [], []
    for fp in external_fp:
        values = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, reference_fp))
        index = int(np.argmax(values))  # Ascending source raw_row resolves exact ties.
        similarities.append(float(values[index]))
        indices.append(index)
    nearest = reference.iloc[indices]
    result['max_tanimoto_full_AZ'] = similarities
    for column in ('raw_row', 'source_id', 'parent_smiles', 'compound_id', 'scaffold'):
        result['nearest_AZ_' + column] = nearest[column].to_numpy()
    return result


class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def candidate_components(external, fingerprints):
    """Join identities first, then any parent-pair similarity >= fixed thresholds."""
    result = external.copy()
    sets = {threshold: DisjointSet(len(result)) for threshold in THRESHOLDS}
    first = {}
    for i, identity in enumerate(result.compound_id):
        if identity in first:
            for ds in sets.values():
                ds.union(i, first[identity])
        else:
            first[identity] = i
    edge_counts = {threshold: 0 for threshold in THRESHOLDS}
    identities = result.compound_id.to_numpy()
    for i in range(1, len(result)):
        values = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        for j, value in enumerate(values):
            if identities[i] == identities[j]:
                continue
            for threshold, ds in sets.items():
                if value >= threshold:
                    ds.union(i, j)
                    edge_counts[threshold] += 1
    summaries = []
    for threshold, ds in sets.items():
        roots = [ds.find(i) for i in range(len(result))]
        sizes = pd.DataFrame({'root': roots, 'compound': identities}).groupby('root').compound.nunique()
        order = sorted(sizes.index, key=lambda r: (-sizes.loc[r], r))
        names = {root: f'T{int(threshold * 100)}_C{j + 1:03d}' for j, root in enumerate(order)}
        column = f'candidate_component_t{int(threshold * 100)}'
        result[column] = [names[root] for root in roots]
        for name, group in result.groupby(column):
            summaries.append(dict(threshold=threshold, component=name, n_records=len(group),
                                  n_compounds=group.compound_id.nunique(), n_scaffolds=group.scaffold.nunique(),
                                  min_max_tanimoto=float(group.max_tanimoto_full_AZ.min()),
                                  median_max_tanimoto=float(group.max_tanimoto_full_AZ.median()),
                                  max_max_tanimoto=float(group.max_tanimoto_full_AZ.max()),
                                  uncertain_stereo_records=int(group.uncertain_stereo.sum()),
                                  source_split_counts=json.dumps(group.source_split.value_counts().to_dict(), sort_keys=True)))
    return result, pd.DataFrame(summaries), edge_counts


def distribution_summary(frame, dataset):
    rows = []
    tables = {'records': frame[DESCRIPTORS],
              'compound_mean_parent_descriptor': frame.groupby('compound_id')[DESCRIPTORS].mean()}
    for weighting, values in tables.items():
        for name in DESCRIPTORS:
            col = values[name].to_numpy()
            quantiles = np.quantile(col, [.05, .25, .5, .75, .95])
            rows.append(dict(dataset=dataset, weighting=weighting, descriptor=name, n=len(col),
                             mean=float(col.mean()), std=float(col.std(ddof=1)), minimum=float(col.min()),
                             q05=float(quantiles[0]), q25=float(quantiles[1]), median=float(quantiles[2]),
                             q75=float(quantiles[3]), q95=float(quantiles[4]), maximum=float(col.max())))
    return rows


def similarity_summary(external):
    rows = []
    tables = {'records': external.max_tanimoto_full_AZ,
              'compound_max_across_parent_records': external.groupby('compound_id').max_tanimoto_full_AZ.max()}
    for weighting, column in tables.items():
        a = column.to_numpy()
        row = dict(weighting=weighting, n=len(a), mean=float(a.mean()), minimum=float(a.min()),
                   q05=float(np.quantile(a, .05)), q25=float(np.quantile(a, .25)), median=float(np.median(a)),
                   q75=float(np.quantile(a, .75)), q95=float(np.quantile(a, .95)), maximum=float(a.max()))
        for threshold in (.3, .4, .5, .6, .7, .8, .9):
            row[f'count_ge_{threshold:.1f}'] = int((a >= threshold).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def scaffold_tables(reference, external):
    parts = []
    for name, frame in [('AZ', reference), ('ASAP', external)]:
        part = frame.groupby('scaffold', dropna=False).agg(**{
            name + '_records': ('raw_row', 'size'), name + '_compounds': ('compound_id', 'nunique')})
        parts.append(part)
    overlap = pd.concat(parts, axis=1).fillna(0).astype(int).reset_index()
    overlap['shared'] = (overlap.AZ_records > 0) & (overlap.ASAP_records > 0)
    overlap['acyclic_empty_scaffold'] = overlap.scaffold.eq('')
    overlap = overlap.sort_values(['ASAP_records', 'ASAP_compounds', 'scaffold'], ascending=[False, False, True])
    overlap['scaffold_id'] = [f'S{i + 1:04d}' for i in range(len(overlap))]
    ext = overlap.loc[overlap.ASAP_records > 0].copy()
    return overlap, ext


def load_priors(cache, frame, require_ready=False):
    if require_ready:
        ready_path = cache / 'READY.json'
        if not ready_path.exists():
            return None
        ready = json.loads(ready_path.read_text(encoding='utf-8'))
        if not ready.get('cache_complete', False):
            return None
        for name in ('ionization_features.npz', 'states.jsonl'):
            if name not in ready.get('outputs', {}) or sha(cache / name) != ready['outputs'][name]:
                raise ValueError('Prior cache READY hash mismatch')
    if not all((cache / name).exists() for name in ('ionization_features.npz', 'states.jsonl')):
        return None
    with np.load(cache / 'ionization_features.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['raw_row'], frame.raw_row.to_numpy()):
            raise ValueError('Prior cache record order differs from source')
        names = list(saved['feature_names'])
        fn = saved['Z'][:, names.index('net_zero_fraction')]
        available = saved['available'].astype(bool)
    states = [json.loads(line) for line in (cache / 'states.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if [s['raw_row'] for s in states] != frame.raw_row.tolist() or len(fn) != len(frame):
        raise ValueError('Prior state/feature row identities differ')
    return pd.DataFrame(dict(raw_row=frame.raw_row.to_numpy(), compound_id=frame.compound_id.to_numpy(),
                             net_zero_fraction=np.where(available, fn, np.nan), available=available,
                             selected_states=[len(s['smiles']) for s in states]))


def append_prior_comparison(config, reference, external, out):
    az = load_priors(ROOT / 'data/processed/round3_unipka', reference)
    asap = load_priors(ROOT / config['cache'], external, require_ready=True)
    if az is None or asap is None:
        return dict(status='not_ready_not_waited', labels_used=False)
    rows, counts = [], []
    for name, frame in [('AZ', az), ('ASAP', asap)]:
        frame.to_csv(out / f'{name.lower()}_prior_structure_summary.csv', index=False)
        values = frame.loc[frame.available, 'net_zero_fraction'].to_numpy()
        rows.append(dict(dataset=name, n_records=len(frame), available=int(frame.available.sum()),
                         mean_net_zero=float(values.mean()), median_net_zero=float(np.median(values)),
                         q25_net_zero=float(np.quantile(values, .25)), q75_net_zero=float(np.quantile(values, .75)),
                         net_zero_below_0_1=int((values < .1).sum()),
                         net_zero_0_1_to_below_0_9=int(((values >= .1) & (values < .9)).sum()),
                         net_zero_at_least_0_9=int((values >= .9).sum()),
                         mean_selected_states=float(frame.selected_states.mean())))
        counts.extend(dict(dataset=name, selected_states=int(k), n_records=int(v))
                      for k, v in frame.selected_states.value_counts().sort_index().items())
    pd.DataFrame(rows).to_csv(out / 'prior_distribution_comparison.csv', index=False)
    pd.DataFrame(counts).to_csv(out / 'selected_state_counts.csv', index=False)
    return dict(status='complete', labels_used=False, rows=rows)






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--with-priors', action='store_true')
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.warning')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    reference_path = ROOT / 'data/processed/logd/molecules.csv'
    external_path = ROOT / config['source']['directory'] / 'molecules.csv'
    if sha(external_path) != config['source']['molecules_sha256']:
        raise ValueError('External source differs from its frozen hash')
    hashes = {'AZ_molecules_csv': sha(reference_path), 'ASAP_molecules_csv': sha(external_path),
              'config': sha(args.config), 'script': sha(__file__)}
    if (OUT / 'summary.json').exists():
        previous = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
        if any(previous['input_hashes'][key] != hashes[key] for key in ('AZ_molecules_csv', 'ASAP_molecules_csv', 'config')):
            raise ValueError('Refusing to mix domain results across changed input cohorts')
    reference, reference_fp = featurize(read_structures(reference_path))
    external, external_fp = featurize(read_structures(external_path, external=True))
    if len(reference) != 4200 or len(external) != config['source']['records']:
        raise ValueError('Frozen source cohort counts changed')
    if external.compound_id.nunique() != config['source']['normalized_compounds']:
        raise ValueError('External normalized identity count changed')
    external = nearest_neighbors(external, external_fp, reference, reference_fp)
    external, clusters, edges = candidate_components(external, external_fp)
    similarity = similarity_summary(external)
    distributions = pd.DataFrame(distribution_summary(reference, 'AZ') + distribution_summary(external, 'ASAP'))
    overlap, scaffolds = scaffold_tables(reference, external)
    OUT.mkdir(parents=True, exist_ok=True)
    external.to_csv(OUT / 'external_neighbors.csv', index=False)
    reference.to_csv(OUT / 'az_parent_descriptors.csv', index=False)
    similarity.to_csv(OUT / 'similarity_coverage.csv', index=False)
    distributions.to_csv(OUT / 'descriptor_distributions.csv', index=False)
    overlap.to_csv(OUT / 'scaffold_overlap.csv', index=False)
    scaffolds.to_csv(OUT / 'external_scaffold_counts.csv', index=False)
    clusters.to_csv(OUT / 'candidate_components.csv', index=False)
    charge_counts = {name: {str(int(k)): int(v) for k, v in frame.formal_charge.value_counts().sort_index().items()}
                     for name, frame in [('AZ', reference), ('ASAP', external)]}
    pd.DataFrame([dict(dataset=name, formal_charge=int(charge), n_records=count)
                  for name, table in charge_counts.items() for charge, count in table.items()]).to_csv(OUT / 'formal_charge_counts.csv', index=False)
    summary = dict(status='complete_structure_only', created_utc=datetime.now(timezone.utc).isoformat(),
                   input_hashes=hashes, parsed_columns={'AZ': COMMON_COLUMNS, 'ASAP': EXTERNAL_COLUMNS},
                   endpoint_columns_parsed=[], labels_used=False,
                   reference_scope='Complete AZ 4200-parent source space; not any single fold training set',
                   fingerprint={'algorithm': 'Morgan', 'radius': 2, 'bits': 1024, 'includeChirality': True},
                   scaffold_normalization='Source ACYCLIC sentinel is mapped to the empty nonchiral Murcko scaffold bucket',
                   parent_formal_charge_counts=charge_counts,
                   AZ_records=len(reference), AZ_compounds=reference.compound_id.nunique(),
                   ASAP_records=len(external), ASAP_compounds=external.compound_id.nunique(),
                   source_split_counts=external.source_split.value_counts().to_dict(),
                   uncertain_stereo_records=int(external.uncertain_stereo.sum()),
                   normalized_identity_overlap=len(set(reference.compound_id) & set(external.compound_id)),
                   AZ_scaffolds_including_acyclic_bucket=reference.scaffold.nunique(),
                   ASAP_scaffolds_including_acyclic_bucket=external.scaffold.nunique(),
                   shared_scaffold_keys_including_acyclic_bucket=int(overlap.shared.sum()),
                   ASAP_records_with_AZ_scaffold=int(external.scaffold.isin(set(reference.scaffold)).sum()),
                   ASAP_compounds_with_AZ_scaffold=int(external.loc[external.scaffold.isin(set(reference.scaffold)), 'compound_id'].nunique()),
                   AZ_acyclic_records=int(reference.scaffold.eq('').sum()),
                   ASAP_acyclic_records=int(external.scaffold.eq('').sum()),
                   candidate_component_thresholds=list(THRESHOLDS),
                   cross_compound_parent_pair_edges={str(k): int(v) for k, v in edges.items()},
                   candidate_component_interpretation='Transitive similarity-connected candidate groups, not verified series or independent units',
                   prior_comparison=dict(status='not_requested', labels_used=False))
    if args.with_priors:
        summary['prior_comparison'] = append_prior_comparison(config, reference, external, OUT)
    write_json(OUT / 'summary.json', summary)
    print(json.dumps(dict(status=summary['status'], records=len(external), compounds=external.compound_id.nunique(),
                          median_similarity=float(external.max_tanimoto_full_AZ.median()),
                          report=str(REPORT), output=str(OUT), labels_used=False)), flush=True)


if __name__ == '__main__':
    main()
