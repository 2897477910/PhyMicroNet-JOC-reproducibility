"""Audit AZ pKa compatibility without outcome-dependent transition matching.

Default: analyze the frozen 100-compound round-2 pilot. No model loading, fitting,
or use of downstream endpoints. Unqualified values are exploratory point values;
explicitly censored values never enter MAE/RMSE. Generic prediction-dir allows
the same predeclared mapping to be applied after full inference is available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phymicronet.chemistry import standardize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(row):
    """Fix charge mapping from source labels before inspecting prediction values."""
    formula = re.search(r"Ionisation Formula:\s*(A(\d+)B(\d+))", row.activity_comment)
    label = re.fullmatch(r"pKa_([AB])(\d+)", row.standard_type)
    result = dict(ionisation_formula="", n_acid_sites=None, n_base_sites=None,
                  expected_higher_charge=None, expected_lower_charge=None,
                  mapping_status="unknown_source_label", mapping_basis="")
    if not formula or not label:
        return result
    formula_text, na, nb = formula.group(1), int(formula.group(2)), int(formula.group(3))
    side, rank = label.group(1), int(label.group(2))
    result.update(ionisation_formula=formula_text, n_acid_sites=na, n_base_sites=nb)
    if na > 0 and nb > 0:
        result["mapping_status"] = "amphoteric_no_automatic_charge_assignment"
        return result
    if (side == "A" and (nb != 0 or rank > na)) or (side == "B" and (na != 0 or rank > nb)):
        result["mapping_status"] = "label_formula_inconsistent"
        return result
    if row.parent_formal_charge != 0:
        result["mapping_status"] = "nonzero_reference_charge_needs_review"
        return result
    if not row.parent_locally_uncharged:
        result["mapping_status"] = "internal_formal_charge_needs_review"
        return result
    # For a neutral acid-only reference A1 is 0 -> -1, A2 is -1 -> -2.
    # For a neutral base-only reference B1 is +1 -> 0, B2 is +2 -> +1.
    high = 1-rank if side == "A" else rank
    result.update(expected_higher_charge=high, expected_lower_charge=high-1,
                  mapping_status="predeclared_nonamphoteric_charge_mapping",
                  mapping_basis="source_AxBy_and_rank_neutral_uncharged_parent")
    return result


def metrics(frame):
    if not len(frame):
        return dict(n_records=0, n_compounds=0, mae=None, rmse=None, bias=None)
    errors = frame.predicted_pka.to_numpy(float)-frame.standard_value.to_numpy(float)
    return dict(n_records=len(frame), n_compounds=frame.compound_id.nunique(),
                mae=float(np.mean(np.abs(errors))), rmse=float(np.sqrt(np.mean(errors**2))),
                bias=float(np.mean(errors)))


def resolve_input(row, manifest):
    """Pick a prediction input using provenance and structures, never pKa values.

    A source ID match takes precedence, but must agree on normalized identity.
    Fallbacks require a unique exact parent, then a unique normalized-identity
    input. Multiple raw records of the same input key do not create ambiguity;
    multiple distinct parent input keys do. No averaging across such inputs.
    """
    base = dict(prediction_input_key="", input_match_status="no_matching_input",
                input_match_candidate_count=0, prediction_raw_rows="",
                prediction_parent_smiles="")
    same_id = manifest[manifest.source_id == row.molecule_chembl_id]
    if len(same_id):
        compatible = same_id[same_id.compound_id == row.compound_id]
        if len(compatible) != len(same_id):
            return {**base, "input_match_status": "source_id_normalized_identity_conflict",
                    "input_match_candidate_count": same_id.prediction_key.nunique()}
        selected, basis = compatible, "source_id_and_normalized_identity"
        if selected.prediction_key.nunique() > 1:
            exact = selected[selected.parent_smiles == row.parent_smiles]
            if len(exact):
                selected, basis = exact, "source_id_and_exact_parent"
    else:
        exact = manifest[manifest.parent_smiles == row.parent_smiles]
        if len(exact):
            selected, basis = exact, "exact_parent_fallback"
        else:
            selected = manifest[manifest.compound_id == row.compound_id]
            basis = "unique_normalized_identity_fallback"
    n = selected.prediction_key.nunique()
    if not n:
        return base
    if n != 1:
        return {**base, "input_match_status": "ambiguous_multiple_parent_inputs_no_selection",
                "input_match_candidate_count": n}
    r = selected.iloc[0]
    return {**base, "prediction_input_key": r.prediction_key,
            "input_match_status": basis, "input_match_candidate_count": 1,
            "prediction_raw_rows": ";".join(map(str, selected.raw_row)) if "raw_row" in selected else "",
            "prediction_parent_smiles": r.parent_smiles}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, default=ROOT / "runs/round2_unipka_pilot")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/round3_pka_audit")
    args = parser.parse_args()
    args.prediction_dir = args.prediction_dir.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.warning")
    source_path = ROOT / "data/source_archive/AZ_pka_all_long.csv"
    pred_path = args.prediction_dir / "predicted_pka_transitions.csv"
    sample_path = args.prediction_dir / "record_manifest.csv"
    if not sample_path.exists():
        sample_path = args.prediction_dir / "sample_manifest.csv"
    source = pd.read_csv(source_path, keep_default_na=False)
    sample = pd.read_csv(sample_path, keep_default_na=False)
    pred = pd.read_csv(pred_path, keep_default_na=False)
    key_column = "input_key" if "input_key" in pred else "compound_id"
    assert key_column in sample
    assert not pred.duplicated([key_column, "higher_charge", "lower_charge"]).any()
    sample["prediction_key"] = sample[key_column]
    # Multiple raw records may refer to one exact parent, but an input key must
    # never mix different parent structures or normalized compound identities.
    assert sample.groupby("prediction_key").parent_smiles.nunique().max() == 1
    assert sample.groupby("prediction_key").compound_id.nunique().max() == 1
    pred_index = pred.set_index([key_column, "higher_charge", "lower_charge"])
    sample_ids = set(sample.compound_id)
    sample_source_ids = set(sample.source_id)
    unique_smiles = {s: standardize(s) for s in source.canonical_smiles.unique()}
    for field in ("compound_id", "parent_smiles", "group_smiles"):
        source[field] = source.canonical_smiles.map(lambda s: unique_smiles[s][field])
    source["parent_formal_charge"] = source.parent_smiles.map(lambda s: Chem.GetFormalCharge(Chem.MolFromSmiles(s)))
    source["parent_locally_uncharged"] = source.parent_smiles.map(lambda s: all(a.GetFormalCharge() == 0 for a in Chem.MolFromSmiles(s).GetAtoms()))
    mapping = pd.DataFrame([classify(r) for r in source.itertuples(index=False)])
    source = pd.concat([source, mapping], axis=1)
    inputs = pd.DataFrame([resolve_input(r, sample) for r in source.itertuples(index=False)])
    source = pd.concat([source, inputs], axis=1)
    source["in_prediction_normalized"] = source.compound_id.isin(sample_ids)
    source["in_prediction_source_id"] = source.molecule_chembl_id.isin(sample_source_ids)
    source["reported_relation"] = source.standard_relation.where(source.standard_relation != "", source.relation)
    source["relation_missing"] = source.reported_relation == ""
    source["predicted_pka"] = np.nan
    source["prediction_status"] = "not_in_prediction_set"
    source["checkpoint_sha256"] = ""
    for idx, row in source.iterrows():
        if not row.prediction_input_key:
            source.loc[idx, "prediction_status"] = row.input_match_status
            continue
        if row.mapping_status != "predeclared_nonamphoteric_charge_mapping":
            source.loc[idx, "prediction_status"] = row.mapping_status
            continue
        key = (row.prediction_input_key, int(row.expected_higher_charge), int(row.expected_lower_charge))
        if key not in pred_index.index:
            source.loc[idx, "prediction_status"] = "mapped_charge_transition_not_generated"
            continue
        p = pred_index.loc[key]
        source.loc[idx, "predicted_pka"] = float(p.predicted_macro_transition_pka)
        source.loc[idx, "checkpoint_sha256"] = p.get("checkpoint_sha256", "see_inference_provenance")
        source.loc[idx, "prediction_status"] = "matched_by_fixed_charge_transition"
    source["numeric_value"] = pd.to_numeric(source.standard_value, errors="coerce")
    source["exploratory_point_eligible"] = (source.prediction_status == "matched_by_fixed_charge_transition") & source.reported_relation.isin(["", "="]) & source.numeric_value.notna()
    source["confirmed_exact_eligible"] = source.exploratory_point_eligible & (source.reported_relation == "=")
    source["absolute_error_if_point_assumption"] = np.where(source.exploratory_point_eligible, abs(source.predicted_pka-source.numeric_value), np.nan)
    source["inequality_violation"] = np.nan
    for idx, row in source[source.predicted_pka.notna()].iterrows():
        if row.reported_relation in ["<", "<="]:
            source.loc[idx, "inequality_violation"] = max(0., row.predicted_pka-row.numeric_value)
        elif row.reported_relation in [">", ">="]:
            source.loc[idx, "inequality_violation"] = max(0., row.numeric_value-row.predicted_pka)
    points = source[source.exploratory_point_eligible].copy()
    points["standard_value"] = points.numeric_value
    # No prediction or label-driven row deletion. Compound-balanced MAE is optional
    # descriptive aggregation, not a way to choose one favorable pKa per molecule.
    compound_abs = points.groupby("compound_id").absolute_error_if_point_assumption.mean()
    overlap = source[source.in_prediction_normalized]
    summary = dict(
        script_version="1.1_exact_parent_input_mapping_fixed_before_full_inference", endpoint_labels_read=False,
        source_records=len(source), source_ids=source.molecule_chembl_id.nunique(),
        source_normalized_compounds=source.compound_id.nunique(), prediction_records=len(sample),
        prediction_unique_inputs=sample.prediction_key.nunique(),
        prediction_normalized_compounds=sample.compound_id.nunique(),
        overlap_normalized_records=len(overlap), overlap_normalized_compounds=overlap.compound_id.nunique(),
        overlap_source_id_compounds=source[source.in_prediction_source_id].molecule_chembl_id.nunique(),
        source_relation_counts=source.reported_relation.replace("", "missing").value_counts().to_dict(),
        source_records_missing_numeric_value=int(source.numeric_value.isna().sum()),
        overlap_relation_counts=overlap.reported_relation.replace("", "missing").value_counts().to_dict(),
        matched_predictions_missing_measured_value=int((source.predicted_pka.notna() & source.numeric_value.isna()).sum()),
        matched_predictions_with_explicit_inequality=int((source.predicted_pka.notna() & source.reported_relation.isin(["<", "<=", ">", ">="])).sum()),
        source_mapping_status_counts=source.mapping_status.value_counts().to_dict(),
        overlap_prediction_status_counts=overlap.prediction_status.value_counts().to_dict(),
        overlap_input_match_status_counts=overlap.input_match_status.value_counts().to_dict(),
        input_mapping="source_ID_plus_identity > exact_parent > unique_normalized_identity; ambiguous_distinct_inputs_unmatched; never_nearest_pKa",
        exploratory_point_assumption=metrics(points),
        compound_balanced_mae_if_point_assumption=float(compound_abs.mean()) if len(compound_abs) else None,
        confirmed_exact_values=metrics(points[points.confirmed_exact_eligible]),
        by_endpoint={label: metrics(g) for label,g in points.groupby("standard_type")},
        eligible_single_site_formula=metrics(points[(points.n_acid_sites+points.n_base_sites) == 1]),
        no_closest_pka_matching=True, predictions_used_to_define_eligible_subset=False,
        training_overlap_status="not_established_independent; ChEMBL_pretraining_structure_overlap_and_Dwar_iBond_label_overlap_possible",
        inference_source=str(args.prediction_dir),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (source_path,pred_path,sample_path)},
        meaning="Exploratory charge-compatible macro-transition check under missing-relation point assumption; not calibrated microstate populations or independent external pKa accuracy.")
    source.to_csv(args.out / "az_pka_mapping_all_records.csv", index=False)
    overlap.to_csv(args.out / "az_pka_prediction_overlap.csv", index=False)
    points.to_csv(args.out / "az_pka_exploratory_point_comparisons.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
