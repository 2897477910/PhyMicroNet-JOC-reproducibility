"""Exploratory, saved-prediction robustness analysis; never fits or edits R5.

Run `self-test`, then `plan`, then `run`. The plan is created exclusively and
locks this source plus all input bytes before any new empirical results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/round6/robustness"
REPORT = ROOT / "reports/ROUND6_ROBUSTNESS.md"
SEED, DRAWS = 261024, 10000
MODELS = ["parent_only", "parent_fn", "parent_ionization", "dominant", "uniform", "prior"]
CONTRASTS = [("parent_ionization", "parent_only"), ("dominant", "parent_ionization"),
             ("uniform", "dominant"), ("prior", "uniform")]
SETS = {"logd": "combined_logd", "solubility": "combined_solubility", "external": "external_asap"}
EPS = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_dump(path, obj, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")


def inputs():
    paths = {f"{cohort}_{kind}": f"runs/round5/{folder}/{kind}.csv"
             for cohort, folder in SETS.items()
             for kind in ("predictions", "metrics", "paired_comparisons")}
    paths.update({"components": "runs/round5_external_domain/candidate_components.csv",
                  "neighbors": "runs/round5_external_domain/external_neighbors.csv"})
    return paths


def make_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise RuntimeError("Plan/output directory must be empty; never overwrite a frozen analysis.")
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory_post_R5_results_plan_frozen_before_new_R6_evaluation",
        "not_preregistered": True, "no_new_fitting": True, "no_primary_residual_exclusion": True,
        "source": str(Path(__file__).relative_to(ROOT)), "source_sha256": sha(__file__),
        "inputs": {k: {"path": v, "sha256": sha(ROOT / v)} for k, v in inputs().items()},
        "contrasts": [list(x) for x in CONTRASTS], "seed": SEED, "draws": DRAWS,
        "estimand": "Mean across normalized compounds of within-compound mean paired absolute-error difference; candidate minus reference.",
        "predictions": "Saved ensemble columns, not averaging errors over seeds; no new response transformations.",
        "compound_bootstrap": "Draw N compound indices with replacement; preserve all duplicate records through their within-compound error means.",
        "component_bootstrap": "For external uniform-dominant and prior-uniform at Tanimoto 0.60 and 0.70, draw G existing components with replacement, calculate sum of sampled component delta sums / sum of sampled compound counts. Retain compound-weighted estimand, not equal-component mean.",
        "scopes": {"all_cohorts": ["primary_all"], "external_extra": ["source_Train", "source_Test", "no_AND_OR"],
                   "solubility_extra": ["exclude_upper_flag", "overlap_logd_identity", "no_overlap_logd_identity"]},
        "component_subset_policy": "Use existing full-external component labels; subset members without rebuilding components.",
        "interval": "Percentile 2.5/97.5%, numpy linear quantiles; unadjusted and conditional on saved fits and fixed cohorts.",
        "rng": "Fresh numpy default_rng(261024) per scope/resampling unit; common draws across all contrasts within that table. Sorted compound and component IDs; batch size 256.",
        "influence": "Leave one compound for all four contrasts/all scopes; leave one component for both external graph contrasts/all external scopes. Compute exact remaining-compound-weighted estimate. Flag opposite sign or absolute change >25% of abs(full effect); if abs(full)<1e-12 disable relative rule, report absolute change and sign flips.",
        "component_diagnostics": "Number, singletons, largest compound/record counts, largest compound fraction, Kish concentration-equivalent count N^2/sum(n_g^2), explicitly not independent series count.",
        "extremes": "Descriptive union across six graph models of top ceil(1% of records) by absolute residual and values strictly above Q3+1.5 IQR; raw_row breaks top-count ties. Save all six predictions and signed/absolute residuals, source/category/stereo/identity metadata. No removal or fitting.",
        "source_heterogeneity": "All four exploratory source Test-minus-Train effect differences; resample union compounds jointly and retain per-category within-compound means, so any identity crossing categories is kept together. Category denominators remain numbers of sampled compounds represented in that category.",
        "robustness_label": "For primary external two graph contrasts, call direction resolved at all levels only if compound and both component intervals all exclude zero with the same direction; otherwise compound-resolved/series-unresolved or unresolved. Keep influence diagnostics separate.",
        "existing_point_check": "Recompute primary ensemble record MAE/RMSE/R2 for every saved model and four compound-equal contrasts; compare existing CSV values at atol=1e-12.",
    }
    json_dump(OUT / "PLAN.json", plan, exclusive=True)
    (OUT / "PLAN.sha256").write_text(sha(OUT / "PLAN.json") + "\n", encoding="ascii")
    print(json.dumps({"plan": str(OUT / "PLAN.json"), "sha256": sha(OUT / "PLAN.json")}, indent=2))


def bootstrap_ratio(sums, counts, draws=DRAWS, seed=SEED):
    """Ratio of resampled totals. Rows are clusters, columns are contrasts."""
    sums = np.asarray(sums, dtype=np.float64)
    if sums.ndim == 1:
        sums = sums[:, None]
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim == 1:
        counts = counts[:, None]
    assert len(sums) == len(counts) and len(sums) >= 2
    assert np.isfinite(sums).all() and np.isfinite(counts).all() and (counts >= 0).all()
    rng = np.random.default_rng(seed)
    result = np.empty((draws, sums.shape[1]))
    for start in range(0, draws, 256):
        ix = rng.integers(0, len(sums), size=(min(256, draws - start), len(sums)))
        denominator = counts[ix].sum(axis=1)
        if np.any(denominator <= 0):
            raise RuntimeError("A bootstrap category has zero denominator; report rather than silently redraw.")
        result[start:start + len(ix)] = sums[ix].sum(axis=1) / denominator
    return result


def influence(sums, counts):
    sums, counts = np.asarray(sums, float), np.asarray(counts, float)
    full = sums.sum() / counts.sum()
    if np.any(counts.sum() - counts <= 0):
        raise ValueError("Leave-one-group estimate requires at least two groups.")
    leave = (sums.sum() - sums) / (counts.sum() - counts)
    change = leave - full
    relative = np.abs(change) / abs(full) if abs(full) >= EPS else np.full(len(leave), np.nan)
    flips = full * leave < 0
    relative_flag = relative > .25
    return full, leave, change, relative, flips, relative_flag


def as_bool(series):
    s = series.astype(str).str.lower()
    if not s.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Unknown Boolean value in scope flags.")
    return s.isin(["true", "1"])


def scopes(cohort, frame):
    result = {"primary_all": frame}
    if cohort == "external":
        assert set(frame.source_split) == {"Train", "Test"}
        result.update({"source_Train": frame[frame.source_split == "Train"],
                       "source_Test": frame[frame.source_split == "Test"],
                       "no_AND_OR": frame[~as_bool(frame.uncertain_stereo)]})
    if cohort == "solubility":
        overlap = as_bool(frame.overlaps_logd_normalized_compound)
        result.update({"exclude_upper_flag": frame[~as_bool(frame.above_stated_upper_1500_uM)],
                       "overlap_logd_identity": frame[overlap],
                       "no_overlap_logd_identity": frame[~overlap]})
    return result


def compound_errors(frame, models):
    errors = frame[["compound_id"]].copy()
    for model in models:
        errors[model] = np.abs(frame[model].to_numpy(float) - frame.y.to_numpy(float))
    return errors.groupby("compound_id", sort=True)[models].mean()


def mapped_external(frame, neighbors, components):
    assert frame.raw_row.is_unique and neighbors.raw_row.is_unique
    assert set(frame.raw_row) == set(neighbors.raw_row)
    joined = frame.merge(neighbors, on="raw_row", how="left", validate="one_to_one", suffixes=("", "__neighbor"))
    for field in ["source_id", "compound_id", "parent_smiles", "source_split", "uncertain_stereo"]:
        assert joined[field].astype(str).equals(joined[field + "__neighbor"].astype(str)), field
    rows = []
    for threshold, column in [(0.6, "candidate_component_t60"), (0.7, "candidate_component_t70")]:
        assert joined[column].notna().all()
        assert (joined.groupby("compound_id")[column].nunique() == 1).all()
        actual = joined.groupby(column).agg(n_records=("raw_row", "size"), n_compounds=("compound_id", "nunique"))
        expected = components[np.isclose(components.threshold, threshold)].set_index("component")
        assert set(actual.index) == set(expected.index)
        for field in ["n_records", "n_compounds"]:
            assert np.array_equal(actual[field].to_numpy(), expected.loc[actual.index, field].to_numpy())
        rows.append({"threshold": threshold, "records": len(joined), "compounds": joined.compound_id.nunique(),
                     "components": len(actual), "duplicate_identity_split_components": 0, "missing_rows": 0})
    return joined, rows


def self_test():
    checks = []
    x = np.array([[1., -2.], [3., 4.], [8., -1.]])
    b = bootstrap_ratio(x, np.ones(3), draws=1000)
    # Singleton components are exactly the compound bootstrap with identical RNG.
    np.testing.assert_array_equal(b, bootstrap_ratio(x.copy(), np.ones(3), draws=1000))
    checks.append("singleton components exactly reproduce compound resampling")
    # Unequal components preserve total-compound weighting, including each draw.
    sums, counts = np.array([4., 10.]), np.array([2., 1.])
    boot = bootstrap_ratio(sums, counts, draws=1000)[:, 0]
    ix = np.random.default_rng(SEED).integers(0, 2, size=(1000, 2))
    np.testing.assert_allclose(boot, sums[ix].sum(1) / counts[ix].sum(1), atol=0, rtol=0)
    assert sums.sum() / counts.sum() != np.mean(sums / counts)
    checks.append("unequal component ratio, not equal-component effect")
    f, leave, change, relative, flips, flag = influence(sums, counts)
    for k in range(2):
        np.testing.assert_allclose(leave[k], np.delete(sums, k).sum() / np.delete(counts, k).sum())
        np.testing.assert_allclose(change[k], leave[k] - f)
    assert not flips.any() and flag.all()
    checks.append("leave-one-group formula matches brute force")
    f, leave, _, rel, flips, _ = influence([-.2, .1], [1, 1])
    assert flips.tolist() == [True, False]
    assert np.isnan(influence([-1., 1.], [1, 1])[3]).all()
    checks.append("opposite sign and zero-full-effect relative rule")
    toy = pd.DataFrame({"compound_id": ["a", "a", "b"], "y": [0., 0., 0.], "a": [1., 3., 10.], "b": [0., 0., 1.]})
    ce = compound_errors(toy, ["a", "b"])
    assert (ce.a - ce.b).mean() == 5.5
    assert (np.abs(toy.a - toy.y) - np.abs(toy.b - toy.y)).mean() != 5.5
    checks.append("duplicate averaging precedes compound averaging")
    print(json.dumps({"status": "passed", "checks": checks}, indent=2))
    return checks


def run():
    plan_path = OUT / "PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert sha(plan_path) == (OUT / "PLAN.sha256").read_text().strip()
    assert sha(__file__) == plan["source_sha256"], "Analysis changed after plan freeze."
    assert set(p.name for p in OUT.iterdir()) == {"PLAN.json", "PLAN.sha256"}, "Refuse overwriting prior analysis."
    assert not REPORT.exists(), "Refuse overwriting an existing R6 report."
    for item in plan["inputs"].values():
        assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
    verification = {"self_tests": self_test(), "input_hashes_verified": True}
    frames = {cohort: pd.read_csv(ROOT / plan["inputs"][cohort + "_predictions"]["path"])
              for cohort in SETS}
    for cohort, df in frames.items():
        assert df.raw_row.is_unique and df.compound_id.notna().all()
        assert np.isfinite(df[["y"] + MODELS].to_numpy(float)).all()
        assert len(df) == {"logd": 4200, "solubility": 353, "external": 473}[cohort]
        assert df.compound_id.nunique() == {"logd": 4178, "solubility": 350, "external": 413}[cohort]
        for model in MODELS:
            np.testing.assert_allclose(df[model], df[[f"{model}__seed{s}" for s in [11, 29, 47]]].mean(axis=1), atol=1e-12, rtol=0)
    frames["external"], verification["component_mapping"] = mapped_external(
        frames["external"], pd.read_csv(ROOT / plan["inputs"]["neighbors"]["path"]),
        pd.read_csv(ROOT / plan["inputs"]["components"]["path"]))
    external = frames["external"]
    cross = external.groupby("compound_id").source_split.nunique()
    verification["external_compounds_crossing_source_categories"] = int((cross > 1).sum())
    metric_rows, points, effects, influence_rows, sizes, compound_rows, residual_rows = [], [], [], [], [], [], []
    boot_distributions = {}

    def add_resampling(cohort, scope, errors, contrasts, group_labels, unit, raw_frame):
        delta = pd.DataFrame({f"{a}__minus__{b}": errors[a] - errors[b] for a, b in contrasts})
        delta["group"] = np.asarray(group_labels)
        grouped = delta.groupby("group", sort=True)
        group_sums = grouped[delta.columns[:-1]].sum()
        group_counts = grouped.size().astype(float)
        samples = bootstrap_ratio(group_sums.to_numpy(), group_counts.to_numpy())
        key = f"{cohort}__{scope}__{unit}"
        boot_distributions[key] = samples
        quantiles = np.quantile(samples, [.025, .975], axis=0, method="linear")
        n = len(errors)
        effective = float(n * n / np.sum(group_counts.to_numpy() ** 2))
        for col, (a, b) in enumerate(contrasts):
            full, leave, change, relative, flips, rel_flag = influence(group_sums.iloc[:, col], group_counts)
            np.testing.assert_allclose(full, (errors[a] - errors[b]).mean(), atol=1e-14, rtol=0)
            lo, hi = map(float, quantiles[:, col])
            effects.append({"cohort": cohort, "scope": scope, "resampling_unit": unit, "candidate": a, "reference": b,
                            "delta_MAE": full, "ci_low": lo, "ci_high": hi, "n_records": len(raw_frame), "n_compounds": n,
                            "n_units": len(group_counts), "singleton_units": int((group_counts == 1).sum()),
                            "largest_unit_compounds": int(group_counts.max()),
                            "largest_unit_compound_fraction": float(group_counts.max() / n),
                            "concentration_effective_units": effective, "draws": DRAWS, "seed": SEED,
                            "interval_direction": "negative" if hi < 0 else "positive" if lo > 0 else "unresolved",
                            "n_sign_flips": int(flips.sum()), "n_relative_change_gt25pct": int(rel_flag.sum()),
                            "max_abs_leave_one_change": float(np.abs(change).max()),
                            "max_relative_leave_one_change": float(np.nanmax(relative)) if np.isfinite(relative).any() else np.nan,
                            "influence_sensitive": bool((flips | rel_flag).any()),
                            "bootstrap_key": key, "bootstrap_column": col, "status": "exploratory_conditional_unadjusted"})
            for k, group in enumerate(group_sums.index):
                influence_rows.append({"cohort": cohort, "scope": scope, "unit": unit, "group_id": group,
                                       "candidate": a, "reference": b, "n_compounds_removed": int(group_counts.iloc[k]),
                                       "removed_delta_sum": float(group_sums.iloc[k, col]), "full_delta": full,
                                       "leave_one_delta": leave[k], "change": change[k], "relative_abs_change": relative[k],
                                       "sign_flip": bool(flips[k]), "change_gt25pct": bool(rel_flag[k]),
                                       "flagged": bool(flips[k] or rel_flag[k])})

    for cohort, frame in frames.items():
        models = MODELS + (["tree_parent", "tree_ionization"] if cohort == "solubility" else [])
        old_metrics = pd.read_csv(ROOT / plan["inputs"][cohort + "_metrics"]["path"])
        old_pairs = pd.read_csv(ROOT / plan["inputs"][cohort + "_paired_comparisons"]["path"])
        for scope, df in scopes(cohort, frame).items():
            ce = compound_errors(df, models)
            for model in models:
                residual = df[model].to_numpy(float) - df.y.to_numpy(float)
                values = {"MAE": float(np.mean(np.abs(residual))), "RMSE": float(np.sqrt(np.mean(residual ** 2))),
                          "R2": float(1 - np.sum(residual ** 2) / np.sum((df.y - df.y.mean()) ** 2))}
                metric_rows.append({"cohort": cohort, "scope": scope, "model": model, "n_records": len(df), "n_compounds": len(ce),
                                    "record_MAE": values["MAE"], "compound_MAE": float(ce[model].mean()),
                                    "record_RMSE": values["RMSE"], "record_R2": values["R2"]})
                if scope == "primary_all":
                    old = old_metrics[(old_metrics.scope == scope) & (old_metrics.model == model) & (old_metrics.seed.astype(str) == "ensemble")]
                    assert len(old) == 1
                    for field, value in values.items():
                        previous = float(old.iloc[0][field])
                        assert abs(value - previous) <= 1e-12
                        points.append({"cohort": cohort, "kind": field, "model_or_contrast": model, "recomputed": value,
                                       "previous": previous, "absolute_difference": abs(value - previous)})
            for a, b in CONTRASTS:
                delta = ce[a] - ce[b]
                for compound, value in delta.items():
                    compound_rows.append({"cohort": cohort, "scope": scope, "compound_id": compound,
                                          "candidate": a, "reference": b, "delta": value,
                                          "candidate_mean_absolute_error": ce.loc[compound, a],
                                          "reference_mean_absolute_error": ce.loc[compound, b],
                                          "n_records": int((df.compound_id == compound).sum())})
                if scope == "primary_all":
                    old = old_pairs[(old_pairs.scope == scope) & (old_pairs.candidate == a) & (old_pairs.reference == b)]
                    if "seed" in old:
                        old = old[old.seed.astype(str) == "ensemble"]
                    assert len(old) == 1
                    previous = float(old.iloc[0].delta_MAE)
                    assert abs(delta.mean() - previous) <= 1e-12
                    points.append({"cohort": cohort, "kind": "paired_delta", "model_or_contrast": f"{a}-{b}",
                                   "recomputed": float(delta.mean()), "previous": previous,
                                   "absolute_difference": abs(delta.mean() - previous)})
            add_resampling(cohort, scope, ce, CONTRASTS, ce.index, "compound", df)
            if cohort == "external":
                for cutoff in [60, 70]:
                    col = f"candidate_component_t{cutoff}"
                    labels = df.groupby("compound_id")[col].first().reindex(ce.index)
                    assert labels.notna().all()
                    add_resampling(cohort, scope, ce, CONTRASTS[2:], labels.to_numpy(), f"component_t{cutoff}", df)
                    for component, sub in df.groupby(col, sort=True):
                        sizes.append({"scope": scope, "threshold": cutoff / 100, "component": component,
                                      "n_records": len(sub), "n_compounds": sub.compound_id.nunique(),
                                      "n_source_categories": sub.source_split.nunique(),
                                      "n_Train_records": int((sub.source_split == "Train").sum()),
                                      "n_Test_records": int((sub.source_split == "Test").sum())})
        # Residual flagging is fully descriptive, uses all six models, and never filters metrics.
        absolute = pd.DataFrame({m: np.abs(frame[m] - frame.y) for m in MODELS})
        reasons = {i: [] for i in frame.index}
        for model in MODELS:
            top = frame.assign(_abs=absolute[model]).sort_values(["_abs", "raw_row"], ascending=[False, True]).head(math.ceil(.01 * len(frame))).index
            q1, q3 = absolute[model].quantile([.25, .75])
            for i in top:
                reasons[i].append(model + ":top1pct")
            for i in frame.index[absolute[model] > q3 + 1.5 * (q3 - q1)]:
                reasons[i].append(model + ":above_Tukey_fence")
        keep = ["raw_row", "source_id", "compound_id", "parent_smiles", "y", "source_split", "uncertain_stereo",
                "fold", "parent_charge", "above_stated_upper_1500_uM", "molecular_weight", "formal_charge", "TPSA",
                "max_tanimoto_full_AZ", "candidate_component_t60", "candidate_component_t70"]
        counts = frame.compound_id.value_counts()
        for i, why in reasons.items():
            if not why:
                continue
            row = {"cohort": cohort, "flag_reasons": "|".join(why), "primary_retained": True,
                   "compound_record_count": int(counts[frame.loc[i, "compound_id"]])}
            row.update({k: frame.loc[i, k] for k in keep if k in frame})
            for model in MODELS:
                row[model] = float(frame.loc[i, model])
                row[model + "__residual"] = float(frame.loc[i, model] - frame.loc[i, "y"])
                row[model + "__absolute_residual"] = float(absolute.loc[i, model])
            residual_rows.append(row)

    # Joint identity resampling preserves cross-category dependence if present.
    ids = sorted(external.compound_id.unique())
    cat_effects, cat_counts = [], []
    for category in ["Train", "Test"]:
        ce = compound_errors(external[external.source_split == category], MODELS).reindex(ids)
        cat_counts.append(ce.iloc[:, 0].notna().to_numpy(float))
        cat_effects.append(np.column_stack([(ce[a] - ce[b]).fillna(0).to_numpy() for a, b in CONTRASTS]))
    hetero_sums = np.hstack(cat_effects)
    hetero_counts = np.repeat(np.column_stack(cat_counts), len(CONTRASTS), axis=1)
    sampled = bootstrap_ratio(hetero_sums, hetero_counts)
    differences = sampled[:, 4:] - sampled[:, :4]
    boot_distributions["external__source_Test_minus_Train__compound_joint"] = differences
    heterogeneity = []
    for k, (a, b) in enumerate(CONTRASTS):
        train = cat_effects[0][:, k].sum() / cat_counts[0].sum()
        test = cat_effects[1][:, k].sum() / cat_counts[1].sum()
        low, high = np.quantile(differences[:, k], [.025, .975], method="linear")
        heterogeneity.append({"candidate": a, "reference": b, "effect_Train": train, "effect_Test": test,
                              "delta_Test_minus_Train": test - train, "ci_low": low, "ci_high": high,
                              "n_union_compounds": len(ids), "n_Train_compounds": int(cat_counts[0].sum()),
                              "n_Test_compounds": int(cat_counts[1].sum()),
                              "n_cross_category_compounds": verification["external_compounds_crossing_source_categories"],
                              "draws": DRAWS, "seed": SEED, "status": "exploratory_joint_compound_bootstrap"})
    effect_frame = pd.DataFrame(effects)
    classifications = []
    for a, b in CONTRASTS[2:]:
        sub = effect_frame[(effect_frame.cohort == "external") & (effect_frame.scope == "primary_all") &
                           (effect_frame.candidate == a) & (effect_frame.reference == b)]
        directions = dict(zip(sub.resampling_unit, sub.interval_direction))
        all_resolved = len(set(directions.values())) == 1 and "unresolved" not in directions.values()
        opposite = directions["compound"] != "unresolved" and any(
            x not in ["unresolved", directions["compound"]] for x in directions.values())
        classifications.append({"candidate": a, "reference": b, "directions": directions,
                                "direction_resolved_all_three_levels": all_resolved,
                                "interpretation": "Direction resolved at all three resampling levels" if all_resolved else
                                "Direction differs across resampling levels" if opposite else
                                "Compound-level resolved; analogue-component uncertainty remains" if directions["compound"] != "unresolved" else
                                "Direction unresolved under compound resampling",
                                "compound_influence_sensitive": bool(sub.loc[sub.resampling_unit == "compound", "influence_sensitive"].iloc[0]),
                                "any_component_influence_sensitive": bool(sub.loc[sub.resampling_unit != "compound", "influence_sensitive"].any())})
    tables = {"metrics": pd.DataFrame(metric_rows), "paired_effects": effect_frame, "compound_effects": pd.DataFrame(compound_rows),
              "influence": pd.DataFrame(influence_rows), "component_sizes": pd.DataFrame(sizes),
              "extreme_residuals": pd.DataFrame(residual_rows), "source_heterogeneity": pd.DataFrame(heterogeneity),
              "existing_point_verification": pd.DataFrame(points)}
    for name, table in tables.items():
        table.to_csv(OUT / (name + ".csv"), index=False, float_format="%.17g")
    tables["influence"][tables["influence"].flagged].to_csv(OUT / "influence_flagged.csv", index=False, float_format="%.17g")
    external[[c for c in external.columns if c in ["raw_row", "source_id", "compound_id", "source_split", "uncertain_stereo", "candidate_component_t60", "candidate_component_t70"]]].to_csv(OUT / "external_component_membership.csv", index=False)
    np.savez_compressed(OUT / "bootstrap_draws.npz", **boot_distributions)
    verification["primary_point_checks"] = len(points)
    verification["maximum_primary_point_difference"] = float(pd.DataFrame(points).absolute_difference.max())
    verification["plan_unchanged"] = sha(plan_path) == (OUT / "PLAN.sha256").read_text().strip()
    verification["old_inputs_unchanged"] = all(sha(ROOT / v["path"]) == v["sha256"] for v in plan["inputs"].values())
    assert verification["plan_unchanged"] and verification["old_inputs_unchanged"]
    json_dump(OUT / "verification.json", verification)
    json_dump(OUT / "external_classification.json", classifications)
    write_report(tables, classifications, verification)
    artifacts = {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}
    json_dump(OUT / "READY.json", {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                                  "plan_sha256": sha(plan_path), "source_sha256": sha(__file__), "artifact_hashes": artifacts,
                                  "report_sha256": sha(REPORT), "external_interpretations": classifications}, exclusive=True)
    print(json.dumps({"status": "complete", "report": str(REPORT), "external": classifications}, indent=2))


def markdown_table(frame, columns):
    head = "| " + " | ".join(columns) + " |"
    rows = [head, "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(f"{row[c]:.4f}" if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns) + " |")
    return "\n".join(rows)


def write_report(t, classifications, verification):
    eff = t["paired_effects"]
    primary = eff[(eff.scope == "primary_all") & (eff.resampling_unit == "compound")].copy()
    external = eff[(eff.scope == "primary_all") & (eff.cohort == "external") & eff.candidate.isin(["uniform", "prior"])].copy()
    lines = ["# R6 saved-prediction statistical robustness", "",
             "This is an exploratory post-R5 analysis. The analysis plan, input hashes, and code hash were frozen before calculating these R6 results; the contrasts were not preregistered before R5 outcomes. No model was fitted, no R5 result was edited, and no residual-based observation exclusion was applied.", "",
             "## Estimand and uncertainty", "",
             "Each contrast is candidate minus reference in absolute error, first averaged within a normalized compound and then equally across compounds. A negative value favors the candidate. Ten thousand percentile bootstrap draws use seed 261024. Intervals are unadjusted and conditional on the saved fits and fixed cohorts. They do not include partition selection, retraining, or upstream-model uncertainty.", "",
             "Analogue resampling draws the existing connected components, retaining all their compounds. Every draw divides the sampled sum of compound differences by the sampled compound count. This preserves the compound-weighted point estimand rather than giving large and small components equal estimand weight. Subsets keep their original full-cohort component labels. Components may include transitive bridges and are candidate analogue groups, not experimentally verified independent series.", "",
             "## Compound-bootstrap primary contrasts", "",
             markdown_table(primary, ["cohort", "candidate", "reference", "delta_MAE", "ci_low", "ci_high", "n_compounds"]), "",
             "## External analogue-component sensitivity", "",
             markdown_table(external, ["candidate", "reference", "resampling_unit", "delta_MAE", "ci_low", "ci_high", "n_units", "singleton_units", "largest_unit_compounds", "concentration_effective_units"]), "",
             "Concentration-effective units are (total compound count)^2 / sum(component compound count^2), a size-concentration diagnostic; they are not an estimate of the number of independent chemical series.", ""]
    for obj in classifications:
        lines.append(f"- {obj['candidate']} minus {obj['reference']}: **{obj['interpretation']}**. Compound influence-sensitive: {obj['compound_influence_sensitive']}; any component influence-sensitive: {obj['any_component_influence_sensitive']}.")
    lines += ["", "## Record and compound balance", "",
              markdown_table(t["metrics"][t["metrics"].scope == "primary_all"], ["cohort", "model", "record_MAE", "compound_MAE"]), ""]
    for cohort, group in t["metrics"][t["metrics"].scope == "primary_all"].groupby("cohort"):
        rec = group.sort_values("record_MAE").model.tolist()
        comp = group.sort_values("compound_MAE").model.tolist()
        lines.append(f"- {cohort}: record ranking {' < '.join(rec)}; compound ranking {' < '.join(comp)}. " +
                     ("The rank was sensitive to whether records or normalized compounds were weighted equally." if rec != comp else "The complete numerical ranking is unchanged."))
    scope_eff = eff[(eff.cohort == "external") & (eff.scope != "primary_all") & (eff.resampling_unit == "compound")]
    lines += ["", "## Complete external scope heterogeneity", "",
              markdown_table(scope_eff, ["scope", "candidate", "reference", "delta_MAE", "ci_low", "ci_high", "n_compounds"]), "",
              "Both released source categories are external to the AZ-trained models. Test-minus-Train differences below jointly resample the union of identities, so an identity appearing in both categories would remain linked.", "",
              markdown_table(t["source_heterogeneity"], ["candidate", "reference", "effect_Train", "effect_Test", "delta_Test_minus_Train", "ci_low", "ci_high", "n_cross_category_compounds"]), "",
              "## Influence and extreme residuals", "",
              "Exact leave-one-compound and leave-one-component results are saved for all relevant contrasts and scopes. Flags identify an opposite point-estimate sign or a change greater than 25% of the full absolute effect; the relative rule is disabled below an absolute effect of 1e-12. Flags are diagnostic and do not change the primary observations or estimand.", "",
              markdown_table(external, ["candidate", "resampling_unit", "n_sign_flips", "n_relative_change_gt25pct", "max_abs_leave_one_change", "max_relative_leave_one_change"]), ""]
    top = t["influence"][(t["influence"].cohort == "external") & (t["influence"].scope == "primary_all") & (t["influence"].candidate == "prior")].copy()
    top["absolute_change"] = top.change.abs()
    top = top.sort_values(["unit", "absolute_change"], ascending=[True, False]).groupby("unit", sort=False).head(3)
    lines += [markdown_table(top, ["unit", "group_id", "n_compounds_removed", "full_delta", "leave_one_delta", "change", "relative_abs_change", "sign_flip"]), "",
              "The extreme-residual table uses the union of the top 1% and Tukey-fence flags from all six graph models. It retains source/stereogroup categories, all six predictions, and signed and absolute residuals. Prior-confidence variables are not inferred from older descriptor fields here and can be joined separately by raw_row. No flagged record is removed from any primary calculation.", "",
              "## Verification and deliverables", "",
              f"Verified {verification['primary_point_checks']} existing primary MAE/RMSE/R2/paired point values; maximum absolute discrepancy {verification['maximum_primary_point_difference']:.3g}. Component membership is complete and one-to-one by raw_row, and all duplicate normalized compounds remain within one component at each threshold. There are {verification['external_compounds_crossing_source_categories']} identities crossing source categories. Synthetic checks cover duplicate balancing, unequal component ratios, singleton-component equivalence, exact influence, and zero-effect handling. All input hashes and the immutable plan hash remain unchanged.", "",
              "Machine-readable results in `runs/round6/robustness/`: `paired_effects.csv` (all scopes and units), `metrics.csv`, `compound_effects.csv`, `component_sizes.csv`, `external_component_membership.csv`, `source_heterogeneity.csv`, `influence.csv`/`influence_flagged.csv`, `extreme_residuals.csv`, `existing_point_verification.csv`, `bootstrap_draws.npz`, `verification.json`, `external_classification.json`, and the immutable `PLAN.json`. Solubility range and identity-overlap sensitivities remain available in the complete tidy tables, without retraining or changing labels.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["self-test", "plan", "run"])
    arg = parser.parse_args()
    {"self-test": self_test, "plan": make_plan, "run": run}[arg.command]()
