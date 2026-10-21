"""Fixed, exploratory saved-prediction sensitivity to recovered corpus matches.

PLAN.json precedes this analysis. Only chembl_train and dwar-iBond form the
candidate training-corpus union. No claim of exact checkpoint membership.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/round6/exposure_sensitivity"
REPORT = ROOT / "reports/ROUND6_EXPOSURE_SENSITIVITY.md"
ENDPOINTS = {"logd": "AZ_logD", "solubility": "AZ_solubility", "external": "ASAP_logD"}
MODELS = ["parent_only", "parent_fn", "parent_ionization", "dominant", "uniform", "prior"]
CONTRASTS = [("parent_ionization", "parent_only"), ("dominant", "parent_ionization"),
             ("uniform", "dominant"), ("prior", "uniform")]
RULES = ["project_normalized", "stereo_stripped_normalized"]
CORPORA = {"chembl_train", "dwar-iBond"}
SEED, DRAWS = 261024, 10000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def classify(queries, matches, endpoint, rule):
    """Any matching query record exposes its entire normalized compound."""
    query = queries[queries.endpoint == endpoint].copy()
    assert query.raw_row.is_unique
    relevant = matches[(matches.endpoint == endpoint) & (matches.strategy == rule) & matches.corpus.isin(CORPORA)]
    assert set(relevant.raw_row).issubset(set(query.raw_row))
    matching_rows = set(relevant.raw_row)
    matching_compounds = set(query.loc[query.raw_row.isin(matching_rows), "compound_id"])
    query["direct_matching_record"] = query.raw_row.isin(matching_rows)
    query["matched_candidate_corpus"] = query.compound_id.isin(matching_compounds)
    assert (query.groupby("compound_id").matched_candidate_corpus.nunique() == 1).all()
    query["matching_candidate_corpora"] = query.compound_id.map(
        relevant.merge(query[["raw_row", "compound_id"]], on="raw_row", validate="many_to_one")
        .groupby("compound_id").corpus.agg(lambda x: "|".join(sorted(set(x)))))
    query["matching_candidate_corpora"] = query.matching_candidate_corpora.fillna("")
    return query


def self_tests(kernel):
    query = pd.DataFrame({"endpoint": ["test"] * 4, "raw_row": [0, 1, 2, 3],
                          "compound_id": ["a", "a", "b", "c"]})
    matches = pd.DataFrame({"endpoint": ["test"] * 5, "raw_row": [0, 2, 3, 0, 2],
                            "strategy": ["project_normalized"] * 3 + ["stereo_stripped_normalized"] * 2,
                            "corpus": ["chembl_train", "chembl_valid", "novartis_acid", "dwar-iBond", "chembl_train"]})
    p = classify(query, matches, "test", "project_normalized")
    assert p.matched_candidate_corpus.tolist() == [True, True, False, False]
    assert p.direct_matching_record.tolist() == [True, False, False, False]
    s = classify(query, matches, "test", "stereo_stripped_normalized")
    assert s.matched_candidate_corpus.tolist() == [True, True, True, False]
    np.testing.assert_array_equal(kernel.bootstrap_ratio([2., 3.], [1, 1], draws=500),
                                  kernel.bootstrap_ratio([2., 3.], [1, 1], draws=500))
    return ["Any matching row carries duplicate compound into matched stratum",
            "Validation and Novartis are excluded from candidate training union",
            "Stereo-stripped matches may expand normalized compound membership",
            "Verified singleton-component kernel has identical draws to compound kernel"]


def compound_errors(df):
    result = df[["compound_id"]].copy()
    for model in MODELS:
        result[model] = (df[model] - df.y).abs()
    return result.groupby("compound_id", sort=True)[MODELS].mean()


def main():
    plan_path = OUT / "PLAN.json"
    assert sha(plan_path) == (OUT / "PLAN.sha256").read_text().strip()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert set(x.name for x in OUT.iterdir()) == {"PLAN.json", "PLAN.sha256"}, "Never overwrite prior stratum evaluation."
    assert not REPORT.exists()
    assert set(plan["candidate_training_corpora"]) == CORPORA
    assert plan["identity_rules"] == RULES and plan["models"] == MODELS
    assert plan["contrasts"] == [list(v) for v in CONTRASTS]
    assert plan["bootstrap"]["seed"] == SEED and plan["bootstrap"]["draws"] == DRAWS
    for item in plan["inputs"].values():
        assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
    ready = json.loads((ROOT / plan["inputs"]["upstream_ready"]["path"]).read_text(encoding="utf-8"))
    assert ready["audit_run_complete"] and ready["full_released_corpora_scanned"]
    for key in ["identity_matches", "query_identities", "corpus_inventory"]:
        p = ROOT / plan["inputs"][key]["path"]
        assert sha(p) == ready["artifacts"][p.name]
    # Freeze execution source before reading match assignments or scoring strata.
    write_json(OUT / "EXECUTION.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                      "plan_sha256": sha(plan_path), "source_sha256": sha(__file__),
                                      "kernel_sha256": plan["inputs"]["verified_kernel"]["sha256"],
                                      "upstream_gate_complete": True}, exclusive=True)
    spec = importlib.util.spec_from_file_location("r6_verified_kernel", ROOT / plan["inputs"]["verified_kernel"]["path"])
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    tests = self_tests(kernel)
    queries = pd.read_csv(ROOT / plan["inputs"]["query_identities"]["path"])
    matches = pd.read_csv(ROOT / plan["inputs"]["identity_matches"]["path"])
    inventory = pd.read_csv(ROOT / plan["inputs"]["corpus_inventory"]["path"])
    assert set(queries.endpoint) == set(ENDPOINTS.values())
    assert not queries.duplicated(["endpoint", "raw_row"]).any()
    # Validate the relation between every relevant match and query before grouping.
    relevant = matches[matches.corpus.isin(CORPORA) & matches.strategy.isin(RULES)]
    relation = relevant.merge(queries[["endpoint", "raw_row", "source_id"] + RULES],
                              on=["endpoint", "raw_row"], suffixes=("", "__query"), validate="many_to_one")
    assert len(relation) == len(relevant)
    assert (relation.source_id.astype(str) == relation.source_id__query.astype(str)).all()
    for rule in RULES:
        sub = relation[relation.strategy == rule]
        assert (sub.identity_key == sub[rule]).all()
    frames = {c: pd.read_csv(ROOT / plan["inputs"][c + "_predictions"]["path"]) for c in ENDPOINTS}
    neighbor = pd.read_csv(ROOT / plan["inputs"]["external_neighbors"]["path"])
    assert neighbor.raw_row.is_unique
    assert set(frames["external"].raw_row) == set(neighbor.raw_row)
    ext = frames["external"].merge(neighbor[["raw_row", "compound_id", "candidate_component_t60", "candidate_component_t70"]],
                                    on="raw_row", validate="one_to_one", suffixes=("", "__neighbor"))
    assert (ext.compound_id == ext.compound_id__neighbor).all()
    for column in ["candidate_component_t60", "candidate_component_t70"]:
        assert ext[column].notna().all() and (ext.groupby("compound_id")[column].nunique() == 1).all()
    frames["external"] = ext
    assignments, counts, metrics, effects, distributions = [], [], [], [], {}
    verification = {"self_tests": tests, "upstream_ready_complete": True, "duplicate_compounds_split_between_strata": 0,
                    "identity_relation_rows_verified": len(relevant), "reconstruction_checks": []}

    def add_effects(cohort, rule, stratum, df, ce, unit, labels, contrasts):
        delta = pd.DataFrame({f"{a}__minus__{b}": ce[a] - ce[b] for a, b in contrasts})
        delta["group"] = np.asarray(labels)
        grouped = delta.groupby("group", sort=True)
        sums = grouped[delta.columns[:-1]].sum()
        sizes = grouped.size().astype(float)
        n_units = len(sizes)
        if n_units >= 2:
            draws = kernel.bootstrap_ratio(sums.to_numpy(), sizes.to_numpy(), draws=DRAWS, seed=SEED)
            key = f"{cohort}__{rule}__{stratum}__{unit}"
            distributions[key] = draws
            q = np.quantile(draws, [.025, .975], axis=0, method="linear")
            status = "exploratory_conditional_unadjusted"
        else:
            q = np.full((2, len(contrasts)), np.nan)
            key = ""
            status = "empty_stratum" if len(ce) == 0 else "insufficient_resampling_units"
        for j, (a, b) in enumerate(contrasts):
            point = float((ce[a] - ce[b]).mean()) if len(ce) else np.nan
            if len(ce):
                np.testing.assert_allclose(sums.iloc[:, j].sum() / sizes.sum(), point, atol=1e-14, rtol=0)
            effects.append({"cohort": cohort, "identity_rule": rule, "stratum": stratum,
                            "resampling_unit": unit, "candidate": a, "reference": b,
                            "n_records": len(df), "n_compounds": len(ce), "n_units": n_units,
                            "largest_unit_compounds": int(sizes.max()) if n_units else 0,
                            "concentration_effective_units": float(len(ce) ** 2 / np.sum(sizes ** 2)) if n_units else np.nan,
                            "delta_MAE": point, "ci_low": q[0, j], "ci_high": q[1, j],
                            "draws": DRAWS if n_units >= 2 else 0, "seed": SEED,
                            "status": status, "bootstrap_key": key, "bootstrap_column": j})

    for cohort, endpoint in ENDPOINTS.items():
        frame = frames[cohort]
        assert frame.raw_row.is_unique
        assert np.isfinite(frame[["y"] + MODELS].to_numpy(float)).all()
        full_errors = compound_errors(frame)
        for rule in RULES:
            q = classify(queries, matches, endpoint, rule)
            selected = frame.merge(q[["raw_row", "source_id", "compound_id", "direct_matching_record", "matched_candidate_corpus", "matching_candidate_corpora"]],
                                   on="raw_row", suffixes=("", "__query"), validate="one_to_one")
            assert len(selected) == len(frame)
            for field in ["compound_id", "source_id"]:
                assert (selected[field].astype(str) == selected[field + "__query"].astype(str)).all()
            assert (selected.groupby("compound_id").matched_candidate_corpus.nunique() == 1).all()
            for _, row in selected.iterrows():
                assignments.append({"cohort": cohort, "endpoint": endpoint, "identity_rule": rule,
                                    "raw_row": row.raw_row, "source_id": row.source_id, "compound_id": row.compound_id,
                                    "direct_matching_record": bool(row.direct_matching_record),
                                    "matched_candidate_corpus": bool(row.matched_candidate_corpus),
                                    "stratum": "matched" if row.matched_candidate_corpus else "nonmatched",
                                    "matching_candidate_corpora": row.matching_candidate_corpora})
            reconstructed = []
            for stratum, match_flag in [("matched", True), ("nonmatched", False)]:
                df = selected[selected.matched_candidate_corpus == match_flag]
                ce = compound_errors(df)
                reconstructed.append(ce)
                counts.append({"cohort": cohort, "identity_rule": rule, "stratum": stratum,
                               "n_records": len(df), "n_compounds": len(ce),
                               "total_evaluated_records": len(frame), "total_evaluated_compounds": len(full_errors),
                               "record_fraction": len(df) / len(frame), "compound_fraction": len(ce) / len(full_errors),
                               "source_query_records": len(q), "source_query_compounds": q.compound_id.nunique(),
                               "direct_match_evaluated_records": int(df.direct_matching_record.sum()),
                               "status": "empty_stratum" if not len(df) else "descriptive_exposure_stratum"})
                for model in MODELS:
                    metrics.append({"cohort": cohort, "identity_rule": rule, "stratum": stratum, "model": model,
                                    "n_records": len(df), "n_compounds": len(ce),
                                    "record_MAE": float((df[model] - df.y).abs().mean()) if len(df) else np.nan,
                                    "compound_MAE": float(ce[model].mean()) if len(ce) else np.nan,
                                    "status": "empty_stratum" if not len(df) else "descriptive_exposure_stratum"})
                add_effects(cohort, rule, stratum, df, ce, "compound", ce.index, CONTRASTS)
                if cohort == "external":
                    for cutoff in [60, 70]:
                        labels = df.groupby("compound_id")[f"candidate_component_t{cutoff}"].first().reindex(ce.index)
                        assert labels.notna().all()
                        add_effects(cohort, rule, stratum, df, ce, f"component_t{cutoff}", labels.to_numpy(), CONTRASTS[2:])
            restored = pd.concat(reconstructed).sort_index()
            assert restored.index.is_unique and restored.index.equals(full_errors.index)
            np.testing.assert_allclose(restored.to_numpy(), full_errors.to_numpy(), atol=0, rtol=0)
            verification["reconstruction_checks"].append({"cohort": cohort, "rule": rule, "all_compounds_retained_once": True,
                                                          "compound_error_partition_exact": True})
    tables = {"sample_counts": pd.DataFrame(counts), "record_exposure_assignments": pd.DataFrame(assignments),
              "metrics": pd.DataFrame(metrics), "paired_effects": pd.DataFrame(effects)}
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False, float_format="%.17g")
    inventory[inventory.corpus.isin(CORPORA)].to_csv(OUT / "candidate_training_corpus_inventory.csv", index=False)
    np.savez_compressed(OUT / "bootstrap_draws.npz", **distributions)
    verification.update({"plan_unchanged": sha(plan_path) == (OUT / "PLAN.sha256").read_text().strip(),
                         "inputs_unchanged": all(sha(ROOT / v["path"]) == v["sha256"] for v in plan["inputs"].values()),
                         "empty_effect_rows": int((tables["paired_effects"].status == "empty_stratum").sum()),
                         "new_fits": 0, "new_forwards": 0})
    assert verification["plan_unchanged"] and verification["inputs_unchanged"]
    write_json(OUT / "verification.json", verification)
    make_report(tables, inventory, ready, kernel.markdown_table)
    write_json(OUT / "READY.json", {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                                   "plan_sha256": sha(plan_path), "source_sha256": sha(__file__),
                                   "artifacts": {p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
                                   "report_sha256": sha(REPORT), "scope": plan["interpretation"]}, exclusive=True)
    print(json.dumps({"status": "complete", "counts": tables["sample_counts"][["cohort", "identity_rule", "stratum", "n_records", "n_compounds"]].to_dict("records")}, indent=2))


def make_report(tables, inventory, ready, markdown_table):
    counts, metrics, effects = (tables[k] for k in ["sample_counts", "metrics", "paired_effects"])
    lines = ["# R6 recovered-public-corpus exposure sensitivity", "",
             "This is an exploratory, post-result analysis of unchanged saved R5 ensemble predictions. The local analysis plan was frozen before these stratum calculations. Only **chembl_train and dwar-iBond** enter the union of candidate public training corpora. Released ChEMBL validation, Novartis, and SAMPL identities are excluded from that union.", "",
             "A recovered public-corpus match does not prove membership in the exact deployed checkpoint training data. A nonmatch is not certified unseen: the exact checkpoint row manifests, version attribution, and some failed structures remain unresolved. Matching is a structure-exposure descriptor rather than proof of downstream response leakage or a causal explanation of error. These strata do not replace the full-cohort primary analysis.", "",
             "Within each endpoint and identity rule, any matching source record places its entire normalized compound in the matched stratum; duplicate records are never split. Stereo-stripped matches may encompass additional project-normalized compounds. Only evaluated records with frozen R5 ensemble predictions enter the scores: 4,200 internal logD, 353 fixed solubility-test, and 473 external logD records.", "",
             "## Coverage", "",
             markdown_table(counts, ["cohort", "identity_rule", "stratum", "n_records", "n_compounds", "compound_fraction"]), "",
             "## Six-model errors", "",
             "Record MAE weights observations equally. Compound MAE first averages absolute errors within each normalized identity and then weights identities equally. Empty strata have NA values, not zero errors.", "",
             markdown_table(metrics, ["cohort", "identity_rule", "stratum", "model", "record_MAE", "compound_MAE"]), "",
             "## Four designated paired contrasts", "",
             "Differences are candidate minus reference in absolute error, averaged within compounds and then equally across compounds. Negative favors the candidate. Intervals use 10,000 percentile compound draws, seed 261024, and condition on saved fits and fixed cohorts; they are unadjusted and exploratory. These are within-stratum model comparisons, not causal estimates of the effect of upstream exposure. Singleton or empty strata receive no bootstrap interval.", "",
             markdown_table(effects[effects.resampling_unit == "compound"], ["cohort", "identity_rule", "stratum", "candidate", "reference", "delta_MAE", "ci_low", "ci_high", "status"]), "",
             "## External analogue-component sensitivity", "",
             "Existing T60/T70 labels are retained without rebuilding components. With at least two represented components, each draw divides sampled compound-difference totals by sampled compound counts, preserving the compound-weighted estimand. Empty or insufficient-unit strata receive NA intervals. The graph comparisons use identical fitted models and saved predictions.", "",
             markdown_table(effects[(effects.cohort == "external") & (effects.resampling_unit != "compound")], ["identity_rule", "stratum", "candidate", "reference", "resampling_unit", "n_units", "delta_MAE", "ci_low", "ci_high", "status"]), ""]
    external_counts = counts[(counts.cohort == "external") & (counts.stratum == "matched")]
    if (external_counts.n_compounds == 0).all():
        lines += ["No evaluated ASAP compound matches either candidate-training union under either requested identity rule. Consequently, the external nonmatched stratum is exactly the full primary cohort, reproducing the original compound and component sensitivity; there is no matched external comparator and no new external validation evidence from this subdivision. This does not certify that the deployed upstream checkpoint was unexposed to ASAP chemistry.", ""]
    failed = int(inventory[inventory.corpus.isin(CORPORA)].failed_states.sum())
    lines += ["## Verification and remaining attribution limits", "",
              f"The consumed upstream READY reports completion of all released corpora, with {ready['failed_states']} failed states overall and {failed} in the two selected candidate-training corpora. Exact deployed-checkpoint exposure remains nonexhaustive. The public Dwar-iBond release and checkpoint task/version attribution must not be conflated.", "",
              "All relevant identity-match keys and source IDs were checked against the audit query table. Six cohort/rule partitions reconstruct the complete saved compound-error matrices exactly. Synthetic checks verify whole-compound propagation and exclusion of validation/Novartis matches. The verified R6 bootstrap kernel is reused by hash; no model code is run. All inputs and the frozen plan remain unchanged.", "",
              "Machine-readable `sample_counts.csv`, `record_exposure_assignments.csv`, `metrics.csv`, `paired_effects.csv`, `candidate_training_corpus_inventory.csv`, and `bootstrap_draws.npz` are in `runs/round6/exposure_sensitivity/`, with plan, execution hashes, verification and completion metadata. NA fields are blank in CSV and carry an explicit status. No strata are omitted because of sample size or performance.", ""]
    REPORT.write_text("\n".join(lines).replace(" nan ", " NA "), encoding="utf-8")


if __name__ == "__main__":
    main()
