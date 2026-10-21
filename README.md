# PhyMicroNet: fixed-pH property prediction reproducibility

**Statistics-only, analysis-only repository.** Recompute the numerical analyses for controlled comparisons of ionization information, molecular state representation, and occupancy weighting in fixed-pH molecular property prediction, using frozen R6/R5 inputs.

## What this reproduces

- Three fixed cohorts: AZ logD (4,200 records / 4,178 compounds), AZ solubility test set (353 / 350), and ASAP external logD (473 / 413).
- Six model columns; 12 primary compound-equal paired contrasts.
- 10,000-draw compound and Tanimoto-component robustness bootstrap; influence and extreme-residual descriptions.
- Candidate-public-training-corpus overlap sensitivity, 5,000-draw prior diagnostics, fixed bins and pH flank descriptions.
- External chemical-domain statistics, six transparently selected cases, experimental pKa comparison and integrated diagnostic tables.
- All 46 expected CSV/NPZ artifacts are recalculated and compared at absolute tolerance `1e-10`; numeric NaN positions and nonnumeric labels must also match.

**Not included:** model weights, model training, neural forward inference, upstream million-row corpus rescanning, or unverified new training results. Bootstrap intervals are conditional on the frozen predictions and cohorts; this is not training-uncertainty reproduction. Historical hardware timing is not remeasured.

## Run

Python 3.12 (historical analysis: 3.12.13) or Python 3.13 is recommended. The repository was also reproduced in a newly created isolated Python 3.13 environment (actual interpreter recorded in `PUBLIC_VALIDATION.json`: 3.13.14): all 46 CSV/NPZ comparisons passed with maximum numerical difference zero. RDKit and the four direct scientific dependencies in `requirements.txt` are required; Torch and Chemprop are not. RDKit installs Pillow transitively, but repository code does not import plotting or imaging libraries.

```bash
python -m venv .venv
# Activate .venv using your operating system's normal command.
python -m pip install -r requirements.txt
python reproduce.py --mode verify
python reproduce.py --mode stats --run-id my_statistics
```

Run from the repository root. No machine-specific source path, internet connection, checkpoint or dataset download is required after installing dependencies. Every invocation needs a new run ID; existing output directories are protected.

Outputs: `reproduced/<run-id>/verification.json`, seven numerical-analysis directories, `execution.json`, and logs. `verification.json` contains all comparison results, the 12 primary contrasts, and installed versions. Exit zero alone is not sufficient evidence: inspect `complete`, `comparison_files`, and each comparison.

## Metadata sanitation

Numeric input CSVs/NPZs and benchmark tables are copied byte-for-byte. Historical absolute paths are removed from selected JSON metadata. Non-public references and unused fields were omitted; dependent metadata hashes were regenerated. Scientific definitions, RNG seeds, resampling counts and data are unchanged. Git automatic line-ending conversion is disabled because input/source hash gates are byte-sensitive.

`package_manifest.json` protects every distributed file except itself. Generated runs and local environments are untracked. `PUBLIC_VALIDATION.json` records the validation run.

## Data attribution and permissions

See `DATA_LICENSES.txt`: AZ experimental data trace to ChEMBL/MoleculeNet; ASAP traces to the CC-BY-4.0 Zenodo release. Upstream candidate-corpus diagnostics are derived overlap summaries, not redistribution of the full candidate corpora. No blanket permissive license is applied to third-party data. Project code has no newly granted license; contact the rights holder for reuse beyond applicable law.

## Scientific boundaries

Internal partitions support a controlled development comparison, not independent-source validation. AZ logD and solubility share source/structures. ASAP is independent-source but contains structural-component dependence. Compound and component CI disagreements are retained. Public-corpus nonmatching is not proof that the exact deployed pKa checkpoint never saw a compound. Prior associations and case selection are exploratory, not mechanistic or causal validation. New complete retraining is a separate local task and is excluded here.
