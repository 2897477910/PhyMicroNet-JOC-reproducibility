"""Auditable structure handling and deliberately limited heuristic speciation.

The input parent is never neutralized or tautomer-canonicalized for modeling.
Those transformations apply ONLY to the split/group identity. Enumerated states
are candidate representations, not measured equilibrium populations. The HH
prior is restricted to simple one-site acids/bases and uses broad functional-
group pKa means, not compound-specific pKa predictions.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


RULE_SOURCE = (
    "https://raw.githubusercontent.com/durrantlab/dimorphite_dl/v2.0.2/"
    "dimorphite_dl/smarts/site_substructures.smarts"
)
RULES = {
    "Carboxyl": (3.456652971502591, 1.2871420886834017),
    "Phenyl_carboxyl": (3.463441968255319, 1.2518054407928614),
    "Amines_primary_secondary_tertiary": (8.159107682388349, 2.5183597445318147),
}
# Only unequivocal counterions are removed. Unknown organic mixtures are kept.
COUNTERIONS = frozenset(
    {"[F-]", "[Cl-]", "[Br-]", "[I-]", "[Li+]", "[Na+]", "[K+]",
     "[Mg+2]", "[Ca+2]", "[Zn+2]"}
)
_CARBOXYL = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1,OX1-]")
_PHENOL_ENOL = Chem.MolFromSmarts("[c,CX3]-[OX2H1,OX1-]")
_BETA_DICARBONYL = Chem.MolFromSmarts("[CX3](=O)-[CX4]-[CX3](=O)")
_PEROXIDE = Chem.MolFromSmarts("[O]-[O]")


def _mol(smiles: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return molecule


def _smiles(molecule: Chem.Mol) -> str:
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _strip_counterions(molecule: Chem.Mol) -> tuple[Chem.Mol, list[str]]:
    fragments = list(Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True))
    if len(fragments) == 1:
        return molecule, []
    retained = [fragment for fragment in fragments if _smiles(fragment) not in COUNTERIONS]
    removed = [_smiles(fragment) for fragment in fragments if _smiles(fragment) in COUNTERIONS]
    if len(retained) == 1 and any(atom.GetAtomicNum() == 6 for atom in retained[0].GetAtoms()):
        return retained[0], ["removed_explicit_counterions:" + ",".join(sorted(removed))]
    return molecule, ["unresolved_multifragment_input"]


def _group_mol(molecule: Chem.Mol) -> Chem.Mol:
    # Permanent charges remain permanent; Uncharger must not invent chemistry.
    uncharged = rdMolStandardize.Uncharger(canonicalOrder=True).uncharge(Chem.Mol(molecule))
    enumerator = rdMolStandardize.TautomerEnumerator()
    canonical = enumerator.Canonicalize(uncharged)
    return canonical


@lru_cache(maxsize=20000)
def standardize(smiles: str) -> dict:
    """Return a preserved parent representation and separate split identity.

    ``compound_id`` groups reversible protonation/tautomer alternatives while
    preserving stereo where defined after tautomer canonicalization. It is a
    conservative grouping identifier, not a registry compound identifier.
    """
    original = _mol(smiles)
    parent, warnings = _strip_counterions(original)
    parent_smiles = _smiles(parent)
    grouped = _group_mol(parent)
    group_smiles = _smiles(grouped)
    try:
        inchikey = Chem.MolToInchiKey(parent)
    except Exception:
        inchikey = ""
        warnings.append("inchikey_generation_failed")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=grouped, includeChirality=False)
    return {
        "original_smiles": smiles,
        "parent_smiles": parent_smiles,
        "compound_id": hashlib.sha256(group_smiles.encode()).hexdigest()[:24],
        "group_smiles": group_smiles,
        "inchikey": inchikey,
        "scaffold": scaffold,
        "warning": ";".join(warnings),
    }


def _heavy_elements(molecule: Chem.Mol) -> Counter:
    return Counter(atom.GetAtomicNum() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1)


@lru_cache(maxsize=20000)
def enumerate_states(
    parent_smiles: str,
    max_states: int = 16,
    *,
    ph_min: float = 6.4,
    ph_max: float = 8.4,
    include_tautomers: bool = True,
) -> dict:
    """Deterministically rank a bounded protonation/tautomer candidate set.

    The reference parent is retained. A common fixed pH window is used for all
    molecules; this is not quantitative population prediction. Internal or final
    enumeration caps are reported, including when truncation is only possible.
    """
    if not isinstance(max_states, int) or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if not (math.isfinite(ph_min) and math.isfinite(ph_max) and 0 <= ph_min <= ph_max <= 14):
        raise ValueError("Require finite 0 <= ph_min <= ph_max <= 14")
    parent = _mol(parent_smiles)
    reference = _smiles(parent)
    result = {"states": [reference], "status": "OK", "truncated": False,
              "warning": "", "population_semantics": "unweighted_candidate_representations",
              "ph_min": ph_min, "ph_max": ph_max}
    if len(Chem.GetMolFrags(parent)) != 1:
        result.update(status="FAIL", warning="multifragment_parent;canonical_fallback")
        return result
    try:
        from dimorphite_dl import protonate_smiles
        internal_cap = max(128, max_states * 8)
        raw = protonate_smiles(reference, ph_min=ph_min, ph_max=ph_max,
                               precision=1.0, max_variants=internal_cap)
        if not raw:
            raise ValueError("enumerator returned no states")
        capped = len(raw) >= internal_cap
        all_states = {reference}
        protonated = set()
        for item in raw:
            candidate = _mol(item)
            if _heavy_elements(candidate) != _heavy_elements(parent):
                raise ValueError("enumerator changed heavy atom composition")
            protonated.add(_smiles(candidate))
        if include_tautomers:
            tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
            tautomer_enumerator.SetMaxTautomers(64)
            tautomer_enumerator.SetMaxTransforms(256)
            for candidate_smiles in sorted(protonated | {reference}):
                enumeration = tautomer_enumerator.Enumerate(_mol(candidate_smiles))
                capped |= enumeration.status != rdMolStandardize.TautomerEnumeratorStatus.Completed
                all_states.update(_smiles(tautomer) for tautomer in enumeration)
        else:
            all_states.update(protonated)
        ordered = [reference] + sorted(all_states - {reference})
        capped |= len(ordered) > max_states
        result.update(states=ordered[:max_states], truncated=bool(capped),
                      status="TRUNCATED" if capped else "OK",
                      n_candidates_before_final_cap=len(ordered),
                      warning="enumeration_cap_reached_or_possible" if capped else "")
    except Exception as error:
        result.update(status="FAIL", warning=f"{type(error).__name__}:{error};canonical_fallback")
    return result


def _rejected(reason: str) -> dict:
    return {"weights": None, "prior_states": None, "eligible": False, "rule": None,
            "pka": None, "pka_sd": None, "reason": reason,
            "weight_kind": "unavailable", "source": RULE_SOURCE}


def _set_charge_and_h(molecule: Chem.Mol, index: int, charge: int, hydrogens: int) -> Chem.Mol:
    result = Chem.Mol(molecule)
    atom = result.GetAtomWithIdx(index)
    atom.SetFormalCharge(charge)
    atom.SetNumExplicitHs(hydrogens)
    atom.SetNoImplicit(True)
    Chem.SanitizeMol(result)
    return result


def _fraction_from_log10_odds(log10_odds: float) -> float:
    if log10_odds >= 0:
        return 1.0 / (1.0 + 10.0 ** (-log10_odds))
    odds = 10.0 ** log10_odds
    return odds / (1.0 + odds)


def heuristic_prior(
    parent_smiles: str, states: list[str] | tuple[str, ...], pH: float,
    *, pka_shift: float = 0.0,
) -> dict:
    """Construct a two-state HH *heuristic*, or explicitly decline.

    Weights ALWAYS correspond to ``prior_states``, not the supplied enumeration.
    Eligible families: one carboxylic acid with no nitrogen; or one saturated
    aliphatic amine as the molecule's only nitrogen. Additional acidic motifs,
    other elements with unresolved ionization, permanent/extra charges, and
    conjugated/amide nitrogens are excluded. The broad pKa means/SDs come from
    Dimorphite-DL 2.0.2. They are not compound-specific experimental pKa values.
    """
    try:
        ph = float(pH)
        shift = float(pka_shift)
    except (TypeError, ValueError):
        return _rejected("missing_or_invalid_pH_or_pka_shift")
    if not (math.isfinite(ph) and 0 <= ph <= 14 and math.isfinite(shift)):
        return _rejected("missing_or_out_of_range_pH_or_pka_shift")
    molecule = _mol(parent_smiles)
    if len(Chem.GetMolFrags(molecule)) != 1:
        return _rejected("multifragment_parent")
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        return _rejected("radical_species_outside_scope")
    if any(atom.GetAtomicNum() not in {1, 6, 7, 8, 9, 17, 35, 53} for atom in molecule.GetAtoms()):
        return _rejected("unsupported_element_or_other_ionizable_chemistry")
    if molecule.HasSubstructMatch(_BETA_DICARBONYL) or molecule.HasSubstructMatch(_PEROXIDE):
        return _rejected("additional_acidic_or_tautomeric_motif")
    acid_matches = molecule.GetSubstructMatches(_CARBOXYL)
    nitrogens = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 7]
    kind = None
    if len(acid_matches) == 1 and not nitrogens:
        carbon_index, carbonyl_o, target_index = acid_matches[0]
        acid_atom_indices = set(acid_matches[0])
        # Exclude phenols/enols beyond the target acid OH; alcohols remain neutral
        # spectators for this restricted assay-pH heuristic.
        if any(match[-1] not in acid_atom_indices for match in molecule.GetSubstructMatches(_PHENOL_ENOL)):
            return _rejected("additional_phenol_or_enol")
        if any(atom.GetFormalCharge() != 0 for atom in molecule.GetAtoms()
               if atom.GetIdx() != target_index):
            return _rejected("additional_or_permanent_charge")
        target = molecule.GetAtomWithIdx(target_index)
        if target.GetFormalCharge() not in {-1, 0}:
            return _rejected("unsupported_carboxyl_charge")
        carbon = molecule.GetAtomWithIdx(carbon_index)
        aromatic_neighbor = any(atom.GetIsAromatic() for atom in carbon.GetNeighbors())
        rule = "Phenyl_carboxyl" if aromatic_neighbor else "Carboxyl"
        neutral = _set_charge_and_h(molecule, target_index, 0, 1)
        ionized = _set_charge_and_h(molecule, target_index, -1, 0)
        kind = "monoprotic_carboxylic_acid"
    elif not acid_matches and len(nitrogens) == 1:
        target = nitrogens[0]
        target_index = target.GetIdx()
        if target.GetIsAromatic() or target.GetFormalCharge() not in {0, 1}:
            return _rejected("non_aliphatic_or_permanent_nitrogen")
        heavy_neighbors = [atom for atom in target.GetNeighbors() if atom.GetAtomicNum() != 1]
        if not 1 <= len(heavy_neighbors) <= 3:
            return _rejected("quaternary_or_unsupported_amine")
        if any(atom.GetAtomicNum() != 6 or atom.GetIsAromatic()
               or atom.GetHybridization() != Chem.HybridizationType.SP3 for atom in heavy_neighbors):
            return _rejected("conjugated_amide_or_heteroatom_substituted_nitrogen")
        if any(bond.GetBondType() != Chem.BondType.SINGLE for bond in target.GetBonds()):
            return _rejected("non_single_bonded_nitrogen")
        if any(atom.GetFormalCharge() != 0 for atom in molecule.GetAtoms()
               if atom.GetIdx() != target_index):
            return _rejected("additional_or_permanent_charge")
        if molecule.HasSubstructMatch(_PHENOL_ENOL):
            return _rejected("additional_phenol_or_enol")
        neutral = _set_charge_and_h(molecule, target_index, 0, 3 - len(heavy_neighbors))
        ionized = _set_charge_and_h(molecule, target_index, 1, 4 - len(heavy_neighbors))
        rule = "Amines_primary_secondary_tertiary"
        kind = "monoprotic_aliphatic_amine"
    else:
        return _rejected("outside_simple_single_site_subset")
    prior_states = [_smiles(neutral), _smiles(ionized)]
    if prior_states[0] == prior_states[1]:
        return _rejected("conjugate_pair_construction_failed")
    pka_mean, pka_sd = RULES[rule]
    pka = pka_mean + shift
    log10_odds = ph - pka if kind == "monoprotic_carboxylic_acid" else pka - ph
    charged_fraction = _fraction_from_log10_odds(log10_odds)
    observed_states = set(states)
    return {
        "weights": [1.0 - charged_fraction, charged_fraction],
        "prior_states": prior_states,
        "eligible": True, "rule": rule, "pka": pka, "pka_sd": pka_sd,
        "pka_shift": shift, "pH": ph, "kind": kind,
        "weight_kind": "heuristic_prior",
        "reason": "single_site_HH_with_functional_group_mean_pKa;not_measured_population",
        "source": RULE_SOURCE,
        "enumerated_contains_both_prior_states": set(prior_states) <= observed_states,
        "scope_warning": "pKa uncertainty is large; neutral spectator assumptions may fail at extreme pH",
    }
