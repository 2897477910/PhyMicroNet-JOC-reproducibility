"""Join outcome-flagged diagnostics to already computed chemistry; no exclusions."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors,rdMolDescriptors

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/round6/integrated_diagnostics'
OUT.mkdir(exist_ok=True)
inp=[ROOT/'runs/round6/robustness/extreme_residuals.csv',
     ROOT/'runs/round6/robustness/influence.csv',
     ROOT/'runs/round6/prior_diagnostics/per_record.csv',
     ROOT/'runs/round6/prior_diagnostics/per_compound.csv']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
extreme=pd.read_csv(inp[0]);prior=pd.read_csv(inp[2])
mapping={'az_logd':'logd','az_solubility':'solubility','asap_external':'external'}
prior.cohort=prior.cohort.map(mapping);assert prior.cohort.notna().all()
cols=['cohort','raw_row','compound_id','parent_smiles','full_entropy','dominant_occupancy',
      'effective_state_number','full_state_count','parent_is_dominant','full_parent_is_dominant',
      'neutral_fraction','selected_retained_mass','selected_state_count','conformer_fallback']
joined=extreme.merge(prior[cols],on=['cohort','raw_row'],validate='one_to_one',how='left',suffixes=('','__prior'))
for col in ['compound_id','parent_smiles']:
    assert joined[col].equals(joined[col+'__prior'])
    joined=joined.drop(columns=[col+'__prior'])
assert len(joined)==len(extreme) and joined.full_entropy.notna().all()
for row in joined.itertuples():
    mol=Chem.MolFromSmiles(row.parent_smiles);assert mol is not None
    values={'molecular_weight':Descriptors.MolWt(mol),'TPSA':rdMolDescriptors.CalcTPSA(mol),
            'formal_charge':Chem.GetFormalCharge(mol)}
    for col,value in values.items():
        previous=joined.at[row.Index,col]
        if pd.notna(previous):assert abs(float(previous)-value)<1e-6,(col,previous,value)
        joined.at[row.Index,col]=value
joined.to_csv(OUT/'extreme_cases_enriched.csv',index=False,float_format='%.17g')
inf=pd.read_csv(inp[1]);inf=inf[(inf.cohort=='external')&(inf.scope=='primary_all')&
                            (inf.unit=='compound')&(inf.candidate=='prior')&(inf.reference=='uniform')]
pc=pd.read_csv(inp[3]);pc=pc[pc.cohort=='asap_external']
enriched=inf.merge(pc,left_on='group_id',right_on='compound_id',how='left',validate='one_to_one',suffixes=('','__diagnostic'))
assert len(enriched)==413 and enriched.full_entropy.notna().all()
enriched.to_csv(OUT/'external_weighting_influence_enriched.csv',index=False,float_format='%.17g')
result=dict(complete=True,extreme_records=len(joined),counts=joined.groupby('cohort').size().to_dict(),
            external_compound_influence_rows=len(enriched),source_identity_and_parent_matches=True,
            parent_descriptors_recomputed=True,primary_records_excluded=0,
            scope='Descriptive joins only; outcome-based extreme flags are not subgroups selected for primary inference.',
            input_hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in inp},
            script_sha256=sha(Path(__file__)),
            outputs={p.name:sha(p) for p in OUT.glob('*.csv')})
(OUT/'READY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['complete','extreme_records','counts','external_compound_influence_rows']}))
