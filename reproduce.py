"""Self-contained numerical reproduction from frozen predictions; no fitting or neural inference."""


from __future__ import annotations


import argparse


import ast


from datetime import datetime, timezone


import hashlib


import importlib.metadata


import json


import os
import re


from pathlib import Path


import shutil


import subprocess


import sys


import time


ROOT=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
    return h.hexdigest()


def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def verify_bundle():
    manifest=ROOT/'package_manifest.json'
    if not manifest.exists():raise FileNotFoundError('package_manifest.json missing')
    data=json.loads(manifest.read_text(encoding='utf-8'))
    for item in data['files']:
        p=ROOT/item['path']
        if not p.is_file() or sha(p)!=item['sha256']:raise AssertionError('Package integrity failure: '+item['path'])
    violations=[]
    for folder in ['src','scripts','tests','data']:
        for p in (ROOT/folder).rglob('*.py'):
            text=p.read_text(encoding='utf-8-sig');ast.parse(text)
            for word in ['matplotlib','seaborn','MolsToGridImage','MolToImage','rdMolDraw2D','savefig','DistributionPlotter','from PIL','import Draw']:
                if word in text:violations.append(str(p.relative_to(ROOT))+': '+word)
    assert not violations,violations
    return dict(files_verified=len(data['files']),no_plot_code=True)


def environment():
    result=dict(python=sys.version,executable=sys.executable,platform=sys.platform)
    for p in ['numpy','pandas','scipy','rdkit']:
        try:result[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:result[p]=None
    return result


def run_command(argv,cwd,logdir,jobs):
    logdir.mkdir(parents=True,exist_ok=True)
    label=f'{len(jobs)+1:03d}'
    env=os.environ.copy();env['PYTHONPATH']=str(cwd/'src');env['PYTHONIOENCODING']='utf-8';env['PYTHONDONTWRITEBYTECODE']='1'
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[name]='4'
    start=time.time()
    command=[sys.executable,*map(str,argv)]
    print('RUN',label,' '.join(map(str,argv)),flush=True)
    with (logdir/f'{label}.stdout.txt').open('wb') as out,(logdir/f'{label}.stderr.txt').open('wb') as err:
        p=subprocess.Popen(command,cwd=cwd,env=env,stdout=out,stderr=err)
        code=p.wait()
    job=dict(argv=command,cwd=str(cwd),pid=p.pid,started_utc=datetime.fromtimestamp(start,timezone.utc).isoformat(),finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.time()-start,returncode=code,stdout=f'logs/{label}.stdout.txt',stderr=f'logs/{label}.stderr.txt')
    jobs.append(job);save(logdir.parent/'execution.json',dict(jobs=jobs))
    if code:
        print((logdir/f'{label}.stderr.txt').read_text(encoding='utf-8',errors='replace')[-5000:],file=sys.stderr)
        raise RuntimeError(f'Job {label} failed; see {logdir}')


def compare_csv(original,new,atol=1e-10):
    import numpy as np
    import pandas as pd
    a=pd.read_csv(original,float_precision='round_trip');b=pd.read_csv(new,float_precision='round_trip')
    assert list(a.columns)==list(b.columns),(str(original),'columns')
    assert a.shape==b.shape,(str(original),a.shape,b.shape)
    maximum=0.
    for column in a:
        if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
            x=a[column].to_numpy(float);y=b[column].to_numpy(float)
            np.testing.assert_allclose(x,y,rtol=0,atol=atol,equal_nan=True,err_msg=str(original)+':'+column)
            good=np.isfinite(x)&np.isfinite(y)
            if good.any():maximum=max(maximum,float(np.max(np.abs(x[good]-y[good]))))
        else:
            assert a[column].fillna('<NA>').astype(str).equals(b[column].fillna('<NA>').astype(str)),(str(original),column)
    return dict(file=str(original.relative_to(ROOT)),rows=len(a),columns=len(a.columns),max_abs_numeric_difference=maximum,absolute_tolerance=atol,passed=True)


def compare_npz(original,new,atol=1e-10):
    import numpy as np
    maximum=0.
    with np.load(original,allow_pickle=False) as a,np.load(new,allow_pickle=False) as b:
        assert set(a.files)==set(b.files),(str(original),'npz keys')
        for key in a.files:
            if np.issubdtype(a[key].dtype,np.number):
                np.testing.assert_allclose(a[key],b[key],rtol=0,atol=atol,equal_nan=True)
                good=np.isfinite(a[key])&np.isfinite(b[key])
                if good.any():maximum=max(maximum,float(np.max(np.abs(a[key][good]-b[key][good]))))
            else:np.testing.assert_array_equal(a[key],b[key])
    return dict(file=str(original.relative_to(ROOT)),max_abs_numeric_difference=maximum,absolute_tolerance=atol,passed=True)


def statistics(destination,jobs):
    log=destination/'logs'
    robust=destination/'robustness';prior=destination/'prior_diagnostics';exposure=destination/'exposure_sensitivity'
    commands=[['scripts/reproduce_round6_robustness.py','--out',robust],
        ['scripts/reproduce_round6_exposure_sensitivity.py','--out',exposure],
        ['scripts/analyze_round6_prior_diagnostics.py','--plan',ROOT/'runs/round6/prior_diagnostics/plan.json','--out',prior],
        ['scripts/replay_auxiliary.py','domain','--out',destination/'external_domain'],
        ['scripts/replay_auxiliary.py','cases','--out',destination/'cases'],
        ['scripts/audit_unipka_experimental_pka.py','--prediction-dir',ROOT/'runs/round3_unipka_full','--out',destination/'pka'],
        ['scripts/replay_auxiliary.py','join','--out',destination/'integrated_diagnostics','--input-root',destination]]
    for argv in commands:run_command(argv,ROOT,log,jobs)
    pairs=[('runs/round6/robustness',robust),('runs/round6/exposure_sensitivity',exposure),('runs/round6/prior_diagnostics',prior),('runs/round5_external_domain',destination/'external_domain'),('runs/round5_cases/r5_final',destination/'cases/r5_final'),('runs/round3_pka_audit_full',destination/'pka'),('runs/round6/integrated_diagnostics',destination/'integrated_diagnostics')]
    comparisons=[]
    for old,new in pairs:
        expected_files = sorted(p for p in (ROOT/old).iterdir() if p.suffix in {'.csv', '.npz'})
        actual_files = sorted(p for p in new.iterdir() if p.suffix in {'.csv', '.npz'})
        assert {p.name for p in actual_files} == {p.name for p in expected_files}, (old, 'Missing or unexpected numerical outputs')
        for expected in expected_files:
            actual = new / expected.name
            comparisons.append(compare_csv(expected, actual) if expected.suffix == '.csv' else compare_npz(expected, actual))
    assert len(comparisons) == 46, ('Expected every one of 46 numerical artifacts', len(comparisons))
    effects=__import__('pandas').read_csv(robust/'paired_effects.csv')
    core=effects[(effects.scope=='primary_all')&(effects.resampling_unit=='compound')]
    results=dict(complete=True,mode='statistics',trained_models=0,upstream_neural_inference=0,comparison_files=len(comparisons),comparisons=comparisons,core_contrasts=core[['cohort','candidate','reference','delta_MAE','ci_low','ci_high']].to_dict('records'),environment=environment(),notes=['Numeric tables and bootstrap arrays compared; historical timestamps/source digests are not rewritten.', 'Historical runtime measurements are retained; newly measured runtime cannot equal historical wall-clock times.'])
    save(destination/'verification.json',results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['stats', 'verify'], default='stats')
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = parser.parse_args()
    integrity = verify_bundle()
    print(json.dumps(integrity), flush=True)
    if args.mode == 'verify':
        return
    if Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'} or not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('run-id must contain only letters, digits, underscore or hyphen')
    destination = ROOT / 'reproduced' / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    jobs = []
    save(destination / 'environment.json', dict(environment(), integrity=integrity))
    try:
        result = statistics(destination, jobs)
        print(json.dumps(dict(complete=result['complete'], comparison_files=result['comparison_files'], output=str(destination)), indent=2), flush=True)
    except Exception as error:
        save(destination / 'FAILURE.json', dict(complete=False, error=repr(error), executed_jobs=jobs))
        raise

if __name__ == '__main__':
    main()
