"""Replay domain, case selection and diagnostic joins; no model loading."""
from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', choices=['domain', 'cases', 'join'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--input-root', type=Path)
    a = parser.parse_args()
    out = a.out.resolve()
    out.relative_to(ROOT.resolve())
    if out.exists():
        raise FileExistsError('Fresh destination required: ' + str(out))
    out.mkdir(parents=True)
    if a.analysis == 'domain':
        m = importlib.import_module('analyze_round5_external_domain')
        m.OUT = out
        m.REPORT = out / 'REPORT.md'
        sys.argv = [m.__file__, '--config', str(ROOT / 'configs/round5_external.json'), '--with-priors']
        m.main()
    elif a.analysis == 'cases':
        m = importlib.import_module('analyze_round5_cases')
        m.OUT = out
        m.select_cases(SimpleNamespace(stage='r5_final', predictions=ROOT / 'runs/round5/combined_logd/predictions.csv', completion_manifest=ROOT / 'runs/round5/combined_logd/READY.json'))
    else:
        if a.input_root is None:
            parser.error('--input-root is required for join')
        inputs = a.input_root.resolve()
        inputs.relative_to(ROOT.resolve())
        text = (ROOT / 'scripts/join_round6_diagnostics.py').read_text(encoding='utf-8')
        output_token = "OUT=ROOT/'runs/round6/integrated_diagnostics'"
        assert text.count(output_token) == 1, 'Join output redirect not uniquely matched'
        text = text.replace(output_token, 'OUT=Path(' + repr(str(out)) + ')')
        for directory in ['robustness', 'prior_diagnostics']:
            token = "ROOT/'runs/round6/" + directory + '/'
            assert text.count(token) == 2, ('Join input redirect mismatch', directory)
            text = text.replace(token, 'Path(' + repr(str(inputs / directory)) + ")/'")
        assert "ROOT/'runs/round6/" not in text, 'Frozen output/input directory remains in join execution'
        exec(compile(text, str(ROOT / 'scripts/join_round6_diagnostics.py'), 'exec'), {'__file__': str(ROOT / 'scripts/join_round6_diagnostics.py'), '__name__': '__main__'})

if __name__ == '__main__':
    main()
