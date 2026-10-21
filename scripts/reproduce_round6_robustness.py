"""Replay the frozen robustness implementation into a fresh project subdirectory."""
from pathlib import Path
import argparse
import analyze_round6_robustness as study

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/round6/robustness_reproduced'))
    args=parser.parse_args()
    target=(study.ROOT/args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    target.relative_to(study.ROOT.resolve())
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('Reproduction destination must be new or empty.')
    study.OUT=target
    study.REPORT=target/'REPORT.md'
    study.self_test()
    study.make_plan()
    study.run()
