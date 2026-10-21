"""Replay exposure sensitivity into a fresh project subdirectory using its frozen plan.

Original inputs, analysis code, plans and results remain unchanged. Relative
--out paths are resolved from the project root, not the shell working directory.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "runs/round6/exposure_sensitivity"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("runs/round6/exposure_sensitivity_reproduced"),
        help="New directory within the project; must not already exist. Relative paths use the project root.",
    )
    args = parser.parse_args()
    target = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    if not target.is_relative_to(ROOT.resolve()):
        parser.error("Reproduction output must remain within the project directory.")
    if target.exists():
        parser.error("Reproduction output must be a new directory; existing paths are protected.")

    plan_bytes = (FROZEN / "PLAN.json").read_bytes()
    digest_bytes = (FROZEN / "PLAN.sha256").read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != digest_bytes.decode("ascii").strip():
        parser.error("The original frozen plan does not match its recorded SHA256.")

    # Import only after guards, so --help and rejected destinations perform no analysis.
    import analyze_round6_exposure_sensitivity as study

    assert study.ROOT.resolve() == ROOT.resolve()
    target.mkdir(parents=True, exist_ok=False)
    for name, content in [("PLAN.json", plan_bytes), ("PLAN.sha256", digest_bytes)]:
        with (target / name).open("xb") as stream:
            stream.write(content)
    study.OUT = target
    study.REPORT = target / "REPORT.md"
    study.main()


if __name__ == "__main__":
    main()
