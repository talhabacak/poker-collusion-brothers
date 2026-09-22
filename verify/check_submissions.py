"""Compare rebuilt submissions with the files that were submitted.

    python verify/check_submissions.py                  # both finals
    python verify/check_submissions.py --file X.csv     # one file, by name in submissions/

A rebuilt file either has the recorded sha256 or it does not; when it does not, the
useful question is which of the three scored columns moved, so this prints that too:
how many of the 112,540 rows differ in risk, in behaviour, and in the five evidence
slots, and how far the risk ranking moved. Evidence and behaviour are compared as
text; risk is compared both as text and as a ranking, because the metric reads only
the ranking and two runs can differ in the last digit without differing in score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "submissions"
RECORDED = json.loads((Path(__file__).resolve().parent / "checksums.json").read_text(encoding="utf-8"))
EV = [f"evidence_hand_{i}" for i in range(1, 6)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(name: str, expected: str, reference: Path | None) -> bool:
    path = SUBMISSIONS / name
    if not path.exists():
        print(f"{name}: not built yet")
        return False
    digest = sha256(path)
    if digest == expected:
        print(f"{name}: sha256 matches the submitted file\n  {digest}")
        return True
    print(f"{name}: sha256 DIFFERS\n  rebuilt   {digest}\n  submitted {expected}")
    if reference is None or not reference.exists():
        print("  (no copy of the submitted file here to diff against)")
        return False
    new, old = pl.read_csv(path, infer_schema=False), pl.read_csv(reference, infer_schema=False)
    if new["pair_id"].to_list() != old["pair_id"].to_list():
        print("  pair_id order differs - the files are not comparable row by row")
        return False
    risk_text = int((new["risk_score"] != old["risk_score"]).sum())
    beh = int((new["predicted_behavior"] != old["predicted_behavior"]).sum())
    ev = sum(int((new[c] != old[c]).sum()) for c in EV)
    rho = spearman(new["risk_score"].cast(pl.Float64).to_numpy(),
                   old["risk_score"].cast(pl.Float64).to_numpy())
    print(f"  rows differing: risk text {risk_text}, behaviour {beh}, evidence cells {ev} of 562,700")
    print(f"  risk rank correlation with the submitted file: {rho:.6f}")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="")
    ap.add_argument("--reference-dir", default="", help="directory holding the submitted copies")
    args = ap.parse_args()
    ref_dir = Path(args.reference_dir) if args.reference_dir else SUBMISSIONS / "submitted"
    wanted = {**RECORDED["selected_submissions"], **RECORDED["inputs_to_the_finals"]}
    names = [args.file] if args.file else list(RECORDED["selected_submissions"])
    results = {}
    for name in names:
        if name not in wanted:
            raise SystemExit(f"{name}: no recorded checksum")
        results[name] = compare(name, wanted[name]["sha256"], ref_dir / name)
    built = {k: v for k, v in results.items() if (SUBMISSIONS / k).exists()}
    if not built:
        print("\nnothing rebuilt yet - run run_all.py first")
    elif all(built.values()) and len(built) == len(results):
        print("\nevery rebuilt file carries the checksum of the file that was submitted")
    else:
        print(f"\n{sum(built.values())} of {len(results)} files matched - see above")


if __name__ == "__main__":
    main()
