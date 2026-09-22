"""Stage 608: the last cross - A's risk under B's behaviour head and the MIL evidence.

Every other cell of the three-component cross has a board reading:

    R_B + B_B + E_B      v55                 0.91748
    R_B + B_B + E_MIL    MIL file            0.91834   (primary)
    R_B + B_B + E_A      x_v55_ourEv         0.91233   -> E_A - E_B = -0.00515
    R_B + B_A + E_MIL    behA                0.91756   -> B_A - B_B = -0.00078
    R_A + B_A + E_A      v64                 0.90543   (second final)

This builds `R_A + B_B + E_MIL`: the MIL file with only `risk_score` replaced
by v64's. Behaviour and evidence are copied as text, so the difference to the
primary is the risk column alone (plus the behaviour term's ranking-by-risk,
which follows the risk column by construction).

Nothing here submits anything.

    python scripts/608_build_hybrid_riskA.py
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pokercol import config as C

s08 = importlib.import_module("08_validate_submission")
MIL = C.SUBMISSIONS / "tarik_v55_mil_evidence.csv"
# REPO NOTE: during the competition the risk column reached this file inside
# `v64_union_oppaware_risk.csv`, a full submission whose evidence and behaviour columns
# this stage throws away. The risk text in it is byte-identical to stage 156's own
# output, verified row by row, so the repository reads 156's output directly and uses
# the competition-era file only if someone puts it here. --risk overrides both.
_COMBINED = C.SUBMISSIONS / "v64_union_oppaware_risk.csv"
_RISK_ONLY = C.SUBMISSIONS / "_risk_v64_union_oppaware.csv"
V64 = _COMBINED if _COMBINED.exists() else _RISK_ONLY
if "--risk" in sys.argv:
    V64 = C.SUBMISSIONS / sys.argv[sys.argv.index("--risk") + 1]
OUT = C.SUBMISSIONS / "hybrid_riskA_behB_evMIL.csv"
REPORT = C.ARTIFACTS / "608_build_hybrid_riskA.json"
EV = [f"evidence_hand_{i}" for i in range(1, 6)]


def main() -> None:
    mil = pl.read_csv(MIL, infer_schema=False)
    v64 = pl.read_csv(V64, infer_schema=False).select("pair_id", pl.col("risk_score").alias("risk_a"))
    assert list(mil.columns) == list(C.SUBMISSION_COLUMNS)
    cand = mil.join(v64, on="pair_id", how="left", maintain_order="left")
    assert cand["risk_a"].null_count() == 0 and cand.height == mil.height
    cand = cand.with_columns(pl.col("risk_a").alias("risk_score")).select(list(C.SUBMISSION_COLUMNS))
    # the six asserts
    assert cand["pair_id"].to_list() == mil["pair_id"].to_list(), "pair_id order changed"
    v64full = pl.read_csv(V64, infer_schema=False)
    assert cand["risk_score"].to_list() == v64full["risk_score"].to_list(), "risk is not v64's text"
    assert cand["predicted_behavior"].to_list() == mil["predicted_behavior"].to_list(), "behaviour changed"
    for c in EV:
        assert cand[c].to_list() == mil[c].to_list(), f"{c} changed"
    n_risk_diff = int((cand["risk_score"] != mil["risk_score"]).sum())
    assert n_risk_diff > 0
    cand.write_csv(OUT)
    rep = s08.validate(OUT)
    ra = cand["risk_score"].cast(pl.Float64).to_numpy()
    rb = mil["risk_score"].cast(pl.Float64).to_numpy()
    top_a, top_b = set(np.argsort(-ra)[:1000]), set(np.argsort(-rb)[:1000])
    out = {"path": str(OUT), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
           "risk_from": str(V64), "behaviour_and_evidence_from": str(MIL),
           "asserts": {"pair_id identical": True, "risk == v64 text": True, "behaviour == MIL text": True,
                       "evidence_1..5 == MIL text": True, "risk rows differing from MIL": n_risk_diff,
                       "behaviour rows differing from MIL": 0, "evidence rows differing from MIL": 0},
           "risk_diversity_vs_MIL": {"spearman": round(float(spearmanr(ra, rb).correlation), 4),
                                     "top1000_overlap": len(top_a & top_b)},
           "validation": rep}
    REPORT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
