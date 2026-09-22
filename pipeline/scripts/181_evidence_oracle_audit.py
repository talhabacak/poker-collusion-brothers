"""Stage 181: where evidence retrieval loses its score.

Separates two failure modes with different fixes:

* coverage - planted hands are not even near the top (low recall@50): the hand
  representation is missing them, and ranker changes cannot help;
* ordering - planted hands are near the top but not in the first five: the
  ranker needs better within-pair discrimination.

Also splits the ranker into a mechanism-only model (everything except the
episode-position features) and a position-only model, and tests
`rank(mechanism) + lambda * rank(position)`, which keeps the position prior as an
explicit, tunable term instead of letting it hide inside one model.

Pooled lambdarank, table-grouped OOF. Writes artifacts/evidence_oracle_audit.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C
from pokercol.cv import table_folds

import importlib
exp = importlib.import_module("11_evidence_experiments")
ev_mod = importlib.import_module("06_evidence_model")


def oof_scores(df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    x = df.select(feats).to_numpy().astype(np.float32)
    folds = table_folds(df["table_id"].to_numpy(), 5)
    out = np.zeros(df.height)
    for f in range(5):
        tr = folds != f
        out[~tr] = exp.fit_ranker(df, feats, tr).predict(x[~tr])
    return out


def recall_at(df: pl.DataFrame, score: np.ndarray, ks=(5, 10, 20, 50, 100)) -> dict:
    d = df.select("pair_id", "behavior_family", "is_evidence").with_columns(
        pl.Series("s", score)).with_columns(
        pl.col("s").rank("ordinal", descending=True).over("pair_id").alias("r"))
    ev = d.filter(pl.col("is_evidence") == 1)
    out = {"all": {f"recall@{k}": float((ev["r"] <= k).mean()) for k in ks}}
    for fam in C.FAMILIES:
        e = ev.filter(pl.col("behavior_family") == fam)
        out[fam] = {f"recall@{k}": float((e["r"] <= k).mean()) for k in ks}
    return out


def main() -> None:
    df, feats = exp.build()
    evidence = pl.read_csv(C.DEV_EVIDENCE).select("pair_id", "hand_id").unique()
    position = [f for f in feats if f in ev_mod.EPISODE_POSITION]
    mechanism = [f for f in feats if f not in ev_mod.EPISODE_POSITION]

    full = oof_scores(df, feats)
    mech = oof_scores(df, mechanism)
    pos = oof_scores(df, position)
    wr = lambda s: exp.within_pair_rank(df, s)

    report = {
        "map5": {"full": exp.score_map5(df, full, evidence),
                 "mechanism_only": exp.score_map5(df, mech, evidence),
                 "position_only": exp.score_map5(df, pos, evidence)},
        "recall_full": recall_at(df, full),
        "recall_mechanism_only": recall_at(df, mech),
        "lambda_blend": {},
    }
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        report["lambda_blend"][str(lam)] = exp.score_map5(df, wr(mech) + lam * wr(pos), evidence)

    OUT = C.ARTIFACTS / "evidence_oracle_audit.json"
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
