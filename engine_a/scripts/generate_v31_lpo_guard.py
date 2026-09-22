"""Generate v31: v24 with the leave-partner-out surprise guard.

Stage 69: on the v24 feature set, max(tail(model), tail(lpo_ps_top5) - 0.3)
is free where all three families are trained (cleaned AP 0.9693 -> 0.9699, no
family down) and recovers an unseen family: leave-one-family-out mixtures at a
5% / 10% unseen share 0.940 -> 0.958 / 0.911 -> 0.943. The production guard it
replaces (ps_top5, c = 0.5) costs 0.001 in both harnesses, matching its -0.0009
on the public board. Behaviour and evidence are v24's, byte for byte.

`--base FILE --out FILE` applies the same guard to another risk file.
`--band LO` (v34) sets c = 0 for pairs whose best known-family top-5 hand
score lies in [LO, 0.95): stage 72 found evaluation's excess of unexplained
suspicious pairs (the hidden family) concentrated there, and on development the
targeted guard is itself a gain (cleaned AP 0.9699 -> 0.9724 at LO = 0.3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C

GUARD_C = float(sys.argv[sys.argv.index("--c") + 1]) if "--c" in sys.argv else 0.3


def arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


BAND = float(sys.argv[sys.argv.index("--band") + 1]) if "--band" in sys.argv else None


def lpo_guard(risk: np.ndarray, signal: np.ndarray, c=GUARD_C) -> np.ndarray:
    n = risk.size
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable") / n
    tail = lambda r: -np.log10(1.0 - r + 1.0 / n)
    guarded = np.maximum(tail(rank(risk)), tail(rank(np.nan_to_num(signal, nan=-1e9))) - c)
    return (rank(guarded) * n + 1) / n


def main() -> None:
    base_path = C.SUBMISSIONS / arg("--base", "v24_v7feats_relational.csv")
    out_path = C.SUBMISSIONS / arg("--out", "v31_v24_lpo_guard.csv")
    base = pl.read_csv(base_path, infer_schema_length=0)
    eval_pairs = pl.read_csv(C.EVAL_PAIRS).with_columns(
        pl.min_horizontal("player_1", "player_2").alias("p1"), pl.max_horizontal("player_1", "player_2").alias("p2"))
    lpo = pl.read_parquet(C.CACHE / "lpo_pair.parquet").filter(pl.col("phase") == "evaluation").select("p1", "p2", "lpo_ps_top5")
    hsf = pl.read_parquet(C.CACHE / "hand_suspicion_family.parquet").filter(pl.col("phase") == "evaluation").select(
        "p1", "p2", pl.max_horizontal("hsdir_top5", "hssoft_top5", "hsiso_top5").alias("hmax"))
    rows = base.select("pair_id").join(eval_pairs.select("pair_id", "p1", "p2"), on="pair_id", how="left").join(
        lpo, on=["p1", "p2"], how="left").join(hsf, on=["p1", "p2"], how="left")
    assert rows.height == C.N_EVAL_PAIRS, rows.height
    print(f"lpo missing for {rows['lpo_ps_top5'].null_count()} pairs")
    risk = base["risk_score"].cast(float).to_numpy()
    c = GUARD_C
    if BAND is not None:
        hmax = rows["hmax"].fill_null(0).to_numpy()
        band = (hmax >= BAND) & (hmax < 0.95)
        print(f"band pairs: {band.sum()}")
        c = np.where(band, 0.0, GUARD_C)
    new = lpo_guard(risk, rows["lpo_ps_top5"].to_numpy().astype(float), c)
    base.with_columns(pl.Series("risk_score", new)).write_csv(out_path)
    top = lambda r, k: set(np.argsort(-r)[:k])
    for k in (500, 1000, 2000, 5000):
        print(f"top-{k} overlap with base: {len(top(new, k) & top(risk, k))}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
