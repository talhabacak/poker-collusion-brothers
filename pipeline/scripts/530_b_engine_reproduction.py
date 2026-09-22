"""Stage 530: is the teammate's v55 evidence engine still here and still reading what it read?

Everything from stage 532 onward is measured inside that engine, so before any of it exists the
engine has to be shown to be the same object stage 448 measured. Two readings are asserted, not
quoted: the v55 evidence side (`USE_OAH=1 USE_OCH=1`, six seeds) at **0.66085** and the v45-chain
baseline (`USE_OAH=0 USE_OCH=0`) at **0.65164**, both rerun under their `05b_evidence_reranker.py`
in CV-only mode and both read from the out-of-fold dumps the reruns write. The patched copy
`05b_mil.py` (stage 533's harness) is run once more with none of its hooks set and has to land on
the same number, or the harness is a different code path and nothing measured in it counts.

Then note 003's two measurements, on B's engine rather than A's:

*Within-pair AUC among B's own top 20.* Evidence MAP@5 is a within-pair ordering problem; A's
production engine reads 0.9116 there and a simulation of MAP@5 at that AUC reproduces the engine
to within 0.009. The same statistic for B's v55 reranker says whether B has room A does not.

*A direct discriminator on B's own hand-feature table.* The reranker's training frame (372 pairs
x their top-50 candidates, every column the v55 reranker sees) is fed to a plain binary
classifier, cross-fit by B's `tidx % 5`, and its within-pair AUC on the same top-20 shortlist is
compared with the reranker's. If it beats the reranker, B has signal on the table and the fix is
the objective, not the features; if it does not, better fitting of these columns closes nothing.

    TARIK_INTERIM=<repro>/data/interim python scripts/530_b_engine_reproduction.py

Writes artifacts/530_b_engine_reproduction.json.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C
from pokercol.metrics import evidence_map5

s474 = importlib.import_module("474_within_pair_ceiling")

INTERIM = Path(os.environ.get("TARIK_INTERIM", ""))
REPORT = C.ARTIFACTS / "530_b_engine_reproduction.json"
RECORDED = {"v55ev": 0.66085, "basenew": 0.65164}     # stage 448, same harness
TOL = 1e-5
TOP = 20
N_FOLDS = 5
STAGE1_COLS = ["s_ev", "s_dt", "s_sp", "s_ci", "s_fam_max", "r", "s_rel_max", "s_minus_5th",
               "s1b", "r1b", "s1b_rel_max", "s1b_minus_5th",
               "prior_gt0.1", "prior_gt0.2", "prior_gt0.3", "after_gt0.1", "after_gt0.2", "after_gt0.3",
               "prior_s_sum", "after_s_sum", "mass_before", "cand_chrono"]


def lists_from_dump(dump: pl.DataFrame, hands: pl.DataFrame) -> dict:
    """Top five per pair, B's own tie-break (s2 desc, hidx asc)."""
    top = (dump.join(hands, on="hidx", how="left")
           .sort(["pair_id", "s2", "hidx"], descending=[False, True, False])
           .group_by("pair_id", maintain_order=True).head(5))
    return {pid: hs for pid, hs in top.group_by("pair_id", maintain_order=True)
            .agg(pl.col("hand_id")).iter_rows()}


def within_pair_auc(dump: pl.DataFrame, score: str, shortlist: str, top: int) -> dict:
    """Mean AUC of `score` against gold inside each pair's top-`top` by `shortlist`."""
    d = dump.with_columns(pl.col(shortlist).rank("ordinal", descending=True).over("pair_id").alias("_r"))
    d = d.filter(pl.col("_r") <= top)
    aucs = [roc_auc_score(g["y"].to_numpy(), g[score].to_numpy())
            for _, g in d.group_by("pair_id") if 0 < g["y"].sum() < g.height]
    gold_in = d.group_by("pair_id").agg(pl.col("y").sum().alias("g")).join(
        dump.group_by("pair_id").agg(pl.col("y").sum().alias("t")), on="pair_id")
    return {"mean within-pair AUC": round(float(np.mean(aucs)), 5), "pairs": len(aucs),
            "gold recall inside the shortlist": round(float(gold_in["g"].sum() / gold_in["t"].sum()), 4)}


def discriminator(frame: pl.DataFrame, feats: list[str], seed: int = 42) -> np.ndarray:
    """A plain binary LightGBM on the reranker's own training frame, out of fold by B's folds."""
    x = frame.select(feats).to_numpy().astype(np.float32)
    y = frame["y"].to_numpy()
    fold = frame["fold"].to_numpy()
    oof = np.zeros(frame.height)
    params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=30,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  verbose=-1, num_threads=16, seed=seed, deterministic=True, force_col_wise=True)
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        m = lgb.train(params, lgb.Dataset(x[tr], y[tr]), 400)
        oof[te] = m.predict(x[te])
    return oof


def main() -> None:
    t0 = time.time()
    if not INTERIM.exists():
        raise SystemExit("set TARIK_INTERIM to the reproduction's data/interim directory")
    out = {"interim": str(INTERIM), "tree": {}}

    # ------------------------------------------------------------ 1. the tree
    expected = {"hands.parquet": None, "players.parquet": None, "actions_feat.parquet": C.N_ACTIONS,
                "actions_scored.parquet": C.N_ACTIONS, "seats_feat.parquet": C.N_SEATS,
                "oppaware_hand.parquet": None, "oppcond_hand.parquet": None}
    for name, n in expected.items():
        p = INTERIM / name
        rows = pl.scan_parquet(p).select(pl.len()).collect().item() if p.exists() else None
        out["tree"][name] = {"exists": p.exists(), "rows": rows}
        assert p.exists(), f"{p} missing"
        if n is not None:
            assert rows == n, f"{name} has {rows} rows, expected {n}"
    for d, n in (("pairhand", 20), ("handscores", 20), ("stage1b", 40), ("actmodel", 21),
                 ("relational", 10), ("extra", 10)):
        files = sorted((INTERIM / d).glob("*.parquet"))
        out["tree"][d] = {"files": len(files)}
        assert len(files) == n, f"{d}: {len(files)} files, expected {n}"
    print(f"tree intact ({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------- 2. the two readings, asserted
    hands = pl.read_parquet(INTERIM / "hands.parquet").select("hidx", "hand_id", "tidx", "table_id")
    gold_table = pl.read_csv(C.DEV_EVIDENCE).select("pair_id", "hand_id")
    gold = {pid: set(hs) for pid, hs in gold_table.group_by("pair_id").agg(pl.col("hand_id")).iter_rows()}
    out["reproduction"] = {}
    dumps = {}
    for tag, rec in RECORDED.items():
        d = pl.read_parquet(INTERIM / f"rerank_oof_{tag}_530.parquet")
        dumps[tag] = d
        lists = lists_from_dump(d, hands.select("hidx", "hand_id"))
        assert set(lists) == set(gold), "the dump does not cover the 372 target pairs"
        got = evidence_map5(lists, gold)
        prev = pl.read_parquet(INTERIM / f"rerank_oof_{tag}.parquet")   # stage 448's dump, same command
        j = d.select("pair_id", "hidx", "s2").join(prev.select("pair_id", "hidx", pl.col("s2").alias("s2_448")),
                                                   on=["pair_id", "hidx"])
        out["reproduction"][tag] = {
            "recorded (stage 448)": rec, "reread now": round(got, 5),
            "rows": d.height, "pairs": d["pair_id"].n_unique(),
            "max |s2 - s2 of stage 448's dump|": float(np.abs(j["s2"] - j["s2_448"]).max()),
            "status": "reproduced" if abs(got - rec) < TOL else "NOT REPRODUCED"}
        print(f"  {tag}: {got:.5f} (recorded {rec}) -> {out['reproduction'][tag]['status']}; "
              f"bit-agreement with 448's dump: max |ds2| = {out['reproduction'][tag]['max |s2 - s2 of stage 448\'s dump|']}",
              flush=True)
        assert abs(got - rec) < TOL, f"{tag} does not reproduce: {got:.5f} vs {rec}"
    # the patched harness, no hooks set, must be the same code path
    d = pl.read_parquet(INTERIM / "rerank_oof_v55ev_harness.parquet")
    got = evidence_map5(lists_from_dump(d, hands.select("hidx", "hand_id")), gold)
    j = d.select("pair_id", "hidx", "s2").join(dumps["v55ev"].select("pair_id", "hidx", pl.col("s2").alias("s2_ref")),
                                               on=["pair_id", "hidx"])
    out["reproduction"]["05b_mil.py with no hooks"] = {
        "reread": round(got, 5), "max |s2 - original 05b|": float(np.abs(j["s2"] - j["s2_ref"]).max()),
        "status": "same code path" if abs(got - RECORDED["v55ev"]) < TOL else "DIFFERENT"}
    assert abs(got - RECORDED["v55ev"]) < TOL, "the patched harness is not the same code path"
    print(f"  05b_mil.py (no hooks): {got:.5f}, max |ds2| against the original {out['reproduction']['05b_mil.py with no hooks']['max |s2 - original 05b|']}", flush=True)

    # ------------------------------------------ 3. within-pair AUC among B's top 20
    v55 = dumps["v55ev"]
    out["within_pair"] = {
        "v55 reranker, its own top 20": within_pair_auc(v55, "s2", "s2", TOP),
        "v55 reranker, its own top 50 (whole candidate frame)": within_pair_auc(v55, "s2", "s2", 50),
        "v45-chain reranker, its own top 20": within_pair_auc(dumps["basenew"], "s2", "s2", TOP)}
    cur = out["within_pair"]["v55 reranker, its own top 20"]["mean within-pair AUC"]
    sim = s474.map5_at_auc(cur)
    need = {t: s474.auc_for(t, cur) for t in (0.70, 0.7677, 0.80, 0.90)}
    out["within_pair"]["simulated MAP@5 at B's AUC (five gold among twenty)"] = round(sim, 4)
    out["within_pair"]["engine reads"] = RECORDED["v55ev"]
    out["within_pair"]["AUC required on the same shortlist"] = {str(k): round(v, 4) for k, v in need.items()}
    out["within_pair"]["A's engine for reference (stage 474)"] = {"within-pair AUC": 0.9116, "simulated": 0.6807, "engine": 0.68921}
    print(f"  B within-pair AUC (top 20) {cur:.4f}; simulated MAP@5 {sim:.4f} vs engine {RECORDED['v55ev']}; "
          f"required {json.dumps({k: round(v, 4) for k, v in need.items()})} ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------- 4. direct discriminator on B's frame
    frame = pl.read_parquet(INTERIM / "533_frame_v55.parquet")
    assert frame.height == v55.height, "the frame dump and the OOF dump are different objects"
    skip = {"pair_id", "hidx", "p", "q", "tidx", "fold", "y", "evidence_rank", "t_order"}
    feats = [c for c in frame.columns if c not in skip]
    key = frame.select("pair_id", "hidx", "y", "fold")
    key = key.join(v55.select("pair_id", "hidx", "s2"), on=["pair_id", "hidx"], how="left")
    assert key["s2"].null_count() == 0
    arms = {"all v55 reranker inputs": feats,
            "without the stage-1 / stage-1b scores and their ranks": [c for c in feats if c not in STAGE1_COLS]}
    out["discriminator"] = {"rows": frame.height, "pairs": frame["pair_id"].n_unique(),
                            "reranker (lambdarank, 6-seed bag) on its own top 20": cur}
    for name, cols in arms.items():
        oof = discriminator(frame, cols)
        kd = key.with_columns(pl.Series("disc", oof))
        lists = lists_from_dump(kd.select("pair_id", "hidx", pl.col("disc").alias("s2")), hands.select("hidx", "hand_id"))
        out["discriminator"][name] = {
            "columns": len(cols),
            "within-pair AUC on the reranker's top 20": within_pair_auc(kd, "disc", "s2", TOP)["mean within-pair AUC"],
            "within-pair AUC on its own top 20": within_pair_auc(kd, "disc", "disc", TOP)["mean within-pair AUC"],
            "MAP@5 as a ranker": round(evidence_map5(lists, gold), 5)}
        print(f"  discriminator [{name}]: {json.dumps(out['discriminator'][name])} ({time.time() - t0:.0f}s)", flush=True)

    # gold against the pair's other top-20 candidates, within-pair z, for the record
    d = key.join(frame.drop("y", "fold"), on=["pair_id", "hidx"]).with_columns(
        pl.col("s2").rank("ordinal", descending=True).over("pair_id").alias("_r")).filter(pl.col("_r") <= TOP)
    y = d["y"].to_numpy()
    rows = []
    for c in feats:
        z = d.select("pair_id", c).with_columns(
            ((pl.col(c) - pl.col(c).mean().over("pair_id")) / (pl.col(c).std().over("pair_id") + 1e-9)).alias("z"))["z"].to_numpy()
        a, b = z[y == 1], z[y == 0]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 50 or len(b) < 50:
            continue
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        rows.append({"feature": c, "gold_minus_other_z": round(float(diff), 4), "t": round(float(diff / se) if se > 0 else 0.0, 1)})
    rows.sort(key=lambda r: -abs(r["t"]))
    out["separation (gold vs the pair's other top-20 candidates, within-pair z)"] = rows[:25]
    out["runtime_s"] = round(time.time() - t0, 1)
    REPORT.write_text(json.dumps(out, indent=2))
    print(f"written to {REPORT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
