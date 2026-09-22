"""Generate v64's risk column: the 311-column union recipe with stage 150's
opponent-aware block joined on.

`generate_v27_union` built the union feature set - the production 311 plus the
relational and dyadic tables - and shipped it as v27, which is also the risk
column v59 carries. This adds one table to that frame and changes nothing else,
so v64 minus v59 is the block measured on the 311 feature set, and v64 minus v62
is the feature-set axis with the block held fixed.

The point of the file is to be a *different* candidate rather than another reading
of the same one: only two submissions can be selected and the better of the two
counts, so two strong candidates that differ on both axes are worth more than two
views of one. Whether it earns the slot is decided by stage 155 against v62's
feature set, before this runs.

    python pipeline/scripts/156_generate_v64_union_oppaware.py

REPO NOTE: during the competition this column was then carried into a full submission
by `build_combo.py`, which attached an evidence and a behaviour column. Neither
selected file uses those columns - stage 608 reads the risk column straight from this
script's output - so build_combo is not part of this reproduction repository.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C
from pokercol.cv import attach_labels, candidate_pairs, load_pair_table

pair_mod = importlib.import_module("05_pair_model")

KEY = ["phase", "table_id", "p1", "p2"]
# REPO NOTE: the competition run read an earlier submission purely as a row template
# (pair_id order plus the columns build_combo later replaces). That file is a
# pipeline output, not an input, so this repository falls back to the competition
# sample submission, whose pair_id order is identical - asserted in main().
_LEGACY_BASE = C.SUBMISSIONS / "v20_lgb_xgb_blend_other.csv"
BASE = _LEGACY_BASE if _LEGACY_BASE.exists() else C.SAMPLE_SUB
OUT = C.SUBMISSIONS / "_risk_v64_union_oppaware.csv"
OPPAWARE = "oppaware_pair_nostyle"


def main() -> None:
    t0 = time.time()
    pairs = load_pair_table()
    for t in ("relational_pair", "dyadic_pair", OPPAWARE):
        pairs = pairs.join(pl.read_parquet(C.CACHE / f"{t}.parquet"), on=KEY, how="left")
    dev = attach_labels(candidate_pairs(pairs, "development"), pl.read_csv(C.DEV_LABELS))
    feats = pair_mod.feature_columns(dev)
    added = [f for f in feats if f.startswith("oa_")]
    print(f"features {len(feats)} ({len(added)} from {OPPAWARE}); "
          f"development candidates {dev.height:,}", flush=True)

    base = pl.read_csv(BASE, infer_schema_length=0)
    sample = pl.read_csv(C.SAMPLE_SUB, infer_schema_length=0)
    assert base["pair_id"].to_list() == sample["pair_id"].to_list(), "row template reordered"
    eval_pairs = pl.read_csv(C.EVAL_PAIRS).with_columns(
        pl.min_horizontal("player_1", "player_2").alias("p1"),
        pl.max_horizontal("player_1", "player_2").alias("p2"))
    eval_rows = (base.select("pair_id")
                 .join(eval_pairs.select("pair_id", "p1", "p2"), on="pair_id", how="left")
                 .join(pairs.filter(pl.col("phase") == "evaluation"), on=["p1", "p2"], how="left"))
    assert eval_rows.height == C.N_EVAL_PAIRS and eval_rows["shared_hands"].null_count() == 0
    assert max(eval_rows[c].null_count() for c in added) == 0, "block misses evaluation rows"

    x = dev.select(feats).to_numpy().astype(np.float32)
    y = dev["y"].to_numpy()
    target = (y == 1).astype(np.int8)
    w = np.where(y == 1, 10.0, np.where(y == 0, 1.0, 0.3)).astype(float)
    xe = eval_rows.select(feats).to_numpy().astype(np.float32)

    lgb_params = dict(**C.LGB_REPRO, objective="binary", learning_rate=0.03, num_leaves=31,
                      min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=C.N_THREADS,
                      seed=C.SEED)
    p_lgb = lgb.train(lgb_params, lgb.Dataset(x, label=target, weight=w), 450).predict(xe)
    xgb_params = {"objective": "binary:logistic", "eval_metric": "logloss", "learning_rate": 0.03,
                  "max_depth": 5, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8,
                  "reg_lambda": 3.0, "nthread": C.N_THREADS, "random_state": C.SEED,
                  "tree_method": "hist"}
    p_xgb = xgb.train(xgb_params, xgb.DMatrix(x, label=target, weight=w),
                      400).predict(xgb.DMatrix(xe))
    rank = lambda v: (np.argsort(np.argsort(v, kind="stable"), kind="stable") + 1) / v.size
    risk = rank(0.5 * rank(p_lgb) + 0.5 * rank(p_xgb))

    base.with_columns(pl.Series("risk_score", risk)).write_csv(OUT)
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
