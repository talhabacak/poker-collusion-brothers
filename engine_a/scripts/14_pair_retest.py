"""Stage 14: re-test pair-model decisions on the cleaned surrogate.

Every earlier pair-model decision was made on the plain population surrogate,
which is biased against exactly the models that rank undisclosed colluders
highly. This freezes the exclusion set once and re-runs the choices that
surrogate rejected or never resolved, all on the same folds and the same rows.

Run with `--freeze` first to write the exclusion set from the reference model.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C
from pokercol.cv import (
    EXCLUSION_PATH,
    attach_labels,
    candidate_pairs,
    cleaned_mask,
    load_pair_table,
    population_average_precision,
    table_folds,
)

N_FOLDS = 5
ID_COLS = {"phase", "table_id", "p1", "p2", "pair_id", "label", "label_status",
           "behavior_family", "y", "fold"}
BASE = dict(**C.LGB_REPRO, objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40,
            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
            max_bin=127, verbosity=-1, num_threads=C.N_THREADS, seed=C.SEED)


def load(extra: tuple[str, ...] = ()) -> pl.DataFrame:
    # Optional tables are skipped when absent: stage 045b calls this before the
    # family hand-suspicion table exists.
    pairs = load_pair_table(required=False)
    for name in extra:
        pairs = pairs.join(pl.read_parquet(C.CACHE / f"{name}.parquet"),
                           on=["phase", "table_id", "p1", "p2"], how="left")
    return attach_labels(candidate_pairs(pairs, "development"), pl.read_csv(C.DEV_LABELS))


def matrix(dev: pl.DataFrame):
    feats = [c for c, t in zip(dev.columns, dev.dtypes) if c not in ID_COLS and t.is_numeric()]
    return dev.select(feats).to_numpy().astype(np.float32), feats


def lgb_oof(x, y, folds, *, params=BASE, rounds=700, pos_w=10.0, unk_w=0.3, pseudo=None):
    oof = np.zeros(len(y))
    for f in range(N_FOLDS):
        train = folds != f
        target = (y == 1).astype(np.int8)
        weight = np.where(y == 1, pos_w, np.where(y == 0, 1.0, unk_w)).astype(float)
        if pseudo is not None:
            conf = train & pseudo[f]
            target = target.copy()
            target[conf] = 1
            weight[conf] = 3.0
        use = train & (weight > 0)
        booster = lgb.train(params, lgb.Dataset(x[use], label=target[use], weight=weight[use]),
                            num_boost_round=rounds)
        oof[~train] = booster.predict(x[~train])
    return oof


def main() -> None:
    t0 = time.time()
    dev = load()
    x, feats = matrix(dev)
    y = dev["y"].to_numpy()
    folds = table_folds(dev["table_id"].to_numpy(), N_FOLDS)

    if "--freeze" in sys.argv or not EXCLUSION_PATH.exists():
        ref = lgb_oof(x, y, folds)
        dev.filter(pl.Series((y == -1) & (ref > 0.5))).select("table_id", "p1", "p2") \
           .write_parquet(EXCLUSION_PATH)
        print(f"froze exclusion set: {int(((y == -1) & (ref > 0.5)).sum())} pairs")

    keep = cleaned_mask(dev)
    results = []

    def report(name, oof, keep_rows=keep, yy=y):
        row = dict(variant=name,
                   cleaned_ap=population_average_precision((yy == 1)[keep_rows], oof[keep_rows]),
                   population_ap=population_average_precision(yy == 1, oof))
        results.append(row)
        print(f"{name:36s} cleaned={row['cleaned_ap']:.4f}  pop={row['population_ap']:.4f}  "
              f"({time.time() - t0:.0f}s)")
        return oof

    base = report("baseline", lgb_oof(x, y, folds))

    for unk in (0.0, 0.05, 0.1, 1.0):
        report(f"unknown weight {unk}", lgb_oof(x, y, folds, unk_w=unk))
    for pw in (3.0, 30.0):
        report(f"positive weight {pw}", lgb_oof(x, y, folds, pos_w=pw))
    report("rounds 1500", lgb_oof(x, y, folds, rounds=1500))
    report("leaves 63", lgb_oof(x, y, folds, params={**BASE, "num_leaves": 63}))
    report("leaves 15", lgb_oof(x, y, folds, params={**BASE, "num_leaves": 15}))

    # XGBoost and a rank blend, previously judged on the biased surrogate.
    xo = np.zeros(len(y))
    w = np.where(y == 1, 10.0, np.where(y == 0, 1.0, 0.3))
    xp = dict(objective="binary:logistic", eta=0.03, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist",
              max_bin=127, nthread=C.N_THREADS, seed=C.SEED + 3)
    for f in range(N_FOLDS):
        tr = folds != f
        b = xgb.train(xp, xgb.DMatrix(x[tr], label=(y[tr] == 1).astype(int), weight=w[tr]), 700)
        xo[~tr] = b.predict(xgb.DMatrix(x[~tr]))
    report("xgboost", xo)
    rank = lambda s: np.argsort(np.argsort(s)) / s.size
    report("rank blend lgb+xgb", (rank(base) + rank(xo)) / 2)

    # Phase contrast features (stage 4d), previously judged on the biased surrogate.
    dev_pc = load(("phase_contrast",))
    x_pc, _ = matrix(dev_pc)
    report("+ phase contrast features", lgb_oof(x_pc, y, folds))

    (C.ARTIFACTS / "pair_retest.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
