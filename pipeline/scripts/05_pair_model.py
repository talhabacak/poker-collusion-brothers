"""Stage 5: pair risk model, scored on the population surrogate.

Reports two numbers deliberately kept apart:

* `labelled AP` - 372 targets against the 1,488 disclosed hard negatives. Useful
  only as a regression check; it saturates.
* `population AP` - the same 372 targets ranked against every development pair
  that passes the evaluation co-seating filter. This is the number that tracks
  the official Pair AP, and the one to optimise.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C
from pokercol.cv import (
    EXCLUSION_PATH,
    attach_labels,
    load_pair_table,
    candidate_pairs,
    cleaned_mask,
    population_average_precision,
    table_folds,
)

N_FOLDS = 5
UNKNOWN_WEIGHT = 0.3
POSITIVE_WEIGHT = 10.0  # swept: 0.559 vs 0.546 at 3.0

ID_COLS = {"phase", "table_id", "p1", "p2", "pair_id", "label", "label_status",
           "behavior_family", "y", "fold"}

PARAMS = dict(**C.LGB_REPRO,
    objective="binary",
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=40,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    max_bin=127,
    verbosity=-1,
    num_threads=C.N_THREADS,
    seed=C.SEED,
)
N_ROUNDS = 700


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, t in zip(df.columns, df.dtypes)
            if c not in ID_COLS and t.is_numeric()]


def main() -> None:
    t0 = time.time()
    # Stage 04 features plus bag-of-hands summaries and directed surprise contrasts.
    pairs = load_pair_table()
    labels = pl.read_csv(C.DEV_LABELS)

    dev = attach_labels(candidate_pairs(pairs, "development"), labels)
    print(f"development candidates: {dev.height:,}")
    print(dev.group_by("label_status").len().sort("len", descending=True))

    feats = feature_columns(dev)
    print(f"features: {len(feats)}")

    x = dev.select(feats).to_numpy().astype(np.float32)
    y = dev["y"].to_numpy()
    folds = table_folds(dev["table_id"].to_numpy(), N_FOLDS)

    oof = np.zeros(dev.height, dtype=np.float64)
    for fold in range(N_FOLDS):
        train = folds != fold
        # Unknown pairs are soft negatives, never hard ones.
        target = (y == 1).astype(np.int8)
        weight = np.where(y == 1, POSITIVE_WEIGHT, np.where(y == 0, 1.0, UNKNOWN_WEIGHT))

        booster = lgb.train(
            PARAMS,
            lgb.Dataset(x[train], label=target[train], weight=weight[train]),
            num_boost_round=N_ROUNDS,
        )
        oof[~train] = booster.predict(x[~train])
        print(f"  fold {fold}: trained on {train.sum():,} rows ({time.time() - t0:.0f}s)")

    pop_ap = population_average_precision(y == 1, oof)
    known = y >= 0
    lab_ap = population_average_precision(y[known] == 1, oof[known])

    per_fold = [
        population_average_precision(y[folds == f] == 1, oof[folds == f])
        for f in range(N_FOLDS)
    ]
    print("\n=== pair risk ===")
    if EXCLUSION_PATH.exists():
        keep = cleaned_mask(dev)
        clean_ap = population_average_precision((y == 1)[keep], oof[keep])
        print(f"cleaned AP    : {clean_ap:.5f}   <- calibrated to the leaderboard Pair AP")
    print(f"population AP : {pop_ap:.5f}   (biased low by undisclosed colluders)")
    print(f"labelled AP   : {lab_ap:.5f}   (372 vs 1,488, saturates)")
    print(f"per-fold pop AP: {[round(v, 4) for v in per_fold]}")
    print(f"worst fold     : {min(per_fold):.5f}")

    imp = sorted(zip(feats, booster.feature_importance("gain")), key=lambda kv: -kv[1])
    print("\ntop features by gain:")
    for name, gain in imp[:20]:
        print(f"  {gain:12.0f}  {name}")

    dev.select("phase", "table_id", "p1", "p2", "pair_id", "y", "behavior_family").with_columns(
        pl.Series("risk_oof", oof), pl.Series("fold", folds)
    ).write_parquet(C.ARTIFACTS / "pair_oof.parquet")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
