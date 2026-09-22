"""Stage 4b: score every shared hand for "this looks like a planted event",
then summarise each pair by its most incriminating hands.

A colluding pair plays ~120 hands together and cheats in about five of them.
Any pair-level average is therefore ~96% noise, which is why rate features
plateau. The fix is to treat the pair as a bag of hands, score the hands, and
let the pair inherit its extremes - the classic multiple-instance framing, and
the reason `development_evidence.csv` is useful far beyond the 20% it is scored
on: it is 1,817 labelled examples of what a collusion event looks like at the
hand level.

Leakage control: a development pair's aggregates always come from a model
trained on other tables. Evaluation pairs are scored by a model fitted on all
development tables.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C
from pokercol.cv import table_folds

import importlib
ev_mod = importlib.import_module("06_evidence_model")

N_FOLDS = 5
TABLES_PER_BATCH = 25
OUT = C.CACHE / "hand_suspicion.parquet"

PARAMS = dict(**C.LGB_REPRO,
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    max_bin=127,
    verbosity=-1,
    num_threads=C.N_THREADS,
    seed=C.SEED,
)
N_ROUNDS = 500
EXTRA_UNKNOWN_PAIRS = int(__import__("os").environ.get("HS_EXTRA_UNKNOWN", "3000"))


def labelled_frame() -> pl.DataFrame:
    """Planted evidence hands as positives, hard-negative pairs' hands as negatives.

    Hands from positive pairs that were *not* planted are dropped rather than
    used as negatives: they may well be collusive too, and calling them clean
    would teach the model to suppress exactly the signal it is looking for.
    """
    labels = pl.read_csv(C.DEV_LABELS).with_columns(
        pl.min_horizontal("player_1", "player_2").alias("p1"),
        pl.max_horizontal("player_1", "player_2").alias("p2"),
    )
    pos_pairs = labels.filter(pl.col("label") == 1).select("pair_id", "p1", "p2")
    neg_pairs = labels.filter(pl.col("label") == 0).select("pair_id", "p1", "p2")
    evidence = pl.read_csv(C.DEV_EVIDENCE).select("pair_id", "hand_id").unique()

    pos = ev_mod.load_pair_hands(pos_pairs, "development").join(
        evidence.with_columns(pl.lit(1, dtype=pl.Int8).alias("target")),
        on=["pair_id", "hand_id"], how="inner",
    )
    neg = ev_mod.load_pair_hands(neg_pairs, "development").with_columns(
        pl.lit(0, dtype=pl.Int8).alias("target")
    )
    frames = [pos, neg]
    if EXTRA_UNKNOWN_PAIRS:
        frames.append(unknown_negatives(labels, pos_pairs))
    # Fixed row order: bagging samples by position.
    return pl.concat(frames, how="diagonal").sort("pair_id", "hand_id")


def unknown_negatives(labels: pl.DataFrame, pos_pairs: pl.DataFrame) -> pl.DataFrame:
    """Hands from a random sample of ordinary unlabelled pairs, as negatives.

    The 1,488 confirmed non-targets were chosen to look suspicious, so on their
    own they show the model a narrow, hard slice of "clean". A random sample of
    unlabelled pairs is what the model actually has to score. Pairs a reference
    model rated as likely colluders (the frozen surrogate exclusion set) and
    pairs touching a disclosed colluder are left out, since some of those will
    be undisclosed targets.
    """
    universe = pl.read_parquet(C.CACHE / "pair_universe.parquet").filter(
        (pl.col("phase") == "development") & (pl.col("shared_hands") >= 38))
    colluders = pl.concat([pos_pairs["p1"], pos_pairs["p2"]]).unique().to_list()
    candidates = (
        universe.join(labels.select("p1", "p2"), on=["p1", "p2"], how="anti")
        .join(pl.read_parquet(C.ARTIFACTS / "surrogate_exclusion.parquet"),
              on=["table_id", "p1", "p2"], how="anti")
        .filter(~pl.col("p1").is_in(colluders) & ~pl.col("p2").is_in(colluders))
        # A seeded sample is only reproducible if the rows arrive in a fixed order.
        .sort("table_id", "p1", "p2")
        .sample(n=EXTRA_UNKNOWN_PAIRS, seed=C.SEED)
        .with_columns(pl.concat_str([pl.lit("U"), pl.col("p1"), pl.col("p2")]).alias("pair_id"))
        .select("pair_id", "p1", "p2")
    )
    return ev_mod.load_pair_hands(candidates, "development").with_columns(
        pl.lit(0, dtype=pl.Int8).alias("target"))


def aggregate_scores(df: pl.DataFrame, score: np.ndarray) -> pl.DataFrame:
    """Per-pair summary of its hand scores: the extremes, not the average."""
    scored = df.select("phase", "table_id", "p1", "p2").with_columns(
        pl.Series("hand_score", score)
    )
    s = pl.col("hand_score")
    return scored.group_by("phase", "table_id", "p1", "p2").agg(
        s.max().alias("hs_max"),
        s.top_k(3).mean().alias("hs_top3"),
        s.top_k(5).mean().alias("hs_top5"),
        s.top_k(10).mean().alias("hs_top10"),
        s.mean().alias("hs_mean"),
        s.quantile(0.99).alias("hs_p99"),
        s.quantile(0.95).alias("hs_p95"),
        (s > 0.5).sum().alias("hs_n_over_50"),
        (s > 0.2).sum().alias("hs_n_over_20"),
        (s > 0.05).sum().alias("hs_n_over_05"),
        s.sum().alias("hs_sum"),
        s.top_k(5).sum().alias("hs_top5_sum"),
    )


HAND_SCORES = C.CACHE / "hand_scores_dev.parquet"


def score_phase(booster: lgb.Booster, feats: list[str], phase: str,
                tables: list[str], hand_rows: list | None = None) -> pl.DataFrame:
    """Aggregate hand scores per pair; optionally keep the per-hand scores too.

    Stage 23 re-aggregates development hand scores over cropped windows of a
    pair's hands, which needs the scores hand by hand rather than summarised.
    """
    parts = []
    for i in range(0, len(tables), TABLES_PER_BATCH):
        batch = tables[i : i + TABLES_PER_BATCH]
        lf = pl.scan_parquet(C.CACHE / "pair_hands" / "*.parquet").filter(
            (pl.col("phase") == phase) & pl.col("table_id").is_in(batch))
        df = ev_mod.add_within_pair(ev_mod.derive(lf).collect())
        score = booster.predict(df.select(feats).to_numpy().astype(np.float32))
        parts.append(aggregate_scores(df, score))
        if hand_rows is not None:
            hand_rows.append(df.select("table_id", "p1", "p2", "hand_id")
                             .with_columns(pl.Series("hs", score)))
    return pl.concat(parts)


def main() -> None:
    t0 = time.time()
    feats = ev_mod.feature_names()
    train = labelled_frame()
    print(f"hand-level training rows: {train.height:,} "
          f"({int(train['target'].sum())} planted, {time.time() - t0:.0f}s)")

    x = train.select(feats).to_numpy().astype(np.float32)
    y = train["target"].to_numpy()
    folds = table_folds(train["table_id"].to_numpy(), N_FOLDS)

    all_tables = sorted(
        pl.scan_parquet(C.CACHE / "pair_hands" / "*.parquet")
        .select("table_id").unique().collect()["table_id"].to_list()
    )
    fold_of_table = dict(zip(train["table_id"].to_list(), folds.tolist()))
    # Tables with no labelled pair still need a fold so they can be scored.
    unseen = [t for t in all_tables if t not in fold_of_table]
    for j, t in enumerate(unseen):
        fold_of_table[t] = j % N_FOLDS

    dev_parts = []
    hand_rows: list[pl.DataFrame] = []
    for fold in range(N_FOLDS):
        mask = folds != fold
        booster = lgb.train(PARAMS, lgb.Dataset(x[mask], label=y[mask]),
                            num_boost_round=N_ROUNDS)
        held = [t for t in all_tables if fold_of_table[t] == fold]
        dev_parts.append(score_phase(booster, feats, "development", held, hand_rows))
        print(f"  fold {fold}: scored {len(held)} held-out tables "
              f"({time.time() - t0:.0f}s)")

    full = lgb.train(PARAMS, lgb.Dataset(x, label=y), num_boost_round=N_ROUNDS)
    eval_scores = score_phase(full, feats, "evaluation", all_tables)
    print(f"  evaluation scored ({time.time() - t0:.0f}s)")

    out = pl.concat([*dev_parts, eval_scores])
    out.write_parquet(OUT, compression="zstd")
    pl.concat(hand_rows).sort("table_id", "p1", "p2", "hand_id").write_parquet(
        HAND_SCORES, compression="zstd")
    print(f"wrote {OUT}: {out.height:,} pairs x {out.width} cols "
          f"({time.time() - t0:.0f}s)")

    imp = sorted(zip(feats, full.feature_importance("gain")), key=lambda kv: -kv[1])
    print("\ntop hand-suspicion features:")
    for name, gain in imp[:12]:
        print(f"  {gain:12.0f}  {name}")


if __name__ == "__main__":
    main()
