"""Stage 11: evidence retrieval variants, measured on strict OOF MAP@5.

Questions this answers, in order of how much they could be worth:

* Does a pointwise classifier beat a pairwise ranker here? MAP@5 rewards putting
  a handful of specific hands on top, and the planted hands are absolutely - not
  just relatively - distinctive, so the ranking objective's advantage is not
  obvious.
* Do per-family specialists beat one pooled model? The three families leave very
  different traces, and an oracle-routed specialist bounds what perfect routing
  could buy.
* Does blending the two help, and how much does imperfect routing cost?

Folds are by table throughout, so a pair's own table never trains the model that
scores it.
"""
from __future__ import annotations

import json
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
from pokercol.metrics import evidence_map5

import importlib
ev_mod = importlib.import_module("06_evidence_model")

N_FOLDS = 5

CLF_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
    verbosity=-1, num_threads=C.N_THREADS, seed=C.SEED,
)


def build() -> tuple[pl.DataFrame, list[str]]:
    labels = pl.read_csv(C.DEV_LABELS).with_columns(
        pl.min_horizontal("player_1", "player_2").alias("p1"),
        pl.max_horizontal("player_1", "player_2").alias("p2"),
    )
    pos = labels.filter(pl.col("label") == 1).select("pair_id", "p1", "p2", "behavior_family")
    evidence = pl.read_csv(C.DEV_EVIDENCE).select("pair_id", "hand_id").unique()
    df = ev_mod.load_pair_hands(pos, "development")
    df = df.join(evidence.with_columns(pl.lit(1, dtype=pl.Int8).alias("is_evidence")),
                 on=["pair_id", "hand_id"], how="left").with_columns(
        pl.col("is_evidence").fill_null(0)).sort("pair_id")
    return df, ev_mod.feature_names()


def score_map5(df: pl.DataFrame, score: np.ndarray, evidence: pl.DataFrame) -> float:
    scored = df.with_columns(pl.Series("score", score))
    top5 = (
        scored.sort(["pair_id", "score", "hand_id"], descending=[False, True, False])
        .group_by("pair_id", maintain_order=True).head(5)
    )
    submitted = {p: g["hand_id"].to_list() for p, g in top5.group_by("pair_id", maintain_order=True)}
    valid = {p: set(g["hand_id"].to_list())
             for p, g in scored.group_by("pair_id", maintain_order=True)}
    # The average runs over the pairs actually in `df`. Taking the gold table
    # whole instead divided a subset's score by all 372 target pairs, so any
    # stage handed a slice - a holdout fold, an evaluation-like half - printed a
    # level roughly half the true one. Comparisons within a slice were unharmed,
    # since both sides carried the same denominator, but the levels were wrong.
    relevant = {p: set(g["hand_id"].to_list())
                for p, g in evidence.group_by("pair_id", maintain_order=True)
                if p in valid}
    return evidence_map5(submitted, relevant, valid)


def fit_ranker(df: pl.DataFrame, feats: list[str], rows: np.ndarray, rounds=400) -> lgb.Booster:
    tr = df.filter(pl.Series(rows)).sort("pair_id")
    groups = tr.group_by("pair_id", maintain_order=True).len()["len"].to_numpy()
    return lgb.train(ev_mod.PARAMS,
                     lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32),
                                 label=tr["is_evidence"].to_numpy(), group=groups),
                     num_boost_round=rounds)


def fit_classifier(df: pl.DataFrame, feats: list[str], rows: np.ndarray, rounds=400) -> lgb.Booster:
    tr = df.filter(pl.Series(rows))
    return lgb.train(CLF_PARAMS,
                     lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32),
                                 label=tr["is_evidence"].to_numpy()),
                     num_boost_round=rounds)


def within_pair_rank(df: pl.DataFrame, score: np.ndarray) -> np.ndarray:
    """Percentile inside each pair, so two models can be blended on equal terms."""
    tmp = df.select("pair_id").with_columns(pl.Series("s", score))
    return (tmp.with_columns(
        (pl.col("s").rank("average").over("pair_id") / pl.len().over("pair_id")).alias("r")
    )["r"].to_numpy())


def main() -> None:
    t0 = time.time()
    df, feats = build()
    evidence = pl.read_csv(C.DEV_EVIDENCE).select("pair_id", "hand_id").unique()
    x = df.select(feats).to_numpy().astype(np.float32)
    folds = table_folds(df["table_id"].to_numpy(), N_FOLDS)
    family = df["behavior_family"].to_numpy()
    print(f"{df.height:,} rows, {len(feats)} features ({time.time() - t0:.0f}s)\n")

    oof_rank = np.zeros(df.height)
    oof_clf = np.zeros(df.height)
    oof_spec = np.zeros(df.height)

    for fold in range(N_FOLDS):
        train = folds != fold
        held = ~train
        oof_rank[held] = fit_ranker(df, feats, train).predict(x[held])
        oof_clf[held] = fit_classifier(df, feats, train).predict(x[held])
        # Specialists, routed by the true family: an upper bound on routing.
        for fam in C.FAMILIES:
            rows = train & (family == fam)
            target = held & (family == fam)
            if rows.sum() == 0 or target.sum() == 0:
                continue
            oof_spec[target] = fit_ranker(df, feats, rows, rounds=300).predict(x[target])
        print(f"  fold {fold} ({time.time() - t0:.0f}s)")

    results = {
        "lambdarank (pooled)": score_map5(df, oof_rank, evidence),
        "binary classifier (pooled)": score_map5(df, oof_clf, evidence),
        "family specialists (oracle routing)": score_map5(df, oof_spec, evidence),
    }

    r_rank = within_pair_rank(df, oof_rank)
    r_clf = within_pair_rank(df, oof_clf)
    r_spec = within_pair_rank(df, oof_spec)
    results["rank + classifier 50/50"] = score_map5(df, r_rank + r_clf, evidence)
    results["rank .6 / classifier .4"] = score_map5(df, 0.6 * r_rank + 0.4 * r_clf, evidence)
    results["specialist .6 / pooled .4 (oracle)"] = score_map5(df, 0.6 * r_spec + 0.4 * r_rank, evidence)
    results["all three equal (oracle)"] = score_map5(df, r_rank + r_clf + r_spec, evidence)

    print(f"\n{'variant':40s} {'MAP@5':>8s}")
    for name, value in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"{name:40s} {value:8.5f}")

    print("\nper-family MAP@5 (pooled lambdarank):")
    for fam in C.FAMILIES:
        sub = df.filter(pl.Series(family == fam))
        sub_ev = evidence.join(sub.select("pair_id").unique(), on="pair_id", how="semi")
        print(f"  {fam:24s} {score_map5(sub, oof_rank[family == fam], sub_ev):.5f}")

    (C.ARTIFACTS / "evidence_experiments.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
