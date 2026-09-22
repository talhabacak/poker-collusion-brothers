"""Stage 13: treat confidently-scored unknown pairs as the colluders they likely are.

Out of fold, 468 development pairs score above 0.5 but only ~309 of the top 450
are disclosed targets. The rest are unlabelled pairs the model is confident
about - most plausibly undisclosed colluders, the same population that makes the
development surrogate read low. Training has been weighting them as soft
negatives, i.e. actively teaching the model to push real colluders down.

The plain population surrogate cannot judge a fix for this, because it scores
those same pairs as false positives. So this stage also reports a *cleaned*
surrogate that drops a fixed exclusion set - unknowns the reference model scores
above 0.5 - from the evaluation negatives. The set is chosen once, before any
variant is trained, so every variant is measured on identical rows.

Pseudo-labels for a fold come from out-of-fold scores computed without that
fold's own tables, so a held-out table never decides its own labels.
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

from pokercol import config as C
from pokercol.cv import attach_labels, candidate_pairs, load_pair_table, population_average_precision, table_folds

N_FOLDS = 5
ID_COLS = {"phase", "table_id", "p1", "p2", "pair_id", "label", "label_status",
           "behavior_family", "y", "fold"}
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              max_bin=127, verbosity=-1, num_threads=C.N_THREADS, seed=C.SEED)
ROUNDS = 700
POS_W, NEG_W, UNK_W = 10.0, 1.0, 0.3


def load() -> pl.DataFrame:
    pairs = load_pair_table()
    return attach_labels(candidate_pairs(pairs, "development"), pl.read_csv(C.DEV_LABELS))


def fit_predict(x, target, weight, train, held):
    booster = lgb.train(PARAMS, lgb.Dataset(x[train], label=target[train], weight=weight[train]),
                        num_boost_round=ROUNDS)
    return booster.predict(x[held])


def reference_oof(x, y, folds) -> np.ndarray:
    target = (y == 1).astype(np.int8)
    weight = np.where(y == 1, POS_W, np.where(y == 0, NEG_W, UNK_W))
    oof = np.zeros(len(y))
    for f in range(N_FOLDS):
        train = folds != f
        oof[~train] = fit_predict(x, target, weight, train, ~train)
    return oof


def nested_scores(x, y, folds, f) -> np.ndarray:
    """Scores for fold-`f` training rows from models that never saw fold `f`."""
    train_idx = np.flatnonzero(folds != f)
    inner = folds[train_idx]
    target = (y == 1).astype(np.int8)
    weight = np.where(y == 1, POS_W, np.where(y == 0, NEG_W, UNK_W))
    scores = np.full(len(y), np.nan)
    for g in np.unique(inner):
        tr = np.zeros(len(y), bool)
        tr[train_idx[inner != g]] = True
        held = np.zeros(len(y), bool)
        held[train_idx[inner == g]] = True
        scores[held] = fit_predict(x, target, weight, tr, held)
    return scores


def main() -> None:
    t0 = time.time()
    dev = load()
    feats = [c for c, t in zip(dev.columns, dev.dtypes) if c not in ID_COLS and t.is_numeric()]
    x = dev.select(feats).to_numpy().astype(np.float32)
    y = dev["y"].to_numpy()
    folds = table_folds(dev["table_id"].to_numpy(), N_FOLDS)

    ref = reference_oof(x, y, folds)
    exclude = (y == -1) & (ref > 0.5)
    keep = ~exclude
    print(f"reference model ready ({time.time() - t0:.0f}s); exclusion set: "
          f"{int(exclude.sum())} unknowns scoring > 0.5")

    def report(name: str, oof: np.ndarray) -> dict:
        row = dict(
            variant=name,
            population_ap=population_average_precision(y == 1, oof),
            cleaned_ap=population_average_precision((y == 1)[keep], oof[keep]),
            labelled_ap=population_average_precision((y == 1)[y >= 0], oof[y >= 0]),
        )
        print(f"{name:34s} pop={row['population_ap']:.4f} cleaned={row['cleaned_ap']:.4f} "
              f"labelled={row['labelled_ap']:.4f} ({time.time() - t0:.0f}s)")
        return row

    results = [report("baseline (unknowns soft-negative)", ref)]

    inner_scores = {f: nested_scores(x, y, folds, f) for f in range(N_FOLDS)}
    print(f"nested pseudo-label scores ready ({time.time() - t0:.0f}s)")

    variants = {
        "drop unknowns > 0.9": dict(threshold=0.9, mode="drop"),
        "drop unknowns > 0.5": dict(threshold=0.5, mode="drop"),
        "positive unknowns > 0.9 (w=3)": dict(threshold=0.9, mode="positive", w=3.0),
        "positive unknowns > 0.9 (w=10)": dict(threshold=0.9, mode="positive", w=10.0),
        "positive unknowns > 0.7 (w=3)": dict(threshold=0.7, mode="positive", w=3.0),
    }
    for name, v in variants.items():
        oof = np.zeros(len(y))
        for f in range(N_FOLDS):
            train = folds != f
            s = inner_scores[f]
            confident = train & (y == -1) & (np.nan_to_num(s, nan=0.0) > v["threshold"])
            target = (y == 1).astype(np.int8)
            weight = np.where(y == 1, POS_W, np.where(y == 0, NEG_W, UNK_W)).astype(float)
            if v["mode"] == "drop":
                weight[confident] = 0.0
            else:
                target = target.copy()
                target[confident] = 1
                weight[confident] = v["w"]
            use = train & (weight > 0)
            oof[~train] = fit_predict(x, target, weight, use, ~train)
        results.append(report(name, oof))

    (C.ARTIFACTS / "pseudo_labels.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
