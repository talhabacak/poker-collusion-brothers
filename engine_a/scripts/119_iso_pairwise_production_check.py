"""Stage 119b: does the isolation comparator still gain on the production code path?

Stages 115-117 measured the comparator through the 43/75/82/90 research harness.
That harness has disagreed with production before, by 0.007, over nothing more
than the order the template and sizing blocks were appended in - `feature_fraction`
samples columns by position, so a permutation behaves like a seed. A re-ranker
that sits on top of the stage-1 order is exactly the kind of change that could
be carried by such a difference rather than by its own merit, so the gain is
re-measured here entirely through `06_evidence_model.py`: its `load_pair_hands`,
its `feature_names`, its `oof_stage1`, its specialist settings.

Two routings are reported. Development can gate on the true behaviour family,
but evaluation has no labels and must gate on the behaviour model's prediction,
so the predicted-family gate is measured here too - it is the one that will
actually ship, and the gap between the two is the cost of routing.

If the production path does not show the gain, nothing gets built.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# The production evidence configuration, set before the engine is imported so the
# feature list and the specialist settings are the ones a submission would use.
for _k, _v in {"EV_SIZING": "1", "EV_TEMPLATES": "1", "EV_SPEC_TUNED": "1"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C
from pokercol.cv import table_folds
from pokercol.metrics import evidence_map5

import importlib
ev = importlib.import_module("06_evidence_model")
rt = importlib.import_module("14_pair_retest")
m34 = importlib.import_module("34_other_routing")


def map5(df: pl.DataFrame, score: np.ndarray, evidence: pl.DataFrame,
         subset: np.ndarray | None = None) -> float:
    """MAP@5 over the pairs in `subset`, ordered by `score`."""
    scored = df.with_columns(pl.Series("score", score))
    if subset is not None:
        scored = scored.filter(pl.Series(subset))
    top5 = (scored.sort(["pair_id", "score", "hand_id"], descending=[False, True, False])
            .group_by("pair_id", maintain_order=True).head(5))
    submitted = {p: g["hand_id"].to_list() for p, g in top5.group_by("pair_id", maintain_order=True)}
    valid = {p: set(g["hand_id"].to_list()) for p, g in scored.group_by("pair_id", maintain_order=True)}
    keep = scored.select("pair_id").unique()
    relevant = {p: set(g["hand_id"].to_list())
                for p, g in evidence.join(keep, on="pair_id").group_by("pair_id", maintain_order=True)}
    return evidence_map5(submitted, relevant, valid)


def report(df, evidence, score, label, results) -> None:
    fam = df["behavior_family"].to_numpy()
    row = {"all": round(map5(df, score, evidence), 4)}
    for f in C.FAMILIES:
        row[f[:4]] = round(map5(df, score, evidence, subset=fam == f), 4)
    results[label] = row
    print(f"{label:38s} {row}", flush=True)


def predicted_family(df: pl.DataFrame) -> np.ndarray:
    """The behaviour model's out-of-fold argmax family per hand row.

    This is the same column the submission's `predicted_behavior` carries, which
    is what makes the evaluation gate and the file's own claim about a pair agree.
    """
    dev = rt.load()
    xp, _ = rt.matrix(dev)
    proba = m34.known_behaviour(xp, dev["y"].to_numpy(),
                                dev["behavior_family"].fill_null("none").to_numpy(),
                                table_folds(dev["table_id"].to_numpy(), 5), list(C.FAMILIES))
    argmax = {p: list(C.FAMILIES)[i] for p, i in zip(dev["pair_id"].to_list(), proba.argmax(1))}
    return np.array([argmax.get(p, "none") for p in df["pair_id"].to_list()])


def main() -> None:
    labels = pl.read_csv(C.DEV_LABELS).with_columns(
        pl.min_horizontal("player_1", "player_2").alias("p1"),
        pl.max_horizontal("player_1", "player_2").alias("p2"))
    pos = labels.filter(pl.col("label") == 1).select("pair_id", "p1", "p2", "behavior_family")
    evidence = pl.read_csv(C.DEV_EVIDENCE).select("pair_id", "hand_id").unique()

    df = ev.load_pair_hands(pos, "development")
    df = df.join(evidence.with_columns(pl.lit(1, dtype=pl.Int8).alias("is_evidence")),
                 on=["pair_id", "hand_id"], how="left").with_columns(
        pl.col("is_evidence").fill_null(0)).sort("pair_id", "hand_id")
    feats = ev.feature_names()
    folds = table_folds(df["table_id"].to_numpy(), ev.N_FOLDS)
    print(f"production frame {df.height:,} rows, {df['pair_id'].n_unique()} pairs, "
          f"{len(feats)} features, truncation {ev.PARAMS['lambdarank_truncation_level']}, "
          f"{ev.EVIDENCE_SEEDS} seeds", flush=True)

    results = {}
    stage1 = ev.oof_stage1(df, feats, folds, seeds=ev.EVIDENCE_SEEDS)
    report(df, evidence, stage1, "production listwise baseline", results)

    borda = ev.pairwise_oof(df, feats, stage1, folds)
    blended = ev.pairwise_blend(df["pair_id"], stage1, borda)
    report(df, evidence, blended, "comparator on every family", results)

    true_iso = df["behavior_family"].to_numpy() == ev.PAIRWISE_FAMILY
    report(df, evidence, np.where(true_iso, blended, stage1),
           "isolation only, true family", results)

    pred = predicted_family(df)
    pred_iso = pred == ev.PAIRWISE_FAMILY
    agree = (pred_iso == true_iso).mean()
    results["routing"] = {
        "pairs gated by predicted family": int(df.filter(pl.Series(pred_iso))["pair_id"].n_unique()),
        "pairs gated by true family": int(df.filter(pl.Series(true_iso))["pair_id"].n_unique()),
        "row agreement": round(float(agree), 4)}
    print(f"routing {results['routing']}", flush=True)
    report(df, evidence, np.where(pred_iso, blended, stage1),
           "isolation only, predicted family", results)

    (C.ARTIFACTS / "iso_pairwise_production_check.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
