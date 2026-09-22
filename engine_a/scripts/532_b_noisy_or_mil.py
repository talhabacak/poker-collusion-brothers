"""Stage 532: the noisy-OR / smooth-max action head, trained in the teammate's folds.

The model, the loss, the censoring rule, the pooling pre-registration, the nested epoch choice
and the strict-holdout protocol are stage 481's, imported rather than copied, so nothing about
the method differs from what A measured. What differs is the frame: stage 531's bags, built
from B's tables, with the fold of every bag being B's `tidx % 5`. A column built here is out of
fold for B's reranker; a column built on A's folds is not (note 001 §6.1).

The four things note 001 says decide it, each asserted in code:

    censoring     `training_index` (481) drops every bag with label -1 and returns the count;
                  asserted here against stage 531's census, and again inside `fit` before the
                  first gradient step (`assert (label[train_bags] >= 0).all()`).
    pooling       noisy-OR against log-sum-exp, chosen on folds 0-3 only by the bag score used
                  alone as a within-pair ranking (`part_pooling`), then frozen. Fold 4 takes
                  no part.
    fold contract for reranker outer fold k, a row in fold j reads a column from a head trained
                  on universe - {k, j} (`blocks`, nested=True), asserted per block.
    holdout       fold 4 - the fold B's own `run_milhold.sh` held out - is in no fit, no
                  standardiser, no epoch choice; scored once from heads that saw folds 0-3.

    TARIK_INTERIM=<repro>/data/interim python scripts/532_b_noisy_or_mil.py --part pooling
    ...                                                                      --part holdout
    ...                                                                      --part columns

Writes artifacts/532_b_noisy_or_mil.json, artifacts/532_mil_full_nested_s*.parquet,
artifacts/532_mil_inner_s*.parquet and artifacts/532_mil_holdout_s*.parquet. Nothing here
submits anything.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C

s481 = importlib.import_module("481_noisy_or_mil")
s531 = importlib.import_module("531_b_action_bag_frame")

N_FOLDS = 5
HOLDOUT_FOLD = 4                        # B's run_milhold.sh held out fold 4 of tidx % 5
REPORT = C.ARTIFACTS / "532_b_noisy_or_mil.json"
FEATURES = s531.FEATURES


def load_frame() -> tuple[pl.DataFrame, np.ndarray]:
    if not s531.FRAME.exists():
        raise SystemExit(f"{s531.FRAME} is missing; run scripts/531_b_action_bag_frame.py")
    rows = pl.read_parquet(s531.FRAME).sort(["pair_id", "hidx", "action_no", "player"])
    codes = (rows.select("pair_id", "hidx").with_row_index("row")
             .group_by(["pair_id", "hidx"], maintain_order=True)
             .agg(pl.col("row").min().alias("start"), pl.len().alias("n")))
    bag_idx = np.repeat(np.arange(codes.height), codes["n"].to_numpy())
    return rows, bag_idx


def bag_table(rows: pl.DataFrame, bag_idx: np.ndarray) -> pl.DataFrame:
    """One row per bag: key, label, B's fold, family, and its length."""
    first = np.concatenate([[0], np.flatnonzero(np.diff(bag_idx)) + 1])
    out = rows[first].select("pair_id", "hand_id", "hidx", "tidx", "table_id", "fold",
                             "behavior_family", "is_pos_pair", "bag_label")
    out = out.with_columns(pl.col("fold").cast(pl.Int64),
                           pl.Series("n_actions", np.bincount(bag_idx)))
    assert (out["fold"] == out["tidx"] % N_FOLDS).all(), "the fold column is not tidx % 5"
    return out


def part_holdout(rows, bag_idx, bags, x, sizes, starts, report, form, seed, epochs) -> None:
    """Stage 481's holdout protocol on B's folds, written under this stage's names."""
    t0 = time.time()
    inner = [f for f in range(N_FOLDS) if f != HOLDOUT_FOLD]
    log: dict = {}
    a = s481.blocks(rows, bag_idx, bags, x, sizes, starts, inner, inner, form, seed, epochs, True, log)
    a.filter(pl.col("row_fold") != HOLDOUT_FOLD).write_parquet(C.ARTIFACTS / f"532_mil_inner_s{seed}.parquet")
    b = s481.blocks(rows, bag_idx, bags, x, sizes, starts, inner, [HOLDOUT_FOLD], form, seed, epochs, True, log)
    b.write_parquet(C.ARTIFACTS / f"532_mil_holdout_s{seed}.parquet")
    trained_on = [t for t in log if t.startswith(f"seed {seed} trained on")]
    assert trained_on, "no fit was recorded"
    assert all(str(HOLDOUT_FOLD) not in t.split("on ")[1] for t in trained_on), \
        "the holdout fold appears in a training set"
    report.setdefault("holdout protocol by seed", {})[str(seed)] = {
        "holdout fold": HOLDOUT_FOLD, "epochs": epochs,
        "holdout pairs": int(bags.filter((pl.col("fold") == HOLDOUT_FOLD) & pl.col("is_pos_pair"))["pair_id"].n_unique()),
        "inner pairs": int(bags.filter((pl.col("fold") != HOLDOUT_FOLD) & pl.col("is_pos_pair"))["pair_id"].n_unique()),
        "training sets used": sorted(t.split("on ")[1] for t in trained_on),
        "fits": log, "seconds": round(time.time() - t0, 1)}
    print(f"  holdout seed {seed}: {report['holdout protocol by seed'][str(seed)]['training sets used']} "
          f"({time.time() - t0:.0f}s)", flush=True)


def part_columns(rows, bag_idx, bags, x, sizes, starts, report, form, seeds, epochs) -> None:
    t0 = time.time()
    prev = report.get("columns (five-fold, nested)", {})
    log: dict = dict(prev.get("fits", {}))
    for seed in seeds:
        path = C.ARTIFACTS / f"532_mil_full_nested_s{seed}.parquet"
        if path.exists():
            # the first run lost this seed's in-memory fit log to a transient CUDA error on a later
            # seed; the training sets are deterministic and their censuses are the ones the holdout
            # protocol records, so the block on disk is reused and the reuse is written down
            log[f"seed {seed} reused from disk"] = str(path.name)
            print(f"  seed {seed}: nested blocks already on disk, reused", flush=True)
            continue
        log = {k: v for k, v in log.items() if not k.startswith(f"seed {seed} ")}
        out = s481.blocks(rows, bag_idx, bags, x, sizes, starts, list(range(N_FOLDS)),
                          list(range(N_FOLDS)), form, seed, epochs, True, log)
        out.write_parquet(path)
        print(f"  seed {seed}: nested blocks written ({time.time() - t0:.0f}s)", flush=True)
        report["columns (five-fold, nested)"] = {"epochs": epochs, "seeds": seeds, "fits": log}
        REPORT.write_text(json.dumps(report, indent=2))
    report["columns (five-fold, nested)"] = {"epochs": epochs, "seeds": seeds, "fits": log}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="all", choices=("pooling", "epochs", "columns", "holdout", "all"))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()

    t0 = time.time()
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    rows, bag_idx = load_frame()
    bags = bag_table(rows, bag_idx)
    x = rows.select(FEATURES).to_numpy().astype(np.float32)
    sizes = np.bincount(bag_idx)
    starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    frame_census = json.loads(s531.REPORT.read_text())["frame"]
    _, census = s481.training_index(bags, np.ones(bags.height, dtype=bool))
    # the frame holds bags that carry at least one action; the 531 census counts every cell and
    # records, per label, the cells in which neither member acted
    miss = frame_census.get("cells with no action rows by label", {})
    assert census["censored bags excluded from the loss"] == frame_census["bag_label counts"]["-1"] - miss.get("-1", 0), \
        "censoring does not match the frame census"
    assert census["positive"] == frame_census["bag_label counts"]["1"] - miss.get("1", 0), "positive bag count differs from the census"
    assert census["negative"] == frame_census["bag_label counts"]["0"] - miss.get("0", 0), "negative bag count differs from the census"
    assert frame_census["fold isolation violation count"] == 0
    report["setup"] = {
        "instance rows": int(rows.height), "bags": int(bags.height), "features": len(FEATURES),
        "device": str(s481.DEVICE), "fold map": "B's tidx % 5", "holdout fold": HOLDOUT_FOLD,
        "censoring": census,
        "architecture": {"dim": s481.DIM, "dropout": s481.DROPOUT, "lr": s481.LR,
                         "weight decay": s481.WEIGHT_DECAY, "max epochs": s481.MAX_EPOCHS,
                         "batch bags": s481.BATCH_BAGS, "neg per pos": s481.NEG_PER_POS, "tau": s481.TAU},
        "fold isolation violations": frame_census["fold isolation violation count"],
        "bags per fold (positive pairs)": {int(k): int(v) for k, v in
                                           bags.filter(pl.col("is_pos_pair")).group_by("fold").len().sort("fold").iter_rows()}}
    print(json.dumps(report["setup"], indent=2), flush=True)

    form = report.get("pooling", {}).get("chosen")
    if args.part in ("pooling", "all") or form is None:
        form = s481.part_pooling(rows, bag_idx, bags, x, sizes, starts, report)
        REPORT.write_text(json.dumps(report, indent=2))
    if args.part in ("epochs", "all") or "epoch budget" not in report.get("pooling", {}):
        sub: dict = {}
        inner = [f for f in range(N_FOLDS) if f != HOLDOUT_FOLD]
        best = s481.nested_epochs(x, bag_idx, bags, inner, bags["fold"].to_numpy(), form, C.SEED, sub)
        report.setdefault("pooling", {}).update(
            {"epoch budget": int(best), "epoch budget curve": sub["validation curve"],
             "epoch budget chosen on": "folds 0-3 rotation only; fold 4 took no part"})
        REPORT.write_text(json.dumps(report, indent=2))
    epochs = int(report["pooling"]["epoch budget"])
    print(f"pooling: {form}, epoch budget {epochs} ({time.time() - t0:.0f}s)", flush=True)

    if args.part in ("holdout", "all"):
        for s in [C.SEED + int(v) for v in args.seeds.split(",")]:
            part_holdout(rows, bag_idx, bags, x, sizes, starts, report, form, s, epochs)
            REPORT.write_text(json.dumps(report, indent=2))
    if args.part in ("columns", "all"):
        seeds = [C.SEED + int(s) for s in args.seeds.split(",")]
        part_columns(rows, bag_idx, bags, x, sizes, starts, report, form, seeds, epochs)
        REPORT.write_text(json.dumps(report, indent=2))
    report["runtime_s"] = round(time.time() - t0, 1)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"written to {REPORT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
