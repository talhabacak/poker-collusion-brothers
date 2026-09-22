"""Stage 534: the candidate file, built only if stage 533 cleared the bar.

`tarik_v55_best.csv` with its evidence columns replaced by B's v55 reranker carrying the
seed-bagged action-head columns; risk and behaviour carried over byte for byte (the file is
read and written as text, so not even a float's spelling changes).

The evaluation side needs the head's columns for every evaluation hand of every evaluation
pair - all of them, not only the reranker's fifty candidates, because `pct_mil_max_z` is a
within-pair percentile and on the development side it was taken over every hand of the pair.
Five heads (one per stage 532 seed) are fitted on all five development folds with the frozen
pooling and epoch budget, every evaluation bag is scored by each, and the columns are averaged
exactly as stage 533's `seed_bag` averaged the development blocks.

    TARIK_INTERIM=... TARIK_RUN=... python scripts/534_b_mil_candidate.py [--dry-run]

`--dry-run` builds and scores one evaluation chunk and stops, to prove the path before the
verdict exists. Writes submissions/tarik_v55_mil_evidence.csv and artifacts/534_b_mil_candidate.json.
Nothing here submits anything.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
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
s532 = importlib.import_module("532_b_noisy_or_mil")
s533 = importlib.import_module("533_b_mil_evidence_arm")
s08 = importlib.import_module("08_validate_submission")

INTERIM = s533.INTERIM
RUN = s533.RUN
REPORT = C.ARTIFACTS / "534_b_mil_candidate.json"
EVAL_BLOCK = INTERIM / "534_mil_eval_block.parquet"
DEV_BLOCK = s533.BLOCKS / "full_mil_bag.parquet"
OUT_SUB = C.SUBMISSIONS / "tarik_v55_mil_evidence.csv"
SHIPPED = C.SUBMISSIONS / "tarik_v55_best.csv"
NCH = 20
EV = [f"evidence_hand_{i}" for i in range(1, 6)]


def heads(seeds: list[int], form: str, epochs: int):
    """One head per seed, fitted on every eligible development bag (all five folds)."""
    rows, bag_idx = s532.load_frame()
    bags = s532.bag_table(rows, bag_idx)
    x = rows.select(s531.FEATURES).to_numpy().astype(np.float32)
    tr, census = s481.training_index(bags, np.ones(bags.height, dtype=bool))
    frame_census = json.loads(s531.REPORT.read_text())["frame"]
    assert census["censored bags excluded from the loss"] == frame_census["bag_label counts"]["-1"]
    fitted = []
    for seed in seeds:
        model, pars, _ = s481.fit(x, bag_idx, bags, tr, form, epochs, seed)
        fitted.append((model, pars))
    return fitted, census


def eval_bags(ch: int, evp: pl.DataFrame, hands: pl.DataFrame) -> pl.DataFrame:
    files = sorted((INTERIM / "pairhand").glob("chunk*.parquet"))
    b = pl.concat([pl.scan_parquet(f).filter((pl.col("is_eval") == 1) & (pl.col("tidx") % NCH == ch))
                   .select("hidx", "p", "q", "tidx").join(evp.lazy(), on=["p", "q"]).collect() for f in files])
    return b.join(hands, on="hidx", how="left")


def score_chunk(bags: pl.DataFrame, fitted, form: str) -> pl.DataFrame:
    rows = s531.instances(bags, 1)
    rows = (rows.join(bags.select("pair_id", "hidx", "hand_id", "players_at_showdown"), on=["pair_id", "hidx"], how="left")
            .select(["pair_id", "hidx", "hand_id", "player"] + s531.FEATURES)
            .with_columns([pl.col(c).cast(pl.Float32) for c in s531.FEATURES])
            .sort(["pair_id", "hidx", "action_no", "player"]))
    codes = (rows.select("pair_id", "hidx").with_row_index("row").group_by(["pair_id", "hidx"], maintain_order=True)
             .agg(pl.col("row").min().alias("start"), pl.len().alias("n")))
    bag_idx = np.repeat(np.arange(codes.height), codes["n"].to_numpy())
    sizes = np.bincount(bag_idx)
    starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    first = np.concatenate([[0], np.flatnonzero(np.diff(bag_idx)) + 1])
    btab = rows[first].select("pair_id", "hidx", "hand_id")
    x = rows.select(s531.FEATURES).to_numpy().astype(np.float32)
    which = np.arange(btab.height)
    acc = None
    for model, pars in fitted:
        xs = torch.from_numpy(s481.apply_standardiser(x, pars)).to(s481.DEVICE)
        _, z = s481.bag_logits(model, xs, sizes, starts, which, form)
        cols = s481.action_columns(rows, btab, which, z, sizes, starts).select(s533.MIL_COLS)
        acc = cols if acc is None else acc.with_columns([(pl.col(c) + cols[c]).alias(c) for c in s533.MIL_COLS])
    acc = acc.with_columns([(pl.col(c) / len(fitted)).alias(c) for c in s533.MIL_COLS])
    out = btab.with_columns(acc).with_columns(
        (pl.col("mil_max_z").rank("average").over("pair_id") / pl.len().over("pair_id")).cast(pl.Float32).alias("pct_mil_max_z"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    # The bar in stage 533 was +0.010 real over placebo; it read +0.00729. The
    # user chose to build and send it anyway (2026-09-20, ~12:30 UTC), on the
    # grounds that it is the first evidence gain measured in the engine whose
    # development metric transfers. That decision is recorded in the report
    # rather than by editing the verdict, which stays what it measured.
    parser.add_argument("--user-approved", action="store_true",
                        help="build although stage 533 did not clear the bar; recorded in the report")
    args = parser.parse_args()
    t0 = time.time()
    verdict = json.loads(s533.REPORT.read_text())["verdict"]
    if not args.dry_run and not verdict.get("cleared") and not args.user_approved:
        raise SystemExit(f"stage 533 did not clear the bar; no candidate is built: {json.dumps(verdict)}")
    r532 = json.loads(s532.REPORT.read_text())
    form, epochs = r532["pooling"]["chosen"], int(r532["pooling"]["epoch budget"])
    seeds = [int(s) for s in r532["columns (five-fold, nested)"]["seeds"]]
    out = {"verdict from 533": verdict, "pooling": form, "epochs": epochs, "seeds": seeds,
           "built_despite_bar": bool(args.user_approved and not verdict.get("cleared")),
           "authorised_by": "user, explicit, 2026-09-20" if args.user_approved else None}

    fitted, census = heads(seeds, form, epochs)
    out["heads on all five folds"] = census
    print(f"{len(fitted)} heads fitted on all folds ({time.time() - t0:.0f}s)", flush=True)

    players = pl.read_parquet(INTERIM / "players.parquet").select("player_id", "pidx")
    evp = s533.to_pq(pl.read_csv(C.EVAL_PAIRS), players).select("pair_id", "p", "q")
    hands = pl.read_parquet(INTERIM / "hands.parquet").select("hidx", "hand_id", "players_at_showdown")
    parts = []
    for ch in range(NCH):
        b = eval_bags(ch, evp, hands)
        parts.append(score_chunk(b, fitted, form))
        print(f"  eval chunk {ch}: {b.height:,} bags scored ({time.time() - t0:.0f}s)", flush=True)
        if args.dry_run:
            print(parts[0].describe(), flush=True)
            return
    block = pl.concat(parts)
    key = block.select("pair_id", "hidx").join(evp, on="pair_id").select("hidx", "p", "q")
    block = key.with_columns(block.select(s533.MIL_COLS))
    assert block.select("hidx", "p", "q").n_unique() == block.height
    block.write_parquet(EVAL_BLOCK)
    out["evaluation block"] = {"rows": block.height, "pairs": int(evp.height), "path": str(EVAL_BLOCK)}
    print(f"evaluation block written: {block.height:,} rows ({time.time() - t0:.0f}s)", flush=True)

    # ---- the full 05b pass: development models with the seed-bagged nested block, evaluation with this one
    assert DEV_BLOCK.exists(), f"{DEV_BLOCK} missing - stage 533 did not build the seed-bagged arm"
    env = {**os.environ, **{k: v for k, v in s533.BASE_ENV.items() if k != "RERANK_CV_ONLY"},
           "MIL_BLOCK": str(DEV_BLOCK), "MIL_BLOCK_EVAL": str(EVAL_BLOCK),
           "RERANK_OUT": "submission_534_mil.csv", "RERANK_DUMP": "534_final"}
    with open(RUN / "logs" / "05b_534.log", "w") as lo, open(RUN / "logs" / "05b_534.err", "w") as le:
        subprocess.run([sys.executable, s533.HARNESS], cwd=RUN, env=env, stdout=lo, stderr=le, check=True)
    cv = json.loads((INTERIM / "reranker_cv.json").read_text())
    out["final pass cv"] = {k: v for k, v in cv.items() if k != "mil_cols"}
    print(f"final pass: dev reranker_map5_oof {cv['reranker_map5_oof']:.5f} ({time.time() - t0:.0f}s)", flush=True)

    # ---- splice: risk and behaviour from the shipped v55 file as text, evidence from the new pass
    new = pl.read_csv(RUN / "submission_534_mil.csv", infer_schema=False).select(["pair_id"] + EV)
    old = pl.read_csv(SHIPPED, infer_schema=False)
    assert list(old.columns) == list(C.SUBMISSION_COLUMNS)
    cand = old.select("pair_id", "risk_score", "predicted_behavior").join(new, on="pair_id", how="left", maintain_order="left")
    assert cand.height == old.height and cand.select(EV).null_count().to_numpy().sum() == 0
    assert cand["pair_id"].to_list() == old["pair_id"].to_list()
    assert cand["risk_score"].to_list() == old["risk_score"].to_list(), "risk text changed"
    assert cand["predicted_behavior"].to_list() == old["predicted_behavior"].to_list(), "behaviour text changed"
    cand.select(list(C.SUBMISSION_COLUMNS)).write_csv(OUT_SUB)
    changed = float(np.mean([(old[c] != cand[c]).mean() for c in EV]))
    same_set = float(np.mean([set(a) == set(b) for a, b in zip(zip(*[old[c].to_list() for c in EV]),
                                                               zip(*[cand[c].to_list() for c in EV]))]))
    rep = s08.validate(OUT_SUB)
    out["candidate"] = {"path": str(OUT_SUB), "sha256": hashlib.sha256(OUT_SUB.read_bytes()).hexdigest(),
                        "risk and behaviour": "byte-identical text to tarik_v55_best.csv (asserted)",
                        "evidence cells changed vs v55": round(changed, 4),
                        "pairs whose set of five is unchanged": round(same_set, 4),
                        "validation": rep}
    out["runtime_s"] = round(time.time() - t0, 1)
    REPORT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["candidate"], indent=1, default=str), flush=True)
    print(f"written to {REPORT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
