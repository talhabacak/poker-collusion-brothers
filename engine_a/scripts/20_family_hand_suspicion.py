"""Stage 4b-iii: one hand-suspicion model per behaviour family.

The single hand model scores the planted hands of the hardest disclosed targets
at hs_top5 0.35 against 0.98 for easy ones, even though those hands stand out
16x on partner-directed surprise. Soft play, directed transfer and isolation
plant very different hands, and a pooled classifier is pulled towards the
loudest. Here each family gets its own hand model (its planted hands against
all negatives, other families' planted hands left out), and the pair inherits
each model's extremes.

Out of fold by table exactly as stage 4b. Writes hand_suspicion_family.parquet.

On the cleaned surrogate, on top of pseudo-labels and the family pair-model
blend: 0.9793 -> 0.9813, with soft play, directed transfer and isolation all up.
Runs after 045b, since its unlabelled negatives avoid the frozen exclusion set.
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
hs = importlib.import_module("045_hand_suspicion")
ev_mod = hs.ev_mod

OUT = C.CACHE / "hand_suspicion_family.parquet"
SHORT = {"directed_transfer": "dir", "soft_play": "soft", "coordinated_isolation": "iso"}


def main() -> None:
    t0 = time.time()
    feats = ev_mod.feature_names()
    train = hs.labelled_frame()
    fam_of_pair = pl.read_csv(C.DEV_LABELS).select("pair_id", "behavior_family")
    train = train.join(fam_of_pair, on="pair_id", how="left").with_columns(
        pl.col("behavior_family").fill_null("none"))
    x = train.select(feats).to_numpy().astype(np.float32)
    y = train["target"].to_numpy()
    fam = train["behavior_family"].to_numpy()
    folds = table_folds(train["table_id"].to_numpy(), hs.N_FOLDS)

    tables = sorted(pl.scan_parquet(C.CACHE / "pair_hands" / "*.parquet")
                    .select("table_id").unique().collect()["table_id"].to_list())
    fold_of = dict(zip(train["table_id"].to_list(), folds.tolist()))
    for j, t in enumerate([t for t in tables if t not in fold_of]):
        fold_of[t] = j % hs.N_FOLDS

    boosters: dict[tuple[str, int], lgb.Booster] = {}
    for name in C.FAMILIES:
        keep = (y == 0) | (fam == name)
        for fold in [*range(hs.N_FOLDS), -1]:
            rows = keep & ((folds != fold) if fold >= 0 else True)
            boosters[(name, fold)] = lgb.train(hs.PARAMS, lgb.Dataset(x[rows], label=y[rows]),
                                               num_boost_round=hs.N_ROUNDS)
        print(f"  {name}: models fitted ({time.time() - t0:.0f}s)", flush=True)

    parts = []
    hand_rows: list[pl.DataFrame] = []
    for phase in ("development", "evaluation"):
        for i in range(0, len(tables), hs.TABLES_PER_BATCH):
            batch = tables[i : i + hs.TABLES_PER_BATCH]
            lf = pl.scan_parquet(C.CACHE / "pair_hands" / "*.parquet").filter(
                (pl.col("phase") == phase) & pl.col("table_id").is_in(batch))
            df = ev_mod.add_within_pair(ev_mod.derive(lf).collect()).sort("p1", "p2", "hand_id")
            xb = df.select(feats).to_numpy().astype(np.float32)
            row_fold = np.array([fold_of[t] for t in df["table_id"].to_list()])
            agg = None
            per_hand = df.select("table_id", "p1", "p2", "hand_id")
            for name in C.FAMILIES:
                score = np.zeros(df.height)
                if phase == "evaluation":
                    score = boosters[(name, -1)].predict(xb)
                else:
                    for fold in np.unique(row_fold):
                        m = row_fold == fold
                        score[m] = boosters[(name, int(fold))].predict(xb[m])
                if per_hand is not None:
                    per_hand = per_hand.with_columns(pl.Series(f"hs{SHORT[name]}", score))
                s = hs.aggregate_scores(df, score)
                s = s.rename({c: f"{c.replace('hs_', 'hs' + SHORT[name] + '_')}"
                              for c in s.columns if c.startswith("hs_")})
                agg = s if agg is None else agg.join(s, on=["phase", "table_id", "p1", "p2"])
            parts.append(agg)
            hand_rows.append(per_hand)
        # Per-hand scores: development (out of fold) for stage 23's cropped
        # re-aggregation and the isolation token model's training hands,
        # evaluation (full models) for its candidates.
        pl.concat(hand_rows).sort("table_id", "p1", "p2", "hand_id").write_parquet(
            C.CACHE / f"hand_scores_family_{'dev' if phase == 'development' else 'eval'}.parquet",
            compression="zstd")
        hand_rows = []
        print(f"  {phase} scored ({time.time() - t0:.0f}s)", flush=True)

    out = pl.concat(parts).sort("phase", "table_id", "p1", "p2")
    out.write_parquet(OUT, compression="zstd")
    print(f"wrote {OUT}: {out.height:,} pairs x {out.width} cols ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
