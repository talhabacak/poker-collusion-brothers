"""Stage 25 (experiment): does an opponent's hidden hand explain a player's actions?

An honest player's action depends on the public state and their own cards. A
player who knows their partner's cards - or who is steering chips to them - acts
in a way that the partner's *hidden* hand predicts. So fit an auxiliary policy

    P(action | state, own cards, one active opponent's cards)

on the whole population, next to the stage-2b-style base policy without the
opponent's cards, and score every (action, active opponent) pair by

    d_base = log P_aux - log P_base      information the opponent's cards add
    d_cent = log P_aux - mean over the action's active opponents of log P_aux

`d_cent` needs at least two active opponents and cancels any calibration gap
between the two models; `d_base` also covers heads-up spots. Summed over a table
and contrasted against the actor's other opponents, a colluding pair should
stand out: the partner's cards explain the actor's play and nobody else's do.

    python scripts/25_hidden_info_residual.py --score   # ~15 min, writes caches
    python scripts/25_hidden_info_residual.py --eval    # pair AP + evidence MAP

    python scripts/25_hidden_info_residual.py --production  # pseudo + family blend

Kill criterion (docs/ROADMAP_REVIEW.md): pair cleaned AP +0.003 or evidence +0.005.

Result - rejected. The opponent's cards do add information population-wide
(held-out log-loss 0.4194 -> 0.4143), but little of it is collusion-specific:
plain pair model 0.9769 -> 0.9780 (soft play +0.005, isolation -0.002), the
production setup 0.9812 -> 0.9814, evidence MAP@5 0.610 -> 0.604.
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

import importlib
pol = importlib.import_module("02b_action_surprise")

HAND_OUT = C.CACHE / "hidden_info_hand.parquet"
PAIR_OUT = C.CACHE / "hidden_info_pair.parquet"
TRAIN_ACTIONS = 1_600_000
HOLDOUT_ACTIONS = 300_000
N_CHUNKS = 16

OPP_FEATURES = [
    "opp_strength", "opp_category", "opp_equity", "opp_bucket", "opp_flush_cards",
    "opp_straight_run", "opp_strength_diff", "opp_equity_diff", "opp_ahead",
    "opp_is_responding_to", "opp_rel_position", "opp_best_active", "n_active_opps",
]


def actions_with_strength() -> pl.DataFrame:
    strength = pol.street_strength()
    a = pol.action_frame()
    a = a.join(strength, on=["hand_id", "player_id", "street"], how="left").with_columns(
        pl.col("hand_strength").fill_null(0), pl.col("board_strength").fill_null(0),
        pl.col("flush_cards").fill_null(0), pl.col("straight_run").fill_null(0),
        pl.col("flush_uses_hole").fill_null(0),
    ).with_columns(
        (pl.col("hand_strength") - pl.col("board_strength")).alias("strength_over_board"),
        (pl.col("hand_strength") // (14 ** 5)).cast(pl.Int8).alias("hand_category"),
        pl.col("action").replace_strict({n: i for i, n in enumerate(pol.ACTIONS)},
                                        return_dtype=pl.Int8).alias("y"),
    ).sort("hand_id", "action_no")
    a = a.with_columns(pl.col("hand_id").rank("dense").cast(pl.Int32).alias("hand_idx"))

    wide = pl.scan_parquet(C.SEATS).select("hand_id", "player_id", "seat_no").join(
        pl.scan_parquet(C.CACHE / "seat_features.parquet").select(
            "hand_id", "player_id", "preflop_bucket", "preflop_equity_hu"),
        on=["hand_id", "player_id"], how="left").collect()
    for s in (1, 2, 3):
        wide = wide.join(
            strength.filter(pl.col("street") == s).select(
                "hand_id", "player_id",
                pl.col("hand_strength").alias(f"str_{s}"),
                pl.col("flush_cards").alias(f"fc_{s}"),
                pl.col("straight_run").alias(f"sr_{s}")),
            on=["hand_id", "player_id"], how="left")
    folds = a.filter(pl.col("action") == "fold").group_by("hand_id", "player_id").agg(
        pl.col("action_no").min().alias("fold_no"))
    hand_map = a.select("hand_id", "hand_idx").unique()
    opp = (wide.join(folds, on=["hand_id", "player_id"], how="left")
           .join(hand_map, on="hand_id", how="inner")
           .drop("hand_id")
           .rename({"player_id": "opp_id", "seat_no": "opp_seat"})
           .sort("hand_idx", "opp_id"))
    return a, opp


def expand(chunk: pl.DataFrame, opp: pl.DataFrame) -> pl.DataFrame:
    """One row per (action, opponent still in the hand when the action was taken)."""
    street = pl.col("street")
    pick = lambda base: (pl.when(street == 1).then(pl.col(f"{base}_1"))
                         .when(street == 2).then(pl.col(f"{base}_2"))
                         .when(street == 3).then(pl.col(f"{base}_3"))
                         .otherwise(0).fill_null(0))
    e = (
        chunk.select("row", "hand_idx", "hand_id", "player_id", "action_no", "street", "seat_no",
                     "responding_to", "hand_strength", "preflop_equity_hu", "y")
        .join(opp, on="hand_idx", how="inner")
        .filter((pl.col("opp_id") != pl.col("player_id"))
                & (pl.col("fold_no").is_null() | (pl.col("fold_no") > pl.col("action_no"))))
        .with_columns(
            pick("str").cast(pl.Int64).alias("opp_strength"),
            pick("fc").alias("opp_flush_cards"),
            pick("sr").alias("opp_straight_run"),
        )
    )
    # The join suffixes the opponent's copy of `preflop_equity_hu` with `_right`.
    return (
        e.with_columns(
            (pl.col("opp_strength") // (14 ** 5)).cast(pl.Int8).alias("opp_category"),
            pl.col("preflop_equity_hu_right").alias("opp_equity"),
            pl.col("preflop_bucket").alias("opp_bucket"),
            pl.when(street == 0).then(0).otherwise(pl.col("opp_strength") - pl.col("hand_strength"))
            .alias("opp_strength_diff"),
            (pl.col("preflop_equity_hu_right") - pl.col("preflop_equity_hu")).alias("opp_equity_diff"),
            (pl.col("responding_to") == pl.col("opp_id")).fill_null(False).cast(pl.Int8)
            .alias("opp_is_responding_to"),
            ((pl.col("opp_seat") - pl.col("seat_no")) % C.SEATS_PER_HAND).alias("opp_rel_position"),
        )
        .with_columns(
            pl.when(street == 0).then(pl.col("opp_equity_diff") > 0)
            .otherwise(pl.col("opp_strength_diff") > 0).cast(pl.Int8).alias("opp_ahead"),
            pl.when(street == 0).then(pl.col("opp_equity")).otherwise(pl.col("opp_strength").cast(pl.Float64))
            .alias("_opp_value"),
        )
        .with_columns(
            (pl.col("_opp_value") == pl.col("_opp_value").max().over("row")).cast(pl.Int8)
            .alias("opp_best_active"),
            pl.len().over("row").alias("n_active_opps"),
        )
        .sort("row", "opp_id")
    )


def score() -> None:
    t0 = time.time()
    a, opp = actions_with_strength()
    a = a.with_row_index("row")
    x_base = a.select(pol.FEATURES).to_numpy().astype(np.float32)
    print(f"actions {a.height:,}, seats {opp.height:,} ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(C.SEED + 25)
    order = rng.permutation(a.height)
    train_rows = np.sort(order[:TRAIN_ACTIONS])
    hold_rows = np.sort(order[TRAIN_ACTIONS:TRAIN_ACTIONS + HOLDOUT_ACTIONS])

    def matrices(e: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = e["row"].to_numpy()
        xo = e.select(OPP_FEATURES).to_numpy().astype(np.float32)
        return x_base[rows], np.hstack([x_base[rows], xo]), e["y"].to_numpy()

    e_tr = expand(a[train_rows], opp)
    e_ho = expand(a[hold_rows], opp)
    xb_tr, xa_tr, y_tr = matrices(e_tr)
    xb_ho, xa_ho, y_ho = matrices(e_ho)
    print(f"training rows {e_tr.height:,}, holdout rows {e_ho.height:,} ({time.time() - t0:.0f}s)",
          flush=True)
    base = lgb.train(pol.PARAMS, lgb.Dataset(xb_tr, label=y_tr), num_boost_round=pol.N_ROUNDS)
    aux = lgb.train(pol.PARAMS, lgb.Dataset(xa_tr, label=y_tr), num_boost_round=pol.N_ROUNDS)
    ll = lambda p, y: float(-np.log(np.clip(p[np.arange(y.size), y], 1e-9, 1)).mean())
    report = {"holdout_logloss_base": ll(base.predict(xb_ho), y_ho),
              "holdout_logloss_aux": ll(aux.predict(xa_ho), y_ho)}
    print(f"holdout log-loss base {report['holdout_logloss_base']:.4f} "
          f"aux {report['holdout_logloss_aux']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    imp = dict(zip(pol.FEATURES + OPP_FEATURES, aux.feature_importance("gain").tolist()))
    report["aux_gain_opp_features"] = {k: imp[k] / sum(imp.values()) for k in OPP_FEATURES}
    del e_tr, xb_tr, xa_tr, e_ho, xb_ho, xa_ho

    fold_id = pol.ACTIONS.index("fold")
    hand_idx = a["hand_idx"].to_numpy()
    bounds = np.searchsorted(hand_idx, np.linspace(1, hand_idx.max() + 1, N_CHUNKS + 1))
    parts = []
    for c in range(N_CHUNKS):
        lo, hi = int(bounds[c]), int(bounds[c + 1])
        chunk = a[lo:hi]
        yb = chunk["y"].to_numpy()
        lp_base_action = np.log(np.clip(base.predict(x_base[lo:hi])[np.arange(hi - lo), yb], 1e-9, 1))
        e = expand(chunk, opp)
        _, xa, ye = matrices(e)
        lp_aux = np.log(np.clip(aux.predict(xa)[np.arange(ye.size), ye], 1e-9, 1))
        lp_base = lp_base_action[e["row"].to_numpy() - lo]
        scored = e.select("hand_id", "player_id", "opp_id", "row", "opp_is_responding_to").with_columns(
            pl.Series("d_base", lp_aux - lp_base),
            pl.Series("lp_aux", lp_aux),
            pl.Series("is_fold", (ye == fold_id).astype(np.int8)),
        ).with_columns(
            pl.when(pl.len().over("row") > 1)
            .then(pl.col("lp_aux") - pl.col("lp_aux").mean().over("row")).otherwise(0.0)
            .alias("d_cent"),
        )
        parts.append(
            scored.group_by("hand_id", "player_id", "opp_id").agg(
                pl.col("d_base").sum().alias("hi_base"),
                pl.col("d_cent").sum().alias("hi_cent"),
                pl.col("d_cent").max().alias("hi_cent_max"),
                pl.len().alias("hi_n"),
                (pl.col("d_base") * pl.col("is_fold")).sum().alias("hi_fold"),
                (pl.col("d_base") * pl.col("opp_is_responding_to")).sum().alias("hi_resp"),
            )
        )
        print(f"  chunk {c}: {e.height:,} scored rows ({time.time() - t0:.0f}s)", flush=True)
    hands = pl.concat(parts).sort("hand_id", "player_id", "opp_id")
    hands.write_parquet(HAND_OUT, compression="zstd")
    report["hand_rows"] = hands.height
    (C.ARTIFACTS / "hidden_info_policy.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {HAND_OUT}: {hands.height:,} rows ({time.time() - t0:.0f}s)")
    pair_features(hands)


def pair_features(hands: pl.DataFrame | None = None) -> None:
    if hands is None:
        hands = pl.read_parquet(HAND_OUT)
    meta = pl.scan_parquet(C.HANDS).select("hand_id", "table_id", "phase").collect()
    sums = ["hi_base", "hi_cent", "hi_fold", "hi_resp"]
    d = (
        hands.with_columns([pl.col(c).cast(pl.Float64) for c in sums])
        .join(meta, on="hand_id", how="inner")
        .with_columns((pl.col("hi_cent") > 2.0).cast(pl.Int32).alias("hi_hot"))
        .sort("phase", "table_id", "player_id", "opp_id", "hand_id")
        .group_by("phase", "table_id", "player_id", "opp_id", maintain_order=True)
        .agg(*[pl.col(c).sum() for c in sums], pl.col("hi_n").sum(), pl.col("hi_hot").sum(),
             pl.col("hi_cent").max().alias("hi_hand_max"), pl.len().alias("hi_hands"))
    )
    tot = d.group_by("phase", "table_id", "player_id", maintain_order=True).agg(
        *[pl.col(c).sum().alias(f"t_{c}") for c in [*sums, "hi_n", "hi_hot", "hi_hands"]])
    d = d.join(tot, on=["phase", "table_id", "player_id"], how="left")
    n = d["hi_n"].to_numpy().astype(np.float64)
    fn = (d["t_hi_n"] - d["hi_n"]).to_numpy().astype(np.float64)
    cols = {}
    for c in sums:
        own = d[c].to_numpy() / np.maximum(n, 1)
        field = (d[f"t_{c}"] - d[c]).to_numpy() / np.maximum(fn, 1)
        cols[f"{c}_rate"] = own
        cols[f"{c}_delta"] = own - field
        cols[f"{c}_excess"] = (own - field) * np.sqrt(np.minimum(n, 400.0))
    h = d["hi_hands"].to_numpy().astype(np.float64)
    fh = (d["t_hi_hands"] - d["hi_hands"]).to_numpy().astype(np.float64)
    hot_rate = d["hi_hot"].to_numpy() / np.maximum(h, 1)
    cols["hi_hot_rate"] = hot_rate
    cols["hi_hot_delta"] = hot_rate - (d["t_hi_hot"] - d["hi_hot"]).to_numpy() / np.maximum(fh, 1)
    d = d.with_columns([pl.Series(k, v) for k, v in cols.items()])
    feat = [*sums, "hi_hot", "hi_hand_max", *cols]
    keyed = d.select("phase", "table_id",
                     pl.min_horizontal("player_id", "opp_id").alias("p1"),
                     pl.max_horizontal("player_id", "opp_id").alias("p2"), *feat)
    folded = keyed.group_by("phase", "table_id", "p1", "p2", maintain_order=True).agg(
        *[pl.col(c).max().alias(f"{c}_hi") for c in feat],
        *[pl.col(c).min().alias(f"{c}_lo") for c in feat],
    ).sort("phase", "table_id", "p1", "p2")
    folded.write_parquet(PAIR_OUT, compression="zstd")
    print(f"wrote {PAIR_OUT}: {folded.height:,} pairs x {folded.width} cols")


def evaluate() -> None:
    from pokercol.cv import cleaned_mask, population_average_precision as ap, table_folds
    rt = importlib.import_module("14_pair_retest")
    exp = importlib.import_module("11_evidence_experiments")
    audit = importlib.import_module("181_evidence_oracle_audit")
    ev_mod = audit.ev_mod
    results: dict = {"pair": {}, "evidence": {}}

    dev = rt.load()
    x, feats = rt.matrix(dev)
    dev_hi = rt.load(("hidden_info_pair",))
    x_hi, feats_hi = rt.matrix(dev_hi)
    y = dev["y"].to_numpy()
    keep = cleaned_mask(dev)
    folds = table_folds(dev["table_id"].to_numpy(), 5)
    fam = dev["behavior_family"].fill_null("none").to_numpy()
    for name, xx in (("baseline", x), ("+ hidden info", x_hi)):
        rows = []
        for seed in (C.SEED, C.SEED + 1):
            oof = rt.lgb_oof(xx, y, folds, params={**rt.BASE, "seed": seed})
            r = {"cleaned": ap((y == 1)[keep], oof[keep])}
            for f in C.FAMILIES:
                m = keep & ((y != 1) | (fam == f))
                r[f] = ap((y == 1)[m], oof[m])
            rows.append(r)
        results["pair"][name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        results["pair"][name]["seeds"] = [round(r["cleaned"], 4) for r in rows]
        print(f"pair {name:16s} {results['pair'][name]}", flush=True)

    df, efeats = exp.build()
    hands = pl.read_parquet(HAND_OUT).drop("hi_n")
    hcols = [c for c in hands.columns if c.startswith("hi_")]
    for actor, other, tag in (("p1", "p2", "12"), ("p2", "p1", "21")):
        df = df.join(hands.rename({"player_id": actor, "opp_id": other, **{c: f"{c}_{tag}" for c in hcols}}),
                     on=["hand_id", actor, other], how="left")
    new = []
    for c in hcols:
        df = df.with_columns(
            pl.max_horizontal(f"{c}_12", f"{c}_21").alias(f"{c}_pmax"),
            (pl.col(f"{c}_12").fill_null(0) + pl.col(f"{c}_21").fill_null(0)).alias(f"{c}_psum"))
        new += [f"{c}_12", f"{c}_21", f"{c}_pmax", f"{c}_psum"]
    df = df.with_columns(
        pl.col("hi_cent_psum").rank("average", descending=True).over("pair_id").alias("hi_cent_rank"))
    new.append("hi_cent_rank")
    evidence = pl.read_csv(C.DEV_EVIDENCE).select("pair_id", "hand_id").unique()
    seed0 = ev_mod.PARAMS["seed"]
    for name, fs in (("baseline", efeats), ("+ hidden info", efeats + new)):
        vals = []
        for seed in (seed0, seed0 + 1):
            ev_mod.PARAMS["seed"] = seed
            vals.append(exp.score_map5(df, audit.oof_scores(df, fs), evidence))
        ev_mod.PARAMS["seed"] = seed0
        results["evidence"][name] = {"map5": float(np.mean(vals)), "seeds": [round(v, 4) for v in vals]}
        print(f"evidence {name:16s} {results['evidence'][name]}", flush=True)

    (C.ARTIFACTS / "hidden_info_eval.json").write_text(json.dumps(results, indent=2))


def production_check(extra: tuple[str, ...] = ("hidden_info_pair",),
                     out_name: str = "hidden_info_production") -> None:
    """The pair gain re-measured in the production setup: pseudo-labels chosen by
    each feature set's own nested model, global + family-model rank blend."""
    from pokercol.cv import cleaned_mask, population_average_precision as ap, table_folds
    rt = importlib.import_module("14_pair_retest")
    p13 = importlib.import_module("13_pseudo_labels")
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable") / v.size
    results = {}
    for name, tables in (("baseline", ()), ("+ " + "+".join(extra), extra)):
        dev = rt.load(tables)
        x, _ = rt.matrix(dev)
        y = dev["y"].to_numpy()
        fam = dev["behavior_family"].to_numpy()
        folds = table_folds(dev["table_id"].to_numpy(), 5)
        keep = cleaned_mask(dev)
        pseudo = {k: (y == -1) & (np.nan_to_num(p13.nested_scores(x, y, folds, k), nan=0.0) > 0.7)
                  for k in range(5)}
        rows = []
        for seed in (C.SEED, C.SEED + 1):
            params = {**rt.BASE, "seed": seed}
            glob = np.zeros(len(y))
            family_scores = {f: np.zeros(len(y)) for f in C.FAMILIES}
            for k in range(5):
                tr = folds != k
                target = (y == 1).astype(np.int8)
                weight = np.where(y == 1, 10.0, np.where(y == 0, 1.0, 0.3)).astype(float)
                target[tr & pseudo[k]] = 1
                weight[tr & pseudo[k]] = 3.0
                glob[~tr] = lgb.train(params, lgb.Dataset(x[tr], label=target[tr], weight=weight[tr]),
                                      700).predict(x[~tr])
                for f in C.FAMILIES:
                    tf = (fam == f).astype(np.int8)
                    wf = np.where(fam == f, 10.0, np.where(y == 1, 0.0, np.where(y == 0, 1.0, 0.3)))
                    uf = tr & (np.where(pseudo[k], 0.0, wf) > 0)
                    family_scores[f][~tr] = lgb.train(params, lgb.Dataset(x[uf], label=tf[uf], weight=wf[uf]),
                                                      700).predict(x[~tr])
            blend = 0.7 * rank(glob) + 0.3 * np.max(np.stack([rank(v) for v in family_scores.values()]), axis=0)
            row = {"cleaned": ap((y == 1)[keep], blend[keep])}
            for f in C.FAMILIES:
                m = keep & ((fam == f) | (y != 1))
                row[f] = ap((fam == f)[m], blend[m])
            rows.append(row)
        results[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        results[name]["seeds"] = [round(r["cleaned"], 4) for r in rows]
        print(f"production {name:14s} {results[name]}", flush=True)
    (C.ARTIFACTS / f"{out_name}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    if "--production" in sys.argv:
        production_check()
    if "--score" in sys.argv:
        score()
    if "--pairs" in sys.argv:
        pair_features()
    if "--eval" in sys.argv:
        evaluate()
