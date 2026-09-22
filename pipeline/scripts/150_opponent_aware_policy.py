"""Stage 150: the relational statistic, re-estimated by a model that knows what
kind of opponent the actor is facing.

Stage 4e is the strongest single feature the project has, and it has two limits
that point the same way.

It is estimated by counting. A pair supplies about a hundred shared hands spread
over 96 discrete cells, so most cells hold a handful of actions; raising the
resolution to 768 cells (stage 63) moved the board 0.0000, because the finer
partition gave back in thinner counts whatever it bought in context. A model
over a continuous state has no such trade-off: every action informs every
prediction, and the context can be as fine as the features allow.

And its baseline conflates opponent type with opponent identity. Stage 4e asks
whether the actor plays differently against *this* opponent than against the
others at the table - but a player legitimately plays differently against a
passive opponent than against an aggressive one, so part of every measured
deviation is only "this opponent is not like the others", which is not
collusion. Conditioning the expectation on the opponent's *style* removes that
component and should leave a sharper partner-specific residual.

So: fit P(action | state, actor style, opponent style) on the whole action log,
one row per (action, opponent still in the hand), and ask per directed pair how
far the actor's action mix against that opponent departed from what the model
expected of anyone in those spots facing that kind of opponent. Two readings of
the departure are kept, because they fail differently:

    likelihood ratio     G = 2 N sum_a p_obs(a) log( p_obs(a) / p_model(a) )
                         with Dirichlet smoothing on both, and its chi-square
                         tail depth - the model-based twin of stage 4e's G.
    calibrated residual  (O_a - sum_k P_k(a)) / sqrt(sum_k P_k(a)(1 - P_k(a))),
                         which uses the model's own variance instead of a
                         multinomial approximation, so it is honest about how
                         much of the deviation a run of easy spots could explain.

Both directions i->j and j->i are kept and folded to hi/lo over the pair, on the
same keys stage 4e writes, so the columns drop straight into the pair table.

No label of any kind enters this model. The target is the action that was taken,
the features come from the raw logs and from per-player rates over
`seat_features.parquet`, and the training rows are a random sample of all 18.6M
actions across both phases. Nothing here reads `development_labels.csv` or
`development_evidence.csv`, so there is no fold to be safe with respect to: the
stage is a deterministic transform of the public logs, exactly like stages 2b
and 4e that already ship.

    python scripts/150_opponent_aware_policy.py          # ~45 min, writes caches

Writes oppaware_pair.parquet / oppaware_hand.parquet and the ablation pair that
drops the opponent-style block, oppaware_pair_nostyle.parquet /
oppaware_hand_nostyle.parquet.
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C

pol = importlib.import_module("02b_action_surprise")
h25 = importlib.import_module("25_hidden_info_residual")

PARTS = C.CACHE / "_oppaware_parts"
REPORT = C.ARTIFACTS / "oppaware_policy.json"
N_CHUNKS = 24
ALPHA = 0.5
N_ACTIONS = 6
AGGRESSIVE = (3, 4, 5)
FOLD, CALL = 0, 2

# The same draw stage 25 used, so the plain-feature model below reproduces that
# stage's 0.4194 exactly and the opponent block is measured against a number
# already on record rather than against a fresh baseline of its own.
TRAIN_ACTIONS = 1_600_000
HOLDOUT_ACTIONS = 300_000
SAMPLE_SEED = C.SEED + 25

# Opponent-specific state the actor can actually see. The opponent's cards are
# deliberately absent: stage 25 measured what they are worth (pair +0.0002,
# evidence -0.006) and a policy that reads them answers a different question.
CTX = ["opp_is_responding_to", "opp_rel_position", "n_active_opps"]
STYLE = ["style_vpip", "style_pfr", "style_aggression", "style_fold_to_bet",
         "style_showdown", "style_postflop_aggression"]
OPP_STYLE = [f"opp_{c}" for c in STYLE]

FEATURES = ["oa_g", "oa_g_tail", "oa_g_per_action", "oa_n",
            "oa_fold_dev", "oa_call_dev", "oa_aggr_dev",
            "oa_fold_z", "oa_call_z", "oa_aggr_z",
            "oa_surprise", "oa_surprise_excess",
            "oa_n_facing", "oa_fold_z_facing", "oa_aggr_z_facing", "oa_aggr_dev_facing"]

VARIANTS = ("full", "nostyle")

# `--quick` exercises the whole path on a fraction of the data, under its own
# file names, so a mistake shows up in two minutes instead of forty.
QUICK = "--quick" in sys.argv
SUFFIX = "_quick" if QUICK else ""


def opponent_style() -> pl.DataFrame:
    """The same six rates stage 2b computes for the actor, keyed to the opponent."""
    s = pol.player_style().collect()
    return s.rename({c: f"opp_{c}" for c in STYLE}).rename({"player_id": "opp_id"})


def prepare() -> tuple[pl.DataFrame, pl.DataFrame]:
    a, opp = h25.actions_with_strength()
    a = a.with_row_index("row")
    meta = pl.scan_parquet(C.HANDS).select("hand_id", "table_id", "phase").collect()
    # `expand` carries every column of `opp` through, so the opponent's phase,
    # table and style ride along with the seat instead of costing a join on the
    # expanded frame.
    opp = (opp.join(meta.join(a.select("hand_id", "hand_idx").unique(), on="hand_id", how="inner")
                    .select("hand_idx", "table_id", "phase"), on="hand_idx", how="left")
           .join(opponent_style(), on=["opp_id", "phase"], how="left"))
    return a, opp


def directed_index() -> tuple[pl.DataFrame, dict]:
    """Every co-seated ordered pair, and its row in the accumulator arrays.

    The universe is the same one the pair table is built on, so a directed group
    that never fires simply stays at zero and is dropped at the end.
    """
    u = pl.read_parquet(C.CACHE / "pair_universe.parquet").select("phase", "table_id", "p1", "p2")
    both = pl.concat([
        u.select("phase", "table_id", pl.col("p1").alias("player_id"), pl.col("p2").alias("opp_id")),
        u.select("phase", "table_id", pl.col("p2").alias("player_id"), pl.col("p1").alias("opp_id")),
    ]).unique().sort("phase", "table_id", "player_id", "opp_id").with_row_index("gidx")
    return both, {}


class Accumulator:
    """Additive sufficient statistics per directed pair, filled chunk by chunk."""

    def __init__(self, n_groups: int):
        z = lambda k=1: np.zeros((n_groups, k) if k > 1 else n_groups)
        self.n = z()
        self.obs = z(N_ACTIONS)
        self.exp = z(N_ACTIONS)
        self.var = z(3)          # fold, call, aggression
        self.nll = z()
        self.ent = z()
        self.n_f = z()
        self.obs_f = z(2)        # fold, aggression, answering this opponent
        self.exp_f = z(2)
        self.var_f = z(2)

    def add(self, gidx: np.ndarray, p: np.ndarray, y: np.ndarray, facing: np.ndarray,
            m: int) -> None:
        bc = lambda w: np.bincount(gidx, weights=w, minlength=m)
        self.n += bc(None)
        for a in range(N_ACTIONS):
            self.obs[:, a] += bc((y == a).astype(np.float64))
            self.exp[:, a] += bc(p[:, a])
        aggr = p[:, AGGRESSIVE].sum(axis=1)
        for j, q in enumerate((p[:, FOLD], p[:, CALL], aggr)):
            self.var[:, j] += bc(q * (1.0 - q))
        self.nll += bc(-np.log(np.clip(p[np.arange(y.size), y], 1e-9, 1.0)))
        self.ent += bc(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1))
        f = facing.astype(np.float64)
        self.n_f += bc(f)
        for j, (o, q) in enumerate(((y == FOLD, p[:, FOLD]), (np.isin(y, AGGRESSIVE), aggr))):
            self.obs_f[:, j] += bc(o.astype(np.float64) * f)
            self.exp_f[:, j] += bc(q * f)
            self.var_f[:, j] += bc(q * (1.0 - q) * f)


def directed_features(acc: Accumulator, keys: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    """Per-direction columns, and the log tilt vector the hand table needs."""
    live = acc.n > 0
    n = np.maximum(acc.n, 1.0)
    denom = (acc.n + ALPHA * N_ACTIONS)[:, None]
    p_obs = (acc.obs + ALPHA) / denom
    p_mod = (acc.exp + ALPHA) / denom
    log_t = np.log(p_obs / p_mod)
    g = 2.0 * acc.n * (p_obs * log_t).sum(axis=1)
    tail = -stats.chi2.logsf(np.maximum(g, 0.0), N_ACTIONS - 1) / np.log(10.0)

    aggr_obs = acc.obs[:, AGGRESSIVE].sum(axis=1)
    aggr_exp = acc.exp[:, AGGRESSIVE].sum(axis=1)
    z = lambda o, e, v: (o - e) / np.sqrt(np.maximum(v, 1.0))
    nf = np.maximum(acc.n_f, 1.0)
    cols = {
        "oa_g": g,
        "oa_g_tail": tail,
        "oa_g_per_action": g / n,
        "oa_n": acc.n,
        "oa_fold_dev": (acc.obs[:, FOLD] - acc.exp[:, FOLD]) / n,
        "oa_call_dev": (acc.obs[:, CALL] - acc.exp[:, CALL]) / n,
        "oa_aggr_dev": (aggr_obs - aggr_exp) / n,
        "oa_fold_z": z(acc.obs[:, FOLD], acc.exp[:, FOLD], acc.var[:, 0]),
        "oa_call_z": z(acc.obs[:, CALL], acc.exp[:, CALL], acc.var[:, 1]),
        "oa_aggr_z": z(aggr_obs, aggr_exp, acc.var[:, 2]),
        "oa_surprise": acc.nll / n,
        "oa_surprise_excess": (acc.nll - acc.ent) / n,
        "oa_n_facing": acc.n_f,
        "oa_fold_z_facing": z(acc.obs_f[:, 0], acc.exp_f[:, 0], acc.var_f[:, 0]),
        "oa_aggr_z_facing": z(acc.obs_f[:, 1], acc.exp_f[:, 1], acc.var_f[:, 1]),
        "oa_aggr_dev_facing": (acc.obs_f[:, 1] - acc.exp_f[:, 1]) / nf,
    }
    frame = keys.with_columns([pl.Series(k, v) for k, v in cols.items()]).filter(pl.Series(live))
    return frame, log_t


def fold_to_pairs(per: pl.DataFrame) -> pl.DataFrame:
    """Stage 4e's output contract: hi/lo over the two directions of the pair."""
    keyed = per.select("phase", "table_id",
                       pl.min_horizontal("player_id", "opp_id").alias("p1"),
                       pl.max_horizontal("player_id", "opp_id").alias("p2"), *FEATURES)
    return keyed.group_by("phase", "table_id", "p1", "p2").agg(
        *[pl.col(f).max().alias(f"{f}_hi") for f in FEATURES],
        *[pl.col(f).min().alias(f"{f}_lo") for f in FEATURES],
        pl.col("oa_g_tail").sum().alias("oa_g_tail_sum"),
    ).sort("phase", "table_id", "p1", "p2")


def hand_table(variant: str, log_t: np.ndarray) -> None:
    """Per (hand, actor, opponent): the tilt the pair carries, and the model's surprise."""
    parts = sorted((PARTS / variant).glob("*.parquet"))
    lt = pl.DataFrame({"gidx": np.arange(log_t.shape[0], dtype=np.uint32),
                       **{f"lt{a}": log_t[:, a] for a in range(N_ACTIONS)}})
    out = C.CACHE / f"oppaware_hand{'' if variant == 'full' else '_nostyle'}{SUFFIX}.parquet"
    frame = (pl.scan_parquet(parts)
     .join(lt.lazy(), on="gidx", how="left")
     .select(
         "hand_id", "player_id", "opp_id",
         pl.sum_horizontal([pl.col(f"o{a}") * pl.col(f"lt{a}") for a in range(N_ACTIONS)]).alias("oa_lr"),
         pl.col("nll").alias("oa_nll"),
         (pl.col("nll") - pl.col("ent")).alias("oa_excess"),
         pl.col("res_fold").alias("oa_res_fold"),
         pl.col("res_aggr").alias("oa_res_aggr"),
         pl.col("acts").alias("oa_acts"),
     )
     .sort("hand_id", "player_id", "opp_id")
     .collect(engine="streaming"))
    frame.write_parquet(out, compression="zstd")
    print(f"wrote {out}: {frame.height:,} rows", flush=True)


def main() -> None:
    global TRAIN_ACTIONS, HOLDOUT_ACTIONS
    if QUICK:
        TRAIN_ACTIONS, HOLDOUT_ACTIONS = 200_000, 50_000
    t0 = time.time()
    a, opp = prepare()
    x_base = a.select(pol.FEATURES).to_numpy().astype(np.float32)
    print(f"actions {a.height:,}, opponent seats {opp.height:,} ({time.time() - t0:.0f}s)", flush=True)

    order = np.random.default_rng(SAMPLE_SEED).permutation(a.height)
    train_rows = np.sort(order[:TRAIN_ACTIONS])
    hold_rows = np.sort(order[TRAIN_ACTIONS:TRAIN_ACTIONS + HOLDOUT_ACTIONS])
    e_tr = h25.expand(a[train_rows], opp)
    e_ho = h25.expand(a[hold_rows], opp)
    y_tr, y_ho = e_tr["y"].to_numpy(), e_ho["y"].to_numpy()
    print(f"training rows {e_tr.height:,}, holdout rows {e_ho.height:,} "
          f"({time.time() - t0:.0f}s)", flush=True)

    def block(e: pl.DataFrame, cols: list[str]) -> np.ndarray:
        xb = x_base[e["row"].to_numpy()]
        if not cols:
            return xb
        return np.hstack([xb, e.select(cols).to_numpy().astype(np.float32)])

    ll = lambda p, y: float(-np.log(np.clip(p[np.arange(y.size), y], 1e-9, 1)).mean())
    specs = {"plain": [], "nostyle": CTX, "full": CTX + OPP_STYLE}
    models, report = {}, {"holdout_logloss": {}, "features": {k: len(pol.FEATURES) + len(v)
                                                             for k, v in specs.items()}}
    for name, extra in specs.items():
        booster = lgb.train(pol.PARAMS, lgb.Dataset(block(e_tr, extra), label=y_tr),
                            num_boost_round=pol.N_ROUNDS)
        report["holdout_logloss"][name] = round(ll(booster.predict(block(e_ho, extra)), y_ho), 4)
        print(f"  {name:8s} holdout log-loss {report['holdout_logloss'][name]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if name in VARIANTS:
            models[name] = booster
            imp = dict(zip(pol.FEATURES + extra, booster.feature_importance("gain").tolist()))
            report[f"gain_{name}"] = {k: round(imp[k] / sum(imp.values()), 4)
                                      for k in (OPP_STYLE + CTX) if k in imp}
    del e_tr, e_ho

    keys, _ = directed_index()
    m = keys.height
    accs = {v: Accumulator(m) for v in VARIANTS}
    if PARTS.exists():
        shutil.rmtree(PARTS)
    for v in VARIANTS:
        (PARTS / v).mkdir(parents=True)

    hand_idx = a["hand_idx"].to_numpy()
    bounds = np.searchsorted(hand_idx, np.linspace(1, hand_idx.max() + 1, N_CHUNKS + 1))
    key_index = keys.select("phase", "table_id", "player_id", "opp_id", "gidx")
    total = 0
    for c in range(2 if QUICK else N_CHUNKS):
        lo, hi = int(bounds[c]), int(bounds[c + 1])
        e = h25.expand(a[lo:hi], opp).join(key_index, on=["phase", "table_id", "player_id", "opp_id"],
                                           how="left")
        assert e["gidx"].null_count() == 0, "a directed pair is outside the co-seating universe"
        gidx = e["gidx"].to_numpy().astype(np.int64)
        y = e["y"].to_numpy()
        facing = e["opp_is_responding_to"].to_numpy().astype(bool)
        xb = x_base[e["row"].to_numpy()]
        ctx = e.select(CTX).to_numpy().astype(np.float32)
        sty = e.select(OPP_STYLE).to_numpy().astype(np.float32)
        base = e.select("hand_id", "player_id", "opp_id").with_columns(
            pl.Series("gidx", gidx.astype(np.uint32)))
        for v in VARIANTS:
            x = np.hstack([xb, ctx]) if v == "nostyle" else np.hstack([xb, ctx, sty])
            p = models[v].predict(x)
            del x
            accs[v].add(gidx, p, y, facing, m)
            aggr = p[:, AGGRESSIVE].sum(axis=1)
            rows = base.with_columns(
                *[pl.Series(f"o{k}", (y == k).astype(np.uint8)) for k in range(N_ACTIONS)],
                pl.Series("nll", -np.log(np.clip(p[np.arange(y.size), y], 1e-9, 1.0)), dtype=pl.Float32),
                pl.Series("ent", -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1), dtype=pl.Float32),
                pl.Series("res_fold", (y == FOLD).astype(np.float64) - p[:, FOLD], dtype=pl.Float32),
                pl.Series("res_aggr", np.isin(y, AGGRESSIVE).astype(np.float64) - aggr, dtype=pl.Float32),
            )
            del p
            (rows.group_by("hand_id", "player_id", "opp_id", "gidx")
             .agg(*[pl.col(f"o{k}").sum().cast(pl.UInt8) for k in range(N_ACTIONS)],
                  pl.col("nll").sum(), pl.col("ent").sum(),
                  pl.col("res_fold").sum(), pl.col("res_aggr").sum(),
                  pl.len().cast(pl.UInt8).alias("acts"))
             .write_parquet(PARTS / v / f"{c:02d}.parquet", compression="zstd"))
            del rows
        total += e.height
        print(f"  chunk {c}: {e.height:,} action x opponent rows ({time.time() - t0:.0f}s)", flush=True)
        del e, xb, ctx, sty, base
    report["action_opponent_rows"] = total
    del x_base, a, opp

    for v in VARIANTS:
        per, log_t = directed_features(accs[v], keys)
        pair = fold_to_pairs(per)
        out = C.CACHE / f"oppaware_pair{'' if v == 'full' else '_nostyle'}{SUFFIX}.parquet"
        pair.write_parquet(out, compression="zstd")
        report[f"pairs_{v}"] = pair.height
        print(f"wrote {out}: {pair.height:,} pairs x {pair.width} cols "
              f"({time.time() - t0:.0f}s)", flush=True)
        hand_table(v, log_t)
    shutil.rmtree(PARTS)
    report["seconds"] = round(time.time() - t0)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
