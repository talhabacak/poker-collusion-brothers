"""Stage 4e: relational action likelihood ratios - does a player act differently
against one opponent than against everyone else in the same spot?

Pipeline A's strongest single addition (+0.019 public). For every action and
every opponent still in the hand, the spot is

    street x own strength (weak/medium/strong) x ahead of that opponent
    x answering that opponent's aggression x heads-up

(96 contexts) and the action one of six classes. For an actor i and opponent j
the counts n_ij[c, a] are compared with the actor's counts against all other
opponents at the same table, n_i\\j[c, a], through a G statistic

    G_ij = 2 sum_c sum_a n_ij[c,a] log( p_ij(a|c) / p_i\\j(a|c) )

with Dirichlet smoothing on both distributions, its chi-square tail depth, and
signed rate deviations for folding, calling and aggression. Both directions of
a pair are kept, folded to hi/lo/sum over the pair.

Per (hand, actor, opponent) the summed log ratio and the actor's log-likelihood
under the "against others" distribution are written for the evidence ranker.

Writes relational_pair.parquet (pair features) and relational_hand.parquet.

With --blocks N the same statistic is computed inside N equal blocks of each
table's timeline (pipeline A's "time-local baseline": players drift and
coordination is episodic, so the comparison is against the same period) and
folded as sum / max over blocks; written to relational_pair_blocks.parquet.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C

import importlib
h25 = importlib.import_module("25_hidden_info_residual")

PAIR_OUT = C.CACHE / "relational_pair.parquet"
HAND_OUT = C.CACHE / "relational_hand.parquet"
N_CHUNKS = 16
ALPHA = 0.5
N_ACTIONS = 6


def contexts(e: pl.DataFrame) -> pl.DataFrame:
    """Context id and action class per (action, active opponent) row."""
    street = pl.col("street")
    own = (pl.when(street == 0)
           .then(pl.when(pl.col("preflop_equity_hu") < 0.45).then(0).when(pl.col("preflop_equity_hu") < 0.6).then(1).otherwise(2))
           .otherwise(pl.when(pl.col("hand_strength") // (14 ** 5) == 0).then(0)
                      .when(pl.col("hand_strength") // (14 ** 5) == 1).then(1).otherwise(2)))
    return e.select(
        "hand_id", "player_id", "opp_id", "action_no",
        (street * 24 + own * 8 + pl.col("opp_ahead").cast(pl.Int64) * 4
         + pl.col("opp_is_responding_to").cast(pl.Int64) * 2 + (pl.col("n_active_opps") == 1).cast(pl.Int64)).alias("ctx"),
        pl.col("y").cast(pl.Int64).alias("a"),
    )


def main() -> None:
    t0 = time.time()
    a, opp = h25.actions_with_strength()
    a = a.with_row_index("row")
    hand_idx = a["hand_idx"].to_numpy()
    bounds = np.searchsorted(hand_idx, np.linspace(1, hand_idx.max() + 1, N_CHUNKS + 1))
    rows = []
    for c in range(N_CHUNKS):
        lo, hi = int(bounds[c]), int(bounds[c + 1])
        rows.append(contexts(h25.expand(a[lo:hi], opp)))
        print(f"  chunk {c}: {rows[-1].height:,} rows ({time.time() - t0:.0f}s)", flush=True)
    r = pl.concat(rows)
    meta = pl.scan_parquet(C.HANDS).select("hand_id", "table_id", "phase", "started_at").collect()
    n_blocks = int(sys.argv[sys.argv.index("--blocks") + 1]) if "--blocks" in sys.argv else 0
    if n_blocks:
        meta = meta.sort("table_id", "started_at", "hand_id").with_columns(
            (pl.int_range(pl.len()).over("table_id") * n_blocks // pl.len().over("table_id")).alias("blk"))
        meta = meta.with_columns((pl.col("phase") + "|" + pl.col("blk").cast(pl.Utf8)).alias("phase"))
    r = r.join(meta.drop("started_at", *(["blk"] if n_blocks else [])), on="hand_id", how="inner")
    print(f"action x opponent rows: {r.height:,} ({time.time() - t0:.0f}s)", flush=True)

    # Counts against each opponent and against everyone at the table.
    n_ij = r.group_by("phase", "table_id", "player_id", "opp_id", "ctx", "a").len("n")
    n_i = n_ij.group_by("phase", "table_id", "player_id", "ctx", "a").agg(pl.col("n").sum().alias("n_all"))
    ctx_tot_ij = n_ij.group_by("phase", "table_id", "player_id", "opp_id", "ctx").agg(pl.col("n").sum().alias("t_ij"))
    ctx_tot_i = n_i.group_by("phase", "table_id", "player_id", "ctx").agg(pl.col("n_all").sum().alias("t_all"))
    cells = (n_ij.join(n_i, on=["phase", "table_id", "player_id", "ctx", "a"], how="left")
             .join(ctx_tot_ij, on=["phase", "table_id", "player_id", "opp_id", "ctx"], how="left")
             .join(ctx_tot_i, on=["phase", "table_id", "player_id", "ctx"], how="left")
             .with_columns(
                 ((pl.col("n") + ALPHA) / (pl.col("t_ij") + ALPHA * N_ACTIONS)).alias("p_ij"),
                 ((pl.col("n_all") - pl.col("n") + ALPHA) / (pl.col("t_all") - pl.col("t_ij") + ALPHA * N_ACTIONS)).alias("p_rest"),
             )
             .with_columns((pl.col("p_ij") / pl.col("p_rest")).log().alias("lr"),
                           (pl.col("p_ij") - pl.col("p_rest")).alias("dp")))
    # Per-cell table for models that read the whole context x action matrix.
    cells.select("phase", "table_id", "player_id", "opp_id", "ctx", "a", "n", "lr", "dp").write_parquet(
        C.CACHE / "relational_cells.parquet", compression="zstd")
    fold_id, call_id = 0, 2
    aggr = [3, 4, 5]
    # Context subsets where the families leave different marks: the actor ahead
    # of this opponent, answering this opponent's aggression, heads-up.
    ahead = ((pl.col("ctx") // 4) % 2) == 1
    facing = ((pl.col("ctx") // 2) % 2) == 1
    hu = (pl.col("ctx") % 2) == 1
    per = cells.group_by("phase", "table_id", "player_id", "opp_id").agg(
        (2 * (pl.col("n") * pl.col("lr")).sum()).alias("rel_g"),
        (2 * (pl.col("n") * pl.col("lr")).filter(ahead).sum()).alias("rel_g_ahead"),
        (2 * (pl.col("n") * pl.col("lr")).filter(~ahead).sum()).alias("rel_g_behind"),
        (2 * (pl.col("n") * pl.col("lr")).filter(facing).sum()).alias("rel_g_facing"),
        (2 * (pl.col("n") * pl.col("lr")).filter(hu).sum()).alias("rel_g_hu"),
        ((pl.col("n") * pl.col("dp")).filter((pl.col("a") == 0) & ahead).sum() / pl.col("n").filter(ahead).sum().clip(lower_bound=1)).alias("rel_fold_ahead_dev"),
        ((pl.col("n") * pl.col("dp")).filter((pl.col("a") == 2) & ~ahead).sum() / pl.col("n").filter(~ahead).sum().clip(lower_bound=1)).alias("rel_call_behind_dev"),
        ((pl.col("n") * pl.col("dp")).filter(pl.col("a").is_in([3, 4, 5]) & facing).sum() / pl.col("n").filter(facing).sum().clip(lower_bound=1)).alias("rel_aggr_facing_dev"),
        pl.col("n").sum().alias("rel_n"),
        pl.col("ctx").n_unique().alias("rel_ctx"),
        ((pl.col("n") * pl.col("dp")).filter(pl.col("a") == fold_id).sum() / pl.col("n").sum()).alias("rel_fold_dev"),
        ((pl.col("n") * pl.col("dp")).filter(pl.col("a") == call_id).sum() / pl.col("n").sum()).alias("rel_call_dev"),
        ((pl.col("n") * pl.col("dp")).filter(pl.col("a").is_in(aggr)).sum() / pl.col("n").sum()).alias("rel_aggr_dev"),
        (pl.col("n") * pl.col("lr")).max().alias("rel_cell_max"),
    )
    g = per["rel_g"].to_numpy().astype(float)
    df_ = (per["rel_ctx"].to_numpy() * (N_ACTIONS - 1)).astype(float)
    tail = -stats.chi2.logsf(np.maximum(g, 0), df_) / np.log(10)
    per = per.with_columns(pl.Series("rel_g_tail", tail), (pl.col("rel_g") / pl.col("rel_n").clip(lower_bound=1)).alias("rel_g_per_action"))
    feat = ["rel_g", "rel_g_tail", "rel_g_per_action", "rel_n", "rel_fold_dev", "rel_call_dev", "rel_aggr_dev", "rel_cell_max",
            "rel_g_ahead", "rel_g_behind", "rel_g_facing", "rel_g_hu",
            "rel_fold_ahead_dev", "rel_call_behind_dev", "rel_aggr_facing_dev"]
    keyed = per.select("phase", "table_id", pl.min_horizontal("player_id", "opp_id").alias("p1"),
                       pl.max_horizontal("player_id", "opp_id").alias("p2"), *feat)
    pair = keyed.group_by("phase", "table_id", "p1", "p2").agg(
        *[pl.col(f).max().alias(f"{f}_hi") for f in feat],
        *[pl.col(f).min().alias(f"{f}_lo") for f in feat],
        pl.col("rel_g_tail").sum().alias("rel_g_tail_sum"),
    ).sort("phase", "table_id", "p1", "p2")
    if n_blocks:
        cols = [c for c in pair.columns if c.startswith("rel_")]
        pair = (pair.with_columns(pl.col("phase").str.split("|").list.first().alias("phase"))
                .group_by("phase", "table_id", "p1", "p2")
                .agg(*[pl.col(c).sum().alias(f"blk_{c}_sum") for c in cols], *[pl.col(c).max().alias(f"blk_{c}_max") for c in cols],
                     pl.len().alias("blk_n"))
                .sort("phase", "table_id", "p1", "p2"))
        out = C.CACHE / "relational_pair_blocks.parquet"
        pair.write_parquet(out, compression="zstd")
        print(f"wrote {out}: {pair.height:,} pairs x {pair.width} cols ({time.time() - t0:.0f}s)")
        return
    pair.write_parquet(PAIR_OUT, compression="zstd")
    print(f"wrote {PAIR_OUT}: {pair.height:,} pairs x {pair.width} cols ({time.time() - t0:.0f}s)", flush=True)

    # Hand-level: the actor's actions in this hand scored under the two distributions.
    hand = (r.join(cells.select("phase", "table_id", "player_id", "opp_id", "ctx", "a", "lr", "p_rest"),
                   on=["phase", "table_id", "player_id", "opp_id", "ctx", "a"], how="left")
            .group_by("hand_id", "player_id", "opp_id").agg(pl.col("lr").sum().alias("rel_lr"),
                                                            (-pl.col("p_rest").log()).sum().alias("rel_rest_nll"),
                                                            pl.len().alias("rel_acts"))
            .sort("hand_id", "player_id", "opp_id"))
    hand.write_parquet(HAND_OUT, compression="zstd")
    print(f"wrote {HAND_OUT}: {hand.height:,} rows ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
