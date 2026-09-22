"""Stage 4: pair-level features, built around partner-vs-field surprise.

The central statistic is not a rate but an improbability. For a directed event
(player X transfers value to player Y) we know X's rate against everyone else at
the table, `p`, the number of hands X shared with Y, `n`, and the observed count
`k`. The feature is then

    surprise = -log10 P(Binomial(n, p) >= k)

which answers "how unlikely is this much of it, against this one opponent, for a
player who behaves like this in general". A rate difference cannot distinguish
3-of-40 from 30-of-400; a tail probability can, and it automatically discounts
small samples instead of needing a separate shrinkage term.

Symmetric aggregates and empirical-Bayes rate contrasts are kept alongside it,
because donor/receiver structure (one-sided) and mutual soft play (two-sided)
leave different fingerprints.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C

PAIR_HANDS = C.CACHE / "pair_hands"
OUT = C.CACHE / "pair_features.parquet"

EB_LAMBDA = 25.0
EPS = 1e-9

# Directed events: counted as X -> Y and compared against X's own field rate.
DIRECTED = ["fold_better", "paid", "transfer_event", "folded_to_partner", "outsider_folds_to",
            "ahead_fold"]

# Symmetric rates contrasted against each player's field baseline.
CONTRAST_RATES = [
    "both_showdown_rate", "mutual_passive_rate", "pair_heads_up_rate",
    "both_in_pot_rate", "transfer_gross_mean", "strong_check_rate",
    "outsiders_folded_to_pair_mean", "relay_rate", "relay_cleared_rate",
    "both_aggressive_rate", "pair_net_mean",
    "both_raised_pf_rate", "squeeze_rate", "weak_pf_raise_mean", "aggressive_loss_rate",
]


def aggregate_pairs(lf: pl.LazyFrame | None = None) -> pl.DataFrame:
    """Aggregate pair-hand rows to one row per pair. `lf` defaults to every
    pair-hand; stage 23 passes cropped subsets through the same aggregation."""
    if lf is None:
        lf = pl.scan_parquet(PAIR_HANDS / "*.parquet")

    t = pl.col("transfer_gross")
    transfer_12 = pl.col("fold_better_1_to_2") | pl.col("paid_1_to_2")
    transfer_21 = pl.col("fold_better_2_to_1") | pl.col("paid_2_to_1")
    paid_any = pl.col("paid_1_to_2") | pl.col("paid_2_to_1")
    fold_better_any = pl.col("fold_better_1_to_2") | pl.col("fold_better_2_to_1")
    transfer_any = transfer_12 | transfer_21
    strong_check_1 = (pl.col("pair_heads_up") & (pl.col("strength_rank_1") == 1)
                      & (pl.col("postflop_aggressive_1") == 0)).fill_null(False)
    strong_check_2 = (pl.col("pair_heads_up") & (pl.col("strength_rank_2") == 1)
                      & (pl.col("postflop_aggressive_2") == 0)).fill_null(False)
    sd_loss = pl.col("both_showdown") & (pl.col("net_bb_1") * pl.col("net_bb_2") < 0)
    pair_surprise = pl.col("surprise_sum_1") + pl.col("surprise_sum_2")
    pair_fold_surprise = pl.col("fold_surprise_1") + pl.col("fold_surprise_2")

    agg = lf.group_by("phase", "table_id", "p1", "p2", maintain_order=True).agg(
        pl.len().alias("shared_hands"),
        # Directed event counts, consumed by the surprise stage.
        transfer_12.sum().alias("k_transfer_event_12"),
        transfer_21.sum().alias("k_transfer_event_21"),
        pl.col("fold_better_1_to_2").sum().alias("k_fold_better_12"),
        pl.col("fold_better_2_to_1").sum().alias("k_fold_better_21"),
        pl.col("paid_1_to_2").sum().alias("k_paid_12"),
        pl.col("paid_2_to_1").sum().alias("k_paid_21"),
        pl.col("p1_folded_to_p2").sum().alias("k_folded_to_partner_12"),
        pl.col("p2_folded_to_p1").sum().alias("k_folded_to_partner_21"),
        pl.col("outsiders_folded_to_p1").sum().alias("k_outsider_folds_to_12"),
        pl.col("outsiders_folded_to_p2").sum().alias("k_outsider_folds_to_21"),
        pl.col("ahead_fold_1_to_2").sum().alias("k_ahead_fold_12"),
        pl.col("ahead_fold_2_to_1").sum().alias("k_ahead_fold_21"),
        (pl.col("ahead_fold_preflop_1_to_2") | pl.col("ahead_fold_preflop_2_to_1"))
        .mean().alias("ahead_fold_preflop_rate"),
        (pl.col("ahead_fold_1_to_2") | pl.col("ahead_fold_2_to_1")).mean().alias("ahead_fold_rate"),
        pl.col("ahead_fold_equity_gap").sum().alias("ahead_fold_equity_sum"),
        # Value flow.
        pl.col("transfer_1_to_2").sum().alias("flow_12"),
        pl.col("transfer_2_to_1").sum().alias("flow_21"),
        t.mean().alias("transfer_gross_mean"),
        t.max().alias("transfer_max"),
        t.sum().alias("transfer_total"),
        t.top_k(3).mean().alias("transfer_top3"),
        t.top_k(5).mean().alias("transfer_top5"),
        t.quantile(0.95).alias("transfer_p95"),
        (t > 0).mean().alias("transfer_any_rate"),
        # Strength-conditioned movement.
        transfer_any.mean().alias("transfer_event_rate"),
        transfer_any.sum().alias("transfer_event_n"),
        paid_any.mean().alias("paid_any"),
        fold_better_any.mean().alias("fold_better_any"),
        (pl.col("conceded_1_to_2") | pl.col("conceded_2_to_1")).mean().alias("conceded_any"),
        pl.when(transfer_any).then(t).otherwise(None).mean().alias("transfer_event_size"),
        pl.when(transfer_any).then(t).otherwise(None).sum().alias("transfer_event_value"),
        pl.when(transfer_any).then(pl.col("pot_bb")).otherwise(None).mean().alias("transfer_event_pot"),
        # Per-hand implausibility, summarised by its tail. A pair that played one
        # genuinely inexplicable hand matters more than one that was mildly odd
        # throughout, so max and top-k carry more than the mean.
        pair_surprise.max().alias("ps_max"),
        pair_surprise.top_k(3).mean().alias("ps_top3"),
        pair_surprise.top_k(5).mean().alias("ps_top5"),
        pair_surprise.mean().alias("ps_mean"),
        pair_surprise.sum().alias("ps_sum"),
        pair_fold_surprise.max().alias("pfs_max"),
        pair_fold_surprise.top_k(3).mean().alias("pfs_top3"),
        pair_fold_surprise.sum().alias("pfs_sum"),
        (pl.col("surprise_sum_1") - pl.col("surprise_sum_2")).abs().max().alias("ps_imbalance_max"),
        (pl.col("aggression_excess_1") + pl.col("aggression_excess_2")).sum().alias("ax_pair_sum"),
        # Implausibility restricted to the hands where value actually moved.
        pl.when(transfer_any).then(pair_surprise).otherwise(None).mean().alias("ps_on_events"),
        pl.when(transfer_any).then(pair_surprise).otherwise(0.0).max().alias("ps_on_events_max"),
        # Contact and passivity.
        pl.col("both_showdown").mean().alias("both_showdown_rate"),
        pl.col("both_in_pot").mean().alias("both_in_pot_rate"),
        pl.col("pair_heads_up").mean().alias("pair_heads_up_rate"),
        pl.col("mutual_passive").mean().alias("mutual_passive_rate"),
        pl.col("one_folded").mean().alias("one_folded_rate"),
        (strong_check_1 | strong_check_2).mean().alias("strong_check_rate"),
        sd_loss.mean().alias("showdown_loss_rate"),
        pl.col("pair_aggressive").mean().alias("pair_aggressive_mean"),
        pl.col("contribution_gap").abs().mean().alias("contribution_gap_mean"),
        pl.col("net_gap").abs().mean().alias("net_gap_mean"),
        # Isolation: preflop double-raises, the sandwich on an outsider, and how
        # little the cards justified them.
        pl.col("both_raised_preflop").mean().alias("both_raised_pf_rate"),
        pl.col("both_raised_preflop").sum().alias("both_raised_pf_n"),
        pl.col("squeeze_sandwich").mean().alias("squeeze_rate"),
        pl.col("squeeze_sandwich").sum().alias("squeeze_n"),
        pl.col("weak_preflop_raise_weight").mean().alias("weak_pf_raise_mean"),
        pl.col("weak_preflop_raise_weight").top_k(5).mean().alias("weak_pf_raise_top5"),
        (1.0 - pl.col("weaker_raiser_strength")).max().alias("weakest_double_raise"),
        pl.col("pair_aggressive_loss").mean().alias("aggressive_loss_rate"),
        # Isolation: the relay motif, and whether it actually cleared the table.
        pl.col("consecutive_aggression").mean().alias("relay_rate"),
        pl.col("consecutive_aggression").sum().alias("relay_n"),
        pl.col("relay_cleared_outsiders").mean().alias("relay_cleared_rate"),
        pl.col("relay_cleared_outsiders").sum().alias("relay_cleared_n"),
        pl.col("both_aggressive").mean().alias("both_aggressive_rate"),
        pl.col("pair_net_bb").mean().alias("pair_net_mean"),
        pl.col("pair_net_bb").sum().alias("pair_net_total"),
        pl.col("outsider_net_bb").mean().alias("outsider_net_mean"),
        pl.when(pl.col("consecutive_aggression")).then(pl.col("pair_net_bb"))
        .otherwise(None).mean().alias("pair_net_when_relay"),
        # Magnitude of concession, not just its frequency. Soft play is rare per
        # pair but enormous per hand, so the tail matters more than the mean.
        pl.col("concession_weight_1").max().alias("concession_max_1"),
        pl.col("concession_weight_2").max().alias("concession_max_2"),
        pl.col("concession_weight_1").sum().alias("concession_sum_1"),
        pl.col("concession_weight_2").sum().alias("concession_sum_2"),
        pl.col("concession_weight_1").top_k(3).mean().alias("concession_top3_1"),
        pl.col("concession_weight_2").top_k(3).mean().alias("concession_top3_2"),
        pl.col("fold_odds_declined_1").max().alias("fold_odds_max_1"),
        pl.col("fold_odds_declined_2").max().alias("fold_odds_max_2"),
        # Biggest single value movement against hand strength, either direction.
        pl.when(transfer_any).then(t).otherwise(0.0).max().alias("transfer_event_max"),
        pl.when(transfer_any).then(t).otherwise(0.0).top_k(3).mean().alias("transfer_event_top3"),
        # Isolation.
        pl.col("outsiders_folded_to_pair").mean().alias("outsiders_folded_to_pair_mean"),
        pl.col("outsiders_folded_to_pair").sum().alias("outsiders_folded_to_pair_n"),
        pl.when(pl.col("both_in_pot")).then(pl.col("outsiders_folded_to_pair"))
        .otherwise(None).mean().alias("outsiders_folded_when_both_in"),
        pl.col("outsider_folds").mean().alias("outsider_folds_mean"),
        pl.col("outsider_aggressive").mean().alias("outsider_aggressive_mean"),
        pl.col("outsider_contribution_bb").mean().alias("outsider_contribution_mean"),
        # Pot context.
        pl.col("pot_bb").mean().alias("pot_bb_mean"),
        pl.when(t > 0).then(pl.col("pot_bb")).otherwise(None).mean().alias("pot_bb_when_transfer"),
        # Episodic concentration.
        pl.col("started_at").min().alias("first_seen"),
        pl.col("started_at").max().alias("last_seen"),
        pl.when(transfer_any).then(pl.col("started_at")).otherwise(None).min().alias("first_event"),
        pl.when(transfer_any).then(pl.col("started_at")).otherwise(None).max().alias("last_event"),
    )

    flow_hi = pl.max_horizontal("flow_12", "flow_21")
    flow_lo = pl.min_horizontal("flow_12", "flow_21")
    tr_hi = pl.max_horizontal("k_transfer_event_12", "k_transfer_event_21")
    tr_lo = pl.min_horizontal("k_transfer_event_12", "k_transfer_event_21")
    conc_hi = pl.max_horizontal("concession_max_1", "concession_max_2")
    conc_sum_hi = pl.max_horizontal("concession_sum_1", "concession_sum_2")

    return agg.with_columns(
        flow_hi.alias("flow_hi"),
        (flow_hi - flow_lo).alias("flow_imbalance"),
        ((flow_hi - flow_lo) / (flow_hi + flow_lo + EPS)).alias("flow_oneway"),
        tr_hi.alias("transfer_hi_n"),
        ((tr_hi - tr_lo) / (tr_hi + tr_lo + EPS)).alias("transfer_oneway"),
        (pl.col("transfer_top5") * 5 / (pl.col("transfer_total") + EPS)).alias("transfer_top5_share"),
        (pl.col("transfer_max") / (pl.col("transfer_gross_mean") + EPS)).alias("transfer_peakiness"),
        (pl.col("transfer_total") / pl.col("shared_hands")).alias("transfer_per_hand"),
        (pl.col("transfer_event_value") / (pl.col("transfer_total") + EPS)).alias("transfer_event_value_share"),
        conc_hi.alias("concession_max_hi"),
        conc_sum_hi.alias("concession_sum_hi"),
        (pl.col("concession_sum_1") + pl.col("concession_sum_2")).alias("concession_sum_both"),
        pl.max_horizontal("concession_top3_1", "concession_top3_2").alias("concession_top3_hi"),
        pl.max_horizontal("fold_odds_max_1", "fold_odds_max_2").alias("fold_odds_max_hi"),
        (conc_sum_hi / (pl.col("shared_hands"))).alias("concession_per_hand"),
        # How tightly the flagged hands cluster inside the pair's lifetime.
        ((pl.col("last_event") - pl.col("first_event")).dt.total_seconds()
         / ((pl.col("last_seen") - pl.col("first_seen")).dt.total_seconds() + 1.0)
         ).alias("event_time_span_ratio"),
    ).drop("first_seen", "last_seen", "first_event", "last_event").collect(engine="streaming")


def directed_long(pairs: pl.DataFrame) -> pl.DataFrame:
    """One row per ordered (actor, partner) with event counts and exposure."""
    sides = []
    for actor, other, suffix in (("p1", "p2", "12"), ("p2", "p1", "21")):
        sides.append(
            pairs.select(
                "phase", "table_id",
                pl.col(actor).alias("actor"),
                pl.col(other).alias("partner"),
                pl.col("shared_hands").alias("n"),
                *[pl.col(f"k_{e}_{suffix}").alias(f"k_{e}") for e in DIRECTED],
            )
        )
    return pl.concat(sides)


def add_directed_surprise(pairs: pl.DataFrame) -> pl.DataFrame:
    long = directed_long(pairs)
    totals = long.group_by("phase", "table_id", "actor", maintain_order=True).agg(
        pl.col("n").sum().alias("tot_n"),
        *[pl.col(f"k_{e}").sum().alias(f"tot_{e}") for e in DIRECTED],
    )
    long = long.join(totals, on=["phase", "table_id", "actor"], how="left")

    n = long["n"].to_numpy().astype(np.float64)
    field_n = (long["tot_n"] - long["n"]).to_numpy().astype(np.float64)
    out: dict[str, np.ndarray] = {}
    for e in DIRECTED:
        k = long[f"k_{e}"].to_numpy().astype(np.float64)
        field_k = (long[f"tot_{e}"] - long[f"k_{e}"]).to_numpy().astype(np.float64)
        # The player's own rate against every other opponent at the table.
        p = np.clip(field_k / np.maximum(field_n, 1.0), 1e-6, 1 - 1e-6)
        # P(X >= k) under the field rate; sf(k-1) because sf is strictly greater.
        tail = stats.binom.sf(k - 1, n, p)
        out[f"s_{e}"] = -np.log10(np.clip(tail, 1e-300, 1.0))
        out[f"z_{e}"] = (k - n * p) / np.sqrt(np.maximum(n * p * (1 - p), 1e-9))
        out[f"exc_{e}"] = k - n * p
        out[f"rate_{e}"] = k / np.maximum(n, 1.0)
        out[f"field_{e}"] = p

    long = long.with_columns([pl.Series(name, values) for name, values in out.items()])
    keyed = long.select(
        "phase", "table_id",
        pl.min_horizontal("actor", "partner").alias("p1"),
        pl.max_horizontal("actor", "partner").alias("p2"),
        *[pl.col(c) for c in out],
    )
    # Fold the two directions back together: `hi` is the donor side of a
    # one-way relationship, `lo` the quieter side, `sum` the joint weight.
    folded = keyed.group_by("phase", "table_id", "p1", "p2", maintain_order=True).agg(
        *[pl.col(c).max().alias(f"{c}_hi") for c in out],
        *[pl.col(c).min().alias(f"{c}_lo") for c in out],
        *[pl.col(f"s_{e}").sum().alias(f"s_{e}_sum") for e in DIRECTED],
    )
    return pairs.join(folded, on=["phase", "table_id", "p1", "p2"], how="left")


def add_symmetric_contrasts(pairs: pl.DataFrame) -> pl.DataFrame:
    """Empirical-Bayes rate contrasts for the symmetric (non-directed) signals."""
    sides = []
    for me, other in (("p1", "p2"), ("p2", "p1")):
        sides.append(
            pairs.select(
                "phase", "table_id",
                pl.col(me).alias("player"), pl.col(other).alias("partner"),
                pl.col("shared_hands").alias("n"),
                *[pl.col(c) for c in CONTRAST_RATES],
            )
        )
    long = pl.concat(sides)
    totals = long.group_by("phase", "table_id", "player", maintain_order=True).agg(
        pl.col("n").sum().alias("tot_n"),
        *[(pl.col(c) * pl.col("n")).sum().alias(f"tot_{c}") for c in CONTRAST_RATES],
    )
    long = long.join(totals, on=["phase", "table_id", "player"], how="left")
    field_n = pl.col("tot_n") - pl.col("n")
    long = long.with_columns([
        (((pl.col(c) - (pl.col(f"tot_{c}") - pl.col(c) * pl.col("n")) / (field_n + EPS))
          * pl.col("n") / (pl.col("n") + EB_LAMBDA)).alias(f"d_{c}"))
        for c in CONTRAST_RATES
    ])
    keyed = long.select(
        "phase", "table_id",
        pl.min_horizontal("player", "partner").alias("p1"),
        pl.max_horizontal("player", "partner").alias("p2"),
        *[pl.col(f"d_{c}") for c in CONTRAST_RATES],
    )
    folded = keyed.group_by("phase", "table_id", "p1", "p2", maintain_order=True).agg(
        *[pl.col(f"d_{c}").max().alias(f"d_{c}_hi") for c in CONTRAST_RATES],
        *[pl.col(f"d_{c}").min().alias(f"d_{c}_lo") for c in CONTRAST_RATES],
    )
    return pairs.join(folded, on=["phase", "table_id", "p1", "p2"], how="left")


TABLE_RANKED = ["ps_top5", "pfs_top3", "ps_on_events_max",
                "transfer_event_rate", "transfer_per_hand", "both_showdown_rate",
                "s_transfer_event_hi", "s_fold_better_hi", "s_outsider_folds_to_hi",
                "flow_oneway", "mutual_passive_rate", "outsiders_folded_to_pair_mean",
                "concession_sum_hi", "relay_cleared_rate", "pair_net_mean",
                "transfer_event_max"]


def add_table_ranks(pairs: pl.DataFrame) -> pl.DataFrame:
    ranked = TABLE_RANKED
    n = pl.len().over("phase", "table_id")
    return pairs.with_columns(
        [(pl.col(c).rank("average").over("phase", "table_id") / n).alias(f"pct_{c}")
         for c in ranked]
    )


def main() -> None:
    t0 = time.time()
    # The streaming aggregation emits pairs in no fixed order, and the contrast
    # stages below sum floats over rows derived from this frame. Fixing the order
    # here makes those sums - and everything trained on them - reproducible.
    pairs = aggregate_pairs().sort("phase", "table_id", "p1", "p2")
    print(f"aggregated {pairs.height:,} pairs ({time.time() - t0:.0f}s)")
    pairs = add_directed_surprise(pairs)
    print(f"directed surprise added ({time.time() - t0:.0f}s)")
    pairs = add_symmetric_contrasts(pairs)
    pairs = add_table_ranks(pairs)
    pairs = pairs.fill_nan(None)
    pairs.write_parquet(OUT, compression="zstd")
    print(f"wrote {OUT}: {pairs.height:,} rows x {pairs.width} cols "
          f"({OUT.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
