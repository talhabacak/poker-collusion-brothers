"""Stage 3: one row per (pair, shared hand) for the full co-seating universe.

Roughly 30M rows (2M hands x C(6,2) seats). Built in table batches so peak
memory stays flat, and written as a single parquet dataset that later stages
scan lazily.

The interaction columns are written in canonical player order (`p1 < p2`) so a
pair has one stable identity, and directional quantities are always stated as
`1_to_2` / `2_to_1` rather than "donor"/"receiver" - which side is which is a
modelling question, not a data one.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C

OUT_DIR = C.CACHE / "pair_hands"
TABLES_PER_BATCH = 20

SIDE_COLS = [
    "net_bb", "contribution_bb", "folded", "went_to_showdown", "strength_rank",
    "held_best_hand", "best5", "preflop_bucket", "preflop_equity_hu", "preflop_strength_pct",
    "strength_flop", "strength_turn", "strength_river", "fold_street",
    "position", "n_actions", "raises",
    "bets", "calls", "checks", "folds_facing_bet", "calls_facing_bet",
    "aggressive_actions", "postflop_aggressive", "postflop_checks", "postflop_calls",
    "postflop_actions", "amount_put_in_bb", "max_amount_bb", "max_to_call_bb",
    "min_players_active", "last_street", "vpip_actions", "preflop_raises",
    "folded_to", "folds_induced", "fold_to_call_bb", "fold_pot_odds",
    "surprise_mean", "surprise_max", "surprise_sum", "fold_surprise",
    "call_surprise", "aggr_surprise", "aggression_excess",
]
HAND_COLS = ["table_id", "phase", "started_at", "pot_bb", "board_cards_dealt",
             "players_at_showdown", "big_blind"]


def consecutive_aggression() -> pl.DataFrame:
    """Ordered pairs of players whose aggressive actions were adjacent in a hand.

    Coordinated isolation looks like a relay: one partner raises, the other
    immediately raises again, and the players between them fold. Counting each
    partner's aggression separately misses it entirely - what matters is that
    the second raise answered the first with nobody in between.
    """
    return (
        pl.scan_parquet(C.ACTIONS)
        .filter(pl.col("action").is_in(["bet", "raise", "all_in"]))
        .sort("hand_id", "action_no")
        .with_columns(pl.col("player_id").shift(1).over("hand_id").alias("prev_aggressor"))
        .filter(pl.col("prev_aggressor").is_not_null()
                & (pl.col("prev_aggressor") != pl.col("player_id")))
        .select("hand_id",
                pl.min_horizontal("prev_aggressor", "player_id").alias("p1"),
                pl.max_horizontal("prev_aggressor", "player_id").alias("p2"))
        .unique()
        .with_columns(pl.lit(True).alias("consecutive_aggression"))
        .collect()
    )


def hand_totals(seats: pl.DataFrame) -> pl.DataFrame:
    """Per-hand sums over all six seats, used to derive outsider behaviour."""
    return seats.group_by("hand_id").agg(
        pl.col("folded").sum().alias("h_folds"),
        pl.col("aggressive_actions").sum().alias("h_aggressive"),
        pl.col("contribution_bb").sum().alias("h_contribution_bb"),
        pl.col("folds_facing_bet").sum().alias("h_folds_facing_bet"),
        pl.col("went_to_showdown").sum().alias("h_showdowns"),
        pl.col("n_actions").sum().alias("h_actions"),
        pl.col("best5").max().alias("h_best5"),
        pl.col("net_bb").sum().alias("h_net_bb"),
        pl.col("preflop_raises").sum().alias("h_preflop_raises"),
    )


DIRECTED_SURPRISE = {
    "surprise_vs_target": "sv", "aggr_surprise_vs_target": "asv",
    "fold_surprise_vs_target": "fsv", "call_surprise_vs_target": "csv",
    "aggression_excess_vs_target": "axv", "actions_vs_target": "nv",
}


def partner_directed(directed: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Policy surprise of each partner's actions that answered the *other* partner.

    Seat-level surprise mixes in every action taken against outsiders. The
    concession, the pay-off and the isolation relay are all actions one partner
    takes in response to the other's pressure, so this isolates exactly those.
    """
    cols = list(DIRECTED_SURPRISE)
    one_to_two = directed.filter(pl.col("player_id") < pl.col("responding_to")).select(
        "hand_id", pl.col("player_id").alias("p1"), pl.col("responding_to").alias("p2"),
        *[pl.col(c).alias(f"{DIRECTED_SURPRISE[c]}_1_to_2") for c in cols],
    )
    two_to_one = directed.filter(pl.col("player_id") > pl.col("responding_to")).select(
        "hand_id", pl.col("responding_to").alias("p1"), pl.col("player_id").alias("p2"),
        *[pl.col(c).alias(f"{DIRECTED_SURPRISE[c]}_2_to_1") for c in cols],
    )
    return one_to_two, two_to_one


def build_batch(seats: pl.DataFrame, consec: pl.DataFrame, directed: pl.DataFrame) -> pl.DataFrame:
    totals = hand_totals(seats)
    one_to_two, two_to_one = partner_directed(directed)

    left = seats.select(
        "hand_id", *HAND_COLS,
        pl.col("player_id").alias("p1"),
        *[pl.col(c).alias(f"{c}_1") for c in SIDE_COLS],
    )
    right = seats.select(
        "hand_id",
        pl.col("player_id").alias("p2"),
        *[pl.col(c).alias(f"{c}_2") for c in SIDE_COLS],
    )
    ph = (
        left.join(right, on="hand_id", how="inner")
        .filter(pl.col("p1") < pl.col("p2"))
        .join(totals, on="hand_id", how="left")
        .join(consec, on=["hand_id", "p1", "p2"], how="left")
        .with_columns(pl.col("consecutive_aggression").fill_null(False))
        .join(one_to_two, on=["hand_id", "p1", "p2"], how="left")
        .join(two_to_one, on=["hand_id", "p1", "p2"], how="left")
        .with_columns([
            pl.col(f"{short}_{d}").fill_null(0.0)
            for short in DIRECTED_SURPRISE.values() for d in ("1_to_2", "2_to_1")
        ])
    )

    net1, net2 = pl.col("net_bb_1"), pl.col("net_bb_2")
    # Chips one side lost that the other side could have collected. This is the
    # host's own transfer definition from the starter notebook.
    t12 = pl.min_horizontal((-net1).clip(lower_bound=0), net2.clip(lower_bound=0))
    t21 = pl.min_horizontal((-net2).clip(lower_bound=0), net1.clip(lower_bound=0))

    scored = pl.col("board_cards_dealt") >= 3
    rank1, rank2 = pl.col("strength_rank_1"), pl.col("strength_rank_2")
    best1, best2 = pl.col("best5_1"), pl.col("best5_2")
    in_pot_1 = ~pl.col("folded_1")
    in_pot_2 = ~pl.col("folded_2")
    # `folded_to` is null for anyone who did not fold to an aggressor, and a null
    # comparison would propagate into every count built on it - silently
    # dropping most hands from the outsider-fold totals. Absent means "no".
    folded_to_partner_1 = (pl.col("folded_to_1") == pl.col("p2")).fill_null(False)
    folded_to_partner_2 = (pl.col("folded_to_2") == pl.col("p1")).fill_null(False)

    def strength_on(street: pl.Expr, side: str) -> pl.Expr:
        # Comparable only within one street: equity preflop, best-five after.
        return (pl.when(street == 0).then(pl.col(f"preflop_equity_hu_{side}") * 1e9)
                .when(street == 1).then(pl.col(f"strength_flop_{side}"))
                .when(street == 2).then(pl.col(f"strength_turn_{side}"))
                .when(street == 3).then(pl.col(f"strength_river_{side}"))
                .otherwise(None))

    s1, s2 = pl.col("fold_street_1"), pl.col("fold_street_2")
    # Folded to the partner's own aggression while holding the better hand at
    # that moment. Covers preflop folds, which a final-board comparison cannot
    # see at all, and folds that a later card would have reversed.
    ahead_fold_1 = (folded_to_partner_1
                    & (strength_on(s1, "1") > strength_on(s1, "2"))).fill_null(False)
    ahead_fold_2 = (folded_to_partner_2
                    & (strength_on(s2, "2") > strength_on(s2, "1"))).fill_null(False)

    return ph.with_columns(
        ahead_fold_1.alias("ahead_fold_1_to_2"),
        ahead_fold_2.alias("ahead_fold_2_to_1"),
        (ahead_fold_1 & (s1 == 0)).fill_null(False).alias("ahead_fold_preflop_1_to_2"),
        (ahead_fold_2 & (s2 == 0)).fill_null(False).alias("ahead_fold_preflop_2_to_1"),
        pl.when(ahead_fold_1 & (s1 == 0))
        .then(pl.col("preflop_equity_hu_1") - pl.col("preflop_equity_hu_2"))
        .when(ahead_fold_2 & (s2 == 0))
        .then(pl.col("preflop_equity_hu_2") - pl.col("preflop_equity_hu_1"))
        .otherwise(0.0).alias("ahead_fold_equity_gap"),
        t12.alias("transfer_1_to_2"),
        t21.alias("transfer_2_to_1"),
        (t12 + t21).alias("transfer_gross"),
        (t12 - t21).alias("transfer_signed"),
        (pl.col("contribution_bb_1") - pl.col("contribution_bb_2")).alias("contribution_gap"),
        (net1 - net2).alias("net_gap"),
        (pl.col("went_to_showdown_1") & pl.col("went_to_showdown_2")).alias("both_showdown"),
        (pl.col("folded_1") & pl.col("folded_2")).alias("both_folded"),
        (pl.col("folded_1") != pl.col("folded_2")).alias("one_folded"),
        (in_pot_1 & in_pot_2).alias("both_in_pot"),
        # Only the two partners left in the pot: the cleanest soft-play setting.
        (in_pot_1 & in_pot_2 & (pl.col("min_players_active_1") <= 2)
         & (pl.col("min_players_active_2") <= 2)).alias("pair_heads_up"),
        (pl.col("aggressive_actions_1") + pl.col("aggressive_actions_2")).alias("pair_aggressive"),
        # Both in the pot after the flop yet neither applies pressure.
        (in_pot_1 & in_pot_2 & (pl.col("postflop_actions_1") > 0) & (pl.col("postflop_actions_2") > 0)
         & (pl.col("postflop_aggressive_1") == 0) & (pl.col("postflop_aggressive_2") == 0)
         ).alias("mutual_passive"),
        (pl.when(scored).then(rank2 - rank1).otherwise(None)).alias("strength_gap"),
        # Value moved against hand strength. Both tests compare the two partners
        # *to each other*, not to the rest of the table: a donor who folds a
        # middling hand to a partner holding air is transferring value even when
        # a third player held the actual best hand, and that case is common.
        (scored & (net1 < 0) & (net2 > 0) & (best1 < best2)
         & (pl.col("contribution_bb_1") >= 2)).alias("paid_1_to_2"),
        (scored & (net2 < 0) & (net1 > 0) & (best2 < best1)
         & (pl.col("contribution_bb_2") >= 2)).alias("paid_2_to_1"),
        # Gave up a hand that beat the partner's, and the partner took the pot.
        (scored & pl.col("folded_1") & (best1 > best2) & (net2 > 0)
         & (pl.col("contribution_bb_1") > 0)).alias("fold_better_1_to_2"),
        (scored & pl.col("folded_2") & (best2 > best1) & (net1 > 0)
         & (pl.col("contribution_bb_2") > 0)).alias("fold_better_2_to_1"),
        # Folded the best hand of the six while the partner collected chips.
        (scored & pl.col("folded_1") & (pl.col("held_best_hand_1") == 1) & (net2 > 0)
         ).alias("conceded_1_to_2"),
        (scored & pl.col("folded_2") & (pl.col("held_best_hand_2") == 1) & (net1 > 0)
         ).alias("conceded_2_to_1"),
        # Fold attribution: pressure is only meaningful against its source.
        folded_to_partner_1.alias("p1_folded_to_p2"),
        folded_to_partner_2.alias("p2_folded_to_p1"),
        # Outsiders driven out by the pair's aggression - the isolation motif.
        (pl.col("folds_induced_1") + pl.col("folds_induced_2")
         - folded_to_partner_1.cast(pl.Int32) - folded_to_partner_2.cast(pl.Int32)
         ).alias("outsiders_folded_to_pair"),
        (pl.col("folds_induced_1") - folded_to_partner_2.cast(pl.Int32)
         ).alias("outsiders_folded_to_p1"),
        (pl.col("folds_induced_2") - folded_to_partner_1.cast(pl.Int32)
         ).alias("outsiders_folded_to_p2"),
        # Outsider context, by subtracting the pair from the hand totals.
        (pl.col("h_folds") - pl.col("folded_1").cast(pl.Int32) - pl.col("folded_2").cast(pl.Int32)
         ).alias("outsider_folds"),
        (pl.col("h_aggressive") - pl.col("aggressive_actions_1") - pl.col("aggressive_actions_2")
         ).alias("outsider_aggressive"),
        (pl.col("h_contribution_bb") - pl.col("contribution_bb_1") - pl.col("contribution_bb_2")
         ).alias("outsider_contribution_bb"),
        (pl.col("h_folds_facing_bet") - pl.col("folds_facing_bet_1") - pl.col("folds_facing_bet_2")
         ).alias("outsider_folds_facing_bet"),
        (pl.col("h_showdowns") - pl.col("went_to_showdown_1").cast(pl.Int32)
         - pl.col("went_to_showdown_2").cast(pl.Int32)).alias("outsider_showdowns"),
        # What the pair jointly took out of the hand, and what it cost the rest
        # of the table. Isolation profits the pair at the outsiders' expense
        # even when no value moves between the partners themselves.
        (net1 + net2).alias("pair_net_bb"),
        (pl.col("h_net_bb") - net1 - net2).alias("outsider_net_bb"),
        (pl.col("aggressive_actions_1") > 0).alias("aggr_1"),
        (pl.col("aggressive_actions_2") > 0).alias("aggr_2"),
        ((pl.col("aggressive_actions_1") > 0) & (pl.col("aggressive_actions_2") > 0)
         ).alias("both_aggressive"),
        # The relay motif, and how many outsiders it pushed out.
        (pl.col("consecutive_aggression")
         & (pl.col("h_folds") - pl.col("folded_1").cast(pl.Int32)
            - pl.col("folded_2").cast(pl.Int32) > 0)).alias("relay_cleared_outsiders"),
        # Coordinated isolation, as the planted hands actually play it: both
        # partners put in a preflop raise in the same round, usually sandwiching
        # an outsider who raised too, and usually holding hands that do not
        # justify it. The pair often loses the pot; the point is the pressure.
        ((pl.col("preflop_raises_1") > 0) & (pl.col("preflop_raises_2") > 0)
         ).alias("both_raised_preflop"),
        ((pl.col("preflop_raises_1") > 0) & (pl.col("preflop_raises_2") > 0)
         & (pl.col("h_preflop_raises") - pl.col("preflop_raises_1")
            - pl.col("preflop_raises_2") > 0)).alias("squeeze_sandwich"),
        (pl.col("preflop_raises_1") * (1.0 - pl.col("preflop_strength_pct_1"))
         + pl.col("preflop_raises_2") * (1.0 - pl.col("preflop_strength_pct_2"))
         ).alias("weak_preflop_raise_weight"),
        pl.when((pl.col("preflop_raises_1") > 0) & (pl.col("preflop_raises_2") > 0))
        .then(pl.min_horizontal("preflop_strength_pct_1", "preflop_strength_pct_2"))
        .otherwise(None).alias("weaker_raiser_strength"),
        (pl.col("h_preflop_raises") - pl.col("preflop_raises_1") - pl.col("preflop_raises_2")
         ).alias("outsider_preflop_raises"),
        ((pl.col("aggressive_actions_1") > 0) & (pl.col("aggressive_actions_2") > 0)
         & (net1 < 0) & (net2 < 0)).alias("pair_aggressive_loss"),
        # Size of the concession: pot odds declined, when folding to the partner.
        (pl.when(folded_to_partner_1).then(pl.col("fold_pot_odds_1")).otherwise(0.0)
         ).alias("fold_odds_declined_1"),
        (pl.when(folded_to_partner_2).then(pl.col("fold_pot_odds_2")).otherwise(0.0)
         ).alias("fold_odds_declined_2"),
        # Value conceded weighted by how wrong the fold was: only counted when the
        # folded hand actually beat the partner's.
        (pl.when(scored & folded_to_partner_1 & (best1 > best2))
         .then(pl.col("fold_pot_odds_1") * pl.col("contribution_bb_1"))
         .otherwise(0.0)).alias("concession_weight_1"),
        (pl.when(scored & folded_to_partner_2 & (best2 > best1))
         .then(pl.col("fold_pot_odds_2") * pl.col("contribution_bb_2"))
         .otherwise(0.0)).alias("concession_weight_2"),
    ).drop("h_folds", "h_aggressive", "h_contribution_bb", "h_folds_facing_bet",
           "h_showdowns", "h_actions", "h_best5",
           # Identity strings and raw scores have served their purpose; dropping
           # them keeps the 30M-row dataset compact.
           "folded_to_1", "folded_to_2", "best5_1", "best5_2", "h_net_bb",
           "strength_flop_1", "strength_flop_2", "strength_turn_1", "strength_turn_2",
           "strength_river_1", "strength_river_2",
           "h_preflop_raises")


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.parquet"):
        old.unlink()

    surprise = pl.scan_parquet(C.CACHE / "action_surprise_seat.parquet")
    lf = (
        pl.scan_parquet(C.CACHE / "seat_features.parquet")
        .join(surprise, on=["hand_id", "player_id"], how="left")
        .with_columns([pl.col(c).fill_null(0.0) for c in
                       ["surprise_mean", "surprise_max", "surprise_sum", "fold_surprise",
                        "call_surprise", "aggr_surprise", "aggression_excess"]])
    )
    tables = sorted(lf.select("table_id").unique().collect()["table_id"].to_list())
    consec_all = consecutive_aggression()
    directed_all = pl.read_parquet(C.CACHE / "action_surprise.parquet")
    print(f"{len(tables)} tables, {consec_all.height:,} consecutive-aggression pairs, "
          f"{directed_all.height:,} directed surprise rows")

    total = 0
    for i in range(0, len(tables), TABLES_PER_BATCH):
        batch = tables[i : i + TABLES_PER_BATCH]
        seats = lf.filter(pl.col("table_id").is_in(batch)).collect()
        hand_ids = seats.select("hand_id").unique()
        ph = build_batch(seats, consec_all.join(hand_ids, on="hand_id", how="semi"),
                         directed_all.join(hand_ids, on="hand_id", how="semi"))
        path = OUT_DIR / f"part_{i // TABLES_PER_BATCH:03d}.parquet"
        ph.write_parquet(path, compression="zstd")
        total += ph.height
        print(f"  tables {i}-{i + len(batch) - 1}: {ph.height:,} pair-hands "
              f"({time.time() - t0:.0f}s)")

    size = sum(p.stat().st_size for p in OUT_DIR.glob("*.parquet")) / 1e9
    print(f"total {total:,} pair-hand rows, {size:.2f} GB ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
