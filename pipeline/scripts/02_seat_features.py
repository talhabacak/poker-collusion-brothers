"""Stage 2: one feature row per (hand_id, player_id), 12M rows.

Combines three things the raw tables keep apart:

* what the player held - hole cards are visible for every seat on every hand,
  including folded ones, so hand strength is known even when it was never shown;
* what the player did - action counts by street, aggression, folds facing a bet;
* how that compares inside the hand - strength rank among the six dealt players
  on the board that was actually dealt.

The strength rank is the piece that makes value transfer separable from bad
luck: it answers "who should have won this pot" for every hand, not just the
421,858 that reached a showdown.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C
from pokercol.cards import eval_best5, parse_cards, preflop_bucket, split_board

CHUNK = 1_000_000
OUT = C.CACHE / "seat_features.parquet"


def hand_frame() -> pl.DataFrame:
    return (
        pl.scan_parquet(C.HANDS)
        .select(
            "hand_id", "table_id", "phase", "started_at", "button_seat",
            "big_blind", "board_cards", "final_pot", "players_at_showdown",
        )
        .with_row_index("hand_idx")
        .collect()
    )


def board_matrices(hands: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    board_rank, board_suit = split_board(hands["board_cards"].to_numpy())
    n_board = (board_rank >= 0).sum(axis=1).astype(np.int8)
    return board_rank, board_suit, n_board


def seat_strength(seats: pl.DataFrame, board_rank, board_suit, n_board) -> pl.DataFrame:
    """Best-five score per seat, plus preflop bucket, evaluated in chunks."""
    hand_idx = seats["hand_idx"].to_numpy()
    hole = np.stack(
        [seats["hole_card_1"].to_numpy().astype("U2"),
         seats["hole_card_2"].to_numpy().astype("U2")],
        axis=1,
    )
    hole_rank, hole_suit = parse_cards(hole)

    n = seats.height
    best5 = np.zeros(n, dtype=np.int64)
    bucket = np.zeros(n, dtype=np.int16)
    # Strength as it stood on each street, from only the board dealt by then.
    # A fold has to be judged against the cards on the table when it happened:
    # a donor who folds top pair on the turn has conceded even if the river
    # later hands the partner a straight.
    by_street = {k: np.zeros(n, dtype=np.int64) for k in (3, 4, 5)}

    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        idx = hand_idx[start:stop]
        hr, hs = hole_rank[start:stop], hole_suit[start:stop]
        bucket[start:stop] = preflop_bucket(hr, hs)

        cards_rank = np.concatenate([hr, board_rank[idx]], axis=1)
        cards_suit = np.concatenate([hs, board_suit[idx]], axis=1)
        # A hand that ended preflop has no board, so there is no five-card hand
        # to score; those rows keep 0 and are flagged by board_cards_dealt.
        has_board = n_board[idx] >= 3
        if has_board.any():
            rows = np.flatnonzero(has_board)
            best5[start + rows] = eval_best5(cards_rank[rows], cards_suit[rows])
        for k in (3, 4, 5):
            rows = np.flatnonzero(n_board[idx] >= k)
            if rows.size:
                by_street[k][start + rows] = eval_best5(cards_rank[rows, : 2 + k],
                                                        cards_suit[rows, : 2 + k])

    return seats.with_columns(
        pl.Series("best5", best5),
        pl.Series("preflop_bucket", bucket),
        pl.Series("board_cards_dealt", n_board[hand_idx]),
        pl.Series("strength_flop", by_street[3]),
        pl.Series("strength_turn", by_street[4]),
        pl.Series("strength_river", by_street[5]),
    )


def action_aggregates() -> pl.LazyFrame:
    # Street ordinal is computed once, before grouping. Mapping it inside the
    # aggregation after a filter pushes polars onto a per-group path that takes
    # minutes over 12M (hand, player) groups instead of seconds.
    a = pl.scan_parquet(C.ACTIONS).with_columns(
        pl.col("street")
        .replace_strict({"preflop": 0, "flop": 1, "turn": 2, "river": 3}, return_dtype=pl.Int8)
        .alias("street_id")
    )
    postflop = pl.col("street") != "preflop"
    facing_bet = pl.col("to_call") > 0
    aggressive = pl.col("action").is_in(["raise", "bet"])
    return a.group_by("hand_id", "player_id").agg(
        pl.len().alias("n_actions"),
        (pl.col("action") == "raise").sum().alias("raises"),
        (pl.col("action") == "bet").sum().alias("bets"),
        (pl.col("action") == "call").sum().alias("calls"),
        (pl.col("action") == "check").sum().alias("checks"),
        (pl.col("action") == "fold").sum().alias("folds"),
        (pl.col("action") == "all_in").sum().alias("all_ins"),
        aggressive.sum().alias("aggressive_actions"),
        (aggressive & postflop).sum().alias("postflop_aggressive"),
        ((pl.col("action") == "check") & postflop).sum().alias("postflop_checks"),
        ((pl.col("action") == "call") & postflop).sum().alias("postflop_calls"),
        postflop.sum().alias("postflop_actions"),
        ((pl.col("action") == "fold") & facing_bet).sum().alias("folds_facing_bet"),
        ((pl.col("action") == "call") & facing_bet).sum().alias("calls_facing_bet"),
        # Preflop entry style.
        ((pl.col("street") == "preflop") & pl.col("action").is_in(["call", "raise", "bet", "all_in"]))
        .sum().alias("vpip_actions"),
        ((pl.col("street") == "preflop") & (pl.col("action") == "raise")).sum().alias("preflop_raises"),
        pl.col("amount").sum().alias("amount_put_in"),
        pl.col("amount").max().alias("max_amount"),
        pl.col("to_call").max().alias("max_to_call"),
        pl.col("to_call").sum().alias("sum_to_call"),
        pl.col("pot_before").max().alias("max_pot_before"),
        pl.col("players_active").min().alias("min_players_active"),
        # Context at the moment of folding. A fold is cheap or expensive
        # depending on the price offered, and folding a good hand at a good
        # price is the loudest single-hand signal of a deliberate concession.
        pl.col("to_call").filter(pl.col("action") == "fold").max().alias("fold_to_call"),
        pl.col("pot_before").filter(pl.col("action") == "fold").max().alias("fold_pot_before"),
        pl.when(pl.col("action") == "fold").then(pl.col("street_id")).otherwise(None)
        .min().alias("fold_street"),
        # Street the player stopped acting on, as an ordinal 0..3.
        pl.col("street_id").max().alias("last_street"),
    )


def fold_attribution() -> pl.LazyFrame:
    """Who each folding player was folding *to*, and how many folds each
    aggressor induced.

    A fold is only meaningful against the player who applied the pressure. This
    walks the action sequence per hand, carries the most recent bettor/raiser
    forward, and attributes every fold to them. It is what turns "an outsider
    folded" into "an outsider folded to this pair", which is the coordinated
    isolation motif, and "folded to the partner", which is a directed transfer.
    """
    a = (
        pl.scan_parquet(C.ACTIONS)
        .select("hand_id", "action_no", "player_id", "action")
        .sort("hand_id", "action_no")
        .with_columns(
            pl.when(pl.col("action").is_in(["bet", "raise", "all_in"]))
            .then(pl.col("player_id"))
            .otherwise(None)
            .alias("aggressor")
        )
        # The last aggressor strictly before this row: shift, then carry forward.
        .with_columns(
            pl.col("aggressor").shift(1).forward_fill().over("hand_id").alias("folded_to")
        )
    )
    folds = a.filter((pl.col("action") == "fold") & pl.col("folded_to").is_not_null())

    who_folded = folds.select("hand_id", "player_id", "folded_to")
    induced = folds.group_by("hand_id", pl.col("folded_to").alias("player_id")).agg(
        pl.len().alias("folds_induced")
    )
    return who_folded.join(induced, on=["hand_id", "player_id"], how="full", coalesce=True)


def main() -> None:
    t0 = time.time()
    hands = hand_frame()
    board_rank, board_suit, n_board = board_matrices(hands)
    print(f"hands parsed: {hands.height:,} ({time.time() - t0:.0f}s)")

    seats = (
        pl.scan_parquet(C.SEATS)
        .join(
            hands.lazy().select(
                "hand_id", "hand_idx", "table_id", "phase", "started_at",
                "button_seat", "big_blind", "final_pot", "players_at_showdown",
            ),
            on="hand_id",
            how="inner",
        )
        .collect()
    )
    print(f"seats joined: {seats.height:,} ({time.time() - t0:.0f}s)")

    seats = seat_strength(seats, board_rank, board_suit, n_board)
    # Ordinal starting-hand strength; the bucket id alone is not ordered.
    seats = seats.join(pl.read_parquet(C.CACHE / "preflop_equity.parquet"),
                       on="preflop_bucket", how="left")
    print(f"strength evaluated ({time.time() - t0:.0f}s)")

    seats = seats.join(action_aggregates().collect(), on=["hand_id", "player_id"], how="left")
    seats = seats.join(fold_attribution().collect(), on=["hand_id", "player_id"], how="left")
    seats = seats.with_columns(
        pl.col("folds_induced").fill_null(0),
        # A player who never acted (the big blind in a walk) faced all six seats
        # and stopped on the first street; null here would leak into heads-up flags.
        pl.col("min_players_active").fill_null(C.SEATS_PER_HAND),
        pl.col("last_street").fill_null(0),
    )
    print(f"actions joined ({time.time() - t0:.0f}s)")

    bb = pl.col("big_blind")
    seats = seats.with_columns(
        (pl.col("net_chips") / bb).alias("net_bb"),
        (pl.col("total_contribution") / bb).alias("contribution_bb"),
        (pl.col("final_pot") / bb).alias("pot_bb"),
        (pl.col("amount_put_in").fill_null(0) / bb).alias("amount_put_in_bb"),
        (pl.col("max_amount").fill_null(0) / bb).alias("max_amount_bb"),
        (pl.col("max_to_call").fill_null(0) / bb).alias("max_to_call_bb"),
        (pl.col("fold_to_call").fill_null(0) / bb).alias("fold_to_call_bb"),
        (pl.col("fold_pot_before").fill_null(0) / bb).alias("fold_pot_before_bb"),
        # Pot odds the folder turned down: pot / price. Folding a winner at 14:1
        # is a far stronger signal than folding one at 2:1.
        (pl.col("fold_pot_before").fill_null(0)
         / pl.max_horizontal(pl.col("fold_to_call").fill_null(0), pl.lit(1))
         ).alias("fold_pot_odds"),
        # Seats are dealt clockwise from the button; position 0 is the button.
        ((pl.col("seat_no") - pl.col("button_seat")) % C.SEATS_PER_HAND).alias("position"),
    ).with_columns(
        [pl.col(c).fill_null(0) for c in
         ["n_actions", "raises", "bets", "calls", "checks", "folds", "all_ins",
          "aggressive_actions", "postflop_aggressive", "postflop_checks",
          "postflop_calls", "postflop_actions", "folds_facing_bet",
          "calls_facing_bet", "vpip_actions", "preflop_raises"]]
    )

    # Strength rank within the hand: 1 = strongest of the six dealt players.
    # Ties share the minimum rank so a chopped pot does not invent an ordering.
    scored = pl.col("board_cards_dealt") >= 3
    seats = seats.with_columns(
        pl.when(scored)
        .then(pl.col("best5").rank("min", descending=True).over("hand_id"))
        .otherwise(None)
        .cast(pl.Int8)
        .alias("strength_rank"),
        pl.when(scored)
        .then((pl.col("best5") == pl.col("best5").max().over("hand_id")).cast(pl.Int8))
        .otherwise(None)
        .alias("held_best_hand"),
    )

    keep = [
        "hand_id", "player_id", "table_id", "phase", "started_at", "seat_no", "position",
        "big_blind", "pot_bb", "board_cards_dealt", "players_at_showdown",
        "preflop_bucket", "preflop_equity_hu", "preflop_equity_6max", "preflop_strength_pct",
        "best5", "strength_rank", "held_best_hand",
        "strength_flop", "strength_turn", "strength_river", "fold_street",
        "net_bb", "contribution_bb", "won_share", "folded", "went_to_showdown",
        "n_actions", "raises", "bets", "calls", "checks", "folds", "all_ins",
        "aggressive_actions", "postflop_aggressive", "postflop_checks",
        "postflop_calls", "postflop_actions", "folds_facing_bet", "calls_facing_bet",
        "vpip_actions", "preflop_raises", "amount_put_in_bb", "max_amount_bb",
        "max_to_call_bb", "min_players_active", "last_street",
        "folded_to", "folds_induced", "fold_to_call_bb", "fold_pot_before_bb",
        "fold_pot_odds",
    ]
    seats = seats.select(keep)
    seats.write_parquet(OUT, compression="zstd")
    print(f"wrote {OUT} : {seats.height:,} rows, {OUT.stat().st_size / 1e6:.0f} MB "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
