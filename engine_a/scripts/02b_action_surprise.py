"""Stage 2b: how improbable was each action, given the cards the actor held?

Aggregate features say a pair moved value. They do not say the moves were
*implausible*. A player who folds top pair to a single bet, or calls a pot-sized
bet with nothing, is taking an action that almost nobody in this population
takes from that spot with those cards - and a colluder does it selectively,
against one opponent.

So: fit the population's policy P(action | situation, own hand strength) over
all 18.6M actions, then score every action by -log P(observed). Conditioning on
the actor's own cards only - never on anyone else's - keeps the quantity
interpretable as "unusual play", not "play that looks odd to an observer with
hidden information".

Each action is also attributed to the player whose aggression it answered, so
surprise can be split into "unusual against this partner" versus "unusual in
general", which is the contrast that matters.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C
from pokercol.cards import draw_features, eval_best5, parse_cards, split_board

OUT = C.CACHE / "action_surprise.parquet"
STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
ACTIONS = ["fold", "check", "call", "bet", "raise", "all_in"]
BOARD_AT_STREET = {1: 3, 2: 4, 3: 5}
TRAIN_SAMPLE = 4_000_000
CHUNK = 2_000_000

PARAMS = dict(**C.LGB_REPRO,
    objective="multiclass", num_class=len(ACTIONS), learning_rate=0.08,
    num_leaves=127, min_data_in_leaf=200, feature_fraction=0.9,
    bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, max_bin=127,
    verbosity=-1, num_threads=C.N_THREADS, seed=C.SEED,
)
N_ROUNDS = 250

FEATURES = [
    "street", "position", "players_active", "pot_bb", "to_call_bb", "pot_odds",
    "stack_bb", "spr", "bet_faced_ratio", "hand_strength", "board_strength",
    "strength_over_board", "hand_category", "preflop_bucket", "preflop_equity_hu",
    "preflop_equity_6max", "board_cards_seen",
    "n_prior_actions", "prior_aggression",
    # Drawing hands: a call with four to a flush is not a pay-off.
    "flush_cards", "straight_run", "flush_uses_hole",
    # Betting line: what this player and this street have already seen.
    "player_prior_aggression", "street_prior_aggression", "player_prior_actions",
    # The actor's own style, so surprise means "unusual for this player" rather
    # than "unusual for the population". A loose player's loose call is not news.
    "style_vpip", "style_pfr", "style_aggression", "style_fold_to_bet",
    "style_showdown", "style_postflop_aggression",
]
HOLDOUT = 1_000_000


def street_strength() -> pl.DataFrame:
    """Best-five score per (hand, player, street) using only the board dealt by then.

    A player deciding on the flop has seen three board cards; scoring them
    against the river board would measure hindsight, not their decision.
    """
    hands = (
        pl.scan_parquet(C.HANDS)
        .select("hand_id", "board_cards")
        .with_row_index("hand_idx")
        .collect()
    )
    board_rank, board_suit = split_board(hands["board_cards"].to_numpy())
    n_board = (board_rank >= 0).sum(axis=1).astype(np.int8)

    seats = (
        pl.scan_parquet(C.SEATS)
        .select("hand_id", "player_id", "hole_card_1", "hole_card_2")
        .join(hands.lazy().select("hand_id", "hand_idx"), on="hand_id", how="inner")
        .collect()
    )
    hole_rank, hole_suit = parse_cards(
        np.stack([seats["hole_card_1"].to_numpy().astype("U2"),
                  seats["hole_card_2"].to_numpy().astype("U2")], axis=1)
    )
    idx = seats["hand_idx"].to_numpy()

    frames = []
    for street, n_cards in BOARD_AT_STREET.items():
        available = n_board[idx] >= n_cards
        rows = np.flatnonzero(available)
        if rows.size == 0:
            continue
        strength = np.zeros(rows.size, dtype=np.int64)
        board_only = np.zeros(rows.size, dtype=np.int64)
        flush_cards = np.zeros(rows.size, dtype=np.int8)
        straight_run = np.zeros(rows.size, dtype=np.int8)
        flush_hole = np.zeros(rows.size, dtype=np.int8)
        for start in range(0, rows.size, CHUNK):
            block = rows[start : start + CHUNK]
            b = idx[block]
            br = board_rank[b][:, :n_cards]
            bs = board_suit[b][:, :n_cards]
            cr = np.concatenate([hole_rank[block], br], axis=1)
            cs = np.concatenate([hole_suit[block], bs], axis=1)
            sl = slice(start, start + block.size)
            strength[sl] = eval_best5(cr, cs)
            flush_cards[sl], straight_run[sl], flush_hole[sl] = draw_features(cr, cs)
            if n_cards == 5:
                board_only[sl] = eval_best5(br, bs)
        frames.append(
            seats[rows].select("hand_id", "player_id").with_columns(
                pl.lit(street, dtype=pl.Int8).alias("street"),
                pl.Series("hand_strength", strength),
                pl.Series("board_strength", board_only),
                pl.Series("flush_cards", flush_cards),
                pl.Series("straight_run", straight_run),
                pl.Series("flush_uses_hole", flush_hole),
            )
        )
    return pl.concat(frames)


def player_style() -> pl.LazyFrame:
    """Per (player, phase) tendencies, the baseline each action is judged against.

    A colluder's planted hands are a handful out of several hundred, so these
    rates describe how the player normally plays, not how they cheat.
    """
    return (
        pl.scan_parquet(C.CACHE / "seat_features.parquet")
        .group_by("player_id", "phase")
        .agg(
            (pl.col("vpip_actions") > 0).mean().alias("style_vpip"),
            (pl.col("preflop_raises") > 0).mean().alias("style_pfr"),
            pl.col("aggressive_actions").mean().alias("style_aggression"),
            (pl.col("folds_facing_bet").sum()
             / (pl.col("folds_facing_bet").sum() + pl.col("calls_facing_bet").sum()).clip(lower_bound=1))
            .alias("style_fold_to_bet"),
            pl.col("went_to_showdown").mean().alias("style_showdown"),
            (pl.col("postflop_aggressive").sum()
             / pl.col("postflop_actions").sum().clip(lower_bound=1)).alias("style_postflop_aggression"),
        )
    )


def action_frame() -> pl.DataFrame:
    hands = pl.scan_parquet(C.HANDS).select(
        "hand_id", "table_id", "phase", "big_blind", "button_seat", "board_cards"
    ).with_columns(
        ((pl.col("board_cards").str.len_chars() + 1) // 3).cast(pl.Int8).alias("board_cards_dealt")
    ).drop("board_cards")
    seats = pl.scan_parquet(C.SEATS).select("hand_id", "player_id", "seat_no")
    seat_meta = pl.scan_parquet(C.CACHE / "seat_features.parquet").select(
        "hand_id", "player_id", "preflop_bucket", "preflop_equity_hu", "preflop_equity_6max"
    )

    a = (
        pl.scan_parquet(C.ACTIONS)
        .join(hands, on="hand_id", how="inner")
        .join(seats, on=["hand_id", "player_id"], how="inner")
        .join(seat_meta, on=["hand_id", "player_id"], how="left")
        .join(player_style(), on=["player_id", "phase"], how="left")
        .with_columns(
            pl.col("street").replace_strict(STREETS, return_dtype=pl.Int8).alias("street_id")
        )
        .sort("hand_id", "action_no")
        .with_columns(
            # Who the actor is responding to, and how much pressure preceded them.
            pl.when(pl.col("action").is_in(["bet", "raise", "all_in"]))
            .then(pl.col("player_id")).otherwise(None).alias("_aggr")
        )
        .with_columns(
            pl.col("_aggr").shift(1).forward_fill().over("hand_id").alias("responding_to"),
            pl.int_range(pl.len()).over("hand_id").alias("n_prior_actions"),
            pl.col("action").is_in(["bet", "raise", "all_in"]).cast(pl.Int8)
            .cum_sum().over("hand_id").alias("_aggr_count"),
        )
        .with_columns(
            (pl.col("_aggr_count") - pl.col("action").is_in(["bet", "raise", "all_in"]).cast(pl.Int8))
            .alias("prior_aggression")
        )
        .with_columns(pl.col("action").is_in(["bet", "raise", "all_in"]).cast(pl.Int16).alias("_is_aggr"))
        .with_columns(
            (pl.col("_is_aggr").cum_sum().over("hand_id", "player_id") - pl.col("_is_aggr"))
            .alias("player_prior_aggression"),
            (pl.col("_is_aggr").cum_sum().over("hand_id", "street_id") - pl.col("_is_aggr"))
            .alias("street_prior_aggression"),
            pl.int_range(pl.len()).over("hand_id", "player_id").alias("player_prior_actions"),
        )
        .drop("_aggr", "_aggr_count", "_is_aggr")
    )

    bb = pl.col("big_blind")
    return (
        a.with_columns(
            pl.col("street_id").alias("street"),
            ((pl.col("seat_no") - pl.col("button_seat")) % C.SEATS_PER_HAND).alias("position"),
            (pl.col("pot_before") / bb).alias("pot_bb"),
            (pl.col("to_call") / bb).alias("to_call_bb"),
            (pl.col("stack_before") / bb).alias("stack_bb"),
            (pl.col("to_call") / (pl.col("pot_before") + pl.col("to_call") + 1e-9)).alias("pot_odds"),
            (pl.col("stack_before") / (pl.col("pot_before") + 1e-9)).alias("spr"),
            (pl.col("to_call") / (pl.col("stack_before") + 1e-9)).alias("bet_faced_ratio"),
            pl.col("board_cards_dealt").alias("board_cards_seen"),
        )
        .collect()
    )


def main() -> None:
    t0 = time.time()
    strength = street_strength()
    print(f"street strengths: {strength.height:,} rows ({time.time() - t0:.0f}s)")

    a = action_frame()
    print(f"actions: {a.height:,} ({time.time() - t0:.0f}s)")

    a = a.join(strength, on=["hand_id", "player_id", "street"], how="left").with_columns(
        pl.col("hand_strength").fill_null(0), pl.col("board_strength").fill_null(0),
        pl.col("flush_cards").fill_null(0), pl.col("straight_run").fill_null(0),
        pl.col("flush_uses_hole").fill_null(0),
    ).with_columns(
        (pl.col("hand_strength") - pl.col("board_strength")).alias("strength_over_board"),
        (pl.col("hand_strength") // (14 ** 5)).cast(pl.Int8).alias("hand_category"),
        pl.col("action").replace_strict({n: i for i, n in enumerate(ACTIONS)},
                                        return_dtype=pl.Int8).alias("y"),
    )

    # The training sample is drawn by row position, so the row order must not
    # depend on how the joins above happened to emit rows.
    a = a.sort("hand_id", "action_no")
    x = a.select(FEATURES).to_numpy().astype(np.float32)
    y = a["y"].to_numpy()

    rng = np.random.default_rng(C.SEED)
    order = rng.permutation(a.height)
    sample, holdout = order[:TRAIN_SAMPLE], order[TRAIN_SAMPLE : TRAIN_SAMPLE + HOLDOUT]
    booster = lgb.train(PARAMS, lgb.Dataset(x[sample], label=y[sample]),
                        num_boost_round=N_ROUNDS)
    hp = booster.predict(x[holdout])
    holdout_logloss = float(-np.log(np.clip(hp[np.arange(holdout.size), y[holdout]], 1e-9, 1)).mean())
    print(f"policy model fitted on {sample.size:,} actions; held-out log-loss "
          f"{holdout_logloss:.4f} ({time.time() - t0:.0f}s)")

    surprise = np.zeros(a.height, dtype=np.float32)
    aggr_prob = np.zeros(a.height, dtype=np.float32)
    aggressive_ids = [ACTIONS.index(n) for n in ("bet", "raise", "all_in")]
    for start in range(0, a.height, CHUNK):
        stop = min(start + CHUNK, a.height)
        proba = booster.predict(x[start:stop])
        taken = proba[np.arange(stop - start), y[start:stop]]
        surprise[start:stop] = -np.log(np.clip(taken, 1e-9, 1.0))
        aggr_prob[start:stop] = proba[:, aggressive_ids].sum(axis=1)
    print(f"scored all actions ({time.time() - t0:.0f}s)  mean surprise {surprise.mean():.4f}")

    fold_id = ACTIONS.index("fold")
    call_id = ACTIONS.index("call")
    scored = a.select(
        "hand_id", "player_id", "responding_to", "street", "phase",
    ).with_columns(
        pl.Series("surprise", surprise),
        pl.Series("expected_aggression", aggr_prob),
        pl.Series("was_aggressive", np.isin(y, aggressive_ids).astype(np.int8)),
        pl.Series("is_fold", (y == fold_id).astype(np.int8)),
        pl.Series("is_call", (y == call_id).astype(np.int8)),
    ).with_columns(
        # Splitting by action type matters: an improbable fold is a concession,
        # an improbable call is a pay-off, and an improbable raise is pressure.
        # Averaged together they partly cancel.
        (pl.col("surprise") * pl.col("is_fold")).alias("fold_surprise"),
        (pl.col("surprise") * pl.col("is_call")).alias("call_surprise"),
        (pl.col("surprise") * pl.col("was_aggressive")).alias("aggr_surprise"),
    )

    # Per (hand, player): overall, and split by which opponent was being answered.
    per_seat = scored.group_by("hand_id", "player_id").agg(
        pl.col("surprise").mean().alias("surprise_mean"),
        pl.col("surprise").max().alias("surprise_max"),
        pl.col("surprise").sum().alias("surprise_sum"),
        pl.col("fold_surprise").max().alias("fold_surprise"),
        pl.col("call_surprise").sum().alias("call_surprise"),
        pl.col("aggr_surprise").sum().alias("aggr_surprise"),
        (pl.col("was_aggressive") - pl.col("expected_aggression")).sum().alias("aggression_excess"),
    )
    per_target = scored.filter(pl.col("responding_to").is_not_null()).group_by(
        "hand_id", "player_id", "responding_to"
    ).agg(
        pl.col("surprise").sum().alias("surprise_vs_target"),
        pl.col("surprise").max().alias("surprise_vs_target_max"),
        pl.len().alias("actions_vs_target"),
        pl.col("fold_surprise").sum().alias("fold_surprise_vs_target"),
        pl.col("call_surprise").sum().alias("call_surprise_vs_target"),
        pl.col("aggr_surprise").sum().alias("aggr_surprise_vs_target"),
        pl.col("is_fold").sum().alias("folds_vs_target"),
        (pl.col("was_aggressive") - pl.col("expected_aggression")).sum()
        .alias("aggression_excess_vs_target"),
    )

    per_seat.write_parquet(C.CACHE / "action_surprise_seat.parquet", compression="zstd")
    per_target.write_parquet(OUT, compression="zstd")
    print(f"wrote {per_seat.height:,} seat rows and {per_target.height:,} directed rows "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
