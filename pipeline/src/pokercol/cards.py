"""Vectorised poker hand evaluation.

Every seat's hole cards are visible in `seats.parquet` for every hand, including
hands the player folded. That makes counterfactual strength observable, which is
what separates a value transfer ("folded the best hand", "paid off with air")
from ordinary bad play. This module turns card strings into numbers fast enough
to run over all 12M seat rows.

Two strength notions are produced:

* `preflop_bucket` / `preflop_equity` - always available, from hole cards alone.
* `best5` score - the standard 5-card category+kicker ranking, evaluated from
  hole cards plus whatever board was dealt (5, 6 or 7 cards total).

Scores are plain integers, comparable with `>`; larger is stronger.
"""
from __future__ import annotations

import numpy as np

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
RANK_OF = {c: i for i, c in enumerate(RANK_CHARS)}   # 0..12, 12 = ace
SUIT_OF = {c: i for i, c in enumerate(SUIT_CHARS)}

CAT_HIGH, CAT_PAIR, CAT_TWO_PAIR, CAT_TRIPS = 0, 1, 2, 3
CAT_STRAIGHT, CAT_FLUSH, CAT_FULL_HOUSE, CAT_QUADS, CAT_STRAIGHT_FLUSH = 4, 5, 6, 7, 8

# Kickers are stored as `rank + 1`, so an ace-high kicker is 13 and an absent
# kicker is 0. Base 14 is the smallest radix that holds every digit without
# carrying into the category.
_KICKER_BASE = 14


def parse_cards(cards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parse an array of two-character card strings into (rank, suit) int8 arrays.

    Empty strings yield -1 in both outputs.
    """
    flat = np.asarray(cards, dtype="U2").ravel()
    rank = np.full(flat.shape, -1, dtype=np.int8)
    suit = np.full(flat.shape, -1, dtype=np.int8)
    valid = flat != ""
    if valid.any():
        chars = flat[valid].view("U1").reshape(-1, 2)
        rank[valid] = np.vectorize(RANK_OF.__getitem__, otypes=[np.int8])(chars[:, 0])
        suit[valid] = np.vectorize(SUIT_OF.__getitem__, otypes=[np.int8])(chars[:, 1])
    shape = np.asarray(cards).shape
    return rank.reshape(shape), suit.reshape(shape)


def split_board(board: np.ndarray, max_cards: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Split `"Ah Kd 7c"`-style board strings into padded (rank, suit) matrices."""
    board = np.asarray(board, dtype=object)
    n = board.shape[0]
    cards = np.full((n, max_cards), "", dtype="U2")
    for i, text in enumerate(board):
        if not text:
            continue
        parts = text.split()
        cards[i, : len(parts)] = parts
    return parse_cards(cards)


def _straight_lookup() -> np.ndarray:
    """table[mask] = top rank index of the best straight in `mask`, else -1.

    The wheel (A-2-3-4-5) is represented with the five as its top card.
    """
    table = np.full(1 << 13, -1, dtype=np.int8)
    patterns = []
    for top in range(4, 13):  # 6-high .. ace-high, top is the rank index
        patterns.append((sum(1 << (top - k) for k in range(5)), top))
    wheel = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
    patterns.append((wheel, 3))
    for mask in range(1 << 13):
        best = -1
        for pattern, top in patterns:
            if mask & pattern == pattern and top > best:
                best = top
        table[mask] = best
    return table


STRAIGHT_TABLE = _straight_lookup()


def _high_bit_lookup() -> np.ndarray:
    """table[mask] = index of the highest set bit, or -1 for an empty mask."""
    table = np.full(1 << 13, -1, dtype=np.int8)
    for mask in range(1, 1 << 13):
        table[mask] = mask.bit_length() - 1
    return table


HIGH_BIT_TABLE = _high_bit_lookup()


def _run_lookup() -> np.ndarray:
    """table[mask] = longest run of consecutive ranks in `mask`, ace also low."""
    table = np.zeros(1 << 13, dtype=np.int8)
    for mask in range(1 << 13):
        bits = [(mask >> r) & 1 for r in range(13)]
        bits = [bits[12]] + bits  # ace can play below the deuce
        best = cur = 0
        for b in bits:
            cur = cur + 1 if b else 0
            best = max(best, cur)
        table[mask] = best
    return table


RUN_TABLE = _run_lookup()


def draw_features(rank: np.ndarray, suit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drawing potential of hole cards (columns 0-1) plus a partial board.

    Returns (cards in the most common suit, longest run of consecutive ranks,
    whether a hole card belongs to that suit). Four suited cards or a four-card
    run is a draw: a call that looks unjustified by made-hand strength alone is
    often just a draw, and without this the policy model reads it as surprise.
    """
    rank = np.asarray(rank, dtype=np.int8)
    suit = np.asarray(suit, dtype=np.int8)
    n, k = rank.shape
    present = rank >= 0
    rows = np.repeat(np.arange(n, dtype=np.int64), k)[present.ravel()]
    suit_counts = np.bincount(rows * 4 + suit.ravel()[present.ravel()].astype(np.int64),
                              minlength=n * 4).reshape(n, 4)
    top_suit = suit_counts.argmax(axis=1)
    rank_counts = np.bincount(rows * 13 + rank.ravel()[present.ravel()].astype(np.int64),
                              minlength=n * 13).reshape(n, 13)
    mask = (rank_counts > 0).astype(np.int32) @ (1 << np.arange(13, dtype=np.int32))
    hole_in_suit = (suit[:, :2] == top_suit[:, None]).any(axis=1)
    return (suit_counts.max(axis=1).astype(np.int8), RUN_TABLE[mask],
            hole_in_suit.astype(np.int8))


def _pack(category: np.ndarray, kickers: np.ndarray) -> np.ndarray:
    """Pack a category and up to five kicker ranks into one comparable integer."""
    score = category.astype(np.int64)
    for i in range(kickers.shape[1]):
        score = score * _KICKER_BASE + (kickers[:, i].astype(np.int64) + 1)
    return score


def eval_best5(rank: np.ndarray, suit: np.ndarray) -> np.ndarray:
    """Best five-card score for each row of a (n, k) card matrix, k in 5..7.

    Cards with rank -1 are ignored, so ragged inputs can be padded with -1.
    Returns int64 scores where a higher value is a stronger hand.
    """
    rank = np.asarray(rank, dtype=np.int8)
    suit = np.asarray(suit, dtype=np.int8)
    n, k = rank.shape
    present = rank >= 0

    # bincount on a flattened (row, value) index is far faster than np.add.at,
    # which matters because this runs over 12M seat rows.
    rows = np.repeat(np.arange(n, dtype=np.int64), k)[present.ravel()]
    counts = np.bincount(
        rows * 13 + rank.ravel()[present.ravel()].astype(np.int64), minlength=n * 13
    ).reshape(n, 13).astype(np.int8)
    suit_counts = np.bincount(
        rows * 4 + suit.ravel()[present.ravel()].astype(np.int64), minlength=n * 4
    ).reshape(n, 4).astype(np.int8)

    rank_mask = (counts > 0).astype(np.int32) @ (1 << np.arange(13, dtype=np.int32))
    straight_top = STRAIGHT_TABLE[rank_mask]

    # Rank order by (count desc, rank desc) puts quads/trips/pairs first, then kickers.
    order_key = counts.astype(np.int16) * 16 + np.arange(13, dtype=np.int16)[None, :]
    order = np.argsort(-order_key, axis=1, kind="stable")
    sorted_ranks = order.astype(np.int8)
    sorted_counts = np.take_along_axis(counts, order, axis=1)

    c0, c1 = sorted_counts[:, 0], sorted_counts[:, 1]
    r = sorted_ranks

    category = np.full(n, CAT_HIGH, dtype=np.int8)
    kickers = np.full((n, 5), -1, dtype=np.int8)

    high = np.ones(n, dtype=bool)
    kickers[high] = r[high, :5]

    is_pair = c0 == 2
    one_pair = is_pair & (c1 == 1)
    category[one_pair] = CAT_PAIR
    kickers[one_pair, :4] = r[one_pair, :4]
    kickers[one_pair, 4] = -1

    two_pair = is_pair & (c1 == 2)
    category[two_pair] = CAT_TWO_PAIR
    kickers[two_pair, :2] = r[two_pair, :2]
    # Seven cards can hold three pairs, and then the kicker is the best card
    # outside the top two pairs - which may be a lone high card rather than the
    # third pair's rank. Sorting by (count, rank) alone gets this wrong.
    remaining = rank_mask & ~(1 << r[:, 0].astype(np.int32)) & ~(1 << r[:, 1].astype(np.int32))
    kickers[two_pair, 2] = HIGH_BIT_TABLE[remaining[two_pair]]
    kickers[two_pair, 3:] = -1

    is_trips = c0 == 3
    trips_only = is_trips & (c1 == 1)
    category[trips_only] = CAT_TRIPS
    kickers[trips_only, :3] = r[trips_only, :3]
    kickers[trips_only, 3:] = -1

    boat = is_trips & (c1 >= 2)
    category[boat] = CAT_FULL_HOUSE
    kickers[boat, :2] = r[boat, :2]
    kickers[boat, 2:] = -1

    quads = c0 == 4
    category[quads] = CAT_QUADS
    kickers[quads, 0] = r[quads, 0]
    # Same trap as two pair: with quads plus a pair, the (count, rank) order puts
    # the pair ahead of a higher lone card, which is not the best kicker.
    outside_quads = rank_mask & ~(1 << r[:, 0].astype(np.int32))
    kickers[quads, 1] = HIGH_BIT_TABLE[outside_quads[quads]]
    kickers[quads, 2:] = -1

    has_straight = straight_top >= 0
    take_straight = has_straight & (category < CAT_STRAIGHT)
    category[take_straight] = CAT_STRAIGHT
    kickers[take_straight, 0] = straight_top[take_straight]
    kickers[take_straight, 1:] = -1

    flush_suit = np.argmax(suit_counts, axis=1)
    has_flush = suit_counts[np.arange(n), flush_suit] >= 5
    if has_flush.any():
        idx = np.flatnonzero(has_flush)
        fs = flush_suit[idx][:, None]
        in_flush = (suit[idx] == fs) & present[idx]
        bits = np.where(in_flush, 1 << np.maximum(rank[idx], 0).astype(np.int32), 0)
        fmask = np.bitwise_or.reduce(bits, axis=1)
        sf_top = STRAIGHT_TABLE[fmask]

        # Highest five flush ranks, read straight out of the bitmask. A stable
        # argsort over rank-descending bit columns puts the set bits first.
        set_bits = (fmask[:, None] >> np.arange(12, -1, -1)[None, :]) & 1
        order = np.argsort(-set_bits, axis=1, kind="stable")[:, :5]
        top5 = np.where(
            np.take_along_axis(set_bits, order, axis=1) == 1,
            (12 - order).astype(np.int8),
            -1,
        ).astype(np.int8)

        plain_flush = sf_top < 0
        rows = idx[plain_flush]
        category[rows] = CAT_FLUSH
        kickers[rows] = top5[plain_flush]

        rows = idx[~plain_flush]
        category[rows] = CAT_STRAIGHT_FLUSH
        kickers[rows, 0] = sf_top[~plain_flush]
        kickers[rows, 1:] = -1

    return _pack(category, kickers)


def preflop_equity_table(n_opponents: int, trials: int = 20_000, seed: int = 0) -> np.ndarray:
    """All-in equity of each of the 169 starting hands against random holdings.

    Indexed like `preflop_bucket`. Monte Carlo over the evaluator above: deal a
    board and `n_opponents` random hands from the 50 unseen cards, credit a win
    as 1 and a split as 1/(number of tied winners).

    The bucket id itself is not ordered by strength - suited and offsuit hands
    are interleaved - so a model given only the id cannot tell 72o from AKs
    without memorising all 169 values. Equity is the ordinal quantity it needs.
    """
    rng = np.random.default_rng(seed)
    table = np.full(169, np.nan)
    need = 5 + 2 * n_opponents
    for hi in range(13):
        for lo in range(hi + 1):
            for suited in ((False,) if hi == lo else (True, False)):
                hole = [(hi, 0), (lo, 0 if suited else 1)]
                hole_ids = {r * 4 + s for r, s in hole}
                deck = np.array([c for c in range(52) if c not in hole_ids])
                picks = deck[np.argsort(rng.random((trials, deck.size)), axis=1)[:, :need]]
                board = picks[:, :5]

                def score(cards: np.ndarray) -> np.ndarray:
                    return eval_best5((cards // 4).astype(np.int8), (cards % 4).astype(np.int8))

                hole_matrix = np.tile([[hi * 4, lo * 4 + (0 if suited else 1)]], (trials, 1))
                hero = score(np.concatenate([hole_matrix, board], axis=1))
                best_opp = np.zeros(trials, dtype=np.int64)
                opp_scores = []
                for k in range(n_opponents):
                    opp = score(np.concatenate([picks[:, 5 + 2 * k : 7 + 2 * k], board], axis=1))
                    opp_scores.append(opp)
                    best_opp = np.maximum(best_opp, opp)
                ties = 1 + sum((o == hero).astype(np.int32) for o in opp_scores)
                share = np.where(hero > best_opp, 1.0, np.where(hero == best_opp, 1.0 / ties, 0.0))

                bucket = hi * 13 + lo if (suited or hi == lo) else lo * 13 + hi
                table[bucket] = share.mean()
    return table


def preflop_bucket(rank: np.ndarray, suit: np.ndarray) -> np.ndarray:
    """Map two hole cards to one of the 169 strategically distinct starting hands.

    Index layout: `hi * 13 + lo` for suited/pairs, `lo * 13 + hi` for offsuit, so
    the value is a stable id in 0..168 that never mixes suited with offsuit.
    """
    hi = np.maximum(rank[:, 0], rank[:, 1]).astype(np.int32)
    lo = np.minimum(rank[:, 0], rank[:, 1]).astype(np.int32)
    suited = rank[:, 0] != rank[:, 1]
    suited &= suit[:, 0] == suit[:, 1]
    return np.where(suited, hi * 13 + lo, lo * 13 + hi).astype(np.int16)
