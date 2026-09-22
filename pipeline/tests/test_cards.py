"""Correctness tests for the vectorised evaluator.

The 7-card path is checked against an independent brute-force reference that
enumerates all C(7,5) subsets, so a packing or ordering bug cannot pass quietly.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol.cards import (  # noqa: E402
    CAT_FLUSH,
    CAT_FULL_HOUSE,
    CAT_HIGH,
    CAT_PAIR,
    CAT_QUADS,
    CAT_STRAIGHT,
    CAT_STRAIGHT_FLUSH,
    CAT_TRIPS,
    CAT_TWO_PAIR,
    eval_best5,
    parse_cards,
    preflop_bucket,
    split_board,
)

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def score_of(cards: str) -> int:
    rank, suit = parse_cards(np.array([cards.split()]))
    return int(eval_best5(rank, suit)[0])


def category_of(cards: str) -> int:
    # Category is the most significant digit of the base-13 packing.
    return score_of(cards) // (14 ** 5)


def test_categories_are_detected():
    cases = {
        "As Ks Qs Js Ts 2c 3d": CAT_STRAIGHT_FLUSH,
        "7c 7d 7h 7s 2c 3d 9h": CAT_QUADS,
        "8c 8d 8h 4s 4c 2d 9h": CAT_FULL_HOUSE,
        "Ac 9c 7c 4c 2c 3d 8h": CAT_FLUSH,
        "5c 6d 7h 8s 9c 2d 3h": CAT_STRAIGHT,
        "Kc Kd Kh 4s 9c 2d 7h": CAT_TRIPS,
        "Qc Qd 8h 8s 3c 2d 7h": CAT_TWO_PAIR,
        "Jc Jd 9h 5s 3c 2d 7h": CAT_PAIR,
        "Ac Jd 9h 5s 3c 2d 7h": CAT_HIGH,
    }
    for cards, expected in cases.items():
        assert category_of(cards) == expected, cards


def test_wheel_straight_is_five_high():
    wheel = score_of("Ac 2d 3h 4s 5c 9d Kh")
    six_high = score_of("2d 3h 4s 5c 6d 9h Kd")
    assert category_of("Ac 2d 3h 4s 5c 9d Kh") == CAT_STRAIGHT
    assert wheel < six_high, "the wheel must be the weakest straight"


def test_ordering_within_and_across_categories():
    # Kicker comparisons.
    assert score_of("Ac Ad Kh 9s 3c 2d 7h") > score_of("Ac Ad Qh 9s 3c 2d 7h")
    assert score_of("Ac Ad Ah Kh Qs 2d 3c") > score_of("Kc Kd Kh As Qs 2d 3c")
    # Category comparisons.
    assert score_of("2c 2d 3h 3s 9c 5d 7h") > score_of("Ac Ad 9h 5s 3c 2d 7h")
    assert score_of("Ac 9c 7c 4c 2c 3d 8h") > score_of("5c 6d 7h 8s 9c 2d 3h")
    assert score_of("8c 8d 8h 4s 4c 2d 9h") > score_of("Ac 9c 7c 4c 2c 3d 8h")


def _reference_five(cards: list[tuple[int, int]]) -> tuple:
    """Independent 5-card ranking returning a comparable tuple."""
    ranks = sorted((r for r, _ in cards), reverse=True)
    suits = [s for _, s in cards]
    counts: dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    by_count = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    shape = [c for _, c in by_count]
    ordered = [r for r, _ in by_count]

    flush = len(set(suits)) == 1
    distinct = sorted(set(ranks), reverse=True)
    straight_top = None
    if len(distinct) == 5:
        if distinct[0] - distinct[4] == 4:
            straight_top = distinct[0]
        elif distinct == [12, 3, 2, 1, 0]:
            straight_top = 3

    if flush and straight_top is not None:
        return (CAT_STRAIGHT_FLUSH, straight_top)
    if shape == [4, 1]:
        return (CAT_QUADS, *ordered)
    if shape == [3, 2]:
        return (CAT_FULL_HOUSE, *ordered)
    if flush:
        return (CAT_FLUSH, *distinct)
    if straight_top is not None:
        return (CAT_STRAIGHT, straight_top)
    if shape == [3, 1, 1]:
        return (CAT_TRIPS, *ordered)
    if shape == [2, 2, 1]:
        return (CAT_TWO_PAIR, *ordered)
    if shape == [2, 1, 1, 1]:
        return (CAT_PAIR, *ordered)
    return (CAT_HIGH, *distinct)


def _show(hand: list[tuple[int, int]]) -> str:
    return " ".join(RANKS[r] + SUITS[s] for r, s in hand)


def _brute_force_check(hands: list[list[tuple[int, int]]]) -> None:
    """Assert the evaluator induces the same total order as the reference."""
    rank = np.array([[c[0] for c in h] for h in hands], dtype=np.int8)
    suit = np.array([[c[1] for c in h] for h in hands], dtype=np.int8)
    got = eval_best5(rank, suit)
    ref = [max(_reference_five(list(c)) for c in itertools.combinations(h, 5)) for h in hands]

    failures = []
    for i in range(len(hands)):
        for j in range(i + 1, len(hands)):
            mine = int(got[i] > got[j]) - int(got[i] < got[j])
            theirs = int(ref[i] > ref[j]) - int(ref[i] < ref[j])
            if mine != theirs:
                failures.append(
                    f"{_show(hands[i])} (score {got[i]}, ref {ref[i]}) vs "
                    f"{_show(hands[j])} (score {got[j]}, ref {ref[j]})"
                )
    assert not failures, f"{len(failures)} ordering disagreements, e.g.\n" + "\n".join(failures[:5])


def test_matches_brute_force_on_random_seven_card_hands():
    rng = np.random.default_rng(7)
    deck = [(r, s) for r in range(13) for s in range(4)]
    hands = [[deck[i] for i in rng.choice(52, size=7, replace=False)] for _ in range(600)]
    _brute_force_check(hands)


def test_matches_brute_force_on_multi_pair_and_quad_hands():
    """Targeted at the (count, rank) sorting trap: three pairs, and quads + pair.

    In both shapes the best kicker can be a lone high card that sorts *after* a
    paired rank, so these are drawn deliberately rather than left to chance.
    """
    rng = np.random.default_rng(11)
    hands: list[list[tuple[int, int]]] = []
    for _ in range(200):
        ranks = rng.choice(13, size=3, replace=False)
        suits = [rng.choice(4, size=2, replace=False) for _ in ranks]
        hand = [(int(r), int(s)) for r, ss in zip(ranks, suits) for s in ss]
        spare = [(r, s) for r in range(13) for s in range(4) if (r, s) not in hand]
        hand.append(spare[int(rng.integers(len(spare)))])
        hands.append(hand)
    for _ in range(200):
        quad, pair = rng.choice(13, size=2, replace=False)
        hand = [(int(quad), s) for s in range(4)]
        hand += [(int(pair), s) for s in rng.choice(4, size=2, replace=False)]
        spare = [(r, s) for r in range(13) for s in range(4) if (r, s) not in hand]
        hand.append(spare[int(rng.integers(len(spare)))])
        hands.append(hand)
    _brute_force_check(hands)


def test_ragged_input_ignores_padding():
    full = score_of("Ac Ad Kh 9s 3c")
    rank, suit = parse_cards(np.array([["Ac", "Ad", "Kh", "9s", "3c", "", ""]]))
    padded = int(eval_best5(rank, suit)[0])
    assert padded == full


def test_split_board_handles_partial_and_empty_boards():
    rank, suit = split_board(np.array(["Ah Kd 7c", "", "2c 3d 4h 5s 6c"], dtype=object))
    assert (rank[1] == -1).all()
    assert (rank[0][:3] >= 0).all() and (rank[0][3:] == -1).all()
    assert (rank[2] >= 0).all()


def test_preflop_bucket_separates_suited_and_offsuit():
    rank, suit = parse_cards(np.array([["Ah", "Kh"], ["Ah", "Kd"], ["Ah", "As"]]))
    buckets = preflop_bucket(rank, suit)
    assert buckets[0] != buckets[1], "AKs and AKo must not share a bucket"
    assert len(set(buckets.tolist())) == 3
    # A pair maps to the diagonal under both conventions.
    assert buckets[2] == 12 * 13 + 12


def test_preflop_equity_orders_known_hands():
    from pokercol.cards import preflop_equity_table

    table = preflop_equity_table(n_opponents=1, trials=4_000, seed=1)
    assert not np.isnan(table).any(), "every one of the 169 buckets must be filled"

    def eq(cards: str) -> float:
        rank, suit = parse_cards(np.array([cards.split()]))
        return float(table[preflop_bucket(rank, suit)[0]])

    # Textbook heads-up equities: AA ~85%, KK ~82%, AKs ~67%, 72o ~35%.
    assert 0.82 < eq("Ah As") < 0.88
    assert eq("Ah As") > eq("Kh Ks") > eq("Ah Kh") > eq("7h 2d")
    assert 0.31 < eq("7h 2d") < 0.38
    assert eq("Ah Kh") > eq("Ah Kd"), "suited must beat the same ranks offsuit"


def test_draw_features_detect_flush_and_straight_draws():
    from pokercol.cards import draw_features

    rank, suit = parse_cards(np.array([
        ["Ah", "Kh", "2h", "7h", "9c"],   # four hearts, hole cards in the suit
        ["5c", "6d", "7h", "8s", "Kd"],   # open-ended four-card run
        ["Ac", "2d", "3h", "4s", "9d"],   # wheel draw, ace plays low
        ["2c", "9d", "Kh", "5s", "7d"],   # nothing
    ]))
    suited, run, hole_in_suit = draw_features(rank, suit)
    assert suited.tolist()[0] == 4 and hole_in_suit[0] == 1
    assert run.tolist()[1] == 4
    assert run.tolist()[2] == 4
    assert suited[3] <= 2 and run[3] <= 1
