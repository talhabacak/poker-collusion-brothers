"""Official metric implementations.

    score = 0.70 * PairAP + 0.20 * EvidenceMAP@5 + 0.10 * BehaviorMAP

These were written against the rules stated on the competition and starter pages,
before the host published the scorer itself. They now follow the published code,
checked against it on the edge cases where two reasonable implementations part
company - an absent behaviour family, a target pair with no gold evidence, heavy
ties in the risk column - by `scripts/127_official_metric_parity.py`, which reads
the host's notebook at run time rather than copying it here.

One deliberate difference remains. The host rejects a submission that repeats an
evidence hand within a row, and says nothing about a hand the pair never played;
`evidence_map5` scores both as misses instead, so an experiment that produces
them is penalised rather than crashing. Submissions are checked for the first by
`scripts/08_validate_submission.py` before they go out.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision, ties broken deterministically by ascending index."""
    y_true = np.asarray(y_true).astype(np.int8)
    score = np.asarray(score, dtype=np.float64)
    if y_true.sum() == 0:
        return 0.0
    order = np.lexsort((np.arange(score.size), -score))
    hits = y_true[order] == 1
    cum = np.cumsum(hits)
    precision = cum / np.arange(1, hits.size + 1)
    return float(precision[hits].sum() / hits.sum())


def evidence_map5(
    submitted: dict[str, list[str]],
    relevant: dict[str, set[str]],
    valid_hands: dict[str, set[str]] | None = None,
) -> float:
    """Mean average precision at 5 over target pairs.

    `submitted` maps pair_id to its ordered evidence slots (with `NO_EVIDENCE`
    entries already removed or left in - both are handled). `relevant` maps a
    target pair_id to its planted evidence hands. `valid_hands`, when given,
    maps pair_id to the hands that pair is actually allowed to cite; anything
    outside it is treated as a miss, as is a repeated hand.

    The average runs over every key of `relevant`, so callers must pass *all*
    target pairs, not only the ones that have gold hands: the host averages over
    every positive pair and a positive with no gold contributes zero. Passing a
    pair with an empty set scores it that way.
    """
    if not relevant:
        return 0.0
    total = 0.0
    for pair_id, gold in relevant.items():
        if not gold:
            continue
        picks = [h for h in submitted.get(pair_id, []) if h != C.NO_EVIDENCE][:5]
        seen: set[str] = set()
        hits = 0
        precision_sum = 0.0
        for rank, hand in enumerate(picks, start=1):
            duplicate = hand in seen
            seen.add(hand)
            allowed = valid_hands is None or hand in valid_hands.get(pair_id, set())
            if not duplicate and allowed and hand in gold:
                hits += 1
                precision_sum += hits / rank
        total += precision_sum / min(len(gold), 5)
    return total / len(relevant)


def behavior_map(
    y_family: np.ndarray,
    predicted_family: np.ndarray,
    risk: np.ndarray,
    families: tuple[str, ...] = C.FAMILIES,
) -> float:
    """Macro average precision across the three disclosed families.

    Within a family, a pair is scored by its risk only when that family was the
    one predicted; otherwise it contributes zero. A confident risk score routed
    to the wrong family is therefore worth nothing, which is why routing matters
    more than its 10% weight suggests.

    The mean is always over all three families. A family absent from `y_family`
    contributes zero rather than being dropped, which is what the host does and
    is the difference that matters on any slice of the data - a leave-one-family-
    out mixture, say - where one of them is missing.
    """
    y_family = np.asarray(y_family)
    predicted_family = np.asarray(predicted_family)
    risk = np.asarray(risk, dtype=np.float64)
    scores = []
    for family in families:
        target = (y_family == family).astype(np.int8)
        if target.sum() == 0:
            scores.append(0.0)
            continue
        routed = np.where(predicted_family == family, risk, 0.0)
        scores.append(average_precision(target, routed))
    return float(np.mean(scores)) if scores else 0.0


def official_score(pair_ap: float, evidence: float, behavior: float) -> float:
    return C.W_PAIR * pair_ap + C.W_EVIDENCE * evidence + C.W_BEHAVIOR * behavior
