"""Fold construction and the development-side scoring surrogate.

Two rules drive the design.

*Group by table.* Players never move tables, so every pair lives on exactly one
of 400 tables. Splitting anywhere finer would let a model see a colluder's other
pairs - and the field baselines computed from them - while scoring a held-out
pair.

*Score the whole population, not the labelled slice.* The official Pair AP ranks
112,540 pairs of which a few hundred are targets. Measuring AP on 372 positives
against 1,488 hand-picked negatives answers a much easier question and saturates
near 1.0. The surrogate here scores every development pair that would have
passed the evaluation filter, counting the 372 disclosed targets as positives
and everything else as negative, which matches the real ranking problem's shape.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import config as C

# The evaluation universe keeps pairs with at least this many shared hands.
MIN_SHARED_HANDS = 38


def candidate_pairs(pairs: pl.DataFrame, phase: str) -> pl.DataFrame:
    """Pairs in `phase` that pass the evaluation universe's co-seating filter."""
    return pairs.filter(
        (pl.col("phase") == phase) & (pl.col("shared_hands") >= MIN_SHARED_HANDS)
    )


# Pair-level feature tables joined onto stage 04's output, in one place so a new
# stage cannot be wired into training but forgotten at scoring time.
PAIR_FEATURE_CACHES = ("hand_suspicion", "hand_suspicion_family", "surprise_pair")
PAIR_KEY = ["phase", "table_id", "p1", "p2"]


def load_pair_table(required: bool = True) -> pl.DataFrame:
    """Stage 04 pair features with every downstream pair feature table joined.

    `required=False` skips tables that do not exist yet, for bootstrap stages that
    run before the later tables are built. Production scoring keeps the default.
    """
    pairs = pl.read_parquet(C.CACHE / "pair_features.parquet")
    for name in PAIR_FEATURE_CACHES:
        path = C.CACHE / f"{name}.parquet"
        if not path.exists():
            if required:
                raise FileNotFoundError(f"{path} is missing; run the stage that builds it")
            continue
        pairs = pairs.join(pl.read_parquet(path), on=PAIR_KEY, how="left")
    return pairs


def attach_labels(dev: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Add `y`, `label_status` and `behavior_family`; unlabelled pairs stay unknown."""
    keyed = labels.with_columns(
        pl.min_horizontal("player_1", "player_2").alias("p1"),
        pl.max_horizontal("player_1", "player_2").alias("p2"),
    ).select("p1", "p2", "pair_id", "label", "label_status", "behavior_family")
    out = dev.join(keyed, on=["p1", "p2"], how="left")
    # Joins and group-bys do not promise a row order, and bagging samples rows by
    # position, so an unsorted frame makes an otherwise deterministic model vary
    # from run to run. Every training frame leaves here in a fixed order.
    return out.with_columns(
        pl.col("label").fill_null(-1).cast(pl.Int8).alias("y"),
        pl.col("label_status").fill_null("unknown"),
        pl.col("behavior_family").fill_null("unknown"),
    ).sort("table_id", "p1", "p2")


_CANONICAL: dict[tuple[int, int], dict[str, int]] = {}


def canonical_fold_map(n_folds: int = 5, seed: int = C.SEED) -> dict[str, int]:
    """table_id -> fold, built once from every table in the dataset.

    The older behaviour shuffled whatever table list the caller happened to pass,
    so a stage that saw only tables with labels put 339 of 397 of them in a
    different fold from a stage that saw all 400. Supervised caches (hand
    suspicion, family hand scores) were then validated on folds their own
    training rows had been part of. One map keyed by (n_folds, seed) removes
    that: every stage looks the same table up in the same place.
    """
    key = (n_folds, seed)
    if key not in _CANONICAL:
        tables = np.sort(pl.scan_parquet(C.HANDS).select("table_id").unique().collect()["table_id"].to_numpy())
        shuffled = np.random.default_rng(seed).permutation(tables)
        _CANONICAL[key] = {t: i % n_folds for i, t in enumerate(shuffled)}
    return _CANONICAL[key]


def table_folds(tables: np.ndarray, n_folds: int = 5, seed: int = C.SEED,
                canonical: bool = True) -> np.ndarray:
    """Assign each row a fold via its table, so a table never spans folds.

    `canonical=False` reproduces the pre-2026-09-18 behaviour, where the fold of
    a table depended on which other tables the caller passed in.
    """
    if canonical:
        fold_of_table = canonical_fold_map(n_folds, seed)
        missing = [t for t in np.unique(tables) if t not in fold_of_table]
        if missing:
            extra = {t: (len(fold_of_table) + i) % n_folds for i, t in enumerate(sorted(missing))}
            fold_of_table = {**fold_of_table, **extra}
    else:
        unique = np.unique(tables)
        shuffled = np.random.default_rng(seed).permutation(unique)
        fold_of_table = {t: i % n_folds for i, t in enumerate(shuffled)}
    return np.array([fold_of_table[t] for t in tables], dtype=np.int8)


EXCLUSION_PATH = C.ARTIFACTS / "surrogate_exclusion.parquet"


def cleaned_mask(dev: pl.DataFrame) -> np.ndarray:
    """Rows kept by the cleaned surrogate: everything except the frozen set of
    unknown pairs a reference model scored as near-certain colluders.

    The plain population surrogate counts those pairs - most plausibly
    undisclosed targets - as false positives, which is why it reads ~0.67 when
    the leaderboard implies ~0.94. Dropping the 150 of them brings it to ~0.95.
    The set is frozen on disk so every experiment is scored on identical rows.
    """
    excluded = pl.read_parquet(EXCLUSION_PATH).with_columns(pl.lit(True).alias("_x"))
    flags = dev.select("table_id", "p1", "p2").join(
        excluded, on=["table_id", "p1", "p2"], how="left"
    )["_x"].fill_null(False).to_numpy()
    return ~flags


def population_average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision with deterministic handling of ties.

    Equal scores are broken by ascending index, which mirrors the deterministic
    `pair_id` tie-break the submission format implies, so a model that emits
    many identical scores is not flattered by a lucky ordering.
    """
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    order = np.lexsort((np.arange(score.size), -score))
    hits = y[order] == 1
    if not hits.any():
        return 0.0
    cum_hits = np.cumsum(hits)
    precision = cum_hits / np.arange(1, hits.size + 1)
    return float(precision[hits].sum() / hits.sum())


def paired_table_bootstrap(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                           tables: np.ndarray, n_boot: int = 500, seed: int = 0) -> dict:
    """AP(b) - AP(a) with a confidence interval from resampling whole tables.

    Pairs at one table share players, so rows are not independent; resampling
    tables keeps that dependence. Both scores see the same resamples, which is
    what makes a ±0.001 difference separable at all.
    """
    y = np.asarray(y)
    tables = np.asarray(tables)
    uniq, inverse = np.unique(tables, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(uniq.size + 1))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([order[starts[t]:starts[t + 1]] for t in pick])
        deltas[b] = (population_average_precision(y[idx], score_b[idx])
                     - population_average_precision(y[idx], score_a[idx]))
    point = population_average_precision(y, score_b) - population_average_precision(y, score_a)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {"delta": float(point), "ci95": (float(lo), float(hi)),
            "p_improves": float((deltas > 0).mean())}
