"""Action-sequence tokens for a (pair, hand), as sparse count features.

Every action of the hand becomes

    street | role | action | strength bucket | facing aggression

where role is C for either pair member and O for anyone else, the strength
bucket comes from the actor's own cards at that street (preflop heads-up equity
below 0.45 / 0.6 / above, postflop high card / one pair / better), and facing
is whether any bet or raise came earlier in the hand. A hand's features are its
unigram, bigram and trigram counts plus the presence of each three-action
pattern of who acted (pair member 1, pair member 2, outsider).

Vectorised in polars so millions of candidate rows fit in memory; the result is
a CSR matrix that LightGBM reads directly. `scripts/32_sequence_tokens.py`
holds the original per-row implementation this reproduces.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import sparse

ACT = {"fold": "F", "check": "X", "call": "C", "bet": "B", "raise": "R", "all_in": "A"}


def prepare_actions(actions: pl.DataFrame) -> pl.DataFrame:
    """Reduce an action frame to what tokens need.

    Expects hand_id, action_no, player_id, street (0-3), action,
    preflop_equity_hu, hand_category and prior_aggression.
    """
    bucket = (pl.when(pl.col("street") == 0)
              .then(pl.when(pl.col("preflop_equity_hu") < 0.45).then(pl.lit("w"))
                    .when(pl.col("preflop_equity_hu") < 0.6).then(pl.lit("m")).otherwise(pl.lit("s")))
              .otherwise(pl.when(pl.col("hand_category") == 0).then(pl.lit("w"))
                         .when(pl.col("hand_category") == 1).then(pl.lit("m")).otherwise(pl.lit("s"))))
    return actions.select(
        "hand_id", "action_no", "player_id",
        pl.col("street").cast(pl.Utf8).alias("st"),
        pl.concat_str(pl.col("action").replace_strict(ACT), bucket,
                      (pl.col("prior_aggression") > 0).cast(pl.Int8).cast(pl.Utf8)).alias("tail"),
    )


def candidate_counts(cands: pl.DataFrame, acts: pl.DataFrame) -> pl.DataFrame:
    """Token counts per candidate row. `cands` needs cid, hand_id, p1, p2."""
    a = (cands.select("cid", "hand_id", "p1", "p2")
         .join(acts, on="hand_id", how="inner")
         .sort("cid", "action_no")
         .with_columns(
             pl.when(pl.col("player_id") == pl.col("p1")).then(pl.lit("1"))
             .when(pl.col("player_id") == pl.col("p2")).then(pl.lit("2")).otherwise(pl.lit("o")).alias("who"))
         .with_columns(
             pl.concat_str("st", pl.when(pl.col("who") == "o").then(pl.lit("O")).otherwise(pl.lit("C")), "tail")
             .alias("tok"))
         .select("cid", "tok", "who"))
    nxt = lambda c, k: pl.col(c).shift(-k).over("cid")
    grams = a.with_columns(
        pl.concat_str(pl.col("tok"), pl.lit("|"), nxt("tok", 1)).alias("bi"),
        pl.concat_str(pl.col("tok"), pl.lit("|"), nxt("tok", 1), pl.lit("|"), nxt("tok", 2)).alias("tri"),
        pl.concat_str(pl.lit("who:"), pl.col("who"), nxt("who", 1), nxt("who", 2)).alias("who3"),
    )
    counted = pl.concat([
        grams.select("cid", pl.col("tok").alias("token")),
        grams.select("cid", pl.col("bi").alias("token")),
        grams.select("cid", pl.col("tri").alias("token")),
    ]).drop_nulls().group_by("cid", "token").len("n")
    # Who-patterns count once per hand, as in the reference implementation.
    who = grams.select("cid", pl.col("who3").alias("token")).drop_nulls().unique().with_columns(
        pl.lit(1, dtype=pl.UInt32).alias("n"))
    return pl.concat([counted, who])


def vocabulary(counts: pl.DataFrame, min_rows: int) -> list[str]:
    """Tokens present in at least `min_rows` candidate rows, sorted."""
    return sorted(counts.group_by("token").len("rows").filter(pl.col("rows") >= min_rows)["token"].to_list())


def to_csr(counts: pl.DataFrame, vocab: list[str], n_rows: int) -> sparse.csr_matrix:
    index = pl.DataFrame({"token": vocab, "col": np.arange(len(vocab), dtype=np.int32)})
    m = counts.join(index, on="token", how="inner")
    return sparse.csr_matrix(
        (m["n"].to_numpy().astype(np.float32), (m["cid"].to_numpy(), m["col"].to_numpy())),
        shape=(n_rows, len(vocab)))
