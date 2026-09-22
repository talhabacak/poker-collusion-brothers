"""Stage 4c: turn directed action surprise into pair features.

`action_surprise.parquet` holds, for each hand, how improbable each player's
actions were and which opponent's aggression they were answering. Rolled up to
(actor, target) over a table, that answers: does this player play more strangely
against this one opponent than against the other twenty-nine?

Both directions are kept. A donor/receiver relationship is one-sided - only the
donor plays strangely - while mutual soft play shows on both sides.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C

OUT = C.CACHE / "surprise_pair.parquet"
EPS = 1e-9

METRICS = ["surprise_vs_target", "actions_vs_target", "aggression_excess_vs_target"]


def directed_totals() -> pl.DataFrame:
    hands = pl.scan_parquet(C.HANDS).select("hand_id", "table_id", "phase")
    float_cols = ["surprise_vs_target", "surprise_vs_target_max", "aggression_excess_vs_target",
                  "fold_surprise_vs_target", "call_surprise_vs_target", "aggr_surprise_vs_target"]
    return (
        pl.scan_parquet(C.CACHE / "action_surprise.parquet")
        # Floating-point sums depend on the order rows are added. Summing in
        # float64 over a fixed row order makes the totals bit-identical across
        # runs; float32 accumulated in join order differed by up to 5e-5.
        .with_columns([pl.col(c).cast(pl.Float64) for c in float_cols])
        .join(hands, on="hand_id", how="inner")
        .sort("hand_id", "player_id", "responding_to")
        .group_by("phase", "table_id", "player_id", "responding_to", maintain_order=True)
        .agg(
            pl.col("surprise_vs_target").sum().alias("sv_sum"),
            pl.col("surprise_vs_target").max().alias("sv_max"),
            pl.col("surprise_vs_target_max").max().alias("sv_hand_max"),
            pl.col("actions_vs_target").sum().alias("sv_actions"),
            pl.col("aggression_excess_vs_target").sum().alias("ax_sum"),
            pl.col("fold_surprise_vs_target").sum().alias("fs_sum"),
            pl.col("fold_surprise_vs_target").max().alias("fs_max"),
            pl.col("call_surprise_vs_target").sum().alias("cs_sum"),
            pl.col("aggr_surprise_vs_target").sum().alias("as_sum"),
            pl.col("folds_vs_target").sum().alias("n_folds"),
            pl.len().alias("sv_hands"),
        )
        .collect()
        .sort("phase", "table_id", "player_id", "responding_to")
    )


def main() -> None:
    t0 = time.time()
    d = directed_totals()
    print(f"directed (actor, target) rows: {d.height:,} ({time.time() - t0:.0f}s)")

    totals = d.group_by("phase", "table_id", "player_id", maintain_order=True).agg(
        pl.col("sv_sum").sum().alias("tot_sv"),
        pl.col("sv_actions").sum().alias("tot_actions"),
        pl.col("ax_sum").sum().alias("tot_ax"),
        pl.col("fs_sum").sum().alias("tot_fs"),
        pl.col("cs_sum").sum().alias("tot_cs"),
        pl.col("n_folds").sum().alias("tot_folds"),
    )
    d = d.join(totals, on=["phase", "table_id", "player_id"], how="left")

    actions = d["sv_actions"].to_numpy().astype(np.float64)
    field_actions = (d["tot_actions"] - d["sv_actions"]).to_numpy().astype(np.float64)
    field_sv = (d["tot_sv"] - d["sv_sum"]).to_numpy().astype(np.float64)
    field_ax = (d["tot_ax"] - d["ax_sum"]).to_numpy().astype(np.float64)

    # Surprise per action, against this opponent versus against everyone else.
    per_action = d["sv_sum"].to_numpy() / np.maximum(actions, 1.0)
    field_rate = field_sv / np.maximum(field_actions, 1.0)
    delta = per_action - field_rate
    # Scale the gap by exposure so a two-action sample cannot dominate.
    excess = delta * np.sqrt(np.minimum(actions, 400.0))

    ax_per_action = d["ax_sum"].to_numpy() / np.maximum(actions, 1.0)
    ax_field = field_ax / np.maximum(field_actions, 1.0)

    # Folding improbably is a concession; calling improbably is a pay-off.
    n_folds = d["n_folds"].to_numpy().astype(np.float64)
    field_folds = (d["tot_folds"] - d["n_folds"]).to_numpy().astype(np.float64)
    fs_per_fold = d["fs_sum"].to_numpy() / np.maximum(n_folds, 1.0)
    fs_field = (d["tot_fs"] - d["fs_sum"]).to_numpy() / np.maximum(field_folds, 1.0)
    cs_per_action = d["cs_sum"].to_numpy() / np.maximum(actions, 1.0)
    cs_field = (d["tot_cs"] - d["cs_sum"]).to_numpy() / np.maximum(field_actions, 1.0)

    d = d.with_columns(
        pl.Series("sv_per_action", per_action),
        pl.Series("sv_field_rate", field_rate),
        pl.Series("sv_delta", delta),
        pl.Series("sv_excess", excess),
        pl.Series("ax_per_action", ax_per_action),
        pl.Series("ax_delta", ax_per_action - ax_field),
        pl.Series("fs_per_fold", fs_per_fold),
        pl.Series("fs_delta", fs_per_fold - fs_field),
        pl.Series("fs_excess", (fs_per_fold - fs_field) * np.sqrt(np.minimum(n_folds, 200.0))),
        pl.Series("cs_delta", cs_per_action - cs_field),
        pl.Series("cs_excess", (cs_per_action - cs_field) * np.sqrt(np.minimum(actions, 400.0))),
    )

    cols = ["sv_sum", "sv_max", "sv_hand_max", "sv_actions", "sv_per_action",
            "sv_field_rate", "sv_delta", "sv_excess", "ax_sum", "ax_per_action", "ax_delta",
            "fs_sum", "fs_max", "fs_per_fold", "fs_delta", "fs_excess",
            "cs_sum", "cs_delta", "cs_excess", "as_sum"]
    keyed = d.select(
        "phase", "table_id",
        pl.min_horizontal("player_id", "responding_to").alias("p1"),
        pl.max_horizontal("player_id", "responding_to").alias("p2"),
        *cols,
    )
    folded = keyed.group_by("phase", "table_id", "p1", "p2", maintain_order=True).agg(
        *[pl.col(c).max().alias(f"{c}_hi") for c in cols],
        *[pl.col(c).min().alias(f"{c}_lo") for c in cols],
        *[pl.col(c).sum().alias(f"{c}_sum2")
          for c in ("sv_excess", "sv_delta", "ax_delta", "fs_excess", "cs_excess")],
    )
    folded.write_parquet(OUT, compression="zstd")
    print(f"wrote {OUT}: {folded.height:,} pairs x {folded.width} cols "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
