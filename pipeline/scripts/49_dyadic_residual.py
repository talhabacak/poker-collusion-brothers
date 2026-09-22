"""Stage 4f: two-way dyadic residuals and relative-partner ranks.

Most pair features compare a player with their own field one way at a time. The
research report's operator removes both players' general tendencies at once:

    r_AB = x_AB - median_C x_AC - median_C x_CB + median_table x
    z_AB = r_AB / (1.4826 * MAD_table(r) + eps)

"A may be a maniac and B may fold a lot; what is left is what appears only when
the two are together." Computed inside each table over every co-seated pair,
for the pair signals that carry the model. Alongside, relative partner ranks:
where x_AB stands among A's pairs and among B's, and the gap to each player's
second-best pair - the information the failed hard-uniqueness post-process
carried, left for the model to weigh.

Writes dyadic_pair.parquet keyed by phase, table_id, p1, p2.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C

OUT = C.CACHE / "dyadic_pair.parquet"
KEY = ["phase", "table_id", "p1", "p2"]
SIGNALS = ["ps_top5", "ps_top3", "ps_max", "pfs_top3", "ax_pair_sum", "ps_on_events_max",
           "transfer_top5", "transfer_event_rate", "paid_any", "fold_better_any", "conceded_any",
           "sv_excess_hi", "sv_excess_lo", "fs_excess_hi", "cs_excess_hi", "s_outsider_folds_to_hi",
           "d_relay_rate_hi", "d_squeeze_rate_hi", "hs_top5", "hsdir_top5", "hssoft_top5", "hsiso_top5",
           "rel_g_tail_hi", "rel_g_per_action_hi", "rel_fold_dev_hi", "rel_call_dev_hi", "rel_aggr_dev_hi"]
RANKED = ["ps_top5", "hs_top5", "rel_g_tail_hi", "transfer_top5"]


def main() -> None:
    t0 = time.time()
    pairs = pl.read_parquet(C.CACHE / "pair_features.parquet")
    for name in ("hand_suspicion", "hand_suspicion_family", "relational_pair"):
        pairs = pairs.join(pl.read_parquet(C.CACHE / f"{name}.parquet"), on=KEY, how="left")
    sig = [s for s in SIGNALS if s in pairs.columns]
    df = pairs.select(*KEY, *sig).with_columns([pl.col(s).cast(pl.Float64).fill_null(0.0) for s in sig])
    # Both players' medians over their pairs at this table: stack (A, B) and (B, A).
    long = pl.concat([df.select(*KEY, pl.col("p1").alias("who"), *sig), df.select(*KEY, pl.col("p2").alias("who"), *sig)])
    med_player = long.group_by("phase", "table_id", "who").agg([pl.col(s).median().alias(f"m_{s}") for s in sig])
    med_table = df.group_by("phase", "table_id").agg([pl.col(s).median().alias(f"t_{s}") for s in sig])
    d = (df.join(med_player.rename({"who": "p1", **{f"m_{s}": f"a_{s}" for s in sig}}), on=["phase", "table_id", "p1"], how="left")
           .join(med_player.rename({"who": "p2", **{f"m_{s}": f"b_{s}" for s in sig}}), on=["phase", "table_id", "p2"], how="left")
           .join(med_table, on=["phase", "table_id"], how="left")
           .with_columns([(pl.col(s) - pl.col(f"a_{s}") - pl.col(f"b_{s}") + pl.col(f"t_{s}")).alias(f"dy_{s}") for s in sig]))
    d = d.with_columns([
        ((pl.col(f"dy_{s}") - pl.col(f"dy_{s}").median().over("phase", "table_id"))
         / (1.4826 * (pl.col(f"dy_{s}") - pl.col(f"dy_{s}").median().over("phase", "table_id")).abs().median().over("phase", "table_id") + 1e-6))
        .alias(f"dyz_{s}") for s in sig])
    # Relative partner ranks and gaps.
    rank_cols = []
    for s in RANKED:
        for side, who in (("a", "p1"), ("b", "p2")):
            grp = ["phase", "table_id", who]
            d = d.with_columns(pl.col(s).rank("ordinal", descending=True).over(grp).alias(f"rk_{side}_{s}"))
            # gap to the player's best *other* pair: x_AB - max_{C != B} x_AC
            best = long.group_by("phase", "table_id", "who").agg(pl.col(s).max().alias("best"), pl.col(s).sort(descending=True).slice(1, 1).first().alias("second"))
            d = d.join(best.rename({"who": who, "best": f"_best_{side}_{s}", "second": f"_second_{side}_{s}"}), on=grp, how="left").with_columns(
                pl.when(pl.col(s) >= pl.col(f"_best_{side}_{s}")).then(pl.col(s) - pl.col(f"_second_{side}_{s}").fill_null(0.0))
                .otherwise(pl.col(s) - pl.col(f"_best_{side}_{s}")).alias(f"gap_{side}_{s}"))
            rank_cols += [f"rk_{side}_{s}", f"gap_{side}_{s}"]
    out_cols = [f"dyz_{s}" for s in sig] + [f"dy_{s}" for s in sig] + rank_cols
    d = d.with_columns([pl.min_horizontal(f"gap_a_{s}", f"gap_b_{s}").alias(f"gap_min_{s}") for s in RANKED]
                       + [pl.max_horizontal(f"rk_a_{s}", f"rk_b_{s}").alias(f"rk_max_{s}") for s in RANKED])
    out_cols += [f"gap_min_{s}" for s in RANKED] + [f"rk_max_{s}" for s in RANKED]
    out = d.select(*KEY, *out_cols).sort(KEY)
    out.write_parquet(OUT, compression="zstd")
    print(f"wrote {OUT}: {out.height:,} pairs x {out.width} cols ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
