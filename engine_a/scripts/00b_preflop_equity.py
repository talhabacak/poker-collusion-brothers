"""Stage 0b: starting-hand equity for all 169 preflop buckets.

Computed once and cached. Two opponents counts are kept: heads-up equity is the
standard strength ordering, and equity against five random hands reflects a
full six-handed pot, where suited and connected hands gain relative to big
off-suit cards.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C
from pokercol.cards import preflop_equity_table

OUT = C.CACHE / "preflop_equity.parquet"


def main() -> None:
    t0 = time.time()
    hu = preflop_equity_table(n_opponents=1, trials=30_000, seed=C.SEED)
    six = preflop_equity_table(n_opponents=5, trials=20_000, seed=C.SEED + 1)
    df = pl.DataFrame({
        "preflop_bucket": np.arange(169, dtype=np.int16),
        "preflop_equity_hu": hu,
        "preflop_equity_6max": six,
    }).with_columns(
        (pl.col("preflop_equity_hu").rank("average") / 169).alias("preflop_strength_pct")
    )
    df.write_parquet(OUT)
    print(df.sort("preflop_equity_hu", descending=True).head(5))
    print(df.sort("preflop_equity_hu").head(3))
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
