"""Stage 4b-bootstrap: freeze the set of unknown pairs a reference model rates as
near-certain colluders.

Two later consumers need it: the hand-suspicion model leaves these pairs out of
its random unlabelled negatives, and the cleaned pair surrogate drops them from
its evaluation negatives. On a clean run the set does not exist yet, so
`run_all.py` builds hand suspicion once without extra negatives, freezes the set
here from an out-of-fold pair model, then rebuilds hand suspicion properly.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C
from pokercol.cv import EXCLUSION_PATH, table_folds

import importlib
rt = importlib.import_module("14_pair_retest")

THRESHOLD = 0.5


def main() -> None:
    t0 = time.time()
    dev = rt.load()
    x, _ = rt.matrix(dev)
    y = dev["y"].to_numpy()
    folds = table_folds(dev["table_id"].to_numpy(), rt.N_FOLDS)
    oof = rt.lgb_oof(x, y, folds)
    flagged = (y == -1) & (oof > THRESHOLD)
    dev.filter(pl.Series(flagged)).select("table_id", "p1", "p2").write_parquet(EXCLUSION_PATH)
    print(f"froze {int(flagged.sum())} unknown pairs scoring > {THRESHOLD} "
          f"-> {EXCLUSION_PATH} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
