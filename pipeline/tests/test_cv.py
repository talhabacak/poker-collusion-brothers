import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol.cv import paired_table_bootstrap


def _data(n_tables=200, per_table=30, seed=1):
    rng = np.random.default_rng(seed)
    tables = np.repeat(np.arange(n_tables), per_table)
    y = (rng.random(tables.size) < 0.05).astype(np.int8)
    return rng, tables, y


def test_identical_scores_give_zero_delta():
    rng, tables, y = _data()
    s = rng.random(y.size) + y
    r = paired_table_bootstrap(y, s, s, tables, n_boot=50)
    assert r["delta"] == 0.0 and r["ci95"] == (0.0, 0.0) and r["p_improves"] == 0.0


def test_clearly_better_score_is_detected():
    rng, tables, y = _data()
    noise = rng.random(y.size)
    weak = noise + 0.3 * y
    strong = noise + 1.0 * y
    r = paired_table_bootstrap(y, weak, strong, tables, n_boot=100)
    assert r["delta"] > 0.1
    assert r["ci95"][0] > 0 and r["p_improves"] == 1.0
