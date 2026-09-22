"""Assemble a submission from independently tested parts.

    python scripts/build_combo.py --risk FILE --guard {none,lpo,band} --evidence FILE --out FILE

risk_score comes from --risk (then the chosen guard is applied), predicted_behavior
from --behavior (default v24), evidence columns from --evidence (default v24).
Every source must cover the same pair_ids; rows follow the risk file's order.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C

import importlib
g31 = importlib.import_module("generate_v31_lpo_guard")

EV_COLS = [f"evidence_hand_{i}" for i in range(1, 6)]


def arg(name: str, default: str | None = None) -> str:
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    if default is None:
        raise SystemExit(f"missing {name}")
    return default


def write_provenance(path: Path) -> None:
    """Record beside the file what it cannot be asked afterwards.

    The 19 September manifest could verify every submission's contents and none
    of its origins: no build had ever written down the environment it ran under,
    so which `EV_*` settings produced which file was a matter of memory. Anything
    recoverable later - the inputs, their checksums - is left to the manifest;
    what is recorded here is only what disappears when the process exits.
    """
    env = {k: v for k, v in os.environ.items()
           if k.startswith(("EV_", "PAIR_", "BEHAVIOR_", "OTHER_"))}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.ROOT,
                          capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=C.ROOT,
                           capture_output=True, text=True)
    record = {
        "output": path.name,
        "argv": sys.argv[1:],
        "environment": env,
        "commit": head.stdout.strip() or "unknown",
        "working_tree_clean": not dirty.stdout.strip(),
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    (path.with_suffix(".provenance.json")).write_text(json.dumps(record, indent=2))


def main() -> None:
    risk_src = pl.read_csv(C.SUBMISSIONS / arg("--risk"), infer_schema_length=0)
    if "--risk2" in sys.argv:
        a = float(arg("--blend", "0.5"))
        b = float(arg("--blend3", "0.0"))
        rk = lambda v: (np.argsort(np.argsort(v, kind="stable"), kind="stable") + 1) / v.size
        joined = risk_src.select("pair_id", "risk_score").join(
            pl.read_csv(C.SUBMISSIONS / arg("--risk2"), infer_schema_length=0).select("pair_id", pl.col("risk_score").alias("r2")),
            on="pair_id", how="left")
        mix = (1 - a - b) * rk(joined["risk_score"].cast(float).to_numpy()) + a * rk(joined["r2"].cast(float).to_numpy())
        if b:
            third = pl.read_csv(C.SUBMISSIONS / arg("--risk3"), infer_schema_length=0).select("pair_id", pl.col("risk_score").alias("r3"))
            mix = mix + b * rk(joined.join(third, on="pair_id", how="left")["r3"].cast(float).to_numpy())
        risk_src = risk_src.with_columns(pl.Series("risk_score", rk(mix)))
        print(f"blended risk: {1 - a - b:.2f} x {arg('--risk')} + {a:.2f} x {arg('--risk2')}"
              + (f" + {b:.2f} x {arg('--risk3')}" if b else ""))
    beh_src = pl.read_csv(C.SUBMISSIONS / arg("--behavior", "v24_v7feats_relational.csv"), infer_schema_length=0)
    ev_src = pl.read_csv(C.SUBMISSIONS / arg("--evidence", "v24_v7feats_relational.csv"), infer_schema_length=0)
    guard = arg("--guard", "none")
    out = (risk_src.select("pair_id", "risk_score")
           .join(beh_src.select("pair_id", "predicted_behavior"), on="pair_id", how="left")
           .join(ev_src.select("pair_id", *EV_COLS), on="pair_id", how="left"))
    assert out.height == C.N_EVAL_PAIRS and out.null_count().sum_horizontal().item() == 0
    risk = out["risk_score"].cast(float).to_numpy()
    if guard != "none":
        eval_pairs = pl.read_csv(C.EVAL_PAIRS).with_columns(
            pl.min_horizontal("player_1", "player_2").alias("p1"), pl.max_horizontal("player_1", "player_2").alias("p2"))
        lpo = pl.read_parquet(C.CACHE / "lpo_pair.parquet").filter(pl.col("phase") == "evaluation").select("p1", "p2", "lpo_ps_top5")
        hsf = pl.read_parquet(C.CACHE / "hand_suspicion_family.parquet").filter(pl.col("phase") == "evaluation").select(
            "p1", "p2", pl.max_horizontal("hsdir_top5", "hssoft_top5", "hsiso_top5").alias("hmax"))
        rows = out.select("pair_id").join(eval_pairs.select("pair_id", "p1", "p2"), on="pair_id", how="left") \
            .join(lpo, on=["p1", "p2"], how="left").join(hsf, on=["p1", "p2"], how="left")
        c = 0.3
        hmax = rows["hmax"].fill_null(0).to_numpy()
        band = (hmax >= 0.3) & (hmax < 0.95)
        if guard == "band":
            c = np.where(band, 0.0, 0.3)
        elif guard == "bandonly":
            # The 2026-09-16 ablation: a guard that lifts 16-21% of all pairs cost
            # 0.0009 on the public split. This one only reaches the band where the
            # evaluation excess sits - 96 development pairs, 51 of them positives.
            c = np.where(band, 0.0, np.inf)
        risk = g31.lpo_guard(risk, rows["lpo_ps_top5"].to_numpy().astype(float), c)
    out = out.with_columns(pl.Series("risk_score", risk)).select(
        "pair_id", "risk_score", "predicted_behavior", *EV_COLS)
    path = C.SUBMISSIONS / arg("--out")
    out.write_csv(path)
    write_provenance(path)
    print(f"wrote {path}: risk from {arg('--risk')} + guard {guard}, behaviour {arg('--behavior', 'v24')}, "
          f"evidence {arg('--evidence', 'v24')}")


if __name__ == "__main__":
    main()
