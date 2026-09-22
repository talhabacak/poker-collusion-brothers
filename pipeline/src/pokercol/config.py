"""Paths and competition constants.

Every number here is asserted against the raw files by `scripts/00_verify_data.py`,
so a silent upstream change fails loudly instead of quietly shifting results.
"""
from __future__ import annotations

import os
from pathlib import Path

# REPO NOTE (reproduction repo): the only change to this file. In the competition
# repository engine A was the repository root; here both engines share one data
# directory at the repository root, so the root is taken from POKERCOL_ROOT when
# it is set (run_all.py sets it) and falls back to the original expression.
ROOT = Path(os.environ.get("POKERCOL_ROOT") or Path(__file__).resolve().parents[2])
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
ARTIFACTS = ROOT / "artifacts"
SUBMISSIONS = ROOT / "submissions"

for _d in (CACHE, ARTIFACTS, SUBMISSIONS):
    _d.mkdir(parents=True, exist_ok=True)

PLAYERS = RAW / "players.parquet"
HANDS = RAW / "hands.parquet"
SEATS = RAW / "seats.parquet"
ACTIONS = RAW / "actions.parquet"
DEV_LABELS = RAW / "development_labels.csv"
DEV_EVIDENCE = RAW / "development_evidence.csv"
EVAL_PAIRS = RAW / "evaluation_pairs.csv"
SAMPLE_SUB = RAW / "sample_submission.csv"

SEED = 20260914

# Disclosed target families. `other_coordination` exists only in the evaluation
# phase and earns no Behavior MAP credit, so it is never a training target.
FAMILIES = ("directed_transfer", "soft_play", "coordinated_isolation")
BEHAVIOR_VALUES = ("none", *FAMILIES, "other_coordination")

SUBMISSION_COLUMNS = (
    "pair_id",
    "risk_score",
    "predicted_behavior",
    "evidence_hand_1",
    "evidence_hand_2",
    "evidence_hand_3",
    "evidence_hand_4",
    "evidence_hand_5",
)
NO_EVIDENCE = "NO_EVIDENCE"

# Official weights.
W_PAIR, W_EVIDENCE, W_BEHAVIOR = 0.70, 0.20, 0.10

# Shape facts verified against the raw files.
N_PLAYERS = 12_000
N_HANDS = 2_000_000
N_DEV_HANDS = 1_200_000
N_EVAL_HANDS = 800_000
N_SEATS = 12_000_000
N_ACTIONS = 18_609_028
N_TABLES = 400
N_EVAL_PAIRS = 112_540
N_DEV_LABELS = 1_860
N_DEV_POSITIVES = 372
N_DEV_NEGATIVES = 1_488
SEATS_PER_HAND = 6

N_THREADS = min(20, os.cpu_count() or 8)

# LightGBM's default multithreaded histogram construction can produce slightly
# different trees run to run. Winner verification needs the selected submission
# reproduced from code, so every production model trains in deterministic mode.
# Results are reproducible for a fixed thread count on the same library version.
LGB_REPRO = dict(deterministic=True, force_col_wise=True)
