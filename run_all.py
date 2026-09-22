"""Rebuild both selected submissions from the raw competition files, stage by stage.

    python run_all.py --list        # every stage, what it writes, and whether that exists
    python run_all.py --dry-run     # the plan and the environment of each stage, run nothing
    python run_all.py               # phases A, C, D, V: v64's risk column, the MIL evidence
                                    # column, both finals, validation
    python run_all.py --with-engine-b          # also rebuild v55 from engine B first
    python run_all.py --from a156              # resume at a stage
    python run_all.py --only C,D               # one or more phases
    python run_all.py --with-measurements      # stage 533's full four-arm protocol, not
                                               # only the arm the submission needs

Phases

    B  engine B: the pair, behaviour and evidence columns of v55 (`tarik_v55_best.csv`).
       Off by default - see README, "What is and is not rebuilt here". The shipped file
       is in `submissions/`, so every later phase runs without it.
    A  engine A: the caches behind v64's risk column, and that column.
    C  the cross: the action-level MIL head, trained in engine B's tables and folds, and
       the primary final it produces.
    D  the second final: v64's risk under the primary's behaviour and evidence.
    V  validation and checksums.

Every stage runs as its own process, in a fixed order, and the run stops at the first
failure. A manifest with the library versions, per-stage timings, input checksums and
both submissions' checksums is written to `artifacts/run_manifest.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
INTERIM = ROOT / "data" / "interim"
SUBMISSIONS = ROOT / "submissions"
ARTIFACTS = ROOT / "artifacts"

RAW_FILES = ("hands.parquet", "seats.parquet", "actions.parquet", "players.parquet",
             "development_labels.csv", "development_evidence.csv", "evaluation_pairs.csv",
             "sample_submission.csv")

PRIMARY = "tarik_v55_mil_evidence.csv"
SECOND = "hybrid_riskA_behB_evMIL.csv"
V55 = "tarik_v55_best.csv"

# The evidence side of v55, as recorded in the teammate's run_A3.sh (the commit that
# scored 0.91748), and the clean-chain settings its pair side inherits from run_clean.sh.
B_BASE = {"USE_VAL": "0", "USE_CT": "0", "USE_CTR": "0", "PYTHONUNBUFFERED": "1",
          "PYTHONIOENCODING": "utf-8"}
B_RERANK_SEEDS = "42,7,2024,11,99,5"


@dataclass
class Stage:
    key: str
    phase: str
    script: str
    what: str
    outputs: tuple[str, ...] = ()
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    optional: bool = False


STAGES: list[Stage] = [
    # ---------------------------------------------------------------- phase B
    Stage("b01", "B", "engine_b/src/01_prep.py",
          "ids, preflop equity, postflop strength, action context, player style",
          ("data/interim/hands.parquet", "data/interim/players.parquet",
           "data/interim/seats_feat.parquet", "data/interim/actions_feat.parquet"), env=B_BASE),
    Stage("b02", "B", "engine_b/src/02_policy.py",
          "population action policy and bet-size model; out-of-fold surprisal",
          ("data/interim/actions_scored.parquet",), env=B_BASE),
    Stage("b03", "B", "engine_b/src/03_pairhand.py",
          "features for every (pair, shared hand) row, partner-active conditioning",
          ("data/interim/pairhand",), env=B_BASE),
    Stage("b03b", "B", "engine_b/src/03b_relational.py",
          "relational likelihood ratio against the player's own baseline",
          ("data/interim/pair_relational.parquet",), env=B_BASE),
    Stage("b03c", "B", "engine_b/src/03c_extra.py",
          "partner-independent features: third-player folds, multiway aggression",
          ("data/interim/extra",), env=B_BASE),
    Stage("b13", "B", "engine_b/src/13_oppaware.py",
          "opponent-aware policy residuals per directed pair (ported from engine A)",
          ("data/interim/oppaware_hand.parquet", "data/interim/oppaware_pair.parquet"), env=B_BASE),
    Stage("b15", "B", "engine_b/src/15_oppcond_policy.py",
          "opponent-conditioned policy and its residuals",
          ("data/interim/oppcond_hand.parquet", "data/interim/oppcond_pair.parquet"), env=B_BASE),
    Stage("b04", "B", "engine_b/src/04_hand_scorer.py",
          "4-class evidence-hand scorer, 5 folds by table",
          ("data/interim/handscores",), env=B_BASE),
    Stage("b04c", "B", "engine_b/src/04c_action_model.py",
          "action-level collusion model, one head per family (MIL refinement off)",
          ("data/interim/actmodel",), env=B_BASE),
    Stage("b04b", "B", "engine_b/src/04b_stage1b.py",
          "within-pair evidence ranker on pair-normalised features",
          ("data/interim/stage1b",), env={**B_BASE, "FAMILY_SOURCE": "model"}),
    Stage("b05", "B", "engine_b/src/05_pair_model.py",
          "pair risk and behaviour columns (v55: opponent-aware block on)",
          ("submission_v55_pairs.csv",),
          env={**B_BASE, "U_WEIGHT": "0.0", "MIXED_NEG_W": "0", "USE_OA": "1",
               "DUMP_DEV": "1", "DUMP_ALL": "1", "BEH_LOFO": "1",
               "SUB_OUT": "submission_v55_pairs.csv"}),
    Stage("b05b", "B", "engine_b/src/05b_evidence_reranker.py",
          "evidence reranker, six seeds, opponent-aware and opponent-conditioned blocks on",
          (f"submissions/{V55}",),
          env={**B_BASE, "USE_OAH": "1", "USE_OCH": "1", "RERANK_SEEDS": B_RERANK_SEEDS,
               "RERANK_IN": "submission_v55_pairs.csv", "RERANK_OUT": f"submissions/{V55}"}),
    Stage("b06", "B", "engine_b/src/06_validate_submission.py", "engine B's own validator",
          (), args=(f"submissions/{V55}",), env=B_BASE),

    # ---------------------------------------------------------------- phase A
    Stage("a00b", "A", "engine_a/scripts/00b_preflop_equity.py",
          "all-in equity of the 169 starting hands", ("data/cache/preflop_equity.parquet",)),
    Stage("a01", "A", "engine_a/scripts/01_pair_universe.py",
          "co-seating universe and shared-hand counts", ("data/cache/pair_universe.parquet",)),
    Stage("a02", "A", "engine_a/scripts/02_seat_features.py",
          "per-seat strength by street, actions, fold attribution",
          ("data/cache/seat_features.parquet",)),
    Stage("a02b", "A", "engine_a/scripts/02b_action_surprise.py",
          "action policy model and -log P(observed) per action",
          ("data/cache/action_surprise.parquet", "data/cache/action_surprise_seat.parquet")),
    Stage("a03", "A", "engine_a/scripts/03_pair_hands.py",
          "pair x shared-hand interactions (30M rows)", ("data/cache/pair_hands",)),
    Stage("a04", "A", "engine_a/scripts/04_pair_features.py",
          "pair features, partner-versus-field improbability",
          ("data/cache/pair_features.parquet",)),
    Stage("a04c", "A", "engine_a/scripts/04c_surprise_pair.py",
          "directed surprise contrasts", ("data/cache/surprise_pair.parquet",)),
    Stage("a045boot", "A", "engine_a/scripts/045_hand_suspicion.py",
          "hand suspicion without unlabelled negatives, to seed the exclusion set",
          (), env={"HS_EXTRA_UNKNOWN": "0"}),
    Stage("a045b", "A", "engine_a/scripts/045b_freeze_exclusion.py",
          "freeze the set of unlabelled pairs the surrogate must not count as negatives",
          ("artifacts/surrogate_exclusion.parquet",)),
    Stage("a045", "A", "engine_a/scripts/045_hand_suspicion.py",
          "hand suspicion bag features", ("data/cache/hand_suspicion.parquet",)),
    Stage("a20", "A", "engine_a/scripts/20_family_hand_suspicion.py",
          "per-family hand suspicion", ("data/cache/hand_suspicion_family.parquet",)),
    Stage("a04e", "A", "engine_a/scripts/04e_relational.py",
          "relational likelihood ratios over 96 decision contexts",
          ("data/cache/relational_pair.parquet",)),
    Stage("a49", "A", "engine_a/scripts/49_dyadic_residual.py",
          "dyadic residual features", ("data/cache/dyadic_pair.parquet",)),
    Stage("a150", "A", "engine_a/scripts/150_opponent_aware_policy.py",
          "opponent-aware policy residuals, full and no-opponent-style variants",
          ("data/cache/oppaware_pair_nostyle.parquet",)),
    Stage("a156", "A", "engine_a/scripts/156_generate_v64_union_oppaware.py",
          "v64's risk column: the 311-feature union recipe plus the opponent-aware block",
          ("submissions/_risk_v64_union_oppaware.csv",)),

    # ---------------------------------------------------------------- phase C
    Stage("c531", "C", "engine_a/scripts/531_b_action_bag_frame.py",
          "action bags for every candidate hand, built from engine B's tables and folds",
          ("data/cache/531_b_action_bags.parquet",)),
    Stage("c532pool", "C", "engine_a/scripts/532_b_noisy_or_mil.py",
          "noisy-OR against log-sum-exp pooling, chosen on folds 0-3 and then frozen",
          (), args=("--part", "pooling")),
    Stage("c532hold", "C", "engine_a/scripts/532_b_noisy_or_mil.py",
          "the strict holdout protocol: fold 4 takes no part in any fit",
          (), args=("--part", "holdout")),
    Stage("c532cols", "C", "engine_a/scripts/532_b_noisy_or_mil.py",
          "the eight MIL columns, nested so a row never reads a head that saw its fold",
          ("artifacts/532_mil_full_nested_s20260914.parquet",), args=("--part", "columns")),
    Stage("c533", "C", "engine_a/scripts/533_b_mil_evidence_arm.py",
          "the seed-bagged MIL block inside engine B's reranker",
          ("data/interim/533_blocks/full_mil_bag.parquet",),
          args=("--protocols", "full", "--arms", "bag")),
    Stage("c534", "C", "engine_a/scripts/534_b_mil_candidate.py",
          "PRIMARY FINAL: v55 with the MIL evidence column, risk and behaviour copied as text",
          (f"submissions/{PRIMARY}",)),

    # ---------------------------------------------------------------- phase D
    Stage("d608", "D", "engine_a/scripts/608_build_hybrid_riskA.py",
          "SECOND FINAL: the primary with v64's risk column in place of engine B's",
          (f"submissions/{SECOND}",)),

    # ---------------------------------------------------------------- phase V
    Stage("v1", "V", "engine_a/scripts/08_validate_submission.py",
          "format, vocabulary and evidence-hand checks on the primary",
          (), args=(f"submissions/{PRIMARY}",)),
    Stage("v2", "V", "engine_a/scripts/08_validate_submission.py",
          "the same checks on the second final", (), args=(f"submissions/{SECOND}",)),
    Stage("v3", "V", "verify/check_submissions.py",
          "sha256 of both files against the ones that were submitted", ()),
]

MEASUREMENT_ARGS = ("--protocols", "full,holdout", "--arms", "baseline,mil,placebo,bag")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def selected(args: argparse.Namespace) -> list[Stage]:
    phases = set((args.only or ("B,A,C,D,V" if args.with_engine_b else "A,C,D,V")).split(","))
    stages = [s for s in STAGES if s.phase in phases]
    if args.from_stage:
        keys = [s.key for s in stages]
        if args.from_stage not in keys:
            raise SystemExit(f"--from {args.from_stage}: not one of {keys}")
        stages = stages[keys.index(args.from_stage):]
    if args.with_measurements:
        stages = [Stage(**{**s.__dict__, "args": MEASUREMENT_ARGS}) if s.key == "c533" else s
                  for s in stages]
    return stages


def stage_env(stage: Stage) -> dict[str, str]:
    env = {**os.environ, "POKERCOL_ROOT": str(ROOT), "PYTHONIOENCODING": "utf-8",
           "TARIK_INTERIM": str(INTERIM), "TARIK_RUN": str(ROOT),
           "TARIK_HARNESS": "engine_b/src/05b_mil.py", **stage.env}
    return env


def run_stage(stage: Stage) -> float:
    print(f"\n=== [{stage.phase}] {stage.key}  {stage.script}"
          + (" " + " ".join(stage.args) if stage.args else ""), flush=True)
    print(f"    {stage.what}", flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, "-u", str(ROOT / stage.script), *stage.args],
                            cwd=ROOT, env=stage_env(stage))
    elapsed = time.time() - t0
    if result.returncode:
        raise SystemExit(f"stage failed: {stage.key} ({stage.script}) "
                         f"exit {result.returncode} after {elapsed:.0f}s")
    missing = [o for o in stage.outputs if not exists(o)]
    if missing:
        raise SystemExit(f"stage {stage.key} finished but did not write: {missing}")
    print(f"=== {stage.key} ok ({elapsed:.0f}s)", flush=True)
    return elapsed


def print_list(stages: list[Stage]) -> None:
    print(f"{'key':<10}{'phase':<7}{'output exists':<15}script")
    for s in stages:
        state = "-" if not s.outputs else ("yes" if all(exists(o) for o in s.outputs) else "no")
        print(f"{s.key:<10}{s.phase:<7}{state:<15}{s.script}"
              + (" " + " ".join(s.args) if s.args else ""))
        print(f"{'':<32}{s.what}")
        for o in s.outputs:
            print(f"{'':<32}-> {o}{'' if exists(o) else '  (missing)'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--from", dest="from_stage", default="")
    ap.add_argument("--with-engine-b", action="store_true")
    ap.add_argument("--with-measurements", action="store_true")
    args = ap.parse_args()

    for d in (CACHE, INTERIM, SUBMISSIONS, ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)
    stages = selected(args)

    if args.list:
        print_list(stages)
        return
    if args.dry_run:
        print_list(stages)
        print("\nenvironment set for every stage: POKERCOL_ROOT, TARIK_INTERIM, TARIK_RUN, "
              "TARIK_HARNESS")
        for s in stages:
            if s.env:
                print(f"  {s.key}: " + " ".join(f"{k}={v}" for k, v in s.env.items()
                                                if k not in ("PYTHONUNBUFFERED", "PYTHONIOENCODING")))
        return

    missing = [f for f in RAW_FILES if not (RAW / f).exists()]
    if missing:
        raise SystemExit(f"missing competition files in {RAW}: {missing}\n"
                         "kaggle competitions download -c detect-suspicious-value-transfers-in-poker "
                         "-p data/raw && unzip ...")
    if any(s.phase in ("C", "D") for s in stages) and not (SUBMISSIONS / V55).exists():
        # Phase C reads v55 and copies its risk and behaviour columns through as text.
        # When phase B has not been run, the submitted copy is the input.
        shipped = SUBMISSIONS / "submitted" / V55
        if not shipped.exists():
            raise SystemExit(f"submissions/{V55} is missing: it is either shipped in "
                             f"submissions/submitted/ or rebuilt by phase B (--with-engine-b)")
        (SUBMISSIONS / V55).write_bytes(shipped.read_bytes())
        print(f"copied the submitted {V55} into submissions/ as the input to phase C")

    started = time.time()
    timings = [{"stage": s.key, "script": s.script, "args": list(s.args), "env": s.env,
                "seconds": round(run_stage(s), 1)} for s in stages]

    manifest = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": {name: metadata.version(name) for name in
                      ("polars", "numpy", "scipy", "lightgbm", "xgboost", "scikit-learn")
                      if _installed(name)},
        "torch": metadata.version("torch") if _installed("torch") else None,
        "stages": timings,
        "total_seconds": round(time.time() - started, 1),
        "inputs_sha256": {f: sha256(RAW / f) for f in RAW_FILES if (RAW / f).exists()},
        "submissions_sha256": {name: sha256(SUBMISSIONS / name)
                               for name in (PRIMARY, SECOND, V55) if (SUBMISSIONS / name).exists()},
    }
    out = ARTIFACTS / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nall stages passed in {manifest['total_seconds']:.0f}s; manifest -> {out}")
    for name, digest in manifest["submissions_sha256"].items():
        print(f"  {name}  {digest}")


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
