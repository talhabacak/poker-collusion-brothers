"""Stage 8: hard validation of a submission file before it is uploaded.

Checks the format rules Kaggle enforces, and the two evidence rules it does not
announce loudly: a cited hand must belong to the evaluation phase, and both
players of the pair must actually have been seated in it. A row that breaks
either earns nothing, so it is worth failing locally instead.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C

EVIDENCE_COLS = [f"evidence_hand_{i}" for i in range(1, 6)]


def validate(path: Path) -> dict:
    t0 = time.time()
    sub = pl.read_csv(path)
    report: dict[str, object] = {"path": str(path)}
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        report[name] = bool(ok)
        if not ok:
            failures.append(f"{name}{': ' + detail if detail else ''}")

    check("column_order", list(sub.columns) == list(C.SUBMISSION_COLUMNS), str(sub.columns))
    check("row_count", sub.height == C.N_EVAL_PAIRS, f"{sub.height}")

    eval_pairs = pl.read_csv(C.EVAL_PAIRS)
    check("pair_ids_unique", sub["pair_id"].n_unique() == sub.height)
    check("pair_ids_match", set(sub["pair_id"]) == set(eval_pairs["pair_id"]))

    check("no_nulls", sub.null_count().to_numpy().sum() == 0)
    risk = sub["risk_score"]
    check("risk_in_unit_interval", bool(risk.min() >= 0.0 and risk.max() <= 1.0),
          f"[{risk.min()}, {risk.max()}]")
    check("risk_not_constant", risk.n_unique() > 1000, f"{risk.n_unique()} distinct")
    check("behavior_vocabulary",
          set(sub["predicted_behavior"]).issubset(set(C.BEHAVIOR_VALUES)),
          str(sorted(set(sub["predicted_behavior"]))))

    long = sub.select("pair_id", *EVIDENCE_COLS).unpivot(
        index="pair_id", on=EVIDENCE_COLS, variable_name="slot", value_name="hand_id"
    ).filter(pl.col("hand_id") != C.NO_EVIDENCE)

    dup = long.group_by("pair_id", "hand_id").len().filter(pl.col("len") > 1)
    check("no_duplicate_evidence_within_row", dup.height == 0, f"{dup.height} duplicates")

    hands = pl.scan_parquet(C.HANDS).select("hand_id", "phase").collect()
    eval_hand_ids = hands.filter(pl.col("phase") == "evaluation").select("hand_id")
    unknown = long.join(hands, on="hand_id", how="anti")
    check("evidence_hands_exist", unknown.height == 0, f"{unknown.height} unknown ids")
    wrong_phase = long.join(eval_hand_ids, on="hand_id", how="anti").join(
        hands, on="hand_id", how="semi")
    check("evidence_hands_are_evaluation_phase", wrong_phase.height == 0,
          f"{wrong_phase.height} development hands")

    # Both players of the pair must have been seated in every cited hand.
    keyed = eval_pairs.select("pair_id", "player_1", "player_2")
    seats = pl.scan_parquet(C.SEATS).select("hand_id", "player_id")
    cited = long.join(keyed, on="pair_id", how="left")
    seated = (
        cited.lazy()
        .join(seats, on="hand_id", how="left")
        .group_by("pair_id", "hand_id", "player_1", "player_2")
        .agg(
            (pl.col("player_id") == pl.col("player_1").first()).any().alias("has_1"),
            (pl.col("player_id") == pl.col("player_2").first()).any().alias("has_2"),
        )
        .filter(~(pl.col("has_1") & pl.col("has_2")))
        .collect()
    )
    check("evidence_hands_are_shared", seated.height == 0, f"{seated.height} non-shared")

    report["rows"] = sub.height
    report["evidence_rows"] = long.height
    report["pairs_with_evidence"] = long["pair_id"].n_unique()
    report["behavior_counts"] = {
        k: int(v) for k, v in zip(*[s.to_list() for s in
        sub.group_by("predicted_behavior").len().sort("len", descending=True)])
    }
    report["seconds"] = round(time.time() - t0, 1)
    report["passed"] = not failures
    report["failures"] = failures
    return report


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else C.SUBMISSIONS / "submission.csv"
    report = validate(path)
    out = C.ARTIFACTS / f"validation_{path.stem}.json"
    out.write_text(json.dumps(report, indent=2))
    for key, value in report.items():
        if key not in ("failures", "behavior_counts"):
            print(f"  {key}: {value}")
    print(f"  behavior_counts: {report['behavior_counts']}")
    if report["failures"]:
        print("\nFAILED:")
        for f in report["failures"]:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll submission checks passed.")


if __name__ == "__main__":
    main()
