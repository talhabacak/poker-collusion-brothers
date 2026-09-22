"""Stage 480: the action-level bag frame, and the baseline it will be judged against.

The competition says that inside every listed evidence hand there is a visible,
behaviour-specific *action*. The latent structure is therefore

    PAIR -> HAND -> TRIGGER ACTION

and supervision arrives at the hand. Every stage so far has treated a hand as an
atom: `06_evidence_model.py` ranks hands and never asks which action inside one
is the manipulation. Stage 140 already wrote down the consequence - "the positive
set for a family is every action the two pair members took in those hands and is
diluted with ordinary play" - and then lived with the dilution.

This stage builds the frame a differentiable multiple-instance model needs, and
nothing else. No model is fitted here and no label is ever attached to an action.

Bags
----
A bag is a `(pair_id, hand_id)` cell; its instances are the actions the two
members of that pair took in that hand. The same physical hand can be a cell of
several pairs at its table, and the instance set differs between them, so the
pair-relative columns below (`resp_partner`, `partner_aggr_before`, ...) are what
make an instance belong to a *pair's* reading of the hand rather than to the hand.

    label  1   a listed evidence hand of a disclosed target pair
    label  0   any hand of one of the 1,488 confirmed non-target pairs
    label -1   **censored**: every other hand of a positive pair

The censoring is the whole point. A positive pair's unlisted hands are not known
negatives - the organiser listed five, not "the only five" - and training on them
as negatives would teach the model to suppress exactly the hands the reranker is
being asked to find. They are excluded from the loss entirely; they are scored at
inference, because that is where the ranking happens.

Features
--------
Per action, from `data/cache/action_context_{dev}.parquet` (stage 140's build of
stage 02b's context), the six-way action identity, and the population policy's
out-of-fold log-probability from `artifacts/family_llr_clean_logp.npy`. That
vector is fitted per table fold on the action taken, reads no collusion label of
any kind, and so introduces no coupling between folds.

On top of those, the pair-relative columns that make an action a *pair's*: which
player it answers, what the partner had already done when it was taken and what
the partner did afterwards, and how the partner's hand compares.

Folds
-----
Players never move tables, so a pair lives on exactly one table and the canonical
table fold map partitions pairs, hands and actions at once. The audit asserts it
rather than assuming it and the violation counts are written to the report.

    python scripts/480_action_bag_frame.py --part baseline   # reproduce 0.68921
    python scripts/480_action_bag_frame.py --part frame      # build the frame

Writes data/cache/480_action_bags.parquet, artifacts/480_action_bag_frame.json
and artifacts/480_baseline_stage1.npy. Nothing here submits anything.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

for _k, _v in {"EV_SIZING": "1", "EV_TEMPLATES": "1", "EV_SPEC_TUNED": "1"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pokercol import config as C
from pokercol.cv import table_folds

ev = importlib.import_module("06_evidence_model")
s02b = importlib.import_module("02b_action_surprise")
s119 = importlib.import_module("119_iso_pairwise_production_check")

N_FOLDS = 5
FRAME = C.CACHE / "480_action_bags.parquet"
REPORT = C.ARTIFACTS / "480_action_bag_frame.json"
BASE_VEC = C.ARTIFACTS / "480_baseline_stage1.npy"
CTX = C.CACHE / "action_context_dev.parquet"
CLEAN_LOGP = C.ARTIFACTS / "family_llr_clean_logp.npy"

RECORDED_BASELINE = 0.68921
TOLERANCE = 0.0035

# Stage 02b's context, unchanged. `hand_category` and `preflop_bucket` are small
# integers and are left as such - the model below embeds nothing, it standardises,
# and a six-level ordinal costs nothing at this width.
CTX_FEATS = list(s02b.FEATURES)

ACTION_NAMES = list(s02b.ACTIONS)          # fold check call bet raise all_in
AGGRESSIVE = {ACTION_NAMES.index(a) for a in ("bet", "raise", "all_in")}

PAIR_FEATS = [
    "surprise", "amt_bb", "amt_to_pot", "raise_over_call",
    "resp_partner", "resp_outsider", "partner_acted_before", "partner_aggr_before",
    "partner_folded_before", "partner_acts_after", "partner_aggr_after",
    "partner_folds_after", "own_acted_before", "own_aggr_before",
    "turns_since_partner", "is_last_action_of_hand",
    "partner_strength_rank", "own_strength_rank", "strength_rank_gap",
    "partner_preflop_equity", "own_preflop_equity", "preflop_equity_gap",
    "partner_net_bb", "own_net_bb", "partner_folded_hand", "own_folded_hand",
    "outsiders_left", "pair_share_of_actions",
]
ACT_ONEHOT = [f"act_{n}" for n in ACTION_NAMES]
FEATURES = CTX_FEATS + ACT_ONEHOT + PAIR_FEATS


# ------------------------------------------------------------------ baseline

def evidence_frame() -> tuple[pl.DataFrame, list[str], pl.DataFrame]:
    """The production evidence training frame, exactly as stage 391 builds it."""
    labels = pl.read_csv(C.DEV_LABELS).with_columns(
        pl.min_horizontal("player_1", "player_2").alias("p1"),
        pl.max_horizontal("player_1", "player_2").alias("p2"))
    pos = labels.filter(pl.col("label") == 1).select("pair_id", "p1", "p2", "behavior_family")
    evidence = pl.read_csv(C.DEV_EVIDENCE).select("pair_id", "hand_id").unique()
    df = ev.load_pair_hands(pos, "development")
    df = df.join(evidence.with_columns(pl.lit(1, dtype=pl.Int8).alias("is_evidence")),
                 on=["pair_id", "hand_id"], how="left").with_columns(
        pl.col("is_evidence").fill_null(0)).sort("pair_id", "hand_id")
    return df, ev.feature_names(), evidence


def baseline(report: dict) -> None:
    """Reproduce the shipped engine's out-of-fold MAP@5 and assert the record."""
    t0 = time.time()
    df, feats, evidence = evidence_frame()
    folds = table_folds(df["table_id"].to_numpy(), N_FOLDS)
    stage1 = ev.oof_stage1(df, feats, folds, seeds=ev.EVIDENCE_SEEDS)
    np.save(BASE_VEC, stage1)
    fam = df["behavior_family"].to_numpy()
    got = s119.map5(df, stage1, evidence)
    report["baseline"] = {
        "rows": int(df.height), "pairs": int(df["pair_id"].n_unique()),
        "features": len(feats), "seeds": int(ev.EVIDENCE_SEEDS),
        "truncation": int(ev.PARAMS["lambdarank_truncation_level"]),
        "specialist_weight": ev.SPECIALIST_WEIGHT,
        "map5": {"all": round(got, 5),
                 **{f[:4]: round(s119.map5(df, stage1, evidence, subset=fam == f), 5)
                    for f in C.FAMILIES}},
        "recorded": RECORDED_BASELINE, "difference": round(got - RECORDED_BASELINE, 5),
        "seconds": round(time.time() - t0, 1)}
    print(json.dumps(report["baseline"], indent=2), flush=True)
    assert abs(got - RECORDED_BASELINE) < TOLERANCE, \
        f"baseline {got:.5f} does not reproduce {RECORDED_BASELINE}"


# --------------------------------------------------------------------- bags

def pair_keys() -> tuple[pl.DataFrame, pl.DataFrame]:
    labels = pl.read_csv(C.DEV_LABELS).with_columns(
        pl.min_horizontal("player_1", "player_2").alias("p1"),
        pl.max_horizontal("player_1", "player_2").alias("p2"))
    cols = ("pair_id", "p1", "p2", "behavior_family")
    return (labels.filter(pl.col("label") == 1).select(cols),
            labels.filter(pl.col("label") == 0).select(cols))


HAND_COLS = ["hand_id", "table_id", "p1", "p2", "pot_bb", "big_blind",
             "strength_rank_1", "strength_rank_2", "preflop_equity_hu_1",
             "preflop_equity_hu_2", "net_bb_1", "net_bb_2", "folded_1", "folded_2",
             "n_actions_1", "n_actions_2", "players_at_showdown"]


def bag_table() -> pl.DataFrame:
    """One row per (pair_id, hand_id) with the three-valued bag label."""
    pos, neg = pair_keys()
    evidence = pl.read_csv(C.DEV_EVIDENCE).select("pair_id", "hand_id").unique()
    lf = (pl.scan_parquet(C.CACHE / "pair_hands" / "*.parquet")
          .filter(pl.col("phase") == "development").select(HAND_COLS))
    keep = pl.concat([pos, neg]).select("pair_id", "p1", "p2", "behavior_family")
    bags = lf.join(keep.lazy(), on=["p1", "p2"], how="inner").collect()
    pos_ids = set(pos["pair_id"].to_list())
    bags = bags.with_columns(
        pl.col("pair_id").is_in(list(pos_ids)).alias("is_pos_pair")).join(
        evidence.with_columns(pl.lit(1, dtype=pl.Int8).alias("_gold")),
        on=["pair_id", "hand_id"], how="left")
    return bags.with_columns(
        pl.when(pl.col("_gold") == 1).then(pl.lit(1, dtype=pl.Int8))
        .when(~pl.col("is_pos_pair")).then(pl.lit(0, dtype=pl.Int8))
        .otherwise(pl.lit(-1, dtype=pl.Int8)).alias("bag_label")
    ).drop("_gold").sort("pair_id", "hand_id")


def context_with_surprise(hand_ids: pl.Series) -> pl.DataFrame:
    """Stage 140's development action context, restricted to the hands we need,
    with the out-of-fold population-policy surprise attached.

    `family_llr_clean_logp.npy` is positionally aligned with the whole context
    frame, so the surprise column is taken before any filtering.
    """
    ctx = pl.read_parquet(CTX)
    logp = np.load(CLEAN_LOGP)
    assert logp.shape[0] == ctx.height, "the clean log-probability is not aligned"
    ctx = ctx.with_columns(pl.Series("surprise", (-logp).astype(np.float32)))
    wanted = pl.DataFrame({"hand_id": hand_ids.unique()})
    return ctx.join(wanted, on="hand_id", how="inner")


def action_amounts(blinds: pl.DataFrame) -> pl.DataFrame:
    """Bet sizing per action, which stage 02b's context deliberately leaves out."""
    amt = (pl.scan_parquet(C.ACTIONS)
           .select("hand_id", "action_no", "amount", "amount_to", "pot_before", "to_call")
           .join(blinds.lazy(), on="hand_id", how="inner").collect())
    return amt.with_columns(
        (pl.col("amount").fill_null(0.0) / pl.col("big_blind")).cast(pl.Float32).alias("amt_bb"),
        (pl.col("amount_to").fill_null(0.0) / (pl.col("pot_before") + 1e-6))
        .cast(pl.Float32).alias("amt_to_pot"),
        # one big blind in the denominator rather than an epsilon: a raise over a
        # zero to-call is a bet, not a division by nothing.
        (pl.col("amount").fill_null(0.0) / (pl.col("to_call").fill_null(0.0) + pl.col("big_blind")))
        .cast(pl.Float32).alias("raise_over_call"),
    ).select("hand_id", "action_no", "amt_bb", "amt_to_pot", "raise_over_call")


def build_frame(report: dict) -> None:
    t0 = time.time()
    bags = bag_table()
    print(f"bags: {bags.height:,} cells over {bags['pair_id'].n_unique()} pairs "
          f"({time.time() - t0:.0f}s)", flush=True)

    ctx = context_with_surprise(bags["hand_id"])
    print(f"context rows for those hands: {ctx.height:,} ({time.time() - t0:.0f}s)", flush=True)
    # The hand's own length, from every seat's actions rather than the pair's.
    hand_shape = ctx.group_by("hand_id").agg(
        pl.col("action_no").max().alias("_hand_last_action"),
        pl.len().alias("_hand_actions"))
    blinds = bags.select("hand_id", "big_blind").unique(subset=["hand_id"])
    ctx = ctx.join(action_amounts(blinds), on=["hand_id", "action_no"], how="left")
    ctx = ctx.join(hand_shape, on="hand_id", how="left")

    # One row per (pair, hand, member); the member's actions come from the join.
    members = pl.concat([
        bags.select("pair_id", "hand_id", pl.col("p1").alias("player_id"),
                    pl.col("p2").alias("partner_id"), pl.lit(1, dtype=pl.Int8).alias("side")),
        bags.select("pair_id", "hand_id", pl.col("p2").alias("player_id"),
                    pl.col("p1").alias("partner_id"), pl.lit(2, dtype=pl.Int8).alias("side")),
    ])
    rows = members.join(ctx, on=["hand_id", "player_id"], how="inner")
    print(f"instance rows: {rows.height:,} ({time.time() - t0:.0f}s)", flush=True)

    # ---- pair-relative columns, computed inside each (pair, hand) bag.
    key = ["pair_id", "hand_id"]
    rows = rows.sort(key + ["action_no"]).with_columns(
        pl.col("y").is_in(sorted(AGGRESSIVE)).cast(pl.Int8).alias("_aggr"),
        (pl.col("y") == ACTION_NAMES.index("fold")).cast(pl.Int8).alias("_fold"),
        pl.lit(1, dtype=pl.Int8).alias("_one"),
    )
    own = ["pair_id", "hand_id", "player_id"]
    rows = rows.with_columns(
        # prior/after counts over the whole bag and over the actor alone; the
        # difference is the partner's, which is what a colluder reads.
        (pl.col("_one").cum_sum().over(key) - pl.col("_one")).alias("_bag_before"),
        (pl.col("_one").cum_sum().over(own) - pl.col("_one")).alias("own_acted_before"),
        (pl.col("_aggr").cum_sum().over(key) - pl.col("_aggr")).alias("_bag_aggr_before"),
        (pl.col("_aggr").cum_sum().over(own) - pl.col("_aggr")).alias("own_aggr_before"),
        (pl.col("_fold").cum_sum().over(key) - pl.col("_fold")).alias("_bag_fold_before"),
        (pl.col("_fold").cum_sum().over(own) - pl.col("_fold")).alias("_own_fold_before"),
        pl.col("_one").sum().over(key).alias("_bag_total"),
        pl.col("_one").sum().over(own).alias("_own_total"),
        pl.col("_aggr").sum().over(key).alias("_bag_aggr_total"),
        pl.col("_aggr").sum().over(own).alias("_own_aggr_total"),
        pl.col("_fold").sum().over(key).alias("_bag_fold_total"),
        pl.col("_fold").sum().over(own).alias("_own_fold_total"),
        # the action immediately before this one inside the bag, and who took it
        pl.col("action_no").shift(1).over(key).alias("_prev_an"),
        pl.col("player_id").shift(1).over(key).alias("_prev_who"),
    )
    rows = rows.with_columns(
        (pl.col("_bag_before") - pl.col("own_acted_before")).alias("partner_acted_before"),
        (pl.col("_bag_aggr_before") - pl.col("own_aggr_before")).alias("partner_aggr_before"),
        (pl.col("_bag_fold_before") - pl.col("_own_fold_before")).alias("partner_folded_before"),
        ((pl.col("_bag_total") - pl.col("_bag_before") - 1)
         - (pl.col("_own_total") - pl.col("own_acted_before") - 1)).alias("partner_acts_after"),
        ((pl.col("_bag_aggr_total") - pl.col("_bag_aggr_before") - pl.col("_aggr"))
         - (pl.col("_own_aggr_total") - pl.col("own_aggr_before") - pl.col("_aggr")))
        .alias("partner_aggr_after"),
        ((pl.col("_bag_fold_total") - pl.col("_bag_fold_before") - pl.col("_fold"))
         - (pl.col("_own_fold_total") - pl.col("_own_fold_before") - pl.col("_fold")))
        .alias("partner_folds_after"),
        (pl.col("responding_to") == pl.col("partner_id")).cast(pl.Int8).alias("resp_partner"),
        (pl.col("responding_to").is_not_null()
         & (pl.col("responding_to") != pl.col("partner_id"))
         & (pl.col("responding_to") != pl.col("player_id"))).cast(pl.Int8).alias("resp_outsider"),
        (pl.col("action_no") == pl.col("_hand_last_action")).cast(pl.Int8)
        .alias("is_last_action_of_hand"),
        (pl.col("_bag_total") / pl.col("_hand_actions"))
        .cast(pl.Float32).alias("pair_share_of_actions"),
        # turns between this action and the partner's most recent one; -1 when
        # the action immediately before was the actor's own or the bag's first.
        pl.when(pl.col("_prev_who").is_not_null() & (pl.col("_prev_who") != pl.col("player_id")))
        .then(pl.col("action_no") - pl.col("_prev_an"))
        .otherwise(pl.lit(-1)).cast(pl.Float32).alias("turns_since_partner"),
    )

    # ---- hand-level partner context, oriented to the actor.
    hand_ctx = bags.select(
        "pair_id", "hand_id", "strength_rank_1", "strength_rank_2",
        "preflop_equity_hu_1", "preflop_equity_hu_2", "net_bb_1", "net_bb_2",
        "folded_1", "folded_2", "players_at_showdown", "n_actions_1", "n_actions_2")
    rows = rows.join(hand_ctx, on=key, how="left")
    one = pl.col("side") == 1
    rows = rows.with_columns(
        pl.when(one).then(pl.col("strength_rank_1")).otherwise(pl.col("strength_rank_2"))
        .cast(pl.Float32).alias("own_strength_rank"),
        pl.when(one).then(pl.col("strength_rank_2")).otherwise(pl.col("strength_rank_1"))
        .cast(pl.Float32).alias("partner_strength_rank"),
        pl.when(one).then(pl.col("preflop_equity_hu_1")).otherwise(pl.col("preflop_equity_hu_2"))
        .cast(pl.Float32).alias("own_preflop_equity"),
        pl.when(one).then(pl.col("preflop_equity_hu_2")).otherwise(pl.col("preflop_equity_hu_1"))
        .cast(pl.Float32).alias("partner_preflop_equity"),
        pl.when(one).then(pl.col("net_bb_1")).otherwise(pl.col("net_bb_2"))
        .cast(pl.Float32).alias("own_net_bb"),
        pl.when(one).then(pl.col("net_bb_2")).otherwise(pl.col("net_bb_1"))
        .cast(pl.Float32).alias("partner_net_bb"),
        pl.when(one).then(pl.col("folded_1")).otherwise(pl.col("folded_2"))
        .cast(pl.Float32).alias("own_folded_hand"),
        pl.when(one).then(pl.col("folded_2")).otherwise(pl.col("folded_1"))
        .cast(pl.Float32).alias("partner_folded_hand"),
        (pl.col("players_active") - 2).cast(pl.Float32).alias("outsiders_left"),
    ).with_columns(
        (pl.col("own_strength_rank") - pl.col("partner_strength_rank"))
        .cast(pl.Float32).alias("strength_rank_gap"),
        (pl.col("own_preflop_equity") - pl.col("partner_preflop_equity"))
        .cast(pl.Float32).alias("preflop_equity_gap"),
    )
    rows = rows.with_columns(
        [(pl.col("y") == i).cast(pl.Int8).alias(f"act_{n}") for i, n in enumerate(ACTION_NAMES)])

    bag_meta = bags.select("pair_id", "hand_id", "table_id", "behavior_family",
                           "is_pos_pair", "bag_label")
    out = rows.join(bag_meta, on=key, how="left").select(
        ["pair_id", "hand_id", "table_id", "player_id", "partner_id", "action_no",
         "behavior_family", "is_pos_pair", "bag_label"] + FEATURES)
    out = out.with_columns([pl.col(c).cast(pl.Float32) for c in FEATURES]).sort(
        key + ["action_no", "player_id"])
    out.write_parquet(FRAME, compression="zstd")
    print(f"frame written: {out.height:,} rows x {len(FEATURES)} features "
          f"({time.time() - t0:.0f}s)", flush=True)

    audit(out, bags, report)


# -------------------------------------------------------------------- audit

def audit(rows: pl.DataFrame, bags: pl.DataFrame, report: dict) -> None:
    """Fold isolation and bag census, asserted in code and reported as counts."""
    fold_of = dict(zip(*[s.to_list() for s in
                         (pl.Series(np.unique(rows["table_id"].to_numpy())),
                          pl.Series(table_folds(np.unique(rows["table_id"].to_numpy()), N_FOLDS)))]))
    rows = rows.with_columns(pl.col("table_id").replace_strict(fold_of).alias("fold"))

    def spans(frame: pl.DataFrame, by: list[str]) -> int:
        g = frame.group_by(by).agg(pl.col("fold").n_unique().alias("n"))
        return int((g["n"] > 1).sum())

    viol = {
        "tables in more than one fold": spans(rows, ["table_id"]),
        "pairs in more than one fold": spans(rows, ["pair_id"]),
        "hands in more than one fold": spans(rows, ["hand_id"]),
        "bags in more than one fold": spans(rows, ["pair_id", "hand_id"]),
        "players in more than one fold": spans(rows, ["player_id"]),
    }
    total = sum(viol.values())
    counts = bags.group_by("bag_label").len().sort("bag_label")
    per_fold = rows.group_by("fold").agg(
        pl.struct("pair_id", "hand_id").n_unique().alias("bags"),
        pl.len().alias("actions"),
        (pl.col("bag_label") == 1).sum().alias("positive_action_rows"))
    gold = pl.read_csv(C.DEV_EVIDENCE)
    shared = (bags.filter(pl.col("bag_label") == 1).select("hand_id").unique()
              .join(bags.filter(~pl.col("is_pos_pair")).select("hand_id").unique(),
                    on="hand_id", how="inner").height)
    report["frame"] = {
        "instance rows": int(rows.height),
        "features": len(FEATURES),
        "bags": {"positive (listed evidence hands)": int(
                     bags.filter(pl.col("bag_label") == 1).height),
                 "safe negative (non-target pairs' hands)": int(
                     bags.filter(pl.col("bag_label") == 0).height),
                 "censored (positive pairs' unlisted hands)": int(
                     bags.filter(pl.col("bag_label") == -1).height),
                 "gold rows in development_evidence.csv": int(gold.height),
                 "gold cells reached by the frame": int(
                     bags.filter(pl.col("bag_label") == 1).height)},
        "bag_label counts": {int(k): int(v) for k, v in counts.iter_rows()},
        "actions per bag": {
            "positive": round(float(rows.filter(pl.col("bag_label") == 1).height
                                    / max(1, bags.filter(pl.col("bag_label") == 1).height)), 3),
            "negative": round(float(rows.filter(pl.col("bag_label") == 0).height
                                    / max(1, bags.filter(pl.col("bag_label") == 0).height)), 3),
            "censored": round(float(rows.filter(pl.col("bag_label") == -1).height
                                    / max(1, bags.filter(pl.col("bag_label") == -1).height)), 3)},
        "evidence hands that are also a non-target pair's hand": int(shared),
        "fold isolation violations": viol,
        "fold isolation violation count": int(total),
        "per fold": {int(f): {"bags": int(b), "actions": int(a), "positive action rows": int(p)}
                     for f, b, a, p in per_fold.sort("fold").iter_rows()},
    }
    print(json.dumps(report["frame"], indent=2), flush=True)
    assert total == 0, f"{total} objects cross a fold boundary"
    assert bags.filter(pl.col("bag_label") == 1).height == gold.select(
        "pair_id", "hand_id").unique().height, "a listed evidence hand is missing from the frame"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="both", choices=("baseline", "frame", "both"))
    args = parser.parse_args()
    report = {}
    if REPORT.exists():
        report = json.loads(REPORT.read_text())
    if args.part in ("baseline", "both"):
        baseline(report)
        REPORT.write_text(json.dumps(report, indent=2))
    if args.part in ("frame", "both"):
        build_frame(report)
        REPORT.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
