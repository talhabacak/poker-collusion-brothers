"""Stage 531: the action-level bag frame, rebuilt on the teammate's tables and in the teammate's folds.

Stage 480 built this frame for pipeline A. Note 001 §6.1 says why it cannot be imported: A's
column was built on A's fold map (a seeded shuffle of the 400 tables) which is effectively
independent of B's `tidx % 5` (adjusted Rand index 0.0028), so a column built on A's folds is
not out of fold for B's reranker. This stage rebuilds the bags from B's own interim tables -
their `actions_feat` / `actions_scored` (their population policy's out-of-fold surprise and
class probabilities, fitted by table parity with no label read), their `seats_feat` for the
partner's hand context - with the fold of every bag being B's `tidx % 5`.

Bags
----
A bag is a `(pair, hidx)` cell; its instances are the actions the two pair members took in
that hand. The universe is every development hand in which both members of one of the 1,860
labelled pairs were seated, taken from `seats_feat` (co-seating), and asserted equal, as a set
of keys, to B's own `pairhand` rows for the same pairs - two tables, one census.

    label  1   a listed evidence hand of a disclosed target pair
    label  0   any hand of one of the 1,488 confirmed non-target pairs
    label -1   **censored**: every other hand of a positive pair

A positive pair's unlisted hands are not negatives. They are out of the loss entirely (stage
532 asserts the count it drops against this stage's census, and again before the first
gradient step) and scored at inference, which is where the ranking happens.

Features
--------
B's per-action columns as `04c_action_model.py` uses them (context, policy probabilities and
surprise, sizing, the partner's strength and state, who the action answers), the six-way
action identity, and the pair-relative columns of stage 480 computed inside each bag (what the
partner had done before this action and does after it, turns since the partner's last action,
the bag's share of the hand). Column names `street`, `surprise`, `resp_partner` and `act_*`
are kept identical to stage 480's so stage 481's trigger descriptors read them unchanged.

    TARIK_INTERIM=<repro>/data/interim python scripts/531_b_action_bag_frame.py

Writes data/cache/531_b_action_bags.parquet and artifacts/531_b_action_bag_frame.json.
Nothing here submits anything.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C

INTERIM = Path(os.environ.get("TARIK_INTERIM", ""))
FRAME = C.CACHE / "531_b_action_bags.parquet"
REPORT = C.ARTIFACTS / "531_b_action_bag_frame.json"
N_FOLDS = 5

ACTION_NAMES = ["fold", "check", "call", "bet", "raise", "all_in"]      # B's amap order = A's
AGGRESSIVE = [3, 4, 5]
ACT_ONEHOT = [f"act_{n}" for n in ACTION_NAMES]

# B's own action columns, exactly the ones its 04c reads, plus `players_dealt` and the four
# style rates 04c leaves out (02_policy's inputs, so they cost nothing new).
B_ACTION = ["st", "relpos", "hi", "lo", "suited", "pocket", "gap", "eq_pf", "hs", "hc", "potodds",
            "tocall_bb", "pot_bb", "stack_bb", "tocall_stack", "spr", "players_active", "players_dealt",
            "n_agg_hand", "n_agg_st", "own_agg_before", "n_act_st", "self_last_agg", "action_no",
            "surp", "p_fold", "p_pass", "p_agg", "size_z",
            "ps_n", "ps_open_raise", "ps_open_fold", "ps_vs_raise_agg", "ps_vs_raise_fold",
            "ps_post_agg", "ps_post_fold_vs_bet"]
# sizing and partner context, as 04c derives them
B_DERIVED = ["amt_bb", "amt_pot", "inv_bb", "raise_over_call", "b_active", "b_folded_before", "b_str",
             "rel", "facing", "b_relpos", "b_relpos_diff", "b_contrib_bb", "a_contrib_bb", "a_net_bb",
             "b_net_bb", "n_third_active"]
# stage 480's pair-relative columns, computed inside the bag
PAIR_FEATS = ["surprise", "resp_partner", "resp_outsider", "partner_acted_before", "partner_aggr_before",
              "partner_folded_before", "partner_acts_after", "partner_aggr_after", "partner_folds_after",
              "own_acted_before_bag", "own_aggr_before_bag", "turns_since_partner", "is_last_action_of_hand",
              "own_folded_hand", "partner_folded_hand", "outsiders_left", "players_at_showdown",
              "pair_share_of_actions", "street"]
FEATURES = B_ACTION + ACT_ONEHOT + B_DERIVED + PAIR_FEATS
KEY = ["pair_id", "hidx"]


def to_pq(df: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    df = (df.join(players.rename({"player_id": "player_1", "pidx": "i1"}), on="player_1")
            .join(players.rename({"player_id": "player_2", "pidx": "i2"}), on="player_2"))
    return df.with_columns(pl.min_horizontal("i1", "i2").alias("p"),
                           pl.max_horizontal("i1", "i2").alias("q")).drop("i1", "i2")


def bag_table(players, hands) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One row per (pair, hidx) with the three-valued bag label, from co-seating; and the
    same keys counted from B's own pairhand table as the independent census."""
    lab = to_pq(pl.read_csv(C.DEV_LABELS), players).select(
        "pair_id", "p", "q", "label", "behavior_family")
    ev = (pl.read_csv(C.DEV_EVIDENCE).join(hands.select("hidx", "hand_id"), on="hand_id")
          .select("pair_id", "hidx").unique())
    seats = (pl.scan_parquet(INTERIM / "seats_feat.parquet").filter(pl.col("is_eval") == 0)
             .select("hidx", "pidx").collect())
    members = set(lab["p"].to_list()) | set(lab["q"].to_list())
    seats = seats.filter(pl.col("pidx").is_in(list(members)))
    bags = (lab.join(seats.rename({"pidx": "p"}), on="p")
               .join(seats.rename({"pidx": "q"}), on=["hidx", "q"]))
    bags = bags.join(hands.select("hidx", "hand_id", "tidx", "table_id", "big_blind",
                                  "players_at_showdown"), on="hidx")
    bags = bags.join(ev.with_columns(pl.lit(1, dtype=pl.Int8).alias("_gold")),
                     on=["pair_id", "hidx"], how="left")
    bags = bags.with_columns(
        (pl.col("label") == 1).alias("is_pos_pair"),
        pl.when(pl.col("_gold") == 1).then(pl.lit(1, dtype=pl.Int8))
        .when(pl.col("label") == 0).then(pl.lit(0, dtype=pl.Int8))
        .otherwise(pl.lit(-1, dtype=pl.Int8)).alias("bag_label"),
        (pl.col("tidx") % N_FOLDS).cast(pl.Int8).alias("fold"),
    ).drop("_gold").sort(KEY)
    # the independent census: B's pairhand rows for the same pairs
    census = pl.concat([
        pl.scan_parquet(f).filter(pl.col("is_eval") == 0).select("hidx", "p", "q")
        .join(lab.lazy().select("p", "q", "pair_id"), on=["p", "q"]).collect()
        for f in sorted((INTERIM / "pairhand").glob("chunk*.parquet"))])
    return bags, census


def build(report: dict) -> None:
    t0 = time.time()
    players = pl.read_parquet(INTERIM / "players.parquet").select("player_id", "pidx")
    hands = pl.read_parquet(INTERIM / "hands.parquet").select(
        "hidx", "hand_id", "tidx", "table_id", "is_eval", "big_blind", "players_dealt", "players_at_showdown")
    bags, census = bag_table(players, hands)
    print(f"bags: {bags.height:,} cells over {bags['pair_id'].n_unique()} pairs "
          f"({time.time() - t0:.0f}s)", flush=True)
    a = set(bags.select("pair_id", "hidx").iter_rows())
    b = set(census.select("pair_id", "hidx").iter_rows())
    assert a == b, (f"co-seating and B's pairhand table disagree on the bag universe: "
                    f"{len(a - b)} only in seats, {len(b - a)} only in pairhand")
    ev_rows = pl.read_csv(C.DEV_EVIDENCE).select("pair_id", "hand_id").unique().height
    assert int((bags["bag_label"] == 1).sum()) == ev_rows, "a listed evidence hand is missing from the frame"

    rows = instances(bags, 0, t0)

    meta = bags.select(KEY + ["hand_id", "table_id", "fold", "behavior_family", "is_pos_pair", "bag_label",
                              "players_at_showdown"])
    out = rows.join(meta, on=KEY, how="left").select(
        KEY + ["hand_id", "tidx", "table_id", "fold", "player", "partner",
               "behavior_family", "is_pos_pair", "bag_label"] + FEATURES)   # action_no is in FEATURES
    out = out.with_columns([pl.col(c).cast(pl.Float32) for c in FEATURES]).sort(KEY + ["action_no", "player"])
    out.write_parquet(FRAME, compression="zstd")
    print(f"frame written: {out.height:,} rows x {len(FEATURES)} features ({time.time() - t0:.0f}s)", flush=True)
    audit(out, bags, report)
    report["frame"]["seconds"] = round(time.time() - t0, 1)


def instances(bags: pl.DataFrame, is_eval: int, t0: float | None = None) -> pl.DataFrame:
    """Instance rows for the (pair_id, hidx, p, q) cells in `bags`: every action either member
    took in the hand, with every column of FEATURES except `players_at_showdown` (a property of
    the bag, joined by the caller). Used unchanged for the development frame and, by stage 534,
    for the evaluation-side bags."""
    t0 = time.time() if t0 is None else t0
    hand_keys = bags.select("hidx").unique()
    af = (pl.scan_parquet(INTERIM / "actions_feat.parquet").filter(pl.col("is_eval") == is_eval)
          .join(hand_keys.lazy(), on="hidx").collect())
    asc = (pl.scan_parquet(INTERIM / "actions_scored.parquet")
           .select("hidx", "action_no", "surp", "p_fold", "p_pass", "p_agg", "size_z")
           .join(hand_keys.lazy(), on="hidx").collect())
    af = af.join(asc, on=["hidx", "action_no"], how="left").sort(["hidx", "action_no"])
    assert af["surp"].null_count() == 0, "actions_scored does not cover the bag hands"
    print(f"action rows in those hands: {af.height:,} ({time.time() - t0:.0f}s)", flush=True)
    hand_shape = af.group_by("hidx").agg(pl.col("action_no").max().alias("_hand_last_action"),
                                         pl.len().alias("_hand_actions"))
    af = af.with_columns(
        (pl.col("amount").cum_sum().over(["hidx", "pidx"]) - pl.col("amount")).alias("_inv_before"))

    members = pl.concat([
        bags.select("pair_id", "hidx", pl.col("p").alias("player"), pl.col("q").alias("partner"),
                    pl.lit(1, dtype=pl.Int8).alias("side")),
        bags.select("pair_id", "hidx", pl.col("q").alias("player"), pl.col("p").alias("partner"),
                    pl.lit(2, dtype=pl.Int8).alias("side")),
    ])
    rows = members.join(af.rename({"pidx": "player"}), on=["hidx", "player"], how="inner")
    rows = rows.join(hand_shape, on="hidx", how="left")
    print(f"instance rows: {rows.height:,} ({time.time() - t0:.0f}s)", flush=True)

    seat = (pl.scan_parquet(INTERIM / "seats_feat.parquet").filter(pl.col("is_eval") == is_eval)
            .select("hidx", "pidx", "fold_no", "eq_pf", "hs1", "hs2", "hs3", "relpos",
                    "total_contribution", "net_chips", "folded")
            .join(hand_keys.lazy(), on="hidx").collect())
    partner = seat.rename({"pidx": "partner", "fold_no": "b_fold_no", "eq_pf": "b_eq", "hs1": "b_hs1",
                           "hs2": "b_hs2", "hs3": "b_hs3", "relpos": "b_relpos",
                           "total_contribution": "b_contrib", "net_chips": "b_net", "folded": "b_folded"})
    own = seat.select("hidx", pl.col("pidx").alias("player"), pl.col("total_contribution").alias("a_contrib"),
                      pl.col("net_chips").alias("a_net"), pl.col("folded").alias("a_folded"))
    rows = rows.join(partner, on=["hidx", "partner"], how="left").join(own, on=["hidx", "player"], how="left")

    pre = pl.col("st") == 0
    rows = rows.with_columns(
        (pl.col("amount") / pl.col("big_blind")).alias("amt_bb"),
        (pl.col("amount") / pl.col("pot_before").clip(1)).alias("amt_pot"),
        (pl.col("_inv_before") / pl.col("big_blind")).alias("inv_bb"),
        (pl.col("amount") / (pl.col("to_call") + pl.col("big_blind"))).alias("raise_over_call"),
        (pl.col("b_fold_no").is_null() | (pl.col("b_fold_no") > pl.col("action_no"))).cast(pl.Int8).alias("b_active"),
        (pl.col("b_fold_no").is_not_null() & (pl.col("b_fold_no") < pl.col("action_no"))).cast(pl.Int8).alias("b_folded_before"),
        pl.when(pre).then(pl.col("b_eq")).when(pl.col("st") == 1).then(pl.col("b_hs1"))
        .when(pl.col("st") == 2).then(pl.col("b_hs2")).otherwise(pl.col("b_hs3")).alias("b_str"),
        pl.when(pl.col("last_agg_pidx").is_null()).then(0).when(pl.col("last_agg_pidx") == pl.col("player")).then(1)
        .when(pl.col("last_agg_pidx") == pl.col("partner")).then(2).otherwise(3).alias("facing"),
        (pl.col("b_relpos") - pl.col("relpos")).alias("b_relpos_diff"),
        (pl.col("b_contrib") / pl.col("big_blind")).alias("b_contrib_bb"),
        (pl.col("a_contrib") / pl.col("big_blind")).alias("a_contrib_bb"),
        (pl.col("a_net") / pl.col("big_blind")).alias("a_net_bb"),
        (pl.col("b_net") / pl.col("big_blind")).alias("b_net_bb"),
        (pl.col("last_agg_pidx") == pl.col("partner")).fill_null(False).cast(pl.Int8).alias("resp_partner"),
        (pl.col("last_agg_pidx").is_not_null() & (pl.col("last_agg_pidx") != pl.col("partner"))
         & (pl.col("last_agg_pidx") != pl.col("player"))).cast(pl.Int8).alias("resp_outsider"),
        pl.col("surp").alias("surprise"), pl.col("st").alias("street"),
        pl.col("a_folded").cast(pl.Int8).alias("own_folded_hand"),
        pl.col("b_folded").cast(pl.Int8).alias("partner_folded_hand"),
        (pl.col("players_active") - 2).alias("outsiders_left"),
        (pl.col("action_no") == pl.col("_hand_last_action")).cast(pl.Int8).alias("is_last_action_of_hand"),
    )
    rows = rows.with_columns(
        (pl.when(pre).then(pl.col("eq_pf")).otherwise(pl.col("hs")) - pl.col("b_str")).fill_null(0.0).alias("rel"),
        (pl.col("players_active") - 1 - pl.col("b_active")).alias("n_third_active"),
        pl.col("y").is_in(AGGRESSIVE).cast(pl.Int8).alias("_aggr"),
        (pl.col("y") == 0).cast(pl.Int8).alias("_fold"),
        pl.lit(1, dtype=pl.Int8).alias("_one"),
    )
    rows = rows.with_columns([(pl.col("y") == i).cast(pl.Int8).alias(n) for i, n in enumerate(ACT_ONEHOT)])

    # ---- pair-relative columns inside each bag (stage 480's construction, unchanged)
    ownk = KEY + ["player"]
    rows = rows.sort(KEY + ["action_no"]).with_columns(
        (pl.col("_one").cum_sum().over(KEY) - pl.col("_one")).alias("_bag_before"),
        (pl.col("_one").cum_sum().over(ownk) - pl.col("_one")).alias("own_acted_before_bag"),
        (pl.col("_aggr").cum_sum().over(KEY) - pl.col("_aggr")).alias("_bag_aggr_before"),
        (pl.col("_aggr").cum_sum().over(ownk) - pl.col("_aggr")).alias("own_aggr_before_bag"),
        (pl.col("_fold").cum_sum().over(KEY) - pl.col("_fold")).alias("_bag_fold_before"),
        (pl.col("_fold").cum_sum().over(ownk) - pl.col("_fold")).alias("_own_fold_before"),
        pl.col("_one").sum().over(KEY).alias("_bag_total"),
        pl.col("_one").sum().over(ownk).alias("_own_total"),
        pl.col("_aggr").sum().over(KEY).alias("_bag_aggr_total"),
        pl.col("_aggr").sum().over(ownk).alias("_own_aggr_total"),
        pl.col("_fold").sum().over(KEY).alias("_bag_fold_total"),
        pl.col("_fold").sum().over(ownk).alias("_own_fold_total"),
        pl.col("action_no").shift(1).over(KEY).alias("_prev_an"),
        pl.col("player").shift(1).over(KEY).alias("_prev_who"),
    )
    rows = rows.with_columns(
        (pl.col("_bag_before") - pl.col("own_acted_before_bag")).alias("partner_acted_before"),
        (pl.col("_bag_aggr_before") - pl.col("own_aggr_before_bag")).alias("partner_aggr_before"),
        (pl.col("_bag_fold_before") - pl.col("_own_fold_before")).alias("partner_folded_before"),
        ((pl.col("_bag_total") - pl.col("_bag_before") - 1)
         - (pl.col("_own_total") - pl.col("own_acted_before_bag") - 1)).alias("partner_acts_after"),
        ((pl.col("_bag_aggr_total") - pl.col("_bag_aggr_before") - pl.col("_aggr"))
         - (pl.col("_own_aggr_total") - pl.col("own_aggr_before_bag") - pl.col("_aggr"))).alias("partner_aggr_after"),
        ((pl.col("_bag_fold_total") - pl.col("_bag_fold_before") - pl.col("_fold"))
         - (pl.col("_own_fold_total") - pl.col("_own_fold_before") - pl.col("_fold"))).alias("partner_folds_after"),
        (pl.col("_bag_total") / pl.col("_hand_actions")).alias("pair_share_of_actions"),
        pl.when(pl.col("_prev_who").is_not_null() & (pl.col("_prev_who") != pl.col("player")))
        .then(pl.col("action_no") - pl.col("_prev_an")).otherwise(pl.lit(-1)).alias("turns_since_partner"),
    )
    return rows


def audit(rows: pl.DataFrame, bags: pl.DataFrame, report: dict) -> None:
    """Fold isolation under B's `tidx % 5` and the bag census, asserted in code."""
    def spans(frame: pl.DataFrame, by: list[str]) -> int:
        g = frame.group_by(by).agg(pl.col("fold").n_unique().alias("n"))
        return int((g["n"] > 1).sum())

    viol = {
        "tables in more than one fold": spans(rows, ["tidx"]),
        "pairs in more than one fold": spans(rows, ["pair_id"]),
        "hands in more than one fold": spans(rows, ["hidx"]),
        "bags in more than one fold": spans(rows, KEY),
        "players in more than one fold": spans(rows, ["player"]),
    }
    total = sum(viol.values())
    counts = bags.group_by("bag_label").len().sort("bag_label")
    per_fold = rows.group_by("fold").agg(
        pl.struct(KEY).n_unique().alias("bags"), pl.len().alias("actions"),
        (pl.col("bag_label") == 1).sum().alias("positive_action_rows"))
    gold = pl.read_csv(C.DEV_EVIDENCE)
    nulls = {c: int(v) for c, v in zip(FEATURES, rows.select(FEATURES).null_count().row(0)) if v}
    bag_len = rows.group_by(KEY + ["bag_label"]).len()
    # cells in which neither member acted carry no instance and so cannot be in the frame; the
    # count per label is what stage 532 subtracts when it asserts its censoring against this census
    missing = bags.join(rows.select(KEY).unique(), on=KEY, how="anti").group_by("bag_label").len()
    report["frame"] = {
        "source": "B's interim tables (actions_feat, actions_scored, seats_feat, hands, pairhand) under "
                  "B's tidx % 5 folds; nothing of A's fold map or A's features is used",
        "instance rows": int(rows.height), "features": len(FEATURES), "feature names": FEATURES,
        "bags": {"positive (listed evidence hands)": int((bags["bag_label"] == 1).sum()),
                 "safe negative (non-target pairs' hands)": int((bags["bag_label"] == 0).sum()),
                 "censored (positive pairs' unlisted hands)": int((bags["bag_label"] == -1).sum()),
                 "gold rows in development_evidence.csv": int(gold.height),
                 "gold cells reached by the frame": int((bags["bag_label"] == 1).sum()),
                 "bags with no action by either member (not in the frame)": int(
                     bags.height - rows.select(KEY).n_unique())},
        "bag_label counts": {int(k): int(v) for k, v in counts.iter_rows()},
        "cells with no action rows by label": {str(int(k)): int(v) for k, v in missing.iter_rows()},
        "actions per bag": {name: round(float(bag_len.filter(pl.col("bag_label") == lbl)["len"].mean()), 3)
                            for name, lbl in (("positive", 1), ("negative", 0), ("censored", -1))},
        "columns with nulls (standardiser imputes the training median)": nulls,
        "fold isolation violations": viol, "fold isolation violation count": int(total),
        "fold map": "tidx % 5 (B's 04_hand_scorer / 04b / 04c / 05b)",
        "per fold": {int(f): {"bags": int(b), "actions": int(a), "positive action rows": int(p)}
                     for f, b, a, p in per_fold.sort("fold").iter_rows()},
    }
    print(json.dumps({k: v for k, v in report["frame"].items() if k != "feature names"}, indent=2), flush=True)
    assert total == 0, f"{total} objects cross a fold boundary"
    assert int((bags["bag_label"] == 1).sum()) == gold.select("pair_id", "hand_id").unique().height


def main() -> None:
    if not INTERIM.exists():
        raise SystemExit("set TARIK_INTERIM to the reproduction's data/interim directory")
    report = {"interim": str(INTERIM)}
    build(report)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"written to {REPORT}", flush=True)


if __name__ == "__main__":
    main()
