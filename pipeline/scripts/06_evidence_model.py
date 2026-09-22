"""Stage 6: evidence retrieval - rank a pair's shared hands, keep the top five.

This is a within-pair ranking problem, not a classification one. A pair that
transfers value in every hand and a pair that does it twice both need their
*relative* ordering right, so every continuous signal enters twice: as its raw
value, and as its rank inside the pair. Without the second form the model learns
"big pot" rather than "unusual for this pair".

Trained on the 372 disclosed positive pairs against `development_evidence.csv`,
validated out-of-fold by table so a pair's own table never trains its ranker.

Retrieval is routed: one specialist per behaviour family, mixed by the family
probabilities and then blended 0.6/0.4 with a pooled ranker. Family routing is
accurate to 98.7% out of fold on the disclosed positives, so the specialists
almost always see the right kind of hand, while the pooled term keeps a pair
whose family is genuinely ambiguous from being scored by the wrong expert.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C
from pokercol.cv import table_folds
from pokercol.metrics import evidence_map5

N_FOLDS = 5

# Hand-level signals that are also meaningful as a within-pair rank.
RANKED = [
    "transfer_gross", "transfer_abs_signed", "pot_bb", "contribution_gap_abs",
    "net_gap_abs", "paid_flag", "conceded_flag", "strength_gap_abs",
    "outsider_folds", "pair_aggressive", "value_against_strength",
    "pair_surprise", "pair_fold_surprise", "surprise_imbalance",
    "unjustified_investment", "donor_surprise", "donor_weakness",
    "pair_won_small_pot", "outsider_clear_rate", "weak_preflop_raise_weight",
    "ahead_fold_equity_gap", "pair_aggr_surprise", "pair_aggr_excess",
    "ps_partner", "ax_vs_partner",
]
FLAT = [
    "with_dominant_flow", "with_dominant_concession",
    "ps_partner", "ps_partner_hi", "fold_vs_partner", "call_vs_partner",
    "aggr_vs_partner", "ax_vs_partner",
    "pair_aggr_surprise", "pair_aggr_excess", "stronger_side_deficit",
    "both_showdown", "both_folded", "one_folded", "both_in_pot", "pair_heads_up",
    "mutual_passive", "paid_flag", "conceded_flag", "transfer_gross",
    "transfer_abs_signed", "pot_bb", "contribution_gap_abs", "net_gap_abs",
    "strength_gap_abs", "outsider_folds", "outsider_aggressive",
    "outsider_folds_facing_bet", "outsider_showdowns", "pair_aggressive",
    "board_cards_dealt", "players_at_showdown", "value_against_strength",
    "min_strength_rank", "max_strength_rank", "both_postflop_checks",
    "loser_contribution", "winner_is_partner",
    "pair_surprise", "pair_fold_surprise", "pair_call_surprise",
    "surprise_imbalance", "transfer_event", "fold_better_flag",
    "consecutive_aggression", "outsiders_folded_to_pair", "pair_net_bb",
    "concession_weight",
    "unjustified_investment", "donor_surprise", "receiver_surprise",
    "donor_weakness", "receiver_weakness", "donor_contribution",
    "surprise_per_action", "pair_won_pot", "pair_won_small_pot",
    "outsider_clear_rate", "aggr_with_weak_hand", "donor_called_facing_bet",
    "ahead_fold", "ahead_fold_preflop", "ahead_fold_equity_gap",
    "both_raised_preflop", "squeeze_sandwich", "weak_preflop_raise_weight",
    "weaker_raiser_strength", "outsider_preflop_raises", "pair_aggressive_loss",
    "preflop_equity_hu_1", "preflop_equity_hu_2",
    "timeline_position", "gap_to_prev_bb", "gap_to_next_bb",
    "neighbour_transfer", "neighbour_surprise", "local_event_density",
    "any_vpip", "both_vpip", "interacted", "contact",
    "contact_before", "contact_total", "contact_position", "contact_index", "events_before",
]

PARAMS = dict(**C.LGB_REPRO,
    objective="lambdarank",
    metric="map",
    eval_at=[5],
    # The competition reads five slots per pair, so gradients spread over fifteen
    # candidates spend most of their budget below the scored region. Eight is the
    # plateau: six seeds give 0.6892 against 0.6832 at fifteen, ahead in all three
    # families (stage 107). `EV_TRUNCATION` rebuilds the older files.
    lambdarank_truncation_level=int(__import__("os").environ.get("EV_TRUNCATION", 8)),
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=2.0,
    verbosity=-1,
    num_threads=C.N_THREADS,
    seed=C.SEED,
)
N_ROUNDS = 400


def derive(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Hand-level evidence signals shared by training and inference."""
    net1, net2 = pl.col("net_bb_1"), pl.col("net_bb_2")
    rank1, rank2 = pl.col("strength_rank_1"), pl.col("strength_rank_2")
    paid = pl.col("paid_1_to_2") | pl.col("paid_2_to_1")
    conceded = pl.col("conceded_1_to_2") | pl.col("conceded_2_to_1")
    loser_contribution = (
        pl.when(net1 < net2).then(pl.col("contribution_bb_1")).otherwise(pl.col("contribution_bb_2"))
    )
    return lf.with_columns(
        # How implausibly the two of them played this hand, and how lopsidedly.
        (pl.col("surprise_sum_1") + pl.col("surprise_sum_2")).alias("pair_surprise"),
        (pl.col("fold_surprise_1") + pl.col("fold_surprise_2")).alias("pair_fold_surprise"),
        (pl.col("call_surprise_1") + pl.col("call_surprise_2")).alias("pair_call_surprise"),
        # Surprise of the aggressive actions specifically, and aggression taken
        # beyond what the policy model expected from those cards. Isolation over-
        # raises (AUC 0.90 on its planted hands); soft play has the stronger
        # partner under-raising (0.77).
        (pl.col("aggr_surprise_1") + pl.col("aggr_surprise_2")).alias("pair_aggr_surprise"),
        # The same surprise restricted to actions one partner took in answer to
        # the other's pressure. Planted concessions and pay-offs live there, and
        # it separates them far better than the seat total (AUC 0.93 / 0.95 for
        # directed transfer / soft play, against 0.88 / 0.88).
        (pl.col("sv_1_to_2") + pl.col("sv_2_to_1")).alias("ps_partner"),
        pl.max_horizontal("sv_1_to_2", "sv_2_to_1").alias("ps_partner_hi"),
        (pl.col("fsv_1_to_2") + pl.col("fsv_2_to_1")).alias("fold_vs_partner"),
        (pl.col("csv_1_to_2") + pl.col("csv_2_to_1")).alias("call_vs_partner"),
        (pl.col("asv_1_to_2") + pl.col("asv_2_to_1")).alias("aggr_vs_partner"),
        (pl.col("axv_1_to_2") + pl.col("axv_2_to_1")).alias("ax_vs_partner"),
        (pl.col("aggression_excess_1") + pl.col("aggression_excess_2")).alias("pair_aggr_excess"),
        pl.when(pl.col("strength_rank_1").fill_null(3) <= pl.col("strength_rank_2").fill_null(3))
        .then(-pl.col("aggression_excess_1")).otherwise(-pl.col("aggression_excess_2"))
        .alias("stronger_side_deficit"),
        (pl.col("surprise_sum_1") - pl.col("surprise_sum_2")).abs().alias("surprise_imbalance"),
        (pl.col("fold_better_1_to_2") | pl.col("fold_better_2_to_1")).cast(pl.Int8)
        .alias("fold_better_flag"),
        ((pl.col("fold_better_1_to_2") | pl.col("fold_better_2_to_1"))
         | (pl.col("paid_1_to_2") | pl.col("paid_2_to_1"))).cast(pl.Int8).alias("transfer_event"),
        pl.max_horizontal("concession_weight_1", "concession_weight_2").alias("concession_weight"),
        pl.col("consecutive_aggression").cast(pl.Int8),
        (pl.col("ahead_fold_1_to_2") | pl.col("ahead_fold_2_to_1")).cast(pl.Int8)
        .alias("ahead_fold"),
        (pl.col("ahead_fold_preflop_1_to_2") | pl.col("ahead_fold_preflop_2_to_1"))
        .cast(pl.Int8).alias("ahead_fold_preflop"),
        # Hard preconditions read off the planted hands: every one has at least
        # one partner putting money in preflop by choice, and nearly all are
        # "contact" hands - both partners voluntarily in, or one folding to the
        # other's pressure.
        ((pl.col("vpip_actions_1") > 0) | (pl.col("vpip_actions_2") > 0)).cast(pl.Int8)
        .alias("any_vpip"),
        ((pl.col("vpip_actions_1") > 0) & (pl.col("vpip_actions_2") > 0)).cast(pl.Int8)
        .alias("both_vpip"),
        ((pl.col("n_actions_1") > 0) & (pl.col("n_actions_2") > 0)
         & ((pl.col("aggressive_actions_1") + pl.col("aggressive_actions_2")
             + pl.col("calls_1") + pl.col("calls_2")) > 0)).cast(pl.Int8).alias("interacted"),
        (pl.col("p1_folded_to_p2") | pl.col("p2_folded_to_p1")
         | ((pl.col("vpip_actions_1") > 0) & (pl.col("vpip_actions_2") > 0))).cast(pl.Int8)
        .alias("contact"),
        pl.col("both_raised_preflop").cast(pl.Int8),
        pl.col("squeeze_sandwich").cast(pl.Int8),
        pl.col("pair_aggressive_loss").cast(pl.Int8),
        pl.col("weaker_raiser_strength").fill_null(1.0),
        pl.col("transfer_signed").abs().alias("transfer_abs_signed"),
        pl.col("contribution_gap").abs().alias("contribution_gap_abs"),
        pl.col("net_gap").abs().alias("net_gap_abs"),
        pl.col("strength_gap").abs().fill_null(0).alias("strength_gap_abs"),
        paid.cast(pl.Int8).alias("paid_flag"),
        conceded.cast(pl.Int8).alias("conceded_flag"),
        # Chips moved multiplied by how wrong the movement was given the cards.
        (pl.col("transfer_gross") * paid.cast(pl.Float64)).alias("value_against_strength"),
        pl.min_horizontal(rank1, rank2).fill_null(0).alias("min_strength_rank"),
        pl.max_horizontal(rank1, rank2).fill_null(0).alias("max_strength_rank"),
        (pl.col("postflop_checks_1") + pl.col("postflop_checks_2")).alias("both_postflop_checks"),
        loser_contribution.alias("loser_contribution"),
        # Did the pot leave the pair, or stay inside it?
        ((net1 > 0) | (net2 > 0)).cast(pl.Int8).alias("winner_is_partner"),
    ).with_columns(
        # Split the pair into the side that lost chips and the side that gained.
        # A natural big loss comes from a strong hand that got beaten; a planted
        # one comes from a weak hand that paid anyway, so the donor's own
        # strength - not the size of the pot - is what separates them.
        pl.when(net1 <= net2).then(pl.col("strength_rank_1")).otherwise(pl.col("strength_rank_2"))
        .fill_null(3).alias("donor_weakness"),
        pl.when(net1 > net2).then(pl.col("strength_rank_1")).otherwise(pl.col("strength_rank_2"))
        .fill_null(3).alias("receiver_weakness"),
        pl.when(net1 <= net2).then(pl.col("surprise_sum_1")).otherwise(pl.col("surprise_sum_2"))
        .alias("donor_surprise"),
        pl.when(net1 > net2).then(pl.col("surprise_sum_1")).otherwise(pl.col("surprise_sum_2"))
        .alias("receiver_surprise"),
        pl.when(net1 <= net2).then(pl.col("contribution_bb_1")).otherwise(pl.col("contribution_bb_2"))
        .alias("donor_contribution"),
        pl.when(net1 <= net2).then(pl.col("calls_facing_bet_1")).otherwise(pl.col("calls_facing_bet_2"))
        .alias("donor_called_facing_bet"),
        (pl.col("pair_surprise")
         / (pl.col("n_actions_1") + pl.col("n_actions_2") + 1)).alias("surprise_per_action"),
        ((net1 > 0) | (net2 > 0)).cast(pl.Int8).alias("pair_won_pot"),
        # Isolation pays off in small pots cleared of outsiders, not big showdowns.
        pl.when(pl.col("outsider_folds") > 0)
        .then(pl.col("outsiders_folded_to_pair") / pl.col("outsider_folds"))
        .otherwise(0.0).alias("outsider_clear_rate"),
        # Applying pressure while holding a bottom-half hand.
        (((pl.col("aggressive_actions_1") > 0) & (pl.col("strength_rank_1") >= 4))
         | ((pl.col("aggressive_actions_2") > 0) & (pl.col("strength_rank_2") >= 4))
         ).cast(pl.Int8).alias("aggr_with_weak_hand"),
    ).with_columns(
        # Chips the losing side committed, weighted by how little their cards
        # justified it.
        (pl.col("donor_contribution") * pl.col("donor_weakness") / 6.0)
        .alias("unjustified_investment"),
        (((net1 > 0) | (net2 > 0)) & (pl.col("pot_bb") < 40)).cast(pl.Int8)
        .alias("pair_won_small_pot"),
    )


def add_episode_features(df: pl.DataFrame) -> pl.DataFrame:
    """Where a hand sits in the pair's own history, and what surrounds it.

    The competition states that coordination is episodic, and the planted hands
    bear that out: they sit earlier in a pair's shared timeline than chance
    would put them (median position 0.32 of the way through, not 0.50) and
    closer together than chance (median gap 11 hands against 16). So a hand is
    more suspicious when its neighbours are too.

    All of this comes from `started_at` - the order the hands were actually
    played - not from row or file order.
    """
    key = ["p1", "p2"]
    df = df.sort([*key, "started_at"])
    n = pl.len().over(key)
    idx = pl.int_range(pl.len()).over(key)
    transfer = pl.col("transfer_gross")
    event = pl.col("transfer_event").cast(pl.Float64)
    return df.with_columns(
        (idx / n).alias("timeline_position"),
        # Neighbourhood sums exclude the hand itself, so the feature describes
        # the company a hand keeps rather than restating the hand.
        ((transfer.shift(1).over(key).fill_null(0.0)
          + transfer.shift(-1).over(key).fill_null(0.0)) / 2).alias("neighbour_transfer"),
        ((pl.col("pair_surprise").shift(1).over(key).fill_null(0.0)
          + pl.col("pair_surprise").shift(-1).over(key).fill_null(0.0)) / 2
         ).alias("neighbour_surprise"),
        (event.rolling_sum(window_size=9, center=True, min_samples=1).over(key) - event
         ).alias("local_event_density"),
    ).with_columns(
        ((pl.col("started_at") - pl.col("started_at").shift(1).over(key))
         .dt.total_seconds().fill_null(-1.0) / 3600.0).alias("gap_to_prev_bb"),
        ((pl.col("started_at").shift(-1).over(key) - pl.col("started_at"))
         .dt.total_seconds().fill_null(-1.0) / 3600.0).alias("gap_to_next_bb"),
        # Position among the pair's *contact* hands only. Every planted hand has
        # at least one partner entering the pot voluntarily, and nearly all are
        # contact hands, so ordering within that subset of ~33 hands locates the
        # episode far more precisely than ordering within all ~120.
        (pl.col("contact").cast(pl.Int32).cum_sum().over(key) - pl.col("contact").cast(pl.Int32))
        .alias("contact_before"),
        pl.col("contact").cast(pl.Int32).sum().over(key).alias("contact_total"),
        (pl.col("transfer_event").cast(pl.Int32).cum_sum().over(key)
         - pl.col("transfer_event").cast(pl.Int32)).alias("events_before"),
    ).with_columns(
        (pl.col("contact_before") / pl.col("contact_total").clip(lower_bound=1))
        .alias("contact_position"),
        pl.when(pl.col("contact") == 1).then(pl.col("contact_before")).otherwise(999)
        .alias("contact_index"),
    )


def add_direction_consistency(df: pl.DataFrame) -> pl.DataFrame:
    """Does this hand move value in the pair's dominant direction?

    A directed-transfer donor is consistent: the same partner keeps losing. The
    pair's dominant direction is read from all of its *other* hands, so a hand
    never votes for itself.
    """
    key = ["p1", "p2"]
    af12 = pl.col("ahead_fold_1_to_2").cast(pl.Int32)
    af21 = pl.col("ahead_fold_2_to_1").cast(pl.Int32)
    flow_12_first = ((pl.col("transfer_1_to_2").sum().over(key) - pl.col("transfer_1_to_2"))
                     >= (pl.col("transfer_2_to_1").sum().over(key) - pl.col("transfer_2_to_1")))
    concede_12_first = (af12.sum().over(key) - af12) >= (af21.sum().over(key) - af21)
    return df.with_columns(
        pl.when(flow_12_first).then(pl.col("transfer_1_to_2") - pl.col("transfer_2_to_1"))
        .otherwise(pl.col("transfer_2_to_1") - pl.col("transfer_1_to_2")).alias("with_dominant_flow"),
        pl.when(concede_12_first).then(af12 - af21).otherwise(af21 - af12)
        .alias("with_dominant_concession"),
    )


def add_within_pair(df: pl.DataFrame) -> pl.DataFrame:
    """Rank percentile and share-of-max for each ranked signal, within the pair."""
    df = add_direction_consistency(add_episode_features(df))
    key = ["p1", "p2"]
    n = pl.len().over(key)
    exprs = []
    for c in RANKED:
        exprs.append((pl.col(c).rank("average").over(key) / n).alias(f"pct_{c}"))
        exprs.append((pl.col(c) / (pl.col(c).abs().max().over(key) + 1e-9)).alias(f"rel_{c}"))
    return df.with_columns(exprs)


EPISODE_POSITION = [
    "timeline_position", "contact_position", "contact_before", "contact_total",
    "contact_index", "events_before", "gap_to_prev_bb", "gap_to_next_bb",
    "neighbour_transfer", "neighbour_surprise", "local_event_density",
]


def feature_names() -> list[str]:
    names = FLAT + [f"pct_{c}" for c in RANKED] + [f"rel_{c}" for c in RANKED]
    # Ablation switch for a leaderboard risk check: EV_NO_POSITION=1 removes the
    # episode-position features, which carry about a fifth of evidence MAP@5
    # offline. Unset in the production pipeline.
    if __import__("os").environ.get("EV_NO_POSITION") == "1":
        names = [n for n in names if n not in EPISODE_POSITION]
    # Templates before sizing, the order stages 82, 83, 87 and 90 measured in.
    # `feature_fraction` samples columns by position, so the permutation acts like
    # a seed: the same routed-similarity engine scores 0.6912 in this order and
    # 0.6870 with sizing first, against a base that moves by 0.0001 either way.
    if __import__("os").environ.get("EV_TEMPLATES") == "1":
        names = names + TEMPLATE_NAMES
    if __import__("os").environ.get("EV_SIZING") == "1":
        names = names + sizing_names()
    return names


# Chosen on seeds 0-5 of stage 30 and confirmed on fresh seeds 6-11: specialist
# weight 0.6 -> 0.75 and a six-seed bag lift out-of-fold MAP@5 0.658 -> 0.668.
# EV_SPEC_WEIGHT overrides the blend: 0 scores every pair with the pooled ranker
# alone, which is the control for "the specialists are fitted to 372 development
# pairs and do not carry to the public split".
SPECIALIST_WEIGHT = float(__import__("os").environ.get("EV_SPEC_WEIGHT", 0.75))
SPECIALIST_ROUNDS = 300
EVIDENCE_SEEDS = int(__import__("os").environ.get("EV_SEED_BAG", 6))


def fit_bag(df: pl.DataFrame, feats: list[str]) -> list[tuple[lgb.Booster, dict[str, lgb.Booster]]]:
    """Pooled ranker and specialists, once per seed. With ~300 training pairs
    a single seed's ranking moves by about ±0.005 MAP@5; the bag averages it out."""
    seed0 = PARAMS["seed"]
    members = []
    for s in range(EVIDENCE_SEEDS):
        PARAMS["seed"] = seed0 + s
        members.append((fit_ranker(df, feats), fit_specialists(df, feats)))
    PARAMS["seed"] = seed0
    return members


def fit_routed_specialists(df: pl.DataFrame, feats: list[str],
                           spec_feats: dict[str, list[str]],
                           base: list[dict[str, lgb.Booster]]) -> list[dict[str, lgb.Booster]]:
    """Refit the families named in `spec_feats` on their longer feature lists and
    merge them over `base`, seed by seed, so both sets share the bag's seeds."""
    seed0 = PARAMS["seed"]
    out = []
    for s, spec in enumerate(base):
        PARAMS["seed"] = seed0 + s
        out.append({**spec, **fit_specialists(df, feats, spec_feats, families=spec_feats)})
    PARAMS["seed"] = seed0
    return out


def fit_ranker(df: pl.DataFrame, feats: list[str], rounds: int = N_ROUNDS) -> lgb.Booster:
    """Fit a lambdarank model grouped by pair over the rows given."""
    tr = df.sort("pair_id", "hand_id")
    groups = tr.group_by("pair_id", maintain_order=True).len()["len"].to_numpy()
    return lgb.train(
        PARAMS,
        lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32),
                    label=tr["is_evidence"].to_numpy(), group=groups),
        num_boost_round=rounds,
    )


# Swept per family in stage 83 against a fixed pooled ranker, then confirmed in
# the production blend (stage 90): 0.6858 -> 0.6949 out of fold. The families
# want different capacity - isolation has 92 training pairs and prefers small
# trees (0.6832 -> 0.7005), soft play has the most and prefers more rounds.
TUNED_SPECIALISTS = {
    "directed_transfer": ({"num_leaves": 7, "min_data_in_leaf": 10}, 300),
    "soft_play": ({"num_leaves": 31, "min_data_in_leaf": 10}, 500),
    "coordinated_isolation": ({"num_leaves": 7, "min_data_in_leaf": 20}, 300),
}


def fit_specialists(df: pl.DataFrame, feats: list[str],
                    spec_feats: dict[str, list[str]] | None = None,
                    families=None) -> dict[str, lgb.Booster]:
    """One ranker per behaviour family. `spec_feats` overrides the feature list
    for individual families, `families` restricts which ones are fitted."""
    tuned = __import__("os").environ.get("EV_SPEC_TUNED") == "1"
    saved = dict(PARAMS)
    out: dict[str, lgb.Booster] = {}
    for family in sorted(C.FAMILIES):
        if families is not None and family not in families:
            continue
        sub = df.filter(pl.col("behavior_family") == family)
        if sub.height == 0:
            continue
        cols = (spec_feats or {}).get(family, feats)
        if tuned and family in TUNED_SPECIALISTS:
            params, rounds = TUNED_SPECIALISTS[family]
            PARAMS.update(params)
            out[family] = fit_ranker(sub, cols, rounds)
            PARAMS.clear()
            PARAMS.update(saved)
        else:
            out[family] = fit_ranker(sub, cols, SPECIALIST_ROUNDS)
    return out


def within_pair_rank(pair_ids: pl.Series, score: np.ndarray) -> np.ndarray:
    """Percentile of a score inside its pair, so two models blend on equal terms."""
    tmp = pl.DataFrame({"pair_id": pair_ids, "s": score})
    return tmp.with_columns(
        (pl.col("s").rank("average").over("pair_id") / pl.len().over("pair_id")).alias("r")
    )["r"].to_numpy()


def routed_score(df: pl.DataFrame, feats: list[str], pooled: lgb.Booster,
                 specialists: dict[str, lgb.Booster], family_proba: np.ndarray,
                 classes: list[str], spec_feats: dict[str, list[str]] | None = None,
                 fam_columns: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Family-weighted specialist ranks, blended with the pooled ranker's ranks.

    `pooled` and `specialists` may also be lists (a seed bag from `fit_bag`);
    within-pair ranks are then averaged over the members. `spec_feats` gives a
    family a longer feature list than the pooled ranker's, and `fam_columns`
    supplies the family-dependent similarity distance: a specialist must read the
    distance to its own family's gold hands, which is what it was trained on."""
    pooled_list = pooled if isinstance(pooled, list) else [pooled]
    spec_list = specialists if isinstance(specialists, list) else [specialists]
    x = df.select(feats).to_numpy().astype(np.float32)
    pair_ids = df["pair_id"]
    soft = np.zeros(df.height)
    for i, family in enumerate(classes):
        if family not in spec_list[0]:
            continue
        cols = (spec_feats or {}).get(family)
        if cols is None:
            xf = x
        else:
            sub = df
            if fam_columns is not None and family in fam_columns:
                sub = sub.with_columns(pl.Series("knn_gold_fam_d", fam_columns[family]))
            xf = sub.select(cols).to_numpy().astype(np.float32)
        ranks = np.mean([within_pair_rank(pair_ids, s[family].predict(xf)) for s in spec_list], axis=0)
        soft += family_proba[:, i] * ranks
    pooled_rank = np.mean([within_pair_rank(pair_ids, p.predict(x)) for p in pooled_list], axis=0)
    return SPECIALIST_WEIGHT * soft + (1.0 - SPECIALIST_WEIGHT) * pooled_rank


def sizing_names() -> list[str]:
    cols = [c for c in pl.scan_parquet(C.CACHE / "pair_sizing_dev.parquet").collect_schema().names()
            if c.startswith("sz_")]
    return cols + [f"pct_{c}" for c in cols]


def add_sizing(df: pl.DataFrame, phase: str) -> pl.DataFrame:
    """Bet size, stack depth and table context per (pair, hand) - stage 81.

    Offline these lift stage-1 MAP@5 0.6652 -> 0.6836, nearly all of it isolation
    (0.594 -> 0.686), which is what the token rerank used to carry.
    """
    s = pl.read_parquet(C.CACHE / f"pair_sizing_{phase[:3]}.parquet")
    cols = [c for c in s.columns if c.startswith("sz_")]
    key = ["p1", "p2"]
    return df.join(s, on=["p1", "p2", "hand_id"], how="left").with_columns(
        [(pl.col(c).rank("average").over(key) / pl.len().over(key)).alias(f"pct_{c}") for c in cols])


TEMPLATE_FOLD = ("chk_bet_fold", "bet_fold", "re_fold")
TEMPLATE_PAY = ("chk_bet_call", "bet_call", "re_call", "shove_call")
TEMPLATE_AGGR = ("chk_bet_raise", "bet_raise", "re_raise")
TEMPLATE_NAMES = ["t_fold_don", "t_fold_rec", "t_pay_don", "t_pay_rec", "t_aggr_don", "t_aggr_rec", "t_chk_chk"]


def add_templates(df: pl.DataFrame, phase: str) -> pl.DataFrame:
    """Partner-to-partner action templates (stage 74), oriented to the pair's
    dominant donor: check-bet-fold, bet-call, shove-call and so on. Gold hands
    carry more concession templates than unlisted planted ones."""
    t = pl.read_parquet(C.CACHE / f"pair_templates_{phase[:3]}.parquet")
    raw = [c for c in t.columns if c not in ("p1", "p2", "hand_id")]
    df = df.join(t, on=["p1", "p2", "hand_id"], how="left").with_columns([pl.col(c).fill_null(0) for c in raw])

    def grp(names, side):
        cols = [f"{n}_{g}_{side}" for n in names for g in ("pre", "post") if f"{n}_{g}_{side}" in raw]
        return pl.sum_horizontal([pl.col(c) for c in cols]) if cols else pl.lit(0)

    df = df.with_columns(grp(TEMPLATE_FOLD, 1).alias("_tf1"), grp(TEMPLATE_FOLD, 2).alias("_tf2"),
                         grp(TEMPLATE_PAY, 1).alias("_tp1"), grp(TEMPLATE_PAY, 2).alias("_tp2"),
                         grp(TEMPLATE_AGGR, 1).alias("_ta1"), grp(TEMPLATE_AGGR, 2).alias("_ta2"))
    key = ["p1", "p2"]
    don1 = ((pl.col("_tf1") + pl.col("_tp1")).sum().over(key) - (pl.col("_tf1") + pl.col("_tp1"))) >= \
           ((pl.col("_tf2") + pl.col("_tp2")).sum().over(key) - (pl.col("_tf2") + pl.col("_tp2")))
    chk = (pl.col("chk_chk_pre") + pl.col("chk_chk_post")) if "chk_chk_pre" in raw else pl.lit(0)
    return df.with_columns(
        pl.when(don1).then(pl.col("_tf1")).otherwise(pl.col("_tf2")).alias("t_fold_don"),
        pl.when(don1).then(pl.col("_tf2")).otherwise(pl.col("_tf1")).alias("t_fold_rec"),
        pl.when(don1).then(pl.col("_tp1")).otherwise(pl.col("_tp2")).alias("t_pay_don"),
        pl.when(don1).then(pl.col("_tp2")).otherwise(pl.col("_tp1")).alias("t_pay_rec"),
        pl.when(don1).then(pl.col("_ta1")).otherwise(pl.col("_ta2")).alias("t_aggr_don"),
        pl.when(don1).then(pl.col("_ta2")).otherwise(pl.col("_ta1")).alias("t_aggr_rec"),
        chk.alias("t_chk_chk"),
    ).drop([c for c in ("_tf1", "_tf2", "_tp1", "_tp2", "_ta1", "_ta2") if c in df.columns])


SIM_FEATS = ["knn_gold_d", "knn_planted_d", "knn_ratio", "knn_gold_fam_d",
             "sim_pair_mean", "sim_pair_max", "type_agree_pair"]
# Stage 87: soft play's planted hands are passive and look like the pair's
# ordinary ones, so "similar to the pair's other strong candidates" points the
# wrong way there (-0.013). The columns reach only the two families they help:
# routed, 0.6858 -> 0.6919 out of fold with default specialist settings.
SIM_ROUTED = ("directed_transfer", "coordinated_isolation")
SIM_K = 5
SIM_TOP = 20
SIM_STAGE1_SEEDS = 2


def similarity_on() -> bool:
    return __import__("os").environ.get("EV_SIMILARITY") == "1"


def spec_feature_map(feats: list[str]) -> dict[str, list[str]]:
    return {f: feats + SIM_FEATS for f in SIM_ROUTED} if similarity_on() else {}


def type_a_expr() -> pl.Expr:
    """Fold / concession-type event - the first list block for directed transfer
    and soft play."""
    return ((pl.col("ahead_fold") > 0) | (pl.col("fold_better_flag") > 0)
            | (pl.col("conceded_1_to_2").cast(pl.Int8)
               + pl.col("conceded_2_to_1").cast(pl.Int8) > 0)).cast(pl.Float64)


def add_planted_scores(df: pl.DataFrame) -> pl.DataFrame:
    """The pair family's planted-hand score per hand (stage 45), which picks the
    non-gold planted reference set. Development only - the evaluation rows are
    queries against those references, never references themselves."""
    hs = pl.read_parquet(C.CACHE / "hand_scores_family_dev.parquet")
    own = (pl.when(pl.col("behavior_family") == "directed_transfer").then(pl.col("hsdir"))
           .when(pl.col("behavior_family") == "soft_play").then(pl.col("hssoft"))
           .otherwise(pl.col("hsiso")))
    out = df.join(hs, on=["table_id", "p1", "p2", "hand_id"], how="left").with_columns(own.alias("own"))
    assert out.height == df.height, (out.height, df.height)
    return out.sort("pair_id", "hand_id")


def sim_space(df: pl.DataFrame, base: list[str],
              moments: tuple[np.ndarray, np.ndarray] | None = None):
    """Standardised base-feature matrix. `moments` reuses the development frame's
    mean and sd so evaluation rows land in the same space. Held in float32: an
    evaluation batch is over a million rows wide enough to cost gigabytes at f8,
    and the distances feed a tree model."""
    x = df.select(base).fill_null(0).to_numpy().astype(np.float64)
    if moments is None:
        moments = (x.mean(axis=0), x.std(axis=0))
    mu, sd = moments
    return ((x - mu) / np.where(sd < 1e-9, 1.0, sd)).astype(np.float32), moments


class SimilarityIndex:
    """Nearest-neighbour references in the standardised space: gold hands, non-gold
    planted hands, and gold hands per family. Fitted on development rows."""

    def __init__(self, z: np.ndarray, gold: np.ndarray, planted: np.ndarray, family: np.ndarray):
        self.gold = NearestNeighbors(n_neighbors=min(SIM_K, int(gold.sum()))).fit(z[gold])
        self.planted = NearestNeighbors(n_neighbors=min(SIM_K, int(planted.sum()))).fit(z[planted])
        self.family = {}
        for f in C.FAMILIES:
            m = gold & (family == f)
            if int(m.sum()) >= SIM_K:
                self.family[f] = NearestNeighbors(n_neighbors=SIM_K).fit(z[m])

    def distances(self, z: np.ndarray):
        return (self.gold.kneighbors(z)[0].mean(axis=1),
                self.planted.kneighbors(z)[0].mean(axis=1),
                {f: nn.kneighbors(z)[0].mean(axis=1) for f, nn in self.family.items()})


def pair_rank(pair_ids: pl.Series, score: np.ndarray) -> np.ndarray:
    """Ordinal position inside the pair, best hand first."""
    return pl.DataFrame({"pair_id": pair_ids, "s": score}).with_columns(
        pl.col("s").rank("ordinal", descending=True).over("pair_id").alias("r"))["r"].to_numpy()


def pair_consistency(pair_ids: pl.Series, type_a: np.ndarray, z: np.ndarray, rank: np.ndarray):
    """A pair looks for the same kind of evidence twice: similarity to its own
    other strong candidates, weighted by stage-1 rank and leaving the hand itself
    out, plus agreement with the pair's dominant event type."""
    sim_mean = np.zeros(rank.size)
    sim_max = np.zeros(rank.size)
    w = np.clip(1.0 / rank, 0, 1)
    for _, idx in pl.DataFrame({"p": pair_ids}).with_row_index("i").group_by("p").agg("i").rows():
        cand = np.asarray(idx)
        cand = cand[rank[cand] <= SIM_TOP]
        if cand.size < 2:
            continue
        d = np.linalg.norm(z[cand][:, None, :] - z[cand][None, :, :], axis=2).astype(np.float64)
        np.fill_diagonal(d, np.nan)
        ww = w[cand]
        sim = -d
        sim_mean[cand] = np.nansum(sim * ww[None, :], axis=1) / np.nansum(ww[None, :] * ~np.isnan(sim), axis=1)
        sim_max[cand] = np.nanmax(sim, axis=1)
    dom = pl.DataFrame({"p": pair_ids, "t": type_a * w}).select(pl.col("t").sum().over("p"))["t"].to_numpy()
    return sim_mean, sim_max, type_a * dom


def _sim_columns(df: pl.DataFrame, knn_gold: np.ndarray, knn_planted: np.ndarray,
                 knn_fam: np.ndarray, consistency) -> pl.DataFrame:
    sim_mean, sim_max, agree = consistency
    return df.with_columns(
        pl.Series("knn_gold_d", knn_gold), pl.Series("knn_planted_d", knn_planted),
        pl.Series("knn_ratio", knn_planted / (knn_gold + 1e-9)),
        pl.Series("knn_gold_fam_d", knn_fam),
        pl.Series("sim_pair_mean", sim_mean), pl.Series("sim_pair_max", sim_max),
        pl.Series("type_agree_pair", agree))


def add_similarity_dev(df: pl.DataFrame, base: list[str], folds: np.ndarray,
                       s1: np.ndarray) -> pl.DataFrame:
    """Out-of-fold similarity columns for the development training frame.

    `df` must carry `own`. The standardisation is fitted on the whole frame; the
    kNN references are fitted per fold on the other folds, so a row never sees a
    gold hand from its own table. `s1` are out-of-fold stage-1 scores."""
    df = df.with_columns(type_a_expr().alias("typeA"))
    z, _ = sim_space(df, base)
    gold = df["is_evidence"].to_numpy() == 1
    planted = (~gold) & (df["own"].fill_null(0).to_numpy() > 0.5)
    family = df["behavior_family"].to_numpy()
    knn_gold = np.zeros(df.height)
    knn_planted = np.zeros(df.height)
    knn_fam = np.zeros(df.height)
    for k in range(N_FOLDS):
        te = folds == k
        rows = np.flatnonzero(te)
        index = SimilarityIndex(z, gold & ~te, planted & ~te, family)
        knn_gold[te] = index.gold.kneighbors(z[te])[0].mean(axis=1)
        knn_planted[te] = index.planted.kneighbors(z[te])[0].mean(axis=1)
        for f, nn in index.family.items():
            m = rows[family[te] == f]
            if m.size:
                knn_fam[m] = nn.kneighbors(z[m])[0].mean(axis=1)
    consistency = pair_consistency(df["pair_id"], df["typeA"].to_numpy(), z, pair_rank(df["pair_id"], s1))
    return _sim_columns(df, knn_gold, knn_planted, knn_fam, consistency)


def fit_similarity(df: pl.DataFrame, base: list[str]):
    """Standardisation and kNN references over all development rows, for scoring
    evaluation rows. Nothing can leak into an evaluation row, so no fold split."""
    z, moments = sim_space(df, base)
    gold = df["is_evidence"].to_numpy() == 1
    planted = (~gold) & (df["own"].fill_null(0).to_numpy() > 0.5)
    return moments, SimilarityIndex(z, gold, planted, df["behavior_family"].to_numpy())


def add_similarity_eval(df: pl.DataFrame, base: list[str], moments, index: SimilarityIndex,
                        s1: np.ndarray) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Similarity columns for evaluation rows against the development references.

    `knn_gold_fam_d` comes back once per family rather than once per row: the
    evaluation family is only a probability, and each routed specialist is fed
    the distance to its own family's gold hands."""
    df = df.with_columns(type_a_expr().alias("typeA"))
    z, _ = sim_space(df, base, moments)
    knn_gold, knn_planted, knn_fam = index.distances(z)
    consistency = pair_consistency(df["pair_id"], df["typeA"].to_numpy(), z, pair_rank(df["pair_id"], s1))
    df = _sim_columns(df, knn_gold, knn_planted, np.zeros(df.height), consistency)
    return df, knn_fam


def oof_stage1(df: pl.DataFrame, feats: list[str], folds: np.ndarray,
               spec_feats: dict[str, list[str]] | None = None,
               seeds: int = SIM_STAGE1_SEEDS) -> np.ndarray:
    """The production blend out of fold over the training frame: bagged pooled
    ranker plus specialists, routed by the true family.

    Used to weight the similarity columns, and by the stage-91 check to score the
    routed engine. Fewer seeds than `EVIDENCE_SEEDS` - this only has to order a
    pair's candidates well enough to weight them."""
    x = df.select(feats).to_numpy().astype(np.float32)
    fam = df["behavior_family"].to_numpy()
    pair_ids = df["pair_id"]
    pooled = np.zeros(df.height)
    spec = np.zeros(df.height)
    seed0 = PARAMS["seed"]
    for s in range(seeds):
        PARAMS["seed"] = seed0 + s
        for k in range(N_FOLDS):
            te = folds == k
            tr = df.filter(pl.Series(folds != k))
            pooled[te] += within_pair_rank(pair_ids.filter(pl.Series(te)),
                                           fit_ranker(tr, feats).predict(x[te])) / seeds
            for f, m in fit_specialists(tr, feats, spec_feats).items():
                t = te & (fam == f)
                if not t.any():
                    continue
                cols = (spec_feats or {}).get(f, feats)
                xf = x[t] if cols == feats else df.filter(pl.Series(t)).select(cols).to_numpy().astype(np.float32)
                spec[t] += within_pair_rank(pair_ids.filter(pl.Series(t)), m.predict(xf)) / seeds
    PARAMS["seed"] = seed0
    return SPECIALIST_WEIGHT * spec + (1 - SPECIALIST_WEIGHT) * pooled


# Stages 115-117. A lambdarank model scores each hand on its own and never sees
# the competitor it is being ranked against; this compares two candidates
# directly and scores a hand by how often it wins against the rest of its pair's
# top 20. Over all three families the blend is +0.0028 with a table-resampled
# interval of [-0.0014, +0.0071], which does not clear zero - so it is not a
# general improvement and is not what this flag turns on. On
# coordinated_isolation alone it is +0.0111 [+0.0014, +0.0209], and restricting
# it to that family keeps the point estimate (+0.0027) while the interval clears
# zero, because the noise of the other 280 pairs drops out. EV_ISO_PAIRWISE=1
# turns on that restricted form only. Unset in the default pipeline.
PAIRWISE_TOP_K = 20
PAIRWISE_WEIGHT = 0.3
PAIRWISE_SEEDS = 6
PAIRWISE_ROUNDS = 300
PAIRWISE_FAMILY = "coordinated_isolation"
# Pairs per prediction chunk. A pair contributes 20*19 couple rows of 3x the
# feature block, so a chunk of 400 pairs is about 600 MB at evaluation width.
PAIRWISE_CHUNK = 400
# A couple row carries three copies of the feature block and neighbouring columns
# inside a block are near-duplicates (a raw signal and its within-pair
# percentile), so the column sample sits well below the ranker's.
PAIRWISE_PARAMS = dict(**C.LGB_REPRO,
    objective="binary", metric="binary_logloss", learning_rate=0.05,
    num_leaves=31, min_data_in_leaf=40, feature_fraction=0.4,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbosity=-1,
    num_threads=C.N_THREADS, seed=C.SEED)


def pairwise_on() -> bool:
    return __import__("os").environ.get("EV_ISO_PAIRWISE") == "1"


def pairwise_rank(pair_ids: pl.Series, score: np.ndarray) -> np.ndarray:
    """1-based position of each hand inside its pair, best first."""
    return (pl.DataFrame({"pair_id": pair_ids, "s": score}).with_columns(
        pl.col("s").rank("ordinal", descending=True).over("pair_id").alias("r"))["r"].to_numpy())


def pairwise_couples(pair_ids: np.ndarray, rank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every ordered couple of distinct top-20 candidates, as row indices."""
    rows = np.flatnonzero(rank <= PAIRWISE_TOP_K)
    pid = pair_ids[rows]
    order = np.argsort(pid, kind="stable")
    rows, pid = rows[order], pid[order]
    bounds = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
    left, right = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        idx = rows[lo:hi]
        a, b = np.repeat(idx, idx.size), np.tile(idx, idx.size)
        keep = a != b
        left.append(a[keep])
        right.append(b[keep])
    return np.concatenate(left), np.concatenate(right)


def pairwise_matrix(x: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Both hands' feature blocks and their difference, in that order."""
    n = x.shape[1]
    z = np.empty((ia.size, 3 * n), dtype=np.float32)
    z[:, :n] = x[ia]
    z[:, n:2 * n] = x[ib]
    np.subtract(z[:, :n], z[:, n:2 * n], out=z[:, 2 * n:])
    return z


def pairwise_borda(prob: np.ndarray, ia: np.ndarray, ib: np.ndarray, n: int) -> np.ndarray:
    """Average win probability against the pair's other candidates.

    Both directions of a couple are scored, so the two views are averaged; that
    cancels the part of the comparator that is not exactly antisymmetric. Rows
    that were never compared - anything outside the top 20 - come back NaN."""
    cnt = np.bincount(ia, minlength=n).astype(np.float64)
    won = np.bincount(ia, weights=prob, minlength=n)
    lost = np.bincount(ib, weights=prob, minlength=n)
    seen = np.maximum(cnt, 1.0)
    out = 0.5 * (won / seen + (1.0 - lost / seen))
    out[cnt == 0] = np.nan
    return out


def pairwise_blend(pair_ids: pl.Series, stage1: np.ndarray, borda: np.ndarray) -> np.ndarray:
    """Rank-blend the comparator with the listwise score, as the engine blends
    its own parts. Hands the comparator never saw keep the listwise order and
    sit below everything it did see."""
    below = -1.0 + 1e-3 * within_pair_rank(pair_ids, stage1)
    pw = np.where(np.isnan(borda), below, borda)
    return (PAIRWISE_WEIGHT * within_pair_rank(pair_ids, pw)
            + (1.0 - PAIRWISE_WEIGHT) * within_pair_rank(pair_ids, stage1))


def pairwise_training_couples(df: pl.DataFrame, stage1: np.ndarray):
    """Couples of one gold and one non-gold top-20 candidate, both orders.

    Labels choose which couples exist and which way round they count. No label
    reaches a feature column: stage 115 asserts that by permuting `is_evidence`
    inside each pair and requiring the whole feature block and the candidate set
    to come back unchanged."""
    rank = pairwise_rank(df["pair_id"], stage1)
    ia, ib = pairwise_couples(df["pair_id"].to_numpy(), rank)
    gold = df["is_evidence"].to_numpy()
    return ia, ib, gold[ia], gold[ia] != gold[ib]


def fit_pairwise(df: pl.DataFrame, feats: list[str], stage1: np.ndarray) -> list[lgb.Booster]:
    """One comparator per seed over every development couple, for scoring the
    evaluation phase where there is no fold to hold out."""
    ia, ib, ya, keep = pairwise_training_couples(df, stage1)
    z = pairwise_matrix(df.select(feats).to_numpy().astype(np.float32), ia[keep], ib[keep])
    label = ya[keep].astype(np.float64)
    return [lgb.train({**PAIRWISE_PARAMS, "seed": C.SEED + s}, lgb.Dataset(z, label=label),
                      PAIRWISE_ROUNDS) for s in range(PAIRWISE_SEEDS)]


def pairwise_oof(df: pl.DataFrame, feats: list[str], stage1: np.ndarray,
                 folds: np.ndarray) -> np.ndarray:
    """Comparator Borda score out of fold, for measuring on development.

    A couple joins two hands of one pair, so it sits wholly inside one table and
    one fold; the assertion below is what makes that a checked fact rather than
    an assumption."""
    ia, ib, ya, keep = pairwise_training_couples(df, stage1)
    cf = folds[ia]
    assert np.array_equal(cf, folds[ib]), "a couple crosses folds"
    x = df.select(feats).to_numpy().astype(np.float32)
    z = pairwise_matrix(x, ia, ib)
    prob = np.zeros(ia.size)
    for s in range(PAIRWISE_SEEDS):
        for k in range(N_FOLDS):
            tr, te = keep & (cf != k), cf == k
            if not tr.any() or not te.any():
                continue
            booster = lgb.train({**PAIRWISE_PARAMS, "seed": C.SEED + s},
                                lgb.Dataset(z[tr], label=ya[tr].astype(np.float64)),
                                PAIRWISE_ROUNDS)
            prob[te] += booster.predict(z[te]) / PAIRWISE_SEEDS
    return pairwise_borda(prob, ia, ib, df.height)


def pairwise_rescore(df: pl.DataFrame, feats: list[str], stage1: np.ndarray,
                     boosters: list[lgb.Booster], apply_to: np.ndarray) -> np.ndarray:
    """Re-order the pairs in `apply_to` with the comparator, leave the rest alone.

    `apply_to` is a row mask that is constant inside a pair, so a pair's ordering
    is produced entirely by one of the two scores and the two scales never mix
    inside one ranking. Couples are built and predicted in chunks of pairs
    because the matrix is 380 rows per pair."""
    pair_ids = df["pair_id"]
    pid = pair_ids.to_numpy()
    assert pl.DataFrame({"p": pid, "m": apply_to}).group_by("p").agg(
        pl.col("m").n_unique().alias("n"))["n"].max() == 1, "the mask splits a pair"
    if not apply_to.any():
        return stage1
    x = df.select(feats).to_numpy().astype(np.float32)
    rank = pairwise_rank(pair_ids, np.where(apply_to, stage1, -np.inf))
    borda = np.full(df.height, np.nan)
    targets = pl.Series(pid).filter(pl.Series(apply_to)).unique(maintain_order=True).to_list()
    for i in range(0, len(targets), PAIRWISE_CHUNK):
        chunk = np.isin(pid, targets[i:i + PAIRWISE_CHUNK])
        ia, ib = pairwise_couples(pid, np.where(chunk, rank, PAIRWISE_TOP_K + 1))
        z = pairwise_matrix(x, ia, ib)
        prob = np.mean([b.predict(z) for b in boosters], axis=0)
        part = pairwise_borda(prob, ia, ib, df.height)
        borda = np.where(np.isnan(part), borda, part)
    return np.where(apply_to, pairwise_blend(pair_ids, stage1, borda), stage1)


def load_pair_hands(pair_keys: pl.DataFrame, phase: str) -> pl.DataFrame:
    lf = pl.scan_parquet(C.CACHE / "pair_hands" / "*.parquet").filter(pl.col("phase") == phase)
    df = derive(lf).join(pair_keys.lazy(), on=["p1", "p2"], how="inner").collect()
    df = add_within_pair(df)
    if __import__("os").environ.get("EV_SIZING") == "1":
        df = add_sizing(df, phase)
    if __import__("os").environ.get("EV_TEMPLATES") == "1":
        df = add_templates(df, phase)
    return df


def main() -> None:
    t0 = time.time()
    labels = pl.read_csv(C.DEV_LABELS).with_columns(
        pl.min_horizontal("player_1", "player_2").alias("p1"),
        pl.max_horizontal("player_1", "player_2").alias("p2"),
    )
    pos = labels.filter(pl.col("label") == 1).select("pair_id", "p1", "p2", "behavior_family")
    evidence = pl.read_csv(C.DEV_EVIDENCE).select("pair_id", "hand_id").unique()

    df = load_pair_hands(pos, "development")
    df = df.join(evidence.with_columns(pl.lit(1, dtype=pl.Int8).alias("is_evidence")),
                 on=["pair_id", "hand_id"], how="left").with_columns(
        pl.col("is_evidence").fill_null(0))
    print(f"training rows: {df.height:,} over {df['pair_id'].n_unique()} pairs "
          f"({df['is_evidence'].sum()} evidence hands, {time.time() - t0:.0f}s)")

    feats = feature_names()
    df = df.sort("pair_id")
    x = df.select(feats).to_numpy().astype(np.float32)
    y = df["is_evidence"].to_numpy()
    pair_ids = df["pair_id"].to_numpy()
    folds = table_folds(df["table_id"].to_numpy(), N_FOLDS)

    oof = np.zeros(df.height)
    for fold in range(N_FOLDS):
        train = folds != fold
        tr = df.filter(pl.Series(train)).sort("pair_id")
        groups = tr.group_by("pair_id", maintain_order=True).len()["len"].to_numpy()
        booster = lgb.train(
            PARAMS,
            lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32),
                        label=tr["is_evidence"].to_numpy(), group=groups),
            num_boost_round=N_ROUNDS,
        )
        oof[~train] = booster.predict(x[~train])
        print(f"  fold {fold} ({time.time() - t0:.0f}s)")

    scored = df.with_columns(pl.Series("score", oof))
    top5 = (
        scored.sort(["pair_id", "score", "hand_id"], descending=[False, True, False])
        .group_by("pair_id", maintain_order=True).head(5)
    )
    submitted = {p: g["hand_id"].to_list()
                 for p, g in top5.group_by("pair_id", maintain_order=True)}
    relevant = {p: set(g["hand_id"].to_list())
                for p, g in evidence.group_by("pair_id", maintain_order=True)}
    valid = {p: set(g["hand_id"].to_list())
             for p, g in scored.group_by("pair_id", maintain_order=True)}

    strict = evidence_map5(submitted, relevant, valid)
    print(f"\n=== evidence ===\nstrict OOF MAP@5 : {strict:.5f}")

    # Baseline: rank by raw chip flow, the starter notebook's heuristic.
    base = (
        scored.sort(["pair_id", "transfer_gross", "hand_id"], descending=[False, True, False])
        .group_by("pair_id", maintain_order=True).head(5)
    )
    base_sub = {p: g["hand_id"].to_list() for p, g in base.group_by("pair_id", maintain_order=True)}
    print(f"chip-flow baseline: {evidence_map5(base_sub, relevant, valid):.5f}")

    imp = sorted(zip(feats, booster.feature_importance("gain")), key=lambda kv: -kv[1])
    print("\ntop evidence features:")
    for name, gain in imp[:15]:
        print(f"  {gain:12.0f}  {name}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
