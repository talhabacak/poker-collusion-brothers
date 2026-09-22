"""Decision-time facts behind an evidence case review, read from the public log only.

`CASE_REVIEWS.md` states, for each reviewed pair, what the submitted evidence hands
show. Everything it states is printed by this script from `hands`, `seats` and
`actions`, so a reviewer can check every sentence without trusting the pipeline.

Per evidence hand it prints, for each street a player acted on:

  * the board as it was dealt at that moment - not the final board, which is what
    makes a fold look better or worse than it was;
  * both players' best five at that moment;
  * who the folding player was answering (the last bet, raise or all-in before the
    fold), so "folded to the partner" is separated from "folded to the table";
  * whether the player who gave up held the better hand when they did.

Per pair it prints the same concession event counted over every hand the two
shared, and - the comparison that matters - the same rate for each player against
every other opponent they faced in the evaluation phase.

    python verify/case_facts.py --submission submissions/tarik_v55_mil_evidence.csv \
        --raw data/raw --pairs P8E67334B0E04,PD91D4EB58627 [--json out.json]

With `--top-per-family N` it takes the N highest-risk pairs of each predicted
family instead of an explicit list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "src"))

from pokercol.cards import eval_best5, parse_cards  # noqa: E402

EVIDENCE_COLS = [f"evidence_hand_{i}" for i in range(1, 6)]
FAMILIES = ("directed_transfer", "soft_play", "coordinated_isolation")
AGGRESSIVE = ("bet", "raise", "all_in")
STREETS = ("preflop", "flop", "turn", "river")
BOARD_CARDS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
CATEGORIES = ("high card", "pair", "two pair", "trips", "straight", "flush",
              "full house", "quads", "straight flush")


def best5(hole: tuple[str, str], board: list[str]) -> tuple[int, str] | None:
    """Best five-card score and its name from two hole cards and a visible board."""
    if len(board) < 3:
        return None
    rank, suit = parse_cards([[hole[0], hole[1], *board]])
    score = int(eval_best5(rank, suit)[0])
    return score, CATEGORIES[score // (14 ** 5)]


def load(raw: Path, hand_ids: list[str]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    hands = pl.scan_parquet(raw / "hands.parquet").filter(pl.col("hand_id").is_in(hand_ids)).collect()
    seats = pl.scan_parquet(raw / "seats.parquet").filter(pl.col("hand_id").is_in(hand_ids)).collect()
    actions = (pl.scan_parquet(raw / "actions.parquet").filter(pl.col("hand_id").is_in(hand_ids))
               .collect().sort("hand_id", "action_no"))
    return hands, seats, actions


def concessions(hands: pl.DataFrame, seats: pl.DataFrame, actions: pl.DataFrame) -> list[dict]:
    """Every fold in `actions`, with who it answered and whether it gave up the better hand.

    A fold is attributed to the last aggressive action before it. Preflop folds are
    reported but never counted as conceding a better hand: two hole cards have no
    best-five to compare, and calling a preflop fold a concession would count normal
    folding as evidence.
    """
    board_of = {r["hand_id"]: (r["board_cards"] or "").split() for r in hands.iter_rows(named=True)}
    bb_of = {r["hand_id"]: r["big_blind"] for r in hands.iter_rows(named=True)}
    hole_of = {(r["hand_id"], r["player_id"]): (r["hole_card_1"], r["hole_card_2"])
               for r in seats.iter_rows(named=True)}
    out: list[dict] = []
    last_agg: dict[str, tuple[str, str, int]] = {}
    current = None
    for a in actions.iter_rows(named=True):
        hid = a["hand_id"]
        if hid != current:
            current, last_agg = hid, {}
        if a["action"] in AGGRESSIVE:
            last_agg[hid] = (a["player_id"], a["street"], a["amount"])
            continue
        if a["action"] != "fold" or hid not in last_agg:
            continue
        agg_id, agg_street, agg_amount = last_agg[hid]
        board = board_of[hid][: BOARD_CARDS[a["street"]]]
        mine = best5(hole_of[(hid, a["player_id"])], board)
        theirs = best5(hole_of.get((hid, agg_id), ("", "")), board) if (hid, agg_id) in hole_of else None
        out.append({
            "hand_id": hid, "street": a["street"], "folder": a["player_id"], "to": agg_id,
            "to_bet_bb": round(agg_amount / bb_of[hid], 1), "to_street": agg_street,
            "folder_hand": mine[1] if mine else None, "aggressor_hand": theirs[1] if theirs else None,
            "better": bool(mine and theirs and mine[0] > theirs[0]),
            "comparable": bool(mine and theirs),
        })
    return out


def pair_profile(a: str, b: str, raw: Path) -> dict:
    """Events between the two, against the same events against every other opponent.

    Tables are closed - a player meets the same group of opponents and nobody else -
    so "folded to the partner 20 times against 110 times to everyone else" says
    nothing on its own: the partner is dealt in with the player in only a fraction of
    their hands. Every count here is divided by its exposure - hands shared with the
    partner on one side, (hand x other opponent) slots on the other - and it is the
    ratio of those two rates that the case reviews quote.
    """
    seats_all = (pl.scan_parquet(raw / "seats.parquet")
                 .filter(pl.col("player_id").is_in([a, b])).select("hand_id", "player_id").collect())
    eval_hands = (pl.scan_parquet(raw / "hands.parquet")
                  .filter(pl.col("phase") == "evaluation").select("hand_id").collect())
    seats_all = seats_all.join(eval_hands, on="hand_id", how="inner")
    shared = (seats_all.group_by("hand_id").len().filter(pl.col("len") == 2)["hand_id"].to_list())
    every = seats_all["hand_id"].unique().to_list()
    hands, seats, actions = load(raw, every)
    events = concessions(hands, seats, actions)
    shared_set = set(shared)

    hands_of: dict[str, set[str]] = {a: set(), b: set()}
    for r in seats_all.iter_rows(named=True):
        hands_of[r["player_id"]].add(r["hand_id"])
    seated: dict[str, list[str]] = {}
    for r in seats.iter_rows(named=True):
        seated.setdefault(r["hand_id"], []).append(r["player_id"])
    preflop_agg: dict[str, set[str]] = {}
    for r in actions.filter((pl.col("street") == "preflop")
                            & pl.col("action").is_in(AGGRESSIVE)).iter_rows(named=True):
        preflop_agg.setdefault(r["hand_id"], set()).add(r["player_id"])
    postflop_agg: dict[str, set[str]] = {}
    for r in actions.filter((pl.col("street") != "preflop")
                            & pl.col("action").is_in(AGGRESSIVE)).iter_rows(named=True):
        postflop_agg.setdefault(r["hand_id"], set()).add(r["player_id"])
    showdown: dict[str, set[str]] = {}
    for r in seats.filter(pl.col("went_to_showdown")).iter_rows(named=True):
        showdown.setdefault(r["hand_id"], set()).add(r["player_id"])

    def rates(player: str, partner: str) -> dict:
        with_partner = len(shared_set)
        with_others = sum(len(seated.get(h, [])) - 1 for h in hands_of[player]) - with_partner
        mine = [e for e in events if e["folder"] == player]
        vs_partner = [e for e in mine if e["to"] == partner]
        vs_field = [e for e in mine if e["to"] != partner]
        conceded_p = [e for e in vs_partner if e["better"]]
        conceded_f = [e for e in vs_field if e["better"]]
        # Both of them aggressive in the same preflop round: the isolation relay.
        co_p = sum(1 for h in hands_of[player] & hands_of[partner]
                   if {player, partner} <= preflop_agg.get(h, set()))
        co_f = sum(1 for h in hands_of[player] if player in preflop_agg.get(h, set())
                   for o in seated.get(h, [])
                   if o not in (player, partner) and o in preflop_agg.get(h, set()))
        # Soft play does not show up as folding the better hand but as not betting it:
        # both reach showdown and neither puts in a bet or raise after the flop.
        sd_p = [h for h in hands_of[player] & hands_of[partner]
                if {player, partner} <= showdown.get(h, set())]
        quiet_p = [h for h in sd_p if not ({player, partner} & postflop_agg.get(h, set()))]
        sd_f = [(h, o) for h in hands_of[player] if player in showdown.get(h, set())
                for o in showdown.get(h, set()) if o not in (player, partner)]
        quiet_f = [(h, o) for h, o in sd_f if not ({player, o} & postflop_agg.get(h, set()))]
        per = lambda n, d: round(n / d, 5) if d else None
        lift = lambda n_p, d_p, n_f, d_f: (round((n_p / d_p) / (n_f / d_f), 1)
                                           if n_p and n_f and d_p and d_f else None)
        return {
            "exposure_hands_with_partner": with_partner,
            "exposure_opponent_hand_slots_with_others": with_others,
            "evaluation_hands_played": len(hands_of[player]),
            "folds_to_partner": len(vs_partner),
            "folds_to_others": len(vs_field),
            "fold_rate_partner": per(len(vs_partner), with_partner),
            "fold_rate_others": per(len(vs_field), with_others),
            "fold_lift": lift(len(vs_partner), with_partner, len(vs_field), with_others),
            "better_hand_conceded_to_partner": len(conceded_p),
            "better_hand_conceded_to_others": len(conceded_f),
            "concession_rate_partner": per(len(conceded_p), with_partner),
            "concession_rate_others": per(len(conceded_f), with_others),
            "concession_lift": lift(len(conceded_p), with_partner, len(conceded_f), with_others),
            "both_raised_preflop_with_partner": co_p,
            "both_raised_preflop_with_others": co_f,
            "co_raise_rate_partner": per(co_p, with_partner),
            "co_raise_rate_others": per(co_f, with_others),
            "co_raise_lift": lift(co_p, with_partner, co_f, with_others),
            "showdowns_with_partner": len(sd_p),
            "showdowns_with_others": len(sd_f),
            "no_postflop_bet_by_either_with_partner": len(quiet_p),
            "no_postflop_bet_by_either_with_others": len(quiet_f),
            "quiet_showdown_rate_partner": per(len(quiet_p), len(sd_p)),
            "quiet_showdown_rate_others": per(len(quiet_f), len(sd_f)),
            "quiet_showdown_lift": lift(len(quiet_p), len(sd_p), len(quiet_f), len(sd_f)),
        }

    net = (seats.filter(pl.col("hand_id").is_in(shared))
           .join(hands.select("hand_id", "big_blind"), on="hand_id")
           .group_by("player_id").agg((pl.col("net_chips") / pl.col("big_blind")).sum().alias("net_bb")))
    return {
        "players": [a, b],
        "shared_evaluation_hands": len(shared_set),
        "net_bb_over_shared_hands": {r["player_id"]: round(r["net_bb"], 1)
                                     for r in net.iter_rows(named=True) if r["player_id"] in (a, b)},
        a: rates(a, b),
        b: rates(b, a),
    }


def hand_report(hid: str, a: str, b: str, hands, seats, actions) -> dict:
    h = hands.filter(pl.col("hand_id") == hid).row(0, named=True)
    bb = h["big_blind"]
    board = (h["board_cards"] or "").split()
    s = {r["player_id"]: r for r in seats.filter(pl.col("hand_id") == hid).iter_rows(named=True)}
    acts = actions.filter(pl.col("hand_id") == hid)
    tag = {a: "A", b: "B"}
    streets_seen = [st for st in STREETS if st in set(acts["street"].to_list())]
    per_street = []
    for st in streets_seen:
        vis = board[: BOARD_CARDS[st]]
        row = {"street": st, "board": " ".join(vis) or "-"}
        for pid, who in ((a, "A"), (b, "B")):
            if pid not in s:
                continue
            got = best5((s[pid]["hole_card_1"], s[pid]["hole_card_2"]), vis)
            row[who] = got[1] if got else "preflop"
        per_street.append(row)
    seq = []
    for act in acts.iter_rows(named=True):
        who = tag.get(act["player_id"], "other")
        amount = f" {act['amount'] / bb:.1f}BB" if act["amount"] else ""
        seq.append(f"{act['street'][0]}:{who} {act['action']}{amount}")
    ev = concessions(hands.filter(pl.col("hand_id") == hid),
                     seats.filter(pl.col("hand_id") == hid), acts)
    return {
        "hand_id": hid,
        "stakes": f"{h['small_blind']}/{bb}",
        "board": h["board_cards"] or "-",
        "final_pot_bb": round(h["final_pot"] / bb, 1),
        "players": {tag[p]: {"hole": f"{s[p]['hole_card_1']} {s[p]['hole_card_2']}",
                             "contributed_bb": round(s[p]["total_contribution"] / bb, 1),
                             "net_bb": round(s[p]["net_chips"] / bb, 1),
                             "folded": bool(s[p]["folded"]),
                             "showdown": bool(s[p]["went_to_showdown"])} for p in (a, b) if p in s},
        "strength_by_street": per_street,
        "actions": seq,
        "folds": [{**e, "folder": tag.get(e["folder"], "other"), "to": tag.get(e["to"], "other")}
                  for e in ev if e["folder"] in (a, b)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--top-per-family", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    raw = Path(args.raw)

    sub = pl.read_csv(args.submission)
    pairs_meta = pl.read_csv(raw / "evaluation_pairs.csv")
    if args.pairs:
        wanted = [p.strip() for p in args.pairs.split(",") if p.strip()]
        picked = sub.filter(pl.col("pair_id").is_in(wanted))
        picked = picked.with_columns(
            pl.col("pair_id").cast(pl.Enum(wanted)).to_physical().alias("_o")).sort("_o").drop("_o")
    else:
        n = args.top_per_family or 2
        picked = (sub.filter(pl.col("predicted_behavior").is_in(list(FAMILIES)))
                  .sort("risk_score", descending=True)
                  .group_by("predicted_behavior", maintain_order=True).head(n))
    picked = picked.join(pairs_meta, on="pair_id", how="left")

    report = []
    for r in picked.iter_rows(named=True):
        a, b = r["player_1"], r["player_2"]
        hand_ids = [r[c] for c in EVIDENCE_COLS if r[c] != "NO_EVIDENCE"]
        hands, seats, actions = load(raw, hand_ids)
        entry = {
            "pair_id": r["pair_id"],
            "predicted_behavior": r["predicted_behavior"],
            "risk_score": r["risk_score"],
            "A": a, "B": b,
            "profile": pair_profile(a, b, raw),
            "evidence": [hand_report(h, a, b, hands, seats, actions) for h in hand_ids],
        }
        report.append(entry)
        p = entry["profile"]
        print(f"\n=== {r['pair_id']}  {r['predicted_behavior']}  risk {r['risk_score']:.4f}")
        print(f"    A {a}  B {b}  shared evaluation hands {p['shared_evaluation_hands']}"
              f"  net BB over shared hands {p['net_bb_over_shared_hands']}")
        for who, pid in (("A", a), ("B", b)):
            q = p[pid]
            print(f"    {who} exposure: {q['exposure_hands_with_partner']} hands with the partner, "
                  f"{q['exposure_opponent_hand_slots_with_others']} opponent-hand slots with others")
            print(f"       folds to partner {q['folds_to_partner']} "
                  f"({q['fold_rate_partner']:.3f}/hand) vs others {q['folds_to_others']} "
                  f"({q['fold_rate_others']:.3f}/slot)  lift {q['fold_lift']}")
            print(f"       better hand given up to partner {q['better_hand_conceded_to_partner']} "
                  f"({q['concession_rate_partner']:.4f}/hand) vs others "
                  f"{q['better_hand_conceded_to_others']} ({q['concession_rate_others']:.4f}/slot)"
                  f"  lift {q['concession_lift']}")
            print(f"       both raised preflop with partner {q['both_raised_preflop_with_partner']} "
                  f"({q['co_raise_rate_partner']:.4f}/hand) vs others "
                  f"{q['both_raised_preflop_with_others']} ({q['co_raise_rate_others']:.4f}/slot)"
                  f"  lift {q['co_raise_lift']}")
            print(f"       showdowns with no postflop bet by either: partner "
                  f"{q['no_postflop_bet_by_either_with_partner']}/{q['showdowns_with_partner']} "
                  f"vs others {q['no_postflop_bet_by_either_with_others']}/{q['showdowns_with_others']}"
                  f"  lift {q['quiet_showdown_lift']}")
        for hand in entry["evidence"]:
            print(f"    {hand['hand_id']} {hand['stakes']} board {hand['board']} pot {hand['final_pot_bb']}BB")
            for who, v in hand["players"].items():
                print(f"       {who} {v['hole']:>7}  in {v['contributed_bb']:>6}BB  net {v['net_bb']:>+7}BB"
                      f"{'  folded' if v['folded'] else ''}{'  showdown' if v['showdown'] else ''}")
            for st in hand["strength_by_street"]:
                print(f"       {st['street']:<8} {st['board']:<15} A {st.get('A', '-'):<12} B {st.get('B', '-')}")
            for f in hand["folds"]:
                print(f"       fold: {f['folder']} on {f['street']} to {f['to']}'s "
                      f"{f['to_bet_bb']}BB ({f['to_street']}) - {f['folder_hand']} vs {f['aggressor_hand']}"
                      f"{'  BETTER HAND GIVEN UP' if f['better'] else ''}")
            print("       " + ", ".join(hand["actions"]))

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
