"""Stage 16: human-readable evidence case reviews for winner verification.

Prize eligibility asks for five evidence case reviews, each with the pair, the
evidence hands, the observable behaviour and a plausible benign explanation.
This drafts candidates straight from a submission file: for the highest-risk
evaluation pairs of each routed family it replays the submitted evidence hands
from the raw action log and states, per hand, only what the public log shows -
cards, actions, amounts, who won - alongside the quantitative signals the
pipeline used.

The "why suspicious" and "benign alternative" lines are drafted from the
observed pattern and are meant to be read and edited by a person before
submission; they are not claims the model makes.

    python scripts/16_case_reviews.py [submission.csv] [--per-family 4]
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokercol import config as C
from pokercol.cards import eval_best5, parse_cards, split_board

OUT = C.ARTIFACTS / "case_reviews.md"
EVIDENCE_COLS = [f"evidence_hand_{i}" for i in range(1, 6)]

BENIGN = {
    "directed_transfer": "Weak or tilted play by the losing player: folding marginal made hands to "
                         "aggression and paying off without reads is common among recreational "
                         "players, and repeated meetings between two regulars at one table can "
                         "concentrate those losses on one opponent by chance.",
    "soft_play": "A cautious, low-variance style: some players check down strong hands against "
                 "opponents they respect or fear, and two passive players who often end up heads-up "
                 "will produce many check-downs without any agreement.",
    "coordinated_isolation": "Two independently aggressive players: loose-aggressive regulars "
                             "frequently 3-bet and 4-bet light preflop, and when two of them share "
                             "a table their raises will collide and squeeze third parties out "
                             "without coordination.",
}
WHY = {
    "directed_transfer": "Chips repeatedly move one way between the same two players on hands where "
                         "the losing side held the better cards at the moment it gave up, and the "
                         "pattern is specific to this opponent rather than the player's general game.",
    "soft_play": "When the two are heads-up the stronger hand repeatedly declines to bet or folds "
                 "to minimal pressure, so pots between them stay small or move against card strength.",
    "coordinated_isolation": "The two raise and re-raise in the same preflop round with hands that "
                             "do not justify it, pushing the other players out, and do so far more "
                             "often together than either does with other opponents.",
}


def load_hand_frames(hand_ids: list[str]):
    hands = pl.scan_parquet(C.HANDS).filter(pl.col("hand_id").is_in(hand_ids)).collect()
    seats = pl.scan_parquet(C.SEATS).filter(pl.col("hand_id").is_in(hand_ids)).collect()
    actions = (pl.scan_parquet(C.ACTIONS).filter(pl.col("hand_id").is_in(hand_ids))
               .collect().sort("hand_id", "action_no"))
    return hands, seats, actions


def made_hand(hole: tuple[str, str], board: str) -> str:
    if not board:
        return "no board"
    names = ["high card", "pair", "two pair", "trips", "straight", "flush",
             "full house", "quads", "straight flush"]
    rank, suit = parse_cards([[hole[0], hole[1], *board.split()]])
    return names[int(eval_best5(rank, suit)[0]) // (14 ** 5)]


def describe_hand(hand_id: str, a: str, b: str, hands, seats, actions) -> list[str]:
    h = hands.filter(pl.col("hand_id") == hand_id).row(0, named=True)
    s = {r["player_id"]: r for r in seats.filter(pl.col("hand_id") == hand_id).iter_rows(named=True)}
    bb = h["big_blind"]
    tag = {a: "A", b: "B"}
    lines = [f"- **{hand_id}** (stakes {h['small_blind']}/{bb}, board `{h['board_cards'] or '-'}`, "
             f"final pot {h['final_pot'] / bb:.1f} BB)"]
    for pid, who in ((a, "A"), (b, "B")):
        r = s[pid]
        lines.append(f"  - {who} held `{r['hole_card_1']} {r['hole_card_2']}` "
                     f"({made_hand((r['hole_card_1'], r['hole_card_2']), h['board_cards'])}), "
                     f"put in {r['total_contribution'] / bb:.1f} BB, net {r['net_chips'] / bb:+.1f} BB"
                     f"{', folded' if r['folded'] else ''}")
    seq = []
    for act in actions.filter(pl.col("hand_id") == hand_id).iter_rows(named=True):
        who = tag.get(act["player_id"], "other")
        amount = f" {act['amount'] / bb:.1f}BB" if act["amount"] else ""
        seq.append(f"{act['street'][0]}:{who} {act['action']}{amount}")
    lines.append("  - action: " + ", ".join(seq))
    return lines


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sub_path = Path(args[0]) if args else C.SUBMISSIONS / "submission.csv"
    per_family = 4
    if "--per-family" in sys.argv:
        per_family = int(sys.argv[sys.argv.index("--per-family") + 1])

    sub = pl.read_csv(sub_path)
    pairs = pl.read_csv(C.EVAL_PAIRS)
    picked = (
        sub.filter(pl.col("predicted_behavior").is_in(list(C.FAMILIES)))
        .sort("risk_score", descending=True)
        .group_by("predicted_behavior", maintain_order=True).head(per_family)
        .join(pairs, on="pair_id", how="left")
    )
    hand_ids = [h for row in picked.select(EVIDENCE_COLS).iter_rows() for h in row
                if h != C.NO_EVIDENCE]
    hands, seats, actions = load_hand_frames(hand_ids)

    out = [f"# Evidence case review candidates\n",
           f"Drafted from `{sub_path.name}`. Players are labelled A (`player_1`) and B "
           f"(`player_2`); amounts are in big blinds. Every line under a hand is read directly "
           f"from `hands`, `seats` and `actions`. The last two paragraphs of each case are drafts "
           f"to be checked and edited by a person.\n"]
    for i, r in enumerate(picked.iter_rows(named=True), start=1):
        family = r["predicted_behavior"]
        out.append(f"\n## Case {i}: `{r['pair_id']}` - {family.replace('_', ' ')}\n")
        out.append(f"- Players: A `{r['player_1']}`, B `{r['player_2']}`; "
                   f"{r['shared_hands']} shared evaluation hands; risk score {r['risk_score']:.4f}\n")
        out.append("### Submitted evidence and observable behaviour\n")
        for col in EVIDENCE_COLS:
            if r[col] != C.NO_EVIDENCE:
                out.extend(describe_hand(r[col], r["player_1"], r["player_2"], hands, seats, actions))
        out.append(f"\n### Why suspicious (draft)\n\n{WHY[family]}\n")
        out.append(f"### Plausible benign alternative (draft)\n\n{BENIGN[family]}\n")

    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {len(picked)} candidate cases -> {OUT}")


if __name__ == "__main__":
    main()
