# Five evidence case reviews

All five are pairs from the team's selected submission
`tarik_v55_mil_evidence.csv`, taken from the top of its risk ranking: ranks 1, 2, 3,
4 and 30 of 112,540. The hands quoted are the five `evidence_hand_*` cells the
submission actually carries for that pair — nothing is picked after the fact.

Every fact below is read from `hands`, `seats` and `actions` by
[`verify/case_facts.py`](verify/case_facts.py), and can be reproduced with

    python verify/case_facts.py --submission submissions/submitted/tarik_v55_mil_evidence.csv \
        --pairs P9B9FA14D2D0E,P8E67334B0E04,PD91D4EB58627,P074E4EC05254,P2AD7AE0B444B

Two conventions matter. **Hand strength is stated as it was at the moment of the
action**, not on the final board: a player who folds on the flop is judged on the
flop. And **the comparison is against exposure, not against raw counts.** Tables are
closed, so a player meets the same opponents throughout; "folded to the partner 16
times against 161 times to everyone else" means nothing until each count is divided
by the hands in which that opponent was dealt in. The lifts quoted are
per-shared-hand rates against per-(hand × opponent) rates with every other opponent.

Counts are small — five to seven events per pair. They are evidence for review, not
proof, which is why each case also states what would explain it innocently.

---

## Case 1 — `P9B9FA14D2D0E`, directed transfer (risk rank 1)

**Players:** A `U56859D4F52EE`, B `UC1304BDF14CE`. 124 shared evaluation hands.
Over those hands B is down **1,052.7 BB** and A is up **386.4 BB**.

**Evidence hands submitted:** `H5010EAD0C11928`, `HBF7975E53EFF09`,
`H158DFD15D40D70`, `HC69BD66A8BFE0B`, `H5D3B4ECE127639`.

**Observable behaviour.** In four of the five, B calls A's preflop raise, and then
folds to A's single continuation bet while holding the better hand *at that moment*:

| hand | flop | B held | A held | B folded to |
|---|---|---|---|---|
| `H5010EAD0C11928` | `9s 9h 2h` | `2c 6s` — two pair | `7s As` — pair | 7.9 BB |
| `HC69BD66A8BFE0B` | `Qd 3d Qc` | `Qs 2d` — trips | `9c 8h` — pair | 5.5 BB |
| `HBF7975E53EFF09` | `Jc 5c Ac` | `3s Js` — pair | `Kh 6s` — high card | 16.0 BB |
| `H158DFD15D40D70` | `6d 9s 5d` | `5s 2c` — pair | `Ah Td` — high card | 9.4 BB |

The fifth, `H5D3B4ECE127639`, is the same shape one street earlier: B limps, A raises
to 2.5 BB, B folds. Across all 124 shared hands B gives up the better hand to A **7
times (0.057 per shared hand)** against **12 times in 1,606 opponent-hand slots
(0.0075)** with everyone else — a **7.6×** lift. It is one-directional: A never does
it to B (0 of 10 folds).

**Plausible benign explanation.** B may simply be a weak, fit-or-fold player who
over-folds to continuation bets, and A an aggressive regular who bets every flop.
Folding second pair or a low two pair on a paired board to a half-pot bet is an
ordinary mistake, not a signal; the concentration on one opponent could follow from
seat order — if B is usually in the blinds against A's button opens, B's fold rate to
A will be high for positional reasons alone.

---

## Case 2 — `P8E67334B0E04`, directed transfer (risk rank 2)

**Players:** A `U56785476817A`, B `UF5BDAA08411D`. 121 shared evaluation hands.
B is down **569.5 BB** over them, A up **374.5 BB**.

**Evidence hands submitted:** `HC0F810438324E0`, `H9F305E58AD5929`,
`HC6F0775882C4AA`, `H945017B5E5C347`, `H80BC63567B4D3C`.

**Observable behaviour.** The first four are heads-up pots B enters by calling A's
2–2.5 BB raise, and then abandons to one bet:

| hand | board when B folded | B held | A held | B folded to |
|---|---|---|---|---|
| `HC0F810438324E0` | `2d Jh Jd 9s` | `4s Js` — trips | `Qh 7s` — pair | 8.0 BB |
| `H9F305E58AD5929` | `3c Qs Qd` | `Qh Tc` — trips | `8h Th` — pair | 8.0 BB |
| `HC6F0775882C4AA` | `7d 5d As 6h` | `6c 5c` — two pair | `Qh Ts` — high card | 5.5 BB |
| `H945017B5E5C347` | `Tc 6s Qh` | `6d Kc` — pair | `Kh As` — high card | 5.5 BB |

Folding trips on a paired board to a small bet, twice, is the part that does not have
an ordinary reading. The fifth hand runs the other way and is entirely preflop: in `H80BC63567B4D3C` B
calls A's raise to 8 BB with `7d 6h`, an outsider re-raises to 24.5 BB, A re-raises
to 72 BB — and B puts his last 39 BB in against that, into A's `6c Kh`. B loses the
48 BB, A wins 75.5. Over the 121 shared hands B gives up the better
hand to A **5 times (0.041 per hand)** against **8 times in 2,454 opponent-hand slots
(0.0033)** elsewhere — a **12.7×** lift, again with nothing in the reverse direction.

**Plausible benign explanation.** A player who reads one specific opponent as
"only bets when strong" will fold good hands to that opponent cheaply and correctly
by their own lights; the all-in with `7d 6h` is consistent with tilt right after the
folds rather than with a plan. Two regulars who meet 121 times will also produce runs
like this by chance, and B's losses could be concentrated on A simply because A is
the most aggressive player at that table.

---

## Case 3 — `PD91D4EB58627`, directed transfer (risk rank 3)

**Players:** A `UAE0BA493C322`, B `UE7644B41C536`. 156 shared evaluation hands.
A is down **1,712.2 BB**, B up **1,845.0 BB** — the largest one-way flow of the five.

**Evidence hands submitted:** `HCF915E3013CFE9`, `H95B832C81665BD`,
`H31ECA94B2242AB`, `HE4B89B6071831C`, `H266DEFC2896AE2`.

**Observable behaviour.** Here the concessions are large and late, and they run from
A to B.

* `HCF915E3013CFE9` — board `Th 9c 5s 3s Ac`. A holds `2d 4s` and makes a wheel
  straight on the river. A has already put in 38 BB, calling 25.5 BB on the turn. B
  moves all in for 24 BB and **A folds the straight** to B's one pair of aces.
* `H31ECA94B2242AB` — flop `3h 9s 7d`. A holds `Jd 9d`, top pair; B holds `Ah Ks`,
  no pair. A folds to B's 32.5 BB bet.
* `HE4B89B6071831C` and `H266DEFC2896AE2` — A calls off 81.5 BB and 99 BB to
  showdown holding second-best hands (`8d 5d` against `2d Ad` two pair; `9s Td`
  against `Kc Kd`), 180 BB of the total flow in two hands.
* `H95B832C81665BD` — A folds `As Js` preflop to B's 1.5 BB raise.

Across the 156 shared hands A gives up the better hand to B **5 times (0.032 per
hand)** against **3 times in 1,874 opponent-hand slots (0.0016)** with everyone else:
a **20×** lift, the highest of the five cases. B never does it to A.

**Plausible benign explanation.** Folding the river to an all-in with a four-card
straight on an ace-high board is a bad fold, not necessarily a bought one — the wheel
is easy to miss when holding `2d 4s`, and it is a known blind spot. Paying off with
second-best hands twice is ordinary losing poker, and A loses money to the rest of
the table too; that A's losses are largest against the strongest opponent at the
table is what one would expect without any arrangement.

---

## Case 4 — `P074E4EC05254`, soft play (risk rank 4)

**Players:** A `U93A206CA0A4E`, B `UC00B6D37BF1A`. 236 shared evaluation hands.

**Evidence hands submitted:** `H602C1D05287A46`, `H8A628461582FBA`,
`H9ABC3071382233`, `HAFEDB7077CCF85`, `H58225841825F8F`.

**Observable behaviour.** Soft play does not show up as folding the better hand but
as declining to bet it. The clearest hand is `H58225841825F8F`: A raises to 3 BB
preflop, B calls, everyone else folds, and the board runs `Kd 9d Ad 6h 7h`. **A holds
`Td Jd` — a flopped flush — and B holds `6s 6c`, improving to trips on the turn.
Both players check the flop, the turn and the river.** A heads-up pot with a flush
against trips ends at 7.5 BB.

`H9ABC3071382233` is the same reluctance in a multiway pot: B makes a straight on the
turn with `Td 6s` and only calls A's 6.5 BB bet, then checks the river. In the other
three, one of them folds the better hand to a small bet from the other
(`HAFEDB7077CCF85`: B folds a pair to A's 5.5 BB with A holding high card;
`H602C1D05287A46` and `H8A628461582FBA`: equal-strength folds on the flop).

Quantitatively: of the 15 hands in which both reach showdown together, **5 (33%) see
no bet or raise from either of them after the flop**, against 6 of 39 (15%) and 7 of
53 (13%) for the same two players with every other opponent — a 2.2× and 2.5× lift.
B also gives up the better hand to A at 2.7× the rate it does to others.

**Plausible benign explanation.** A flopped flush checked down is the most suspicious
single observation here, but slow-playing a flush on a two-tone board is a normal, if
unprofitable, line, and the turn and river both bring cards that can scare a player
into checking back. More generally, two cautious players who often end up heads-up
will produce many check-downs without any agreement, and this pair's lifts (2.2–2.7×)
are far smaller than the directed-transfer cases above. Of the five, this is the case
whose innocent reading is strongest.

---

## Case 5 — `P2AD7AE0B444B`, coordinated isolation (risk rank 30)

**Players:** A `U09C53566E337`, B `UA8E04240D99E`. 97 shared evaluation hands.

**Evidence hands submitted:** `H380F8AB26CCC71`, `H58DE3944B0B659`,
`H4AB3F079B361AE`, `H5BF6A5346F3AC4`, `H258D8F013538CC`.

**Observable behaviour.** The pattern is preflop and it is about the other players at
the table, not about the chips moving between these two.

* `H380F8AB26CCC71` — an outsider opens to 2.2 BB, **B re-raises to 6.6 BB with
  `4h 3c`, and A re-raises to 19.8 BB with `8c Th`**. Two players of the same pair
  raising in the same round with those holdings is the motif; a fourth player then
  shoves 48.2 BB and both fold, losing 26.4 BB between them.
* `H4AB3F079B361AE` — A opens to 2.5 BB, the table folds, **B raises to 6 BB and A
  folds immediately.** B takes the pot uncontested.
* `H258D8F013538CC` — B opens to 2.5 BB, one caller, **A raises to 7.5 BB with
  `Qc Qs`; everyone folds, including B.**
* `H5BF6A5346F3AC4` and `H58DE3944B0B659` — A opens, B folds at once, A plays the
  pot against the rest of the table.

Over the 97 shared hands the two are **both aggressive in the same preflop round 15
times (0.155 per shared hand)**, against 0.035 and 0.016 per opponent-hand slot with
everyone else — lifts of **4.5×** and **9.5×**. Each also folds to the other about
twice as often per hand as to any other opponent (0.216 vs 0.082; 0.186 vs 0.087),
and almost always preflop: they raise around the table but rarely play a pot out
against each other.

**Plausible benign explanation.** This is the family with the most innocent
look-alike. Two loose-aggressive regulars three-bet and four-bet light as a matter of
style; when both sit at the same table their raises will collide often and will
squeeze third parties out with no arrangement at all. In `H258D8F013538CC` A's
re-raise is made with queens, which needs no other explanation. And the pair's own
result argues against a scheme: over these 97 hands A is up only 228.4 BB and B is
down 13.9 BB, and in the one large pot they built together they lost 26.4 BB to an
outsider. Mutual avoidance after a raise is also what two players who respect each
other's ranges would do.

---

### What these five have in common, and what they do not

The three directed-transfer cases share one observable: a player folds a hand that is
ahead at the moment of the fold, to the same opponent, far more often than to anyone
else, while chips move one way in aggregate. The soft-play case shows the mirror
image — strong hands that never get bet — and the isolation case shows neither, only
a preflop raising pattern aimed at the rest of the table.

None of the five is decisive on its own. What the pipeline ranks on is the
repetition: the same event, in the same direction, against the same opponent, at a
rate several times that player's own rate against everybody else.
