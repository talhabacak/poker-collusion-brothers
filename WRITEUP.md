# Two engines, one column: how the Brothers submission was built

Team **Brothers** — private 0.92237 (11th of 371 teams), public 0.91834 (14th).
Code (MIT): [https://github.com/talhabacak/poker-collusion-brothers](https://github.com/talhabacak/poker-collusion-brothers). Five case reviews: [`CASE_REVIEWS.md`](https://github.com/talhabacak/poker-collusion-brothers/blob/main/CASE_REVIEWS.md).

## The problem as we framed it

Rank 112,540 evaluation pairs by collusion risk (70%), retrieve the five hands that
evidence it (20%), name the behaviour family (10%). A colluding pair shares roughly
120 hands and cheats in about five of them, so any pair-level average is mostly
noise. The signal lives in a few implausible hands, and in how one player treats one
specific opponent differently from everyone else at the table. Everything we use is
derived from `hands`, `seats` and `actions`. Identifiers, row order, file order and
player metadata are never read.

The team is two people who worked **independently until the last days** — no shared
code, no shared features, no shared diagnostics — so that neither of us collapsed the
other's reading of the problem into his own. Two complete pipelines came out of that,
engine A and engine B, and they merged on 19 September within 0.0013 of each other on
the board. The submitted file is engine B's risk and behaviour columns with an
evidence column engine A's method produced inside engine B's tables and folds. Both
are in the repository, as one pipeline.

## Engine B — the pair, behaviour and evidence columns

**Policy surprisal.** A LightGBM model of P(action | state) is fitted on all 18.6M
actions, two folds by table, and every action is scored by −log P(observed),
out of fold. The state carries street, pot odds, stack-to-pot, position, made-hand
strength, preflop equity, the betting line and the actor's own style rates, so
"surprising" means unusual *for this player*, not for the population.

**Relational likelihood ratio.** For every action taken while a specific opponent is
still live, the player's action distribution against that opponent is compared with
their own distribution in the same context against everyone else, with the current
action left out. Only repeated, partner-specific behaviour scores.

**Opponent-aware and opponent-conditioned residuals.** The relational statistic uses
the player's own baseline. These ask a different question: did the player depart from
what the population policy expected *in those spots*, normalised by the model's own
variance — and then, with the expectation itself conditioned on who the opponent is
and how that opponent plays, what is left after removing "this opponent is simply not
like the others". This block came from engine A and was the largest late gain of the
project: 0.90858 → 0.91748 on the board when engine B ported it, against +0.0087 and
+0.0100 when engine A measured the same mechanism in its own stack. Two pipelines
sharing no code priced the same mechanism the same way.

**Hand and action scorers.** A four-class hand scorer (non-evidence / directed
transfer / soft play / coordinated isolation) is fitted five-fold by table; a
within-pair ranker is trained only inside positive pairs on pair-normalised features,
so it learns which hand of *this* pair is planted rather than which pair looks
coordinated; an action-level model per family learns the visible action itself.

**Pair, behaviour, evidence.** Risk and behaviour are LightGBM models over the
aggregated pair frame, with unlabelled pairs given zero weight. Evidence is a
lambdarank reranker grouped by pair, six seeds, over the top 50 candidates, carrying
within-pair chronology.

## Engine A's contribution to the shipped file: an action-level MIL head

Hands carry the label, but only one or two actions inside a hand are the transfer.
The head treats each hand as a bag of its actions, scores every action, and pools
with noisy-OR / log-sum-exp so the bag is positive when any action is. Bags whose
label cannot be determined are censored out of the loss rather than called negative.

The first version was measured in engine A and read below our +0.010 bar. Engine A's
development evidence metric had been shown to be 0.054 MAP optimistic, so that
reading settled nothing about the mechanism. Rebuilt in engine B's tables and engine
B's folds, with the head nested so no row ever reads a column from a head that saw
its own fold, and with fold 4 held out of every fit and every choice:

    real minus column-matched placebo   +0.00729  [+0.00260, +0.01250], p 0.9995
    inner folds                         +0.00551  [+0.00100, +0.01016]
    strict holdout (fold 4)             +0.00999  — same sign, larger, no leak signature
    placebo minus baseline              +0.00024  — junk columns do not inflate here

It still missed the +0.010 bar and was shipped anyway, as an explicit decision, with
the expected board effect written down first: 0.20 × 0.0073 ≈ +0.0015, against an
evidence-side public standard error of 0.0021, so the public number could not confirm
or refute it. It scored 0.91834 against v55's 0.91748.

## What the measurements taught us

**Unlabelled is not negative.** About 150 development pairs score out of fold as
near-certain colluders while carrying no label. The plain surrogate counts them as
false positives and reads 0.67 where the leaderboard implies 0.94. We froze that set
and score every experiment on identical rows. We never trained against it: a model
fitted to push those pairs down would be fitted to push undisclosed positives down,
and the evaluation set holds the same population.

**Two defects that had been shaping decisions.** Each stage shuffled its own fold
map, so supervised caches were validated on folds their training rows belonged to;
one canonical map removed that, and 0.011 of an apparent evidence gain with it.
Leave-one-family-out scoring dropped the held-out family's rows — exactly the rows a
recovery rule promotes — so three guards certified that way all lost on the board.

**The board cannot settle small differences.** The public split resolves the pair
term to 0.0026–0.0075 and the evidence term to about 0.0021 of score. Differences
smaller than that are decided offline or not at all. With 57 scored submissions, the
maximum of many noisy draws also carries a winner's curse of roughly +0.007 to +0.021
— which is why the file we made primary was the one whose predicted range was
written down before it was sent, not the one with the best public number.

**Name the engine a negative came from.** Family-routed evidence specialists were
recorded as worthless because they paid nothing in engine A — the engine whose
development metric does not transfer. Rebuilt in engine B's reranker after the
deadline, the same architecture scored 0.92121: +0.0037 over v55 and above the team's
best in-competition file. The mechanism was never the problem. This is the one thing
we would do differently, and it is written up in [`docs/MEASUREMENT_NOTES.md`](https://github.com/talhabacak/poker-collusion-brothers/blob/main/docs/MEASUREMENT_NOTES.md) rather than left out.

## Choosing the two finals

Every file below differs from its neighbours in named columns only — risk (R),
behaviour (B) and evidence (E), each either engine A's, engine B's, or B's with the
MIL head:

| file | composition | public |
|---|---|---|
| v55 | R_B + B_B + E_B | 0.91748 |
| **tarik_v55_mil_evidence** | R_B + B_B + E_MIL | **0.91834** (primary) |
| x_v55_ourEv | R_B + B_B + E_A | 0.91233 |
| tarik_v55_mil_evidence_behA | R_B + B_A + E_MIL | 0.91756 |
| **hybrid_riskA_behB_evMIL** | R_A + B_B + E_MIL | **0.91112** (second) |
| v64 | R_A + B_A + E_A | 0.90543 |

Read as single-column differences: E_A − E_B = −0.00515, B_A − B_B = −0.00078,
R_A − R_B = −0.00722. Engine B's column is the better one on every term, although
engine A's risk leads on development — an inversion its own diagnostics had warned
about.

The second final had to differ where diversification is wanted, the pair ranking, and
nowhere else. The hybrid is v64's risk column under the primary's behaviour and
evidence: Spearman 0.505 against the primary, 679 of the top 1000 shared. It was
predicted from the probes before it was sent — 0.90543 + 0.00601 + 0.00078 = 0.9122,
range 0.909–0.915 — with the selection rule locked in advance. It scored 0.91112,
inside the range, and by that rule became the second final.

## Reproducing it

`python run_all.py --list` prints every stage and whether its output exists;
`--dry-run` prints the plan. The default run rebuilds v64's risk column, the MIL
evidence column, both selected files and their checksums; `--with-base` rebuilds
v55 first. Setup, hardware, runtimes and what is and is not rebuilt here are in the
[README](https://github.com/talhabacak/poker-collusion-brothers#setup).

The two chains, what each reached alone, and every file submitted after they
merged: [`docs/TWO_CHAINS.md`](https://github.com/talhabacak/poker-collusion-brothers/blob/main/docs/TWO_CHAINS.md).
