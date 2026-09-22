# What we measured wrong, and what that cost

Kept in the reproduction repository on purpose. Three of these changed how the
project was run, and one of them cost the team places on the board.

## 1. The development metric of one engine did not transfer — and we generalised from it anyway

Engine A's development evidence metric was measured as **0.054 MAP optimistic**
against the board. That is a fact about an engine, not about a mechanism. It was
nevertheless used to retire a mechanism:

> the board pays nothing for the family-specialist stack, and that stack is our
> entire development lead (+0.0469)

Both readings behind that sentence came from engine A — its whole evidence column
spliced into v55 scored 0.91233 against v55's 0.91748, and its family-agnostic pooled
arm read about the same as four board probes of its specialist stack. What they
establish is that *engine A's evidence engine* does not transfer. Whether routing to
family specialists transfers, measured in an engine whose development number does
transfer, was never tested.

The correct rule had already been written down, before the mistake:

> a +0.008 development gain is worth ~0.0016 of board score in engine A. In engine B,
> where development transfers one for one, the same gain would be readable. The
> method should be tried where the metric works.

It was applied to the MIL action head, which was rebuilt in engine B's tables and
folds, cleared its placebo comparison, and became the primary final. It was not
applied to the routed specialists, which were the larger component.

**What that cost.** Rebuilt inside engine B's reranker after the deadline — pooled
arm 0.10 plus three family specialists 0.90, routed by the behaviour head's
calibrated probabilities, lambdarank truncation 50, twelve seeds — the same
architecture scored **0.92121** public against v55's 0.91748: **+0.0037**, and
+0.0029 over the file the team actually selected. Development evidence CV moved
0.6631 → 0.6812 across three independent six-seed sets. The submission was late and
counts for nothing.

The rule that replaces the claim: before recording a mechanism as dead, name the
engine the reading came from. If that engine's development metric has been shown not
to transfer, the honest statement is *not yet measured where it counts*, and the next
action is the rebuild, not the refusal.

## 2. Fold maps were not shared between stages

Each stage shuffled whatever list of tables it had been handed, so a stage that saw
only labelled tables placed 339 of 397 of them in a different fold from a stage that
saw all 400 — and supervised caches were then validated on folds their own training
rows belonged to. One canonical map, keyed by fold count and seed
(`engine_a/src/pokercol/cv.py::canonical_fold_map`), removed it, and removed 0.011 of
the evidence engine's apparent gain with it.

## 3. Leave-one-family-out scoring dropped the family it held out

The rows it removed are exactly the ones a recovery rule promotes, so any rule that
worked read as free. Three guards were certified that way and all three lost on the
board; with the rows kept as negatives their cost is 0.048 to 0.065. The submitted
pipeline carries no guard.

## 4. The unlabelled-colluder problem

The plain development surrogate — the 372 disclosed targets against every other
candidate pair — read about 0.67 while the leaderboard implied a Pair AP near 0.94.
Out of fold, roughly 150 unlabelled pairs score as near-certain colluders. Dropping
them from the negatives moves the surrogate to 0.95, in line with the board. That set
is frozen on disk (`artifacts/surrogate_exclusion.parquet`) so every experiment is
scored on identical rows.

The obvious next step — promote them to positives — was tried and rejected on
measurement: it improved the surrogate by 0.007 and lost on the board. The opposite
extreme, weight 0 for unlabelled pairs, raises an unseen family's leave-one-family-out
AP from 0.17 to 0.87 but costs 0.0157 of cleaned AP on the families we can see, with
a bootstrap interval clear of zero, and the board agreed.

## 5. What the public leaderboard can and cannot settle

Resampling development at the public split's rate puts the standard error of the
evidence term at 0.0106 of MAP@5 — 0.0021 of score — and the pair term at
0.0026–0.0075 of score. A real difference of 0.0032 in the pair term reads with the
wrong sign in a third of slices. Differences of that size are therefore decided
offline or not at all; several ideas retired on small public readings had to be
re-examined against that standard rather than against the readings.

## 6. Things that were measured and are not in the submission

Player-level colluder detection (collusion is purely relational; AP 0.13 at a 5.8%
base rate). Cross-phase history — development targets look like non-targets in the
evaluation phase, so episodes are confined to one phase. Graph propagation over
shared players. Pseudo-labelling. A coalition Bayes factor, whose whole surface sat
under 0.52 in both engines. Sparse-mixture extremes (Higher Criticism, Berk–Jones),
placebo-indistinguishable. A learned bi-GRU sequence residual over every action,
which moved within-pair AUC from 0.9041 to 0.9042 and read exactly zero on holdout.
Engine A's behaviour head, which beat engine B's by +0.030 MAP on development and
lost 0.0008 on the board — the single-column probe that settled it is
`tarik_v55_mil_evidence_behA`.
