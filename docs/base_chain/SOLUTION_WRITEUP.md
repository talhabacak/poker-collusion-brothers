# Detect Suspicious Value Transfers in Poker — Solution Writeup (draft, ≤1500 words)

*Numbers below describe the selected submission (v33/v34 chain).*

## Summary

The pipeline treats the synthetic generator as a set of player policies and looks for **policy deviations that are specific to one partner**. Three ideas carry the result:

1. **Population policy surprisal.** A LightGBM action model `p(action | state)` is trained on all 18.6M actions with two folds by table. Every action is scored by `-log p(action)`. Only actions taken **while the partner is still in the hand** are aggregated. This single condition removes the intended traps (tilt, weak play, similar styles): those deviations show up against everyone, collusion only against the partner.
2. **Relational likelihood ratio.** For each ordered pair (a → b), a's action distribution in a context (street × own strength × strength vs. b × who a is facing × heads-up) is compared with a's own distribution in the same context against all other opponents. Repeated partner-specific behaviour scores high even when each hand looks ordinary.
3. **Value transfer measured directly (collusion table).** Following Mazrooei, Archibald and Bowling (AAAI 2013), a learned value function `V_j(h)` gives each player's expected outcome at a decision point, and every action by `k` is credited with `V_j(h·a) − V_j(h)` — how much it moved the *other* player's expectation. Summed over a pair's shared hands this is a direct estimate of who benefits from whom. It is a strong pair-level signal (PU pair AP 0.708 → 0.776) but a harmful hand-level one (evidence MAP@5 0.684 → 0.679), so it is used **only** in the pair model; the whole evidence chain is trained without it.
4. **Evidence as a two-stage, chronology-aware ranking.** Planted evidence hands cluster early in the pair's timeline. A within-pair ranker (features normalised inside each pair) proposes candidates; a LambdaRank reranker with chronological-position features and an **action-level collusion model** picks the final five.

All features derive from gameplay only: cards, actions, pot, stack and hand order within a table. No identifiers, row order or generator internals are used.

## Data preparation (`src/01_prep.py`)

- Preflop equity of every hole-card class from Monte Carlo (4,000 samples/class, 169 classes).
- Post-flop made-hand strength for every seat on every reached street (treys evaluator; 12.8M evaluations).
- Decision-point context per action: pot odds, stack-to-pot ratio, players active, aggression counts, whether the actor was last aggressor, player style priors (open-raise, fold-to-raise, post-flop aggression rates).

## Population policy (`src/02_policy.py`)

Six-class LightGBM (fold/check/call/bet/raise/all-in), 4M training actions per fold, early stopping; out-of-fold log-loss 0.50. A second regressor models bet size; its standardised residual is a feature. A sharper policy (log-loss 0.41 with draw and betting-line features) was tested and was neutral: a better population model is also less surprised by collusive actions.

## Pair-hand features (`src/03*.py`)

For every pair of seated players in every hand (30M rows):
- surprisal sums split by action type, street, "facing partner", heads-up; hand strength when folding to the partner; chips put in while behind; dumped value; passive play while ahead;
- third-player folds to the pair's aggression, dead money collected, multiway aggression (partner-independent, important for `coordinated_isolation`);
- relational LLR sums per direction;
- within-pair z-scores, rolling means (episode density) and chronological position.

Winner/loser roles are assigned per hand so directional families (`directed_transfer`) have consistent feature semantics.

## Models

**Hand scorer (`04`)**: 4-class LightGBM (not evidence / DT / SP / CI) on labelled pairs' hands, five folds by table. Out-of-fold everywhere: development rows are only ever scored by the fold that never saw their table. An early in-sample leak inflated evidence MAP@5 from 0.60 to 0.95 offline and cost 0.014 on the leaderboard; that rule is now enforced in every stage.

**Within-pair ranker (`04b`)**: trained only inside public positive pairs, per family, with features z-scored and rank-normalised inside each pair. It answers "which hand of *this* pair" rather than "which pair".

**Action-level collusion model (`04c`)**: each action of a pair member in a candidate hand is a row (own state, partner state, facing role, contributions, hand outcome). Label: the hand is a planted evidence hand of that pair. Per-family LightGBM; hand aggregates (max, sum, per-street max) feed the ranker. This was the single largest evidence gain in the final week (+0.027 MAP@5 offline).

**Collusion table (`09`)**: a LightGBM value function predicts each seated player's net result from the state at every decision point (RMSE ≈ 21 bb). Per hand and pair it yields `help_p_by_q`, `help_q_by_p`, and their mutual, asymmetric, max and positive-part summaries. These enter the pair model only.

**Pair model (`05`)**: LightGBM on 40 pair-level aggregates (top-k of hand scores, surprisal, relational G, player-margin normalisation: how much this pair stands out among each player's other pairs). Trained on the 1,860 labelled pairs only; a single seed, because rank-averaging several seeds measurably hurt with so few labelled pairs (PU pair AP 0.776 → 0.764). Behaviour: 3-class LightGBM on public positives (accuracy 0.979 OOF).

**Fourth family (`08`)**: the public data names three families but the evaluation set contains a fourth, `other_coordination`. Its size was estimated by simulation: a hidden ~10% family drops behaviour MAP from 0.970 to 0.862, matching the 0.883 observed on the leaderboard. A leave-one-family-out study — one family removed from the hand scorer's training entirely — showed an unseen mechanism is still separable at AUC 0.966–0.981: the pair model ranks it high while the evidence model finds its hands weak. That rule, calibrated on the LOFO cost curve, relabels the most anomalous high-risk pairs as `other_coordination`.

**Evidence reranker (`05b`)**: LambdaRank over each pair's top-50 candidates, six seeds rank-averaged, with chronology (position in the pair's timeline, mass of suspicion before/after, rank among candidates) and all stage scores.

## Validation

- Folds are by table (pool), so no pair or player crosses folds.
- Local metric = the official metric reimplemented (`reference/official_metric.py` is the organiser's code).
- Pair AP on labelled pairs saturates (0.998); a second benchmark, positives vs. "public colluder + innocent opponent" pairs (18k, excluded from evaluation), was used for feature decisions.
- Leaderboard decomposition by diagnostic submissions (evidence removed / behaviour removed) gave Pair AP 0.953, Evidence 0.587, Behaviour 0.883 for an early version; later submissions moved evidence to 0.645 and above.
- Candidate-set audit: the top-50 candidate list contains 99.9% of true evidence hands, so the remaining evidence loss is ranking within roughly fifteen hands, not recall.

## What did not work (and was reverted)

- Seed rank-averaging in the pair model (neutral on the leaderboard, negative offline on the final chain).
- The collusion table inside the evidence chain (helps pairs, hurts hands — hence the split).
- Player metadata (region, client, account age): the generator does not correlate it with collusion; OOF AUC 0.504.
- Any use of unlabelled pairs in pair-model training: pseudo-labelling (−0.015 LB), colluder-plus-innocent pairs as hard negatives (−0.007 LB), although all three offline metrics improved.
- Exclusivity post-processing (one partner per player), episode scan statistics, per-family rerankers, chronological ordering of the final five, information-sharing detector (partner's cards in the policy), big-blind attribution for unopened-pot folds, action-sequence n-grams.

## Reproduction

`src/README.md` lists the run order (`01` → `02` → `03`, `03b`, `03c` → `09` → `04` → `04c` → `04b` → `05` → `05b` → `08` → `06`); `run_split.sh` reproduces the final chain end to end, with `USE_CT=0` everywhere except the pair model. Seeds fixed (42), `requirements.txt` pinned, ~45 minutes on 16 cores / 32 GB. `tools/submit_via_mcp.py` submits through Kaggle's MCP endpoint.

## Responsible interpretation

Scores rank synthetic patterns; they do not establish intent. Evidence hands are meant to support a human reviewer, and each case review below lists a plausible innocent explanation.
