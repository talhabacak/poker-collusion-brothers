# Experiment log — base chain

*English translation of the base chain's own log. The original is `EXPERIMENTS.md` at
commit `edddbb5` of the competition repository, written in Turkish by its author.
Experiment ids, numbers, file names, metric readings and decisions are unchanged; only
the prose is translated. Decisions read **KEPT** (in the pipeline), **REJECTED** (hurt
or neutral) and **PENDING** (not submitted).*

Score formula: `0.70·PairAP + 0.20·EvidenceMAP@5 + 0.10·BehaviorMAP`
Measurement base (v2, 15 Sep): Pair **0.953**, Evidence **0.587**, Behavior **0.883**.
**Current reading (v20 diagnosis, 17 Sep):** 0.7·Pair + 0.1·Behavior = **0.77367**,
Evidence = **0.6451**. Assuming Behavior 0.883, Pair ≈ **0.979**. Remaining weighted
headroom: evidence 0.071, pair 0.015, behaviour 0.012.
**The evidence CV tracks the leaderboard** (CV 0.6485 → LB 0.6451), so evidence
decisions can be taken offline — but only for methods that obey Rule 8. On the pair
side there is no usable offline metric (Rule 9).

**Best submission: v41 = 0.90731** (18 Sep). Chain: no MIL, no collusion table,
fourth-family rule on 432 pairs.
**Before submitting:** write the prediction with `scratch/eda/44_predict_lb.py`, then
look at the residual (Rule 11).

---

## 1. Summary table

| # | Experiment | Local reading | Public LB | Δ | Decision |
|---|---|---|---|---|---|
| 1 | Population policy surprisal + pair model (v1) | pair AP 0.965, evidence 0.215 | 0.85842 | — | KEPT |
| 2 | Chronological evidence reranker (v2) | evidence CV 0.470 → 0.566 | 0.87254 | +0.0141 | KEPT |
| 3 | Metric decomposition diagnostics (2 submissions) | — | 0.75510 / 0.78429 | — | information |
| 4 | Relational LLR features (v3) | pair AP 0.9962→0.9975, beh 0.960→0.976 | 0.89100 | +0.0185 | KEPT |
| 5 | Within-pair stage-1b + 50 candidates (v4) | evidence CV 0.564 → 0.584 | 0.89270 | +0.0017 | KEPT |
| 6 | PU self-training + table features + 3 seeds (v5) | locally neutral | 0.87798 | −0.0147 | REJECTED |
| 7 | Truncated development window (TRUNC 0.67) | beh 0.978→0.954, family accuracy 0.987→0.970 | — | — | REJECTED |
| 8 | Per-family stage-1b ranker (v7) | evidence CV 0.584 → 0.611 | **0.89779** | +0.0051 | KEPT |
| 9 | stage-1b pair features + leaking reranker (v8) | leaked CV 0.95 | 0.88417 | −0.0136 | REJECTED |
| 10 | Mixed pairs as hard negatives (w=0.2) (v9) | all three local metrics rose | 0.89095 | −0.0068 | REJECTED |
| 11 | Information-sharing detector (src/07) | AUC 0.58, no excess in evaluation | — | — | REJECTED |
| 12 | Isolation features (src/03c) + 3-seed reranker (v10) | evidence CV 0.604 → 0.620 | — | — | PENDING |
| 13 | Per-family reranker | honest CV 0.618 → 0.620 (neutral) | — | — | REJECTED |
| 14 | Episode-scan statistics | metric 0.933 → 0.937 (neutral) | — | — | REJECTED |
| 15 | Uniqueness post-processing (one partner per player) | mixed metric +0.055, broad metric −0.060 | — | — | REJECTED |
| 16 | Ordering the five evidence hands chronologically | MAP@5 0.618 → 0.562 | — | — | REJECTED |
| 17 | `other_coordination` for ambiguous families (v12) | ambiguity rate dev 2.4% vs eval 14.7% | — | — | PENDING (reserve) |
| 18 | Fold attribution: count the BB as aggressor when the aggressor is null (pipeline B's idea) | 43.6% of folds have no aggressor, measurement pending | — | — | PENDING |
| 19 | Pipeline B's conservative `other` rule (P<0.95 and family hand score<0.2) | not selective here: 44/372 of dev positives fall below the threshold | — | — | REJECTED |
| 20 | Isolation-specific token model, P(iso) weighted (+0.012 OOF in pipeline B) | — | — | — | PENDING |
| 21 | Policy model v2: draws, betting line (src/02_policy_v2, v15) | log-loss 0.50→0.414 but evidence reranker 0.647→0.645, pair AP 0.9978→0.9953, behaviour 0.968→0.973 | — | — | REJECTED (neutral) |
| 22 | **Action-level collusion model** (src/04c): member actions inside evidence hands as positives, per family, 5 folds | alone: hand AUC 0.960, MAP@5 0.444; as a feature: stage-1b 0.517→0.535, reranker **0.620→0.647** | — | — | PENDING (v13, 17 Sep slot 1) |
| 24 | Time-local relational baseline (src/03b_relational_v2, TBIN=400, v16) | reranker 0.6454→0.6455, stage-1b 0.535→0.527, pair AP 0.9953→0.9958 | — | — | REJECTED (neutral) |
| 25 | Isolation action-sequence token model (src/04d, v17) | alone MAP@5 0.25 (iso AUC 0.966); reranker 0.6455→0.6468 | — | — | REJECTED (neutral) |
| 26 | Showdown board texture / loser hand strength | AUC 0.56, 0.50 (noise) | — | — | REJECTED |
| 23 | BB fold attribution (src/02b): in an unopened pot the aggressor is the big blind; whole chain rerun (v14) | stage-1 0.488→0.468, reranker 0.647→0.645, Behavior 0.968→0.956 | — | — | REJECTED |

---

## 2. Details

### 1. Policy surprisal (the core idea, KEPT)
Population action model (LightGBM, 2 folds by table, 18.6M actions, log-loss 0.50).
Every action's surprisal is `-log p(action|state)`. It is aggregated **only while the
partner is live in the pot**. With the partner not live there is no signal (hand-level
AUC 0.21 vs 0.92), which removes the tilt and weak-play traps by construction.

### 2. Chronological reranker (KEPT)
The evidence list is skewed towards a pair's first manipulated hands: high-scoring hands
outside the evidence window are 314 after / 33 before. Inside the candidate pool,
chronological position is the second strongest feature. A reranker without time reads
0.480, with time 0.566.

### 3. Metric decomposition (information)
Copies of v2: evidence blanked → 0.75510, families set to `other_coordination` →
0.78429. The components were solved exactly from these. The official weights were later
confirmed from the Evaluation page.

### 4. Relational likelihood ratio (KEPT)
A player's action distribution against the partner, compared with their distribution in
the same context against other opponents. Context: street × own strength × strength
relative to the partner × who is being faced × heads-up (216 contexts). The pair-level
G statistic alone reads AUC 0.987. 47 seconds for the whole dataset.

### 5. Within-pair stage-1b (KEPT)
Train the evidence model only inside positive pairs, on features z-scored and
percentile-normalised within the pair. It asks "which hand of this pair" instead of
"which pair looks suspicious". MAP@5 0.478 → 0.500.

### 6. PU self-training (REJECTED, −0.0147)
Treat unlabelled pairs scoring above 0.9 as positives and retrain for 2 rounds. False
positives went 160 → 247. The local labelled AP did not move, but Pair AP fell about
0.02 on the leaderboard.

### 7. Truncated development window (REJECTED)
Keep the last 67% of each pair's hands, to imitate the evaluation phase's hand counts.
With less data, family accuracy and Behavior MAP both fell; not submitted.

### 8. Per-family stage-1b (KEPT, +0.0051)
A separate ranker for each of the three families. Isolation 0.270 → 0.328,
directed_transfer 0.570 → 0.584, soft_play 0.572 → 0.582.

### 9. The leak incident (REJECTED, −0.0136)
Development hand scores were averaged over all five folds, so each row's own fold was
included. Evidence CV jumped 0.60 → 0.95 and the leaderboard fell. **Rule: development
scores are always out of fold.** After the fix, stage-1b's contribution as a pair
feature dropped from 0.024 to 0.003 — it had been entirely leakage.

### 10. Mixed pairs as hard negatives (REJECTED, −0.0068)
"One colluder + one innocent player" pairs (18,209 of them) as negatives at weight 0.2.
All three local metrics rose (labelled AP 0.9976→0.9977, against mixed 0.925→0.959,
against all unlabelled 0.684→0.691) and the leaderboard fell. **Rule: the pair model is
trained on the 1,860 labelled pairs only.**

### 11. Information-sharing detector (REJECTED)
The log-likelihood gain of a second policy model that sees the partner's hole cards, in
excess of the player's other opponents. AUC 0.58 on labelled pairs, and the count of
extreme-valued pairs in evaluation is lower than in development. The fourth family is
not this mechanism.

### 12. Isolation features (PENDING)
Without the partner-active condition: number of third players folding to the pair's
aggression (AUC 0.865), aggression in multiway pots (0.825), folding to the partner
(0.772). All three are stronger than the pipeline's conditional versions.

### 15. Uniqueness post-processing (REJECTED)
`score × sigmoid(8·(score − the player's best other pair))`. On the mixed-pair metric
0.926 → 0.980, but against all unlabelled pairs 0.676 → 0.616 and labelled AP
0.9975 → 0.991. For 66 of the positives the player has another pair with a higher score.

### 16. Evidence ordering (REJECTED)
Sorting the chosen five hands chronologically lowered MAP@5. The reranker's own score
order is well calibrated: hit rates by position 0.858 / 0.782 / 0.694 / 0.556 / 0.460.

### 17. `other_coordination` (PENDING)
Share of pairs whose top family probability stays under 0.6: 2.4% among development
positives, 14.7% among high-risk evaluation pairs. Median entropy 0.087 vs 0.190. That
points at the undisclosed fourth family being present in evaluation. Applied to 211
pairs.

---

### 18-21. Ideas carried over from pipeline B
The other pipeline's log was shared on 16 Sep. No code and no data were taken; only
measured findings were carried over as ideas. Detail and the rule note: `PLAN.md` §4b.
Two of our rules that they had independently confirmed: unlabelled pairs do not enter
training; adding features transfers, changing the training set does not.

### 22. Action-level model (BEING MEASURED)
The first bet of the aggressive plan. A row is one action by a pair member plus the
partner context (is the partner live, their strength, who is being faced, their
contribution, the outcome of the hand). Label: the hand is that pair's evidence hand.
Training: the family's positive pairs plus the hands of labelled negative pairs. 10k
positive rows. The most important features: surprisal, position, the partner's position,
the partner's contribution, player style. Alone it does not beat the hand-level model
(0.444 vs 0.517) but it is an independent signal; the combined effect is measured in v13.

### The aggressive round, balance sheet (16 Sep, 5 bets, ~6 hours of compute)
| Bet | Evidence reranker CV | Result |
|---|---|---|
| Action-level model (v13) | 0.620 → **0.647** | the only one that held |
| BB fold attribution (v14) | 0.645, behaviour −0.012 | neutral/negative |
| Policy v2, log-loss 0.50→0.41 (v15) | 0.645 | neutral |
| Time-local relational baseline (v16) | 0.6455 | neutral |
| Isolation token (v17) | 0.6468 | neutral |

Conclusion: the evidence ranker has settled on a plateau in the 0.645-0.647 band, and
adding features no longer looks likely to break out of it. The remaining ~35% of misses
probably comes from a component of the generator's evidence-selection rule that we
cannot see.

### 36. Multiple-instance refinement (MIL)
The competition text says there is **one** visible, behaviour-specific action in every
evidence hand. The model was treating all of a hand's actions as positive; most of the
11,050 positive rows are ordinary actions. Keeping the best 2 actions per hand by the
first round's out-of-fold scores and dropping the rest from training (3,634 rows left)
sharpened the model markedly. This moves the label definition closer to the problem's
real structure, and the measurement agrees.

| 37 | Features describing "the definition of the highest-scoring action" after MIL (street, type, who it faces, surprisal, role) | reranker 0.6838 → 0.6866 | — | — | KEPT |

### The evidence component's trajectory (all out-of-fold by table, tracking the LB within ±0.004)
| Version | Evidence CV | Note |
|---|---|---|
| v2 (chronology) | 0.566 | first reranker |
| v7 (per-family stage-1b) | 0.611 | LB 0.89779 |
| v13 (action model) | 0.647 | LB contribution +0.0029 |
| v20 (family = model) | 0.6485 | LB 0.90268 |
| v24 (MIL 4 rounds) | 0.6838 | +0.035 |
| v26 (top-action description) | **0.6866** | to be submitted |

| 38 | Diagnosis of the behaviour gap | few-hands effect 0.981→0.965 (small); a 10% hidden fourth family 0.970→0.862 (matches the 0.883 seen on the LB) | — | — | information |
| 39 | **Honest LOFO novelty detection** (remove the family from the hand scorer's training entirely) | separates the unseen family at AUC 0.966-0.981; catching 70% of it costs 1-3% of the known families | — | — | KEPT (src/08) |

### 38-39. The fourth family: diagnosis and remedy
The reason Behavior MAP is stuck at 0.883 was measured. Two hypotheses were tested:
(a) evaluation pairs share few hands, which breaks the family prediction, and (b) an
undisclosed fourth family lowers the confidence of every family. (a) only moves
0.981 → 0.965; (b) at a 10% share gives 0.970 → 0.862 and matches the observed value.

The remedy was an honest LOFO: one family was removed **entirely** from the hand
scorer's training (no head for that family), and its pairs were then caught as "a
mechanism I do not recognise". Separation AUC 0.966 (isolation) and 0.981 (soft play).
The rule: the pair model ranks them high but their hands score weakly under the evidence
model. `src/08_fourth_family.py` calibrates that rule with the LOFO cost curve.

An important asymmetry: writing `other_coordination` on a pair that is **not** a target
is useful (it removes a false positive from that family's ranking); the damage is only
in missing one of the three real families, and LOFO keeps that at 1-3%.

## 3. Rules derived so far

1. **Unlabelled pairs do not enter training.** Two independent attempts both hurt on the
   leaderboard (−0.015, −0.007), and in both the local metrics had risen.
2. **Development scores must be out of fold.** Otherwise the CV inflates and the LB falls.
3. **Choosing the local metric is critical.** Labelled AP is saturated (0.998) and cannot
   measure a difference. Usable metrics: evidence MAP@5 (OOF) and the "positives vs one
   colluder + one innocent" metric. But the second one misleads under training-set
   changes; it is reliable only for feature changes.
4. **The gains come from the evidence side.** Three of four leaderboard gains came from
   the evidence model (+0.014, +0.002, +0.005), one from relational features (+0.019).
5. **Adding features transfers; changing the training set does not.**

### 26. Showdown board texture / loser strength (REJECTED)
On hands where both members of the pair went to showdown (2,128 hands, 413 evidence), the
losing hand's strength and the gap between the two hands were tested. AUC 0.56 and 0.50 —
noise. Not a new lever.

### Rule 6: the "scored vs not scored" artefact
If a hand or pair score is computed for a subset only and the rest are filled with 0 (or
any constant), a spurious separation appears in pair-level metrics. It happened twice:
stage-1b (0.953 → 0.927 once fixed) and the action model (0.945 against all unlabelled,
because only labelled pairs had been scored). **If a new score is to be used as a pair
feature, it must be computed for every pair being compared.**

| 27 | Action-model aggregates in the pair model (with honest coverage, v20) | mixed 0.9326→0.9420, all unlabelled 0.6760→0.7105 | — | — | KEPT |
| 28 | Exposure / extreme-value normalisation | mixed +0.004, harmful when used together | — | — | REJECTED |
| 29 | Participation synchrony (phi coefficient) | alone AP 0.79, no contribution to the model | — | — | REJECTED |
| 30 | Family source = behaviour model (instead of the rule) | reranker 0.6473 → 0.6485 | — | — | KEPT |
| 31 | v13e7: v7 risk + action-model evidence | evidence CV 0.611 → 0.647 | 0.90068 | +0.0029 | information |
| 32 | **v20: action aggregates in the pair model + family=model evidence** | mixed 0.942, all unlabelled 0.711 | 0.90268 | +0.0049 | KEPT |
| 33 | v20 diagnostic (evidence blanked) | — | 0.77367 | — | information |
| 34 | **v21: v20 + `other_coordination` on 196 ambiguous pairs** | — | **0.90387** | +0.0012 | KEPT |
| 35 | v22: 3-seed rank-averaged pair model | labelled AP unchanged | 0.90282 | +0.0001 | REJECTED (noise) |
| 36 | Multiple-instance (MIL) refinement in the action model: only the 2 highest-scoring actions of an evidence hand are positive | round/action sweep: 2r2a 0.6547, 2r1a 0.6656, 3r1a 0.6676, 4r1a 0.6838; action model alone 0.444→0.524 | 0.90012 (v26) | −0.0026 | **REJECTED — leakage, see experiment 50 / Rule 8** |

| 40 | Player metadata (players.parquet, the organiser allowed it on the forum) | same region AUC 0.513, same client 0.508, age difference 0.500; all together OOF AUC 0.504 | — | — | REJECTED (no signal) |
| 41 | Counterfactual value transfer (src/03d): the difference between the chips the policy would put in and the actual ones, signed towards the partner | evidence 0.6866→0.6811, pair PU 0.7355→0.7143, Behavior 0.957→0.9755 | — | — | REJECTED |
| 42 | Episode window (episode detection with a sliding window, features from inside the window) | mixed 0.9416→0.9392, all unlabelled 0.7027→0.6954 | — | — | REJECTED |

| 43 | **Collusion table (AAAI 2013, Mazrooei/Archibald/Bowling)** — `src/09_collusion_table.py`: value function V_j (RMSE ~21bb), C(j,k)=Σ[V_j(h·a)−V_j(h)] | pair PU metric 0.7077 → **0.7438**; evidence reranker 0.6838 → 0.6789 (harmful) | — | — | SPLIT USE |
| 44 | **Split configuration (v33)**: evidence chain without CT (04/04c/04b/05b), pair model with CT (05) | pair PU **0.7757** (best so far, previously 0.7438), evidence reranker **0.6869** (best), Behavior PU 0.7483, behaviour OOF accuracy 0.9785 | — | — | KEPT → submission_v33 / v34_novel70 |
| 45 | Candidate-set recall ceiling (`scratch/eda/38_cand_recall.py`) | true evidence recall in the top-50 candidates 0.9989, 1.000 in the top-100; recall@5 = 0.627 | — | — | information: NCAND is not the bottleneck |

### 43-44. The collusion table: where you put it matters
The idea from Alberta's 2013 paper is the sum of "how much did k's action change j's
expected value". It was extracted per hand as `help_p_by_q`, `help_q_by_p`, `ct_mutual`,
`ct_asym`, `ct_max`, `ct_pos`.

The result points in opposite directions on the two sides: **strong at the pair level**
(PU metric 0.7077 → 0.7438), **harmful at the evidence level** (0.6838 → 0.6789). The
reason is plausible: value transfer is a cumulative signature when you look at a pair's
whole history, and noisy on a single hand, where it does not separate the real evidence
hands from other high-variance ones.

So it was split (`run_split.sh`): 04, 04c, 04b and 05b with `USE_CT=0`, only 05 with
`USE_CT=1`. That combination gave the best value on both metrics — pair PU 0.7757,
evidence 0.6869. Since the pair term is 70% of the metric, this is where the hope is.

### 45. Where the evidence loss is
The candidate list (top-50) contains 99.9% of the true evidence hands, so there is no
"the right hand never makes the list" problem. The loss is entirely a ranking problem
among the first 15 hands. That says enlarging NCAND (100, 200) will not help — no need
to try it.

| 46 | 5-seed rank-averaged pair model (with CT, on top of the v33 chain) | pair PU 0.7757 → 0.7638, Behavior PU 0.7483 → 0.7384 | — | — | REJECTED (same result as experiment 35) |

### 46. Seed averaging failed again
In experiment 35 three seeds were neutral on the leaderboard (+0.0001). Retried on the
new chain with five seeds, this time it **hurt offline as well**: pair PU 0.776 → 0.764.
The likely reason is training on only 1,860 labelled pairs — seed averaging reduces
variance but also softens the sharp boundary a single seed captures. This direction is
closed and will not be retried.

## 4. Submission plan for 18 September 00:00 UTC (`run_submit_0918.sh`, scheduler running)

| # | File | What it measures |
|---|---|---|
| 1 | `submission_v33.csv` | new chain, no fourth-family rule — the pure LB effect of CT + MIL (reference: v21 = 0.90387) |
| 2 | `submission_v34_novel70.csv` | v33 + the LOFO rule at the 70% setting (444 pairs) |
| 3 | `submission_v35_novel50.csv` | v33 + the LOFO rule at the 50% setting (309 pairs) |
| 4 | conditional | if the rule curve is rising, `v35_novel85` (463 pairs); otherwise `v26` (old-chain control, which isolates CT's contribution) |
| 5 | reserve | manual, based on the first four scores |

The script waits for the first three scores and picks the fourth slot itself, so a day's
quota is not tied to a single prediction.

| 47 | Reranker LambdaRank truncation level (metric is MAP@5, default 10) | 2 seeds: 5 → 0.6804, 10 → 0.6832, 20 → 0.6852. 6 seeds: 10 → 0.6869, 20 → 0.6876, 35 → 0.6830 | — | — | REJECTED (no peak, noise) |
| 48 | Truncating the dev window, retried with the pair metric (repeat of experiment 7) | TRUNC=0.68: pair PU 0.7757 → 0.7560, Behavior PU 0.7483 → 0.7386 (labelled behaviour rose 0.959 → 0.967 but PU fell). TRUNC=0.50: 0.7172 / 0.6782 | — | — | REJECTED (experiment 7 confirmed) |

### 47-48. Both hyperparameter directions are closed
**Truncation level:** LambdaRank's `lambdarank_truncation_level` was not aligned with the
metric (MAP@5). Lowering it (5) hurt; raising it (20) showed +0.0020 on 2 seeds but only
+0.0007 on 6 seeds, and at 35 it fell below 10 (0.6830). There is no monotone trend — the
three values sit inside seed noise. Left at the default 10.

**Dev window:** evaluation pairs share fewer hands than development pairs (median 76 vs
112, `scratch/eda/39_n_dist.py`), so pulling training towards that distribution looked
sensible. But the pair PU metric falls: truncation weakens each of the 372 positive pairs,
and the loss from training on less data is larger than the gain from matching the
distribution. A telling detail: labelled behaviour MAP rises (0.959 → 0.967) while PU
behaviour falls — exactly the "a local improvement is not evidence on its own" lesson again.

| 49 | **Relational collusion table** (`src/09b_ct_relational.py`): does the partner help more than the player's other opponents — within-player z-score and rank | strong alone (on labelled pairs `ctr_max_hp_mean_rk` AUC **0.846** / AP 0.578, `ctr_min_hp_mean_z` 0.843 / 0.586). In the pair model: 12 features 0.7757 → 0.7594; best 2 features 0.7723; 1 feature 0.7617; 3 features 0.7566. Behavior PU 0.7483 → 0.744-0.746 | — | — | REJECTED |

### 49. A strong feature with no place in the model
The raw collusion table says "how much did k's actions help j"; some players structurally
receive a lot of help (loose table, passive opponents). So the same normalisation as in the
relational LLR was applied: the partner's help compared with the help **the same player**
receives from their other opponents.

Alone the reading is very good: AUC 0.846, one of the strongest single features we have (on
a par with `n3_fold_to_pair` at 0.865). An interesting detail is that the sign of the mean
help is **inverted** (AUC 0.697 in the negative direction): a colluder receives less help on
average from their partner, but the positive part of the help, and its concentration on the
best hands, is very high. The signature is not "constant help", it is "help concentrated on
selected hands".

Against that, it contributed nothing to the pair model in any combination — the 12-, 3-, 2-
and 1-feature variants all stayed below 0.7757. The reason is redundancy: the collusion
table itself (experiment 43) and the relational LLR are already in the pair model and
measure the same thing; with 1,860 labelled pairs every new feature adds variance.

Care was taken not to threshold: a minimum-hands filter would leave the filtered pairs at a
constant fill value and reproduce the Rule 6 artefact.

## 4b. The 18 September submissions — three components measured in isolation

| Submission | What it adds | Offline promise | **LB** | Difference |
|---|---|---|---|---|
| v20 (reference) | — | — | 0.90268 | — |
| v21 (our best) | + fourth-family rule | — | **0.90387** | +0.0012 |
| v26 | MIL evidence chain | evidence CV 0.6485 → 0.6866 (+0.035) | 0.90012 | **−0.0026** |
| v33 | + collusion table (pair model) | pair PU 0.708 → 0.776 (+0.068) | 0.90081 | +0.0007 |
| v34 | + fourth-family rule | behaviour diagnosis suggested up to +0.0087 | 0.90081 | **0.0000** |

### Rule 7 (CORRECTED on the evening of 18 September — the first version below was too general)
**Corrected form:** the evidence metric (OOF MAP@5 on public positives) predicts the
leaderboard **well** — the transfer table in section 5.1 and the formula in section 6 show
it. Two specific things are broken: (a) the pair PU metric (Rule 9), and (b) standard OOF
for methods that rewrite their own labels (Rule 8). Three components falling on the same day
gave the impression that "every metric is broken"; the real reason was that all three fell
into one of those two categories.

#### The first (too general) form, and that day's observations
All three components were strong offline and worthless or harmful on the leaderboard:

- **MIL evidence model**: +0.035 in evidence MAP@5 (weight 0.20 → expected +0.007) → actual
  **−0.0026**. Even the sign is inverted.
- **Collusion table**: +0.068 on the pair PU metric, the largest local jump so far → actual
  +0.0007, noise.
- **Fourth-family rule**: +0.0012 on top of v21, exactly **0.0000** on top of v33. Does not
  reproduce.

Hypothesis for the common cause: since v20 every model decision has been chosen on the
**372 public positive pairs**. The evaluation set contains different pairs and also the
undisclosed fourth family (~10%). Sharpening methods like MIL may improve on the three known
families while breaking on the unseen one — a testable prediction (measure evidence MAP
under LOFO).

**That day's outcome:** best submission v21 = 0.90387. *(Later the same day, with the leak
cleaned, it was passed by v41 = 0.90731 — see experiment 57.)*

## 5. Fixing the offline measurement (18 September)

### 5.1 The evidence metric was working after all — MIL is the single exception
For every step on the evidence side, the offline CV difference was placed next to the LB
difference. The metric is linear, so LB difference / 0.20 = the LB's evidence MAP@5
difference:

| Step | Method | Offline Δ | LB Δ | Implied LB evidence Δ | Transfer |
|---|---|---|---|---|---|
| v1→v2 | chronological reranker | +0.096 | +0.0141 | +0.071 | 74% |
| v3→v4 | stage-1b + 50 candidates | +0.020 | +0.0017 | +0.009 | 43% |
| v4→v7 | per-family stage-1b | +0.027 | +0.0051 | +0.025 | 94% |
| v7→v13e7 | action-model evidence | +0.036 | +0.0029 | +0.015 | 40% |
| **v20→v26** | **MIL evidence chain** | **+0.038** | **−0.0026** | **−0.013** | **−34%** |

In four consecutive steps the offline gain transferred at 40-94%. That makes the evidence CV
a sound metric. MIL is the only exception that changes sign — so "something specific is
wrong with MIL" is more likely than "all metrics are broken".

Reading the MIL code, a candidate mechanism appears: in round r the labels of fold f are
chosen by round r−1's f-model (trained on the folds other than f, g included); then round
r's g-model is trained on f's selected labels and evaluated on g. g's test data leaks into
its own training labels in two steps, and it accumulates over 4 rounds. Test: `HOLDOUT_FOLD`
— one fold takes part in no round and is scored once at the very end (`run_milhold.sh`).

### 5.2 The MIL leak confirmed (`run_milhold.sh`, HOLDOUT_FOLD=4, 70 pairs)

| | Standard OOF (inner folds) | **Strict holdout** |
|---|---|---|
| No MIL (1 round) | 0.4478 | 0.4626 |
| MIL 4 rounds, 1 action | **0.5177** (+0.070) | **0.4616** (−0.001) |

MIL's +0.07 gain **disappears completely** on the fold that took part in no round. What
standard 5-fold OOF cannot see: label refinement couples the folds to each other across
rounds. The −0.0026 on the leaderboard is explained by this. Experiment 36 (MIL, KEPT) →
**REJECTED**; the evidence chain of v26/v33/v34 is invalid.

| 50 | MIL strict-holdout test | inner OOF +0.070, holdout −0.001 | — | — | **MIL REJECTED, leakage** |

### Rule 8: any method that replaces labels with model output (MIL, self-training, pseudo-labels) cannot be measured with standard OOF
The folds become coupled through the labels. Such a method is measured only on a fold that
took part in **no** round. Three attempts of this kind were made in this competition (PU
self-training v5, pseudo-labels, MIL); all three rose locally and fell on the leaderboard.
From now on, every idea in this class goes through the holdout test first.

### Rule 9: there is no offline resolution left on the pair side
- Labelled AP is saturated (0.998): it cannot measure a difference.
- The PU metric (unlabelled counted as negative) misleads: a 0.708 → 0.776 jump gave +0.0007
  on the leaderboard. There are ~1000 hidden positives among the unlabelled; the metric
  **punishes** the model for finding them and rewards it for learning the 372 disclosed
  pairs specifically.
- The only reliable measurement is an LB probe. In a submission that holds v21's evidence and
  behaviour fixed and changes only the risk column, LB Δ / 0.70 = Pair AP Δ exactly. Each
  pair candidate costs one submission; with 12 slots left, 2-3 candidates can be measured.

### The valid protocol for the evidence side (proven to work)
Standard OOF evidence MAP@5 transfers at 40-94% for methods that obey Rule 8. If evidence
changes are bolted onto v21's risk/behaviour and submitted, LB Δ / 0.20 = LB evidence Δ, so
each submission gives both a score and a calibration.

**Valid best chain: v21** (0.90387). No MIL, no CT.

| 51 | **Stale-artefact contamination** (`actmodel/dev_cvonly.parquet` caught by the glob) | reranker CV 0.6535 → 0.897, pair PU 0.721 → 0.933 — entirely spurious | — | — | FIXED + guard added |
| 52 | Clean chain rebuilt (no MIL, no CT = the v21 configuration) | reranker CV **0.6535**, pair PU 0.7209, Behavior PU 0.7047 | — | — | VALID BASELINE (v40) |
| 53 | Recall diagnosis by evidence strength | rank1 0.847, rank2 0.841, rank3 0.769, rank4 0.629, **rank5 0.459**; 3.54 of the 5 hands | — | — | information |
| 54 | Difficulty-graded lambdarank (high grade for weak evidence) | hard 0.5431 (rank5 0.459→0.582 but rank1 0.847→0.500), easy 0.6436, current **0.6542** | — | — | REJECTED (we are at the optimum of the trade-off curve) |

### 51. The third measurement error, and a permanent guard
The first run of the clean chain gave extraordinary numbers: reranker 0.897, pair PU 0.933.
Instead of celebrating, they were checked — the numbers looked like the leak signature we
had been burned by twice.

The cause was not in the model: that day's MIL holdout test had left an
`actmodel/dev_cvonly.parquet` file behind. Downstream stages read that directory with
`*.parquet`, so the dev rows were loaded twice (223,749 → 447,498 rows, unique key count
unchanged). A single hand counted twice in the top 5 inflates MAP@5 by itself.

Three measures were taken: (1) CV output is now written outside the globbed directory,
(2) `_assert_unique_keys` in all three consumers — if (hidx,p,q) is not unique after the
glob, the process fails loudly, (3) it was recorded as a rule.

### Rule 10: if an artefact directory is read with `*.parquet`, nothing else is ever written into it
No side outputs, no CV copies, no backups. Duplication silently inflates the metric and never
raises an error at any stage.

### 53-54. The evidence loss is in the weak manipulations, and it cannot be taken from there
The organiser ordered each pair's five evidence hands by strength, and our recall follows
that order exactly (0.847 → 0.459). Since all five count equally in MAP@5, the hypothesis "we
are spending capacity on the easy hands" suggested itself.

The test refuted it. Giving hard hands a higher grade does work (rank-5 recall 0.459 →
0.582), but it loses far more on rank-1 (0.847 → 0.500) and the net MAP@5 falls. The opposite
direction (`easy`) also falls. The current setting is at the top of that trade-off curve.
Weak manipulation hands are not a capacity problem — they really are less distinguishable.

| 55 | Reranker configuration blend (full / no_rel / no_s1b / easy, averaged as within-pair ranks) | best single 0.6554, best blend 0.6550; rank correlation 0.948-0.988 | — | — | REJECTED (no diversity) |
| 56 | Set selection (choosing the five hands with a temporal-proximity bonus rather than independently) | bonus 0.02 → 0.6540, 0.10 → 0.6371, 0.40 → 0.5098; evidence tpos std 0.2345 vs all hands 0.2915 | — | — | REJECTED |
| 57 | **v41: clean chain (no MIL, no CT) + fourth-family rule** | evidence CV 0.6535 | **0.90731** | **+0.0034** | KEPT — NEW BEST |

### 57. The chain cleaned of leakage brought a real gain
The difference between v21 (0.90387) and v41 (0.90731) is +0.0034. Since the evidence CV
difference is only +0.005 (expected LB effect +0.001), part of the gain comes from elsewhere:
04b's behaviour source and a wider application of the fourth-family rule (196 → 432 pairs).

What matters is this: it is the first step **forward** in three days, and it came from
cleaning up a leak. Removing a method gained more than adding one.

| 58 | v43: widening the rule to the top 3000 pairs (432 → 1478 flagged) | — | **0.90731** | **0.0000** | information (very valuable) |

### 58. Zero difference, large information: the positives are inside the top 1500
Relabelling 1,046 extra pairs as `other_coordination` changed the score by **exactly zero**.
Since Behavior MAP is computed only over the true positive pairs, there is almost no true
positive in the 1500-3000 rank band. Nearly all evaluation positives sit in the first ~1500.

Two consequences: (1) the claim that Pair AP is genuinely high (≈0.979 estimate) is
independently confirmed, (2) the fourth-family effort only means something inside the top
1500, and the **count** there is the only real knob. `NOVEL_KEEP` was a weak lever (0.70→0.85
moved only 20 pairs); `NOVEL_N` was added instead, labelling the N most "novel"-looking pairs
in the band directly.

## 6. The LB prediction formula (18 September) — to be run before every submission

The metric is linear: `0.70·PairAP + 0.20·EvidenceMAP@5 + 0.10·BehaviorMAP`. In clean
versions the first and third terms stay nearly constant, so the only unknown is the evidence
term:

> **LB ≈ 0.7744 + 0.20 × (offline evidence CV)**

The `pair+behavior` constant sits between 0.77128 and 0.77661 across six clean versions, std
**0.0018**. The evidence-blanked probe taken at v20 measured it independently at 0.77367 —
0.0008 from the computed value. The same number by two separate routes; the formula is
reliable. Computation: `scratch/eda/44_predict_lb.py`.

### Its real use: a diagnostic
| Version | Prediction | Actual | Residual |
|---|---|---|---|
| v41 (clean) | 0.9051 | 0.90731 | +0.0022 (the rule lifted behaviour) |
| **v26 (MIL)** | 0.91174 | 0.90012 | **−0.0116** |
| **v33 (MIL+CT)** | 0.91180 | 0.90081 | **−0.0110** |

Normal deviation is ±0.002. In the two leaking versions the residual is **six times** that
and negative. So if this calculation had been done before MIL was submitted, the first result
would have revealed the inflated metric and two further submissions would not have been spent.

**Rule 11: write the prediction before every submission, then look at the residual.** If the
residual is larger than −0.005, that local metric is broken; do not make a second submission
in that direction, run the strict holdout test (Rule 8) first.

### The fourth-family curve — the 19 September sweep REFUTED this inference
The "curve" above was an illusion born of comparing different chains: v20/v21 and v41 are not
the same chain, and attributing the difference between them to the rule was a mistake. A
direct sweep (all on the same chain, only the count changing):

| Pairs labelled | LB |
|---|---|
| 0 (rule off) | **0.90731** |
| 150 | **0.90731** |
| 432 (v41) | **0.90731** |
| 800 | 0.89907 |
| 1200 | 0.87460 |

The rule's contribution is **exactly zero**, and more of it is harmful. Since relabelling 432
pairs did not move the score at all, **none of those pairs is a true positive** — our
"novelty" score systematically selects non-positives. Past 800 we start breaking the family
of real positives. v41's +0.0034 over v21 comes entirely from the **chain** (from cleaning
the MIL and CT leaks).

## 7. The 19 September sweep — CONCLUSION: the fourth-family rule is dead

| 59 | Fourth-family rule 1-D sweep (0/150/432/800/1200 pairs) | — | 0.90731 / 0.90731 / 0.90731 / 0.89907 / 0.87460 | 0.0000 / −0.008 / −0.033 | **REJECTED — rule removed entirely** |

The net result of four submissions: the rule's contribution is zero and more of it hurts. The
Behavior term is at 0.883 and does not open this way. The only real pockets left are Pair
(0.979 → 1.0 = +0.0147) and Evidence (0.645 → ?, weight 0.20).

**An important operational note:** the Kaggle public LB shows the best score, so a bad
submission does not lower our current best. The cost of risk is only the slot spent, which
makes high-variance attempts cheap.

| 60 | **The fourth family = the pairs where the behaviour head is UNDECIDED** (`src/10_ambiguous_family.py`, 326 pairs, top family probability < 0.60) | — | **0.90858** | **+0.0013** | KEPT — NEW BEST |
| 61 | The ceiling of evidence ordering | 3.487 of our five chosen hands are correct; a perfect order gives 0.6540 → **0.7178** (worth +0.0128 on the LB) | — | — | information |
| 62 | Alternative ordering criteria (s_ev, s1b, raw surprisal, chronology, blends) | none beat the reranker's own order (0.6542 vs best alternative 0.6373) | — | — | REJECTED |
| 63 | Single-feature sweep for evidence selection (within-pair z, all features) | the best single feature is already `s_ev` (AUC 0.9332); raw surprisal 0.9176 | — | — | information: no new signal |
| 64 | Evidence order ↔ chronology structure | Spearman median **+0.70**, r>0.8 in 49% of pairs; rank 1 is the earliest (position 0.227) and the strongest | — | — | information (the reranker already uses it) |

### 60. Looking for the fourth family in the right place
Experiment 59 showed the old rule (high risk + LOW evidence score) returned exactly zero:
none of the 432 pairs it chose was a true positive. The new hypothesis is mechanically
sounder: the behaviour head has three classes and cannot place an undisclosed family in any
of them, so it stays **undecided** on those pairs.

The asymmetry is in our favour too: where the model is already unsure, the current label is
probably wrong, so changing it is cheap. Result +0.0013.

### 61-62. Evidence ordering: a large gap that cannot be closed
On average 3.487 of our five chosen hands are correct. Ordering those same five perfectly
would take MAP@5 from 0.6540 to 0.7178 — with weight 0.20 that is **+0.0128 on the LB**, more
than all our recent gains put together.

But every criterion we could try (s_ev, s1b, raw surprisal, chronology, weighted blends)
stayed below the reranker's own order. That is expected: if the score is a monotone function
of P(evidence), ordering by it already maximises expected AP. The gap is the model's
irreducible uncertainty.

## 8. 20 September — the last day's plan (REVISED: target first place, `run_submit_0920.sh` running)

The user's instruction: no submission that has no chance of reaching first place. First is
~0.935, we are at 0.90858, a gap of +0.027. No single-component change can deliver that
(the Pair ceiling is +0.015, Behavior +0.010, and every measured direction on the evidence
side is at its ceiling). So each submission is three high-variance changes **stacked**:
collusion table in the pair model (Pair) + behaviour ambiguity (Behavior) + unsupervised G
blend (Pair).

| # | File | Ambiguity threshold | Blend w | Note |
|---|---|---|---|---|
| 1 | `s1_amb60_bl05` | 0.60 (327) | 0.05 | moderate package |
| 2 | `s2_amb75_bl15` | 0.75 (596) | 0.15 | middle |
| 3 | `s3_amb85_bl30` | 0.85 (781) | 0.30 | aggressive |
| 4-5 | conditional | further in the direction of whichever of the first three is best (0.95 / w=0.50) | | |

Honest probability: ≪5% for first place. But it is the only non-zero strategy; a small
guaranteed gain is worthless to the user.
**Kaggle final selection:** if the final standing is computed from the private LB, the two
final submissions must be ticked by hand before the close (the API has no such operation).
Candidates: the highest-scoring public package + v45 (safe).

| 65 | Blend calibration (dev OOF risk × relational G, `scratch/eda/50_blend_calib.py`) | w=0→0.30: PU-AP 0.7365→0.7023 but the count of known positives in the top band goes 368→363 and their median rank 253→245 (improving); at w=0.50 it is 362, at w=1.0 352 | — | — | information: the blend reshuffles the unlabelled region, not the known positives; w≤0.30 safe, 0.50 starts costing |
| 66 | Strength of the unsupervised signal (`scratch/eda/49_unsup_strength.py`) | the G_rate+G_max rank blend alone reads AUC **0.9959**, PU-AP 0.6318 (model 0.7209); on labelled pairs 0.9883 (model 0.998) | — | — | information: the blend partner is nearly as strong as the model |
| 67 | **Behaviour-head LOFO** (`BEH_LOFO=1`, 2-class head, unseen family) | catching the unseen family through "indecision": threshold 0.75 → 8.5% (2.6% of the known families mislabelled), 0.85 → 21.8% (4.6%), 0.90 → 41.4% (6.3%), **0.95 → 77.5% (11.4%)**; with soft_play unseen, almost nothing is caught up to 0.85 | — | — | information: the indecision rule only catches the unseen family at an aggressive threshold, and then breaks 5-11% of the known families |

### 65-67. Calibrating the last day's packages
Three measurements, all at the limit of what is possible offline:

- **The blend** is safe: up to w=0.30 the known positives stay where they are and only the
  unlabelled region (where the hidden positives are) is reordered. The fall in PU-AP is
  meaningless for exactly that reason (Rule 9).
- **The indecision rule** is a coin: the unseen family is only caught in serious numbers at
  thresholds 0.85-0.95, and at those thresholds 5-11% of the known families also become
  "other". Rough arithmetic: if the fourth family is ~10% of the population and its AP goes
  from 0 to 0.5, Behavior MAP gains +0.125; losing 11% of the three known families costs
  ≈ −0.07 → the net can be positive, which with weight 0.10 is ±0.01. That v44's novelty
  rule lost at 800-1200 pairs supports the same reading: go deeper and you start hitting
  real positives. The difference is that the positives indecision hits are probably
  mislabelled anyway.
- Hence the package order is right: climb 0.60 → 0.75 → 0.85, and try 0.95 only if 0.85 wins.

## 9. 19 September: the team merge — the other side's largest gain, rebuilt on our chain

The teammate's repository (`team-mates-repo/`) contains code only, no CSVs and no artefacts,
and we were told no files would be coming. So I read their **largest late gain**
(`150_opponent_aware_policy.py`, **+0.0087/+0.0100** on the leaderboard) and rewrote it on
our own chain: `src/13_oppaware.py`.

**The idea, and how it differs from our 03b.** Our relational LLR asks "does A play
differently against the partner than against their own other opponents" — the baseline is
the player themselves. This block asks something different: **while the partner was live in
the hand, how far did A's moves depart from what the population policy expected in those
spots**, scaled by the model's own variance (`(O−ΣP)/√ΣP(1−P)`), so that a run of easy hands
cannot manufacture a signal. A counting-based LLR and a model-based residual fail in
different ways, which is why the two add on top of each other. The directed (i→j) statistics
are folded onto the pair as hi/lo: G likelihood ratio + χ² tail depth, fold/call/aggression
deviations and z-scores, surprisal and excess surprisal, and the subset "while A is
responding directly to B's aggression". 37 columns, 12 seconds, no label is read anywhere.

| 68 | **Opponent-aware policy residuals** (`src/13_oppaware.py`) alone, on the 1,860 labelled pairs | `oa_surp_excess_hi` AUC **0.9969**, `oa_g_tail_sum` AUC 0.9932 / **PU-AP 0.6853** | — | — | information |
| 69 | Comparison: our strongest existing feature | relational `G_max` AUC 0.9848 / PU-AP 0.3704; the whole pair model PU-AP 0.7209 | — | — | information |
| 70 | **OA block in the pair model, 37 columns** | labelled AP 0.9977 → **0.9989**, **Behavior MAP (labelled) 0.9604 → 0.9920**, PU 0.7209 → 0.7029 | — | — | CANDIDATE A |
| 71 | OA block, strongest 5 columns | labelled AP **0.9989**, Behavior 0.9632, PU 0.7209 → **0.7351** | — | — | CANDIDATE B |

**Reading.** A single untrained number (`oa_g_tail_sum`, PU-AP 0.685) separates almost as
well as our whole trained pair model (0.721) — twice our best existing feature. With all 37
columns both honest metrics reach their best values (labelled AP 0.9989, behaviour 0.9920);
the only thing that falls is the metric we know to be structurally invalid (Rule 9) — and
since it punishes finding hidden positives, its falling may not be bad news. The 5-column
variant raises PU as well; the two candidates are deliberately in different directions.

| 72 | **The hand-level form of the OA block** (`oppaware_hand.parquet`, 30M rows): how much does a hand sample the pair's characteristic deviation (log tilt) | within-pair single-feature AUC: `oah_nll` 0.9183, `oah_tilt_hi` 0.8857 (s_ev 0.9332) | — | — | information |
| 73 | **OAH in the ranker** (3 seeds) | evidence CV 0.6542 → **0.6608**; rank-4 recall 0.629 → 0.643, **rank-5 0.459 → 0.485** | — | — | KEPT |

### 72-73. The first real movement on the evidence side
The pair-level block says "this couple characteristically deviates like this". The
hand-level form applies that deviation vector (log tilt) to individual hands: a hand scores
high if it is made of the moves the pair produces in excess. The evidence term asks exactly
that, and our ranker did not have the information.

The gain is +0.0066 MAP and — more importantly — **exactly where we were most stuck**: the
"we miss the weak manipulation hands" problem measured in experiment 53 moved for the first
time (rank-5: 0.459 → 0.485). What difficulty weighting could not do in experiment 54, new
information did.

### The metric reality (evening of 19 September, target 0.938)
| Scenario | Evidence MAP needed to catch the leader |
|---|---|
| Pair 1.00, behaviour 1.00 | 0.690 |
| Pair 0.99, behaviour 0.98 | 0.735 |
| Pair 0.98, behaviour 0.97 | 0.771 |

We are at 0.645 on evidence. **Even with perfect Pair and behaviour, if evidence stays at
0.645 the ceiling is 0.929.** So all remaining effort must go to the evidence term; the OA
block's Pair/behaviour gain is valuable but does not cross the threshold on its own.

| 74 | Bet-size signature in the ranker (`src/14_sizing_hand.py`, 18 columns) | evidence CV 0.6608 → **0.6588** (3 seeds, same baseline) | — | — | REJECTED |

### 74. The sizing block did not transfer to us
In the teammate's ablation the sizing block was the second most valuable after chronology
(−0.0244 MAP when removed). Here it gave a small fall instead. The likely reason: our action
model already uses `size_z` at the action level, and hand-level summaries are not new
information but a noisy repeat of the same information. Their chain has no action model, and
the sizing block was filling that gap. **Lesson: another pipeline's ablation value does not
transfer if your own pipeline does not have the same gap.**

## 10. 20 September submission candidates (ready, validated)

| File | Evidence CV | Contents |
|---|---|---|
| `submission_A2_oa_both.csv` | **0.6609** | OA pair + OA hand + family ambiguity rule (495 pairs) |
| `submission_v54.csv` | 0.6609 | the same, rule off |
| `submission_A_oa_full.csv` | 0.6535 | OA pair + old evidence chain + rule |
| (reference) v45 = 0.90858 | 0.6535 | no OA |

| 75 | **Opponent-conditioned policy** (`src/15_oppcond_policy.py`): the expectation itself is conditioned on the opponent's identity, position and six style rates; 17.1M (action × live opponent) rows, fit error 0.469/0.473 | alone PU-AP **0.6744** (step 13: 0.6853) — weaker | — | — | REJECTED on its own |
| 76 | **Step 13 + step 15 together, in the ranker** | evidence CV 0.6608 → **0.6670** (3 seeds); rank-3 0.763→0.780, rank-5 0.485→**0.506** | — | — | KEPT |
| 77 | Both blocks, full chain (6 seeds) | evidence CV **0.6658**, labelled AP **0.9989**, Behavior **0.9920** | — | — | CANDIDATE A3 |

### 75-77. The sharp-model paradox, and why both are needed
The opponent-conditioned expectation is logically more correct: "this opponent is not like
the others" is a legitimate reason, and removing it should leave pure signal. But alone it
came out **weaker** (0.6744 vs 0.6853) — the sharper the model, the less surprised it is by
collusive moves too. This is the same mechanism that rejected policy v2 in experiment 3
(log-loss 0.500→0.414, evidence neutral).

Stacked, however, it gains: 0.6608 → 0.6670. The two blocks are not two readings of the same
thing; one says "deviated from what the situation expected", the other "deviated from an
expectation that also accounts for the opponent", and the difference carries information.
Lesson: a signal that is weak on its own is still worth adding if its conditioning differs.

## 11. 20 September candidates — final (none submitted, awaiting the user's approval)

| File | Evidence CV | Labelled AP | Behavior | Contents |
|---|---|---|---|---|
| `submission_A3_both_blocks.csv` | **0.6658** | **0.9989** | 0.9920 | both blocks + ambiguity rule (481 pairs) |
| `submission_v55.csv` | 0.6658 | 0.9989 | 0.9920 | the same, rule off |
| `submission_A2_oa_both.csv` | 0.6609 | 0.9988 | 0.9922 | one block + rule |
| (current best, submitted) v45 | 0.6535 | 0.9977 | 0.9604 | no block — **LB 0.90858** |

The teams have merged: the submission quota is shared, the teammate's v58-v70 files also
appear in the list, and the combined best is 0.90858.

| 78 | **v55: both opponent-aware blocks, pair model + evidence ranker, no fourth-family rule** | evidence CV 0.6658, labelled AP 0.9989, Behavior 0.9920 | **0.91748** | **+0.0089** | KEPT — NEW BEST |

### 78. Where the gain came from: the prediction formula's residual says so
The prediction is `0.7744 + 0.20 × 0.6658 = 0.90756`, the actual **0.91748**, residual
**+0.0099** — five times the normal deviation (±0.0018) and **positive**. Because the
formula's constant assumes the `0.7P + 0.1B` term, that residual means precisely that this
term rose: 0.7743 → **0.7843**.

Decomposition: evidence side +0.0025 (CV +0.0123 × 0.20), pair+behaviour side **+0.0100**.
That matches exactly what the teammate measured for the same block in their own chain
(+0.0087/+0.0100) — the method transferred at the same magnitude into two independent chains.

**Note:** a large positive residual is Rule 11 in reverse; the formula's constant is no
longer valid, because it rested on the assumption that pair and behaviour stay fixed and this
version broke exactly that assumption. New constant: **0.7843**.

## 12. Current state (20 September 00:15 UTC)
| | |
|---|---|
| Best score | **0.91748** (v55) |
| Previous best | 0.90858 (v45) |
| Merged team, remaining slots | 4 (today, shared quota) |
| Close | 20 September 22:00 UTC |

## 13. 20 September: new-method research (the user: "take risks, do not be timid")

Three new axes were opened. The first is our own idea; the other two came out of a full
inventory of the teammate's repository (that inventory also closed several axes **as
measured** — marked "closed" below).

### Are there collusion rings? (our own idea)
Among the 372 labelled positive pairs, 49 players appear in more than one positive pair; a
permutation null that holds the pair graph fixed says 25.4 ± 4.2 → **5.6 sigma**, so
collusion is a player property too. The number of closed triangles, however, is **zero**:
the structure is star-shaped, not a ring.

| 79 | **Pair-graph features** (`src/16_graph_pair.py`, 28 columns: hub = the player's *other* best partner, exclusivity = rank/z/share inside the player's own partner distribution, triangle support) | alone: the exclusivity columns are a restatement of the baseline (AUC 0.9928 vs baseline 0.9932), hub/triangle alone zero. In the model: labelled OOF AP 0.99856 → 0.99842 (+graph) / 0.99857 (hub only); in the universe **all** 372 known positives are already in the top 1000, median rank 278 → 271 | — | — | WEAK POSITIVE, shelved |

**Reading.** The hypothesis is right but there is no usable gain: 42.5% of the 40 hardest
positives are hub-connected (baseline 25.8%), so the structure does concentrate exactly where
we struggle — but the model already carries the information another way. The teammate's
repository reached the same conclusion twice more (`80_player_graph_features.py` REJECTED;
`300_relational_graph_residual.py`: "the entire apparent gain is leakage, the file was not
written"). **Axis closed.**

### Full inventory of the teammate's repository — the things we had no equivalent of
In their own ablation the block that carried their evidence engine most was
**episode/position (11 columns): −0.1458 MAP@5** — six times the bet-sizing block we had
REJECTED (−0.0244). We had no equivalent.

| 80 | **Contact-restricted chronology + neighbourhood + wall clock** (`src/17_chrono_hand.py`, 17 columns) | a pair shares ~120 hands but real contact happens in a median of **42**; position is now measured inside that subset. Neighbouring-hand (excluding itself) transfer/surprisal means, 9-hand density, seconds to the previous/next hand (median 77 s, long tail = session break) | — | — | being measured |
| 81 | **Leave-one-out direction consistency** (`src/05b`, `USE_DIR=1`, 7 columns) | does this hand carry value in the direction the pair's *other* hands carry (a hand does not vote for itself); plus the preflop equity difference signed the same way — the stronger hand giving up the pot | — | — | being measured |
| 82 | **Within-pair normalisation** (`RERANK_PAIRNORM=1`): within-pair rank + z for all the new opponent-aware columns | the scale is a property of the pair (how many hands, how loose), not of the hand; with 1,860 groups it is hard for the model to learn that from raw values | — | — | being measured |
| 83 | **Soft routed specialists** (`RERANK_ROUTED=1`): a pooled model + three family specialists over within-pair ranks, `0.25·pooled + 0.75·Σ_f P(family=f)·specialist_f`. Our `RERANK_PER_FAMILY` was either/or; in their chain this architecture was worth +0.0091 | — | — | — | being measured |

### Axes the inventory closed (we will not retry them)
- **Neural networks / embeddings / sequence models:** all four were tried. An attention-based
  MIL ranker **lost 0.0301** to GBDT, negative on all five holdout folds. An end-to-end
  learned pair latent (49M rows) is real — it beats its permutation null by 2.8× — but its
  **conditional contribution on top of the `oa_*` aggregates is absent at all 25 steps**. The
  JEPA/transformer route is closed by measurement.
- **Episodicity (clustering):** the gain survives a time-shuffled control → what is being
  measured is density, not episodicity. On top of that, the planted hands are spread over
  55.8% of the timeline.
- **Graph/rings:** above.

### Results of 80-83: six of seven axes closed, one opened
The evidence CV baseline is 0.6670 (v55 chain, 3 seeds, measured with RERANK_CV_ONLY;
0.66696 reproduced exactly).

| Experiment | Method | Evidence CV | Decision |
|---|---|---|---|
| 80 | Contact-restricted chronology + neighbourhood + clock gaps (`src/17_chrono_hand.py`) | **0.6650** | REJECTED on its own |
| 82 | Within-pair normalisation (`RERANK_PAIRNORM=1`) | **0.6649** | REJECTED |
| 80+82 | Both together | **0.6618** | REJECTED |
| 81 | Leave-one-out direction consistency (`USE_DIR=1`) | **0.6621** | REJECTED |
| 83 | Soft routed specialists (`RERANK_ROUTED=1`) alone | **0.6674** | neutral |
| 83+80 | **Specialists + chronology** | **0.6710** | **KEPT (+0.0040)** |

| 84 | **Reordering the top 5** (the same five hands, 24 rules: s_ev, earliness, surprisal and blends with s2) | current order 0.6543, best alternative 0.6545 | — | — | REJECTED (experiment 62 confirmed) |
| 85 | **Phase contrast** (`/tmp/phase.py`): the same pair's value in the other phase, a free pair-specific null | the mechanism is **confirmed** — the evaluation-window AUC of dev positives is 0.477-0.504, i.e. pure chance; collusion is phase-specific. But in the model: labelled AP 0.99860 → 0.99859, universe median 276 → 273 | — | — | REJECTED (the information is already inside) |

### Why most were REJECTED — and what the single exception means
In two of the six axes the hypothesis was **confirmed** (player-level structure 5.6σ;
phase specificity AUC 0.48) and none of them turned into a gain. The recurring pattern: our
chain (action model + stage1b + the two opponent-aware blocks) already carries that
information another way, and the new column is a noisy repeat of it. A generalisation of
experiment 74's lesson: **another pipeline's ablation value does not transfer if your own
pipeline does not have the same gap.**

The single exception is instructive: chronology **hurt on its own (−0.0020) but gained with
the family specialists (+0.0040)**. Timing structure is family-specific — soft play's episode
and directed transfer's episode are laid out differently — and when the pooled model
compresses three families into one mould the signals cancel out. Without the architecture
change the feature is useless; without the feature the architecture change is neutral (0.6674).

## 14. 20 September: the specialist architecture reopens rejected blocks

Experiment 83+80's finding (chronology alone −0.0020, with specialists +0.0040) was not a
coincidence. The same pattern repeated in three blocks at once: **while the pooled model
compresses three families into one mould the signals cancel; split into family specialists,
each can use its own family's signature.**

| # | Configuration (all 3 seeds, baseline 0.66696) | Evidence CV |
|---|---|---|
| 86 | Specialist weight sweep: 0.50 / 0.75 / 0.90 / 1.00 | 0.6681 / **0.6710** / 0.6693 / 0.6671 — peak at 0.75 (the value the teammate had found) |
| 87 | Ranker capacity (never swept before): 7 leaves / 31 leaves / 800 rounds lr.015 / min_child 10 | 0.6653 / 0.6709 / 0.6690 / 0.6706 — the current 15 leaves/400 rounds is already the peak |
| 88 | Specialists + chronology + **within-pair normalisation** | **0.6716** |
| 89 | + **family-specific parameters** (dt 7 leaves/300 rounds, sp 31/500, iso 7/300) | 0.6712 (neutral on its own) |
| 90 | + the sizing block (rejected in experiment 74) | 0.6685; **with family parameters 0.6723** |
| 91 | + **direction consistency** (rejected in experiment 81) + family parameters | **0.6785** |
| 92 | + the value block | 0.6646 (rejected) |

**Experiment 91 is +0.0115 over the baseline.** Three separate REJECTED decisions (chronology
−0.0020, direction −0.0049, family parameters neutral) turn positive when used together. This
gives Rule 12.

> **Rule 12.** A block's value measured on its own is meaningless if the model has no capacity
> to use it. In a ranker that compresses the same feature into one function for three
> families, family-specific signals cancel each other. Before rejecting a block, make sure the
> architecture can express it.

| 93 | Softened specialists (`FAM_SMOOTH`): a specialist also sees the other families at a low weight (3× data) | 0.15 → 0.6732, 0.30 → 0.6718, 0.50 → 0.6700 (reference 0.6748) | — | — | REJECTED, monotonically falling — hard specialists are right |
| 94 | **BEST CONFIGURATION, 6 seeds**: `RERANK_ROUTED=1 USE_CH=1 RERANK_PAIRNORM=1 USE_DIR=1 FAM_TUNE=1` | evidence CV **0.6748** (production baseline 0.6658, **+0.0090**); rank-5 recall 0.506 → **0.5265** | — | — | **CANDIDATE v56** |

### 94. A warning about seed noise
The configuration that read 0.6785 with 3 seeds reads 0.6748 with 6 seeds, and 0.6712 with a
different set of 3. After ~30 configurations have been swept, the highest reading is
systematically inflated. **The honest number is the 6-seed 0.6748**, and its LB equivalent is
`0.20 × 0.0090 = +0.0018` → expected ~0.9193.

Component ablation (3 seeds, from the best configuration's 0.6785): removing family
parameters 0.6732, removing chronology 0.6735, removing within-pair normalisation 0.6697. All
three are necessary; none is valuable on its own.

### The v56 file
`run_v56.sh`. The risk and behaviour columns are **byte-identical** to v55 (the change is on
the evidence side only, a clean isolation). 74.7% of the first evidence hands are the same,
top-5 set overlap 4.47/5. `06_validate_submission` OK.
Note: the evaluation pass now works in buckets of pairs (`EVAL_BUCKETS`, default 6) — with
~70 new columns a single pass no longer fits in memory and was being killed by the OOM killer.

### 95-97. Disciplined verification with three seed sets
After sweeping ~30 configurations, the highest reading from a single seed set is not
trustworthy. From here on every decision was measured on **three independent 6-seed sets**
(A=42,7,2024,11,99,5 · B=3,13,23,33,43,53 · C=101,202,303,404,505,606) and averaged.

| 95 | **Family-specific capacity tuning** (leaf and round sweep for DT/SP/CI, 16 configurations) | best combination on seed set A 0.6805 (+0.0058) but on the **independent seed set B 0.6746 vs reference 0.6743 = +0.0003** | — | — | **REJECTED — the entire gain was selection noise** |
| 96 | **v56 candidate vs the v55 baseline, on three sets** | baseline 0.6658 / 0.6624 / 0.6610 (mean **0.6631**) → candidate 0.6748 / 0.6743 / 0.6730 (mean **0.6740**), **+0.009 to +0.012 on all three** | — | — | **REAL, +0.0110** |
| 97 | **Routing by the behaviour head's calibrated probabilities** (`BEH_ROUTE=1`): the router is now the pair model's OOF softmax instead of stage-1 family score shares | 0.6770 / 0.6766 / 0.6738 (mean **0.6758**) — **positive on all three**, +0.0018 | — | — | KEPT |
| 98 | Specialist weight again with the new router: 0.75 / 0.85 / 0.90 / 1.00 | mean 0.6758 / 0.6764 / **0.6768** / 0.6771 — flat; 1.00 discards the pooled model entirely, and on the worst set (C) 0.90 is better | — | — | **SPEC_W=0.90 chosen** |

### Final: v57
`run_v57.sh` — `RERANK_ROUTED=1 USE_CH=1 RERANK_PAIRNORM=1 USE_DIR=1 FAM_TUNE=1 BEH_ROUTE=1
SPEC_W=0.90`, 6 seeds. The mean of the three sets is **0.6768**, the v55 baseline 0.6631 →
**+0.0137 evidence CV** → `0.20 × 0.0137 = +0.0027` on the LB → expected **~0.9202**. The risk
and behaviour columns are byte-identical to v55; the change is on the evidence side only.

> **Rule 13.** The best reading out of a configuration sweep is not real until it repeats on a
> seed set that was not used in the sweep. In experiment 95 a setting that read +0.0058 came
> out at +0.0003 on an independent set; in experiment 96 a change that read +0.0110 repeated
> on all three sets. That difference is the whole decision.

| 99 | **LambdaRank truncation level, retried in the routed architecture** (experiment 47 had found no peak in the pooled one) | mean of three sets: truncation 5 → 0.6722, 10 → 0.6768, 15 → 0.6786, 20 → 0.6796, 30 → 0.6798, **50 → 0.6812** | — | — | **KEPT, truncation 50 (= no truncation)** |
| 100 | Number of seeds: 18 seeds in one run vs the mean of three 6-seed runs | 0.6794 vs 0.6796 — the same thing; and at truncation 50 the seed spread narrows to 0.6808-0.6815 | — | — | information: 6-12 seeds is enough |
| 101 | 100 candidates + truncation 100 | mean 0.6812 (0.6828/0.6799/0.6811) — no gain, wider spread | — | — | REJECTED, NCAND stays 50 |
| 102 | Specialist weight 1.0 at truncation 50 (the pooled model discarded entirely) | mean 0.6805 vs 0.6812 for 0.90 | — | — | REJECTED, SPEC_W stays 0.90 |

### 99. Why the truncation level matters now
Experiment 47 swept the same parameter in the pooled architecture and found it flat (10 →
0.6869, 20 → 0.6876, 35 → 0.6830). In the routed architecture it is monotone and strong:
0.6722 → 0.6812. The reason is Rule 12 again — the truncation level decides how deep into the
list the model tries to learn; a model compressing three families into one function cannot
learn the deep list anyway, so the parameter had no effect. Split into specialists, each can
learn its own family's full ordering, and the full list (truncation 50 = NCAND) is best.
**Same parameter, same data, different architecture — flat in one, +0.0090 in the other.**

## 15. Final candidate: v58
`run_v58.sh`: `RERANK_ROUTED=1 USE_CH=1 RERANK_PAIRNORM=1 USE_DIR=1 FAM_TUNE=1 BEH_ROUTE=1
SPEC_W=0.90 LR_TRUNC=50`, 12 seeds.

| | Evidence CV (mean of three sets) | LB |
|---|---|---|
| v45 (no opponent-aware block) | 0.6535 | 0.90858 |
| v55 (submitted) | **0.6631** | **0.91748** |
| v56 | 0.6740 | — |
| v57 | 0.6768 | — |
| **v58** | **0.6812** | predicted **~0.9211** |

v55 → v58 is **+0.0181 evidence CV**, worth `0.20 × 0.0181 = +0.0036` on the LB. The risk and
behaviour columns are byte-identical to v55; the whole change is on the evidence side, so the
prediction formula's constant (0.7843) stays valid this time and the prediction should be tight.

| 103 | **Routed risk model** (`RISK_ROUTE_TEST=1`, `src/05_pair_model.py`): the specialist idea from the evidence side applied to the pair model, which carries 70% of the metric — same negatives, three specialists whose positives are restricted to one family, blended by the behaviour OOF probabilities | 5 seeds: top-200 143.2 → **147.8** (positive on 5/5 seeds), median 256.4 → 254.2, but **top-500 362.6 → 360.6** and labelled AP 0.99884 → 0.99873 (lower on 5/5) | — | — | **REJECTED** |
| 104 | Seed averaging in the pair model (against run-to-run instability) | top-200 144 vs the single-seed mean 143.2 — no difference (experiment 46 confirmed) | — | — | REJECTED |

### 103. Why what worked on the evidence side did not work on the pair side
The specialist architecture gave +0.018 in the evidence ranker; in the pair model its
direction is ambiguous: consistently better at the very top (top 200), consistently worse a
little lower (top 500), and slightly lower on labelled AP in five seeds out of five. The
difference is in the nature of the task: **evidence selection really is a different task per
family** (soft play's episode sits in a different place and takes a different shape from
directed transfer's), **risk scoring is not** — in all three families the same question is
asked: is this couple abnormal. Splitting into specialists here only divides the number of
positives each model sees by three.

Changing the column that carries 70% of the metric, for a signal whose direction is ambiguous
and whose only reliable metric is saturated, is a bad bet. **The pair side stays as it is in
v55.**

### Note: the pair model is not fully deterministic between runs
`05_pair_model.py` run twice with the same command produces a different risk column (largest
difference 0.084; 98% of behaviour predictions identical). For that reason
`submission_v55_base.csv` must **never be regenerated**; v56/v57/v58 were all built on top of
that original file, so the risk and behaviour columns are exactly the ones measured at
0.91748 on the leaderboard.

## 16. The teammate's strategy documents (`docs/`, 22 md files) — read and measured

### The fourth-family defence: in production for them, unnecessary for us
`OTHER_COORDINATION_REVIEW.md` carries a **risk floor** in production:
`max(tail(model), tail(ps_top5) − 0.5)`. Its justification is measured: pair models that have
never seen a family collapse on that family, AP **0.958 → 0.113**, 0.916 → 0.031,
0.960 → 0.073; while the untrained `ps_top5` finds it at 0.833 / 0.503 / 0.791. We already
have the same statistic as `surp_top5`.

| 105 | **Unseen-family robustness, on our chain** (`GUARD_TEST=1`; leak-free 82 columns: only `oa_*`, `oc_*`, `G_*`, `rel_*`, `surp*`) | model AP on the held-out family: directed **0.5495**, soft play **0.3172**, isolation **0.2689** — about 5× the teammate's 0.113/0.031/0.073 | — | — | information: **we are already robust** |
| 106 | Risk floor (`max(pct(model), pct(surp_top5) − c)`), c=0.50 / 0.25 | changes exactly nothing — the model already ranks unseen-family positives above the floor | — | — | REJECTED (inert) |
| 107 | **Untrained statistics beat the model on the held-out family** | directed: `surp_top5` 0.665 (model 0.549); soft play: `oa_g_tail_sum` **0.705** (model 0.317); isolation: `surpmax_top5` **0.721** (model 0.269) — a different statistic in each family | — | — | information (very valuable) |
| 108 | A committee of five untrained statistics (mean percentile rank), blend weight w | an "expected PairAP" computed on separate scales said +0.0025/+0.0054/+0.0109 at w=0.1 | — | — | **MISLEADING — protocol error** |
| 109 | **Mixed AP, honest protocol** (`MIXAP_TEST=1`): the hidden family and the known families in **one ranking**; the risk model cross-validated without the held-out family, the hidden family's positives scaled down to a share f | f=0.05: w=0 **0.6726**, w=0.05 −0.0051, w=0.1 −0.0119, w=0.2 −0.0217 · f=0.10: −0.0032 / −0.0091 / −0.0176 · f=0.20: −0.0011 / −0.0050 / −0.0108 | — | — | **REJECTED — harmful at every value of f** |

### 108 → 109: adding scales together manufactures a false gain
Experiment 108's "expected PairAP" was a weighted sum of two APs computed against two
different populations: the known-family AP against 1,488 easy negatives, the unseen-family AP
against 155k unlabelled pairs. They are not on the same scale. Measured honestly in a single
ranking, the sign **reverses**. This is a new face of Rule 9.

> **Rule 14.** Do not manufacture an "expected metric" by taking a weighted sum of two separate
> APs. If the metric is defined over a single ranking, the simulation must be over a single
> ranking too — the populations must be ranked together.

### Official-scorer edge-case audit (`01_KOD_VE_OLCUM_DENETIMI.md` §A6)
The official scorer differs from the local one in four edge cases; the most dangerous is a
**repeated evidence hand within a pair = a submission error**. All three of
`submission_v55 / v58 / v58_amb` are clean: repeated evidence hands 0, `NO_EVIDENCE` 0, all
three families populated, risk finite and inside (0,1), 112,540 unique pairs, no nulls.

## 17. Taking aim at the isolation family — measured, closed

For the first time we measured where we stand per family on the evidence side
(`map5_by_family`, 6 seeds, seed set A):

| Family | pairs | v55 | v58 | difference |
|---|---|---|---|---|
| directed_transfer | 148 | 0.7004 | **0.7131** | +0.0127 |
| soft_play | 132 | 0.6483 | **0.6713** | +0.0230 |
| **coordinated_isolation** | 92 | 0.6353 | **0.6424** | +0.0071 |

Isolation is both the lowest and the least improved; at 25% weight, lifting it to the directed
level would be +0.017 total MAP (+0.0034 on the LB). The **only completely unmeasured** idea in
the teammate's documents (E3, `03_ONCELIKLI_DENEY_KARTLARI.md`) aims at exactly this.

| 110 | **Critical-transition block** (`src/19_isolation_transition.py`, 11 columns): what happens to the outside player when the pair applies pressure — how many withdrew while they still had a decision to make, how many bb they had put in that they cannot get back, and **whether the pressure the pair applies to each other stops once the outsider is gone** (no decision opportunity is **null**, not zero) | 1.05M rows, 6 s. Three seed sets: total **0.68111** (reference 0.68119). Isolation 0.6424 → ~0.6461 **but** soft play 0.6713 → 0.6673 | — | — | REJECTED (total neutral) |
| 111 | **Family-specific feature set** (`FAM_ONLY=coordinated_isolation:it_`): the block is given only to its own specialist, so it does not dilute the others | total **0.68142** (reference 0.68119), isolation 0.6448 | — | — | REJECTED (the difference is noise) |

### 110-111. The limit of Rule 12
Experiment 110 confirmed Rule 12 once more — the block lifted one family and lowered another
by just as much, so the dilution effect is real. But once the dilution is removed (111) the
remaining gain is at noise level. **Opening architectural capacity makes a signal visible; it
does not create a signal that is not there.** Isolation's low MAP is not a model constraint:
the generator plants ~18 hands per isolation pair and lists 5 of them (the teammate's
measurement), so the hands that cannot be selected really are indistinguishable.

| 112 | **Blend of the two architectures** (v58 routed + v55 pooled, within-pair rank blend; rank correlation 0.867) | w=0.1/0.2: 0.6811, w=0.3: 0.6809, w=0.5: 0.6729 — v58 itself is 0.6811 | — | — | REJECTED (diversity exists, contribution does not) |

## 18. End of 20 September — final state

**Submitted:** v55 = **0.91748**. **Ready candidate:** v58, evidence CV 0.6812 (v55: 0.6631),
predicted **~0.9205**.

26 experiments today, **one** accepted. What was accepted is not a feature but an
**architectural constraint**: a single pooled lambdarank had to learn one function for three
behaviour families. Four separate "REJECTED" decisions (chronology −0.0020, direction
consistency −0.0049, within-pair normalisation −0.0020, family parameters neutral) turned into
**+0.0137** once that constraint was lifted, and **+0.0181** together with the truncation level.

**Axes closed by measurement:** neural network/transformer/latent (the teammate's four
attempts), collusion rings (5.6σ real, no usable gain), phase contrast (real at AUC 0.48, no
gain), reordering the top five (24 rules, flat), the specialist architecture in the pair model
(direction ambiguous), the fourth-family risk floor (harmful at every f under an honest
protocol), the isolation critical-transition block (neutral), the blend of two architectures
(no contribution), family-specific capacity and truncation (noise), 100 candidates, softened
specialists.

### The day's three rules
> **Rule 12.** A block's value measured on its own is meaningless if the model has no capacity
> to use it. Before rejecting a block, make sure the architecture can express it.
>
> **Rule 13.** The best reading out of a sweep is not real until it repeats on a seed set that
> was not used in the sweep. (A setting that read +0.0058 came out at +0.0003 on an
> independent set.)
>
> **Rule 14.** Do not manufacture an "expected metric" by weighting two APs computed against
> two different populations; if the metric is defined over a single ranking, the simulation
> must be over a single ranking. (The sign reversed.)

### Submission plan (4 shared slots, close 20 September 22:00 UTC)
1. `submission_v58.csv` — a single isolated change (risk/behaviour bit-identical to v55),
   predicted +0.0036.
2. If v58 confirms, `submission_v58_amb.csv` — tests the fourth-family rule (487 pairs) at the
   new calibration.
3. The remaining two slots to the teammate; the quota is shared.
4. **The two finals must be ticked by hand** (the API has no such operation): the two highest
   public scores.

All three files passed the official-scorer edge-case audit: repeated evidence hands 0 (counted
as a submission error on the official side), `NO_EVIDENCE` 0, all three families populated,
risk finite, 112,540 unique pairs.

## 19. Reading the last submissions — the team record is now 0.91834

Today's three submissions form a clean single-variable experiment: in all three the risk and
behaviour columns are our v55's, **byte-identical**; only the evidence column changes.

| Time (UTC) | Who | Evidence column | LB |
|---|---|---|---|
| 01:55 | B | B's evidence engine v1 | 0.91233 |
| 00:01 | A | **our v55 evidence** | **0.91748** |
| 12:58 | B | B's evidence engine v2 (noisy-OR/LSE action pooling) | **0.91834** |

Since the evidence weight is 0.20: B's best evidence engine is only **+0.004 MAP** ahead of our
v55. Our v58 is **+0.018 MAP** ahead of v55 (confirmed on three seed sets) → v58 ≈ **0.9211**,
**+0.0028** above the current team record. That is no longer a formula prediction; it rests on
a chain measured on the leaderboard.

| 113 | **Noisy-OR and log-sum-exp action pooling** (`src/04c_action_model.py`, 5 columns: `act_nor`, `act_nor_l`, `act_lse10/25/50`) — the method named in the explanation of B's v1→v2 jump (+0.006 LB) | A/B on the same retrained model, three seed sets: without pooling **0.68119**, with **0.67944** (0.6792 / 0.6828 / 0.6763) | — | — | **REJECTED (−0.0018, and unstable)** |

### 113. Why +0.006 for them and −0.002 for us
Our action model already reaches the hand level through seven separate routes: `act_max`,
`act_2nd`, `act_n05`, `act_mean`, `act_sum`, `act_max_pre/post`, `act_max_bactive`. Noisy-OR
brings no new reading on top of those; `act_lse*` converges in practice to `τ·log(n)` at low
scores, i.e. a noisy repeat of the action count. This is the third repetition of the lesson of
experiment 74 (bet sizing) and experiment 80 (chronology): **another chain's gain does not
transfer if your own chain does not have the same gap.**

**Side finding:** the A/B control arm reproduced the old reference to 12 digits
(0.6807803166069296 vs 0.6807803166069294), so `04c_action_model.py` is fully deterministic and
v58 is reproducible. (The only non-deterministic step is `05_pair_model.py`.)

## 20. The graph-theory axis — third and final attempt, closed

The graph axis was tried today on three separate objects (two here, two more in the teammate's
repository):

| # | Object | Result |
|---|---|---|
| 79 | **Player graph** (`src/16_graph_pair.py`): hub, exclusivity, triangle support | labelled AP 0.99856 → 0.99842; universe median 278 → 271 — REJECTED |
| 114 | **Victim graph** (`src/20_victim_graph.py`): the pair–outsider bipartite graph, 11 columns | below — REJECTED |
| (B) | `80_player_graph_features.py` (OOF-risk-weighted player graph) | cleaned AP 0.9693 → 0.9693 — REJECTED |
| (B) | `300_relational_graph_residual.py` (two-way median-polish residual over 400 closed pools, star penalty, triangle and common-neighbour support) | "no column beats its null at any of the three bandwidths... the entire apparent gain is leakage, the file was not written" |

### 114. The victim graph (`src/20_victim_graph.py`)
Edge weight: the bb lost by the outside player in hands where the pair is jointly in profit.
9.77M hands, 5.33M pair–victim edges, 7 seconds. Two questions: **concentration** (is it always
the same victim) and **sharing** (is that victim also drained by the *other* partners of the
pair's members — the only statement in the dataset that needs three players).

| Column | AUC (1,860 labelled pairs) |
|---|---|
| `vg_drained_bb` (total drained) | 0.6220 |
| `vg_n_victims` | 0.5663 |
| `vg_hhi` (concentration) | **0.4249** — inverted |
| `vg_top_share` | **0.4229** — inverted |
| `vg_shared_p` / `vg_shared_q` (sharing) | 0.5209 / 0.5627 |
| `vg_nshare_max` | 0.5284 |

In the pair model (3 seeds): labelled AP 0.99862 → **0.99848**, top-200 139.3 → 137.3, median
~275 (unchanged).

**Two things are instructive.** First, the only meaningful column (`vg_drained_bb`) is not a
graph statistic: it says "this pair makes a lot of money", which the model already knows.
Second, concentration came out **inverted** — colluding pairs spread their winnings over *more*
victims, they do not return to a single mark. The opposite of the hypothesis. The only
genuinely three-player statement (sharing) reads AUC 0.50-0.56, i.e. chance.

### Final verdict on graph theory
The structure is **real but carries no information**: 49 players in more than one positive pair
(null 25.4 ± 4.2, **5.6σ**), **zero closed triangles**, and all 372 known positives are already
in the top 1000 with the base features. The reason is simple: if X colludes with both Y and Z,
both pairs show the behavioural signature directly; a graph would only add prior information
where the direct evidence is weak, and the 2× rate there does not move AP. **The generator
plants collusion pairwise; there is no group structure to discover.**

## 21. Closing: the late v58 submission = 0.92121

| 115 | **v58 (late submission)**: routed family specialists (pooled 0.10 + three specialists 0.90, routed by the behaviour head's calibrated probabilities), truncation 50, 12 seeds, plus contact chronology + within-pair normalisation + leave-one-out direction consistency. Risk and behaviour columns bit-identical to v55 | evidence CV 0.6631 → **0.6812** (on three independent 6-seed sets) | **0.92121** | **+0.00373** | KEPT |

### The prediction was exact for the first time
`0.91748 + 0.20 × 0.0181 = 0.92110`, actual **0.92121**, residual **+0.00011**.
The previous best residual was inside ±0.0018; v55's +0.0099 had been five times the band.
The difference is the design itself: v58 changed only the evidence column and kept the
`0.7P + 0.1B` term bit-identical, so the formula's constant (0.7843) stayed valid. **The
prediction formula works exactly when you isolate the term you change, and does not work when
you change two terms at once.** v55's large residual was not a mystery, it was a violation of
that rule.

### The cost in standings
On the competition's closing table the team (Brothers) is **15th with 0.91834**. Had v58's
0.92121 been submitted in time we would have been **10th** (10th is 0.92090, 11th 0.92060).
**Five places lost to a file that was not submitted.** The file had been ready and validated
~13 hours before the close; it was waiting for submission approval.

| | public |
|---|---|
| 1st | 0.94068 |
| 3rd (the top-three boundary) | 0.93894 |
| 10th | 0.92090 |
| **v58 (late)** | **0.92121** |
| 15th Brothers (official) | 0.91834 |
| v55 (our best in-time submission) | 0.91748 |

### The day's balance sheet
~35 experiments, **one** accepted. What was accepted is not a feature but the removal of an
architectural constraint: a single pooled lambdarank had to learn one function for three
behaviour families. Four blocks that had each been stamped REJECTED separately (chronology
−0.0020, direction consistency −0.0049, within-pair normalisation −0.0020, family parameters
neutral) gave +0.0137 together once that constraint was lifted, and **+0.0181** with the
truncation level — and it showed up on the leaderboard as +0.00373, exactly as predicted.

Axes closed by measurement: graph theory (three objects, all zero), neural
network/transformer/latent, phase contrast, reordering the top five, the specialist
architecture in the pair model, the fourth-family risk floor, the isolation critical-transition
block, noisy-OR/LSE pooling, the blend of two architectures, family-specific capacity and
truncation, the victim graph, 100 candidates, softened specialists.

## 22. Applying the handover plan (`docs/research_20260918/04_AGENT_DEVIR_PLANI.md`)

The document is a **process plan**: verification packages (A–E), a timetable, an experiment
template, acceptance criteria for the final production. It contains no model recipe to apply
directly; two of its items are actionable.

### Package A — verification (the document's "first job")
Item 3 of §7 is a critical warning: on their side, generating folds from different table lists
with the same seed is inconsistent (the fold changes for 199 of 245 positive tables) and the
fix lowered the evidence OOF **0.6930 → 0.6817**, i.e. their measurements had been +0.011
inflated.

**We do not have that problem structurally.** All five stages use `tidx % NF` — a deterministic
function of the table index, with no difference between them: `04_hand_scorer.py:41`,
`04b_stage1b.py:56`, `04c_action_model.py:105`, `05_pair_model.py:127`,
`05b_evidence_reranker.py:154`. Since all of a pair's hands are at one table, no pair crosses a
fold boundary. All stage-1 scores are produced out of fold (04's docstring, 04b:114-124,
04c:191). **Today's numbers are sound in this respect.**

### Experiment card E6 — matched fake-partner null calibration
The card's never-measured idea. `src/21_matched_null.py`: for every directed pair (A→B) a null
is built from A's *other* opponents, but **matched on exposure** — only opponents whose shared
decision count is within a 1.5× band enter the pool, because the spread of the statistic
narrows with exposure and an opponent seen over 300 hands is not a comparable reading to one
seen over 30. Output: an empirical p-value, MAD-scaled distance from the pool centre, and the
difference to the best opponent. 7.07M matched comparisons, 2.4% empty pools. (The difference
from `src/16_graph_pair.py`: there was no matching there.)

| 116 | **E6 matched null** (`src/21_matched_null.py`, 24 columns) | best alone `mn_lead_oa_surp_excess_hi` AUC 0.9874 — but **below** the baseline `oa_g_tail_sum` at 0.9932. In the pair model (3 seeds): labelled AP 0.99862 → **0.99826**, top-200 139.3 → **134.0** | — | — | **REJECTED** (the card's own kill rule: "if it reproduces the existing LLR, drop it") |

### The second late submission: v58_amb
§4 says the second selection should be "a different recipe only if there is a measured
robustness/diversity justification". The only genuinely different recipe we have is
`submission_v58_amb.csv`, which changes the behaviour column.

| 117 | **v58_amb (late submission)**: v58 + the fourth-family rule — inside the top 1500 of the risk ranking, the **487 pairs** whose top behaviour-class probability is below 0.60 were relabelled `other_coordination`. The risk column and the five evidence columns are bit-identical to v58, so only the behaviour column is isolated | — | **0.92197** | **+0.00076** (over v58) | **KEPT — final best** |

### 117. The rule pays at the v55 calibration and at v58 too
At the v55 calibration the same rule gave +0.0013 with 326 pairs (experiment 60); at v58 it
gives +0.00076 with 487 pairs. Since the behaviour weight is 0.10, that is +0.0076 in
BehaviorMAP. The rule still stands after the evidence side improved by 0.0181 — so the
fourth-family signal is independent of evidence quality.

## 23. Final table

| Version | What changed | LB |
|---|---|---|
| v45 | no opponent-aware block | 0.90858 |
| v55 | both opponent-aware blocks (our best in-time submission) | 0.91748 |
| v58 | routed family specialists + three blocks + truncation 50 (late) | 0.92121 |
| **v58_amb** | **+ the fourth-family rule (late)** | **0.92197** |

Today's total gain is **+0.00449** (v55 → v58_amb). On the official closing table the team is
**15th** with 0.91834; had 0.92197 been submitted in time we would have been **10th** (10th
place 0.92090, 9th 0.92501). The top-three boundary was 0.93894 and was not reachable with
today's findings.
