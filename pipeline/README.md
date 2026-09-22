# The pipeline

Every stage lives in `scripts/`, the shared library in `src/pokercol/`, its unit tests
in `tests/`. `run_all.py --list` prints this same list with the state of each output;
what follows is the map, with what each stage writes and why it is here.

The numbering is the project's own stage ids, kept because every measurement in
`docs/`, in the write-up and in the run manifests refers to stages by number. Two
chains were developed in parallel during the competition and merged into this one
tree, which is why the numbers restart: `01_prep` and `01_pair_universe` are the first
stage of each. The run order below, not the file name, is what matters.

## base — raw logs to v55's three columns

*Written by Tarık Bacak, independently of the risk chain below; see*
*[`../docs/TWO_CHAINS.md`](../docs/TWO_CHAINS.md).*

These files are taken from commit `823dce0`, the commit that built v55 — not from
the working copy used for the earlier cross-engine measurements, whose pair model
predates the `USE_OC` block and would silently ignore the opponent-conditioned
residuals. `05_pair_risk_behaviour.py` is that commit's `src/05_pair_model.py`,
renamed because the risk chain also has a `05_pair_model.py`; nothing else about it
changed. The flags every stage runs under are in [`../docs/run_A3.sh`](../docs/run_A3.sh),
the author's own script, included here verbatim.

| stage | writes |
|---|---|
| `01_prep.py` | ids, preflop equity, postflop strength, action context, player style |
| `02_policy.py` | population action policy and bet-size model; out-of-fold surprisal per action |
| `03_pairhand.py` | every (pair, shared hand) row, conditioned on the partner being live |
| `03b_relational.py` | relational likelihood ratio against the player's own baseline |
| `03c_extra.py` | partner-independent features: third-player folds, multiway aggression |
| `13_oppaware.py` | opponent-aware policy residuals per directed pair |
| `15_oppcond_policy.py` | the opponent-conditioned policy and its residuals |
| `04_hand_scorer.py` | 4-class evidence-hand scorer, five folds by table |
| `04c_action_model.py` | action-level collusion model, one head per family |
| `04b_stage1b.py` | within-pair evidence ranker on pair-normalised features |
| `05_pair_risk_behaviour.py` | the pair risk column and the behaviour column |
| `05b_evidence_reranker.py` | lambdarank evidence reranker, six seeds → `tarik_v55_best.csv` |
| `06_validate_submission.py` | this chain's own submission validator |

`05b_mil.py` is `05b_evidence_reranker.py` with hooks: it can read an external block of
columns (`MIL_BLOCK`), hold a fold out (`HOLDOUT_FOLD`) and dump its out-of-fold frame,
which is how stage 533 measures a candidate evidence block inside this engine, in its
folds, against a column-matched placebo. Run with none of its hooks set it reproduced
the plain reranker to the last digit (0.66085, maximum absolute difference 0.0 against
the original out-of-fold dump) — the check that makes anything measured through it
comparable with anything measured without it.

`01_prep.py` carries one structural change from its original, marked in its docstring:
the multiprocessing workers and the card tables stay at module level and the rest moves
under an `if __name__ == "__main__"` guard, because Windows spawns pool workers instead
of forking. No feature, seed, filter or constant is touched.

## risk — the caches behind v64's risk column

*Written by Talha Bacak. The second final is the only thing that uses this
column; the primary final does not.*

| stage | writes |
|---|---|
| `00b_preflop_equity.py` | all-in equity of the 169 starting hands |
| `01_pair_universe.py` | co-seating universe and shared-hand counts |
| `02_seat_features.py` | per-seat strength by street, actions, fold attribution |
| `02b_action_surprise.py` | action policy model and −log P(observed) per action |
| `03_pair_hands.py` | pair × shared-hand interactions, 30M rows |
| `04_pair_features.py` | pair features, partner-versus-field improbability |
| `04c_surprise_pair.py` | directed surprise contrasts |
| `045_hand_suspicion.py` | hand suspicion bag features (run twice: bootstrap, then real) |
| `045b_freeze_exclusion.py` | the frozen exclusion set the surrogate is scored on |
| `20_family_hand_suspicion.py` | per-family hand suspicion |
| `04e_relational.py` | relational likelihood ratios over 96 decision contexts |
| `49_dyadic_residual.py` | dyadic residual features |
| `150_opponent_aware_policy.py` | opponent-aware residuals, full and no-opponent-style |
| `156_generate_v64_union_oppaware.py` | **v64's risk column** |

## mil — the action-level MIL head, and the primary final

*Written by Talha Bacak and run inside the base chain's tables and folds - that
is the whole point of these stages: a mechanism measured where the development
metric transfers. `05b_mil.py` above is Tarık's reranker with the hooks they need.*

| stage | writes |
|---|---|
| `531_b_action_bag_frame.py` | action bags for every candidate hand, in the base chain's tables and folds |
| `532_b_noisy_or_mil.py` | pooling choice, strict-holdout protocol, and the eight nested MIL columns |
| `533_b_mil_evidence_arm.py` | the seed-bagged block measured inside the reranker |
| `534_b_mil_candidate.py` | **the primary final**: v55 with the MIL evidence column |
| `608_build_hybrid_riskA.py` | **the second final**: that file with v64's risk column |
| `08_validate_submission.py` | format, vocabulary and evidence-hand checks |
| `530_b_engine_reproduction.py` | the check that the base chain still reads what it read |

## Modules that never run

Thirteen files in `scripts/` are never executed as a stage and are still required:
production stages import them at module level for shared helpers, and Python will not
import a stage without them.

    04e_relational, 150            -> 25_hidden_info_residual
    25_hidden_info_residual        -> 11_evidence_experiments, 13_pseudo_labels,
                                      14_pair_retest, 181_evidence_oracle_audit
    045b_freeze_exclusion          -> 14_pair_retest
    156                            -> 05_pair_model            (its feature_columns)
    532, 534                       -> 481_noisy_or_mil
    481_noisy_or_mil               -> 480_action_bag_frame
    480_action_bag_frame           -> 06_evidence_model, 119_iso_pairwise_production_check
    119_iso_pairwise_production_check -> 34_other_routing -> 27_leave_family_out
    530_b_engine_reproduction      -> 474_within_pair_ceiling

That coupling is a wart of the competition repository, not of the method. It is listed
here so a reviewer can tell at a glance which files are the pipeline and which are
imported scaffolding, without having to grep for it.
