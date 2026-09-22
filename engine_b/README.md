# Engine B

The pipeline that produced `tarik_v55_best.csv` — the risk, behaviour and candidate
columns the team's primary final carries, and the reranker the MIL evidence column
was built inside. It is the work of the second member of team Brothers; `src/` is
their code and `src/README.md` is their own run order, both as received.

Two things about this copy are worth stating plainly.

**It is the Windows copy.** `src/01_prep.py` differs from its author's original in one
structural respect, marked in its docstring: the two multiprocessing worker functions
and the card tables stay at module level and the rest moves into `main()` under an
`if __name__ == "__main__"` guard, because Windows spawns pool workers instead of
forking and would otherwise re-execute the whole script in every child. No feature,
seed, filter or numeric constant is touched.

**`src/05b_mil.py` is the reranker with hooks.** It is `05b_evidence_reranker.py`
with the ability to read an external block of columns (`MIL_BLOCK`), hold out a fold
(`HOLDOUT_FOLD`) and dump its out-of-fold frame, so that engine A's stage 533 can
measure a candidate evidence block inside this engine, in this engine's folds,
against a column-matched placebo. Run with none of its hooks set it reproduced the
plain reranker's development reading to the last digit (0.66085, maximum absolute
difference 0.0 against the original out-of-fold dump), which is the check that makes
anything measured through it comparable with anything measured without it.

## Running it

`run_all.py --with-engine-b` runs the v55 configuration in order. Directly, from the
repository root, it is:

    python engine_b/src/01_prep.py
    python engine_b/src/02_policy.py
    python engine_b/src/03_pairhand.py
    python engine_b/src/03b_relational.py
    python engine_b/src/03c_extra.py
    python engine_b/src/13_oppaware.py
    python engine_b/src/15_oppcond_policy.py
    python engine_b/src/04_hand_scorer.py
    python engine_b/src/04c_action_model.py           # MIL refinement off: defaults
    FAMILY_SOURCE=model python engine_b/src/04b_stage1b.py
    U_WEIGHT=0.0 MIXED_NEG_W=0 USE_OA=1 USE_VAL=0 USE_CT=0 USE_CTR=0 \
        SUB_OUT=submission_v55_pairs.csv python engine_b/src/05_pair_model.py
    USE_OAH=1 USE_OCH=1 RERANK_SEEDS=42,7,2024,11,99,5 \
        RERANK_IN=submission_v55_pairs.csv RERANK_OUT=submissions/tarik_v55_best.csv \
        python engine_b/src/05b_evidence_reranker.py
    python engine_b/src/06_validate_submission.py submissions/tarik_v55_best.csv

The fourth-family rule (`08_fourth_family.py`) is **not** part of v55; it is in the
tree because v45 used it and because the late v58 experiments did.

`src/02_policy_v2.py`, `src/02b_bb_attribution.py`, `src/03d_value.py`,
`src/04d_token_model.py`, `src/07_infoshare.py`, `src/09*`, `src/10`, `src/11`,
`src/12` and `src/14` are experiments that were measured and rejected. They are kept
so the run order can be read against what was tried, and none of them runs.

See the repository README, "What is and is not rebuilt here", for what was verified
on this machine and what was not.
