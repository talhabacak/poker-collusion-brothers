# Pipeline (policy-surprisal approach)

Setup (Python 3.12):
    uv venv .venv -p 3.12 && uv pip install -p .venv/bin/python -r requirements.txt

Full run order (the pipeline as submitted). Every step is deterministic with seed 42.

    .venv/bin/python src/01_prep.py               # ids, preflop equity, postflop strength, action context
    .venv/bin/python src/02_policy.py             # population policy + bet-size model, OOF surprisal
    .venv/bin/python src/03_pairhand.py           # pair-hand features (partner-active conditioning)
    .venv/bin/python src/03b_relational.py        # relational likelihood ratio vs the player's own baseline
    .venv/bin/python src/03c_extra.py             # partner-independent features (third-player folds, multiway aggression)
    .venv/bin/python src/04_hand_scorer.py        # 4-class hand scorer, 5 folds by table
    MIL_ROUNDS=4 MIL_KEEP=1 .venv/bin/python src/04c_action_model.py   # action-level model with multiple-instance refinement
    FAMILY_SOURCE=model .venv/bin/python src/04b_stage1b.py            # within-pair, per-family evidence ranker
    U_WEIGHT=0 MIXED_NEG_W=0 .venv/bin/python src/05_pair_model.py     # pair risk + behaviour
    RERANK_SEEDS=42,7,2024,11,99,5 .venv/bin/python src/05b_evidence_reranker.py   # evidence reranker -> submission
    .venv/bin/python src/08_fourth_family.py <in.csv> <out.csv>        # other_coordination rule (LOFO calibrated)
    .venv/bin/python src/06_validate_submission.py <out.csv>

Optional / diagnostic: `src/02_policy_v2.py`, `src/02b_bb_attribution.py`, `src/03d_value.py`, `src/04d_token_model.py`,
`src/07_infoshare.py` are experiments that were measured and rejected (see `EXPERIMENTS.md`); they are not part of the run.

Original short form, in order (total ~35 min on 16 cores / 32 GB):
    .venv/bin/python src/01_prep.py            # ids, preflop equity, postflop strength, action context, player style stats (~1 min)
    .venv/bin/python src/02_policy.py          # population policy + bet-size model, 2-fold by table, OOF surprisal (~19 min)
    .venv/bin/python src/03_pairhand.py        # features for all 30M (pair, shared hand) rows (~1 min)
    .venv/bin/python src/04_hand_scorer.py     # supervised evidence scorer, 5-fold by table, scores all rows (~12 min)
    .venv/bin/python src/05_pair_model.py      # pair risk + behavior models, local metrics, submission.csv (~20 s)
    .venv/bin/python src/05b_evidence_reranker.py   # stage-2 evidence reranker with within-pair chronology -> submission_v2.csv (~30 s)
    .venv/bin/python src/06_validate_submission.py submission_v2.csv

All seeds fixed (42). CV folds are by table (pool), so no pair or player leaks across folds.
