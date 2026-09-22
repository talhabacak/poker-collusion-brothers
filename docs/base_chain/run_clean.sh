#!/bin/bash
# Rebuild the v21 chain: no MIL (rule 8), no collusion table (LB-neutral), no value features.
# This is the only configuration whose offline evidence metric has been shown to transfer to the LB.
cd /home/tarik/Projects/kaggle-comp/detect-suspicious-value-transfers-in-poker
set -o pipefail
export U_WEIGHT=0.0 MIXED_NEG_W=0 USE_VAL=0 USE_CT=0 USE_CTR=0 RERANK_SEEDS=42,7,2024,11,99,5
echo "=== 04 hand scorer"; .venv/bin/python src/04_hand_scorer.py > logs/cl_04.txt 2>&1 || { echo FAILED04; exit 1; }
grep "Evidence MAP" logs/cl_04.txt | cut -c1-200
echo "=== 04c action model (MIL off)"; MIL_ROUNDS=1 MIL_KEEP=2 .venv/bin/python src/04c_action_model.py > logs/cl_04c.txt 2>&1 || { echo FAILED04c; exit 1; }
grep "CV (positive" logs/cl_04c.txt | cut -c1-200
echo "=== 04b stage1b"; FAMILY_SOURCE=model EVAL_BEHAVIOR=submission_v31_base.csv .venv/bin/python src/04b_stage1b.py > logs/cl_04b.txt 2>&1 || { echo FAILED04b; exit 1; }
grep CV logs/cl_04b.txt | cut -c1-200
echo "=== 05 pair model"; SUB_OUT=submission_v40_base.csv .venv/bin/python src/05_pair_model.py > logs/cl_05.txt 2>&1 || { echo FAILED05; exit 1; }
grep -E "u_weight=0.0|Behavior MAP labeled" logs/cl_05.txt | cut -c1-200
echo "=== 05b reranker"; RERANK_IN=submission_v40_base.csv RERANK_OUT=submission_v40.csv .venv/bin/python src/05b_evidence_reranker.py > logs/cl_05b.txt 2>&1 || { echo FAILED05b; exit 1; }
grep CV logs/cl_05b.txt | cut -c1-200
.venv/bin/python src/06_validate_submission.py submission_v40.csv | tail -1
echo CLEAN_DONE
