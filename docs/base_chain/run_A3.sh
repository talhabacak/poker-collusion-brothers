#!/bin/bash
cd /home/tarik/Projects/kaggle-comp/detect-suspicious-value-transfers-in-poker
set -o pipefail
export U_WEIGHT=0.0 MIXED_NEG_W=0 USE_VAL=0 USE_CT=0 USE_CTR=0 USE_OA=1 USE_OC=1 OA_COLS=""
export USE_OAH=1 USE_OCH=1 RERANK_SEEDS=42,7,2024,11,99,5
echo "=== 05 (both blocks)"
SUB_OUT=submission_v55_base.csv .venv/bin/python src/05_pair_model.py > logs/A3_05.txt 2>&1 || { echo FAIL05; tail -4 logs/A3_05.txt; exit 1; }
grep -E "u_weight=0.0|Behavior MAP labeled" logs/A3_05.txt | cut -c1-165
echo "=== 05b (both hand blocks, 6 seeds)"
RERANK_IN=submission_v55_base.csv RERANK_OUT=submission_v55.csv .venv/bin/python src/05b_evidence_reranker.py > logs/A3_05b.txt 2>&1 || { echo FAIL05B; tail -4 logs/A3_05b.txt; exit 1; }
grep CV logs/A3_05b.txt | cut -c1-330
AMB_THR=0.60 AMB_TOPN=1500 .venv/bin/python src/10_ambiguous_family.py submission_v55.csv submission_A3_both_blocks.csv | head -1
.venv/bin/python src/06_validate_submission.py submission_A3_both_blocks.csv | tail -1
.venv/bin/python src/06_validate_submission.py submission_v55.csv | tail -1
echo A3_DONE
