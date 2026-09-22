#!/bin/bash
cd /home/tarik/Projects/kaggle-comp/detect-suspicious-value-transfers-in-poker
set -o pipefail
export USE_CT=0 USE_VAL=0 USE_OAH=1 RERANK_SEEDS=42,7,2024,11,99,5
RERANK_IN=submission_v52_base.csv RERANK_OUT=submission_v54.csv .venv/bin/python src/05b_evidence_reranker.py > logs/A2_05b.txt 2>&1 || { echo FAILA2; tail -4 logs/A2_05b.txt; exit 1; }
grep CV logs/A2_05b.txt | cut -c1-320
AMB_THR=0.60 AMB_TOPN=1500 .venv/bin/python src/10_ambiguous_family.py submission_v54.csv submission_A2_oa_both.csv | head -1
.venv/bin/python src/06_validate_submission.py submission_A2_oa_both.csv | tail -1
.venv/bin/python src/06_validate_submission.py submission_v54.csv | tail -1
echo A2_DONE
