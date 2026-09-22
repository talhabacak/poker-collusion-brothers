#!/bin/bash
cd /home/tarik/Projects/kaggle-comp/detect-suspicious-value-transfers-in-poker
export USE_CT=0 USE_VAL=0 RERANK_SEEDS=42,7,2024
echo "=== referans (OAH yok): 0.6535"
echo "=== OAH acik"
USE_OAH=1 RERANK_IN=submission_v52_base.csv RERANK_OUT=/tmp/oah_test.csv .venv/bin/python src/05b_evidence_reranker.py 2>&1 | grep -E "CV |Error" | cut -c1-320
echo OAH_DONE
