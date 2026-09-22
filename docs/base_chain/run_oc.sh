#!/bin/bash
cd /home/tarik/Projects/kaggle-comp/detect-suspicious-value-transfers-in-poker
export USE_CT=0 USE_VAL=0 RERANK_SEEDS=42,7,2024
echo "=== referans: OAH tek basina (3 tohum) = 0.6608"
echo "=== OAH + OCH birlikte"
USE_OAH=1 USE_OCH=1 RERANK_IN=submission_v52_base.csv RERANK_OUT=/tmp/oc_test.csv .venv/bin/python src/05b_evidence_reranker.py 2>&1 | grep -E "CV |Error" | cut -c1-330
echo OC_DONE
