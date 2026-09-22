#!/bin/bash
cd /home/tarik/Projects/kaggle-comp/detect-suspicious-value-transfers-in-poker
export U_WEIGHT=0.0 MIXED_NEG_W=0 USE_VAL=0 USE_CT=0 USE_CTR=0
echo "=== baseline (clean chain, no OA)"; grep -E "u_weight=0.0" logs/cl3_05.txt | head -1 | cut -c1-160
for C in "" "oa_g_tail_sum,oa_surp_lo,oa_surp_excess_hi,oa_g_hi,oa_fold_z_lo"; do
  L=$([ -z "$C" ] && echo "ALL 37 columns" || echo "best 5 columns")
  echo "=== OA: $L"
  USE_OA=1 OA_COLS="$C" SUB_OUT=/tmp/oa_sub.csv .venv/bin/python src/05_pair_model.py 2>&1 | grep -E "u_weight=0.0|Behavior MAP labeled" | cut -c1-165
done
echo OA_TEST_DONE
