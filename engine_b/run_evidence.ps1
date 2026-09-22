# Stage 440/444: the evidence half of the teammate's clean chain (04b stage-1b, then the
# stage-2 reranker), so their development out-of-fold hand ordering exists to fuse against
# ours. RERANK_SEEDS is the six-seed set run_clean.sh exports. EVAL_BEHAVIOR is their own
# shipped v45 file, which is exactly the eval-side behaviour column 04b expects.
#
# NOTE: stdout and stderr are redirected to SEPARATE files via Start-Process. PowerShell 5.1
# turns any native-executable stderr write into a NativeCommandError, so `*>` plus
# ErrorActionPreference=Stop kills the chain on a harmless DeprecationWarning - which is
# exactly what happened to the first run of 05_pair_model.
$run = "C:\Users\TALHAB~1\AppData\Local\Temp\claude\c--Users-Talha-Bacak-Desktop-work-repo-detect-suspicious-value-transfers-in-poker\b98735ec-0cf8-4563-a8b7-de75c3f004c7\scratchpad\tarikrun"
Set-Location $run
$env:PYTHONUNBUFFERED = "1"
$env:USE_VAL = "0"; $env:USE_CT = "0"; $env:USE_CTR = "0"
$env:FAMILY_SOURCE = "model"
$env:EVAL_BEHAVIOR = "tarik_v45_best.csv"
$env:RERANK_SEEDS = "42,7,2024,11,99,5"

Write-Output "=== START 04b_stage1b $(Get-Date -Format o)"
$p = Start-Process -FilePath "python" -ArgumentList "src/04b_stage1b.py" -WorkingDirectory $run -WindowStyle Hidden -RedirectStandardOutput "$run\logs\04b_stage1b.log" -RedirectStandardError "$run\logs\04b_stage1b.err" -PassThru -Wait
if ($p.ExitCode -ne 0) { Write-Output "=== FAILED 04b exit $($p.ExitCode)"; Get-Content "$run\logs\04b_stage1b.err" -Tail 20; exit 1 }
Get-Content "$run\logs\04b_stage1b.log" -Tail 3

$env:RERANK_IN = "submission_repro_base.csv"
$env:RERANK_OUT = "submission_repro_v2.csv"
$env:RERANK_DUMP = "repro"
Write-Output "=== START 05b_evidence_reranker $(Get-Date -Format o)"
$q = Start-Process -FilePath "python" -ArgumentList "src/05b_evidence_reranker.py" -WorkingDirectory $run -WindowStyle Hidden -RedirectStandardOutput "$run\logs\05b_reranker.log" -RedirectStandardError "$run\logs\05b_reranker.err" -PassThru -Wait
if ($q.ExitCode -ne 0) { Write-Output "=== FAILED 05b exit $($q.ExitCode)"; Get-Content "$run\logs\05b_reranker.err" -Tail 20; exit 1 }
Get-Content "$run\logs\05b_reranker.log" -Tail 4
Write-Output "=== EVIDENCE_COMPLETE"
