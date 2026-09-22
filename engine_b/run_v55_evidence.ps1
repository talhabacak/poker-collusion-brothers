# Stage 448: rebuild the EVIDENCE side of their v55 exactly as run_A3.sh specifies
#   export USE_OAH=1 USE_OCH=1 RERANK_SEEDS=42,7,2024,11,99,5
# plus the two hand-level blocks it needs (src/13, src/15). The pair model is not rerun:
# 05b's RERANK_IN only supplies the eval-side risk/behaviour columns it copies through,
# and the development reranker CV this stage reads does not touch them.
#
# The baseline is rerun under the SAME (newer) 05b so both arms come from one code path;
# the only difference between the two arms is USE_OAH/USE_OCH.
#
# stdout and stderr go to SEPARATE files: PowerShell 5.1 turns any native stderr write
# into a NativeCommandError, which killed the first chain run on a DeprecationWarning.
$run = "C:\Users\TALHAB~1\AppData\Local\Temp\claude\c--Users-Talha-Bacak-Desktop-work-repo-detect-suspicious-value-transfers-in-poker\b98735ec-0cf8-4563-a8b7-de75c3f004c7\scratchpad\tarikrun"
Set-Location $run
$env:PYTHONUNBUFFERED = "1"
$env:USE_VAL = "0"; $env:USE_CT = "0"; $env:USE_CTR = "0"
$env:RERANK_SEEDS = "42,7,2024,11,99,5"
$env:RERANK_IN = "submission_repro_base.csv"

function Step($name, $script, $log) {
  Write-Output "=== START $name $(Get-Date -Format o)"
  $p = Start-Process -FilePath "python" -ArgumentList $script -WorkingDirectory $run -WindowStyle Hidden `
       -RedirectStandardOutput "$run\logs\$log.log" -RedirectStandardError "$run\logs\$log.err" -PassThru -Wait
  if ($p.ExitCode -ne 0) { Write-Output "=== FAILED $name exit $($p.ExitCode)"; Get-Content "$run\logs\$log.err" -Tail 20; exit 1 }
  Get-Content "$run\logs\$log.log" -Tail 2
  Write-Output "=== DONE $name $(Get-Date -Format o)"
}

if (-not (Test-Path "$run\data\interim\oppaware_hand.parquet")) { Step "13_oppaware" "src/13_oppaware.py" "13_oppaware" }
if (-not (Test-Path "$run\data\interim\oppcond_hand.parquet")) { Step "15_oppcond" "src/15_oppcond_policy.py" "15_oppcond" }

# arm 1: baseline (v45 evidence configuration) under the new 05b
$env:USE_OAH = "0"; $env:USE_OCH = "0"
$env:RERANK_OUT = "submission_basenew_v2.csv"; $env:RERANK_DUMP = "basenew"
Step "05b baseline (OAH=0 OCH=0)" "src/05b_evidence_reranker.py" "05b_basenew"

# arm 2: the v55 evidence side
$env:USE_OAH = "1"; $env:USE_OCH = "1"
$env:RERANK_OUT = "submission_v55ev_v2.csv"; $env:RERANK_DUMP = "v55ev"
Step "05b v55 evidence (OAH=1 OCH=1)" "src/05b_evidence_reranker.py" "05b_v55ev"

Write-Output "=== V55_EVIDENCE_COMPLETE"
