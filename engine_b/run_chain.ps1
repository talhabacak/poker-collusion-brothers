# Stage 440: the teammate's clean chain (their run_clean.sh configuration), Windows.
# U_WEIGHT=0 MIXED_NEG_W=0 as their README documents; USE_VAL/USE_CT/USE_CTR=0 as run_clean.sh
# exports them (the value and collusion-table artefact directories are not part of the run
# order, so the defaults of 1 would crash on an empty glob).
# 04c keeps its defaults MIL_ROUNDS=1 MIL_KEEP=2, which is MIL OFF - their experiment 50
# retired MIL as a leak and run_clean.sh names this setting "MIL off".
$ErrorActionPreference = "Stop"
$run = "C:\Users\TALHAB~1\AppData\Local\Temp\claude\c--Users-Talha-Bacak-Desktop-work-repo-detect-suspicious-value-transfers-in-poker\b98735ec-0cf8-4563-a8b7-de75c3f004c7\scratchpad\tarikrun"
Set-Location $run
$env:PYTHONUNBUFFERED = "1"
$env:U_WEIGHT = "0.0"; $env:MIXED_NEG_W = "0"; $env:USE_VAL = "0"; $env:USE_CT = "0"; $env:USE_CTR = "0"
$env:DUMP_DEV = "1"; $env:DUMP_ALL = "1"; $env:BEH_LOFO = "1"; $env:SUB_OUT = "submission_repro_base.csv"

foreach ($step in @("03_pairhand", "03b_relational", "03c_extra", "04_hand_scorer", "04c_action_model", "05_pair_model")) {
  Write-Output "=== START $step $(Get-Date -Format o)"
  & python "src/$step.py" *> "logs/$step.log"
  if ($LASTEXITCODE -ne 0) { Write-Output "=== FAILED $step exit $LASTEXITCODE"; Get-Content "logs/$step.log" -Tail 25; exit 1 }
  Write-Output "=== DONE $step $(Get-Date -Format o)"
  Get-Content "logs/$step.log" -Tail 3
}
Write-Output "=== CHAIN_COMPLETE"
