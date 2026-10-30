# Waits until the Kaggle competition rules are accepted, downloads both Optiver datasets,
# then runs the two Optiver experiments. Safe to re-run: finished steps are skipped.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$raw = Join-Path $root "data\raw"
$logs = Join-Path $root "logs"
$closeCsv = Join-Path $raw "optiver_close\train.csv"
$rvpDir = Join-Path $raw "optiver_rvp"
$rvpZip = Join-Path $rvpDir "optiver-realized-volatility-prediction.zip"
$deadline = (Get-Date).AddHours(48)

while ((Get-Date) -lt $deadline) {
    if (-not (Test-Path $closeCsv)) {
        kaggle competitions download -c optiver-trading-at-the-close -f train.csv -p (Join-Path $raw "optiver_close") 2>&1 | Out-Null
        $zip = Join-Path $raw "optiver_close\train.csv.zip"
        if (Test-Path $zip) { Expand-Archive -Force $zip (Join-Path $raw "optiver_close"); Remove-Item $zip }
    }
    if (-not (Test-Path (Join-Path $rvpDir "train.csv"))) {
        if (-not (Test-Path $rvpZip)) { kaggle competitions download -c optiver-realized-volatility-prediction -p $rvpDir 2>&1 | Out-Null }
        if (Test-Path $rvpZip) { python -m zipfile -e $rvpZip $rvpDir }
    }
    if ((Test-Path $closeCsv) -and (Test-Path (Join-Path $rvpDir "train.csv"))) { break }
    Start-Sleep -Seconds 60
}
"downloads finished: $(Get-Date)" | Out-File -Append (Join-Path $logs "optiver_status.log")
Set-Location $root
python -u -W ignore -m src.experiments.optiver_close *> (Join-Path $logs "optiver_close.log")
python -u -W ignore -m src.experiments.optiver_rvp *> (Join-Path $logs "optiver_rvp.log")
"experiments finished: $(Get-Date)" | Out-File -Append (Join-Path $logs "optiver_status.log")
