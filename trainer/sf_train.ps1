# Windows side of the SF-relabel A/B: preprocess + train net_sf and net_eng.
# Reads the WSL-resident text corpora directly; writes .bin to data\ and nets to nets\.
Set-Location C:\Users\nikod\Documents\uni\chess
$wsl = "\\wsl.localhost\Ubuntu-24.04\home\nikodem\king"

Write-Host "=== [1/4] preprocess SF-labeled (33M) ==="
Remove-Item data\extra_sf.bin -Force -ErrorAction SilentlyContinue
python trainer\preprocess_nnue.py --data "$wsl\nnue_extra_sf.txt" --out data\extra_sf.bin
if ($LASTEXITCODE -ne 0) { "PREPROCESS SF FAILED" | Out-Host; exit 1 }

Write-Host "=== [2/4] preprocess engine-labeled (33M, same positions) ==="
Remove-Item data\extra_eng.bin -Force -ErrorAction SilentlyContinue
python trainer\preprocess_nnue.py --data "$wsl\nnue_extra.txt" --out data\extra_eng.bin
if ($LASTEXITCODE -ne 0) { "PREPROCESS ENG FAILED" | Out-Host; exit 1 }

Write-Host "=== [3/4] train net_sf (HL=512 SCReLU lam0.9) ==="
python trainer\train_nnue.py --cache data\extra_sf.bin --out nets\king_extrasf_hl512.bin --samples trainer\s_extrasf.txt --hl 512 --activation screlu --lam 0.9
if ($LASTEXITCODE -ne 0) { "TRAIN SF FAILED" | Out-Host; exit 1 }

Write-Host "=== [4/4] train net_eng (same recipe) ==="
python trainer\train_nnue.py --cache data\extra_eng.bin --out nets\king_extraeng_hl512.bin --samples trainer\s_extraeng.txt --hl 512 --activation screlu --lam 0.9
if ($LASTEXITCODE -ne 0) { "TRAIN ENG FAILED" | Out-Host; exit 1 }

Write-Host "=== TRAIN DONE: nets\king_extrasf_hl512.bin + nets\king_extraeng_hl512.bin ==="
