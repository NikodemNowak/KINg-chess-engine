# Train the scaled-up final net on the merged ~77M SF-labeled corpus.
Set-Location C:\Users\nikod\Documents\uni\chess
$wsl = "\\wsl.localhost\Ubuntu-24.04\home\nikodem\king"

Write-Host "=== preprocess all_sf (~77M) ==="
Remove-Item data\all_sf.bin -Force -ErrorAction SilentlyContinue
python trainer\preprocess_nnue.py --data "$wsl\all_sf.txt" --out data\all_sf.bin
if ($LASTEXITCODE -ne 0) { "PREPROCESS FAILED" | Out-Host; exit 1 }

Write-Host "=== train net_allsf (HL=512 SCReLU lam1.0 — lam1.0 won the A/B by +17 Elo) ==="
python trainer\train_nnue.py --cache data\all_sf.bin --out nets\king_allsf_hl512.bin --samples trainer\s_allsf.txt --hl 512 --activation screlu --lam 1.0
if ($LASTEXITCODE -ne 0) { "TRAIN FAILED" | Out-Host; exit 1 }

Write-Host "=== ALL_SF NET DONE -> nets\king_allsf_hl512.bin ==="
