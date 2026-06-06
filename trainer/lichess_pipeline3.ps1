Set-Location C:\Users\nikod\Documents\uni\chess
Remove-Item data\lichess_eval.bin -Force -ErrorAction SilentlyContinue
python trainer\preprocess_nnue.py --data data\lichess_eval.txt --out data\lichess_eval.bin
if ($LASTEXITCODE -ne 0) { "PREPROCESS FAILED exit=$LASTEXITCODE" | Out-Host; exit 1 }
$recs = [math]::Floor((Get-Item data\lichess_eval.bin).Length/72)
"=== preprocess OK: $recs records, training (lambda=1) ===" | Out-Host
python trainer\train_nnue.py --cache data\lichess_eval.bin --out nets\king_lichess_hl512.bin --samples trainer\s_lichess_hl512.txt --hl 512 --activation screlu --lam 1.0
if ($LASTEXITCODE -ne 0) { "TRAIN FAILED exit=$LASTEXITCODE" | Out-Host; exit 1 }
"=== LICHESS NET DONE ===" | Out-Host
