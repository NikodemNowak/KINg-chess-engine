Set-Location C:\Users\nikod\Documents\uni\chess
python trainer\lichess_to_text.py "\\wsl.localhost\Ubuntu-24.04\home\nikodem\data\lichess_eval.jsonl.zst" data\lichess_quiet.txt 12 50000000
if ($LASTEXITCODE -ne 0) { "CONVERT FAILED" | Out-Host; exit 1 }
Remove-Item data\lichess_quiet.bin -Force -ErrorAction SilentlyContinue
python trainer\preprocess_nnue.py --data data\lichess_quiet.txt --out data\lichess_quiet.bin
if ($LASTEXITCODE -ne 0) { "PREPROCESS FAILED" | Out-Host; exit 1 }
"=== preprocess OK, training ===" | Out-Host
python trainer\train_nnue.py --cache data\lichess_quiet.bin --out nets\king_lichessq_hl512.bin --samples trainer\s_lichessq_hl512.txt --hl 512 --activation screlu --lam 1.0
if ($LASTEXITCODE -ne 0) { "TRAIN FAILED" | Out-Host; exit 1 }
"=== LICHESS QUIET NET DONE ===" | Out-Host
