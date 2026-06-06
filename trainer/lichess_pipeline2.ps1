Set-Location C:\Users\nikod\Documents\uni\chess
python trainer\preprocess_nnue.py --data data\lichess_eval.txt --out data\lichess_eval.bin
if ($LASTEXITCODE -ne 0) { "PREPROCESS FAILED exit=$LASTEXITCODE" | Out-Host; exit 1 }
"=== preprocess OK, training (lambda=1, pure SF eval) ===" | Out-Host
python trainer\train_nnue.py --cache data\lichess_eval.bin --out nets\king_lichess_hl512.bin --samples trainer\s_lichess_hl512.txt --hl 512 --activation screlu --lam 1.0
if ($LASTEXITCODE -ne 0) { "TRAIN FAILED exit=$LASTEXITCODE" | Out-Host; exit 1 }
"=== LICHESS NET DONE ===" | Out-Host
