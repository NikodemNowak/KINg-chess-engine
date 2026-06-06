Set-Location C:\Users\nikod\Documents\uni\chess
$ErrorActionPreference = "Stop"
# 1. Wait for the converter to finish
while (-not (Select-String -Path trainer\lichess_conv.log -Pattern '\[done\]' -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 20 }
"=== convert done, preprocessing ===" | Out-Host
# 2. Preprocess text -> bin
python trainer\preprocess_nnue.py --data data\lichess_eval.txt --out data\lichess_eval.bin
"=== preprocess done, training (lambda=1, pure SF eval) ===" | Out-Host
# 3. Train HL=512 on the Stockfish-labeled positions
python trainer\train_nnue.py --cache data\lichess_eval.bin --out nets\king_lichess_hl512.bin --samples trainer\s_lichess_hl512.txt --hl 512 --activation screlu --lam 1.0
"=== LICHESS NET TRAINED ===" | Out-Host
