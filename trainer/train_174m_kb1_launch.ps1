# PIVOT after the KB architecture lever FAILED its SPRT (kb8ob8 = -88 Elo vs prod,
# despite better val loss — undertrained king buckets w/o factorizer + ~20% NPS
# cost + quant noise). Refocus on the SPRT-VALIDATED DATA lever: train the PRODUCTION
# architecture (KB=1, OB=1, HL512, SCReLU, lam 0.9 — same recipe as king_nnue.bin) on
# the full 174M SF-relabeled corpus. Isolates the data increase (112M -> 174M).
# Data scaling has been real Elo every round (77M->112M = +59.6 SPRT). Outputs a temp
# net + samples; SPRT vs production decides promotion.
$ErrorActionPreference = "SilentlyContinue"
Set-Location C:\Users\nikod\Documents\uni\chess

while ($true) {
  if ((Test-Path trainer\preprocess_174m.log) -and `
      (Select-String -Path trainer\preprocess_174m.log -Pattern '\[preproc\] Done:' -Quiet)) { break }
  Start-Sleep -Seconds 20
}
Write-Output "[launcher] preprocess done -> training KB=1 (prod arch) on 174M"

py trainer\train_nnue.py --cache data\all3_sf.bin --out nets\king_174m.bin `
   --samples trainer\s_174m.txt --hl 512 --activation screlu --lam 0.9 `
   --epochs 45 --batch 16384 *> trainer\king_174m.log

Write-Output "[launcher] KB=1 174M training DONE -> nets\king_174m.bin"
