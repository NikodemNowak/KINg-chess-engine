# Auto-launch the 174M training the moment the preprocess finishes (minimise GPU
# idle). Trains the best architecture (kb8ob8 — best 112M val loss) on the full
# 174M SF-relabeled corpus. Outputs a temp net + temp samples (never clobbers
# production king_nnue.bin / nnue_samples.txt). The 174M net is the strongest
# candidate (best arch + most data) — SPRT vs production decides promotion.
$ErrorActionPreference = "SilentlyContinue"
Set-Location C:\Users\nikod\Documents\uni\chess

while ($true) {
  if ((Test-Path trainer\preprocess_174m.log) -and `
      (Select-String -Path trainer\preprocess_174m.log -Pattern '\[preproc\] Done:' -Quiet)) { break }
  Start-Sleep -Seconds 20
}
Write-Output "[launcher] preprocess done -> training kb8ob8-174M on GPU"

py trainer\train_nnue.py --cache data\all3_sf.bin --out nets\king_kb8ob8_174m.bin `
   --samples trainer\s_kb8ob8_174m.txt --hl 512 --activation screlu --lam 0.9 `
   --kbuckets 8 --buckets 8 --epochs 45 --batch 16384 *> trainer\kb8ob8_174m.log

Write-Output "[launcher] kb8ob8-174M training DONE -> nets\king_kb8ob8_174m.bin"
