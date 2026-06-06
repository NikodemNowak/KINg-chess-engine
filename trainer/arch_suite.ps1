# Keep the idle GPU busy through the relabel window: after kb4-112M finishes,
# train two more NNUE architecture candidates on the same 112M corpus —
#   kb8     : 8 king buckets (more king-relative resolution)
#   kb4ob8  : 4 king buckets + 8 output buckets (combined orthogonal levers)
# All share the production recipe (HL512, SCReLU, lam 0.9, 45 epochs) so each is a
# clean A/B vs the production net. Outputs temp nets + temp samples (never clobbers
# king_nnue.bin / nnue_samples.txt).
$ErrorActionPreference = "SilentlyContinue"
Set-Location C:\Users\nikod\Documents\uni\chess

function Wait-Done($log) {
  while ($true) {
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern '\[done\] peak RAM' -Quiet)) { return }
    Start-Sleep -Seconds 30
  }
}

Wait-Done "trainer\kb4_112m.log"
Write-Output "[arch_suite] kb4 done -> training kb8-112M"
py trainer\train_nnue.py --cache data\all2_sf.bin --out nets\king_kb8_112m.bin `
   --samples trainer\s_kb8_112m.txt --hl 512 --activation screlu --lam 0.9 `
   --kbuckets 8 --epochs 45 --batch 16384 *> trainer\kb8_112m.log

Write-Output "[arch_suite] kb8 done -> training kb4ob8-112M"
py trainer\train_nnue.py --cache data\all2_sf.bin --out nets\king_kb4ob8_112m.bin `
   --samples trainer\s_kb4ob8_112m.txt --hl 512 --activation screlu --lam 0.9 `
   --kbuckets 4 --buckets 8 --epochs 45 --batch 16384 *> trainer\kb4ob8_112m.log

Write-Output "[arch_suite] all architecture candidates trained"
