# selfplay.ps1 -- Run a cutechess-cli match between two UCI engines via WSL.
#
# Usage:
#   .\tools\selfplay.ps1 [-EngineA <wsl-path>] [-EngineB <wsl-path>] `
#                        [-Games <N>] [-TC <time-control>]
#
# Parameters:
#   -EngineA   WSL path to engine A binary  (default: /tmp/kb/engine)
#   -EngineB   WSL path to engine B binary  (default: /tmp/kb/engine)
#   -Games     Total number of games        (default: 100)
#   -TC        Time control in cutechess format, e.g. "8+0.08" or "4+0.04"
#              (default: "8+0.08")
#
# Example -- baseline self-play sanity check (expect ~50%):
#   .\tools\selfplay.ps1 -EngineA /tmp/kb/engine -EngineB /tmp/kb/engine `
#                        -Games 20 -TC "4+0.04"
#
# cutechess-cli is at ~/cc/cutechess-cli inside WSL Ubuntu-24.04.
# The PGN output is written to /tmp/sp.pgn inside WSL.

param(
    [string]$EngineA = "/tmp/kb/engine",
    [string]$EngineB = "/tmp/kb/engine",
    [int]$Games      = 100,
    [string]$TC      = "8+0.08"
)

# -games 2 -rounds (N/2) -repeat plays exactly N games:
#   each round has 2 games (repeat swaps colors), so rounds = Games/2.
$Rounds = [math]::Max(1, [int]($Games / 2))

$CutechessCmd = @"
~/cc/cutechess-cli/cutechess-cli \
  -engine cmd=$EngineA proto=uci name=A \
  -engine cmd=$EngineB proto=uci name=B \
  -each tc=$TC \
  -games 2 -rounds $Rounds -repeat \
  -recover -concurrency 2 \
  -pgnout /tmp/sp.pgn \
  -ratinginterval 20
"@

Write-Host "Running $Games games ($Rounds rounds) at TC=$TC ..."
Write-Host "  Engine A: $EngineA"
Write-Host "  Engine B: $EngineB"
Write-Host ""

# Run via WSL and capture output, printing each line as it arrives
$Output = wsl -d Ubuntu-24.04 bash -c $CutechessCmd 2>&1

foreach ($line in $Output) {
    Write-Host $line
}

Write-Host ""
Write-Host "=== Summary ==="
# Print the final score line and Elo difference from cutechess output
$ScoreLines = @($Output | Where-Object { $_ -match "^Score of" })
if ($ScoreLines.Count -gt 0) { Write-Host $ScoreLines[-1] }
$Output | Where-Object { $_ -match "^Elo diff" } | Select-Object -Last 1 | ForEach-Object { Write-Host $_ }
