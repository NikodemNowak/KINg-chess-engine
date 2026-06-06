#!/usr/bin/env python3
"""Tally a cutechess PGN from the first-listed engine's POV.

Usage: tally.py <pgn> [candName]
Prints W-L-D, score, Elo (+/- 95% CI), and LOS for `candName` (default "cand").
"""
import sys, math, re

pgn = sys.argv[1]
cand = sys.argv[2] if len(sys.argv) > 2 else "cand"

white = None
w = l = d = 0
for line in open(pgn, encoding="utf-8", errors="ignore"):
    if line.startswith("[White "):
        m = re.search(r'"(.*)"', line); white = m.group(1) if m else None
    elif line.startswith("[Result "):
        m = re.search(r'"(.*)"', line); r = m.group(1) if m else "*"
        if r == "1/2-1/2":
            d += 1
        elif r == "1-0":
            (w if white == cand else l).__class__  # noop
            if white == cand: w += 1
            else: l += 1
        elif r == "0-1":
            if white == cand: l += 1
            else: w += 1

n = w + l + d
if n == 0:
    print("no games"); sys.exit(0)
score = (w + 0.5 * d) / n
if 0 < score < 1:
    elo = -400 * math.log10(1 / score - 1)
else:
    elo = float("inf") if score == 1 else float("-inf")
# Approx 95% CI on Elo via score stderr.
mu = score
var = (w * (1 - mu) ** 2 + l * (0 - mu) ** 2 + d * (0.5 - mu) ** 2) / n
se = math.sqrt(var / n) if n else 0
def s2e(s):
    s = min(max(s, 1e-9), 1 - 1e-9); return -400 * math.log10(1 / s - 1)
ci = (s2e(mu + 1.96 * se) - s2e(mu - 1.96 * se)) / 2 if 0 < mu < 1 else float("nan")
los = 0.5 * (1 + math.erf((mu - 0.5) / (math.sqrt(2) * se))) * 100 if se > 0 else (100.0 if mu > 0.5 else 0.0)
print(f"{cand}:  W:{w} L:{l} D:{d}  n={n}  score={score:.3f}  "
      f"Elo={elo:+.1f} +/- {ci:.1f}  LOS={los:.1f}%")
