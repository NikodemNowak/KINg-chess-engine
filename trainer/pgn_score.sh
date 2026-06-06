#!/usr/bin/env bash
# Proper per-engine score: engine1 = first [White] name in the PGN (cutechess passes
# engine1 first -> it is White in game 1). Scores engine1 across all games accounting
# for color alternation, then prints score%, +/- and Elo (with ~95% CI).
cd /home/nikodem/king || exit 1
python3 - "$@" <<'PY'
import sys, glob, math, re
files = sys.argv[1:] or sorted(glob.glob("m_sprt_SE_*.pgn") + ["m_corrhistvsbest.pgn"])
for path in files:
    try:
        txt = open(path, encoding="utf-8", errors="ignore").read()
    except FileNotFoundError:
        continue
    games = re.split(r'(?=\[Event )', txt)
    e1 = None
    w = d = l = 0
    for g in games:
        mw = re.search(r'\[White "([^"]+)"\]', g)
        mb = re.search(r'\[Black "([^"]+)"\]', g)
        mr = re.search(r'\[Result "([^"]+)"\]', g)
        if not (mw and mb and mr): continue
        if e1 is None: e1 = mw.group(1)
        white, black, res = mw.group(1), mb.group(1), mr.group(1)
        # score from engine1 POV
        if res == "1/2-1/2": d += 1
        elif res == "1-0":  (lambda x: None)(0); w += 1 if white == e1 else 0; l += 1 if white != e1 else 0
        elif res == "0-1":  w += 1 if black == e1 else 0; l += 1 if black != e1 else 0
    n = w + d + l
    if n == 0: continue
    score = (w + 0.5 * d) / n
    # Elo + CI
    if 0 < score < 1:
        elo = -400 * math.log10(1/score - 1)
    else:
        elo = float('inf') if score == 1 else float('-inf')
    # stderr of score
    p = score
    var = (w*(1-p)**2 + d*(0.5-p)**2 + l*(0-p)**2) / n
    se = math.sqrt(var/n)
    # elo CI via score +/- 1.96 se
    def e(s):
        s = min(max(s,1e-9),1-1e-9); return -400*math.log10(1/s-1)
    lo, hi = e(score-1.96*se), e(score+1.96*se)
    e2 = re.search(r'\[Black "([^"]+)"\]', games[1] if len(games)>1 else "")
    opp = ""
    print(f"{path:30s} {e1:14s} N={n:4d}  +{w}-{l}={d}  {100*score:5.1f}%  Elo {elo:+6.1f}  [{lo:+.0f}, {hi:+.0f}]")
PY
