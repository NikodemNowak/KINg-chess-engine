#!/usr/bin/env python3
# Validate the current SPSA theta: play v-tune (with the tuned params set via UCI
# options) vs v-best (defaults) at SLOW TC, write a PGN. Score with pgn_score.sh.
# This is the REAL test of whether SPSA's exploration is an improvement.
import json, subprocess, sys

K   = "/home/nikodem/king"
CC  = "/home/nikodem/cc/cutechess-cli/cutechess-cli"
ENG = f"{K}/v-tune/engine"
BOOK= f"{K}/book.epd"
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 600
CONC  = int(sys.argv[2]) if len(sys.argv) > 2 else 16

theta = json.load(open(f"{K}/spsa_theta.json"))["theta"]
opts = [f"option.{p}={int(round(v))}" for p, v in theta.items()]
print("tuned params:", " ".join(opts), flush=True)

cmd = [CC,
    "-engine", f"cmd={ENG}", "name=spsa", "proto=uci", *opts,
    "-engine", f"cmd={K}/v-best/engine", "name=vbest", "proto=uci",
    "-each", "tc=40+0.4", "option.Hash=128", "option.Threads=1", "option.Move Overhead=30",
    "-openings", f"file={BOOK}", "format=epd", "order=random",
    "-repeat", "-games", "2", "-rounds", str(GAMES // 2),
    "-draw", "movenumber=40", "movecount=8", "score=10",
    "-resign", "movecount=4", "score=700",
    "-concurrency", str(CONC), "-recover", "-ratinginterval", "50",
    "-pgnout", f"{K}/m_spsavsbest.pgn"]
subprocess.run(cmd)
print("DONE -> m_spsavsbest.pgn", flush=True)
