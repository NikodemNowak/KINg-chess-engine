#!/usr/bin/env python3
"""sf_label.py -- relabel a KINg NNUE text dataset with Stockfish evaluations.

Reads lines "FEN | old_score_cp | result" and rewrites the score column with a
fresh Stockfish eval at a fixed node budget.  The FEN and result columns are kept
verbatim, so the position distribution (self-play) is preserved while the labels
are upgraded from the weak engine's own search to Stockfish's.

Stockfish reports `score cp` from the side-to-move POV; we convert to White POV
to match the trainer's "FEN | white_cp | result" convention.

Usage: python sf_label.py <sf_binary> <in_shard> <out_shard> [nodes]
Single shard, single SF process -- parallelism is achieved by launching many of
these on pre-split shards (see sf_label_run.sh).
"""
import sys, subprocess

SF    = sys.argv[1]
INP   = sys.argv[2]
OUTP  = sys.argv[3]
NODES = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
CLAMP = 2000

p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     text=True, bufsize=1)

def send(s):
    p.stdin.write(s + "\n"); p.stdin.flush()

def wait_for(tok):
    while True:
        ln = p.stdout.readline()
        if not ln or ln.startswith(tok):
            return

send("uci");            wait_for("uciok")
send("setoption name Threads value 1")
send("setoption name Hash value 16")
send("isready");        wait_for("readyok")

n = 0
fout = open(OUTP, "w", encoding="ascii", buffering=1 << 22)
with open(INP, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.split("|")
        if len(parts) != 3:
            continue
        fen = parts[0].strip()
        result = parts[2].strip()
        flds = fen.split()
        if len(flds) < 2:
            continue
        stm_black = (flds[1] == "b")

        send("position fen " + fen)
        send("go nodes %d" % NODES)
        score = None
        while True:
            ln = p.stdout.readline()
            if not ln or ln.startswith("bestmove"):
                break
            i = ln.find(" score ")
            if i >= 0:
                tok = ln[i + 7:].split()
                if tok[0] == "cp":
                    score = int(tok[1])
                elif tok[0] == "mate":
                    m = int(tok[1]); score = CLAMP if m > 0 else -CLAMP
        if score is None:
            continue
        if stm_black:
            score = -score                       # STM POV -> White POV
        if score > CLAMP:  score = CLAMP
        if score < -CLAMP: score = -CLAMP
        fout.write("%s | %d | %s\n" % (fen, score, result))
        n += 1

fout.close()
send("quit")
print("[sf_label] %s -> %s : %d positions" % (INP, OUTP, n), flush=True)
