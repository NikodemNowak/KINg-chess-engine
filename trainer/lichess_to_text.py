#!/usr/bin/env python3
"""Convert the Lichess evaluations database (lichess_db_eval.jsonl.zst) to the
KINg trainer's text format: "FEN | score_cp_white_pov | result".

Lichess evals are Stockfish centipawns from WHITE's POV (verified against startpos).
There is no game result, so we emit a dummy result of 0.5 and train with --lam 1.0
(pure eval target). Mate scores map to +/-MATE_CP (clamped to the trainer's range).

Usage: python lichess_to_text.py <src.jsonl.zst> <out.txt> [min_depth] [limit]
"""
import sys, json, io
import zstandard

SRC = sys.argv[1]
OUT = sys.argv[2]
MIN_DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 12
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # 0 = no limit
MATE_CP = 2000   # trainer clamps score to +/-2000

n_in = n_out = 0
dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
with open(SRC, 'rb') as fh, open(OUT, 'w', encoding='ascii', buffering=1 << 22) as out:
    reader = dctx.stream_reader(fh)
    text = io.TextIOWrapper(reader, encoding='utf-8')
    for line in text:
        n_in += 1
        try:
            o = json.loads(line)
        except Exception:
            continue
        fen = o.get('fen')
        evals = o.get('evals')
        if not fen or not evals:
            continue
        # Deepest available analysis for this position.
        best = max(evals, key=lambda e: e.get('depth', 0))
        if best.get('depth', 0) < MIN_DEPTH:
            continue
        pvs = best.get('pvs')
        if not pvs:
            continue
        pv0 = pvs[0]
        cp = pv0.get('cp')
        if cp is None:
            m = pv0.get('mate')
            if m is None:
                continue
            cp = MATE_CP if m > 0 else -MATE_CP
        else:
            cp = MATE_CP if cp > MATE_CP else (-MATE_CP if cp < -MATE_CP else int(cp))

        # ── Quiet filter (CRITICAL for NNUE) ─────────────────────────────────
        # NNUE is a static evaluator; it must be trained on QUIET positions.
        # Skip positions whose best move is a capture or promotion (the eval then
        # reflects an imminent tactic, not a static assessment). Capture = the
        # PV's first move lands on an occupied square (en-passant is rare, ignored).
        mv = pv0.get('line', '').split(' ', 1)[0]
        if len(mv) < 4:
            continue
        if len(mv) >= 5:          # promotion
            continue
        tsq = (ord(mv[3]) - 49) * 8 + (ord(mv[2]) - 97)   # destination square 0..63
        occ = set()
        rr, cc = 7, 0
        for ch in fen.split(' ', 1)[0]:
            if ch == '/':
                rr -= 1; cc = 0
            elif ch.isdigit():
                cc += int(ch)
            else:
                occ.add(rr * 8 + cc); cc += 1
        if tsq in occ:            # best move is a capture -> non-quiet
            continue

        out.write(f"{fen} | {cp} | 0.5\n")
        n_out += 1
        if n_out <= 5:
            print(f"[sample] {fen}  cp={cp}", flush=True)
        if n_out % 2000000 == 0:
            print(f"[conv] {n_out:,} written / {n_in:,} read", flush=True)
        if LIMIT and n_out >= LIMIT:
            break
print(f"[done] {n_out:,} positions written / {n_in:,} read -> {OUT}", flush=True)
