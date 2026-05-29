#!/usr/bin/env python3
"""preprocess_nnue.py -- convert KINg NNUE text dataset to a compact binary cache.

Reads lines of the form:
    <FEN> | <score_cp_white> | <result_white_pov>

and writes a numpy memmap file (.bin) with one fixed-size 72-byte record per
position.  The binary cache can then be streamed by the DataLoader in
train_nnue.py without ever loading the whole dataset into RAM.

Record layout (72 bytes, little-endian):
    score   : int16    -- clamped STM-POV centipawn score (clamp ±2000)
    result  : float16  -- STM-POV result in {0, 0.5, 1}
    stm     : uint8    -- side-to-move: 0=White, 1=Black
    n       : uint8    -- number of pieces on board (≤ 32)
    pieces  : uint8[32] -- (color<<4)|ptype for each piece; padding with 0
    squares : uint8[32] -- square index (0=a1..63=h8) for each piece; padding with 0
    _pad    : uint8[2]  -- alignment padding

The feature-index computation is deferred to training time so this file has
NO dependency on the HL/model architecture.

Usage:
    py trainer/preprocess_nnue.py --data data/nnue_big.txt --out data/nnue_big.bin
    py trainer/preprocess_nnue.py --data trainer/smoke_big.txt --out trainer/smoke_big.bin
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── record dtype ──────────────────────────────────────────────────────────────
RECORD_DTYPE = np.dtype([
    ('score',   '<i2'),
    ('result',  '<f2'),
    ('stm',     'u1'),
    ('n',       'u1'),
    ('pieces',  'u1', (32,)),
    ('squares', 'u1', (32,)),
    ('_pad',    'u1', (2,)),
])
assert RECORD_DTYPE.itemsize == 72, RECORD_DTYPE.itemsize

WHITE, BLACK = 0, 1
PT_FROM_CHAR = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
SCORE_CLAMP = 2000


def parse_fen_pieces(fen_board: str):
    """Yield (color, ptype, square) for every piece.  Identical logic to train_nnue.py."""
    ranks = fen_board.split('/')
    pieces = []
    for ri, row in enumerate(ranks):
        rank = 7 - ri
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                color = WHITE if ch.isupper() else BLACK
                ptype = PT_FROM_CHAR[ch.lower()]
                sq = rank * 8 + file
                pieces.append((color, ptype, sq))
                file += 1
    return pieces


def preprocess(data_path: str, out_path: str, limit: int | None = None,
               chunk: int = 500_000) -> int:
    """Convert text dataset at *data_path* to binary cache at *out_path*.

    Returns the number of positions written.
    Two-pass approach is not needed — we pre-allocate a large memmap and
    truncate at the end.  The pre-allocation is ``limit`` (or 35_000_000 for
    the full dataset) records.
    """
    n_alloc = limit if (limit and limit < 35_000_000) else 35_000_000
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Pre-allocate raw binary memmap (NO numpy .npy header — pure record stream
    # so the reader can open it with np.memmap(..., dtype=RECORD_DTYPE) directly).
    mm = np.memmap(str(out), mode='w+', dtype=RECORD_DTYPE, shape=(n_alloc,))

    t0 = time.time()
    n = 0
    bad = 0
    buf: list[tuple] = []

    def flush():
        nonlocal n
        # Build a numpy record array for the chunk, then write in one shot.
        chunk_size = len(buf)
        if chunk_size == 0:
            return
        arr = np.zeros(chunk_size, dtype=RECORD_DTYPE)
        for j, (score_i16, result_f16, stm_u8, piece_bytes, square_bytes) in enumerate(buf):
            arr[j]['score'] = score_i16
            arr[j]['result'] = result_f16
            arr[j]['stm'] = stm_u8
            k = len(piece_bytes)
            arr[j]['n'] = k
            arr[j]['pieces'][:k] = piece_bytes
            arr[j]['squares'][:k] = square_bytes
            # _pad and remaining pieces/squares stay 0 (from np.zeros)
        mm[n:n + chunk_size] = arr
        n += chunk_size
        buf.clear()

    with open(data_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) != 3:
                bad += 1
                continue
            fen = parts[0].strip()
            try:
                score_w = float(parts[1].strip())
                result_w = float(parts[2].strip())
            except ValueError:
                bad += 1
                continue
            fen_fields = fen.split()
            if len(fen_fields) < 2:
                bad += 1
                continue
            board = fen_fields[0]
            stm = WHITE if fen_fields[1] == 'w' else BLACK

            pieces = parse_fen_pieces(board)
            if not pieces:
                bad += 1
                continue

            # Convert to STM POV.
            score = -score_w if stm == BLACK else score_w
            result = (1.0 - result_w) if stm == BLACK else result_w
            score_clamped = max(-SCORE_CLAMP, min(SCORE_CLAMP, score))

            piece_bytes = [(c << 4) | pt for (c, pt, _) in pieces]
            square_bytes = [sq for (_, _, sq) in pieces]

            buf.append((
                np.int16(score_clamped),
                np.float16(result),
                np.uint8(stm),
                piece_bytes,
                square_bytes,
            ))

            if len(buf) >= chunk:
                flush()
                if n % 500_000 == 0:
                    elapsed = time.time() - t0
                    rate = n / max(elapsed, 1e-6)
                    print(f"[preproc] {n:,} positions  {rate/1e3:.1f}k/s  "
                          f"{elapsed:.0f}s elapsed", flush=True)

            if limit and n >= limit:
                break

    flush()  # remaining

    mm.flush()
    del mm

    # Truncate the file to the actual number of records.
    final_size = n * RECORD_DTYPE.itemsize
    with open(out_path, 'r+b') as fh:
        fh.truncate(final_size)

    elapsed = time.time() - t0
    rate = n / max(elapsed, 0.001)
    print(f"[preproc] Done: {n:,} positions ({bad} skipped)  "
          f"{elapsed:.1f}s  {rate/1e3:.1f}k/s  "
          f"output {final_size/1e6:.1f} MB -> {out_path}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Input text dataset')
    ap.add_argument('--out', required=True, help='Output binary cache (.bin)')
    ap.add_argument('--limit', type=int, default=None, help='Cap positions (debug)')
    args = ap.parse_args()

    if Path(args.out).exists():
        print(f"[preproc] Cache already exists: {args.out}  (delete to re-run)")
        sys.exit(0)

    preprocess(args.data, args.out, args.limit)


if __name__ == '__main__':
    sys.exit(main())
