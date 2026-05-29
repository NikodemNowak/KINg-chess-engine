#!/usr/bin/env python3
"""KINg NNUE trainer.

Perspective net (768 -> 256)x2 -> 1, clipped-ReLU. Trains float weights with
PyTorch (GPU if available), then quantizes to integers and exports a
little-endian binary that the C++ inference reproduces bit-for-bit.

The architecture, feature indexing, quantization constants and binary layout
are documented in trainer/README_NNUE.md and MUST match the C++ side exactly.

Usage:
    py trainer/train_nnue.py --data data/nnue_data.txt \
        --out nets/king_nnue.bin --samples trainer/nnue_samples.txt

Run with --help for all options.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants (the contract). These are duplicated in README_NNUE.md and the C++.
# ──────────────────────────────────────────────────────────────────────────────
INPUT_SIZE = 768          # 2 colors * 6 piece types * 64 squares
HL = 256                  # hidden / accumulator size per perspective
QA = 255                  # accumulator (W1,b1) quantization scale
QB = 64                   # output weight (W2) quantization scale
SCALE = 400               # eval scale (cp), == sigmoid divisor used in training
MAGIC = 0x4B4E5545        # "KNUE" big-endian spelled; stored little-endian u32

WHITE, BLACK = 0, 1

# Piece type indices (match src/types.hpp PieceType): P=0,N=1,B=2,R=3,Q=4,K=5.
PT_FROM_CHAR = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
}


# ──────────────────────────────────────────────────────────────────────────────
# Feature indexing
# ──────────────────────────────────────────────────────────────────────────────
def feature_index(persp: int, color: int, ptype: int, sq: int) -> int:
    """Feature index for (perspective P, piece color c, piece type t, square s).

    os = (P==WHITE)? s : s^56     (vertical mirror for black's perspective)
    cr = (c==P)? 0 : 1            (own pieces first, then enemy)
    idx = cr*384 + t*64 + os      (range 0..767)
    """
    os = sq if persp == WHITE else (sq ^ 56)
    cr = 0 if color == persp else 1
    return cr * 384 + ptype * 64 + os


# ──────────────────────────────────────────────────────────────────────────────
# FEN -> piece list (color, ptype, square)  with square index a1=0..h8=63
# (rank-major little-endian, matching src/types.hpp Square enum)
# ──────────────────────────────────────────────────────────────────────────────
def parse_fen_pieces(fen_board: str):
    """Yield (color, ptype, square) for every piece on the board part of a FEN.

    square = rank*8 + file, with rank 0 == rank '1'. FEN ranks are listed from
    rank 8 down to rank 1, files a..h left to right.
    """
    ranks = fen_board.split('/')
    pieces = []
    for ri, row in enumerate(ranks):
        rank = 7 - ri          # ranks[0] is rank 8 -> rank index 7
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


# ──────────────────────────────────────────────────────────────────────────────
# Dataset preprocessing: text -> compact numpy arrays.
#
# For each position we emit, from the side-to-move perspective:
#   * stm feature column indices (variable length, <=32)
#   * nonstm feature column indices
#   * score (STM POV, cp)
#   * result (STM POV, in {0,0.5,1})
#
# We pack features for ALL positions into flat arrays + per-position offsets so
# the training loop can build sparse batches fast without Python-level loops.
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(data_path: str, cache_path: str, limit: int | None = None):
    cache = Path(cache_path)
    if cache.exists():
        z = np.load(cache)
        print(f"[data] loaded cache {cache_path}: {len(z['offsets'])-1} positions")
        return (z['stm_feats'], z['stm_off'], z['nst_feats'], z['nst_off'],
                z['score'], z['result'])

    t0 = time.time()
    stm_feats: list[int] = []
    stm_off = [0]
    nst_feats: list[int] = []
    nst_off = [0]
    scores: list[float] = []
    results: list[float] = []

    n = 0
    bad = 0
    with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
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
                score_w = float(parts[1].strip())     # White POV cp
                result_w = float(parts[2].strip())    # White POV result {0,0.5,1}
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
            # Build feature indices for both perspectives.
            sf = [feature_index(stm, c, t, s) for (c, t, s) in pieces]
            nf = [feature_index(stm ^ 1, c, t, s) for (c, t, s) in pieces]

            # Convert score & result to STM POV.
            if stm == WHITE:
                score = score_w
                result = result_w
            else:
                score = -score_w
                result = 1.0 - result_w

            stm_feats.extend(sf)
            stm_off.append(len(stm_feats))
            nst_feats.extend(nf)
            nst_off.append(len(nst_feats))
            scores.append(score)
            results.append(result)

            n += 1
            if limit and n >= limit:
                break
            if n % 500000 == 0:
                print(f"[data] parsed {n} positions...")

    print(f"[data] parsed {n} positions ({bad} skipped) in {time.time()-t0:.1f}s")

    stm_feats = np.asarray(stm_feats, dtype=np.int32)
    stm_off = np.asarray(stm_off, dtype=np.int64)
    nst_feats = np.asarray(nst_feats, dtype=np.int32)
    nst_off = np.asarray(nst_off, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    results = np.asarray(results, dtype=np.float32)

    np.savez(cache, stm_feats=stm_feats, stm_off=stm_off,
             nst_feats=nst_feats, nst_off=nst_off,
             score=scores, result=results, offsets=stm_off)
    print(f"[data] cached -> {cache_path}")
    return stm_feats, stm_off, nst_feats, nst_off, scores, results


# ──────────────────────────────────────────────────────────────────────────────
# Sparse batch builder. Given a set of position indices, returns coalesced sparse
# COO index tensors (row=position-in-batch, col=feature) for stm and nonstm.
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


def build_sparse(feats, off, idx, device):
    """Return a dense [B, INPUT_SIZE] float tensor of 0/1 features for batch `idx`.

    Fully vectorized (no Python per-row loop): builds COO (row, col) index pairs
    with numpy gather arithmetic, then densifies on `device`.
    """
    starts = off[idx]                       # [B]
    ends = off[idx + 1]                      # [B]
    counts = (ends - starts).astype(np.int64)   # features per position
    total = int(counts.sum())
    B = len(idx)
    if total == 0:
        return torch.zeros((B, INPUT_SIZE), dtype=torch.float32, device=device)
    rows = np.repeat(np.arange(B, dtype=np.int64), counts)
    # Build flat gather positions into `feats`: for row r, positions
    # starts[r] .. starts[r]+counts[r]-1. Equivalent to a per-segment arange.
    seg = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts)
    gather = np.repeat(starts, counts) + seg
    cols = feats[gather].astype(np.int64)
    i = torch.from_numpy(np.stack([rows, cols])).to(device)
    v = torch.ones(total, dtype=torch.float32, device=device)
    sp = torch.sparse_coo_tensor(i, v, (B, INPUT_SIZE), device=device)
    return sp.to_dense()


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        # W1: [HL, INPUT_SIZE], applied as acc = x @ W1.T + b1
        self.l1 = nn.Linear(INPUT_SIZE, HL)
        # W2: [1, 2*HL]
        self.l2 = nn.Linear(2 * HL, 1)

    def forward(self, x_stm, x_nst):
        acc_stm = self.l1(x_stm)          # [B, HL]
        acc_nst = self.l1(x_nst)          # [B, HL]
        # clipped ReLU in float domain: clamp to [0,1] (quant uses [0,QA]).
        c_stm = torch.clamp(acc_stm, 0.0, 1.0)
        c_nst = torch.clamp(acc_nst, 0.0, 1.0)
        x = torch.cat([c_stm, c_nst], dim=1)   # stm first
        # Output is the WIN-LOGIT y. Training uses pred = sigmoid(y); at
        # convergence y ≈ score/SCALE, so the quantized eval (which multiplies by
        # SCALE) comes out in engine centipawns. float eval in cp == y * SCALE.
        return self.l2(x).squeeze(1)      # [B], win-logit


# ──────────────────────────────────────────────────────────────────────────────
# Quantized integer forward pass (pure numpy int math == the C++ contract).
# ──────────────────────────────────────────────────────────────────────────────
class QuantNet:
    def __init__(self, W1q, b1q, W2q, b2q):
        # W1q: [HL, INPUT_SIZE] int16 ; b1q: [HL] int16
        # W2q: [2*HL] int16 (first HL = stm, next HL = nonstm) ; b2q: int32
        self.W1q = W1q.astype(np.int64)
        self.b1q = b1q.astype(np.int64)
        self.W2q = W2q.astype(np.int64)
        self.b2q = int(b2q)

    def eval_features(self, stm_feat_idx, nst_feat_idx) -> int:
        """Quantized eval in cp, STM POV. stm_feat_idx/nst_feat_idx: lists of cols."""
        acc_stm = self.b1q.copy()
        for c in stm_feat_idx:
            acc_stm += self.W1q[:, c]
        acc_nst = self.b1q.copy()
        for c in nst_feat_idx:
            acc_nst += self.W1q[:, c]
        # clipped relu in quantized domain: clamp to [0, QA]
        cr_stm = np.clip(acc_stm, 0, QA)
        cr_nst = np.clip(acc_nst, 0, QA)
        acc = np.concatenate([cr_stm, cr_nst])   # stm first
        dot = int(np.dot(acc, self.W2q))         # int64
        out = self.b2q + dot
        # (b2q + sum) * SCALE / (QA*QB)   integer division (C++ matches)
        return int(out * SCALE // (QA * QB))

    def eval_fen(self, fen: str) -> int:
        fields = fen.split()
        board = fields[0]
        stm = WHITE if fields[1] == 'w' else BLACK
        pieces = parse_fen_pieces(board)
        sf = [feature_index(stm, c, t, s) for (c, t, s) in pieces]
        nf = [feature_index(stm ^ 1, c, t, s) for (c, t, s) in pieces]
        return self.eval_features(sf, nf)

    def acc_range(self, stm_feat_idx, nst_feat_idx):
        """Return (min, max) of the PRE-clamp accumulator. Used to verify the
        int16 accumulator in the C++ contract never overflows."""
        acc_stm = self.b1q.copy()
        for c in stm_feat_idx:
            acc_stm += self.W1q[:, c]
        acc_nst = self.b1q.copy()
        for c in nst_feat_idx:
            acc_nst += self.W1q[:, c]
        return (int(min(acc_stm.min(), acc_nst.min())),
                int(max(acc_stm.max(), acc_nst.max())))


def quantize(model: NNUE):
    W1 = model.l1.weight.detach().cpu().numpy()        # [HL, INPUT]
    b1 = model.l1.bias.detach().cpu().numpy()          # [HL]
    W2 = model.l2.weight.detach().cpu().numpy()[0]     # [2*HL]
    b2 = float(model.l2.bias.detach().cpu().numpy()[0])

    W1q = np.round(W1 * QA).astype(np.int32)
    b1q = np.round(b1 * QA).astype(np.int32)
    W2q = np.round(W2 * QB).astype(np.int32)
    b2q = int(round(b2 * QA * QB))

    # Sanity: warn if anything overflows int16 (would break the C++ contract).
    def chk(name, arr, lo, hi):
        amin, amax = int(arr.min()), int(arr.max())
        if amin < lo or amax > hi:
            print(f"[quant] WARNING {name} range [{amin},{amax}] exceeds "
                  f"[{lo},{hi}] -> clamping")
    chk("W1q", W1q, -32768, 32767)
    chk("b1q", b1q, -32768, 32767)
    chk("W2q", W2q, -32768, 32767)

    W1q = np.clip(W1q, -32768, 32767).astype(np.int16)
    b1q = np.clip(b1q, -32768, 32767).astype(np.int16)
    W2q = np.clip(W2q, -32768, 32767).astype(np.int16)
    return QuantNet(W1q, b1q, W2q, b2q), (W1q, b1q, W2q, b2q)


# ──────────────────────────────────────────────────────────────────────────────
# Binary export. Little-endian. See README_NNUE.md for the exact byte layout.
#   Header:  u32 magic | u16 HL | u16 QA | u16 QB | u16 SCALE
#   W1q:     HL*INPUT int16, row-major [output o][input i]  (o outer, i inner)
#   b1q:     HL       int16
#   W2q:     2*HL     int16  (index 0..HL-1 = stm, HL..2HL-1 = nonstm)
#   b2q:     1        int32
# ──────────────────────────────────────────────────────────────────────────────
def export_bin(path: str, W1q, b1q, W2q, b2q):
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<HHHH', HL, QA, QB, SCALE))
        # W1q row-major [output][input]: W1q has shape [HL, INPUT] already so
        # C-order flatten == [o][i].
        f.write(np.ascontiguousarray(W1q, dtype='<i2').tobytes())
        f.write(np.ascontiguousarray(b1q, dtype='<i2').tobytes())
        f.write(np.ascontiguousarray(W2q, dtype='<i2').tobytes())
        f.write(struct.pack('<i', int(b2q)))
    size = Path(path).stat().st_size
    expected = 4 + 8 + HL * INPUT_SIZE * 2 + HL * 2 + 2 * HL * 2 + 4
    print(f"[export] wrote {path}: {size} bytes (expected {expected})")
    assert size == expected, "binary size mismatch!"


# ──────────────────────────────────────────────────────────────────────────────
# Sample positions for the C++ contract test.
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_FENS = [
    # startpos
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # after 1.e4 (black to move)
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    # after 1.e4 e5 (white to move)
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    # Italian-ish opening
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 3",
    # White up a queen (black queen removed), white to move
    "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Black up a queen (white queen removed), white to move
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
    # White up a rook, white to move
    "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1",
    # White up a full piece (knight), black to move
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR b KQkq - 0 1",
    # endgame: KQ vs K (white winning), white to move
    "8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1",
    # endgame: KR vs K (white winning), black to move
    "8/8/8/4k3/8/8/4R3/4K3 b - - 0 1",
    # endgame: K vs KQ (black winning), white to move
    "4k3/4q3/8/8/8/8/8/4K3 w - - 0 1",
    # KP vs K endgame, white to move
    "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1",
    # rook endgame, white to move
    "8/5k2/8/8/8/8/3R1K2/8 w - - 0 1",
    # symmetric-ish middlegame
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1",
    # material imbalance: white up two pawns
    "rnbqkbnr/pp3ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # tactical middlegame (open position)
    "r2q1rk1/pp1n1ppp/2pbpn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 1",
    # Black up a rook, black to move
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR b KQk - 0 1",
    # Sicilian-ish
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    # Queen vs rook endgame (white winning), white to move
    "4k3/8/8/8/8/8/4r3/3QK3 w - - 0 1",
    # bishop+knight vs lone king (white winning), white to move
    "8/8/8/4k3/8/8/4BN2/4K3 w - - 0 1",
    # white up a bishop, white to move
    "rnbqk1nr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # closed center middlegame
    "r1bqk2r/ppp1bppp/2np1n2/4p3/2P1P3/2NP1N2/PP3PPP/R1BQKB1R w KQkq - 0 1",
    # endgame two rooks vs king (white), white to move
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
    # black up a knight, white to move
    "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # double-edged middlegame
    "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1",
]


def write_samples(path: str, qnet: QuantNet):
    lines = []
    for fen in SAMPLE_FENS:
        cp = qnet.eval_fen(fen)
        lines.append(f"{fen} {cp}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[samples] wrote {len(lines)} samples -> {path}")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/nnue_data.txt')
    ap.add_argument('--cache', default='data/nnue_cache.npz')
    ap.add_argument('--out', default='nets/king_nnue.bin')
    ap.add_argument('--samples', default='trainer/nnue_samples.txt')
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--batch', type=int, default=16384)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.02)
    ap.add_argument('--limit', type=int, default=None,
                    help='cap number of positions (debug)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev_name = (torch.cuda.get_device_name(0)
                if device.type == 'cuda' else 'CPU')
    print(f"[init] torch {torch.__version__}  device={device} ({dev_name})  "
          f"cuda_version={torch.version.cuda}")

    sf, so, nf, no, score, result = preprocess(args.data, args.cache, args.limit)
    N = len(so) - 1
    print(f"[data] {N} positions ready")

    # Train / val split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"[data] train={len(train_idx)} val={len(val_idx)}")

    # Clamp scores for the target: mate/adjudication scores reach ±29999 which
    # saturate the sigmoid and add no signal beyond ~±2000cp. Clamping stabilizes
    # training without changing the WDL blend meaningfully (sigmoid(2000/400)=0.993).
    score_clamped = np.clip(score, -2000.0, 2000.0)
    score_t = torch.from_numpy(score_clamped)
    result_t = torch.from_numpy(result)

    def target_for(idx):
        s = score_t[idx].to(device)
        r = result_t[idx].to(device)
        # WDL-blended target (STM POV). score & result are already STM POV.
        return 0.6 * torch.sigmoid(s / SCALE) + 0.4 * r

    model = NNUE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.3)
    loss_fn = nn.MSELoss()

    def run_val():
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_idx), args.batch):
                bidx = val_idx[i:i + args.batch]
                xs = build_sparse(sf, so, bidx, device)
                xn = build_sparse(nf, no, bidx, device)
                y = model(xs, xn)
                pred = torch.sigmoid(y)        # y is the win-logit
                tgt = target_for(bidx)
                loss = loss_fn(pred, tgt)
                total += loss.item() * len(bidx)
                count += len(bidx)
        model.train()
        return total / max(1, count)

    first_train_loss = None
    best_val = float('inf')
    print("[train] starting")
    for epoch in range(args.epochs):
        t0 = time.time()
        ep_perm = rng.permutation(len(train_idx))
        run_total, run_count = 0.0, 0
        for i in range(0, len(train_idx), args.batch):
            bidx = train_idx[ep_perm[i:i + args.batch]]
            xs = build_sparse(sf, so, bidx, device)
            xn = build_sparse(nf, no, bidx, device)
            y = model(xs, xn)
            pred = torch.sigmoid(y)            # y is the win-logit
            tgt = target_for(bidx)
            loss = loss_fn(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_total += loss.item() * len(bidx)
            run_count += len(bidx)
        sched.step()
        train_loss = run_total / max(1, run_count)
        val_loss = run_val()
        if first_train_loss is None:
            first_train_loss = train_loss
            first_val_loss = val_loss
        best_val = min(best_val, val_loss)
        print(f"[train] epoch {epoch+1:3d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  "
              f"{time.time()-t0:.1f}s")

    print(f"[train] DONE  first_train={first_train_loss:.6f} -> "
          f"last_train={train_loss:.6f}  "
          f"first_val={first_val_loss:.6f} -> last_val={val_loss:.6f}  "
          f"best_val={best_val:.6f}")

    # ── Quantize + export ──
    qnet, (W1q, b1q, W2q, b2q) = quantize(model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    export_bin(args.out, W1q, b1q, W2q, b2q)

    # ── Float vs quant agreement on a validation subset ──
    model.eval()
    check_n = min(2000, len(val_idx))
    cidx = val_idx[:check_n]
    diffs = []
    acc_lo, acc_hi = 0, 0
    with torch.no_grad():
        xs = build_sparse(sf, so, cidx, device)
        xn = build_sparse(nf, no, cidx, device)
        # model output is the win-logit; float eval in cp == y * SCALE.
        yf = model(xs, xn).cpu().numpy() * SCALE
    for k, pi in enumerate(cidx):
        a, b = so[pi], so[pi + 1]
        an, bn = no[pi], no[pi + 1]
        si = sf[a:b].tolist()
        ni = nf[an:bn].tolist()
        q = qnet.eval_features(si, ni)
        diffs.append(abs(q - float(yf[k])))
        lo, hi = qnet.acc_range(si, ni)
        acc_lo = min(acc_lo, lo)
        acc_hi = max(acc_hi, hi)
    diffs = np.asarray(diffs)
    print(f"[quant] float-vs-quant cp diff: mean={diffs.mean():.3f}  "
          f"median={np.median(diffs):.3f}  max={diffs.max():.3f}  "
          f"p99={np.percentile(diffs,99):.3f}  (n={check_n})")
    print(f"[quant] pre-clamp accumulator range over subset: "
          f"[{acc_lo}, {acc_hi}]  (int16 safe if within [-32768, 32767])")
    if acc_lo < -32768 or acc_hi > 32767:
        print("[quant] WARNING: accumulator overflows int16! C++ must use int32 "
              "accumulators or weights need regularization.")

    # ── Samples ──
    Path(args.samples).parent.mkdir(parents=True, exist_ok=True)
    lines = write_samples(args.samples, qnet)

    print("\n[sanity] selected sample evals (FEN -> cp, STM POV):")
    show = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        "8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1",
        "4k3/4q3/8/8/8/8/8/4K3 w - - 0 1",
    ]
    for fen in show:
        print(f"   {qnet.eval_fen(fen):+6d}  {fen}")


if __name__ == '__main__':
    sys.exit(main())
