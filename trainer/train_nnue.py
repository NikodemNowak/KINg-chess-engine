#!/usr/bin/env python3
"""KINg NNUE trainer — large-scale streaming version.

Perspective net ``(768 → HL)×2 → 1``, clipped-ReLU.  Default ``HL=512``.
Trains float weights with PyTorch (GPU if available), then quantizes to
integers and exports a little-endian binary that the C++ inference reproduces
bit-for-bit.

Architecture, feature indexing, quantization constants and binary layout are
documented in trainer/README_NNUE.md and MUST match the C++ side exactly.

Data pipeline (designed for ~30M positions, ~2GB binary cache):
    1. Run ``trainer/preprocess_nnue.py`` ONCE to convert the text dataset to a
       compact binary cache (72 bytes/position, numpy memmap format).
    2. This script reads the cache via ``StreamingNNUEDataset`` backed by
       ``np.memmap`` — never loads the whole dataset into RAM.
       Multi-worker DataLoader OK (workers=0 default on Windows).

Usage:
    # Pre-process (once)
    py trainer/preprocess_nnue.py --data data/nnue_big.txt --out data/nnue_big.bin

    # Train (full retrain, HL=512)
    py trainer/train_nnue.py --cache data/nnue_big.bin \\
        --out nets/king_nnue.bin --samples trainer/nnue_samples.txt

    # Smoke run on partial data (TEMP net — do NOT clobber king_nnue.bin)
    py trainer/train_nnue.py --cache trainer/smoke_big.bin \\
        --out nets/smoke_nnue.bin --epochs 5 --hl 512

Run with --help for all options.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Constants (contract with C++ / quantization / feature indexing).
# HL is a CLI argument (default 512).  All other constants are fixed.
# ──────────────────────────────────────────────────────────────────────────────
INPUT_SIZE = 768          # 2 colors * 6 piece types * 64 squares
QA = 255                  # accumulator (W1,b1) quantization scale
QB = 64                   # output weight (W2) quantization scale
SCALE = 400               # eval scale (cp), == sigmoid divisor
MAGIC = 0x4B4E5545        # "KNUE" big-endian spelled; stored little-endian u32

WHITE, BLACK = 0, 1

PT_FROM_CHAR = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}

# Binary record dtype — must match preprocess_nnue.py EXACTLY.
RECORD_DTYPE = np.dtype([
    ('score',   '<i2'),        # STM-POV score, clamped ±2000 cp
    ('result',  '<f2'),        # STM-POV result {0, 0.5, 1}
    ('stm',     'u1'),         # 0=White, 1=Black
    ('n',       'u1'),         # number of pieces (≤32)
    ('pieces',  'u1', (32,)),  # (color<<4)|ptype per piece; zeros = padding
    ('squares', 'u1', (32,)),  # square index 0=a1..63=h8 per piece
    ('_pad',    'u1', (2,)),   # alignment to 72 bytes
])
assert RECORD_DTYPE.itemsize == 72


# ──────────────────────────────────────────────────────────────────────────────
# Feature indexing — EXACT match with README_NNUE.md and C++.
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
# FEN -> piece list  (used only by QuantNet.eval_fen and sample writing).
# ──────────────────────────────────────────────────────────────────────────────
def parse_fen_pieces(fen_board: str):
    """Yield (color, ptype, square) for every piece.  square=rank*8+file, a1=0."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Streaming dataset backed by numpy memmap.
# Workers each open their own file handle via _open() on first __getitem__.
# ──────────────────────────────────────────────────────────────────────────────
class StreamingNNUEDataset(Dataset):
    """Memory-mapped NNUE position dataset.  Never loads full data into RAM."""

    def __init__(self, cache_path: str, indices: np.ndarray):
        self.cache_path = str(cache_path)
        self.indices = indices
        self._mm: np.memmap | None = None
        # Validate cache size.
        file_size = Path(cache_path).stat().st_size
        if file_size % RECORD_DTYPE.itemsize != 0:
            raise ValueError(
                f"Cache file {cache_path} size {file_size} is not a multiple "
                f"of record size {RECORD_DTYPE.itemsize}")
        self._total = file_size // RECORD_DTYPE.itemsize

    def _open(self):
        if self._mm is None:
            self._mm = np.memmap(self.cache_path, dtype=RECORD_DTYPE,
                                 mode='r', shape=(self._total,))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        self._open()
        # Return the raw structured record; ALL feature-index math is done
        # vectorised over the whole batch in collate_fn (no Python per-piece
        # loop), which keeps the GPU fed instead of starved.
        return self._mm[self.indices[i]]


def collate_fn(batch):
    """Vectorised collate: a list of raw records -> EmbeddingBag inputs.

    Computes every active feature index for the whole batch with numpy array ops
    (no Python per-piece loop), then packs them into a flat index tensor + offsets
    for nn.EmbeddingBag(mode='sum'). Both perspectives share the SAME piece order,
    so they share offsets. The feature_index math here MUST match feature_index()
    and the C++ inference EXACTLY (bit-exact contract).
    """
    recs = np.asarray(batch)                       # structured array [B]
    B = len(recs)
    stm = recs['stm'].astype(np.int64)             # [B]
    n   = recs['n'].astype(np.int64)               # [B]
    pieces  = recs['pieces']                        # [B,32] uint8
    squares = recs['squares']                       # [B,32] uint8

    scores_t  = torch.from_numpy(np.ascontiguousarray(recs['score']).astype(np.float32))
    results_t = torch.from_numpy(np.ascontiguousarray(recs['result']).astype(np.float32))

    # Mask of valid piece slots (k < n) and flat per-piece attributes.
    valid = np.arange(32)[None, :] < n[:, None]    # [B,32] bool
    rows  = np.repeat(np.arange(B, dtype=np.int64), n)   # sample index per piece
    p = pieces[valid]                               # [T] uint8
    s = squares[valid].astype(np.int64)            # [T]
    color = ((p >> 4) & 1).astype(np.int64)
    ptype = (p & 0x0F).astype(np.int64)
    stm_f = stm[rows]                               # [T] perspective owner

    # STM perspective (persp = stm).
    os_s = np.where(stm_f == WHITE, s, s ^ 56)
    cr_s = (color != stm_f).astype(np.int64)
    sf = cr_s * 384 + ptype * 64 + os_s
    # NSTM perspective (persp = stm ^ 1).
    nstm = stm_f ^ 1
    os_n = np.where(nstm == WHITE, s, s ^ 56)
    cr_n = (color != nstm).astype(np.int64)
    nf = cr_n * 384 + ptype * 64 + os_n

    offsets = np.zeros(B, dtype=np.int64)
    if B > 1:
        np.cumsum(n[:-1], out=offsets[1:])
    off_t = torch.from_numpy(offsets)
    return (torch.from_numpy(sf), off_t,
            torch.from_numpy(nf), off_t.clone(), scores_t, results_t)


# ──────────────────────────────────────────────────────────────────────────────
# Model — HL is a parameter; EXACT same architecture as before, just flexible.
# ──────────────────────────────────────────────────────────────────────────────
class NNUE(nn.Module):
    def __init__(self, hl: int, squared: bool = False):
        super().__init__()
        self.hl = hl
        self.squared = squared  # SCReLU (clamp then square) vs CReLU
        # Feature transformer as a sparse EmbeddingBag (sum over active features).
        # ft.weight is [INPUT_SIZE, HL]; the equivalent dense Linear weight W1 is
        # [HL, INPUT_SIZE] == ft.weight.T (used by quantize()/export). The bias b1
        # is a separate parameter added after the bag sum.
        self.ft = nn.EmbeddingBag(INPUT_SIZE, hl, mode='sum')
        self.b1 = nn.Parameter(torch.zeros(hl))
        # W2: [1, 2*HL]
        self.l2 = nn.Linear(2 * hl, 1)
        # Match the old nn.Linear default init scale so training dynamics/quant
        # ranges stay comparable.
        nn.init.kaiming_uniform_(self.ft.weight, a=5 ** 0.5)

    def forward(self, s_idx, s_off, n_idx, n_off):
        acc_stm = self.ft(s_idx, s_off) + self.b1
        acc_nst = self.ft(n_idx, n_off) + self.b1
        # Clipped ReLU in float domain: clamp to [0,1] (quant uses [0,QA]).
        c_stm = torch.clamp(acc_stm, 0.0, 1.0)
        c_nst = torch.clamp(acc_nst, 0.0, 1.0)
        if self.squared:  # SCReLU: square the clamped activation
            c_stm = c_stm * c_stm
            c_nst = c_nst * c_nst
        x = torch.cat([c_stm, c_nst], dim=1)   # stm first
        # Output is the WIN-LOGIT y.  Float eval in cp == y * SCALE.
        return self.l2(x).squeeze(1)            # [B]

    @torch.no_grad()
    def clip_weights(self):
        """Quant-safety guard: keep weights inside int16-safe magnitudes so the
        exported net can never silently saturate/diverge. Bounds are loose enough
        that they essentially never bind in healthy training (catastrophe guard).
        FT: |w|<=1.98 keeps the 32-active accumulator inside int16 for any HL
        (32*1.98*QA + bias < 32767); real trained values peak ~1.86 so it rarely
        binds. L2: |w|<=64 keeps W2q in int16 (64*QB=4096) while never binding on
        real values (which peak ~4.75) — pure divergence insurance."""
        self.ft.weight.clamp_(-1.98, 1.98)
        self.b1.clamp_(-1.98, 1.98)
        self.l2.weight.clamp_(-64.0, 64.0)


# ──────────────────────────────────────────────────────────────────────────────
# Quantized integer forward pass (pure numpy int math == the C++ contract).
# ──────────────────────────────────────────────────────────────────────────────
class QuantNet:
    def __init__(self, W1q, b1q, W2q, b2q, hl: int, squared: bool = False):
        self.W1q = W1q.astype(np.int64)  # [HL, INPUT]
        self.b1q = b1q.astype(np.int64)  # [HL]
        self.W2q = W2q.astype(np.int64)  # [2*HL]
        self.b2q = int(b2q)
        self.hl = hl
        self.squared = squared
        # SCReLU squares the [0,QA] activation, adding one factor of QA to scale.
        self.den = (QA * QA * QB) if squared else (QA * QB)

    def eval_features(self, stm_feat_idx, nst_feat_idx) -> int:
        """Quantized eval in cp, STM POV. Mirrors the C++ integer inference."""
        acc_stm = self.b1q.copy()
        for c in stm_feat_idx:
            acc_stm += self.W1q[:, c]
        acc_nst = self.b1q.copy()
        for c in nst_feat_idx:
            acc_nst += self.W1q[:, c]
        cr_stm = np.clip(acc_stm, 0, QA)
        cr_nst = np.clip(acc_nst, 0, QA)
        if self.squared:
            cr_stm = cr_stm * cr_stm
            cr_nst = cr_nst * cr_nst
        acc = np.concatenate([cr_stm, cr_nst])   # stm first
        dot = int(np.dot(acc, self.W2q))
        out = self.b2q + dot
        # Python floor division — matches the C++ floor-div snippet exactly.
        return int(out * SCALE // self.den)

    def eval_fen(self, fen: str) -> int:
        fields = fen.split()
        board = fields[0]
        stm = WHITE if fields[1] == 'w' else BLACK
        pieces = parse_fen_pieces(board)
        sf = [feature_index(stm, c, t, s) for (c, t, s) in pieces]
        nf = [feature_index(stm ^ 1, c, t, s) for (c, t, s) in pieces]
        return self.eval_features(sf, nf)

    def acc_range(self, stm_feat_idx, nst_feat_idx):
        """Pre-clamp accumulator range (to check int16 safety)."""
        acc_stm = self.b1q.copy()
        for c in stm_feat_idx:
            acc_stm += self.W1q[:, c]
        acc_nst = self.b1q.copy()
        for c in nst_feat_idx:
            acc_nst += self.W1q[:, c]
        return (int(min(acc_stm.min(), acc_nst.min())),
                int(max(acc_stm.max(), acc_nst.max())))


def quantize(model: NNUE):
    hl = model.hl
    # EmbeddingBag weight is [INPUT, HL]; the dense-equivalent W1 is its transpose
    # [HL, INPUT]. b1 is the separate bias parameter.
    W1 = model.ft.weight.detach().cpu().numpy().T   # [HL, INPUT]
    b1 = model.b1.detach().cpu().numpy()            # [HL]
    W2 = model.l2.weight.detach().cpu().numpy()[0]  # [2*HL]
    b2 = float(model.l2.bias.detach().cpu().numpy()[0])

    squared = getattr(model, 'squared', False)
    W1q = np.round(W1 * QA).astype(np.int32)
    b1q = np.round(b1 * QA).astype(np.int32)
    W2q = np.round(W2 * QB).astype(np.int32)
    # SCReLU's squared activation carries an extra QA factor into the output scale.
    b2q = int(round(b2 * (QA * QA * QB if squared else QA * QB)))

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
    return QuantNet(W1q, b1q, W2q, b2q, hl, squared), (W1q, b1q, W2q, b2q)


# ──────────────────────────────────────────────────────────────────────────────
# Binary export — see README_NNUE.md for byte layout.
#   Header:  u32 magic | u16 HL | u16 QA | u16 QB | u16 SCALE  (12 bytes)
#   W1q:     HL*INPUT_SIZE int16, row-major [output o][input i]
#   b1q:     HL int16
#   W2q:     2*HL int16  (0..HL-1 = stm, HL..2HL-1 = nonstm)
#   b2q:     int32
# HL is written in the header — C++ reads it back at load time.
# ──────────────────────────────────────────────────────────────────────────────
def export_bin(path: str, hl: int, W1q, b1q, W2q, b2q):
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<HHHH', hl, QA, QB, SCALE))
        f.write(np.ascontiguousarray(W1q, dtype='<i2').tobytes())
        f.write(np.ascontiguousarray(b1q, dtype='<i2').tobytes())
        f.write(np.ascontiguousarray(W2q, dtype='<i2').tobytes())
        f.write(struct.pack('<i', int(b2q)))
    size = Path(path).stat().st_size
    expected = 4 + 8 + hl * INPUT_SIZE * 2 + hl * 2 + 2 * hl * 2 + 4
    print(f"[export] wrote {path}: {size} bytes (expected {expected})")
    assert size == expected, f"binary size mismatch: got {size}, expected {expected}"


# ──────────────────────────────────────────────────────────────────────────────
# Sample positions for the C++ contract test (bit-exact eval).
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 3",
    "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
    "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR b KQkq - 0 1",
    "8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1",
    "8/8/8/4k3/8/8/4R3/4K3 b - - 0 1",
    "4k3/4q3/8/8/8/8/8/4K3 w - - 0 1",
    "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1",
    "8/5k2/8/8/8/8/3R1K2/8 w - - 0 1",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1",
    "rnbqkbnr/pp3ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r2q1rk1/pp1n1ppp/2pbpn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR b KQk - 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "4k3/8/8/8/8/8/4r3/3QK3 w - - 0 1",
    "8/8/8/4k3/8/8/4BN2/4K3 w - - 0 1",
    "rnbqk1nr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqk2r/ppp1bppp/2np1n2/4p3/2P1P3/2NP1N2/PP3PPP/R1BQKB1R w KQkq - 0 1",
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
    "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
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


def _rss_gb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:
        return float('nan')


# ──────────────────────────────────────────────────────────────────────────────
# GPU-resident dataset: precompute padded per-position feature indices (both
# perspectives) + labels ONCE and keep them on the GPU, so training needs no CPU
# dataloader and the GPU runs at full utilisation. Feature math mirrors
# feature_index()/collate_fn EXACTLY (bit-exact contract). ~4 GB on the GPU for
# 30M positions; falls back to the DataLoader path on CPU / huge datasets.
# ──────────────────────────────────────────────────────────────────────────────
MAXP = 32  # max pieces per position (record layout)

def build_gpu_dataset(cache_path: str, total: int, device):
    # Disk-cache the padded feature tensors (feature-only → reusable across ALL
    # trainings on this data cache, regardless of HL/activation/lambda). First run
    # builds + saves (~60s); later runs load (~15s) so the GPU never idles on a
    # rebuild during a sweep.
    npz = cache_path + ".gpures.npz"
    if Path(npz).exists() and Path(npz).stat().st_size > 0:
        print(f"[data] loading cached GPU tensors from {npz}")
        d = np.load(npz)
        return (torch.from_numpy(d['sf']).to(device),
                torch.from_numpy(d['nf']).to(device),
                torch.from_numpy(d['ln']).to(device),
                torch.from_numpy(d['sc']).to(device),
                torch.from_numpy(d['rs']).to(device))
    mm = np.memmap(cache_path, dtype=RECORD_DTYPE, mode='r', shape=(total,))
    sf_pad = np.zeros((total, MAXP), dtype=np.int16)
    nf_pad = np.zeros((total, MAXP), dtype=np.int16)
    lengths = np.empty(total, dtype=np.int64)
    scores  = np.empty(total, dtype=np.float32)
    results = np.empty(total, dtype=np.float32)
    CH = 1_000_000
    for lo in range(0, total, CH):
        hi = min(lo + CH, total)
        rec = mm[lo:hi]
        stm = rec['stm'].astype(np.int64)[:, None]      # [c,1]
        squares = rec['squares'].astype(np.int64)        # [c,32]
        pieces  = rec['pieces']                           # [c,32] uint8
        color = ((pieces >> 4) & 1).astype(np.int64)
        ptype = (pieces & 0x0F).astype(np.int64)
        os_s = np.where(stm == WHITE, squares, squares ^ 56)
        sf_pad[lo:hi] = ((color != stm).astype(np.int64) * 384 + ptype * 64 + os_s).astype(np.int16)
        nstm = stm ^ 1
        os_n = np.where(nstm == WHITE, squares, squares ^ 56)
        nf_pad[lo:hi] = ((color != nstm).astype(np.int64) * 384 + ptype * 64 + os_n).astype(np.int16)
        lengths[lo:hi] = rec['n'].astype(np.int64)
        scores[lo:hi]  = rec['score'].astype(np.float32)
        results[lo:hi] = rec['result'].astype(np.float32)
    del mm
    try:
        np.savez(npz, sf=sf_pad, nf=nf_pad, ln=lengths, sc=scores, rs=results)
        print(f"[data] cached GPU tensors -> {npz}")
    except Exception as e:
        print(f"[data] (cache save skipped: {e})")
    return (torch.from_numpy(sf_pad).to(device),
            torch.from_numpy(nf_pad).to(device),
            torch.from_numpy(lengths).to(device),
            torch.from_numpy(scores).to(device),
            torch.from_numpy(results).to(device))

def gpu_batch_inputs(sf_g, nf_g, len_g, rows, device):
    """EmbeddingBag (flat indices + offsets) for a batch of rows — all on GPU.
    Both perspectives share the same per-row piece counts, hence the same offsets."""
    L = len_g[rows]                                       # [B]
    mask = torch.arange(MAXP, device=device)[None, :] < L[:, None]
    s_flat = sf_g[rows][mask].long()
    n_flat = nf_g[rows][mask].long()
    offsets = torch.zeros(rows.numel(), device=device, dtype=torch.long)
    if rows.numel() > 1:
        offsets[1:] = torch.cumsum(L, 0)[:-1]
    return s_flat, offsets, n_flat, offsets


# ──────────────────────────────────────────────────────────────────────────────
# Training entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Force line-buffered stdout so progress prints appear immediately when the
    # process is run in the background (Python buffers stdout when not a TTY).
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(
        description='KINg NNUE trainer (streaming, large-scale)')
    ap.add_argument('--cache', required=True,
                    help='Binary cache (.bin) from preprocess_nnue.py')
    ap.add_argument('--out', default='nets/king_nnue.bin')
    ap.add_argument('--samples', default='trainer/nnue_samples.txt')
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--batch', type=int, default=16384)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hl', type=int, default=512,
                    help='Hidden layer size per perspective (default 512)')
    ap.add_argument('--activation', choices=['crelu', 'screlu'], default='crelu',
                    help='Hidden activation: crelu (default) or screlu (squared). '
                         'screlu nets MUST be compiled with -DNNUE_SCRELU.')
    ap.add_argument('--lam', type=float, default=0.6,
                    help='Target blend: lam*sigmoid(eval) + (1-lam)*game_result (default 0.6)')
    ap.add_argument('--val-frac', type=float, default=0.02)
    ap.add_argument('--workers', type=int, default=0,
                    help='DataLoader worker processes (default 0 = main thread; '
                         'safe on Windows; increase for CPU-bound preprocessing)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    HL = args.hl

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev_name = (torch.cuda.get_device_name(0)
                if device.type == 'cuda' else 'CPU')
    print(f"[init] torch {torch.__version__}  device={device} ({dev_name})  "
          f"cuda={torch.version.cuda}  HL={HL}")
    print(f"[init] RAM at start: {_rss_gb():.2f} GB")

    cache_path = args.cache
    if not Path(cache_path).exists():
        print(f"[data] ERROR: cache not found: {cache_path}")
        print("[data] Run first: py trainer/preprocess_nnue.py "
              "--data <txt> --out <bin>")
        sys.exit(1)

    total = Path(cache_path).stat().st_size // RECORD_DTYPE.itemsize
    print(f"[data] {total:,} positions in {cache_path}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(total)
    n_val = max(1, int(total * args.val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"[data] train={len(train_idx):,}  val={n_val:,}")

    use_gpu = (device.type == 'cuda')

    squared = (args.activation == 'screlu')
    model = NNUE(HL, squared=squared).to(device)
    print(f"[model] activation={args.activation}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] HL={HL}  params={n_params:,}  "
          f"({n_params*4/1e6:.2f} MB float32)  "
          f"export size={(4+8+HL*INPUT_SIZE*2+HL*2+2*HL*2+4)/1024:.1f} KB")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.3)
    loss_fn = nn.MSELoss()

    first_train, first_val, last_train, last_val = None, None, 0.0, 0.0
    best_val = float('inf')

    if use_gpu:
        # ── GPU-resident path: whole dataset on the GPU, no CPU dataloader ────
        print("[data] building GPU-resident dataset (one-time) ...")
        t_b = time.time()
        sf_g, nf_g, len_g, sc_g, rs_g = build_gpu_dataset(cache_path, total, device)
        tr_g = torch.from_numpy(train_idx).to(device)
        va_g = torch.from_numpy(val_idx).to(device)
        gb = (sf_g.numel()*2 + nf_g.numel()*2 + len_g.numel()*8
              + sc_g.numel()*4 + rs_g.numel()*4) / 1e9
        print(f"[data] GPU dataset ready in {time.time()-t_b:.1f}s  (~{gb:.2f} GB on device)")

        def make_tgt(rows):
            return args.lam * torch.sigmoid(sc_g[rows] / SCALE) + (1.0 - args.lam) * rs_g[rows]

        def run_val():
            model.eval()
            tot, cnt = 0.0, 0
            with torch.no_grad():
                for b in range(0, va_g.numel(), args.batch):
                    rows = va_g[b:b + args.batch]
                    si, so, ni, no = gpu_batch_inputs(sf_g, nf_g, len_g, rows, device)
                    pred = torch.sigmoid(model(si, so, ni, no))
                    bs = rows.numel()
                    tot += loss_fn(pred, make_tgt(rows)).item() * bs
                    cnt += bs
            model.train()
            return tot / max(1, cnt)

        print("[train] starting (GPU-resident)")
        for epoch in range(args.epochs):
            t0 = time.time()
            order = tr_g[torch.randperm(tr_g.numel(), device=device)]
            model.train()
            ep_loss, ep_cnt = 0.0, 0
            for b in range(0, order.numel(), args.batch):
                rows = order[b:b + args.batch]
                si, so, ni, no = gpu_batch_inputs(sf_g, nf_g, len_g, rows, device)
                pred = torch.sigmoid(model(si, so, ni, no))
                loss = loss_fn(pred, make_tgt(rows))
                opt.zero_grad(); loss.backward(); opt.step()
                model.clip_weights()
                bs = rows.numel()
                ep_loss += loss.item() * bs; ep_cnt += bs
            sched.step()
            train_loss = ep_loss / max(1, ep_cnt); val_loss = run_val()
            if first_train is None: first_train, first_val = train_loss, val_loss
            last_train, last_val = train_loss, val_loss
            best_val = min(best_val, val_loss)
            print(f"[train] epoch {epoch+1:3d}/{args.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.1f}s")
    else:
        # ── CPU fallback: streaming DataLoader path ──────────────────────────
        num_workers = args.workers
        loader_kw = dict(batch_size=args.batch, collate_fn=collate_fn,
                         num_workers=num_workers, pin_memory=False,
                         persistent_workers=(num_workers > 0),
                         prefetch_factor=(2 if num_workers > 0 else None))
        val_loader = DataLoader(StreamingNNUEDataset(cache_path, val_idx),
                                shuffle=False, **loader_kw)

        def make_target(sc, rs):
            return args.lam * torch.sigmoid(sc.to(device) / SCALE) + (1.0 - args.lam) * rs.to(device)

        def run_val():
            model.eval(); tot, cnt = 0.0, 0
            with torch.no_grad():
                for si, so, ni, no, sc, rs in val_loader:
                    pred = torch.sigmoid(model(si.to(device), so.to(device),
                                               ni.to(device), no.to(device)))
                    bs = sc.shape[0]
                    tot += loss_fn(pred, make_target(sc, rs)).item() * bs; cnt += bs
            model.train(); return tot / max(1, cnt)

        print("[train] starting (DataLoader)")
        for epoch in range(args.epochs):
            t0 = time.time()
            ep_order = train_idx[rng.permutation(len(train_idx))]
            ep_loader = DataLoader(StreamingNNUEDataset(cache_path, ep_order),
                                   shuffle=False, **loader_kw)
            model.train(); ep_loss, ep_cnt = 0.0, 0
            for si, so, ni, no, sc, rs in ep_loader:
                pred = torch.sigmoid(model(si.to(device), so.to(device),
                                           ni.to(device), no.to(device)))
                loss = loss_fn(pred, make_target(sc, rs))
                opt.zero_grad(); loss.backward(); opt.step(); model.clip_weights()
                bs = sc.shape[0]; ep_loss += loss.item() * bs; ep_cnt += bs
            sched.step()
            train_loss = ep_loss / max(1, ep_cnt); val_loss = run_val()
            if first_train is None: first_train, first_val = train_loss, val_loss
            last_train, last_val = train_loss, val_loss
            best_val = min(best_val, val_loss)
            print(f"[train] epoch {epoch+1:3d}/{args.epochs}  train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  lr={opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.1f}s")

    print(f"[train] DONE  "
          f"first_train={first_train:.6f} -> last_train={last_train:.6f}  "
          f"first_val={first_val:.6f} -> last_val={last_val:.6f}  "
          f"best_val={best_val:.6f}")

    # ── Quantize + export ──
    qnet, (W1q, b1q, W2q, b2q) = quantize(model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    export_bin(args.out, HL, W1q, b1q, W2q, b2q)

    # ── Float vs quant agreement on a validation subset ──
    model.eval()
    check_n = min(2000, n_val)
    check_idx = val_idx[:check_n]
    check_ds = StreamingNNUEDataset(cache_path, check_idx)
    check_loader = DataLoader(check_ds, batch_size=check_n, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    with torch.no_grad():
        si, so, ni, no, _, _ = next(iter(check_loader))
        yf = model(si.to(device), so.to(device),
                   ni.to(device), no.to(device)).cpu().numpy() * SCALE

    mm = np.memmap(cache_path, dtype=RECORD_DTYPE, mode='r', shape=(total,))
    diffs = []
    acc_lo, acc_hi = 0, 0
    for k, pi in enumerate(check_idx):
        if k >= len(yf):
            break
        rec = mm[pi]
        stm = int(rec['stm'])
        n_p = int(rec['n'])
        sf, nf = [], []
        for j in range(n_p):
            p = int(rec['pieces'][j])
            color = (p >> 4) & 1
            ptype = p & 0x0F
            sq = int(rec['squares'][j])
            sf.append(feature_index(stm, color, ptype, sq))
            nf.append(feature_index(stm ^ 1, color, ptype, sq))
        q = qnet.eval_features(sf, nf)
        diffs.append(abs(q - float(yf[k])))
        lo, hi = qnet.acc_range(sf, nf)
        acc_lo = min(acc_lo, lo)
        acc_hi = max(acc_hi, hi)
    del mm

    diffs = np.asarray(diffs)
    print(f"[quant] float-vs-quant cp diff: mean={diffs.mean():.3f}  "
          f"median={np.median(diffs):.3f}  max={diffs.max():.3f}  "
          f"p99={np.percentile(diffs,99):.3f}  (n={len(diffs)})")
    print(f"[quant] pre-clamp acc range: [{acc_lo}, {acc_hi}]  "
          f"(int16 safe if within [-32768, 32767])")
    if acc_lo < -32768 or acc_hi > 32767:
        print("[quant] WARNING: accumulator overflows int16!")

    # ── Samples ──
    Path(args.samples).parent.mkdir(parents=True, exist_ok=True)
    write_samples(args.samples, qnet)

    print("\n[sanity] selected sample evals (FEN -> cp, STM POV):")
    show = [
        ("startpos",       "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("up-a-queen",     "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("down-a-queen",   "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"),
        ("KQ-vs-K",        "8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1"),
        ("K-vs-KQ",        "4k3/4q3/8/8/8/8/8/4K3 w - - 0 1"),
    ]
    for label, fen in show:
        print(f"   {qnet.eval_fen(fen):+6d}  {label:15s}  {fen}")

    print(f"\n[done] peak RAM: {_rss_gb():.2f} GB")


if __name__ == '__main__':
    # On Windows, DataLoader with num_workers>0 requires __main__ guard.
    sys.exit(main())
