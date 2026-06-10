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
import functools
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
MAGIC_V2 = 0x4B4E5532     # "KNU2" — output-bucket format (u16 OB in header)
MAGIC_V3 = 0x4B4E5533     # "KNU3" — king-bucket format (u16 OB, u16 KB in header)

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


def output_bucket(piece_count: int, nb: int) -> int:
    """Output-bucket index from total piece count. MUST match C++ ob_index():
    bucket = clamp((piece_count - 2) // 4, 0, nb-1). For nb==1 always 0."""
    b = (piece_count - 2) // 4
    return 0 if b < 0 else (nb - 1 if b > nb - 1 else b)


def bucket_tensor(counts, nb: int):
    """Vectorised output_bucket over a tensor of piece counts (floor div, clamped)."""
    b = torch.div(counts - 2, 4, rounding_mode='floor')
    return b.clamp_(0, nb - 1).long()


def king_bucket_np(ksq, kb: int):
    """King-square -> bucket (HalfKP-style). ksq is the ORIENTED own-king square
    (0..63) for the perspective. Buckets primarily by king file-pair, then rank,
    so kingside/queenside is always distinguished. MUST match the C++ side when
    that is implemented. Supported kb: 1,4,8,16,32,64."""
    if kb == 1:
        return np.zeros_like(ksq)
    if kb == 64:
        return ksq.copy()
    file2 = (ksq % 8) // 2          # 0..3 (file pair)
    rank = ksq // 8                 # 0..7
    if kb == 4:
        return file2
    if kb == 8:
        return file2 + 4 * (rank // 4)
    if kb == 16:
        return file2 + 4 * (rank // 2)
    if kb == 32:
        return file2 + 4 * rank
    raise ValueError(f"unsupported kbuckets={kb}")


def king_bucket_scalar(oks: int, kb: int) -> int:
    """Scalar king_bucket (matches king_bucket_np and the C++ king_bucket)."""
    if kb == 1:
        return 0
    if kb == 64:
        return oks
    file2 = (oks % 8) // 2
    rank = oks // 8
    if kb == 4:
        return file2
    if kb == 8:
        return file2 + 4 * (rank // 4)
    if kb == 16:
        return file2 + 4 * (rank // 2)
    if kb == 32:
        return file2 + 4 * rank
    raise ValueError(f"unsupported kbuckets={kb}")


def king_bucket_torch(oks, kb: int):
    """Vectorised king_bucket over a long tensor of ORIENTED own-king squares.
    MUST match king_bucket_np / king_bucket_scalar / the C++ king_bucket EXACTLY.
    Supported kb: 1,4,8,16,32,64. Inputs are non-negative so // is floor div."""
    if kb == 1:
        return torch.zeros_like(oks)
    if kb == 64:
        return oks
    file2 = torch.div(oks % 8, 2, rounding_mode='floor')   # 0..3 (file pair)
    rank = torch.div(oks, 8, rounding_mode='floor')        # 0..7
    if kb == 4:
        return file2
    if kb == 8:
        return file2 + 4 * torch.div(rank, 4, rounding_mode='floor')
    if kb == 16:
        return file2 + 4 * torch.div(rank, 2, rounding_mode='floor')
    if kb == 32:
        return file2 + 4 * rank
    raise ValueError(f"unsupported kbuckets={kb}")


def piece_features(pieces, persp: int, kb: int):
    """King-bucketed feature indices for `pieces` (list of (color,ptype,sq)) from
    perspective `persp`. The bucket uses persp's OWN king square. kb=1 → plain 768."""
    king_sq = next(sq for (c, t, sq) in pieces if t == 5 and c == persp)
    oks = king_sq if persp == WHITE else (king_sq ^ 56)
    kbk = king_bucket_scalar(oks, kb)
    base = kbk * INPUT_SIZE
    out = []
    for (c, t, sq) in pieces:
        os = sq if persp == WHITE else (sq ^ 56)
        cr = 0 if c == persp else 1
        out.append(base + cr * 384 + t * 64 + os)
    return out


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


def collate_fn(batch, kb: int = 1):
    """Vectorised collate: a list of raw records -> EmbeddingBag inputs.

    Computes every active feature index for the whole batch with numpy array ops
    (no Python per-piece loop), then packs them into a flat index tensor + offsets
    for nn.EmbeddingBag(mode='sum'). Both perspectives share the SAME piece order,
    so they share offsets. The feature_index math here MUST match feature_index()
    and the C++ inference EXACTLY (bit-exact contract). kb>1 adds the per-perspective
    king-bucket offset (kb*768), matching gpu_batch_inputs / piece_features.
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

    if kb > 1:
        # Per-sample own-king square per perspective (king whose colour == persp;
        # exactly one per row). Bucket the oriented square and shift by kb*768.
        col_all = ((pieces >> 4) & 1).astype(np.int64)    # [B,32]
        pt_all  = (pieces & 0x0F).astype(np.int64)        # [B,32]
        sq_all  = squares.astype(np.int64)                # [B,32]
        isk = (pt_all == 5) & valid                       # [B,32]
        ks_stm = (sq_all * (isk & (col_all == stm[:, None]))).sum(axis=1)   # [B]
        oks_stm = np.where(stm == WHITE, ks_stm, ks_stm ^ 56)
        sf = sf + king_bucket_np(oks_stm, kb)[rows] * INPUT_SIZE
        nstm_b = stm ^ 1
        ks_nst = (sq_all * (isk & (col_all == nstm_b[:, None]))).sum(axis=1)
        oks_nst = np.where(nstm_b == WHITE, ks_nst, ks_nst ^ 56)
        nf = nf + king_bucket_np(oks_nst, kb)[rows] * INPUT_SIZE

    offsets = np.zeros(B, dtype=np.int64)
    if B > 1:
        np.cumsum(n[:-1], out=offsets[1:])
    off_t = torch.from_numpy(offsets)
    counts_t = torch.from_numpy(np.ascontiguousarray(n))   # [B] piece counts
    return (torch.from_numpy(sf), off_t,
            torch.from_numpy(nf), off_t.clone(), scores_t, results_t, counts_t)


# ──────────────────────────────────────────────────────────────────────────────
# Model — HL is a parameter; EXACT same architecture as before, just flexible.
# ──────────────────────────────────────────────────────────────────────────────
class NNUE(nn.Module):
    def __init__(self, hl: int, squared: bool = False, buckets: int = 1, kbuckets: int = 1,
                 factorizer: bool = False, l2clamp: float = 64.0):
        super().__init__()
        self.hl = hl
        self.squared = squared  # SCReLU (clamp then square) vs CReLU
        self.buckets = buckets  # output buckets by piece count (1 = legacy)
        self.kbuckets = kbuckets  # king-square input buckets (1 = none / plain 768)
        self.l2clamp = float(l2clamp)  # output-weight clamp; 1.98 -> |W2q|<=127 (fast kernel)
        # Factorizer (virtual features): a king-INDEPENDENT parallel FT (plain 768->HL)
        # summed into the real bucketed FT during TRAINING ONLY. It learns the pattern
        # shared across all king buckets (gets KB* more gradient), so rarely-visited
        # buckets inherit a sensible baseline -> fixes the undertraining that made plain
        # KB regress (-88 Elo). Folded into every real bucket at quantize time, so the
        # exported binary + C++ inference are BYTE-IDENTICAL to a plain KB net.
        self.factorizer = bool(factorizer) and kbuckets > 1
        # Feature transformer as a sparse EmbeddingBag (sum over active features).
        # Input space is kbuckets*768 (king-bucketed HalfKP-style); kbuckets=1 is the
        # plain 768. ft.weight is [kbuckets*768, HL]; dense-equivalent W1 is its
        # transpose. b1 is a separate bias added after the bag sum.
        self.ft = nn.EmbeddingBag(kbuckets * INPUT_SIZE, hl, mode='sum')
        self.b1 = nn.Parameter(torch.zeros(hl))
        # W2: [buckets, 2*HL] — one output row per piece-count bucket.
        self.l2 = nn.Linear(2 * hl, buckets)
        # Match the old nn.Linear default init scale so training dynamics/quant
        # ranges stay comparable.
        nn.init.kaiming_uniform_(self.ft.weight, a=5 ** 0.5)
        if self.factorizer:
            # King-independent virtual FT: plain 768 inputs (no king-bucket offset).
            self.ft_virtual = nn.EmbeddingBag(INPUT_SIZE, hl, mode='sum')
            nn.init.kaiming_uniform_(self.ft_virtual.weight, a=5 ** 0.5)

    def forward(self, s_idx, s_off, n_idx, n_off, bucket):
        acc_stm = self.ft(s_idx, s_off) + self.b1
        acc_nst = self.ft(n_idx, n_off) + self.b1
        if self.factorizer:
            # Virtual feature index = real index stripped of the king-bucket offset
            # (the offset is a multiple of 768, so % INPUT_SIZE recovers it). Same
            # offsets (piece order is identical). Summed into the bucketed accumulator.
            acc_stm = acc_stm + self.ft_virtual(s_idx % INPUT_SIZE, s_off)
            acc_nst = acc_nst + self.ft_virtual(n_idx % INPUT_SIZE, n_off)
        # Clipped ReLU in float domain: clamp to [0,1] (quant uses [0,QA]).
        c_stm = torch.clamp(acc_stm, 0.0, 1.0)
        c_nst = torch.clamp(acc_nst, 0.0, 1.0)
        if self.squared:  # SCReLU: square the clamped activation
            c_stm = c_stm * c_stm
            c_nst = c_nst * c_nst
        x = torch.cat([c_stm, c_nst], dim=1)   # [B, 2*HL], stm first
        out = self.l2(x)                       # [B, buckets] win-logits
        # Select each sample's bucket. For buckets==1 this is column 0, identical
        # to the old single-output net. Float eval in cp == y * SCALE.
        return out.gather(1, bucket.view(-1, 1)).squeeze(1)  # [B]

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
        if self.factorizer:
            # Folded (real+virtual) weight must stay int16-safe; clamp the virtual
            # table tighter so |real|+|virtual| <= ~1.98 after the fold.
            self.ft_virtual.weight.clamp_(-0.99, 0.99)
            self.ft.weight.clamp_(-0.99, 0.99)
        self.b1.clamp_(-1.98, 1.98)
        self.l2.weight.clamp_(-self.l2clamp, self.l2clamp)


# ──────────────────────────────────────────────────────────────────────────────
# Quantized integer forward pass (pure numpy int math == the C++ contract).
# ──────────────────────────────────────────────────────────────────────────────
class QuantNet:
    def __init__(self, W1q, b1q, W2q, b2q, hl: int, squared: bool = False,
                 buckets: int = 1, kbuckets: int = 1):
        self.W1q = W1q.astype(np.int64)  # [HL, kbuckets*INPUT]
        self.b1q = b1q.astype(np.int64)  # [HL]
        self.W2q = np.asarray(W2q).astype(np.int64).reshape(buckets, 2 * hl)  # [buckets, 2*HL]
        self.b2q = np.asarray(b2q).astype(np.int64).reshape(buckets)          # [buckets]
        self.hl = hl
        self.squared = squared
        self.buckets = buckets
        self.kbuckets = kbuckets
        # SCReLU squares the [0,QA] activation, adding one factor of QA to scale.
        self.den = (QA * QA * QB) if squared else (QA * QB)

    def eval_features(self, stm_feat_idx, nst_feat_idx, bucket: int = 0) -> int:
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
        dot = int(np.dot(acc, self.W2q[bucket]))
        out = int(self.b2q[bucket]) + dot
        # Python floor division — matches the C++ floor-div snippet exactly.
        return int(out * SCALE // self.den)

    def eval_fen(self, fen: str) -> int:
        fields = fen.split()
        board = fields[0]
        stm = WHITE if fields[1] == 'w' else BLACK
        pieces = parse_fen_pieces(board)
        sf = piece_features(pieces, stm, self.kbuckets)
        nf = piece_features(pieces, stm ^ 1, self.kbuckets)
        return self.eval_features(sf, nf, output_bucket(len(pieces), self.buckets))

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
    buckets = getattr(model, 'buckets', 1)
    kbuckets = getattr(model, 'kbuckets', 1)
    W1 = model.ft.weight.detach().cpu().numpy().T.copy()   # [HL, kbuckets*INPUT]
    if getattr(model, 'factorizer', False):
        # Fold the king-independent virtual table into EVERY real bucket. After this
        # the exported W1 (and thus the binary + C++ inference) is byte-identical to a
        # plain KB net — the factorizer existed only during training.
        Wv = model.ft_virtual.weight.detach().cpu().numpy().T   # [HL, INPUT]
        for b in range(kbuckets):
            W1[:, b * INPUT_SIZE:(b + 1) * INPUT_SIZE] += Wv
        print(f"[quant] folded virtual factorizer into {kbuckets} king buckets")
    b1 = model.b1.detach().cpu().numpy()            # [HL]
    W2 = model.l2.weight.detach().cpu().numpy()     # [buckets, 2*HL]
    b2 = model.l2.bias.detach().cpu().numpy()       # [buckets]

    squared = getattr(model, 'squared', False)
    W1q = np.round(W1 * QA).astype(np.int32)
    b1q = np.round(b1 * QA).astype(np.int32)
    W2q = np.round(W2 * QB).astype(np.int32)        # [buckets, 2*HL]
    # SCReLU's squared activation carries an extra QA factor into the output scale.
    b2q = np.round(b2 * (QA * QA * QB if squared else QA * QB)).astype(np.int64)  # [buckets]

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
    return (QuantNet(W1q, b1q, W2q, b2q, hl, squared, buckets, kbuckets),
            (W1q, b1q, W2q, b2q))


# ──────────────────────────────────────────────────────────────────────────────
# Binary export — see README_NNUE.md for byte layout.
#   Header:  u32 magic | u16 HL | u16 QA | u16 QB | u16 SCALE  (12 bytes)
#   W1q:     HL*INPUT_SIZE int16, row-major [output o][input i]
#   b1q:     HL int16
#   W2q:     2*HL int16  (0..HL-1 = stm, HL..2HL-1 = nonstm)
#   b2q:     int32
# HL is written in the header — C++ reads it back at load time.
# ──────────────────────────────────────────────────────────────────────────────
def export_bin(path: str, hl: int, W1q, b1q, W2q, b2q, buckets: int = 1, kbuckets: int = 1):
    ftin = kbuckets * INPUT_SIZE
    W1b = np.ascontiguousarray(W1q, dtype='<i2')                            # [HL, ftin]
    b1b = np.ascontiguousarray(b1q, dtype='<i2')                            # [HL]
    W2a = np.ascontiguousarray(np.asarray(W2q).reshape(buckets, 2 * hl), dtype='<i2')
    b2a = np.asarray(b2q).reshape(buckets).astype('<i4')
    with open(path, 'wb') as f:
        if kbuckets > 1:
            # KNU3 — u16 OB then u16 KB after SCALE; W1 is [HL, KB*768].
            f.write(struct.pack('<I', MAGIC_V3))
            f.write(struct.pack('<HHHHHH', hl, QA, QB, SCALE, buckets, kbuckets))
            f.write(W1b.tobytes()); f.write(b1b.tobytes())
            f.write(W2a.tobytes()); f.write(b2a.tobytes())
            expected = 4 + 12 + hl * ftin * 2 + hl * 2 + buckets * 2 * hl * 2 + buckets * 4
        elif buckets > 1:
            # KNU2 format — u16 OB after SCALE, W2[OB][2*HL] then b2[OB].
            f.write(struct.pack('<I', MAGIC_V2))
            f.write(struct.pack('<HHHHH', hl, QA, QB, SCALE, buckets))
            f.write(W1b.tobytes()); f.write(b1b.tobytes())
            f.write(W2a.tobytes())          # bucket-major [OB][2*HL]
            f.write(b2a.tobytes())          # [OB] int32
            expected = 4 + 10 + hl * INPUT_SIZE * 2 + hl * 2 + buckets * 2 * hl * 2 + buckets * 4
        else:
            # Legacy KNUE format — byte-identical to the original single-bucket net.
            f.write(struct.pack('<I', MAGIC))
            f.write(struct.pack('<HHHH', hl, QA, QB, SCALE))
            f.write(W1b.tobytes()); f.write(b1b.tobytes())
            f.write(W2a.tobytes())
            f.write(struct.pack('<i', int(b2a[0])))
            expected = 4 + 8 + hl * INPUT_SIZE * 2 + hl * 2 + 2 * hl * 2 + 4
    size = Path(path).stat().st_size
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

def build_gpu_dataset(cache_path: str, total: int, device, kb: int = 1):
    # Memory-COMPACT GPU dataset: keep the raw pieces/squares (uint8, ~75 B/pos)
    # resident and compute the per-perspective feature indices on the fly per batch
    # (see gpu_batch_inputs). The old path precomputed int16 sf/nf tensors (~144
    # B/pos), which overflowed 16 GB VRAM past ~85M positions and thrashed; this
    # fits 112M in ~8 GB and scales to ~200M. The raw tensors are king-INDEPENDENT,
    # so the SAME cache serves any kb — the king-bucket offset is applied per batch
    # in gpu_batch_inputs (kb is accepted here only for signature symmetry).
    npz = cache_path + ".gpures2.npz"
    if Path(npz).exists() and Path(npz).stat().st_size > 0:
        print(f"[data] loading cached compact GPU tensors from {npz}")
        d = np.load(npz)
        return (torch.from_numpy(d['pc']).to(device),
                torch.from_numpy(d['sq']).to(device),
                torch.from_numpy(d['st']).to(device),
                torch.from_numpy(d['ln']).to(device),
                torch.from_numpy(d['sc']).to(device),
                torch.from_numpy(d['rs']).to(device))
    mm = np.memmap(cache_path, dtype=RECORD_DTYPE, mode='r', shape=(total,))
    pc = np.ascontiguousarray(mm['pieces'])                       # [N,32] uint8 (color<<4|ptype)
    sq = np.ascontiguousarray(mm['squares'])                      # [N,32] uint8
    st = np.ascontiguousarray(mm['stm'])                          # [N]   uint8
    ln = np.ascontiguousarray(mm['n']).astype(np.int16)           # [N]   piece count
    sc = np.ascontiguousarray(mm['score']).astype(np.float32)
    rs = np.ascontiguousarray(mm['result']).astype(np.float32)
    del mm
    try:
        np.savez(npz, pc=pc, sq=sq, st=st, ln=ln, sc=sc, rs=rs)
        print(f"[data] cached compact GPU tensors -> {npz}")
    except Exception as e:
        print(f"[data] (cache save skipped: {e})")
    return (torch.from_numpy(pc).to(device), torch.from_numpy(sq).to(device),
            torch.from_numpy(st).to(device), torch.from_numpy(ln).to(device),
            torch.from_numpy(sc).to(device), torch.from_numpy(rs).to(device))

def gpu_batch_inputs(pc_g, sq_g, st_g, len_g, rows, device, kb: int = 1):
    """Compute both perspectives' EmbeddingBag (flat indices + offsets) for a batch
    of rows ON GPU, from the compact raw pieces/squares/stm tensors. Mirrors the
    numpy feature formula exactly: sf = kb_base + (color!=stm)*384 + ptype*64 +
    oriented_sq, where kb_base = king_bucket(oriented own-king sq)*768 (kb=1 -> 0)."""
    pc  = pc_g[rows].long()                                       # [B,32] color<<4|ptype
    sq  = sq_g[rows].long()                                       # [B,32]
    stm = st_g[rows].long().unsqueeze(1)                          # [B,1]
    L   = len_g[rows].long()                                      # [B]
    mask = torch.arange(MAXP, device=device)[None, :] < L[:, None]  # [B,32]
    color = (pc >> 4) & 1
    ptype = pc & 0x0F
    os_s = torch.where(stm == WHITE, sq, sq ^ 56)
    sf = (color != stm).long() * 384 + ptype * 64 + os_s         # [B,32]
    nstm = stm ^ 1
    os_n = torch.where(nstm == WHITE, sq, sq ^ 56)
    nf = (color != nstm).long() * 384 + ptype * 64 + os_n
    if kb > 1:
        # King-bucket offset: per perspective, find the OWN king square (the king
        # whose colour == perspective; exactly one per row, padding slots are pawns
        # so contribute 0), orient it, bucket it, shift the feature block by kb*768.
        is_king = (ptype == 5)                                   # [B,32]
        ks_stm = (sq * (is_king & (color == stm)).long()).sum(dim=1)   # [B] own-king sq
        oks_stm = torch.where(stm.squeeze(1) == WHITE, ks_stm, ks_stm ^ 56)
        sf = sf + (king_bucket_torch(oks_stm, kb) * INPUT_SIZE).unsqueeze(1)
        ks_nst = (sq * (is_king & (color == nstm)).long()).sum(dim=1)
        oks_nst = torch.where(nstm.squeeze(1) == WHITE, ks_nst, ks_nst ^ 56)
        nf = nf + (king_bucket_torch(oks_nst, kb) * INPUT_SIZE).unsqueeze(1)
    s_flat = sf[mask]
    n_flat = nf[mask]
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
    ap.add_argument('--l2clamp', type=float, default=64.0,
                    help='clamp |output-layer weight|; 1.98 keeps |W2q|<=127 so the '
                         'fast 16-wide int16 madd inference kernel is bit-exact')
    ap.add_argument('--step', type=int, default=15,
                    help='StepLR step size in epochs (LR *= gamma every --step epochs). '
                         'Scale with --epochs so the high-LR phase grows proportionally.')
    ap.add_argument('--hl', type=int, default=512,
                    help='Hidden layer size per perspective (default 512)')
    ap.add_argument('--buckets', type=int, default=1,
                    help='Output buckets by piece count (default 1 = legacy single '
                         'output, KNUE format). >1 writes the KNU2 format and needs '
                         'the engine built with -DNNUE_OB=<n>.')
    ap.add_argument('--kbuckets', type=int, default=1,
                    help='King-square input buckets (HalfKP-style, default 1 = plain '
                         '768). >1 is an experiment: trains + reports val loss, but '
                         'skips quantize/export until the C++ side supports it.')
    ap.add_argument('--factorizer', action='store_true',
                    help='With kbuckets>1: add a king-independent virtual feature table '
                         'during training (folded into every bucket at export). Fixes '
                         'undertrained king buckets; binary + inference are unchanged.')
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
    model = NNUE(HL, squared=squared, buckets=args.buckets, kbuckets=args.kbuckets,
                 factorizer=args.factorizer, l2clamp=args.l2clamp).to(device)
    print(f"[model] activation={args.activation}  buckets={args.buckets}  kbuckets={args.kbuckets}"
          f"  factorizer={getattr(model, 'factorizer', False)}")
    n_params = sum(p.numel() for p in model.parameters())
    _hdr = 16 if args.kbuckets > 1 else (14 if args.buckets > 1 else 12)
    _sz = (_hdr + HL * args.kbuckets * INPUT_SIZE * 2 + HL * 2
           + args.buckets * 2 * HL * 2 + args.buckets * 4)
    print(f"[model] HL={HL}  buckets={args.buckets}  kbuckets={args.kbuckets}  "
          f"params={n_params:,}  ({n_params*4/1e6:.2f} MB float32)  "
          f"export size={_sz/1024:.1f} KB")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step, gamma=0.3)
    loss_fn = nn.MSELoss()

    first_train, first_val, last_train, last_val = None, None, 0.0, 0.0
    best_val = float('inf')

    if use_gpu:
        # ── GPU-resident path: whole dataset on the GPU, no CPU dataloader ────
        print("[data] building GPU-resident dataset (one-time) ...")
        t_b = time.time()
        pc_g, sq_g, st_g, len_g, sc_g, rs_g = build_gpu_dataset(cache_path, total, device, args.kbuckets)
        tr_g = torch.from_numpy(train_idx).to(device)
        va_g = torch.from_numpy(val_idx).to(device)
        gb = (pc_g.numel() + sq_g.numel() + st_g.numel() + len_g.numel()*2
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
                    si, so, ni, no = gpu_batch_inputs(pc_g, sq_g, st_g, len_g, rows, device, args.kbuckets)
                    bk = bucket_tensor(len_g[rows], args.buckets)
                    pred = torch.sigmoid(model(si, so, ni, no, bk))
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
                si, so, ni, no = gpu_batch_inputs(pc_g, sq_g, st_g, len_g, rows, device, args.kbuckets)
                bk = bucket_tensor(len_g[rows], args.buckets)
                pred = torch.sigmoid(model(si, so, ni, no, bk))
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
        loader_kw = dict(batch_size=args.batch,
                         collate_fn=functools.partial(collate_fn, kb=args.kbuckets),
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
                for si, so, ni, no, sc, rs, cnts in val_loader:
                    bk = bucket_tensor(cnts.to(device), args.buckets)
                    pred = torch.sigmoid(model(si.to(device), so.to(device),
                                               ni.to(device), no.to(device), bk))
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
            for si, so, ni, no, sc, rs, cnts in ep_loader:
                bk = bucket_tensor(cnts.to(device), args.buckets)
                pred = torch.sigmoid(model(si.to(device), so.to(device),
                                           ni.to(device), no.to(device), bk))
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
    export_bin(args.out, HL, W1q, b1q, W2q, b2q, args.buckets, args.kbuckets)

    # ── Float vs quant agreement on a validation subset ──
    model.eval()
    check_n = min(2000, n_val)
    check_idx = val_idx[:check_n]
    check_ds = StreamingNNUEDataset(cache_path, check_idx)
    check_loader = DataLoader(check_ds, batch_size=check_n, shuffle=False,
                              collate_fn=functools.partial(collate_fn, kb=args.kbuckets),
                              num_workers=0)
    with torch.no_grad():
        si, so, ni, no, _, _, cnts = next(iter(check_loader))
        bk = bucket_tensor(cnts.to(device), args.buckets)
        yf = model(si.to(device), so.to(device),
                   ni.to(device), no.to(device), bk).cpu().numpy() * SCALE

    mm = np.memmap(cache_path, dtype=RECORD_DTYPE, mode='r', shape=(total,))
    diffs = []
    acc_lo, acc_hi = 0, 0
    for k, pi in enumerate(check_idx):
        if k >= len(yf):
            break
        rec = mm[pi]
        stm = int(rec['stm'])
        n_p = int(rec['n'])
        pcs = [((int(rec['pieces'][j]) >> 4) & 1, int(rec['pieces'][j]) & 0x0F,
                int(rec['squares'][j])) for j in range(n_p)]
        sf = piece_features(pcs, stm, args.kbuckets)
        nf = piece_features(pcs, stm ^ 1, args.kbuckets)
        q = qnet.eval_features(sf, nf, output_bucket(n_p, args.buckets))
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
