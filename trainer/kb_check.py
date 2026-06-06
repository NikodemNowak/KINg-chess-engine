#!/usr/bin/env python3
"""Correctness gate for king-bucket feature indexing. Verifies the THREE feature
paths agree bit-for-bit for kb in {1,4,8,16,32,64}:
  - collate_fn(kb)        (CPU DataLoader path + end-of-train self-check)
  - gpu_batch_inputs(kb)  (GPU-resident training path)
  - piece_features(kb)    (numpy reference used by export + engine contract)
kb=1 MUST be byte-identical to the legacy (no king term) indices.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_nnue as T

CACHE = "/mnt/c/Users/nikod/Documents/uni/chess/data/all2_sf.bin"
if len(sys.argv) > 1:
    CACHE = sys.argv[1]
N = 256

total = Path(CACHE).stat().st_size // T.RECORD_DTYPE.itemsize
mm = np.memmap(CACHE, dtype=T.RECORD_DTYPE, mode='r', shape=(total,))
# spread the sample across the file
idx = np.linspace(0, total - 1, N).astype(np.int64)
recs = np.ascontiguousarray(mm[idx])


def reference_sets(rec, kb):
    """piece_features() per perspective -> (sorted stm feats, sorted nstm feats)."""
    stm = int(rec['stm']); npc = int(rec['n'])
    pcs = [((int(rec['pieces'][j]) >> 4) & 1, int(rec['pieces'][j]) & 0x0F,
            int(rec['squares'][j])) for j in range(npc)]
    sf = sorted(T.piece_features(pcs, stm, kb))
    nf = sorted(T.piece_features(pcs, stm ^ 1, kb))
    return sf, nf


def unpack_bag(flat, offsets, B):
    """EmbeddingBag flat indices + offsets -> list of per-sample sorted feature lists."""
    flat = np.asarray(flat); offsets = np.asarray(offsets)
    out = []
    for b in range(B):
        lo = int(offsets[b])
        hi = int(offsets[b + 1]) if b + 1 < B else len(flat)
        out.append(sorted(int(x) for x in flat[lo:hi]))
    return out


fails = 0
for kb in (1, 4, 8, 16, 32, 64):
    # collate path
    batch = [recs[i] for i in range(N)]
    sf_c, off_c, nf_c, off2_c, *_ = T.collate_fn(batch, kb=kb)
    col_sf = unpack_bag(sf_c.numpy(), off_c.numpy(), N)
    col_nf = unpack_bag(nf_c.numpy(), off2_c.numpy(), N)

    # gpu path (run on CPU device — same code)
    pc_g = torch.from_numpy(np.ascontiguousarray(recs['pieces']))
    sq_g = torch.from_numpy(np.ascontiguousarray(recs['squares']))
    st_g = torch.from_numpy(np.ascontiguousarray(recs['stm']))
    ln_g = torch.from_numpy(np.ascontiguousarray(recs['n']).astype(np.int16))
    rows = torch.arange(N)
    gsf, goff, gnf, goff2 = T.gpu_batch_inputs(pc_g, sq_g, st_g, ln_g, rows,
                                               torch.device('cpu'), kb)
    gpu_sf = unpack_bag(gsf.numpy(), goff.numpy(), N)
    gpu_nf = unpack_bag(gnf.numpy(), goff2.numpy(), N)

    bad = 0
    maxfeat = kb * 768
    for i in range(N):
        ref_sf, ref_nf = reference_sets(recs[i], kb)
        if not (ref_sf == col_sf[i] == gpu_sf[i] and ref_nf == col_nf[i] == gpu_nf[i]):
            bad += 1
            if bad <= 2:
                print(f"  kb={kb} sample {i} MISMATCH")
                print(f"    ref_sf ={ref_sf}")
                print(f"    col_sf ={col_sf[i]}")
                print(f"    gpu_sf ={gpu_sf[i]}")
        # range check
        allf = ref_sf + ref_nf
        if allf and (min(allf) < 0 or max(allf) >= maxfeat):
            bad += 1
            print(f"  kb={kb} sample {i} OUT OF RANGE [0,{maxfeat}) got {min(allf)}..{max(allf)}")
    status = "OK" if bad == 0 else f"FAIL ({bad}/{N})"
    print(f"kb={kb:2d}: {status}  (feat range [0,{maxfeat}))")
    fails += bad

print("\nRESULT:", "ALL PATHS AGREE" if fails == 0 else f"{fails} MISMATCHES")
sys.exit(1 if fails else 0)
