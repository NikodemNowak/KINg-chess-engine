# KINg NNUE — architecture, quantization & binary format

This document is the **authoritative contract** between the PyTorch trainer
(`trainer/train_nnue.py`) and the C++ inference (next task). The C++ evaluator
must reproduce the quantized integer eval **bit-for-bit**, and in particular
must reproduce every line in `trainer/nnue_samples.txt` exactly.

All multi-byte values in the binary are **little-endian**.

---

## 1. Architecture

Perspective net, **`(768 → HL)×2 → 1`**, clipped-ReLU.  Default **`HL = 512`**
for new training runs (pass `--hl` to override).  The committed net
`nets/king_nnue.bin` stores its actual HL in the header (bytes 4–5); the C++
loader reads it back at initialisation time.

```
            stm features (768)            nonstm features (768)
                  │                              │
            W1 [HL×768] + b1             W1 [HL×768] + b1      (SAME W1,b1)
                  │                              │
              acc_stm (HL)                  acc_nst (HL)
                  │  clipped-ReLU                │  clipped-ReLU
            crelu(acc_stm)                crelu(acc_nst)
                  └──────────────┬───────────────┘
                       concat (stm first)  → x (2*HL)
                                 │
                        W2 [1×(2*HL)] + b2
                                 │
                              y (scalar)
```

* `HL` (accumulator size per perspective) — default **512** for new training; stored in the binary header so the C++ can load any value without recompilation.
* Both perspectives share the **same** feature transformer `W1`, `b1`.
* The concatenation order is **stm perspective first**, then nonstm.
* Output `y` is the eval from the **side-to-move POV** (positive = good for the
  side to move), matching the engine's `evaluate()` convention.

### 1.1 Feature index

For a piece of color `c ∈ {WHITE=0, BLACK=1}`, type
`t ∈ {PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5}`, on square
`s ∈ 0..63` (square index is rank-major little-endian: `s = rank*8 + file`,
`a1 = 0`, `h8 = 63` — identical to `src/types.hpp`), evaluated from perspective
`P ∈ {WHITE, BLACK}`:

```
os = (P == WHITE) ? s : (s ^ 56);   // vertical mirror (flip rank) for black POV
cr = (c == P)     ? 0 : 1;          // 0 = own piece, 1 = enemy piece
idx = cr * 384 + t * 64 + os;       // range 0..767
```

So feature layout per perspective is: `[own P,N,B,R,Q,K | enemy P,N,B,R,Q,K]`,
each piece type occupying 64 squares.

### 1.2 Forward pass (float, training domain)

```
acc_P = b1 + Σ_{features of P} W1[:, idx]          # W1 shape [HL,768], b1 [HL]
crelu(v) = clamp(v, 0.0, 1.0)                      # float clipped-ReLU
x = concat( crelu(acc_stm), crelu(acc_nst) )       # size 2*HL, stm first
y = b2 + W2 · x                                    # W2 [1,2*HL], b2 scalar
```

The training loss treats `y` as a centipawn-scale logit:
`MSE( sigmoid(y / SCALE), target )`.

---

## 2. Training target

The dataset (`data/nnue_data.txt`, produced by `engine.exe datagen`) stores each
line as `FEN | score_cp_white_pov | result_white_pov`, with
`result ∈ {0.0, 0.5, 1.0}` (1.0 = White won).

Both `score` and `result` are converted to **side-to-move POV** before forming
the target (when stm is Black: `score → −score`, `result → 1 − result`):

```
target = 0.6 * sigmoid(score_stm / 400) + 0.4 * result_stm
loss   = MSE( sigmoid(y / 400), target )
```

Optimizer: Adam, lr `1e-3` with StepLR decay (×0.3 every 15 epochs),
batch `16384`, 45 epochs (val loss plateaus well before then).

---

## 3. Quantization

Constants: **`QA = 255`, `QB = 64`, `SCALE = 400`**.

```
W1q = round(W1 * QA)          # int16,  shape [HL, 768]
b1q = round(b1 * QA)          # int16,  shape [HL]       → accumulator is int16
W2q = round(W2 * QB)          # int16,  shape [2*HL]
b2q = round(b2 * QA * QB)     # int32   scalar
```

Clipped-ReLU in the quantized domain clamps to `[0, QA]` (integer):

```
cr(v) = clamp(v, 0, QA)       # v is an int16 accumulator entry
```

### 3.1 Quantized eval (the EXACT integer computation the C++ performs)

```
acc_stm[i] = b1q[i] + Σ_{features of stm}    W1q[i, idx]      # int16 accumulate
acc_nst[i] = b1q[i] + Σ_{features of nonstm} W1q[i, idx]
acc = concat( cr(acc_stm), cr(acc_nst) )                      # 2*HL ints in [0,QA]

sum = b2q + Σ_{i=0..2*HL-1} acc[i] * W2q[i]                   # accumulate in int32/int64
eval_cp = sum * SCALE / (QA * QB)                             # integer division
```

* `acc` entries are in `[0, 255]`; `W2q` fits in int16. The product `acc[i]*W2q[i]`
  and the running `sum` must be accumulated in **at least int32** (int64 is safe;
  worst case at HL=512: 1024 · 255 · 32767 ≈ 8.6e9 which exceeds int32 — so
  **use int64 for the dot-product accumulator**, then the final value fits comfortably).
* `eval_cp` is from the **side-to-move POV**.
* Integer division truncates toward zero (C/C++ `/`, and Python `int(a*b//c)` for
  non-negative; the reference uses `out * SCALE // (QA*QB)` matching C++ trunc for
  the signs that occur — both round toward −∞ only for negatives, which matches
  C++ `/` toward zero **only if** sign handling is identical; see note below).

> **Sign / rounding note.** The reference Python uses Python floor-division `//`.
> For the C++ to match bit-for-bit it should compute the division the **same way**.
> The simplest portable choice is: compute `num = sum * SCALE` as int64, then
> `eval_cp = num / (QA*QB)` using C++ integer division (truncation toward zero).
> Python `//` floors toward −∞, which differs from C++ for negative numerators.
> **To avoid any mismatch, the C++ MUST replicate Python floor division**, e.g.:
> ```cpp
> int64_t num = (int64_t)sum * SCALE;
> int64_t den = (int64_t)QA * QB;          // 16320
> int64_t q = num / den;
> if ((num % den != 0) && ((num < 0) != (den < 0))) q -= 1;  // floor division
> eval_cp = (int)q;
> ```
> The sample file `trainer/nnue_samples.txt` is generated with Python floor
> division, so the C++ must match it with the snippet above.

### 3.2 Float-vs-quant agreement

The trainer reports mean / median / max centipawn difference between the float
forward pass and the quantized integer forward pass on a validation subset.
Expect agreement within a few centipawns (typically mean < 3 cp).

---

## 4. Binary format — `nets/king_nnue.bin`

Little-endian. `HL` is written in the header (read by the C++ loader at init).
Default for new training runs: `HL = 512`, `INPUT = 768`.

| Offset (bytes)        | Field      | Type            | Count       | Notes                                          |
|-----------------------|------------|-----------------|-------------|------------------------------------------------|
| 0                     | `magic`    | `uint32`        | 1           | `0x4B4E5545` ("KNUE")                          |
| 4                     | `HL`       | `uint16`        | 1           | accumulator size (e.g. 512)                    |
| 6                     | `QA`       | `uint16`        | 1           | `255`                                          |
| 8                     | `QB`       | `uint16`        | 1           | `64`                                           |
| 10                    | `SCALE`    | `uint16`        | 1           | `400`                                          |
| 12                    | `W1q`      | `int16`         | `HL * 768`  | **row-major `[output o][input i]`**: o outer   |
| 12 + HL*768*2         | `b1q`      | `int16`         | `HL`        |                                                |
| 12 + HL*768*2 + HL*2  | `W2q`      | `int16`         | `2*HL`      | `0..HL-1` = **stm**, `HL..2HL-1` = **nonstm** |
| + 2*HL*2              | `b2q`      | `int32`         | 1           |                                                |

* Header size = 12 bytes.
* **Total file size = `12 + HL*768*2 + HL*2 + 2*HL*2 + 4` bytes.**
  * HL=256: 394 768 bytes (old committed net).
  * HL=512: 789 524 bytes (default for the ≥30M retrain).

### 4.1 `W1q` indexing in the accumulator update

`W1q` is stored row-major as `[output][input]`. To add the column for feature
`idx` into the accumulator (`acc[o] += W1q[o, idx]`), element `(o, idx)` lives at
flat index `o * 768 + idx`. A cache-friendly C++ loop is:

```cpp
for (int o = 0; o < HL; ++o)
    acc[o] += W1q[o * 768 + idx];
```

(or, equivalently, transpose `W1q` to `[input][output]` once at load time so each
feature's HL weights are contiguous — that is the usual NNUE layout; either is
fine as long as the math is identical.)

### 4.2 `W2q` ordering

`W2q[0..HL-1]` multiply `cr(acc_stm)` and `W2q[HL..2*HL-1]` multiply
`cr(acc_nst)`, matching the **stm-first** concatenation.

---

## 5. Files

* `trainer/preprocess_nnue.py` — one-time text→binary conversion (72 B/pos,
  numpy memmap format).  Run once; re-run only if the dataset changes.
* `trainer/train_nnue.py` — streaming trainer (reads binary cache via memmap),
  model, quantize, export, sample generation.
* `trainer/README_NNUE.md` — this contract.
* `trainer/nnue_samples.txt` — `~25` lines `FEN <quantized_eval_cp_stm_pov>`; the
  C++ inference must reproduce each integer exactly.
* `nets/king_nnue.bin` — the exported quantized network.

Data lives under `data/` (git-ignored).

---

## 6. Reproduce

```powershell
# 1. Generate large dataset (~30M positions, CPU datagen)
build\engine.exe datagen data\nnue_big.txt ...

# 2. Pre-process text → binary cache (ONCE; ~2GB output for 30M positions)
py trainer\preprocess_nnue.py `
    --data data\nnue_big.txt --out data\nnue_big.bin

# 3. Train (streams from binary cache; HL=512 default)
py trainer\train_nnue.py `
    --cache data\nnue_big.bin `
    --out nets\king_nnue.bin `
    --samples trainer\nnue_samples.txt
```
