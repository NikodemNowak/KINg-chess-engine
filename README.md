# KINg

KINg is a UCI chess engine written in C++20, developed for the **KINo AI Chess
Engine Competition**.

| | |
|---|---|
| **Team** | EPSZ-team |
| **Members** | Bartosz Kołaciński, Nikodem Nowak |
| **Category** | Open (NNUE evaluation). A "No Deep Learning" build is also provided — see [Evaluation](#evaluation). |
| **Protocol** | UCI, over stdin/stdout |
| **Language** | C++20, CPU-only (no GPU at game time) |

---

## Using the engine

KINg speaks the [UCI protocol](https://www.chessprogramming.org/UCI) on
stdin/stdout, so it runs under any UCI GUI (CuteChess, Arena, Banksia, …) or
directly from a terminal:

```
uci
isready
position startpos moves e2e4 e7e5
go movetime 1000
```

It replies with `info` lines while searching and a final `bestmove`, e.g.:

```
info depth 12 seldepth 18 score cp 31 nodes 412233 nps 1850000 hashfull 23 time 222 pv g1f3 b8c6 ...
bestmove g1f3
```

### Supported UCI options

| Option | Type | Default | Meaning |
|---|---|---|---|
| `Hash` | spin | 64 | Transposition-table size in MB (1–1024). |
| `Threads` | spin | #cores | Search threads (Lazy SMP), 1–256. |
| `Move Overhead` | spin | 200 | Time (ms) reserved per move for I/O latency. |
| `Ponder` | check | false | Advertised for GUI compatibility. |
| `SyzygyPath` | string | (empty) | Folder(s) with Syzygy tablebases to probe. |
| `SyzygyProbeDepth` | spin | 1 | Minimum depth at which to probe tablebases. |

### Command-line tools

Besides the UCI loop, the binary exposes a few offline helpers:

```
engine perft <depth> [fen]   # count leaf nodes (move-generator self-test)
engine datagen <args>        # generate self-play training data
engine tune <args>           # in-engine Texel tuner (HCE weights)
```

---

## Building & testing locally

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

This builds the `engine` and `unit_tests` targets and runs the full unit-test
suite (perft, move generation, SEE, draw detection, UCI handshake, time
management, NNUE bit-exact gate, …). Quick move-generator check:

```
./build/engine perft 5
```

## Building in Docker

```
docker build -t chess-engine:latest .
```

Produces a slim `ubuntu:22.04`-based image with the UCI binary at
`/usr/local/bin/engine`. The build is CPU-only (no CUDA): a portable `-march=x86-64`
baseline with AVX2 selected **at runtime**, so the image runs on any x86-64 host
without risking an illegal-instruction crash.

## Running in Docker (competition contract)

The organizer harness starts the engine with:

```
docker run --rm -i --init --memory 2g --network none chess-engine:latest
```

The binary at `/usr/local/bin/engine` is the image `ENTRYPOINT` and speaks UCI on
stdin/stdout — no GPU, no network, 2 GB RAM. UCI handshake smoke test:

```
printf 'uci\nquit\n' | docker run --rm -i chess-engine:latest
```

---

## Techniques and algorithms

KINg is an **original** implementation: the board, move generator, search and
trainer were all written from scratch. The search techniques follow standard
descriptions from the [Chess Programming Wiki](https://www.chessprogramming.org).

**Board & move generation**
- Bitboard board representation; magic-bitboard sliding-piece attacks
- Zobrist hashing for the transposition table and repetition detection
- Staged pseudo-legal generation with a legality filter; `perft`-verified

**Search** — iterative-deepening Principal Variation Search (alpha-beta):
- Aspiration windows
- Lockless transposition table, shared across threads
- Null-move pruning and ProbCut
- Late Move Reductions (LMR) and Late Move Pruning (LMP)
- Singular extensions, including double / negative extensions
- Reverse futility pruning, futility pruning and shallow-SEE pruning
- History pruning; SEE-gated check extensions; mate-distance pruning
- Quiescence search with delta and SEE pruning
- Syzygy endgame tablebases (WDL + DTZ, via the Fathom library)

**Move ordering & history heuristics**
- TT move, then MVV-LVA captures split good/bad by Static Exchange Evaluation
- Killer moves, countermoves, butterfly history, capture history
- 1- and 2-ply continuation history; pawn-keyed correction history

**Parallel search** — Lazy SMP: helper threads share the transposition table,
search a staggered set of depths for tree diversity, and the final move is chosen
by best-thread voting (deepest, then highest score).

**Time management** — two-sided instability control (bank time when the best move
is stable, extend when it is unsettled), bounded by a hard limit so the engine
never forfeits on time.

**Robustness** (a crash or timeout is a lost game, so it is treated as a
first-class concern):
- Portable x86-64 baseline with **runtime** AVX2 dispatch — never executes an
  illegal instruction on a CPU without AVX2
- A crash handler that always has a legal fallback move armed
- Saturating time parsing and a hang-guard on malformed `go` commands

## Evaluation

Two evaluation back-ends are selected at build time via `-DEVAL=`:

- **NNUE** (`-DEVAL=NNUE`, default — "Open" category): a `(768→512)×2 → 8`
  perspective network — 512 neurons per side, eight piece-count output buckets,
  squared clipped-ReLU activation, int8/int16-quantized for a fast 16-wide SIMD
  kernel. It is trained with a custom PyTorch trainer (`trainer/`) on the order of
  10^8 self-play positions generated by the engine's own datagen (`src/datagen.cpp`)
  and then **re-labeled with Stockfish 18 used purely as an offline scoring
  oracle**. Stockfish is a free, publicly available engine and contributes only
  training-data labels — **no Stockfish code, and no Stockfish at game time**: the
  trained net is baked into the binary (`nets/king_int8_174m.bin`) and the engine
  plays completely standalone.

- **HCE** (`-DEVAL=HCE` — "No Deep Learning" category): a handcrafted, tapered
  evaluation built on **PeSTO** piece-square tables (Ronald Friederich, Chess
  Programming Wiki). All material values and structural-term weights are
  **Texel-tuned** with the in-engine coordinate-descent tuner (`src/tune.cpp`) on
  the public **Zurichess quiet-labeled.epd** dataset (Alexandru Moșoi et al.).

## Provenance and attribution

All search, board and trainer code is original to the EPSZ-team. Third-party
material is limited to:

- **PeSTO** piece-square tables (HCE evaluation) — Chess Programming Wiki
- **Stockfish 18** — used *only* as an offline oracle to label NNUE training data
- **Fathom** (`third_party/`) — Syzygy tablebase probing library
- **doctest** (`third_party/`) — unit-test framework
- **CMake**, **PyTorch** (training), **python-chess** (test tooling), **cutechess-cli** (testing)

### AI assistance disclosure (competition regulation §5.8)

KINg was developed with AI assistance: large parts of the implementation,
debugging and design were done interactively with **Claude / Claude Code**
(Anthropic).

AI conversation log (full transcript, exported): [`AI_conversation.zip`](https://github.com/NikodemNowak/KINg-chess-engine/raw/main/AI_conversation.zip)
