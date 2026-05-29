# KINg

KINg is a UCI chess engine written in C++20, developed for the KINo AI Chess Engine Competition.

## Building locally (Windows + MinGW)

```
make unit
```

This runs CMake with the MinGW Makefiles generator, builds both the `engine` and `unit_tests` targets, and runs the full test suite via ctest.

Quick perft check at depth 5:

```
make perft
```

## Building in Docker

```
make build
```

Produces a slim `ubuntu:22.04`-based image with the UCI binary at `/usr/local/bin/engine`.

## Running in Docker (competition contract)

The organizer harness starts the engine with:

```
docker run --rm -i --init --memory 2g --network none king:latest
```

The binary at `/usr/local/bin/engine` must speak UCI on stdin/stdout. No GPU, no network, 2 GB RAM cap.

## Self-play smoke test (requires cutechess-cli on PATH)

```
make test-uci    # UCI handshake only
make test-game   # two-game self-play via engine-adapter.sh
```

`engine-adapter.sh` wraps `docker run` so that cutechess-cli can treat the container as a local UCI binary.

## Provenance

Evaluation weights (material values, piece-square tables, and the handcrafted
structural-term weights) are Texel-tuned with the in-engine coordinate-descent
tuner (`engine tune <epd>`, see `src/tune.cpp`). Training data is the public
Zurichess **quiet-labeled.epd** dataset (~725k FEN + game-result positions,
Alexandru Moșoi et al.), mirrored at
<https://github.com/KierenP/ChessTrainingSets>. The base piece-square tables are
PeSTO by Ronald Friederich (Chess Programming Wiki). The dataset file itself is
not committed (gitignored); fetch it before re-running the tuner.

TODO: list remaining techniques and sources (alpha-beta, MCTS, NNUE references),
third-party libraries, AI-conversation links — to be completed before submission
per competition regulations.
