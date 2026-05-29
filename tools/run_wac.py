#!/usr/bin/env python3
"""
run_wac.py  --  WAC (Win At Chess) tactical suite runner.

Usage:
    python tools/run_wac.py <engine_path> [movetime_ms] [epd_path]

Arguments:
    engine_path   Path to a UCI engine binary.
    movetime_ms   Time per move in milliseconds (default: 200).
    epd_path      Path to the EPD file (default: tools/wac.epd relative to
                  this script's directory).
"""

import sys
import os
import chess
import chess.engine

def parse_bm_field(board: chess.Board, bm_san_tokens: list[str]) -> set[chess.Move]:
    """Convert a list of SAN strings into a set of legal Move objects."""
    moves: set[chess.Move] = set()
    for san in bm_san_tokens:
        san = san.rstrip(";,")
        if not san:
            continue
        try:
            move = board.parse_san(san)
            moves.add(move)
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            # Skip unparseable tokens (shouldn't happen in well-formed EPD)
            pass
    return moves


def run_wac(engine_path: str, movetime_ms: int, epd_path: str) -> None:
    if not os.path.isfile(engine_path):
        sys.exit(f"Error: engine not found: {engine_path}")
    if not os.path.isfile(epd_path):
        sys.exit(f"Error: EPD file not found: {epd_path}")

    solved = 0
    total = 0
    errors = 0

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine_name = engine.id.get("name", os.path.basename(engine_path))
    # Force single-thread so WAC scores stay deterministic and comparable to
    # prior single-thread numbers (the engine now defaults Threads to all cores).
    try:
        engine.configure({"Threads": 1})
    except Exception:
        pass
    limit = chess.engine.Limit(time=movetime_ms / 1000.0)

    with open(epd_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # EPD format: <FEN 4-part> [ops...]
            # FEN has exactly 4 parts: pieces, side, castling, ep
            parts = line.split()
            if len(parts) < 6:
                errors += 1
                continue

            fen = " ".join(parts[:4])  # position FEN without move counters
            ops_str = " ".join(parts[4:])

            # Parse the board
            try:
                board = chess.Board(fen)
            except ValueError:
                errors += 1
                continue

            # Extract bm field: everything between "bm " and the next ";"
            bm_tokens: list[str] = []
            if "bm " in ops_str:
                bm_part = ops_str.split("bm ", 1)[1]
                # bm list ends at the first semicolon
                bm_segment = bm_part.split(";")[0]
                bm_tokens = bm_segment.split()

            if not bm_tokens:
                errors += 1
                continue

            best_moves = parse_bm_field(board, bm_tokens)
            if not best_moves:
                errors += 1
                continue

            total += 1

            # Ask the engine for its best move
            try:
                result = engine.play(board, limit)
            except Exception:
                errors += 1
                total -= 1
                continue

            if result.move in best_moves:
                solved += 1

    engine.quit()

    if total == 0:
        print(f"WAC: no positions parsed (errors={errors})")
        return

    pct = 100.0 * solved / total
    print(f"WAC: {solved}/{total}  ({pct:.1f}%,  movetime {movetime_ms}ms,  engine {engine_name!r})")
    if errors:
        print(f"  ({errors} positions skipped due to parse errors)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    _engine_path = sys.argv[1]
    _movetime_ms = int(sys.argv[2]) if len(sys.argv) >= 3 else 200
    _default_epd = os.path.join(os.path.dirname(__file__), "wac.epd")
    _epd_path = sys.argv[3] if len(sys.argv) >= 4 else _default_epd

    run_wac(_engine_path, _movetime_ms, _epd_path)
