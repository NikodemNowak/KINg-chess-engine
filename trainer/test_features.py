"""Standalone (numpy-only) sanity tests for feature indexing + FEN parsing.

Validates the logic in train_nnue.py without importing torch. Run:
    py trainer/_test_features.py
"""
import numpy as np

INPUT_SIZE = 768
WHITE, BLACK = 0, 1
PT_FROM_CHAR = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}


def feature_index(persp, color, ptype, sq):
    os = sq if persp == WHITE else (sq ^ 56)
    cr = 0 if color == persp else 1
    return cr * 384 + ptype * 64 + os


def parse_fen_pieces(fen_board):
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


def sq_name(s):
    return "abcdefgh"[s & 7] + str((s >> 3) + 1)


def test_fen_squares():
    # startpos: white pieces on ranks 1-2, black on 7-8.
    p = parse_fen_pieces("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    assert len(p) == 32, len(p)
    # white rook a1 = square 0
    assert (WHITE, 3, 0) in p, "white rook a1"
    # white king e1 = square 4
    assert (WHITE, 5, 4) in p, "white king e1"
    # black king e8 = square 60
    assert (BLACK, 5, 60) in p, "black king e8"
    # black rook a8 = square 56
    assert (BLACK, 3, 56) in p, "black rook a8"
    # white pawn e2 = square 12
    assert (WHITE, 0, 12) in p, "white pawn e2"
    print("OK test_fen_squares")


def test_after_e4():
    # 1.e4 : white pawn moved e2(12)->e4(28)
    p = parse_fen_pieces("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")
    assert (WHITE, 0, 28) in p, "white pawn e4"
    assert (WHITE, 0, 12) not in p, "no pawn e2"
    print("OK test_after_e4")


def test_index_range():
    # All indices for startpos must be in [0,767], unique per (persp).
    p = parse_fen_pieces("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    for persp in (WHITE, BLACK):
        idx = [feature_index(persp, c, t, s) for (c, t, s) in p]
        assert all(0 <= i < INPUT_SIZE for i in idx), "range"
        assert len(set(idx)) == len(idx), "uniqueness"
    print("OK test_index_range")


def test_white_king_features():
    # From white's own perspective, the white king on e1 (sq4):
    #   os=4, cr=0, t=5 -> idx = 0 + 5*64 + 4 = 324
    assert feature_index(WHITE, WHITE, 5, 4) == 324
    # From black's perspective, that SAME white king (enemy) on e1:
    #   os = 4^56 = 60, cr=1, t=5 -> idx = 384 + 5*64 + 60 = 384+320+60 = 764
    assert feature_index(BLACK, WHITE, 5, 4) == 764
    print("OK test_white_king_features")


def test_color_mirror_symmetry():
    """A position P with white-to-move and its vertical+color mirror P' with
    black-to-move must produce IDENTICAL stm/nonstm feature multisets.

    This is the core invariant that makes a single perspective net symmetric.
    We mirror each piece: color flips, square ^= 56.
    """
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R"
    pieces = parse_fen_pieces(fen)

    # white to move -> stm = WHITE
    stm = WHITE
    sf = sorted(feature_index(stm, c, t, s) for (c, t, s) in pieces)
    nf = sorted(feature_index(stm ^ 1, c, t, s) for (c, t, s) in pieces)

    # Build mirrored position: flip color and square; black to move.
    mirrored = [(c ^ 1, t, s ^ 56) for (c, t, s) in pieces]
    stm2 = BLACK
    sf2 = sorted(feature_index(stm2, c, t, s) for (c, t, s) in mirrored)
    nf2 = sorted(feature_index(stm2 ^ 1, c, t, s) for (c, t, s) in mirrored)

    assert sf == sf2, "stm features must match under color+vertical mirror"
    assert nf == nf2, "nonstm features must match under color+vertical mirror"
    print("OK test_color_mirror_symmetry")


def test_startpos_symmetry():
    """Startpos with white to move: stm and nonstm feature sets must be equal
    (the position is symmetric), so a symmetric net evals startpos ~0."""
    p = parse_fen_pieces("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    sf = sorted(feature_index(WHITE, c, t, s) for (c, t, s) in p)
    nf = sorted(feature_index(WHITE ^ 1, c, t, s) for (c, t, s) in p)
    assert sf == nf, "startpos stm/nonstm features must be identical"
    print("OK test_startpos_symmetry")


if __name__ == '__main__':
    test_fen_squares()
    test_after_e4()
    test_index_range()
    test_white_king_features()
    test_color_mirror_symmetry()
    test_startpos_symmetry()
    print("\nALL FEATURE TESTS PASSED")
