#include "doctest/doctest.h"
#include "position.hpp"
#include "eval.hpp"
#include "attacks.hpp"
#include "bitboard.hpp"
#include "zobrist.hpp"
#include <cstdlib>

using namespace king;

// These assertions check properties of the handcrafted evaluation (near-zero
// symmetric startpos, exact color symmetry, structural-term magnitudes). The
// NNUE eval is not perfectly symmetric (the startpos sample is +36) and has no
// such structural guarantees, so this whole suite is HCE-only. The NNUE eval is
// gated instead by the bit-exact sample test in test_nnue.cpp.
#ifndef EVAL_NNUE

static void ev_init() {
    bitboard::init();
    attacks::init_leapers();
    attacks::init_magics();
    zobrist::init();
}

TEST_CASE("eval startpos ~ 0 (symmetry)") {
    ev_init();
    Position p;
    p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    CHECK(std::abs(evaluate(p)) <= 30);
}

TEST_CASE("eval color symmetry") {
    ev_init();
    // Mirrored positions should give the same side-to-move-relative score.
    Position a;
    a.set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1");
    Position b;
    b.set_fen("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 0 1");
    CHECK(evaluate(a) == evaluate(b));
}

TEST_CASE("eval up a knight is clearly positive") {
    ev_init();
    // Black is missing a knight — White (side to move) should score clearly positive.
    Position p;
    p.set_fen("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    CHECK(evaluate(p) > 250);
}

TEST_CASE("knight prefers center over corner") {
    ev_init();
    Position c;
    c.set_fen("4k3/8/8/8/8/8/8/N3K3 w - - 0 1"); // Na1 (corner)
    Position d;
    d.set_fen("4k3/8/8/4N3/8/8/8/4K3 w - - 0 1"); // Ne5 (center)
    CHECK(evaluate(d) > evaluate(c));
}

// ── New structural term tests ─────────────────────────────────────────────────

TEST_CASE("passed pawn scores higher than blocked pawn") {
    ev_init();
    // White pawn on e5 with no enemy pawns on d,e,f files ahead — passed
    Position pass;
    pass.set_fen("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1");
    // White pawn on e5 with enemy pawn on e6 blocking it — not passed
    Position block;
    block.set_fen("4k3/8/4p3/4P3/8/8/8/4K3 w - - 0 1");
    CHECK(evaluate(pass) > evaluate(block));
}

TEST_CASE("bishop pair scores higher than single bishop") {
    ev_init();
    // White has two bishops, Black has one
    Position two_bish;
    two_bish.set_fen("4k3/8/8/8/8/8/8/2BBK3 w - - 0 1");
    Position one_bish;
    one_bish.set_fen("4k3/8/8/8/8/8/8/3BK3 w - - 0 1");
    CHECK(evaluate(two_bish) > evaluate(one_bish));
}

TEST_CASE("rook on open file scores higher than closed file") {
    ev_init();
    // Same material: both sides have rook + king + one pawn each on d-file.
    // White rook on e1: e-file has no pawns at all — open file bonus.
    Position open_f;
    open_f.set_fen("4k3/3p4/8/8/8/8/3P4/4RK2 w - - 0 1");
    // White rook on e1: e-file has own pawn — blocked, no bonus.
    // Identical material: white d2-pawn moved to e2; black d7-pawn stays.
    Position closed_f;
    closed_f.set_fen("4k3/3p4/8/8/8/8/4P3/4RK2 w - - 0 1");
    // open file should score better (same material, rook has open-file bonus in open_f)
    CHECK(evaluate(open_f) > evaluate(closed_f));
}

#endif // !EVAL_NNUE
