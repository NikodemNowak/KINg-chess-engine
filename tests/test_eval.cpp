#include "doctest/doctest.h"
#include "position.hpp"
#include "eval.hpp"
#include "attacks.hpp"
#include "bitboard.hpp"
#include "zobrist.hpp"
#include <cstdlib>

using namespace king;

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
