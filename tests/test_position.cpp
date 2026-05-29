#include "doctest/doctest.h"
#include "position.hpp"
#include "attacks.hpp"
#include "bitboard.hpp"
#include "zobrist.hpp"
#include <string>
using namespace king;

static void init_all(){
    bitboard::init();
    attacks::init_leapers();
    attacks::init_magics();
    zobrist::init();
}

TEST_CASE("fen round-trip + key") {
    init_all();
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    };
    for (auto f : fens) {
        Position p;
        p.set_fen(f);
        CHECK(p.fen() == std::string(f));
        CHECK(p.key() == p.compute_key());
    }
}

TEST_CASE("queries + in_check") {
    init_all();
    Position p;
    p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    CHECK(p.in_check(WHITE) == false);
    CHECK(p.in_check(BLACK) == false);
    CHECK(p.piece_on(E1) == W_KING);
    CHECK(p.piece_on(D8) == B_QUEEN);
    CHECK(p.piece_on(E4) == NO_PIECE);
    CHECK(p.king_sq(WHITE) == E1);
    CHECK(p.side_to_move() == WHITE);

    // black rook e8 vs white king e1, empty e-file
    Position c;
    c.set_fen("4r3/8/8/8/8/8/8/4K3 w - - 0 1");
    CHECK(c.in_check(WHITE) == true);

    // white knight f2 does NOT check black king e8
    Position d;
    d.set_fen("4k3/8/8/8/8/8/5N2/4K3 b - - 0 1");
    CHECK(d.in_check(BLACK) == false);
}
