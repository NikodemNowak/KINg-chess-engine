#include "doctest/doctest.h"
#include "position.hpp"
#include "attacks.hpp"
#include "bitboard.hpp"
#include "zobrist.hpp"
using namespace king;

static void d_init(){
    bitboard::init();
    attacks::init_leapers();
    attacks::init_magics();
    zobrist::init();
}

TEST_CASE("repetition") {
    d_init();
    Position p;
    p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move seq[4] = {
        make_move(G1, F3),
        make_move(G8, F6),
        make_move(F3, G1),
        make_move(F6, G8)
    };
    StateInfo sts[4];
    CHECK(p.is_draw() == false);
    for (int i = 0; i < 4; i++) p.do_move(seq[i], sts[i]);
    CHECK(p.is_repetition() == true);   // start position occurred a second time
    CHECK(p.is_draw() == true);
    for (int i = 3; i >= 0; i--) p.undo_move(seq[i]);
    CHECK(p.is_draw() == false);
}

TEST_CASE("fifty move") {
    d_init();
    Position p;
    p.set_fen("4k3/8/8/8/8/8/4R3/4K3 w - - 100 1");   // KR vs K, halfmove=100
    CHECK(p.is_draw() == true);                         // via 50-move

    Position q;
    q.set_fen("4k3/8/8/8/8/8/4R3/4K3 w - - 99 1");
    CHECK(q.is_draw() == false);
}

TEST_CASE("insufficient material") {
    d_init();
    // K vs K
    CHECK([](){
        Position p;
        p.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        return p.is_draw();
    }() == true);
    // KN vs K
    CHECK([](){
        Position p;
        p.set_fen("4k3/8/8/8/8/8/8/4KN2 w - - 0 1");
        return p.insufficient_material();
    }() == true);
    // KB vs K
    CHECK([](){
        Position p;
        p.set_fen("4k3/8/8/8/8/8/8/4KB2 w - - 0 1");
        return p.insufficient_material();
    }() == true);
    // pawn present => not insufficient
    CHECK([](){
        Position p;
        p.set_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
        return p.insufficient_material();
    }() == false);
    // rook present => not insufficient
    CHECK([](){
        Position p;
        p.set_fen("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1");
        return p.insufficient_material();
    }() == false);
}
