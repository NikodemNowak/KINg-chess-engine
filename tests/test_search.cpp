#include "doctest/doctest.h"
#include "search.hpp"
#include "movegen.hpp"
#include "position.hpp"
#include "attacks.hpp"
#include "bitboard.hpp"
#include "zobrist.hpp"
#include <atomic>
#include <chrono>
using namespace king;

static void se_init() {
    bitboard::init();
    attacks::init_leapers();
    attacks::init_magics();
    zobrist::init();
}

static bool legal_move(Position& p, Move m) {
    MoveList ml;
    generate_legal(p, ml);
    for (int i = 0; i < ml.size; ++i)
        if (ml.moves[i] == m) return true;
    return false;
}

TEST_CASE("returns a legal move") {
    se_init();
    Position p;
    p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Limits L;
    L.depth = 4;
    std::atomic<bool> stop{false};
    Move m = search::think(p, L, stop, 50, 1);
    CHECK(legal_move(p, m));
}

TEST_CASE("grabs a free queen") {
    se_init();
    // exd5 wins the undefended queen
    Position p;
    p.set_fen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1");
    Limits L;
    L.depth = 4;
    std::atomic<bool> stop{false};
    Move m = search::think(p, L, stop, 50, 1);
    CHECK(m == make_move(E4, D5));
}

TEST_CASE("respects movetime and returns legal") {
    se_init();
    Position p;
    p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Limits L;
    L.movetime = 200;
    std::atomic<bool> stop{false};
    auto t0 = std::chrono::steady_clock::now();
    Move m = search::think(p, L, stop, 50, 1);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
    CHECK(legal_move(p, m));
    CHECK(ms < 2000);
}
