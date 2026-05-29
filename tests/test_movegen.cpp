#include "doctest/doctest.h"
#include "movegen.hpp"
#include "position.hpp"
#include "attacks.hpp"
#include "bitboard.hpp"
#include "zobrist.hpp"
using namespace king;

static void mg_init(){ bitboard::init(); attacks::init_leapers(); attacks::init_magics(); zobrist::init(); }
static int count_legal(const char* fen){ Position p; p.set_fen(fen); MoveList ml; generate_legal(p, ml); return ml.size; }

TEST_CASE("legal move counts == perft(1)") {
  mg_init();
  CHECK(count_legal("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")==20);
  CHECK(count_legal("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")==48); // kiwipete
  CHECK(count_legal("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")==14);
  CHECK(count_legal("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")==6);
  CHECK(count_legal("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8")==44);
  CHECK(count_legal("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10")==46);
}

TEST_CASE("check evasion + pins") {
  mg_init();
  CHECK(count_legal("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")==3);    // king in check from rook on e2: Kd1,Kf1,Kxe2
  CHECK(count_legal("4r3/8/8/8/8/8/4N3/4K3 w - - 0 1")==4);    // knight on e2 pinned by rook e8 -> 0 knight moves, 4 king
}

TEST_CASE("has_legal_moves / mate / stalemate") {
  mg_init();
  Position s; s.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  CHECK(has_legal_moves(s) == true);
  Position m; m.set_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"); // fool's mate: White is mated
  CHECK(has_legal_moves(m) == false);
  CHECK(m.in_check(WHITE) == true);
  Position st; st.set_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"); // Black stalemated (not in check, no moves)
  CHECK(has_legal_moves(st) == false);
  CHECK(st.in_check(BLACK) == false);
}
