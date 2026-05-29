#include "doctest/doctest.h"
#include "perft.hpp"
#include "position.hpp"
#include "attacks.hpp"
#include "bitboard.hpp"
#include "zobrist.hpp"
using namespace king;

static void pf_init(){ bitboard::init(); attacks::init_leapers(); attacks::init_magics(); zobrist::init(); }
static uint64_t PF(const char* fen,int d){ Position p; p.set_fen(fen); return perft(p,d); }

#define SP   "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#define KIWI "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
#define POS3 "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
#define POS4 "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
#define POS5 "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
#define POS6 "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"

TEST_CASE("perft reference positions (correctness gate)") {
  pf_init();
  CHECK(PF(SP,1)==20ULL);  CHECK(PF(SP,2)==400ULL);  CHECK(PF(SP,3)==8902ULL);
  CHECK(PF(SP,4)==197281ULL); CHECK(PF(SP,5)==4865609ULL);
  CHECK(PF(KIWI,1)==48ULL); CHECK(PF(KIWI,2)==2039ULL); CHECK(PF(KIWI,3)==97862ULL); CHECK(PF(KIWI,4)==4085603ULL);
  CHECK(PF(POS3,4)==43238ULL); CHECK(PF(POS3,5)==674624ULL);
  CHECK(PF(POS4,4)==422333ULL);
  CHECK(PF(POS5,4)==2103487ULL);
  CHECK(PF(POS6,4)==3894594ULL);
}

TEST_CASE("perft startpos depth 6 (slow, opt-in)" * doctest::skip()) {
  pf_init();
  CHECK(PF(SP,6)==119060324ULL);
}
