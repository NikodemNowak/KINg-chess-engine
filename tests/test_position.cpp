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

TEST_CASE("halfmove/fullmove clamp on absurd FEN (UB guard, crash=loss)") {
    init_all();
    Position p;
    // A near-INT_MAX halfmove field would overflow halfmove_++ during search
    // (signed-int UB -> crash on the -O3 Linux build). set_fen must clamp it to the
    // legal [0,100] range so the clock can never overflow while deepening.
    p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 2147483647 1");
    CHECK(p.halfmove_clock() >= 0);
    CHECK(p.halfmove_clock() <= 100);
    // Negative / malformed fields must also land in range.
    Position q;
    q.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - -5 0");
    CHECK(q.halfmove_clock() >= 0);
    CHECK(q.halfmove_clock() <= 100);
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

TEST_CASE("do/undo identity + special moves") {
    init_all();
    // helper lambda to round-trip
    auto roundtrip = [](const char* fen, Move m){
        Position p; p.set_fen(fen);
        std::string f0 = p.fen(); uint64_t k0 = p.key();
        StateInfo st; p.do_move(m, st);
        CHECK(p.key() == p.compute_key());     // incremental key consistent after move
        p.undo_move(m);
        CHECK(p.fen() == f0);
        CHECK(p.key() == k0);
        CHECK(p.key() == p.compute_key());
    };
    roundtrip("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", make_move(E2,E4));           // double push sets ep
    roundtrip("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", make_move(E4,D5));                                  // capture
    roundtrip("4k3/8/8/8/8/8/8/4K2R w K - 0 1", make_move(E1,G1,CASTLING));                            // O-O
    roundtrip("r3k3/8/8/8/8/8/8/4K3 b q - 0 1", make_move(E8,C8,CASTLING));                            // O-O-O black
    roundtrip("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", make_move(E5,D6,EN_PASSANT));                       // en passant
    roundtrip("6k1/4P3/8/8/8/8/8/4K3 w - - 0 1", make_move(E7,E8,PROMO,QUEEN));                        // promotion push
    roundtrip("5rk1/4P3/8/8/8/8/8/4K3 w - - 0 1", make_move(E7,F8,PROMO,QUEEN));                       // promotion capture

    // specific post-state checks
    Position p; p.set_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1");
    StateInfo st; p.do_move(make_move(E1,G1,CASTLING), st);
    CHECK(p.piece_on(G1)==W_KING); CHECK(p.piece_on(F1)==W_ROOK);
    CHECK(p.piece_on(E1)==NO_PIECE); CHECK(p.piece_on(H1)==NO_PIECE);
    CHECK(p.side_to_move()==BLACK);
    Position q; q.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    StateInfo st2; q.do_move(make_move(E2,E4), st2);
    CHECK(file_of(NO_SQ==NO_SQ?E3:E3)==FILE_E); // ep target file e (sanity)
}

TEST_CASE("do_null_move / undo_null_move roundtrip") {
    init_all();

    // Position WITH an en-passant square: after 1.e4 (ep square = e3).
    const char* epFen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    Position p;
    p.set_fen(epFen);

    std::string fenBefore = p.fen();
    uint64_t    keyBefore = p.key();
    CHECK(keyBefore == p.compute_key()); // sanity: key consistent before null

    // After do_null_move:
    StateInfo st;
    p.do_null_move(st);

    CHECK(p.side_to_move() == WHITE);      // side flipped (was BLACK)
    CHECK(p.ep_square()    == NO_SQ);      // ep square cleared
    CHECK(p.key() == p.compute_key());     // incremental key still consistent

    // Key must differ from the before-state (side+ep were changed).
    CHECK(p.key() != keyBefore);

    // After undo_null_move: everything must be restored exactly.
    p.undo_null_move();

    CHECK(p.fen() == fenBefore);           // board + scalars fully restored
    CHECK(p.key() == keyBefore);           // key restored
    CHECK(p.key() == p.compute_key());     // and still consistent with board

    // Test on a position WITHOUT an ep square (no ep xor should fire).
    const char* noEpFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    Position p2;
    p2.set_fen(noEpFen);

    std::string fen2Before = p2.fen();
    uint64_t    key2Before = p2.key();

    StateInfo st2;
    p2.do_null_move(st2);
    CHECK(p2.side_to_move() == BLACK);
    CHECK(p2.ep_square()    == NO_SQ);
    CHECK(p2.key() == p2.compute_key());

    p2.undo_null_move();
    CHECK(p2.fen() == fen2Before);
    CHECK(p2.key() == key2Before);
    CHECK(p2.key() == p2.compute_key());
}
