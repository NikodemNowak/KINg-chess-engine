// NNUE gate tests (only built in the EVAL=NNUE configuration).
//
//  1. PRIMARY GATE: every FEN in trainer/nnue_samples.txt must reproduce the
//     trainer's quantized eval bit-for-bit via nnue::evaluate_from_scratch.
//  2. Incremental == scratch: after do_move the position's incremental
//     accumulator must yield exactly the same eval as a from-scratch rebuild,
//     checked along a played line and across all legal moves of several
//     positions, with the value restored after undo_move.
#include "doctest/doctest.h"

#ifdef EVAL_NNUE

#include "position.hpp"
#include "nnue.hpp"
#include "eval.hpp"
#include "movegen.hpp"
#include "attacks.hpp"
#include "bitboard.hpp"
#include "zobrist.hpp"
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

using namespace king;

static void nn_init() {
    bitboard::init();
    attacks::init_leapers();
    attacks::init_magics();
    zobrist::init();
    nnue::init();
}

// Eval read from the position's live (incremental) accumulator.
static int incremental_eval(const Position& p) {
    return nnue::evaluate_acc(p.accumulator(), p.side_to_move(),
                              popcount(p.occupied()));
}

TEST_CASE("NNUE bit-exact: reproduces all sample evals") {
    nn_init();

#ifndef NNUE_SAMPLES_PATH
#error "NNUE_SAMPLES_PATH must be defined (see CMakeLists.txt)"
#endif
    std::ifstream f(NNUE_SAMPLES_PATH);
    REQUIRE_MESSAGE(f.is_open(), "cannot open " NNUE_SAMPLES_PATH);

    int count = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        // Split off the trailing integer; the rest is the FEN.
        auto pos = line.find_last_of(' ');
        REQUIRE(pos != std::string::npos);
        std::string fen = line.substr(0, pos);
        int expected = std::stoi(line.substr(pos + 1));

        Position p;
        p.set_fen(fen);
        int got = nnue::evaluate_from_scratch(p);
        CHECK_MESSAGE(got == expected,
                      "FEN: ", fen, "  expected ", expected, " got ", got);
        ++count;
    }
    CHECK(count == 25);
}

TEST_CASE("NNUE: dispatcher evaluate == from-scratch on samples") {
    nn_init();
    // The search-facing evaluate() reads the incremental accumulator; it must
    // equal the from-scratch reference on a freshly set position.
    Position p;
    p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    CHECK(evaluate(p) == nnue::evaluate_from_scratch(p));
    CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
}

// Recursively walk all legal moves to a small depth; at every node assert the
// incremental accumulator eval matches a from-scratch rebuild, and that undo
// restores the parent's eval exactly.
static void check_tree(Position& p, int depth) {
    int scratch = nnue::evaluate_from_scratch(p);
    CHECK(incremental_eval(p) == scratch);
    if (depth == 0) return;

    MoveList ml;
    generate_legal(p, ml);
    for (int i = 0; i < ml.size; ++i) {
        StateInfo st;
        p.do_move(ml.moves[i], st);
        // After the move the incremental acc must match a full rebuild.
        CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
        check_tree(p, depth - 1);
        p.undo_move(ml.moves[i]);
        // Undo must restore the accumulator exactly.
        CHECK(incremental_eval(p) == scratch);
    }
}

TEST_CASE("NNUE incremental == scratch: full legal-move tree (several positions)") {
    nn_init();
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // startpos
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // kiwipete (castling, captures)
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", // promotions, EP, castling
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", // EP-rich endgame
    };
    for (const char* fen : fens) {
        Position p;
        p.set_fen(fen);
        check_tree(p, 2); // 2-ply exhaustive: every capture/promo/castle/EP delta exercised
    }
}

TEST_CASE("NNUE incremental == scratch: along a played 20-ply line") {
    nn_init();
    Position p;
    p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::vector<Move> played;
    std::vector<StateInfo> states(64);
    for (int ply = 0; ply < 20; ++ply) {
        MoveList ml;
        generate_legal(p, ml);
        if (ml.size == 0) break;
        // Deterministic pseudo-random pick (no RNG dependency).
        Move m = ml.moves[(ply * 7 + 3) % ml.size];
        p.do_move(m, states[ply]);
        played.push_back(m);
        CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
    }
    // Unwind and re-check at every step.
    for (int i = (int)played.size() - 1; i >= 0; --i) {
        p.undo_move(played[i]);
        CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
    }
}

TEST_CASE("NNUE: null move keeps accumulator consistent (stm swap only)") {
    nn_init();
    Position p;
    p.set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1");
    int before = incremental_eval(p);
    CHECK(before == nnue::evaluate_from_scratch(p));
    StateInfo st;
    p.do_null_move(st);
    // Features unchanged, only stm swapped -> eval read from the other
    // perspective must still equal a from-scratch rebuild of the null'd pos.
    CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
    p.undo_null_move();
    CHECK(incremental_eval(p) == before);
}

// Copy-make specific: a null move pushes NO accumulator slot, so a real move made
// AFTER a null must compute its child from the pre-null slot. Interleave real and
// null moves and verify incremental==scratch at every step and after every undo.
TEST_CASE("NNUE copy-make: real/null move interleaving") {
    nn_init();
    auto first_legal = [](Position& pos) -> Move {
        MoveList ml; generate_legal(pos, ml); return ml.size ? ml.moves[0] : Move(0);
    };
    Position p;
    p.set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1");
    int root = incremental_eval(p);
    CHECK(root == nnue::evaluate_from_scratch(p));

    std::vector<StateInfo> st(8);
    Move m1 = first_legal(p);
    p.do_move(m1, st[0]);
    CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
    p.do_null_move(st[1]);
    CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
    Move m2 = first_legal(p);          // a real move directly after a null move
    p.do_move(m2, st[2]);
    CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
    p.do_null_move(st[3]);
    CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
    Move m3 = first_legal(p);
    p.do_move(m3, st[4]);
    CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
    // Unwind in exact reverse order; every parent must be restored bit-exactly.
    p.undo_move(m3);    CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
    p.undo_null_move(); CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
    p.undo_move(m2);    CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
    p.undo_null_move(); CHECK(incremental_eval(p) == nnue::evaluate_from_scratch(p));
    p.undo_move(m1);    CHECK(incremental_eval(p) == root);
}

#endif // EVAL_NNUE
