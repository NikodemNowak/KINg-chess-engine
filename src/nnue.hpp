#pragma once
// NNUE inference for the KINg engine.
//
// Perspective net (768 -> HL) x2 -> 1, clipped-ReLU, INT16/INT32 quantized.
// HL (accumulator size per perspective) is a compile-time constant controlled
// by -DNNUE_HL=<n> (default 256 for the committed net; 512 for the full retrain).
// The net is embedded in the binary (see nnue_net_data.cpp, generated at build
// time from nets/king_nnue.bin), so no external file is needed at runtime.
//
// This whole translation unit is only compiled when EVAL_NNUE is defined (the
// CMake -DEVAL=NNUE configuration). HCE builds pull in none of it.
#include "types.hpp"

namespace king {

class Position;

namespace nnue {

// Architecture constants (also validated against the embedded net header).
// HL (accumulator size per perspective) is set at build time via -DNNUE_HL=<n>
// (default 256 for the committed HL=256 net; override to 512 for the full
// retrain).  All other constants are fixed.
constexpr int INPUT = 768;
#ifndef NNUE_HL
constexpr int HL    = 256;
#else
constexpr int HL    = NNUE_HL;
#endif
constexpr int QA    = 255;
constexpr int QB    = 64;
constexpr int SCALE = 400;

// Parse + validate the embedded net. Called once at startup. Idempotent.
void init();

// FROM-SCRATCH evaluation: rebuild both accumulators by summing active-piece
// feature columns, then run the forward pass. This is the correctness reference
// (matches the trainer's quantized eval bit-for-bit). Returns centipawns from
// the side-to-move POV, same convention as the HCE evaluate().
int evaluate_from_scratch(const Position& pos);

// Feature index for (color c, type t, square s) from perspective P.
inline int feature_index(Color c, PieceType t, Square s, Color P) {
    int os = (P == WHITE) ? int(s) : (int(s) ^ 56);
    int cr = (c == P) ? 0 : 1;
    return cr * 384 + int(t) * 64 + os;
}

// ── Incremental accumulator ───────────────────────────────────────────────
// Two int16 accumulators (one per perspective, indexed by Color). The live
// accumulator is owned by Position and kept in sync by put/remove_piece; it is
// refreshed from scratch in set_fen / copy_from. The forward pass below reads
// it (stm perspective first) and produces the eval without re-summing features.
struct Accumulator {
    alignas(32) int16_t v[COLOR_NB][HL];
};

// Reset `acc` to the bias and add every piece on the board (full refresh).
void refresh(Accumulator& acc, const Position& pos);

// Run the output layer on a (already up-to-date) accumulator for the given stm.
int evaluate_acc(const Accumulator& acc, Color stm);

// Add / subtract a single feature (a piece of color c, type t on square s) to
// BOTH perspectives of the accumulator. Used by Position::put/remove_piece.
void add_feature(Accumulator& acc, Color c, PieceType t, Square s);
void sub_feature(Accumulator& acc, Color c, PieceType t, Square s);

} // namespace nnue
} // namespace king
