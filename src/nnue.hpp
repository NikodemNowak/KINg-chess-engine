#pragma once
// NNUE inference for the KINg engine.
//
// Perspective net (768 -> HL) x2 -> output buckets, squared clipped-ReLU,
// INT16/INT32 quantized. HL (accumulator size per perspective) is a compile-time
// constant set via -DNNUE_HL=<n> (default 512). The net is embedded in the binary
// (see nnue_net_data.cpp, generated at build time from the committed net binary),
// so no external file is needed at runtime.
//
// This whole translation unit is only compiled when EVAL_NNUE is defined (the
// CMake -DEVAL=NNUE configuration). HCE builds pull in none of it.
#include "types.hpp"

namespace king {

class Position;

namespace nnue {

// Architecture constants (also validated against the embedded net header).
// HL (accumulator size per perspective) is set at build time via -DNNUE_HL=<n>
// (default 512, matching the committed net).  All other constants are fixed.
constexpr int INPUT = 768;
#ifndef NNUE_HL
constexpr int HL    = 512;
#else
constexpr int HL    = NNUE_HL;
#endif
constexpr int QA    = 255;
constexpr int QB    = 64;
constexpr int SCALE = 400;

// Output buckets (by total piece count). 1 = legacy single-bucket behaviour
// (byte-identical to the original KNUE format). >1 selects a per-bucket output
// layer (W2,b2) indexed by piece count — set at build time via -DNNUE_OB=<n>.
#ifndef NNUE_OB
constexpr int OB = 1;
#else
constexpr int OB = NNUE_OB;
#endif

// Output-bucket index from the total piece count (2..32, kings included).
// MUST match the trainer EXACTLY: bucket = clamp((piece_count - 2) / 4, 0, OB-1).
// For OB=1 this is always 0, so legacy nets are unaffected.
inline int ob_index(int piece_count) {
    int b = (piece_count - 2) / 4;
    if (b < 0) b = 0;
    if (b > OB - 1) b = OB - 1;
    return b;
}

// King-square INPUT buckets (HalfKP-style). 1 = plain 768 inputs (no king
// dependence, byte-identical to legacy). >1 makes each perspective's features
// depend on its OWN king square — the input space becomes KB*768. Set at build
// time via -DNNUE_KB=<n>.
#ifndef NNUE_KB
constexpr int KB = 1;
#else
constexpr int KB = NNUE_KB;
#endif
constexpr int FT_IN = KB * INPUT; // feature-transformer input dimension

// King-square -> bucket. `oks` is the ORIENTED own-king square (0..63) of the
// perspective. MUST match the trainer's king_bucket_np EXACTLY. For KB=1 this is
// always 0 (king-independent). Supported KB: 1,4,8,16,32,64.
inline int king_bucket(int oks) {
    if (KB == 1)  return 0;
    if (KB == 64) return oks;
    int file2 = (oks % 8) / 2;   // 0..3 (file pair)
    int rank  = oks / 8;         // 0..7
    if (KB == 4)  return file2;
    if (KB == 8)  return file2 + 4 * (rank / 4);
    if (KB == 16) return file2 + 4 * (rank / 2);
    if (KB == 32) return file2 + 4 * rank;
    return 0; // unreachable for supported KB
}

// Parse + validate the embedded net. Called once at startup. Idempotent.
void init();

// FROM-SCRATCH evaluation: rebuild both accumulators by summing active-piece
// feature columns, then run the forward pass. This is the correctness reference
// (matches the trainer's quantized eval bit-for-bit). Returns centipawns from
// the side-to-move POV, same convention as the HCE evaluate().
int evaluate_from_scratch(const Position& pos);

// Feature index for (color c, type t, square s) from perspective P, given the
// square of P's OWN king (kingP). For KB=1 the king term is 0 → identical to the
// legacy index. MUST match the trainer's feature_index + king bucket exactly.
inline int feature_index(Color c, PieceType t, Square s, Color P, Square kingP) {
    int oks = (P == WHITE) ? int(kingP) : (int(kingP) ^ 56);
    int kb  = king_bucket(oks);
    int os  = (P == WHITE) ? int(s) : (int(s) ^ 56);
    int cr  = (c == P) ? 0 : 1;
    return kb * INPUT + cr * 384 + int(t) * 64 + os;
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

// Rebuild ONE perspective only. Used after a king move: the moving side's king
// bucket changed (so its whole perspective must be rebuilt), but the other
// perspective only saw the enemy king shift squares and was already updated
// incrementally — so refreshing it too would be wasted work.
void refresh_perspective(Accumulator& acc, const Position& pos, Color persp);

// Run the output layer on a (already up-to-date) accumulator for the given stm.
// piece_count (total pieces on the board, 2..32) selects the output bucket.
int evaluate_acc(const Accumulator& acc, Color stm, int piece_count);

// Add a single feature (a piece of color c, type t on square s) to BOTH
// perspectives of `acc`. wking/bking are the White/Black king squares (the
// per-perspective bucket determinants, consulted only for KB>1). Used by the
// full-board refresh.
void add_feature(Accumulator& acc, Color c, PieceType t, Square s, Square wking, Square bking);

// ── Copy-make fused update ──────────────────────────────────────────────────
// One piece placement (color c, type t, square s) that became active (added) or
// inactive (removed) as the result of a move.
struct Feat { Color c; PieceType t; Square s; };

// Compute the CHILD accumulator `dst` from the PARENT `src` by applying up to two
// added and two removed features, fused into ONE pass per perspective. This is
// the copy-make update: `src` (the parent) is only read and left intact, so undo
// is free (the search just steps the ply index back). n_add ∈ [1,2], n_sub ∈
// [1,2]. wking/bking are the per-perspective OWN king squares (consulted only for
// KB>1; ignored for the KB=1 production net). Bit-identical to a memcpy(dst,src)
// followed by applying the n_add added and n_sub removed features one at a time.
void update_accumulator(Accumulator& dst, const Accumulator& src,
                        const Feat* adds, int n_add,
                        const Feat* subs, int n_sub,
                        Square wking, Square bking);

} // namespace nnue
} // namespace king
