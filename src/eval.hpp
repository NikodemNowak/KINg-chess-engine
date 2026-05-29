#pragma once
#include "position.hpp"

namespace king {

// Tapered piece-square-table evaluation.
// Source: PeSTO tables by Ronald Friederich, as published on the Chess Programming Wiki:
//   https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
// Tables are indexed from White's perspective with a1=0, h8=63 (rank-major, little-endian).
// For Black pieces, the square index is mirrored vertically: sq ^ 56 (flips rank).
//
// Material values + handcrafted structural-term weights + the PSQT live in the
// mutable global `g_eval` (an EvalParams) so a Texel tuner can mutate them and
// re-run evaluate(). g_eval is default-initialised to the values committed below,
// so behaviour is identical until the tuner changes anything.

// ── Tunable evaluation parameters ─────────────────────────────────────────────
// All weights the Texel tuner is allowed to move. Anything NOT in here
// (e.g. game-phase increments, the mg/eg taper formula) is fixed.
struct EvalParams {
    // Material (midgame / endgame), indexed by PieceType (KING entry stays 0).
    int mg_value[6];
    int eg_value[6];

    // Bishop pair bonus.
    int bishop_pair_mg;
    int bishop_pair_eg;

    // Rook file bonuses.
    int rook_open_mg,     rook_open_eg;
    int rook_semiopen_mg, rook_semiopen_eg;

    // Pawn-structure penalties (positive = penalty, subtracted in eval).
    int pawn_isolated_mg, pawn_isolated_eg;
    int pawn_doubled_mg,  pawn_doubled_eg;

    // Passed-pawn bonus by relative rank (0 = own back rank, 7 = promotion rank).
    int passed_mg[8];
    int passed_eg[8];

    // Mobility: (count - pivot) * weight, per piece type (0=N,1=B,2=R,3=Q).
    int mob_pivot[4];
    int mob_mg[4];
    int mob_eg[4];

    // King safety.
    int king_attack_units[6]; // per attacking enemy piece type
    int pawn_shield_mg;       // per shielding pawn (mg only)
    int king_safety_max;      // cap on the quadratic penalty magnitude
    int king_safety_divisor;  // penalty = units*units / divisor (kept >= 1)

    // Piece-square tables (White POV, a1=0 .. h8=63), per PieceType.
    int mg_psqt[6][64];
    int eg_psqt[6][64];
};

// The single mutable instance read by evaluate(); default-initialised in eval.cpp.
extern EvalParams g_eval;

// Reset g_eval to the built-in (committed) defaults. Used by the tuner before a run.
void eval_set_defaults();

// Handcrafted (PeSTO + structural) evaluation. Always available — it is the
// "No Deep Learning" build and the NNUE-less fallback. Score in centipawns
// relative to the side to move.
int evaluate_hce(const Position& pos);

// The single evaluation symbol the search calls. Dispatches at compile time:
//   -DEVAL=NNUE  -> nnue::evaluate (incremental accumulator)
//   -DEVAL=HCE   -> evaluate_hce
int evaluate(const Position& pos);

} // namespace king
