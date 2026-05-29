// Tapered piece-square-table (PeSTO) evaluation.
//
// Source: PeSTO's Evaluation Function by Ronald Friederich, published on the
//   Chess Programming Wiki:
//   https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
//
// Orientation note: The Wiki tables are printed rank-8-first (a8=row0, a1=row7).
// Our engine uses a1=0, h8=63 (rank-major, rank 1 first — see types.hpp).
// All tables below have been flipped vertically so that index 0 = a1, index 63 = h8.
//
// For Black pieces we mirror vertically: sq ^ 56 (swaps rank 1 ↔ rank 8, etc.),
// which converts the square to the equivalent White-perspective index.
// The taper weight is a game-phase counter (24 = full midgame, 0 = full endgame).

#include "eval.hpp"
#include "bitboard.hpp"

namespace king {

// ── Piece values (midgame / endgame) ─────────────────────────────────────────
// P, N, B, R, Q, K
static constexpr int MG_VALUE[6] = {  82, 337, 365, 477, 1025,    0 };
static constexpr int EG_VALUE[6] = {  94, 281, 297, 512,  936,    0 };

// ── Phase increment per piece type ───────────────────────────────────────────
// P=0, N=1, B=1, R=2, Q=4, K=0  (total for both sides at start = 24)
static constexpr int PHASE_INC[6] = { 0, 1, 1, 2, 4, 0 };

// ── PSQT tables ──────────────────────────────────────────────────────────────
// All from White's POV. Index 0 = a1 (rank 1), index 63 = h8 (rank 8).
// Row 0 below = rank 1 (a1..h1), row 7 = rank 8 (a8..h8).
// (The original PeSTO Wiki tables start at rank 8; they are reversed here.)

// Pawn — rank 1 and rank 8 are always 0 (impossible positions for a pawn)
static constexpr int MG_PSQT_PAWN[64] = {
     // rank 1 (a1..h1)
      0,   0,   0,   0,   0,   0,   0,   0,
     // rank 2
    -35,  -1, -20, -23, -15,  24,  38, -22,
     // rank 3
    -26,  -4,  -4, -10,   3,   3,  33, -12,
     // rank 4
    -27,  -2,  -5,  12,  17,   6,  10, -25,
     // rank 5
    -14,  13,   6,  21,  23,  12,  17, -23,
     // rank 6
     -6,   7,  26,  31,  65,  56,  25, -20,
     // rank 7
     98, 134,  61,  95,  68, 126,  34, -11,
     // rank 8
      0,   0,   0,   0,   0,   0,   0,   0,
};
static constexpr int EG_PSQT_PAWN[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     13,   8,   8,  10,  13,   0,   2,  -7,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
     32,  24,  13,   5,  -2,   4,  17,  17,
     94, 100,  85,  67,  56,  53,  82,  84,
    178, 173, 158, 134, 147, 132, 165, 187,
      0,   0,   0,   0,   0,   0,   0,   0,
};

// Knight
static constexpr int MG_PSQT_KNIGHT[64] = {
    -105, -21, -58, -33, -17, -28, -19,  -23,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -13,   4,  16,  13,  28,  19,  21,   -8,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -47,  60,  37,  65,  84, 129,  73,   44,
     -73, -41,  72,  36,  23,  62,   7,  -17,
    -167, -89, -34, -49,  61, -97, -15, -107,
};
static constexpr int EG_PSQT_KNIGHT[64] = {
    -29, -51, -23, -15, -22, -18, -50, -64,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -58, -38, -13, -28, -31, -27, -63, -99,
};

// Bishop
static constexpr int MG_PSQT_BISHOP[64] = {
    -33,  -3, -14, -21, -13, -12, -39, -21,
      4,  15,  16,   0,   7,  21,  33,   1,
      0,  15,  15,  15,  14,  27,  18,  10,
     -6,  13,  13,  26,  34,  12,  10,   4,
     -4,   5,  19,  50,  37,  37,   7,  -2,
    -16,  37,  43,  40,  35,  50,  37,  -2,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -29,   4, -82, -37, -25, -42,   7,  -8,
};
static constexpr int EG_PSQT_BISHOP[64] = {
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
     -3,   9,  12,   9, 14,  10,   3,   2,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
    -14, -21, -11,  -8, -7,  -9, -17, -24,
};

// Rook
static constexpr int MG_PSQT_ROOK[64] = {
    -19, -13,   1,  17, 16,  7, -37, -26,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -24, -11,   7,  26, 24, 35,  -8, -20,
     -5,  19,  26,  36, 17, 45,  61,  16,
     27,  32,  58,  62, 80, 67,  26,  44,
     32,  42,  32,  51, 63,  9,  31,  43,
};
static constexpr int EG_PSQT_ROOK[64] = {
    -9,  2,  3, -1, -5, -13,   4, -20,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -4,  0, -5, -1, -7, -12,  -8, -16,
     3,  5,  8,  4, -5,  -6,  -8, -11,
     4,  3, 13,  1,  2,   1,  -1,   2,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
    11, 13, 13, 11, -3,   3,   8,   3,
    13, 10, 18, 15, 12,  12,   8,   5,
};

// Queen
static constexpr int MG_PSQT_QUEEN[64] = {
     -1, -18,  -9,  10, -15, -25, -31, -50,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -28,   0,  29,  12,  59,  44,  43,  45,
};
static constexpr int EG_PSQT_QUEEN[64] = {
    -33, -28, -22, -43,  -5, -32, -20, -41,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -18,  28,  19,  47,  31,  34,  39,  23,
      3,  22,  24,  45,  57,  40,  57,  36,
    -20,   6,   9,  49,  47,  35,  19,   9,
    -17,  20,  32,  41,  58,  25,  30,   0,
     -9,  22,  22,  27,  27,  19,  10,  20,
};

// King
static constexpr int MG_PSQT_KING[64] = {
    -15,  36,  12, -54,   8, -28,  24,  14,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -14, -14, -22, -46, -44, -30, -15, -27,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -17, -20, -12, -27, -30, -25, -14, -36,
     -9,  24,   2, -16, -20,   6,  22, -22,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
    -65,  23,  16, -15, -56, -34,   2,  13,
};
static constexpr int EG_PSQT_KING[64] = {
    -53, -34, -21, -11, -28, -14, -24, -43,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -18,  -4,  21,  24,  27,  23,   9, -11,
     -8,  22,  24,  27,  26,  33,  26,   3,
     10,  17,  23,  15,  20,  45,  44,  13,
    -12,  17,  14,  17,  17,  38,  23,  11,
    -74, -35, -18, -18, -11,  15,   4, -17,
};

// Aggregate per piece type (indexed by PieceType enum: P=0..K=5)
static const int* MG_PSQT[6] = {
    MG_PSQT_PAWN, MG_PSQT_KNIGHT, MG_PSQT_BISHOP,
    MG_PSQT_ROOK, MG_PSQT_QUEEN,  MG_PSQT_KING
};
static const int* EG_PSQT[6] = {
    EG_PSQT_PAWN, EG_PSQT_KNIGHT, EG_PSQT_BISHOP,
    EG_PSQT_ROOK, EG_PSQT_QUEEN,  EG_PSQT_KING
};

// ── evaluate ─────────────────────────────────────────────────────────────────
// Returns score in centipawns relative to the side to move.
// Algorithm:
//   1. Accumulate mg/eg scores and phase counter over all pieces.
//   2. Taper: score = (mg * phase + eg * (24 - phase)) / 24.
//   3. Flip sign for Black to move.
int evaluate(const Position& pos) {
    int mg = 0, eg = 0, phase = 0;

    for (int c = WHITE; c <= BLACK; ++c) {
        const int sign = (c == WHITE) ? +1 : -1;
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard bb = pos.pieces((Color)c, (PieceType)pt);
            while (bb) {
                Square s = pop_lsb(bb);
                // Mirror rank for Black: rank 1 ↔ rank 8, so Black a8 (idx 56)
                // maps to idx 0 (White a1 view), Black a1 (idx 0) maps to idx 56.
                int idx = (c == WHITE) ? s : (s ^ 56);
                mg += sign * (MG_VALUE[pt] + MG_PSQT[pt][idx]);
                eg += sign * (EG_VALUE[pt] + EG_PSQT[pt][idx]);
                phase += PHASE_INC[pt];
            }
        }
    }

    if (phase > 24) phase = 24; // clamp (early promotions)
    int score = (mg * phase + eg * (24 - phase)) / 24;
    return (pos.side_to_move() == WHITE) ? score : -score;
}

} // namespace king
