// Tapered piece-square-table (PeSTO) evaluation + handcrafted structural terms.
//
// PeSTO Source: Ronald Friederich, Chess Programming Wiki:
//   https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
//
// Orientation: a1=0, h8=63 (rank-major, rank 1 first — see types.hpp).
// Black pieces are mirrored vertically: sq ^ 56.
// The taper weight is a game-phase counter (24 = full midgame, 0 = full endgame).
//
// Structural terms (all color-symmetric, added to the same mg/eg accumulators):
//   bishop_pair, rook_open_file, rook_semiopen_file,
//   pawn_isolated, pawn_doubled, passed_pawn, mobility, king_safety.

#include "eval.hpp"
#include "bitboard.hpp"
#include "attacks.hpp"

namespace king {

// ── Piece values (midgame / endgame) ─────────────────────────────────────────
static constexpr int MG_VALUE[6] = {  82, 337, 365, 477, 1025,    0 };
static constexpr int EG_VALUE[6] = {  94, 281, 297, 512,  936,    0 };

// ── Phase increment per piece type ───────────────────────────────────────────
static constexpr int PHASE_INC[6] = { 0, 1, 1, 2, 4, 0 };

// ── PSQT tables ──────────────────────────────────────────────────────────────
// All from White's POV. Index 0 = a1 (rank 1), index 63 = h8 (rank 8).

static constexpr int MG_PSQT_PAWN[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
    -35,  -1, -20, -23, -15,  24,  38, -22,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -14,  13,   6,  21,  23,  12,  17, -23,
     -6,   7,  26,  31,  65,  56,  25, -20,
     98, 134,  61,  95,  68, 126,  34, -11,
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

static const int* MG_PSQT[6] = {
    MG_PSQT_PAWN, MG_PSQT_KNIGHT, MG_PSQT_BISHOP,
    MG_PSQT_ROOK, MG_PSQT_QUEEN,  MG_PSQT_KING
};
static const int* EG_PSQT[6] = {
    EG_PSQT_PAWN, EG_PSQT_KNIGHT, EG_PSQT_BISHOP,
    EG_PSQT_ROOK, EG_PSQT_QUEEN,  EG_PSQT_KING
};

// ── Structural term weights ───────────────────────────────────────────────────
// Format: {mg, eg}  (Texel-tunable — keep as named constants)

// Bishop pair
static constexpr int BISHOP_PAIR_MG = 30;
static constexpr int BISHOP_PAIR_EG = 50;

// Rook file bonuses
static constexpr int ROOK_OPEN_MG     = 25;
static constexpr int ROOK_OPEN_EG     = 10;
static constexpr int ROOK_SEMIOPEN_MG = 12;
static constexpr int ROOK_SEMIOPEN_EG =  5;

// Pawn structure penalties (positive values = penalty, subtracted below)
static constexpr int PAWN_ISOLATED_MG = 15;
static constexpr int PAWN_ISOLATED_EG = 18;
static constexpr int PAWN_DOUBLED_MG  = 10;
static constexpr int PAWN_DOUBLED_EG  = 25;

// Passed pawn bonus by relative rank (0-indexed, rank 0 = own back rank, never occupied by pawn)
static constexpr int PASSED_MG[8] = { 0,  2,  6, 12, 22, 40,  65, 0 };
static constexpr int PASSED_EG[8] = { 0,  5, 15, 30, 55, 85, 130, 0 };

// Mobility: (count - pivot) * weight  per piece type  (N, B, R, Q)
// Conservative weights — Texel tuning will refine these
static constexpr int MOB_PIVOT[4]  = { 4, 6,  7, 14 };  // indexed 0=N,1=B,2=R,3=Q
static constexpr int MOB_MG[4]     = { 3, 2,  1,  1 };
static constexpr int MOB_EG[4]     = { 3, 2,  2,  1 };

// King safety — conservative weights to avoid over-penalizing normal piece activity
// Attack units: per piece type that has ≥1 square in king zone (not per square)
static constexpr int KING_ATTACK_UNITS[6]  = { 0, 1, 1, 2, 3, 0 };  // per attacking piece
static constexpr int PAWN_SHIELD_MG = 5;    // per shielding pawn, mg only
static constexpr int KING_SAFETY_MAX = 100; // cap penalty magnitude (conservative for untuned)
// Coefficient for quadratic penalty: units*units / KING_SAFETY_DIVISOR
static constexpr int KING_SAFETY_DIVISOR = 8;

// ── Helper: adjacent-file mask ────────────────────────────────────────────────
// Returns a mask of the files adjacent to the file of square s.
static inline Bitboard adjacent_files(File f) {
    Bitboard m = 0;
    if (f > FILE_A) m |= file_bb(File(f - 1));
    if (f < FILE_H) m |= file_bb(File(f + 1));
    return m;
}

// Forward span: all squares strictly ahead of sq for color c (same file).
// For WHITE: squares above sq on same file; for BLACK: squares below sq on same file.
static inline Bitboard forward_file_span(Color c, Square sq) {
    // file mask minus the square itself and behind
    Bitboard f = file_bb(file_of(sq));
    if (c == WHITE) {
        // ranks above sq
        Bitboard above = ~((Bitboard(1) << (sq + 1)) - 1); // all bits >= sq+1... not quite
        // simpler: shift the square bit north repeatedly — just use rank mask arithmetic
        // squares on same file with rank > rank_of(sq)
        Rank r = rank_of(sq);
        Bitboard ahead = 0;
        for (int rr = r + 1; rr <= RANK_8; ++rr)
            ahead |= (RANK_1_BB << (rr * 8));
        return f & ahead;
    } else {
        Rank r = rank_of(sq);
        Bitboard ahead = 0;
        for (int rr = 0; rr < r; ++rr)
            ahead |= (RANK_1_BB << (rr * 8));
        return f & ahead;
    }
}

// Forward span on file AND adjacent files ahead of sq for color c.
// Used for passed pawn detection: no enemy pawn on file or adjacent files ahead.
static inline Bitboard passed_pawn_mask(Color c, Square sq) {
    File f = file_of(sq);
    Bitboard files = file_bb(f) | adjacent_files(f);
    Rank r = rank_of(sq);
    Bitboard ahead = 0;
    if (c == WHITE) {
        for (int rr = r + 1; rr <= RANK_8; ++rr)
            ahead |= (RANK_1_BB << (rr * 8));
    } else {
        for (int rr = 0; rr < r; ++rr)
            ahead |= (RANK_1_BB << (rr * 8));
    }
    return files & ahead;
}

// Relative rank: rank from the perspective of color c (0 = own back rank, 7 = promotion rank)
static inline int relative_rank(Color c, Square sq) {
    int r = rank_of(sq);
    return (c == WHITE) ? r : (7 - r);
}

// ── Per-side structural score ─────────────────────────────────────────────────
// Returns {mg_score, eg_score} for one side.
static void eval_side(const Position& pos, Color c, int& mg, int& eg) {

    const Color them = Color(c ^ 1);
    const Bitboard occ        = pos.occupied();
    const Bitboard own_pawns  = pos.pieces(c,    PAWN);
    const Bitboard their_pawns = pos.pieces(them, PAWN);

    // ── Bishop pair ──────────────────────────────────────────────────────────
    if (popcount(pos.pieces(c, BISHOP)) >= 2) {
        mg += BISHOP_PAIR_MG;
        eg += BISHOP_PAIR_EG;
    }

    // ── Pawn structure ───────────────────────────────────────────────────────
    {
        Bitboard pawns = own_pawns;
        while (pawns) {
            Square sq = pop_lsb(pawns);
            File f = file_of(sq);

            // Isolated: no own pawns on adjacent files
            if (!(own_pawns & adjacent_files(f))) {
                mg -= PAWN_ISOLATED_MG;
                eg -= PAWN_ISOLATED_EG;
            }

            // Doubled: count own pawns on same file; penalise extras only
            // (we'll count total pawns on file, subtract 1 for base, penalise rest)
            // We do this once per file to avoid double-counting; track via file set.
        }

        // Doubled pawn penalty — per-file, once
        for (int fi = FILE_A; fi <= FILE_H; ++fi) {
            Bitboard on_file = own_pawns & file_bb(File(fi));
            int cnt = popcount(on_file);
            if (cnt >= 2) {
                // penalise the extras (cnt-1 extra pawns)
                mg -= (cnt - 1) * PAWN_DOUBLED_MG;
                eg -= (cnt - 1) * PAWN_DOUBLED_EG;
            }
        }

        // Passed pawn bonus
        Bitboard pp = own_pawns;
        while (pp) {
            Square sq = pop_lsb(pp);
            // A pawn is passed if there are no enemy pawns in its forward span
            // (same file + adjacent files, all ranks ahead)
            if (!(their_pawns & passed_pawn_mask(c, sq))) {
                int relrank = relative_rank(c, sq);
                mg += PASSED_MG[relrank];
                eg += PASSED_EG[relrank];
            }
        }
    }

    // ── Rook file bonuses ────────────────────────────────────────────────────
    {
        Bitboard rooks = pos.pieces(c, ROOK);
        while (rooks) {
            Square sq = pop_lsb(rooks);
            Bitboard fbb = file_bb(file_of(sq));
            bool no_own   = !(own_pawns   & fbb);
            bool no_their = !(their_pawns & fbb);
            if (no_own && no_their) {
                mg += ROOK_OPEN_MG;
                eg += ROOK_OPEN_EG;
            } else if (no_own) {
                mg += ROOK_SEMIOPEN_MG;
                eg += ROOK_SEMIOPEN_EG;
            }
        }
    }

    // ── Mobility ─────────────────────────────────────────────────────────────
    // mobilityArea = squares NOT attacked by enemy pawns, excluding own pieces
    {
        // Build enemy pawn attack mask
        Bitboard ep_attacks = 0;
        Bitboard their_pw = their_pawns;
        while (their_pw) {
            Square s = pop_lsb(their_pw);
            ep_attacks |= pawn_attacks[them][s];
        }
        Bitboard own_pieces  = pos.pieces(c);
        Bitboard mob_area    = ~(ep_attacks | own_pieces);

        // Knights (mob index 0)
        {
            Bitboard nb = pos.pieces(c, KNIGHT);
            while (nb) {
                Square s = pop_lsb(nb);
                int cnt = popcount(knight_attacks[s] & mob_area);
                mg += (cnt - MOB_PIVOT[0]) * MOB_MG[0];
                eg += (cnt - MOB_PIVOT[0]) * MOB_EG[0];
            }
        }
        // Bishops (mob index 1)
        {
            Bitboard bb2 = pos.pieces(c, BISHOP);
            while (bb2) {
                Square s = pop_lsb(bb2);
                int cnt = popcount(bishop_attacks(s, occ) & mob_area);
                mg += (cnt - MOB_PIVOT[1]) * MOB_MG[1];
                eg += (cnt - MOB_PIVOT[1]) * MOB_EG[1];
            }
        }
        // Rooks (mob index 2)
        {
            Bitboard rb = pos.pieces(c, ROOK);
            while (rb) {
                Square s = pop_lsb(rb);
                int cnt = popcount(rook_attacks(s, occ) & mob_area);
                mg += (cnt - MOB_PIVOT[2]) * MOB_MG[2];
                eg += (cnt - MOB_PIVOT[2]) * MOB_EG[2];
            }
        }
        // Queens (mob index 3)
        {
            Bitboard qb = pos.pieces(c, QUEEN);
            while (qb) {
                Square s = pop_lsb(qb);
                int cnt = popcount(queen_attacks(s, occ) & mob_area);
                mg += (cnt - MOB_PIVOT[3]) * MOB_MG[3];
                eg += (cnt - MOB_PIVOT[3]) * MOB_EG[3];
            }
        }
    }

    // ── King safety ──────────────────────────────────────────────────────────
    {
        Square ksq  = pos.king_sq(c);
        Bitboard zone = king_attacks[ksq] | square_bb(ksq);

        // Sum up attack units from enemy pieces attacking the king zone
        int units = 0;
        for (int pt = KNIGHT; pt <= QUEEN; ++pt) {
            Bitboard att = pos.pieces(them, (PieceType)pt);
            while (att) {
                Square s = pop_lsb(att);
                Bitboard piece_att;
                switch (pt) {
                    case KNIGHT: piece_att = knight_attacks[s]; break;
                    case BISHOP: piece_att = bishop_attacks(s, occ); break;
                    case ROOK:   piece_att = rook_attacks(s, occ); break;
                    default:     piece_att = queen_attacks(s, occ); break;
                }
                int overlap = popcount(piece_att & zone);
                if (overlap > 0)
                    units += KING_ATTACK_UNITS[pt];
            }
        }
        // Quadratic penalty, mg-heavy, conservative divisor
        int penalty = (units * units) / KING_SAFETY_DIVISOR;
        if (penalty > KING_SAFETY_MAX) penalty = KING_SAFETY_MAX;
        mg -= penalty;
        // eg penalty is 0 (king safety is primarily a midgame concern)

        // Pawn shield: own pawns on the 3 files around king, on the 2 ranks in front
        File kf = file_of(ksq);
        Rank kr = rank_of(ksq);
        Bitboard shield_files = file_bb(kf) | adjacent_files(kf);
        // The 2 ranks directly in front
        Bitboard shield_ranks = 0;
        if (c == WHITE) {
            if (kr + 1 <= RANK_8) shield_ranks |= rank_bb(Rank(kr + 1));
            if (kr + 2 <= RANK_8) shield_ranks |= rank_bb(Rank(kr + 2));
        } else {
            if (kr - 1 >= RANK_1) shield_ranks |= rank_bb(Rank(kr - 1));
            if (kr - 2 >= RANK_1) shield_ranks |= rank_bb(Rank(kr - 2));
        }
        int shield = popcount(own_pawns & shield_files & shield_ranks);
        mg += shield * PAWN_SHIELD_MG;
    }
}

// ── evaluate ─────────────────────────────────────────────────────────────────
// Returns score in centipawns relative to the side to move.
int evaluate(const Position& pos) {
    int mg = 0, eg = 0, phase = 0;

    // PSQT + material (unchanged from PeSTO base)
    for (int c = WHITE; c <= BLACK; ++c) {
        const int sign = (c == WHITE) ? +1 : -1;
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard bb = pos.pieces((Color)c, (PieceType)pt);
            while (bb) {
                Square s = pop_lsb(bb);
                int idx = (c == WHITE) ? s : (s ^ 56);
                mg += sign * (MG_VALUE[pt] + MG_PSQT[pt][idx]);
                eg += sign * (EG_VALUE[pt] + EG_PSQT[pt][idx]);
                phase += PHASE_INC[pt];
            }
        }
    }

    // Structural terms: compute for white, subtract for black (perfectly symmetric)
    {
        int w_mg = 0, w_eg = 0;
        int b_mg = 0, b_eg = 0;
        eval_side(pos, WHITE, w_mg, w_eg);
        eval_side(pos, BLACK, b_mg, b_eg);
        mg += (w_mg - b_mg);
        eg += (w_eg - b_eg);
    }

    if (phase > 24) phase = 24;
    int score = (mg * phase + eg * (24 - phase)) / 24;
    return (pos.side_to_move() == WHITE) ? score : -score;
}

} // namespace king
