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
//
// All tunable weights (material, structural-term weights, and the PSQT) live in
// the mutable global `g_eval`. The committed defaults below are the Texel-tuned
// values; eval_set_defaults() also reinstates them (used by the tuner).

#include "eval.hpp"
#include "bitboard.hpp"
#include "attacks.hpp"
#ifdef EVAL_NNUE
#include "nnue.hpp"
#endif

namespace king {

// ── Phase increment per piece type (not tuned) ────────────────────────────────
static constexpr int PHASE_INC[6] = { 0, 1, 1, 2, 4, 0 };

// ── Default (committed) evaluation parameters ─────────────────────────────────
// These are the values evaluate() uses out of the box. The PSQT blocks are the
// PeSTO tables (White POV, a1=0 .. h8=63); the scalar/array weights are the
// handcrafted-term weights (Texel-tuned).
static const EvalParams DEFAULT_EVAL = {
    // mg_value[6]
    { 99, 401, 418, 514, 1210, 0 },
    // eg_value[6]
    { 86, 302, 314, 546, 944, 0 },
    30, 51, // bishop_pair mg,eg
    46, -4, // rook_open mg,eg
    12, 17, // rook_semiopen mg,eg
    10, 8, // pawn_isolated mg,eg
    11, 17, // pawn_doubled mg,eg
    // passed_mg[8]
    { 0, -1, -5, -13, 1, 4, -6, 0 },
    // passed_eg[8]
    { 0, 3, 11, 35, 51, 62, 62, 0 },
    // mob_pivot[4]
    { 4, 6, -1, 5 },
    // mob_mg[4]
    { 6, 5, 3, 1 },
    // mob_eg[4]
    { -1, 1, 2, 7 },
    // king_attack_units[6]
    { 0, 9, 12, 13, 15, 0 },
    11, // pawn_shield_mg
    215, // king_safety_max
    19, // king_safety_divisor
    // mg_psqt[6][64]
    {
        // PAWN
        {
               0,    0,    0,    0,    0,    0,    0,    0,
             -35,   -1,  -20,  -23,  -15,   24,   38,  -22,
             -26,   -4,   -4,  -10,    3,    3,   33,  -12,
             -27,   -2,   -5,   12,   17,    6,   10,  -25,
             -14,   13,    6,   21,   23,   12,   17,  -23,
              -6,    7,   26,   31,   65,   56,   25,  -20,
              98,  134,   61,   95,   68,  126,   34,  -11,
               0,    0,    0,    0,    0,    0,    0,    0,
        },
        // KNIGHT
        {
            -105,  -21,  -58,  -33,  -17,  -28,  -19,  -23,
             -29,  -53,  -12,   -3,   -1,   18,  -14,  -19,
             -23,   -9,   12,   10,   19,   17,   25,  -16,
             -13,    4,   16,   13,   28,   19,   21,   -8,
              -9,   17,   19,   53,   37,   69,   18,   22,
             -47,   60,   37,   65,   84,  129,   73,   44,
             -73,  -41,   72,   36,   23,   62,    7,  -17,
            -167,  -89,  -34,  -49,   61,  -97,  -15, -107,
        },
        // BISHOP
        {
             -33,   -3,  -14,  -21,  -13,  -12,  -39,  -21,
               4,   15,   16,    0,    7,   21,   33,    1,
               0,   15,   15,   15,   14,   27,   18,   10,
              -6,   13,   13,   26,   34,   12,   10,    4,
              -4,    5,   19,   50,   37,   37,    7,   -2,
             -16,   37,   43,   40,   35,   50,   37,   -2,
             -26,   16,  -18,  -13,   30,   59,   18,  -47,
             -29,    4,  -82,  -37,  -25,  -42,    7,   -8,
        },
        // ROOK
        {
             -19,  -13,    1,   17,   16,    7,  -37,  -26,
             -44,  -16,  -20,   -9,   -1,   11,   -6,  -71,
             -45,  -25,  -16,  -17,    3,    0,   -5,  -33,
             -36,  -26,  -12,   -1,    9,   -7,    6,  -23,
             -24,  -11,    7,   26,   24,   35,   -8,  -20,
              -5,   19,   26,   36,   17,   45,   61,   16,
              27,   32,   58,   62,   80,   67,   26,   44,
              32,   42,   32,   51,   63,    9,   31,   43,
        },
        // QUEEN
        {
              -1,  -18,   -9,   10,  -15,  -25,  -31,  -50,
             -35,   -8,   11,    2,    8,   15,   -3,    1,
             -14,    2,  -11,   -2,   -5,    2,   14,    5,
              -9,  -26,   -9,  -10,   -2,   -4,    3,   -3,
             -27,  -27,  -16,  -16,   -1,   17,   -2,    1,
             -13,  -17,    7,    8,   29,   56,   47,   57,
             -24,  -39,   -5,    1,  -16,   57,   28,   54,
             -28,    0,   29,   12,   59,   44,   43,   45,
        },
        // KING
        {
             -15,   36,   12,  -54,    8,  -28,   24,   14,
               1,    7,   -8,  -64,  -43,  -16,    9,    8,
             -14,  -14,  -22,  -46,  -44,  -30,  -15,  -27,
             -49,   -1,  -27,  -39,  -46,  -44,  -33,  -51,
             -17,  -20,  -12,  -27,  -30,  -25,  -14,  -36,
              -9,   24,    2,  -16,  -20,    6,   22,  -22,
              29,   -1,  -20,   -7,   -8,   -4,  -38,  -29,
             -65,   23,   16,  -15,  -56,  -34,    2,   13,
        },
    },
    // eg_psqt[6][64]
    {
        // PAWN
        {
               0,    0,    0,    0,    0,    0,    0,    0,
              13,    8,    8,   10,   13,    0,    2,   -7,
               4,    7,   -6,    1,    0,   -5,   -1,   -8,
              13,    9,   -3,   -7,   -7,   -8,    3,   -1,
              32,   24,   13,    5,   -2,    4,   17,   17,
              94,  100,   85,   67,   56,   53,   82,   84,
             178,  173,  158,  134,  147,  132,  165,  187,
               0,    0,    0,    0,    0,    0,    0,    0,
        },
        // KNIGHT
        {
             -29,  -51,  -23,  -15,  -22,  -18,  -50,  -64,
             -42,  -20,  -10,   -5,   -2,  -20,  -23,  -44,
             -23,   -3,   -1,   15,   10,   -3,  -20,  -22,
             -18,   -6,   16,   25,   16,   17,    4,  -18,
             -17,    3,   22,   22,   22,   11,    8,  -18,
             -24,  -20,   10,    9,   -1,   -9,  -19,  -41,
             -25,   -8,  -25,   -2,   -9,  -25,  -24,  -52,
             -58,  -38,  -13,  -28,  -31,  -27,  -63,  -99,
        },
        // BISHOP
        {
             -23,   -9,  -23,   -5,   -9,  -16,   -5,  -17,
             -14,  -18,   -7,   -1,    4,   -9,  -15,  -27,
             -12,   -3,    8,   10,   13,    3,   -7,  -15,
              -6,    3,   13,   19,    7,   10,   -3,   -9,
              -3,    9,   12,    9,   14,   10,    3,    2,
               2,   -8,    0,   -1,   -2,    6,    0,    4,
              -8,   -4,    7,  -12,   -3,  -13,   -4,  -14,
             -14,  -21,  -11,   -8,   -7,   -9,  -17,  -24,
        },
        // ROOK
        {
              -9,    2,    3,   -1,   -5,  -13,    4,  -20,
              -6,   -6,    0,    2,   -9,   -9,  -11,   -3,
              -4,    0,   -5,   -1,   -7,  -12,   -8,  -16,
               3,    5,    8,    4,   -5,   -6,   -8,  -11,
               4,    3,   13,    1,    2,    1,   -1,    2,
               7,    7,    7,    5,    4,   -3,   -5,   -3,
              11,   13,   13,   11,   -3,    3,    8,    3,
              13,   10,   18,   15,   12,   12,    8,    5,
        },
        // QUEEN
        {
             -33,  -28,  -22,  -43,   -5,  -32,  -20,  -41,
             -22,  -23,  -30,  -16,  -16,  -23,  -36,  -32,
             -16,  -27,   15,    6,    9,   17,   10,    5,
             -18,   28,   19,   47,   31,   34,   39,   23,
               3,   22,   24,   45,   57,   40,   57,   36,
             -20,    6,    9,   49,   47,   35,   19,    9,
             -17,   20,   32,   41,   58,   25,   30,    0,
              -9,   22,   22,   27,   27,   19,   10,   20,
        },
        // KING
        {
             -53,  -34,  -21,  -11,  -28,  -14,  -24,  -43,
             -27,  -11,    4,   13,   14,    4,   -5,  -17,
             -19,   -3,   11,   21,   23,   16,    7,   -9,
             -18,   -4,   21,   24,   27,   23,    9,  -11,
              -8,   22,   24,   27,   26,   33,   26,    3,
              10,   17,   23,   15,   20,   45,   44,   13,
             -12,   17,   14,   17,   17,   38,   23,   11,
             -74,  -35,  -18,  -18,  -11,   15,    4,  -17,
        },
    },
    34, 18, // tempo mg,eg
    79, 23, // threat_pawn  mg,eg
    54, 17, // threat_minor mg,eg
    86, -5, // threat_rook  mg,eg
};

// The single mutable instance read by evaluate().
EvalParams g_eval = DEFAULT_EVAL;

void eval_set_defaults() { g_eval = DEFAULT_EVAL; }

// ── Helper: adjacent-file mask ────────────────────────────────────────────────
// Returns a mask of the files adjacent to the file of square s.
static inline Bitboard adjacent_files(File f) {
    Bitboard m = 0;
    if (f > FILE_A) m |= file_bb(File(f - 1));
    if (f < FILE_H) m |= file_bb(File(f + 1));
    return m;
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
        mg += g_eval.bishop_pair_mg;
        eg += g_eval.bishop_pair_eg;
    }

    // ── Pawn structure ───────────────────────────────────────────────────────
    {
        Bitboard pawns = own_pawns;
        while (pawns) {
            Square sq = pop_lsb(pawns);
            File f = file_of(sq);

            // Isolated: no own pawns on adjacent files
            if (!(own_pawns & adjacent_files(f))) {
                mg -= g_eval.pawn_isolated_mg;
                eg -= g_eval.pawn_isolated_eg;
            }
        }

        // Doubled pawn penalty — per-file, once
        for (int fi = FILE_A; fi <= FILE_H; ++fi) {
            Bitboard on_file = own_pawns & file_bb(File(fi));
            int cnt = popcount(on_file);
            if (cnt >= 2) {
                // penalise the extras (cnt-1 extra pawns)
                mg -= (cnt - 1) * g_eval.pawn_doubled_mg;
                eg -= (cnt - 1) * g_eval.pawn_doubled_eg;
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
                mg += g_eval.passed_mg[relrank];
                eg += g_eval.passed_eg[relrank];
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
                mg += g_eval.rook_open_mg;
                eg += g_eval.rook_open_eg;
            } else if (no_own) {
                mg += g_eval.rook_semiopen_mg;
                eg += g_eval.rook_semiopen_eg;
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
                mg += (cnt - g_eval.mob_pivot[0]) * g_eval.mob_mg[0];
                eg += (cnt - g_eval.mob_pivot[0]) * g_eval.mob_eg[0];
            }
        }
        // Bishops (mob index 1)
        {
            Bitboard bb2 = pos.pieces(c, BISHOP);
            while (bb2) {
                Square s = pop_lsb(bb2);
                int cnt = popcount(bishop_attacks(s, occ) & mob_area);
                mg += (cnt - g_eval.mob_pivot[1]) * g_eval.mob_mg[1];
                eg += (cnt - g_eval.mob_pivot[1]) * g_eval.mob_eg[1];
            }
        }
        // Rooks (mob index 2)
        {
            Bitboard rb = pos.pieces(c, ROOK);
            while (rb) {
                Square s = pop_lsb(rb);
                int cnt = popcount(rook_attacks(s, occ) & mob_area);
                mg += (cnt - g_eval.mob_pivot[2]) * g_eval.mob_mg[2];
                eg += (cnt - g_eval.mob_pivot[2]) * g_eval.mob_eg[2];
            }
        }
        // Queens (mob index 3)
        {
            Bitboard qb = pos.pieces(c, QUEEN);
            while (qb) {
                Square s = pop_lsb(qb);
                int cnt = popcount(queen_attacks(s, occ) & mob_area);
                mg += (cnt - g_eval.mob_pivot[3]) * g_eval.mob_mg[3];
                eg += (cnt - g_eval.mob_pivot[3]) * g_eval.mob_eg[3];
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
                    units += g_eval.king_attack_units[pt];
            }
        }
        // Quadratic penalty, mg-heavy, conservative divisor
        int divisor = g_eval.king_safety_divisor;
        if (divisor < 1) divisor = 1; // guard against a tuner setting it to 0
        int penalty = (units * units) / divisor;
        if (penalty > g_eval.king_safety_max) penalty = g_eval.king_safety_max;
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
        mg += shield * g_eval.pawn_shield_mg;
    }

    // ── Threats ────────────────────────────────────────────────────────────────
    // Reward attacking more valuable enemy pieces with cheaper attackers; a static
    // proxy for tactical pressure the search would otherwise have to discover.
    {
        const Bitboard their_minor = pos.pieces(them, KNIGHT) | pos.pieces(them, BISHOP);
        const Bitboard their_rook  = pos.pieces(them, ROOK);
        const Bitboard their_queen = pos.pieces(them, QUEEN);
        const Bitboard their_bigp  = their_minor | their_rook | their_queen; // non-pawn

        // Pawn attacks on any enemy non-pawn piece.
        Bitboard pawn_att = 0;
        { Bitboard p = own_pawns; while (p) pawn_att |= pawn_attacks[c][pop_lsb(p)]; }
        int t = popcount(pawn_att & their_bigp);
        mg += t * g_eval.threat_pawn_mg;  eg += t * g_eval.threat_pawn_eg;

        // Minor (knight/bishop) attacks on enemy rook or queen.
        Bitboard minor_att = 0;
        { Bitboard b = pos.pieces(c, KNIGHT); while (b) minor_att |= knight_attacks[pop_lsb(b)]; }
        { Bitboard b = pos.pieces(c, BISHOP); while (b) minor_att |= bishop_attacks(pop_lsb(b), occ); }
        t = popcount(minor_att & (their_rook | their_queen));
        mg += t * g_eval.threat_minor_mg; eg += t * g_eval.threat_minor_eg;

        // Rook attacks on enemy queen.
        Bitboard rook_att = 0;
        { Bitboard b = pos.pieces(c, ROOK); while (b) rook_att |= rook_attacks(pop_lsb(b), occ); }
        t = popcount(rook_att & their_queen);
        mg += t * g_eval.threat_rook_mg;  eg += t * g_eval.threat_rook_eg;
    }
}

// ── evaluate (HCE) ───────────────────────────────────────────────────────────
// Returns score in centipawns relative to the side to move.
int evaluate_hce(const Position& pos) {
    int mg = 0, eg = 0, phase = 0;

    // PSQT + material
    for (int c = WHITE; c <= BLACK; ++c) {
        const int sign = (c == WHITE) ? +1 : -1;
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard bb = pos.pieces((Color)c, (PieceType)pt);
            while (bb) {
                Square s = pop_lsb(bb);
                int idx = (c == WHITE) ? s : (s ^ 56);
                mg += sign * (g_eval.mg_value[pt] + g_eval.mg_psqt[pt][idx]);
                eg += sign * (g_eval.eg_value[pt] + g_eval.eg_psqt[pt][idx]);
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
    int stm_score = (pos.side_to_move() == WHITE) ? score : -score;
    // Tempo: a small (tapered) bonus for having the move.
    int tempo = (g_eval.tempo_mg * phase + g_eval.tempo_eg * (24 - phase)) / 24;
    return stm_score + tempo;
}

// ── evaluate (dispatcher) ────────────────────────────────────────────────────
// Single symbol the search calls; selected at compile time by the EVAL CMake
// option (-DEVAL=NNUE default, or -DEVAL=HCE).
int evaluate(const Position& pos) {
#ifdef EVAL_NNUE
    // Read the position's incremental accumulator (kept in sync by do/undo_move
    // and refreshed in set_fen/copy_from). stm perspective first.
    return nnue::evaluate_acc(pos.accumulator(), pos.side_to_move());
#else
    return evaluate_hce(pos);
#endif
}

} // namespace king
