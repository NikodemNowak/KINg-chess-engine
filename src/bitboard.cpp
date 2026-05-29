#include "bitboard.hpp"
#include <cstdlib>
#include <string>

namespace king {

// ── Static tables ─────────────────────────────────────────────────────────
static Bitboard s_between[64][64];
static Bitboard s_line[64][64];
static bool     s_initialized = false;

// ── Geometric helpers (used only during init) ─────────────────────────────

// Returns a unit delta (in square index space) if a and b are on the same
// rank, file, or diagonal, otherwise 0.
// Also validates that the step does NOT wrap files for non-vertical moves.
static int alignment_delta(Square a, Square b) {
    int da = static_cast<int>(a);
    int db = static_cast<int>(b);
    int diff = db - da;

    int fa = file_of(a);
    int fb = file_of(b);
    int ra = rank_of(a);
    int rb = rank_of(b);

    // Same file (vertical)
    if (fa == fb) {
        return (diff > 0) ? 8 : -8;
    }
    // Same rank (horizontal)
    if (ra == rb) {
        return (diff > 0) ? 1 : -1;
    }
    // Diagonal check: |rank diff| == |file diff|
    int dr = rb - ra;
    int df = fb - fa;
    if (dr == df) {
        return (dr > 0) ? 9 : -9;   // NORTH_EAST / SOUTH_WEST
    }
    if (dr == -df) {
        return (dr > 0) ? 7 : -7;   // NORTH_WEST / SOUTH_EAST
    }
    return 0; // not aligned
}

namespace bitboard {

void init() {
    if (s_initialized) return;

    for (int ai = 0; ai < 64; ++ai) {
        for (int bi = 0; bi < 64; ++bi) {
            s_between[ai][bi] = 0ULL;
            s_line[ai][bi]    = 0ULL;
        }
    }

    for (int ai = 0; ai < 64; ++ai) {
        for (int bi = 0; bi < 64; ++bi) {
            if (ai == bi) {
                s_line[ai][bi] = square_bb(static_cast<Square>(ai));
                continue;
            }
            Square a = static_cast<Square>(ai);
            Square b = static_cast<Square>(bi);

            int delta = alignment_delta(a, b);
            if (delta == 0) {
                // not aligned; leave both tables as 0
                continue;
            }

            // ── between: walk from a toward b, collect strictly interior squares ──
            Bitboard between = 0ULL;
            int cur = ai + delta;
            int prev_file = file_of(a);
            bool valid = true;

            // For non-vertical steps, check each file step is exactly ±1
            while (cur != bi) {
                // Guard: cur must stay in [0,63]
                if (cur < 0 || cur > 63) { valid = false; break; }

                // For diagonal / horizontal moves, check no file wrap
                if (delta != 8 && delta != -8) {
                    int cur_file = cur & 7;
                    int expected_file_diff = (delta > 0) ? 1 : -1;
                    if (delta == 9)   expected_file_diff =  1;
                    if (delta == 7)   expected_file_diff = -1;
                    if (delta == -9)  expected_file_diff = -1;
                    if (delta == -7)  expected_file_diff =  1;
                    if (delta == 1)                  expected_file_diff =  1;
                    if (delta == -1)                 expected_file_diff = -1;
                    if (cur_file - prev_file != expected_file_diff) {
                        valid = false; break;
                    }
                    prev_file = cur_file;
                }
                between |= square_bb(static_cast<Square>(cur));
                cur += delta;
            }

            if (!valid) {
                // alignment_delta should have caught this; defensive only
                continue;
            }

            s_between[ai][bi] = between;

            // ── line: extend in both directions to board edges ──
            Bitboard line = square_bb(a) | square_bb(b) | between;

            // Extend from a in -delta direction
            {
                int c = ai - delta;
                int pf = file_of(a);
                while (c >= 0 && c <= 63) {
                    if (delta != 8 && delta != -8) {
                        int cf = c & 7;
                        int efd = (delta > 0) ? -1 : 1;
                        if (delta == 9)   efd = -1;
                        if (delta == 7)   efd =  1;
                        if (delta == -9)  efd =  1;
                        if (delta == -7)  efd = -1;
                        if (delta == 1)                  efd = -1;
                        if (delta == -1)                 efd =  1;
                        if (cf - pf != efd) break;
                        pf = cf;
                    }
                    line |= square_bb(static_cast<Square>(c));
                    c -= delta;
                }
            }

            // Extend from b in +delta direction
            {
                int c = bi + delta;
                int pf = file_of(b);
                while (c >= 0 && c <= 63) {
                    if (delta != 8 && delta != -8) {
                        int cf = c & 7;
                        int efd = (delta > 0) ? 1 : -1;
                        if (delta == 9)   efd =  1;
                        if (delta == 7)   efd = -1;
                        if (delta == -9)  efd = -1;
                        if (delta == -7)  efd =  1;
                        if (delta == 1)                  efd =  1;
                        if (delta == -1)                 efd = -1;
                        if (cf - pf != efd) break;
                        pf = cf;
                    }
                    line |= square_bb(static_cast<Square>(c));
                    c += delta;
                }
            }

            s_line[ai][bi] = line;
        }
    }

    s_initialized = true;
}

} // namespace bitboard

// ── Table accessors ───────────────────────────────────────────────────────
Bitboard between_bb(Square a, Square b) { return s_between[a][b]; }
Bitboard line_bb(Square a, Square b)    { return s_line[a][b]; }

// ── Pretty printer ────────────────────────────────────────────────────────
std::string pretty(Bitboard b) {
    std::string s;
    s.reserve(9 * 8);
    for (int r = 7; r >= 0; --r) {
        for (int f = 0; f < 8; ++f) {
            s += (b & square_bb(make_square(static_cast<File>(f), static_cast<Rank>(r)))) ? 'X' : '.';
        }
        s += '\n';
    }
    return s;
}

} // namespace king
