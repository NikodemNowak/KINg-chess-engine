#include "attacks.hpp"
#include "bitboard.hpp"

#include <cstdint>

namespace king {

// Global attack tables
Bitboard pawn_attacks[2][64];
Bitboard knight_attacks[64];
Bitboard king_attacks[64];

// ── Magic bitboards (plain, multiply-based — no PEXT) ──────────────────────
namespace {

struct Magic {
  Bitboard  mask;   // relevant-occupancy mask
  Bitboard  magic;  // magic multiplier
  Bitboard* table;  // pointer into the per-piece attack table for this square
  unsigned  shift;  // 64 - popcount(mask)

  unsigned index(Bitboard occ) const {
    return unsigned(((occ & mask) * magic) >> shift);
  }
};

Magic rook_magic[64];
Magic bishop_magic[64];

// Per-square attack tables. Worst-case rook mask has 12 bits (corners less, but
// 4096 covers all); bishop worst-case is 9 bits (512). ~2.25 MB total.
Bitboard rook_table[64][4096];
Bitboard bishop_table[64][512];

// Reference slider: step in each of 4 directions, include first blocker, stop.
Bitboard slider_attacks(Square s, Bitboard occ, bool rook) {
  Bitboard att = 0;
  int r = rank_of(s), f = file_of(s);
  static const int R[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
  static const int B[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
  for (int d = 0; d < 4; ++d) {
    int dr = rook ? R[d][0] : B[d][0];
    int df = rook ? R[d][1] : B[d][1];
    int rr = r + dr, ff = f + df;
    while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) {
      Square t = make_square(File(ff), Rank(rr));
      att |= square_bb(t);
      if (occ & square_bb(t)) break;
      rr += dr; ff += df;
    }
  }
  return att;
}

// Relevant-occupancy mask: rays from s minus the board-edge squares (edges
// never affect what is seen beyond them) and minus s itself.
Bitboard slider_mask(Square s, bool rook) {
  Bitboard mask = 0;
  int r = rank_of(s), f = file_of(s);
  static const int R[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
  static const int B[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
  for (int d = 0; d < 4; ++d) {
    int dr = rook ? R[d][0] : B[d][0];
    int df = rook ? R[d][1] : B[d][1];
    int rr = r + dr, ff = f + df;
    // Walk the ray; include a square only if the NEXT step is still on board,
    // i.e. drop the board-edge square in this direction.
    while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) {
      int nr = rr + dr, nf = ff + df;
      bool next_on_board = (nr >= 0 && nr < 8 && nf >= 0 && nf < 8);
      if (next_on_board) mask |= square_bb(make_square(File(ff), Rank(rr)));
      rr = nr; ff = nf;
    }
  }
  return mask;
}

// splitmix64 with a fixed seed → deterministic magics every run.
uint64_t magic_rng_state = 0xD06C659D9C46A3B1ULL;
uint64_t magic_rng() {
  uint64_t z = (magic_rng_state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
// Sparse candidate (few set bits) → better magic distribution.
uint64_t sparse_rand() { return magic_rng() & magic_rng() & magic_rng(); }

void init_magic_for(bool rook) {
  Magic* magics = rook ? rook_magic : bishop_magic;
  for (Square s = A1; s <= H8; s = Square(s + 1)) {
    Bitboard mask = slider_mask(s, rook);
    int bits = popcount(mask);
    unsigned size = 1u << bits;

    Magic& m = magics[s];
    m.mask  = mask;
    m.shift = 64u - unsigned(bits);
    m.table = rook ? rook_table[s] : bishop_table[s];

    // Enumerate every occupancy subset (carry-rippler) and its reference attack.
    // Stored in parallel arrays so we can validate candidate magics quickly.
    static Bitboard occs[4096];
    static Bitboard refs[4096];
    Bitboard b = 0;
    unsigned n = 0;
    do {
      occs[n] = b;
      refs[n] = slider_attacks(s, b, rook);
      ++n;
      b = (b - mask) & mask;
    } while (b);

    // Search for a magic that produces no destructive collisions.
    static Bitboard used[4096];
    for (;;) {
      Bitboard cand = sparse_rand();
      // Heuristic: require the high byte of mask*magic to be reasonably populated.
      if (popcount((mask * cand) & 0xFF00000000000000ULL) < 6) continue;

      for (unsigned i = 0; i < size; ++i) used[i] = 0;
      bool ok = true;
      for (unsigned i = 0; i < n; ++i) {
        unsigned idx = unsigned((occs[i] * cand) >> m.shift);
        if (used[idx] == 0) {
          used[idx] = refs[i] ? refs[i] : ~Bitboard(0); // sentinel for empty attack
        } else if (used[idx] != (refs[i] ? refs[i] : ~Bitboard(0))) {
          ok = false; // destructive collision
          break;
        }
      }
      if (!ok) continue;

      m.magic = cand;
      // Commit: fill the real table with actual reference attacks.
      for (unsigned i = 0; i < size; ++i) m.table[i] = 0;
      for (unsigned i = 0; i < n; ++i) {
        unsigned idx = unsigned((occs[i] * cand) >> m.shift);
        m.table[idx] = refs[i];
      }
      break;
    }
  }
}

} // anonymous namespace

Bitboard rook_attacks(Square s, Bitboard occ) {
  const Magic& m = rook_magic[s];
  return m.table[m.index(occ)];
}

Bitboard bishop_attacks(Square s, Bitboard occ) {
  const Magic& m = bishop_magic[s];
  return m.table[m.index(occ)];
}

namespace attacks {

void init_leapers() {
  // Initialize pawn attacks for both colors
  for (Square s = A1; s <= H8; s = Square(s + 1)) {
    Bitboard b = square_bb(s);

    // White pawns attack diagonally upward (NORTH)
    pawn_attacks[WHITE][s] = shift<NORTH_WEST>(b) | shift<NORTH_EAST>(b);

    // Black pawns attack diagonally downward (SOUTH)
    pawn_attacks[BLACK][s] = shift<SOUTH_WEST>(b) | shift<SOUTH_EAST>(b);
  }

  // Initialize knight attacks (branchless, wrap-safe)
  for (Square s = A1; s <= H8; s = Square(s + 1)) {
    Bitboard b = square_bb(s);

    Bitboard l1 = (b >> 1) & ~FILE_H_BB;
    Bitboard l2 = (b >> 2) & ~(FILE_G_BB | FILE_H_BB);
    Bitboard r1 = (b << 1) & ~FILE_A_BB;
    Bitboard r2 = (b << 2) & ~(FILE_A_BB | FILE_B_BB);
    Bitboard h1 = l1 | r1;
    Bitboard h2 = l2 | r2;
    knight_attacks[s] = (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
  }

  // Initialize king attacks (all 8 directions)
  for (Square s = A1; s <= H8; s = Square(s + 1)) {
    Bitboard b = square_bb(s);
    king_attacks[s] = shift<NORTH>(b) | shift<SOUTH>(b) | shift<EAST>(b) | shift<WEST>(b) |
                      shift<NORTH_EAST>(b) | shift<NORTH_WEST>(b) | shift<SOUTH_EAST>(b) | shift<SOUTH_WEST>(b);
  }
}

void init_magics() {
  static bool done = false;
  if (done) return;          // idempotent
  init_magic_for(true);      // rooks
  init_magic_for(false);     // bishops
  done = true;
}

} // namespace attacks

} // namespace king
