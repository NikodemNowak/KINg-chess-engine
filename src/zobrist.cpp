#include "zobrist.hpp"
#include <cstdint>

namespace king {
namespace zobrist {

uint64_t psq[12][64];
uint64_t side;
uint64_t castling[16];
uint64_t enpassant[8];

void init() {
  static bool done = false;
  if (done) return;

  // Fixed seed splitmix64 PRNG
  static uint64_t s = 0x9E3779B97F4A7C15ULL;
  auto rnd = [&]() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  };

  // Fill psq (12 pieces × 64 squares)
  for (int piece = 0; piece < 12; ++piece) {
    for (int sq = 0; sq < 64; ++sq) {
      psq[piece][sq] = rnd();
    }
  }

  // Fill side
  side = rnd();

  // Fill castling (all 16 masks)
  for (int mask = 0; mask < 16; ++mask) {
    castling[mask] = rnd();
  }

  // Fill enpassant (all 8 files)
  for (int file = 0; file < 8; ++file) {
    enpassant[file] = rnd();
  }

  done = true;
}

} // namespace zobrist
} // namespace king
