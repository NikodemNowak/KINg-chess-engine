#pragma once
#include <cstdint>

namespace king {
namespace zobrist {

extern uint64_t psq[12][64];      // [piece 0..11][square 0..63]
extern uint64_t side;             // XOR'd when side to move is BLACK
extern uint64_t castling[16];     // indexed by 4-bit castling-rights mask
extern uint64_t enpassant[8];     // indexed by file of the ep square

void init();                       // fills all of the above; idempotent

} // namespace zobrist
} // namespace king
