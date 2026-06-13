#pragma once
#include <cstdint>
#include "position.hpp"

namespace king {

// Returns the number of leaf nodes at exactly `depth` plies from `pos`.
uint64_t perft(Position& pos, int depth);

} // namespace king
