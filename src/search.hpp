#pragma once
#include <atomic>
#include "types.hpp"
#include "position.hpp"
#include "timeman.hpp"

namespace king {
namespace search {

// Find the best move for the side to move in `pos`.
// Returns 0 (null move) only if there are no legal moves (mate/stalemate).
// `stop` is polled frequently and causes early return when set.
// `threads` is accepted for API compatibility but ignored (SMP is a later plan).
Move think(Position& pos, const Limits& limits, std::atomic<bool>& stop,
           int move_overhead, int threads);

} // namespace search
} // namespace king
