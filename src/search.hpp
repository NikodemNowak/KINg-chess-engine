#pragma once
#include <atomic>
#include <iostream>
#include <ostream>
#include <mutex>
#include "types.hpp"
#include "position.hpp"
#include "timeman.hpp"

namespace king {
namespace search {

// Find the best move for the side to move in `pos`.
// Returns 0 (null move) only if there are no legal moves (mate/stalemate).
// `stop` is polled frequently and causes early return when set.
// `threads` selects the number of Lazy-SMP search threads (clamped to [1,256]);
// each thread gets a private Searcher + Position clone and they share the global
// TT and `stop`. `pos` is not mutated (each thread searches its own clone).
// `out` receives info lines (only the main thread prints); writes are
// serialized by `out_mtx` when non-null.
Move think(Position& pos, const Limits& limits, std::atomic<bool>& stop,
           int move_overhead, int threads,
           std::ostream& out = std::cout, std::mutex* out_mtx = nullptr);

} // namespace search
} // namespace king
