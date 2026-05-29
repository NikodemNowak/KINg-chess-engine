#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include "types.hpp"

namespace king {

// Bound type of a stored score relative to the search window at store time.
//   BOUND_EXACT  — score is exact (was inside the alpha/beta window, a PV node)
//   BOUND_LOWER  — score is a lower bound (fail-high, score >= beta)
//   BOUND_UPPER  — score is an upper bound (fail-low, score <= alpha)
enum Bound : uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

// A single transposition-table slot. Kept small (16 bytes) so many fit per
// cache line. `score`/`eval` are mate-adjusted by the search (by ply) at the
// store/probe boundary — the TT itself stores them verbatim.
struct TTEntry {
    uint16_t key16;    // upper 16 bits of the zobrist key (cheap collision check)
    Move     move;     // best / refutation move (used for move ordering)
    int16_t  score;    // search score (mate scores are ply-adjusted by search)
    int16_t  eval;     // static eval (optional; 0 if unused)
    uint8_t  depth;    // search depth this entry was produced at
    uint8_t  genBound; // (generation << 2) | Bound
};

class TT {
public:
    // Allocate the largest power-of-two number of entries that fits `mb`
    // megabytes, then clear. Always allocates at least one entry.
    void resize(size_t mb);

    // Zero all entries (empty). Keeps the current size and generation.
    void clear();

    // Begin a new search: bump the generation so older entries become
    // preferred replacement victims.
    void new_search();

    // Probe the slot for `key`. Returns true and fills `tte` iff the slot holds
    // an entry whose key16 matches (collision-checked). On a miss `tte` is left
    // with whatever was in the slot (caller should ignore it on false).
    bool probe(uint64_t key, TTEntry& tte) const;

    // Store/refresh the entry for `key`. Replacement policy: write if the slot
    // is empty, holds the same position, is from an older generation, or the
    // new depth is at least (stored.depth - 2). A non-zero stored move is
    // preserved when the incoming move is 0 (don't clobber a good move).
    void store(uint64_t key, Move m, int16_t score, int16_t eval,
               uint8_t depth, Bound b);

    // Approximate fill level in permille (0..1000), sampled over the first
    // 1000 slots. For UCI `info hashfull`.
    int hashfull() const;

    // Number of allocated entries (0 before the first resize). Lets callers
    // lazily size the table before the first search.
    size_t size() const { return table_.size(); }

private:
    std::vector<TTEntry> table_;
    size_t  mask_       = 0; // (number of entries - 1); index = key & mask_
    uint8_t generation_ = 0;
};

// Global single instance. Single-threaded for now; SMP-safety is a later task.
extern TT tt;

} // namespace king
