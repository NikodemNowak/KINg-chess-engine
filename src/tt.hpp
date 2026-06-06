#pragma once
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <vector>
#include "types.hpp"

// Transposition-table associativity. TT_WAYS slots share one bucket (probed
// together); a store evicts the least-valuable slot in the bucket (depth-/age-
// preferred) instead of always clobbering the single direct-mapped slot — fewer
// useful entries lost to collisions → higher hit rate → effective NPS. Default 1
// = the original direct-mapped table (bit-identical behaviour); -DTT_WAYS=4 makes
// it 4-way (one 64-byte cache line per bucket, so a bucket probe is one miss).
#ifndef TT_WAYS
#define TT_WAYS 1
#endif

namespace king {

// Bound type of a stored score relative to the search window at store time.
//   BOUND_EXACT  — score is exact (was inside the alpha/beta window, a PV node)
//   BOUND_LOWER  — score is a lower bound (fail-high, score >= beta)
//   BOUND_UPPER  — score is an upper bound (fail-low, score <= alpha)
enum Bound : uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

// A single transposition-table slot as seen by callers (probe output / store
// input). `score`/`eval` are mate-adjusted by the search (by ply) at the
// store/probe boundary — the TT itself stores them verbatim.
//
// Internally the table does NOT store this struct directly. Each physical slot
// is a pair of std::atomic<uint64_t> words (see TTSlot below): one "data" word
// packing {move, score, eval, depth, genBound}, and one "key" word holding
// (zobristKey XOR data). This is the lockless XOR scheme (Hyatt): a torn,
// half-updated slot fails the recomputed-key check on probe and is treated as a
// miss, so concurrent probe/store can never return a corrupt entry and no locks
// are needed in the hot path.
struct TTEntry {
    uint16_t key16;    // upper 16 bits of the zobrist key (collision-check, informational)
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
    // an entry whose recomputed key matches the full 64-bit `key` (lockless XOR
    // check, which also rejects torn entries). On a miss `tte` is left
    // untouched-meaningful (caller must ignore it on false).
    bool probe(uint64_t key, TTEntry& tte) const;

    // Store/refresh the entry for `key`. Replacement policy: write if the slot
    // is empty, holds the same position, is from an older generation, or the
    // new depth is at least (stored.depth - 2). A non-zero stored move is
    // preserved when the incoming move is 0 (don't clobber a good move).
    void store(uint64_t key, Move m, int16_t score, int16_t eval,
               uint8_t depth, Bound b);

    // Prefetch the cache line of `key`'s slot into cache. Called right after a
    // do_move (once the child's zobrist key is known) so the ~100-cycle DRAM
    // latency of the child's TT probe overlaps the rest of the move-loop / eval
    // work instead of stalling the recursive probe. Harmless before resize
    // (data() is valid; prefetch never dereferences). Guarded for GCC/Clang.
    void prefetch(uint64_t key) const {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(table_.data() + (key & mask_));
#else
        (void)key;
#endif
    }

    // Approximate fill level in permille (0..1000), sampled over the first
    // 1000 slots. For UCI `info hashfull`.
    int hashfull() const;

    // Number of allocated entries (0 before the first resize). Lets callers
    // lazily size the table before the first search.
    size_t size() const { return table_.size(); }

private:
    // One physical slot: two relaxed-atomic words implementing the XOR scheme.
    //   data = packed {move:16, score:16, eval:16, depth:8, genBound:8}
    //   key  = zobristKey XOR data
    // Reading key^data must reproduce the original zobristKey for the slot to be
    // considered a valid (non-torn) match. Relaxed atomics keep the accesses
    // data-race-free under the C++ memory model (so ThreadSanitizer is clean)
    // while imposing no ordering cost; correctness of the *value* relies only on
    // the XOR self-consistency check, not on inter-thread ordering.
    struct TTSlot {
        std::atomic<uint64_t> key{0};
        std::atomic<uint64_t> data{0};
    };

    // A bucket = TT_WAYS slots probed together. Aligned to a cache line when the
    // bucket fills one (TT_WAYS=4 → 64 bytes) so a bucket probe is a single miss.
    struct alignas(TT_WAYS * sizeof(TTSlot) >= 64 ? 64 : alignof(TTSlot)) TTBucket {
        TTSlot slot[TT_WAYS];
    };

    std::vector<TTBucket> table_;
    size_t  mask_       = 0; // (number of buckets - 1); bucket index = key & mask_
    uint8_t generation_ = 0;
};

// Global single instance. Lockless / thread-safe (XOR scheme) for Lazy SMP.
extern TT tt;

} // namespace king
