#include "tt.hpp"
#include <cstring>
#include <climits>

// Depth-preferred replacement (OFF until SPRT, stacked after the gen fix): a
// shallow same-position store must not clobber a meaningfully-deeper entry's
// score/move. -DTT_DEPTHREPL=1.
#ifndef TT_DEPTHREPL
#define TT_DEPTHREPL 1
#endif

namespace king {

TT tt; // global instance

// ── XOR-lockless packing ──────────────────────────────────────────────────────
// The "data" word packs the entry payload into 64 bits:
//   bits  0..15 : move      (uint16)
//   bits 16..31 : score     (int16, stored as raw bits)
//   bits 32..47 : eval      (int16, stored as raw bits)
//   bits 48..55 : depth     (uint8)
//   bits 56..63 : genBound  (uint8)
// The "key" word stores (zobristKey XOR data). On probe we recompute
// key XOR data and compare to the position's full 64-bit zobrist key: a match
// proves the slot belongs to this position AND that the two words are mutually
// consistent (i.e. the slot was not read mid-write by another thread).
static inline uint64_t pack_data(Move m, int16_t score, int16_t eval,
                                 uint8_t depth, uint8_t genBound) {
    return  (uint64_t)(uint16_t)m
         | ((uint64_t)(uint16_t)score    << 16)
         | ((uint64_t)(uint16_t)eval     << 32)
         | ((uint64_t)depth              << 48)
         | ((uint64_t)genBound           << 56);
}

static inline void unpack_data(uint64_t d, TTEntry& e) {
    e.move     = (Move)(uint16_t)(d & 0xFFFF);
    e.score    = (int16_t)(uint16_t)((d >> 16) & 0xFFFF);
    e.eval     = (int16_t)(uint16_t)((d >> 32) & 0xFFFF);
    e.depth    = (uint8_t)((d >> 48) & 0xFF);
    e.genBound = (uint8_t)((d >> 56) & 0xFF);
}

// An entry whose packed data is 0 (move==0, score==0, eval==0, depth==0,
// genBound==0/BOUND_NONE) is treated as empty. Such an entry carries no useful
// information anyway, so this is harmless.
static inline bool is_empty_data(uint64_t d) { return d == 0; }

void TT::resize(size_t mb) {
    // Largest power-of-two bucket count whose byte size fits `mb` MiB.
    size_t bytes   = mb * 1024 * 1024;
    size_t buckets = bytes / sizeof(TTBucket);
    if (buckets < 1) buckets = 1;

    // floor to power of two
    size_t pow2 = 1;
    while (pow2 * 2 <= buckets) pow2 *= 2;

    // std::atomic is not copyable/movable, so resize the vector to the target
    // size (default-constructs slots = {0,0}) rather than assign(n, value).
    table_.clear();
    table_ = std::vector<TTBucket>(pow2);
    mask_ = pow2 - 1;
}

void TT::clear() {
    // Relaxed stores are fine: clear() is only called between searches with no
    // workers running, but using atomic stores keeps the access data-race-free
    // under the memory model regardless.
    for (auto& b : table_)
        for (auto& s : b.slot) {
            s.key.store(0, std::memory_order_relaxed);
            s.data.store(0, std::memory_order_relaxed);
        }
}

void TT::new_search() {
    ++generation_; // wraps at 256; old entries simply look "different"
}

bool TT::probe(uint64_t key, TTEntry& tte) const {
    if (table_.empty()) return false;
    const TTBucket& bucket = table_[key & mask_];
    // Scan the bucket's slots; the XOR self-consistency check (not memory
    // ordering) is what guarantees we never accept a torn/foreign entry.
    for (int j = 0; j < TT_WAYS; ++j) {
        uint64_t data = bucket.slot[j].data.load(std::memory_order_relaxed);
        uint64_t k    = bucket.slot[j].key.load(std::memory_order_relaxed);
        if (is_empty_data(data)) continue;
        if ((k ^ data) != key) continue;   // foreign position OR torn read → skip
        unpack_data(data, tte);
        tte.key16 = (uint16_t)(key >> 48);  // informational (collision-check bits)
        return true;
    }
    return false;
}

void TT::store(uint64_t key, Move m, int16_t score, int16_t eval,
               uint8_t depth, Bound b) {
    if (table_.empty()) return;

    TTBucket& bucket = table_[key & mask_];
    // genBound packs (gen<<2)|bound into 8 bits, so only the low 6 bits of the
    // generation survive. Mask here so the store, the slotGen compare, and the
    // victim-age math are all 6-bit-consistent — otherwise once generation_>=64
    // (≈ move 64 of a match) slotGen!=gen is ALWAYS true and depth-preferred
    // replacement + aging silently break for the rest of the match.
    const uint8_t gen = generation_ & 63;

    // Pick the target slot in the bucket: an empty or same-position slot if one
    // exists, else the least-valuable victim (shallowest + oldest generation).
    // For TT_WAYS=1 this always resolves to slot[0] → identical to direct-mapped.
    int target = 0;
    {
        bool found = false;
        for (int j = 0; j < TT_WAYS; ++j) {
            uint64_t d = bucket.slot[j].data.load(std::memory_order_relaxed);
            if (is_empty_data(d)) { target = j; found = true; break; }
            uint64_t k = bucket.slot[j].key.load(std::memory_order_relaxed);
            if ((k ^ d) == key) { target = j; found = true; break; } // same position
        }
        if (!found) {
            int worst = INT_MAX;
            for (int j = 0; j < TT_WAYS; ++j) {
                uint64_t d = bucket.slot[j].data.load(std::memory_order_relaxed);
                TTEntry e{}; unpack_data(d, e);
                int age   = ((int)gen - (int)(e.genBound >> 2)) & 63; // modular 6-bit age
                int value = (int)e.depth - 2 * age; // shallow + old ⇒ low ⇒ best victim
                if (value < worst) { worst = value; target = j; }
            }
        }
    }
    TTSlot& slot = bucket.slot[target];

    // Read the current slot once (consistent snapshot via the XOR check).
    uint64_t curData = slot.data.load(std::memory_order_relaxed);
    uint64_t curKey  = slot.key.load(std::memory_order_relaxed);
    const bool curEmpty = is_empty_data(curData);
    const bool curValid = !curEmpty && ((curKey ^ curData) == key); // same position, intact
    TTEntry cur{};
    if (!curEmpty) unpack_data(curData, cur);

    const uint8_t slotGen = curEmpty ? 0 : (uint8_t)(cur.genBound >> 2);

    // Replace if: slot empty, same position, slot is from an older search, or
    // the new entry is searched at least as deep (with a 2-ply slack).
    // (A torn/foreign slot — !curEmpty && !curValid — is replaced too: depth
    //  slack vs. its decoded depth is a fine heuristic, and overwriting a
    //  garbage slot is always safe under the XOR scheme.)
    bool replace = curEmpty
#if TT_DEPTHREPL
                || (curValid && ((int)depth >= (int)cur.depth || b == BOUND_EXACT))
#else
                || curValid
#endif
                || slotGen != gen
                || (int)depth + 2 >= (int)cur.depth;
    if (!replace) return;

    // Preserve a useful stored move if the incoming move is null and we're
    // overwriting the same position (don't lose good ordering info).
    if (m == 0 && curValid && cur.move != 0)
        m = cur.move;

    uint64_t newData = pack_data(m, score, eval, depth, (uint8_t)((gen << 2) | (uint8_t)b));
    uint64_t newKey  = key ^ newData;

    // Publish the slot. Write the key word first then the data word with
    // relaxed ordering: a concurrent probe will only accept the slot if
    // key ^ data == its lookup key, so any interleaving of these two stores
    // with another thread's loads is rejected as a torn read rather than
    // returning a half-updated entry. No torn pointers / UB (atomic stores).
    slot.key.store(newKey, std::memory_order_relaxed);
    slot.data.store(newData, std::memory_order_relaxed);
}

int TT::hashfull() const {
    if (table_.empty()) return 0;
    size_t totalSlots = table_.size() * (size_t)TT_WAYS;
    size_t sample = totalSlots < 1000 ? totalSlots : 1000;
    int used = 0;
    for (size_t i = 0; i < sample; ++i) {
        const TTSlot& s = table_[i / TT_WAYS].slot[i % TT_WAYS];
        uint64_t d = s.data.load(std::memory_order_relaxed);
        if (!is_empty_data(d) && (uint8_t)(((d >> 56) >> 2) & 63) == (generation_ & 63))
            ++used;
    }
    return (int)((int64_t)used * 1000 / (int64_t)sample);
}

} // namespace king
