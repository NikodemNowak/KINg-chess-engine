#include "tt.hpp"
#include <cstring>

namespace king {

TT tt; // global instance

// Is this slot unused? An all-zero entry is treated as empty. (A genuine entry
// could in principle look empty — key16==0, depth==0, move==0 — but such an
// entry carries no useful info anyway, so treating it as empty is harmless.)
static inline bool is_empty(const TTEntry& e) {
    return e.key16 == 0 && e.depth == 0 && e.move == 0;
}

void TT::resize(size_t mb) {
    // Largest power-of-two entry count whose byte size fits `mb` MiB.
    size_t bytes   = mb * 1024 * 1024;
    size_t entries = bytes / sizeof(TTEntry);
    if (entries < 1) entries = 1;

    // floor to power of two
    size_t pow2 = 1;
    while (pow2 * 2 <= entries) pow2 *= 2;

    table_.assign(pow2, TTEntry{}); // allocate + zero (clears)
    mask_ = pow2 - 1;
}

void TT::clear() {
    std::memset(table_.data(), 0, table_.size() * sizeof(TTEntry));
}

void TT::new_search() {
    ++generation_; // wraps at 256; old entries simply look "different"
}

bool TT::probe(uint64_t key, TTEntry& tte) const {
    if (table_.empty()) return false;
    const TTEntry& e = table_[key & mask_];
    if (e.key16 == (uint16_t)(key >> 48) && !is_empty(e)) {
        tte = e;
        return true;
    }
    return false;
}

void TT::store(uint64_t key, Move m, int16_t score, int16_t eval,
               uint8_t depth, Bound b) {
    if (table_.empty()) return;

    TTEntry&       slot  = table_[key & mask_];
    const uint16_t key16 = (uint16_t)(key >> 48);
    const uint8_t  gen   = generation_;

    const bool sameKey   = (slot.key16 == key16) && !is_empty(slot);
    const uint8_t slotGen = (uint8_t)(slot.genBound >> 2);

    // Replace if: slot empty, same position, slot is from an older search, or
    // the new entry is searched at least as deep (with a 2-ply slack).
    bool replace = is_empty(slot)
                || sameKey
                || slotGen != gen
                || depth + 2 >= slot.depth;
    if (!replace) return;

    // Preserve a useful stored move if the incoming move is null and we're
    // overwriting the same position (don't lose good ordering info).
    if (m == 0 && sameKey && slot.move != 0)
        m = slot.move;

    slot.key16    = key16;
    slot.move     = m;
    slot.score    = score;
    slot.eval     = eval;
    slot.depth    = depth;
    slot.genBound = (uint8_t)((gen << 2) | (uint8_t)b);
}

int TT::hashfull() const {
    if (table_.empty()) return 0;
    size_t sample = table_.size() < 1000 ? table_.size() : 1000;
    int used = 0;
    for (size_t i = 0; i < sample; ++i)
        if (!is_empty(table_[i])
            && (uint8_t)(table_[i].genBound >> 2) == generation_)
            ++used;
    return (int)((int64_t)used * 1000 / (int64_t)sample);
}

} // namespace king
