// NNUE inference implementation. Only compiled in EVAL_NNUE builds (the whole
// file body is guarded so an HCE build that happens to glob this .cpp still
// produces an empty TU).
#include "nnue.hpp"

#ifdef EVAL_NNUE

#include "position.hpp"
#include "bitboard.hpp"
#include <cstdint>
#include <cstring>
#include <cstdlib>

namespace king {
namespace nnue {

// Embedded net bytes (generated at build time from nets/king_nnue.bin).
extern "C" const unsigned char king_nnue_data[];
extern "C" const unsigned int  king_nnue_data_len;

namespace {

// Parsed network. W1 is transposed to [input][output] so that the 256 weights
// belonging to one feature are contiguous (the usual NNUE layout) — that makes
// the accumulator update a single contiguous add/sub over HL int16s.
struct Net {
    int16_t W1t[INPUT][HL]; // transposed: W1t[i][o] == W1q[o][i]
    int16_t b1[HL];
    int16_t W2[2 * HL];     // [0..255] stm, [256..511] nonstm
    int32_t b2;
};

Net g_net;
bool g_loaded = false;

template <typename T>
T read_le(const unsigned char* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v; // engine targets little-endian (x86-64 / AVX2); memcpy keeps layout
}

} // namespace

void init() {
    if (g_loaded) return;

    const unsigned char* p = king_nnue_data;
    const unsigned int   n = king_nnue_data_len;

    // Header: magic(u32) HL(u16) QA(u16) QB(u16) SCALE(u16) = 12 bytes.
    constexpr unsigned int kExpected =
        12u + (unsigned)(HL * INPUT) * 2u + (unsigned)HL * 2u +
        (unsigned)(2 * HL) * 2u + 4u; // 394768
    if (n != kExpected) std::abort();

    uint32_t magic = read_le<uint32_t>(p + 0);
    uint16_t hl    = read_le<uint16_t>(p + 4);
    uint16_t qa    = read_le<uint16_t>(p + 6);
    uint16_t qb    = read_le<uint16_t>(p + 8);
    uint16_t scale = read_le<uint16_t>(p + 10);
    if (magic != 0x4B4E5545u) std::abort();
    if (hl != HL || qa != QA || qb != QB || scale != SCALE) std::abort();

    unsigned int off = 12;

    // W1q is stored row-major [output o][input i] at flat o*INPUT + i.
    // Transpose into W1t[i][o].
    for (int o = 0; o < HL; ++o)
        for (int i = 0; i < INPUT; ++i) {
            g_net.W1t[i][o] = read_le<int16_t>(p + off);
            off += 2;
        }

    for (int o = 0; o < HL; ++o) { g_net.b1[o] = read_le<int16_t>(p + off); off += 2; }
    for (int i = 0; i < 2 * HL; ++i) { g_net.W2[i] = read_le<int16_t>(p + off); off += 2; }
    g_net.b2 = read_le<int32_t>(p + off); off += 4;

    g_loaded = true;
}

// ── crelu ─────────────────────────────────────────────────────────────────
static inline int crelu(int v) {
    if (v < 0) return 0;
    if (v > QA) return QA;
    return v;
}

// ── Output layer on an up-to-date accumulator ───────────────────────────────
int evaluate_acc(const Accumulator& acc, Color stm) {
    const Color nonstm = Color(!stm);
    int64_t sum = g_net.b2;
    // stm perspective FIRST -> W2[0..255]; nonstm -> W2[256..511].
    const int16_t* a_stm = acc.v[stm];
    const int16_t* a_nst = acc.v[nonstm];
    for (int i = 0; i < HL; ++i)
        sum += (int64_t)crelu(a_stm[i]) * g_net.W2[i];
    for (int i = 0; i < HL; ++i)
        sum += (int64_t)crelu(a_nst[i]) * g_net.W2[HL + i];

    // eval = floor( sum * SCALE / (QA*QB) ) with floor toward -inf (Python //).
    int64_t num = sum * (int64_t)SCALE;
    int64_t den = (int64_t)QA * QB; // 16320
    int64_t q = num / den;
    if ((num % den != 0) && ((num < 0) != (den < 0))) q -= 1;
    return (int)q;
}

// ── Accumulator maintenance ─────────────────────────────────────────────────
void refresh(Accumulator& acc, const Position& pos) {
    if (!g_loaded) init(); // lazy safety net: refresh is the single choke point
    // Start both perspectives at the bias.
    for (int c = 0; c < COLOR_NB; ++c)
        std::memcpy(acc.v[c], g_net.b1, sizeof(g_net.b1));

    // Add every piece on the board to both perspectives.
    for (int c = WHITE; c <= BLACK; ++c)
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard bb = pos.pieces((Color)c, (PieceType)pt);
            while (bb) {
                Square s = pop_lsb(bb);
                add_feature(acc, (Color)c, (PieceType)pt, s);
            }
        }
}

void add_feature(Accumulator& acc, Color c, PieceType t, Square s) {
    const int iw = feature_index(c, t, s, WHITE);
    const int ib = feature_index(c, t, s, BLACK);
    const int16_t* cw = g_net.W1t[iw];
    const int16_t* cb = g_net.W1t[ib];
    int16_t* aw = acc.v[WHITE];
    int16_t* ab = acc.v[BLACK];
    for (int o = 0; o < HL; ++o) { aw[o] += cw[o]; ab[o] += cb[o]; }
}

void sub_feature(Accumulator& acc, Color c, PieceType t, Square s) {
    const int iw = feature_index(c, t, s, WHITE);
    const int ib = feature_index(c, t, s, BLACK);
    const int16_t* cw = g_net.W1t[iw];
    const int16_t* cb = g_net.W1t[ib];
    int16_t* aw = acc.v[WHITE];
    int16_t* ab = acc.v[BLACK];
    for (int o = 0; o < HL; ++o) { aw[o] -= cw[o]; ab[o] -= cb[o]; }
}

// ── From-scratch reference evaluation ────────────────────────────────────────
int evaluate_from_scratch(const Position& pos) {
    Accumulator acc;
    refresh(acc, pos);
    return evaluate_acc(acc, pos.side_to_move());
}

} // namespace nnue
} // namespace king

#endif // EVAL_NNUE
