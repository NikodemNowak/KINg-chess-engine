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

#ifdef __AVX2__
#include <immintrin.h>
#endif

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

// ── Half-accumulator dot product: Σ crelu(a[i]) * w[i] over HL int16s ─────────
// Returns an int64. This is the per-perspective inner product used by the output
// layer. BIT-EXACTNESS CONTRACT: this MUST equal the scalar sum below for any
// inputs — integer add is associative and no intermediate overflows (each
// crelu(a)∈[0,QA=255], |w|≤32767 → |product|≤8.36M, two summed in madd ≤16.7M <
// 2^31; the int32 partials are then widened to int64 before accumulating, so the
// running sum never wraps). The AVX2 path therefore reproduces the reference int.
static inline int64_t dot_half(const int16_t* a, const int16_t* w) {
#ifdef __AVX2__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16((short)QA);
    __m256i acc0 = _mm256_setzero_si256(); // int64 x4
    __m256i acc1 = _mm256_setzero_si256(); // int64 x4
    for (int i = 0; i < HL; i += 16) {
        __m256i av = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i wv = _mm256_loadu_si256((const __m256i*)(w + i));
        // crelu: clamp to [0, QA] exactly as the scalar helper.
        av = _mm256_min_epi16(_mm256_max_epi16(av, zero), qa);
        // 16 int16 pairs -> 8 int32 partial sums (each = a[2k]*w[2k]+a[2k+1]*w[2k+1]).
        __m256i prod = _mm256_madd_epi16(av, wv);
        // Sign-extend the 8 int32 to int64 and accumulate (no int32 overflow risk).
        acc0 = _mm256_add_epi64(acc0, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod)));
        acc1 = _mm256_add_epi64(acc1, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod, 1)));
    }
    __m256i acc = _mm256_add_epi64(acc0, acc1);
    int64_t lanes[4];
    _mm256_storeu_si256((__m256i*)lanes, acc);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
#else
    int64_t sum = 0;
    for (int i = 0; i < HL; ++i)
        sum += (int64_t)crelu(a[i]) * w[i];
    return sum;
#endif
}

// ── Output layer on an up-to-date accumulator ───────────────────────────────
int evaluate_acc(const Accumulator& acc, Color stm) {
    const Color nonstm = Color(!stm);
    // stm perspective FIRST -> W2[0..255]; nonstm -> W2[256..511].
    int64_t sum = (int64_t)g_net.b2;
    sum += dot_half(acc.v[stm],    g_net.W2);
    sum += dot_half(acc.v[nonstm], g_net.W2 + HL);

    // eval = floor( sum * SCALE / (QA*QB) ) with floor toward -inf (Python //).
    // Kept SCALAR on the int64 sum so the result is bit-exact vs the reference.
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

// dst[o] += col[o] over HL int16s (wraps mod 2^16 identically to the scalar +=,
// which is what the int16 accumulator contract relies on).
static inline void acc_add(int16_t* dst, const int16_t* col) {
#ifdef __AVX2__
    for (int o = 0; o < HL; o += 16) {
        __m256i d = _mm256_loadu_si256((const __m256i*)(dst + o));
        __m256i c = _mm256_loadu_si256((const __m256i*)(col + o));
        _mm256_storeu_si256((__m256i*)(dst + o), _mm256_add_epi16(d, c));
    }
#else
    for (int o = 0; o < HL; ++o) dst[o] += col[o];
#endif
}

// dst[o] -= col[o] over HL int16s.
static inline void acc_sub(int16_t* dst, const int16_t* col) {
#ifdef __AVX2__
    for (int o = 0; o < HL; o += 16) {
        __m256i d = _mm256_loadu_si256((const __m256i*)(dst + o));
        __m256i c = _mm256_loadu_si256((const __m256i*)(col + o));
        _mm256_storeu_si256((__m256i*)(dst + o), _mm256_sub_epi16(d, c));
    }
#else
    for (int o = 0; o < HL; ++o) dst[o] -= col[o];
#endif
}

void add_feature(Accumulator& acc, Color c, PieceType t, Square s) {
    const int iw = feature_index(c, t, s, WHITE);
    const int ib = feature_index(c, t, s, BLACK);
    acc_add(acc.v[WHITE], g_net.W1t[iw]);
    acc_add(acc.v[BLACK], g_net.W1t[ib]);
}

void sub_feature(Accumulator& acc, Color c, PieceType t, Square s) {
    const int iw = feature_index(c, t, s, WHITE);
    const int ib = feature_index(c, t, s, BLACK);
    acc_sub(acc.v[WHITE], g_net.W1t[iw]);
    acc_sub(acc.v[BLACK], g_net.W1t[ib]);
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
