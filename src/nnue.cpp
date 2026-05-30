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

// ── Portable build + runtime AVX2 dispatch ───────────────────────────────────
// The binary is compiled for a portable x86-64 baseline (SSE2 only — universal
// on every x86-64 CPU) so it NEVER executes an illegal instruction on hardware
// without AVX2.  The three hot SIMD kernels below are each provided in a scalar
// form AND an `__attribute__((target("avx2")))` form; init() probes the CPU once
// via __builtin_cpu_supports("avx2") and points the dispatch pointers at the
// AVX2 versions only when the running CPU actually supports them.  This removes
// the SIGILL-on-startup catastrophe (crash == lost game) while keeping full AVX2
// speed wherever AVX2 is present.  See CMakeLists.txt (no global -mavx2).
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define KING_X86 1
#  include <immintrin.h>
#endif

namespace king {
namespace nnue {

// Embedded net bytes (generated at build time from nets/king_nnue.bin).
extern "C" const unsigned char king_nnue_data[];
extern "C" const unsigned int  king_nnue_data_len;

namespace {

// Parsed network. W1 is transposed to [input][output] so that the HL weights
// belonging to one feature are contiguous (the usual NNUE layout) — that makes
// the accumulator update a single contiguous add/sub over HL int16s.
// HL is a compile-time constant set by -DNNUE_HL (default 256; use 512 for the
// full retrain). All loops and array sizes below use HL, so changing the define
// is sufficient — nothing here is hardcoded to 256.
struct Net {
    int16_t W1t[FT_IN][HL];   // transposed: W1t[i][o] == W1q[o][i]; FT_IN = KB*768
    int16_t b1[HL];
    int16_t W2[OB][2 * HL];   // per output bucket: [0..HL-1] stm, [HL..2*HL-1] nonstm
    int32_t b2[OB];           // per output bucket bias
};

// Net binary magics. V1 (KNUE) legacy single-bucket (12-byte header). V2 (KNU2)
// adds a u16 OB after SCALE (14-byte header). V3 (KNU3) adds u16 OB then u16 KB
// (16-byte header) for king-bucketed (HalfKP-style) inputs. The loader accepts
// all three; a KB=1,OB=1 build still accepts the legacy V1 byte-for-byte.
constexpr uint32_t MAGIC_V1 = 0x4B4E5545u; // "KNUE"
constexpr uint32_t MAGIC_V2 = 0x4B4E5532u; // "KNU2"
constexpr uint32_t MAGIC_V3 = 0x4B4E5533u; // "KNU3"

// AVX2 loops in dot_half / acc_add / acc_sub process 16 int16s per iteration;
// HL must be a multiple of 16 for them to be correct.
static_assert(HL % 16 == 0, "NNUE_HL must be a multiple of 16 (AVX2 loop requirement)");

Net g_net;
bool g_loaded = false;

template <typename T>
T read_le(const unsigned char* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v; // engine targets little-endian (x86-64 / AVX2); memcpy keeps layout
}

} // namespace

static void init_simd_dispatch(); // defined below (after the SIMD kernels)

void init() {
    if (g_loaded) return;

    // Pick scalar vs AVX2 kernels for THIS CPU before any eval runs.
    init_simd_dispatch();

    const unsigned char* p = king_nnue_data;
    const unsigned int   n = king_nnue_data_len;

    // Header: magic(u32) HL(u16) QA(u16) QB(u16) SCALE(u16) [V2: OB(u16)].
    uint32_t magic = read_le<uint32_t>(p + 0);
    uint16_t hl    = read_le<uint16_t>(p + 4);
    uint16_t qa    = read_le<uint16_t>(p + 6);
    uint16_t qb    = read_le<uint16_t>(p + 8);
    uint16_t scale = read_le<uint16_t>(p + 10);
    if (hl != HL || qa != QA || qb != QB || scale != SCALE) std::abort();

    unsigned int off;
    unsigned int ob;
    unsigned int kb;
    if (magic == MAGIC_V1) {
        // Legacy single-bucket format. Only valid for an OB=1, KB=1 build.
        if (OB != 1 || KB != 1) std::abort();
        ob = 1; kb = 1; off = 12;
    } else if (magic == MAGIC_V2) {
        if (KB != 1) std::abort();
        ob = read_le<uint16_t>(p + 12);
        if ((int)ob != OB) std::abort();
        kb = 1; off = 14;
    } else if (magic == MAGIC_V3) {
        ob = read_le<uint16_t>(p + 12);
        kb = read_le<uint16_t>(p + 14);
        if ((int)ob != OB || (int)kb != KB) std::abort(); // must match compiled OB/KB
        off = 16;
    } else {
        std::abort();
    }

    // Validate total size for the resolved (header, OB, KB).
    const unsigned int ftin = kb * (unsigned)INPUT;
    const unsigned int expected =
        off + (unsigned)(HL * ftin) * 2u + (unsigned)HL * 2u +
        (unsigned)(ob * 2 * HL) * 2u + (unsigned)ob * 4u;
    if (n != expected) std::abort();

    // W1q is stored row-major [output o][input i] at flat o*FT_IN + i.
    // Transpose into W1t[i][o].
    for (int o = 0; o < HL; ++o)
        for (int i = 0; i < FT_IN; ++i) {
            g_net.W1t[i][o] = read_le<int16_t>(p + off);
            off += 2;
        }

    for (int o = 0; o < HL; ++o) { g_net.b1[o] = read_le<int16_t>(p + off); off += 2; }
    // W2: bucket-major [bucket][2*HL], stm-first within each bucket.
    for (unsigned int bkt = 0; bkt < ob; ++bkt)
        for (int i = 0; i < 2 * HL; ++i) { g_net.W2[bkt][i] = read_le<int16_t>(p + off); off += 2; }
    for (unsigned int bkt = 0; bkt < ob; ++bkt) { g_net.b2[bkt] = read_le<int32_t>(p + off); off += 4; }

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
// layer. BIT-EXACTNESS CONTRACT: the scalar and AVX2 forms MUST return the same
// value for any inputs — integer add is associative and no intermediate overflows
// (each crelu(a)∈[0,QA=255], |w|≤32767 → |product|≤8.36M, two summed in madd
// ≤16.7M < 2^31; the int32 partials are then widened to int64 before accumulating
// so the running sum never wraps even at HL=1024). Requires HL % 16 == 0.
static int64_t dot_half_scalar(const int16_t* a, const int16_t* w) {
    int64_t sum = 0;
    for (int i = 0; i < HL; ++i) {
        int c = crelu(a[i]);            // clamp to [0,QA]
#ifdef NNUE_SCRELU
        sum += (int64_t)c * c * w[i];   // SCReLU: squared activation
#else
        sum += (int64_t)c * w[i];       // CReLU
#endif
    }
    return sum;
}

static void acc_add_scalar(int16_t* dst, const int16_t* col) {
    for (int o = 0; o < HL; ++o) dst[o] += col[o];
}

static void acc_sub_scalar(int16_t* dst, const int16_t* col) {
    for (int o = 0; o < HL; ++o) dst[o] -= col[o];
}

#if defined(KING_X86) && defined(__GNUC__)
// AVX2 forms — compiled with an AVX2 target attribute so the intrinsics are
// legal even though the rest of the TU targets the portable baseline. These are
// only *called* after init() confirms __builtin_cpu_supports("avx2").
#ifdef NNUE_SCRELU
__attribute__((target("avx2")))
static int64_t dot_half_avx2(const int16_t* a, const int16_t* w) {
    // SCReLU: Σ clamp(a,0,QA)² · w, widened to int32 per element then int64-summed.
    // Bit-exact with dot_half_scalar (same integer products; add is associative).
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi32(QA);
    __m256i acc = _mm256_setzero_si256(); // int64 x4
    for (int i = 0; i < HL; i += 8) {
        __m128i a128 = _mm_loadu_si128((const __m128i*)(a + i)); // 8 int16
        __m128i w128 = _mm_loadu_si128((const __m128i*)(w + i));
        __m256i s  = _mm256_cvtepi16_epi32(a128);               // 8 int32
        s = _mm256_min_epi32(_mm256_max_epi32(s, zero), qa);    // clamp [0,QA]
        __m256i wv  = _mm256_cvtepi16_epi32(w128);
        __m256i sw  = _mm256_mullo_epi32(s, wv);                // scr·w  (int32)
        __m256i ssw = _mm256_mullo_epi32(sw, s);                // scr²·w (int32)
        acc = _mm256_add_epi64(acc, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(ssw)));
        acc = _mm256_add_epi64(acc, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(ssw, 1)));
    }
    int64_t lanes[4];
    _mm256_storeu_si256((__m256i*)lanes, acc);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}
#else
__attribute__((target("avx2")))
static int64_t dot_half_avx2(const int16_t* a, const int16_t* w) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16((short)QA);
    __m256i acc0 = _mm256_setzero_si256(); // int64 x4
    __m256i acc1 = _mm256_setzero_si256(); // int64 x4
    for (int i = 0; i < HL; i += 16) {
        __m256i av = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i wv = _mm256_loadu_si256((const __m256i*)(w + i));
        av = _mm256_min_epi16(_mm256_max_epi16(av, zero), qa);
        __m256i prod = _mm256_madd_epi16(av, wv);
        acc0 = _mm256_add_epi64(acc0, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod)));
        acc1 = _mm256_add_epi64(acc1, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod, 1)));
    }
    __m256i acc = _mm256_add_epi64(acc0, acc1);
    int64_t lanes[4];
    _mm256_storeu_si256((__m256i*)lanes, acc);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}
#endif

__attribute__((target("avx2")))
static void acc_add_avx2(int16_t* dst, const int16_t* col) {
    for (int o = 0; o < HL; o += 16) {
        __m256i d = _mm256_loadu_si256((const __m256i*)(dst + o));
        __m256i c = _mm256_loadu_si256((const __m256i*)(col + o));
        _mm256_storeu_si256((__m256i*)(dst + o), _mm256_add_epi16(d, c));
    }
}

__attribute__((target("avx2")))
static void acc_sub_avx2(int16_t* dst, const int16_t* col) {
    for (int o = 0; o < HL; o += 16) {
        __m256i d = _mm256_loadu_si256((const __m256i*)(dst + o));
        __m256i c = _mm256_loadu_si256((const __m256i*)(col + o));
        _mm256_storeu_si256((__m256i*)(dst + o), _mm256_sub_epi16(d, c));
    }
}
#endif // KING_X86 && __GNUC__

// Dispatch pointers — default to the universally-safe scalar kernels; init()
// upgrades them to the AVX2 kernels when the CPU supports AVX2.
static int64_t (*s_dot_half)(const int16_t*, const int16_t*) = dot_half_scalar;
static void    (*s_acc_add )(int16_t*, const int16_t*)       = acc_add_scalar;
static void    (*s_acc_sub )(int16_t*, const int16_t*)       = acc_sub_scalar;

static void init_simd_dispatch() {
#if defined(KING_X86) && defined(__GNUC__)
    // Escape hatch: KING_NO_AVX2=1 forces the scalar kernels even on AVX2 CPUs
    // (used to validate the portable path; also a safety valve if AVX2 is ever
    // suspect on the target hardware).
    if (const char* e = std::getenv("KING_NO_AVX2"); e && e[0] == '1') return;
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2")) {
        s_dot_half = dot_half_avx2;
        s_acc_add  = acc_add_avx2;
        s_acc_sub  = acc_sub_avx2;
    }
#endif
}

static inline int64_t dot_half(const int16_t* a, const int16_t* w) {
    return s_dot_half(a, w);
}

// ── Output layer on an up-to-date accumulator ───────────────────────────────
int evaluate_acc(const Accumulator& acc, Color stm, int piece_count) {
    const Color nonstm = Color(!stm);
    const int bkt = ob_index(piece_count);
    // stm perspective FIRST -> W2[0..HL-1]; nonstm -> W2[HL..2*HL-1].
    int64_t sum = (int64_t)g_net.b2[bkt];
    sum += dot_half(acc.v[stm],    g_net.W2[bkt]);
    sum += dot_half(acc.v[nonstm], g_net.W2[bkt] + HL);

    // eval = floor( sum * SCALE / den ) with floor toward -inf (Python //).
    // SCReLU's squared activation carries an extra QA factor → den = QA*QA*QB.
    // Kept SCALAR on the int64 sum so the result is bit-exact vs the reference.
    int64_t num = sum * (int64_t)SCALE;
#ifdef NNUE_SCRELU
    int64_t den = (int64_t)QA * QA * QB;
#else
    int64_t den = (int64_t)QA * QB; // 16320
#endif
    int64_t q = num / den;
    if ((num % den != 0) && ((num < 0) != (den < 0))) q -= 1;
    return (int)q;
}

// ── Accumulator maintenance ─────────────────────────────────────────────────
void refresh(Accumulator& acc, const Position& pos) {
    if (!g_loaded) init(); // lazy safety net: refresh is the single choke point
    // refresh is only called on a SETTLED board, so the live king squares are
    // valid and are the correct per-perspective bucket determinants.
    const Square wking = pos.king_sq(WHITE);
    const Square bking = pos.king_sq(BLACK);
    // Start both perspectives at the bias.
    for (int c = 0; c < COLOR_NB; ++c)
        std::memcpy(acc.v[c], g_net.b1, sizeof(g_net.b1));

    // Add every piece on the board to both perspectives.
    for (int c = WHITE; c <= BLACK; ++c)
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard bb = pos.pieces((Color)c, (PieceType)pt);
            while (bb) {
                Square s = pop_lsb(bb);
                add_feature(acc, (Color)c, (PieceType)pt, s, wking, bking);
            }
        }
}

// dst[o] += col[o] over HL int16s (wraps mod 2^16 identically to the scalar +=,
// which is what the int16 accumulator contract relies on). Dispatched to the
// scalar or AVX2 kernel by init_simd_dispatch().
static inline void acc_add(int16_t* dst, const int16_t* col) { s_acc_add(dst, col); }

// dst[o] -= col[o] over HL int16s.
static inline void acc_sub(int16_t* dst, const int16_t* col) { s_acc_sub(dst, col); }

void add_feature(Accumulator& acc, Color c, PieceType t, Square s, Square wking, Square bking) {
    const int iw = feature_index(c, t, s, WHITE, wking);
    const int ib = feature_index(c, t, s, BLACK, bking);
    acc_add(acc.v[WHITE], g_net.W1t[iw]);
    acc_add(acc.v[BLACK], g_net.W1t[ib]);
}

void sub_feature(Accumulator& acc, Color c, PieceType t, Square s, Square wking, Square bking) {
    const int iw = feature_index(c, t, s, WHITE, wking);
    const int ib = feature_index(c, t, s, BLACK, bking);
    acc_sub(acc.v[WHITE], g_net.W1t[iw]);
    acc_sub(acc.v[BLACK], g_net.W1t[ib]);
}

// Rebuild a single perspective from scratch (king bucket of `persp` changed).
void refresh_perspective(Accumulator& acc, const Position& pos, Color persp) {
    if (!g_loaded) init();
    const Square king = pos.king_sq(persp);
    std::memcpy(acc.v[persp], g_net.b1, sizeof(g_net.b1));
    for (int c = WHITE; c <= BLACK; ++c)
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard bb = pos.pieces((Color)c, (PieceType)pt);
            while (bb) {
                Square s = pop_lsb(bb);
                acc_add(acc.v[persp],
                        g_net.W1t[feature_index((Color)c, (PieceType)pt, s, persp, king)]);
            }
        }
}

// ── From-scratch reference evaluation ────────────────────────────────────────
int evaluate_from_scratch(const Position& pos) {
    Accumulator acc;
    refresh(acc, pos);
    return evaluate_acc(acc, pos.side_to_move(), popcount(pos.occupied()));
}

} // namespace nnue
} // namespace king

#endif // EVAL_NNUE
