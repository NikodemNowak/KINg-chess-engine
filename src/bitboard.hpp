#pragma once
#include <bit>
#include <string>
#include "types.hpp"

namespace king {

// ── File / Rank masks ──────────────────────────────────────────────────────
constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
constexpr Bitboard FILE_B_BB = FILE_A_BB << 1;
constexpr Bitboard FILE_C_BB = FILE_A_BB << 2;
constexpr Bitboard FILE_D_BB = FILE_A_BB << 3;
constexpr Bitboard FILE_E_BB = FILE_A_BB << 4;
constexpr Bitboard FILE_F_BB = FILE_A_BB << 5;
constexpr Bitboard FILE_G_BB = FILE_A_BB << 6;
constexpr Bitboard FILE_H_BB = FILE_A_BB << 7;

constexpr Bitboard RANK_1_BB = 0xFFULL;
constexpr Bitboard RANK_2_BB = RANK_1_BB << 8;
constexpr Bitboard RANK_3_BB = RANK_1_BB << 16;
constexpr Bitboard RANK_4_BB = RANK_1_BB << 24;
constexpr Bitboard RANK_5_BB = RANK_1_BB << 32;
constexpr Bitboard RANK_6_BB = RANK_1_BB << 40;
constexpr Bitboard RANK_7_BB = RANK_1_BB << 48;
constexpr Bitboard RANK_8_BB = RANK_1_BB << 56;

constexpr inline Bitboard file_bb(File f) { return FILE_A_BB << f; }
constexpr inline Bitboard rank_bb(Rank r) { return RANK_1_BB << (r * 8); }

// ── Direction ─────────────────────────────────────────────────────────────
enum Direction {
    NORTH      =  8,
    EAST       =  1,
    SOUTH      = -8,
    WEST       = -1,
    NORTH_EAST =  9,
    NORTH_WEST =  7,
    SOUTH_EAST = -7,
    SOUTH_WEST = -9
};

// ── Primitives ────────────────────────────────────────────────────────────
constexpr inline Bitboard square_bb(Square s) { return Bitboard(1) << s; }

constexpr inline int popcount(Bitboard b) { return std::popcount(b); }

constexpr inline Square lsb(Bitboard b) { return Square(std::countr_zero(b)); }

constexpr inline Square msb(Bitboard b) { return Square(63 ^ std::countl_zero(b)); }

constexpr inline Square pop_lsb(Bitboard& b) {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

constexpr inline bool more_than_one(Bitboard b) { return (b & (b - 1)) != 0; }

// ── Shift with wrap protection ────────────────────────────────────────────
template<Direction D>
constexpr Bitboard shift(Bitboard b);

template<> constexpr Bitboard shift<NORTH>     (Bitboard b) { return b << 8; }
template<> constexpr Bitboard shift<SOUTH>     (Bitboard b) { return b >> 8; }
template<> constexpr Bitboard shift<EAST>      (Bitboard b) { return (b & ~FILE_H_BB) << 1; }
template<> constexpr Bitboard shift<WEST>      (Bitboard b) { return (b & ~FILE_A_BB) >> 1; }
template<> constexpr Bitboard shift<NORTH_EAST>(Bitboard b) { return (b & ~FILE_H_BB) << 9; }
template<> constexpr Bitboard shift<NORTH_WEST>(Bitboard b) { return (b & ~FILE_A_BB) << 7; }
template<> constexpr Bitboard shift<SOUTH_EAST>(Bitboard b) { return (b & ~FILE_H_BB) >> 7; }
template<> constexpr Bitboard shift<SOUTH_WEST>(Bitboard b) { return (b & ~FILE_A_BB) >> 9; }

// ── Table accessors (init() must be called first) ─────────────────────────
Bitboard between_bb(Square a, Square b);
Bitboard line_bb(Square a, Square b);

// ── Debug pretty-printer ──────────────────────────────────────────────────
std::string pretty(Bitboard b);

// ── Init ──────────────────────────────────────────────────────────────────
namespace bitboard { void init(); }

} // namespace king
