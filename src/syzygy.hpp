#pragma once
#include <string>
#include "position.hpp"

namespace king {
namespace syzygy {

// Initialize tablebases from the given directory path.
// Empty path or path without valid TB files → disables TBs (tb_free).
void init(const std::string& path);

// Returns true if at least one tablebase piece-count is loaded (TB_LARGEST > 0).
bool enabled();

// Returns TB_LARGEST: max piece count (both sides combined) for which we have TBs.
unsigned largest();

// Probe WDL for the position during search.
// Returns TB_WIN / TB_CURSED_WIN / TB_DRAW / TB_BLESSED_LOSS / TB_LOSS
// or TB_RESULT_FAILED (0xFFFFFFFF) if position is not probe-eligible:
//   - TBs not loaded
//   - castling rights != 0
//   - piece count > TB_LARGEST
// NOTE: tb_probe_wdl already requires castling==0 and rule50==0 internally;
//       it returns FAILED for rule50 != 0, which we propagate.
unsigned probe_wdl(const Position& pos);

// Probe for root use (DTZ), returns the packed TB_RESULT.
// Caller decodes with TB_GET_WDL / TB_GET_FROM / TB_GET_TO / TB_GET_PROMOTES.
// Returns TB_RESULT_FAILED if not eligible.
unsigned probe_root(const Position& pos);

} // namespace syzygy
} // namespace king
