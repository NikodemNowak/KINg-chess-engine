#pragma once
#include <iostream>

namespace king {
namespace uci {

// Testable core: reads from `in`, writes to `out`.
// Initializes all engine tables (idempotent) before entering the command loop.
void run(std::istream& in, std::ostream& out);

// Convenience wrapper: run(std::cin, std::cout)
void loop();

} // namespace uci
} // namespace king
