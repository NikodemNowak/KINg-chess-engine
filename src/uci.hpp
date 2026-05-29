#pragma once
#include <iostream>

namespace king {

// Returns the effective number of CPU cores available to this process.
// On Linux: MIN of sched_getaffinity count, cgroup CPU quota (v2/v1), and
// hardware_concurrency(), clamped to ≥1. On Windows/other: hardware_concurrency()
// clamped to ≥1. Used to default the Threads UCI option.
int available_cores();

namespace uci {

// Testable core: reads from `in`, writes to `out`.
// Initializes all engine tables (idempotent) before entering the command loop.
void run(std::istream& in, std::ostream& out);

// Convenience wrapper: run(std::cin, std::cout)
void loop();

} // namespace uci
} // namespace king
