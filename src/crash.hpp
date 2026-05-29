#pragma once

namespace king {
namespace crash {

// Store a best-effort "bestmove <uci>\n" to emit on a fatal signal.
void arm_fallback(const char* uci_move);

// Install handlers for SIGSEGV, SIGABRT, SIGFPE.
void install_handlers();

} // namespace crash
} // namespace king
