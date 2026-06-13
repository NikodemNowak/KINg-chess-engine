cmake_minimum_required(VERSION 3.20)
project(KINg CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# ── Portable baseline (crash-safety: crash == lost game) ──────────────────────
# Compile for the generic x86-64 baseline (SSE2 only — present on EVERY x86-64
# CPU) so the binary can NEVER hit an illegal instruction on hardware without
# AVX2/POPCNT/BMI.  AVX2 acceleration for the NNUE kernels is selected at RUNTIME
# via __builtin_cpu_supports (see src/nnue.cpp init_simd_dispatch); popcount uses
# std::popcount / __builtin_popcountll which are portable without -mpopcnt; and
# Fathom (third_party/fathom) falls back to a builtin popcount when __SSE4_2__ is
# undefined.  Do NOT add -march=native / -mavx2 here — it reintroduces SIGILL.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=x86-64 -mtune=generic -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE   "-O3 -march=x86-64 -mtune=generic -DNDEBUG")

# ── Evaluation selector ───────────────────────────────────────────────────────
# -DEVAL=NNUE (default): embedded NNUE net + incremental accumulator ("Open").
# -DEVAL=HCE           : handcrafted PeSTO/structural eval ("No Deep Learning").
#
# NNUE (the shipped build) is far stronger than the handcrafted eval; pass
# -DEVAL=HCE only to build the "No Deep Learning" entry.
set(EVAL "NNUE" CACHE STRING "Evaluation: NNUE (default) or HCE")
set_property(CACHE EVAL PROPERTY STRINGS NNUE HCE)
message(STATUS "KINg eval = ${EVAL}")

# ── NNUE hidden-layer size ────────────────────────────────────────────────────
# -DNNUE_HL=512 (default, matches the committed production net). Override with
# -DNNUE_HL=<n> to build against a different-width net (passed through to nnue.hpp).
set(NNUE_HL "512" CACHE STRING "NNUE accumulator hidden-layer size per perspective (default 512)")
message(STATUS "KINg NNUE_HL = ${NNUE_HL}")

# ── NNUE output buckets ───────────────────────────────────────────────────────
# -DNNUE_OB=1 (default): legacy single output layer (KNUE format, byte-identical
# to the original net). -DNNUE_OB=<n> selects a per-piece-count output layer and
# requires a KNU2-format net trained with `--buckets <n>` (bucket =
# clamp((piece_count-2)/4, 0, n-1)). 8 is the usual choice.
set(NNUE_OB "8" CACHE STRING "NNUE output buckets by piece count (default 8 = production net)")
message(STATUS "KINg NNUE_OB = ${NNUE_OB}")

# ── NNUE king-square input buckets (HalfKP-style) ─────────────────────────────
# -DNNUE_KB=1 (default): plain 768 inputs (king-independent). -DNNUE_KB=<n>
# makes each perspective's features depend on its own king square (input space =
# n*768) and requires a KNU3-format net trained with `--kbuckets <n>`. King moves
# trigger a full accumulator refresh. Supported: 1,4,8,16,32,64.
set(NNUE_KB "1" CACHE STRING "NNUE king-square input buckets (default 1 = none)")
message(STATUS "KINg NNUE_KB = ${NNUE_KB}")

# ── NNUE activation ───────────────────────────────────────────────────────────
# -DNNUE_SCRELU=ON (default) uses the squared clipped-ReLU activation, matching the
# committed production net. Pass -DNNUE_SCRELU=OFF only with a plain-CReLU net.
set(NNUE_SCRELU ON CACHE BOOL "Use SCReLU (squared) NNUE activation")
message(STATUS "KINg NNUE_SCRELU = ${NNUE_SCRELU}")

# ── NNUE net binary (override for validation builds) ─────────────────────────
# Default (empty): embed the committed production net (nets/king_int8_174m.bin).
# Override with -DNNUE_NET=/abs/path/to/other.bin for temp-net validation.
set(NNUE_NET "" CACHE FILEPATH "Override NNUE net binary (default: nets/king_int8_174m.bin)")

include_directories(${CMAKE_SOURCE_DIR}/src
                    ${CMAKE_SOURCE_DIR}/third_party
                    ${CMAKE_SOURCE_DIR}/third_party/fathom)

# ── Fathom (Syzygy tablebase probing, self-contained C library) ───────────────
set(FATHOM_SRC ${CMAKE_SOURCE_DIR}/third_party/fathom/tbprobe.c)
# Compile as C (not C++) — tbprobe.c uses C11 atomics when compiled as C.
set_source_files_properties(${FATHOM_SRC} PROPERTIES LANGUAGE C)

file(GLOB_RECURSE ENGINE_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE TEST_SRC   CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)

set(EXTRA_SRC "")

if(EVAL STREQUAL "NNUE")
  add_compile_definitions(EVAL_NNUE)
  add_compile_definitions(NNUE_HL=${NNUE_HL})
  add_compile_definitions(NNUE_OB=${NNUE_OB})
  add_compile_definitions(NNUE_KB=${NNUE_KB})
  if(NNUE_SCRELU)
    add_compile_definitions(NNUE_SCRELU)
  endif()

  # Embed the net into the binary: generate a C++ byte-array source from the
  # committed net (or the -DNNUE_NET override) at build time.
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  if(NNUE_NET STREQUAL "")
    set(NET_BIN ${CMAKE_SOURCE_DIR}/nets/king_int8_174m.bin)
  else()
    set(NET_BIN ${NNUE_NET})
  endif()
  message(STATUS "KINg NNUE net = ${NET_BIN}")
  set(NET_GEN ${CMAKE_BINARY_DIR}/nnue_net_data.cpp)
  add_custom_command(
    OUTPUT  ${NET_GEN}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/embed_net.py ${NET_BIN} ${NET_GEN}
    DEPENDS ${NET_BIN} ${CMAKE_SOURCE_DIR}/tools/embed_net.py
    COMMENT "Embedding NNUE net ${NET_BIN}"
    VERBATIM)
  list(APPEND EXTRA_SRC ${NET_GEN})
else()
  # HCE build: drop the NNUE TU entirely (it compiles to nothing without
  # EVAL_NNUE, but excluding it keeps the build clean and net-free).
  list(FILTER ENGINE_SRC EXCLUDE REGEX "nnue\\.cpp$")
endif()

add_executable(engine ${ENGINE_SRC} ${EXTRA_SRC} ${FATHOM_SRC})

# ── LTO / Interprocedural optimization (default-ON for the shipped engine) ─────
# Cross-TU inlining (see.cpp ↔ attacks.cpp ↔ bitboard/movegen) is worth a few %
# NPS. Default-on so EVERY build path (Dockerfile, ship scripts, dev) gets it
# without remembering -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON. AVX2 stays runtime-
# dispatched, so no SIGILL risk. unit_tests is left non-IPO for fast test builds.
include(CheckIPOSupported)
check_ipo_supported(RESULT _king_ipo_ok OUTPUT _king_ipo_msg)
if(_king_ipo_ok AND CMAKE_BUILD_TYPE STREQUAL "Release")
  set_property(TARGET engine PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  message(STATUS "KINg LTO/IPO = ON (engine)")
else()
  message(STATUS "KINg LTO/IPO = OFF (${_king_ipo_msg})")
endif()

set(ENGINE_SRC_NO_MAIN ${ENGINE_SRC})
list(FILTER ENGINE_SRC_NO_MAIN EXCLUDE REGEX "main\\.cpp$")
add_executable(unit_tests ${TEST_SRC} ${ENGINE_SRC_NO_MAIN} ${EXTRA_SRC} ${FATHOM_SRC})

if(EVAL STREQUAL "NNUE")
  # Absolute path to the sample file so the bit-exact gate test finds it
  # regardless of the working directory the test binary runs from.
  # Override with -DNNUE_SAMPLES=<path> when validating a temp net.
  set(NNUE_SAMPLES "" CACHE FILEPATH "Override NNUE samples file for bit-exact gate test")
  if(NNUE_SAMPLES STREQUAL "")
    set(_samples_path "${CMAKE_SOURCE_DIR}/trainer/s_int8_174m.txt")
  else()
    set(_samples_path "${NNUE_SAMPLES}")
  endif()
  target_compile_definitions(unit_tests PRIVATE
    NNUE_SAMPLES_PATH="${_samples_path}")
endif()

enable_testing()
add_test(NAME unit COMMAND unit_tests)
