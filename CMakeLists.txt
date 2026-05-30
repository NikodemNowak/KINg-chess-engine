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
# Default is NNUE: the HL=512 net retrained on ~30M self-play positions beats the
# Texel-tuned HCE by +193 Elo at 8+0.08 (NNUE-vs-HCE 225-23-152 over 400 games,
# 95% CI [+167, +222]). (The first net — 3.5M positions, HL=256 — had lost ~133
# Elo, which is why HCE shipped previously.) Pass -DEVAL=HCE to build the
# "No Deep Learning" entry.
set(EVAL "NNUE" CACHE STRING "Evaluation: NNUE (default) or HCE")
set_property(CACHE EVAL PROPERTY STRINGS NNUE HCE)
message(STATUS "KINg eval = ${EVAL}")

# ── NNUE hidden-layer size ────────────────────────────────────────────────────
# -DNNUE_HL=512 (default, matches committed king_nnue.bin — the 30M-position net).
# Override with -DNNUE_HL=<n> to build against a different-width net; the value
# is passed through as -DNNUE_HL=<n> so nnue.hpp picks it up at compile time.
set(NNUE_HL "512" CACHE STRING "NNUE accumulator hidden-layer size per perspective (default 512)")
message(STATUS "KINg NNUE_HL = ${NNUE_HL}")

# ── NNUE net binary (override for validation builds) ─────────────────────────
# Default: the committed HL=256 production net.
# Override with -DNNUE_NET=/abs/path/to/other.bin for temp-net validation.
set(NNUE_NET "" CACHE FILEPATH "Override NNUE net binary (default: nets/king_nnue.bin)")

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

  # Embed the net into the binary: generate a C++ byte-array source from
  # nets/king_nnue.bin (or the override) at build time.
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  if(NNUE_NET STREQUAL "")
    set(NET_BIN ${CMAKE_SOURCE_DIR}/nets/king_nnue.bin)
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

set(ENGINE_SRC_NO_MAIN ${ENGINE_SRC})
list(FILTER ENGINE_SRC_NO_MAIN EXCLUDE REGEX "main\\.cpp$")
add_executable(unit_tests ${TEST_SRC} ${ENGINE_SRC_NO_MAIN} ${EXTRA_SRC} ${FATHOM_SRC})

if(EVAL STREQUAL "NNUE")
  # Absolute path to the sample file so the bit-exact gate test finds it
  # regardless of the working directory the test binary runs from.
  # Override with -DNNUE_SAMPLES=<path> when validating a temp net.
  set(NNUE_SAMPLES "" CACHE FILEPATH "Override NNUE samples file for bit-exact gate test")
  if(NNUE_SAMPLES STREQUAL "")
    set(_samples_path "${CMAKE_SOURCE_DIR}/trainer/nnue_samples.txt")
  else()
    set(_samples_path "${NNUE_SAMPLES}")
  endif()
  target_compile_definitions(unit_tests PRIVATE
    NNUE_SAMPLES_PATH="${_samples_path}")
endif()

enable_testing()
add_test(NAME unit COMMAND unit_tests)
