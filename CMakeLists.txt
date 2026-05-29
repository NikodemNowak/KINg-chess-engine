cmake_minimum_required(VERSION 3.20)
project(KINg CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx2 -mpopcnt -DNDEBUG")

# ── Evaluation selector ───────────────────────────────────────────────────────
# -DEVAL=HCE (default): handcrafted PeSTO/structural eval ("No Deep Learning").
# -DEVAL=NNUE          : embedded NNUE net + incremental accumulator.
#
# Default is HCE: the first NNUE net (3.5M positions, 256 HL, 45 epochs) tested
# ~133 Elo WEAKER than the HCE at 8+0.08 (NNUE-vs-HCE 16-89-95 over 200 games),
# so shipping NNUE as default would regress strength. The NNUE inference,
# incremental accumulator, and net embedding are all in place and pass the
# bit-exact gate; once the net is retrained with more data/epochs, flip the
# default back to NNUE (or pass -DEVAL=NNUE).
set(EVAL "HCE" CACHE STRING "Evaluation: HCE (default) or NNUE")
set_property(CACHE EVAL PROPERTY STRINGS NNUE HCE)
message(STATUS "KINg eval = ${EVAL}")

include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)

file(GLOB_RECURSE ENGINE_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE TEST_SRC   CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)

set(EXTRA_SRC "")

if(EVAL STREQUAL "NNUE")
  add_compile_definitions(EVAL_NNUE)

  # Embed the net into the binary: generate a C++ byte-array source from
  # nets/king_nnue.bin at build time (no external file needed at runtime).
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(NET_BIN ${CMAKE_SOURCE_DIR}/nets/king_nnue.bin)
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

add_executable(engine ${ENGINE_SRC} ${EXTRA_SRC})

set(ENGINE_SRC_NO_MAIN ${ENGINE_SRC})
list(FILTER ENGINE_SRC_NO_MAIN EXCLUDE REGEX "main\\.cpp$")
add_executable(unit_tests ${TEST_SRC} ${ENGINE_SRC_NO_MAIN} ${EXTRA_SRC})

if(EVAL STREQUAL "NNUE")
  # Absolute path to the sample file so the bit-exact gate test finds it
  # regardless of the working directory the test binary runs from.
  target_compile_definitions(unit_tests PRIVATE
    NNUE_SAMPLES_PATH="${CMAKE_SOURCE_DIR}/trainer/nnue_samples.txt")
endif()

enable_testing()
add_test(NAME unit COMMAND unit_tests)
