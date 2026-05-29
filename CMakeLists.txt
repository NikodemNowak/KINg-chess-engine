cmake_minimum_required(VERSION 3.20)
project(KINg CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx2 -mpopcnt -DNDEBUG")

include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)

file(GLOB_RECURSE ENGINE_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE TEST_SRC   CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)

add_executable(engine ${ENGINE_SRC})

set(ENGINE_SRC_NO_MAIN ${ENGINE_SRC})
list(FILTER ENGINE_SRC_NO_MAIN EXCLUDE REGEX "main\\.cpp$")
add_executable(unit_tests ${TEST_SRC} ${ENGINE_SRC_NO_MAIN})

enable_testing()
add_test(NAME unit COMMAND unit_tests)
