cmake_minimum_required(VERSION 3.16)
project(ephs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EPHS_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

add_library(ephs
  src/backend.cpp
  src/dump.cpp
  src/gemv_avx2.cpp
  src/gemv_scalar.cpp
  src/sampler.cpp
  src/shadow.cpp
)
target_include_directories(ephs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(ephs PRIVATE -Wall -Wextra -Werror -Wpedantic -ffp-contract=off)
set_source_files_properties(src/gemv_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=off")

if(EPHS_SANITIZERS)
  target_compile_options(ephs PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(ephs PUBLIC -fsanitize=address,undefined)
endif()

add_executable(ephs_tests tests/test_main.cpp)
target_link_libraries(ephs_tests PRIVATE ephs)
target_compile_options(ephs_tests PRIVATE -Wall -Wextra -Werror -Wpedantic)

add_executable(ephs_bench bench/bench_head.cpp)
target_link_libraries(ephs_bench PRIVATE ephs)
target_compile_options(ephs_bench PRIVATE -Wall -Wextra -Werror -Wpedantic -mclflushopt)

enable_testing()
add_test(NAME ephs_tests COMMAND ephs_tests)
