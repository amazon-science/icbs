cmake_minimum_required(VERSION 3.15)

project(sa-card)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -mfma -ftree-vectorize -mavx2 -ffp-contract=fast)
endif()

enable_testing()

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.1 # or a later release
)
FetchContent_MakeAvailable(Catch2)

find_package(OpenMP REQUIRED)

include_directories(include)
add_subdirectory(src)
add_subdirectory(include)
