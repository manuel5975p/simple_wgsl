cmake_minimum_required(VERSION 3.16)
project(simple_wgsl C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

option(WGSL_BUILD_TESTS "Build tests" ON)
option(WGSL_BUILD_EXPRESSION_TESTS "Build expression golden-file tests (requires expressions/ data)" OFF)
option(WGSL_USE_ASAN "Enable Address Sanitizer" OFF)
option(WGSL_USE_UBSAN "Enable Undefined Behavior Sanitizer" OFF)
option(WGSL_USE_LSAN "Enable Leak Sanitizer (standalone, not with ASan)" OFF)
option(WGSL_BUILD_EGL_TESTS "Build EGL/OpenGL compute shader validation tests" OFF)
option(WGSL_BUILD_METAL_EXAMPLES "Build Metal/MSL example programs (macOS only)" OFF)
option(WGSL_BUILD_FUZZ "Build libFuzzer fuzz targets" OFF)
option(WGSL_COVERAGE "Enable code coverage with llvm-cov" OFF)

set(SANITIZER_FLAGS "")
set(SANITIZER_LINK_FLAGS "")

if(WGSL_COVERAGE)
    if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "WGSL_COVERAGE requires Clang (set CMAKE_C_COMPILER=clang CMAKE_CXX_COMPILER=clang++)")
    endif()
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
endif()

if(WGSL_USE_ASAN)
    string(APPEND SANITIZER_FLAGS " -fsanitize=address")
    string(APPEND SANITIZER_LINK_FLAGS " -fsanitize=address")
endif()

if(WGSL_USE_UBSAN)
    string(APPEND SANITIZER_FLAGS " -fsanitize=undefined")
    string(APPEND SANITIZER_LINK_FLAGS " -fsanitize=undefined")
endif()

if(WGSL_USE_LSAN AND NOT WGSL_USE_ASAN)
    string(APPEND SANITIZER_FLAGS " -fsanitize=leak")
    string(APPEND SANITIZER_LINK_FLAGS " -fsanitize=leak")
endif()

if(SANITIZER_FLAGS)
    string(APPEND SANITIZER_FLAGS " -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}${SANITIZER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}${SANITIZER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}${SANITIZER_LINK_FLAGS}")
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
        -Wall
        -Wextra
        -Wno-missing-field-initializers
        -Wno-sign-compare
    )
endif()

add_library(wgsl_compiler STATIC
    wgsl_parser.c
    wgsl_resolve.c
    wgsl_lower.c
    wgsl_raise.c
    glsl_parser.c
    ssir.c
    ssir_to_spirv.c
    ssir_to_wgsl.c
    ssir_to_glsl.c
    ssir_to_msl.c
    ssir_to_hlsl.c
    spirv_to_ssir.c
    msl_parser.c
    ptx_parser.c
    ptx_lower.c
)

set_target_properties(wgsl_compiler PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(wgsl_compiler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_path(SPIRV_INCLUDE_DIR spirv/unified1/spirv.h
    HINTS
        $ENV{VULKAN_SDK}/include
        /usr/include
)
if(SPIRV_INCLUDE_DIR)
    target_include_directories(wgsl_compiler PUBLIC ${SPIRV_INCLUDE_DIR})
else()
    FetchContent_Declare(
        spirv_headers
        GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
        GIT_TAG        vulkan-sdk-1.4.341
        GIT_SHALLOW    True
    )
    FetchContent_MakeAvailable(spirv_headers)
    target_include_directories(wgsl_compiler PUBLIC ${spirv_headers_SOURCE_DIR}/include)
endif()

add_executable(example example/example.c)
target_link_libraries(example wgsl_compiler)

if(WGSL_BUILD_METAL_EXAMPLES AND APPLE)
    enable_language(OBJC)

    add_executable(metal_compute example/metal_compute.m)
    target_link_libraries(metal_compute wgsl_compiler "-framework Metal" "-framework Foundation")

    add_executable(metal_triangle example/metal_triangle.m)
    target_link_libraries(metal_triangle wgsl_compiler "-framework Metal" "-framework Foundation")

    add_executable(msl_roundtrip example/msl_roundtrip.m)
    target_link_libraries(msl_roundtrip wgsl_compiler "-framework Metal" "-framework Foundation")
endif()

if(WGSL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(WGSL_BUILD_FUZZ)
    add_subdirectory(fuzz)
endif()

if(WIN32)
    # Cross-compilation: always build cuvk_runtime if Vulkan lib provided
    if(VULKAN_LIBRARY)
        add_subdirectory(cuvk_runtime)

        # Windows test executables
        add_executable(win_test tests/win_test.c)
        target_link_libraries(win_test cuvk_runtime_shared m)
        target_include_directories(win_test PRIVATE ${CMAKE_SOURCE_DIR})

        # E2E PTX-based tests (cross-compiled for Wine)
        set(WIN_E2E_TESTS test_vector_add test_saxpy test_scale test_dot_product)
        foreach(test ${WIN_E2E_TESTS})
            add_executable(${test} tests/e2e/${test}.c)
            target_link_libraries(${test} cuvk_runtime_shared m)
            target_include_directories(${test} PRIVATE ${CMAKE_SOURCE_DIR})
        endforeach()
    endif()
else()
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        add_subdirectory(cuvk_runtime)

        add_executable(bench_fft_stockham bench/bench_fft_stockham.c)
        target_link_libraries(bench_fft_stockham fft_stockham_gen wgsl_compiler Vulkan::Vulkan m)

        add_executable(test_fft_multistage bench/test_fft_multistage.c)
        target_link_libraries(test_fft_multistage fft_stockham_gen wgsl_compiler Vulkan::Vulkan m)

        add_executable(bench_fft_multistage bench/bench_fft_multistage.c)
        target_link_libraries(bench_fft_multistage fft_stockham_gen wgsl_compiler Vulkan::Vulkan m)

        add_executable(bench_fft_optimal bench/bench_fft_optimal.c)
        target_link_libraries(bench_fft_optimal fft_optimal_gen wgsl_compiler Vulkan::Vulkan m)

        add_executable(bench_fft_fused bench/bench_fft_fused.c)
        target_link_libraries(bench_fft_fused fft_fused_gen wgsl_compiler Vulkan::Vulkan m)
    endif()
endif()
