cmake_minimum_required(VERSION 3.16)
project(simple_wgsl C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

option(WGSL_BUILD_TESTS "Build tests" ON)
option(WGSL_USE_ASAN "Enable Address Sanitizer" OFF)

add_library(wgsl_compiler STATIC
    wgsl_parser.c
    wgsl_resolve.c
    wgsl_lower.c
    wgsl_raise.c
)

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
        GIT_TAG        v1.3.250.0
    )
    FetchContent_MakeAvailable(spirv_headers)
    target_include_directories(wgsl_compiler PUBLIC ${spirv_headers_SOURCE_DIR}/include)
endif()

add_executable(wgsl_reflect_harness wgsl_reflect_harness.c)
target_link_libraries(wgsl_reflect_harness wgsl_compiler)

if(WGSL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(WGSL_USE_ASAN)
    target_compile_options(wgsl_compiler PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(wgsl_compiler PRIVATE -fsanitize=address)
endif()
