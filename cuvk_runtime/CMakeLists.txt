option(CUVK_DEBUG_LOG "Enable cuvk debug logging to stderr" OFF)
option(CUVK_NVJITLINK "Enable nvJitLink for LTO-IR compilation (requires libnvJitLink.so)" OFF)

include(FetchContent)
find_package(PkgConfig QUIET)

# Zstd: prefer system package via PkgConfig, fall back to FetchContent
if(PkgConfig_FOUND)
    pkg_check_modules(ZSTD IMPORTED_TARGET libzstd)
endif()
if(ZSTD_FOUND)
    add_library(zstd_dep ALIAS PkgConfig::ZSTD)
else()
    message(STATUS "System zstd not found, fetching via FetchContent")
    FetchContent_Declare(
        zstd
        URL https://github.com/facebook/zstd/archive/refs/tags/v1.5.7.tar.gz
        SOURCE_SUBDIR build/cmake
    )
    set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(zstd)
    add_library(zstd_dep ALIAS libzstd_static)
endif()

# LZ4: prefer system package via PkgConfig, fall back to FetchContent
if(PkgConfig_FOUND)
    pkg_check_modules(LZ4 IMPORTED_TARGET liblz4)
endif()
if(LZ4_FOUND)
    add_library(lz4_dep ALIAS PkgConfig::LZ4)
else()
    message(STATUS "System LZ4 not found, fetching via FetchContent")
    FetchContent_Declare(
        lz4
        GIT_REPOSITORY https://github.com/lz4/lz4.git
        GIT_TAG        v1.10.0
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  build/cmake
    )
    set(LZ4_BUILD_CLI OFF CACHE BOOL "" FORCE)
    set(LZ4_BUILD_LEGACY_LZ4C OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(lz4)
    add_library(lz4_dep ALIAS lz4_static)
endif()

if(WIN32)
    if(NOT VULKAN_LIBRARY)
        message(FATAL_ERROR "Set -DVULKAN_LIBRARY=<path to libvulkan-1.a or vulkan-1.lib>")
    endif()
    set(_VULKAN_LIB ${VULKAN_LIBRARY})
    set(_VULKAN_INC ${VULKAN_INCLUDE_DIR})
else()
    find_package(Vulkan REQUIRED)
    set(_VULKAN_LIB Vulkan::Vulkan)
    set(_VULKAN_INC "")
endif()

set(CUVK_SOURCES
    cuvk_init.c
    cuvk_memory.c
    cuvk_module.c
    cuvk_launch.c
    cuvk_stream.c
    cuvk_graph.c
    cuvk_stubs.c
    cuvk_fatbin.c
    cuvk_compat.c
)

if(CUVK_NVJITLINK)
    list(APPEND CUVK_SOURCES cuvk_jitlink.c)
endif()

# Helper: apply common settings to a cuvk target
function(cuvk_target_setup tgt)
    target_link_libraries(${tgt} PUBLIC wgsl_compiler ${_VULKAN_LIB} zstd_dep lz4_dep)
    target_include_directories(${tgt} PUBLIC ${CMAKE_SOURCE_DIR} ${_VULKAN_INC})
    if(CUVK_DEBUG_LOG)
        target_compile_definitions(${tgt} PRIVATE CUVK_DEBUG_LOG=1)
    endif()
    if(CUVK_NVJITLINK)
        target_compile_definitions(${tgt} PRIVATE CUVK_NVJITLINK=1)
        target_link_libraries(${tgt} PUBLIC dl)
    endif()
    if(WIN32)
        target_link_libraries(${tgt} PUBLIC -lpthread)
    endif()
endfunction()

# Static library
add_library(cuvk_runtime_static STATIC ${CUVK_SOURCES})
cuvk_target_setup(cuvk_runtime_static)
set_target_properties(cuvk_runtime_static PROPERTIES OUTPUT_NAME cuvk_runtime)

# Shared library: nvcuda.dll / libcuda.so.1
add_library(cuvk_runtime_shared SHARED ${CUVK_SOURCES})
cuvk_target_setup(cuvk_runtime_shared)
if(WIN32)
    set_target_properties(cuvk_runtime_shared PROPERTIES
        OUTPUT_NAME "nvcuda"
        PREFIX ""
    )
else()
    set_target_properties(cuvk_runtime_shared PROPERTIES
        OUTPUT_NAME "cuda"
        SOVERSION 1
    )
endif()

# Shared library: cudart64_*.dll / libcudart.so.1
add_library(cuvk_cudart_shared SHARED cuvk_cudart.c ${CUVK_SOURCES})
cuvk_target_setup(cuvk_cudart_shared)
target_include_directories(cuvk_cudart_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(WIN32)
    set_target_properties(cuvk_cudart_shared PROPERTIES
        OUTPUT_NAME "cudart64_12"
        PREFIX ""
    )
else()
    set_target_properties(cuvk_cudart_shared PROPERTIES
        OUTPUT_NAME "cudart"
        SOVERSION 1
    )
endif()

# Shared library: cublas64_*.dll / libcublas.so.13
add_library(cuvk_cublas_shared SHARED cuvk_cublas.c ${CUVK_SOURCES})
cuvk_target_setup(cuvk_cublas_shared)
target_include_directories(cuvk_cublas_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cuvk_cublas_shared PUBLIC m)
if(WIN32)
    set_target_properties(cuvk_cublas_shared PROPERTIES
        OUTPUT_NAME "cublas64_12"
        PREFIX ""
    )
else()
    set_target_properties(cuvk_cublas_shared PROPERTIES
        OUTPUT_NAME "cublas"
        SOVERSION 13
        LINK_OPTIONS "$<$<NOT:$<PLATFORM_ID:Darwin>>:-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cublas.map>"
    )
endif()

add_subdirectory(fft)
