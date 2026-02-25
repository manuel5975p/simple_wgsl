find_package(Vulkan REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED libzstd)

option(CUVK_DEBUG_LOG "Enable cuvk debug logging to stderr" OFF)

set(CUVK_SOURCES
    cuvk_init.c
    cuvk_memory.c
    cuvk_module.c
    cuvk_launch.c
    cuvk_stream.c
    cuvk_stubs.c
    cuvk_fatbin.c
)

# Static library
add_library(cuvk_runtime_static STATIC ${CUVK_SOURCES})
target_link_libraries(cuvk_runtime_static PUBLIC wgsl_compiler Vulkan::Vulkan ${ZSTD_LIBRARIES})
target_include_directories(cuvk_runtime_static PUBLIC ${CMAKE_SOURCE_DIR} ${ZSTD_INCLUDE_DIRS})
set_target_properties(cuvk_runtime_static PROPERTIES OUTPUT_NAME cuvk_runtime)

if(CUVK_DEBUG_LOG)
    target_compile_definitions(cuvk_runtime_static PRIVATE CUVK_DEBUG_LOG=1)
endif()

# Shared library (acts as libcuda.so.1 for nvcc-compiled binaries)
add_library(cuvk_runtime_shared SHARED ${CUVK_SOURCES})
target_link_libraries(cuvk_runtime_shared PUBLIC wgsl_compiler Vulkan::Vulkan ${ZSTD_LIBRARIES})
target_include_directories(cuvk_runtime_shared PUBLIC ${CMAKE_SOURCE_DIR} ${ZSTD_INCLUDE_DIRS})
set_target_properties(cuvk_runtime_shared PROPERTIES
    OUTPUT_NAME "cuda"
    SOVERSION 1
)

if(CUVK_DEBUG_LOG)
    target_compile_definitions(cuvk_runtime_shared PRIVATE CUVK_DEBUG_LOG=1)
endif()
