option(CUVK_DEBUG_LOG "Enable cuvk debug logging to stderr" OFF)

if(WIN32)
    # Cross-compilation: Vulkan and zstd must be provided via cache vars
    #   -DVULKAN_INCLUDE_DIR=...  -DVULKAN_LIBRARY=...
    #   -DZSTD_INCLUDE_DIR=...    -DZSTD_LIBRARY=...
    if(NOT VULKAN_LIBRARY)
        message(FATAL_ERROR "Set -DVULKAN_LIBRARY=<path to libvulkan-1.a or vulkan-1.lib>")
    endif()
    if(NOT DZSTD_LIBRARY AND NOT ZSTD_LIBRARY)
        message(FATAL_ERROR "Set -DZSTD_LIBRARY=<path to libzstd.a>")
    endif()
    set(_VULKAN_LIB ${VULKAN_LIBRARY})
    set(_VULKAN_INC ${VULKAN_INCLUDE_DIR})
    set(_ZSTD_LIB ${ZSTD_LIBRARY})
    set(_ZSTD_INC ${ZSTD_INCLUDE_DIR})
else()
    find_package(Vulkan REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ZSTD REQUIRED libzstd)
    set(_VULKAN_LIB Vulkan::Vulkan)
    set(_VULKAN_INC "")
    set(_ZSTD_LIB ${ZSTD_LIBRARIES})
    set(_ZSTD_INC ${ZSTD_INCLUDE_DIRS})
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
)

# Helper: apply common settings to a cuvk target
function(cuvk_target_setup tgt)
    target_link_libraries(${tgt} PUBLIC wgsl_compiler ${_VULKAN_LIB} ${_ZSTD_LIB})
    target_include_directories(${tgt} PUBLIC ${CMAKE_SOURCE_DIR} ${_VULKAN_INC} ${_ZSTD_INC})
    if(CUVK_DEBUG_LOG)
        target_compile_definitions(${tgt} PRIVATE CUVK_DEBUG_LOG=1)
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
        LINK_OPTIONS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cublas.map"
    )
endif()
