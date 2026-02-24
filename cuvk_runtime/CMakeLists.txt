find_package(Vulkan REQUIRED)

set(CUVK_SOURCES
    cuvk_init.c
    cuvk_memory.c
    cuvk_module.c
    cuvk_launch.c
    cuvk_stream.c
    cuvk_stubs.c
)

# Static library
add_library(cuvk_runtime_static STATIC ${CUVK_SOURCES})
target_link_libraries(cuvk_runtime_static PUBLIC wgsl_compiler Vulkan::Vulkan)
target_include_directories(cuvk_runtime_static PUBLIC ${CMAKE_SOURCE_DIR})
set_target_properties(cuvk_runtime_static PROPERTIES OUTPUT_NAME cuvk_runtime)

# Shared library (for LD_PRELOAD use)
add_library(cuvk_runtime_shared SHARED ${CUVK_SOURCES})
target_link_libraries(cuvk_runtime_shared PUBLIC wgsl_compiler Vulkan::Vulkan)
target_include_directories(cuvk_runtime_shared PUBLIC ${CMAKE_SOURCE_DIR})
set_target_properties(cuvk_runtime_shared PROPERTIES OUTPUT_NAME cuvk_runtime)
