# FFT shader generator (standalone, no Vulkan dependency)
add_library(fft_stockham_gen STATIC fft_stockham_gen.c)
target_link_libraries(fft_stockham_gen m)
target_include_directories(fft_stockham_gen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# FFT optimal (single-dispatch, register-based) shader generator
add_library(fft_optimal_gen STATIC fft_optimal_gen.c)
target_link_libraries(fft_optimal_gen m)
target_include_directories(fft_optimal_gen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# FFT fused (single-dispatch, shared-memory) shader generator
add_library(fft_fused_gen STATIC fft_fused_gen.c)
target_link_libraries(fft_fused_gen m)
target_include_directories(fft_fused_gen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Shared library: cufft64_*.dll / libcufft.so.12
# Links against cuvk_runtime_shared (libcuda.so) to share g_cuvk state.
add_library(cuvk_cufft_shared SHARED cuvk_cufft.c)
target_link_libraries(cuvk_cufft_shared PUBLIC cuvk_runtime_shared wgsl_compiler fft_stockham_gen ${_VULKAN_LIB} m)
target_include_directories(cuvk_cufft_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}
    ${_VULKAN_INC}
)
if(CUVK_DEBUG_LOG)
    target_compile_definitions(cuvk_cufft_shared PRIVATE CUVK_DEBUG_LOG=1)
endif()
if(WIN32)
    set_target_properties(cuvk_cufft_shared PROPERTIES
        OUTPUT_NAME "cufft64_12"
        PREFIX ""
    )
else()
    set_target_properties(cuvk_cufft_shared PROPERTIES
        OUTPUT_NAME "cufft"
        SOVERSION 12
        LINK_OPTIONS "$<$<NOT:$<PLATFORM_ID:Darwin>>:-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cufft.map>"
    )
endif()
