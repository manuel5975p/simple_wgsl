include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)
find_package(Vulkan)

# --- Compiler unit tests (wgsl_tests binary) ---
if(WGSL_BUILD_COMPILER_TESTS)
    set(TEST_SOURCES
        parser_test.cpp
        resolver_test.cpp
        lower_test.cpp
        raise_test.cpp
        ssir_raise_test.cpp
        spirv_to_ssir_test.cpp
        integration_test.cpp
        glsl_parser_test.cpp
        glsl_raise_test.cpp
        ssir_builder_test.cpp
        ssir_to_hlsl_test.cpp
        ssir_to_msl_test.cpp
        msl_parser_test.cpp
        ptx_parser_test.cpp
        ptx_ssir_test.cpp
        ptx_texture_test.cpp
        spec_compliance_test.cpp
        select_test.cpp
        all_any_test.cpp
        fma_test.cpp
        saturate_test.cpp
        faceforward_refract_test.cpp
        transpose_determinant_test.cpp
        bitmanip_test.cpp
        texture_sampling_test.cpp
        texture_query_test.cpp
        derivative_test.cpp
        atomic_test.cpp
        pack_unpack_test.cpp
        barrier_test.cpp
        immediate_test.cpp
        slug_regression_test.cpp
        slug_pattern_test.cpp
        wgsl_roundtrip_test.cpp
        glsl_roundtrip_test.cpp
    )

    if(WGSL_BUILD_EXPRESSION_TESTS)
        list(APPEND TEST_SOURCES expression_test.cpp)
    endif()

    if(WGSL_BUILD_VULKAN_TESTS AND Vulkan_FOUND)
        list(APPEND TEST_SOURCES
            vulkan_compute_harness.cpp
            vulkan_compute_test.cpp
            cuda_to_vulkan_test.cpp
            vulkan_graphics_harness.cpp
            vulkan_graphics_test.cpp
            vulkan_graphics_complex_test.cpp
            vulkan_graphics_scene_test.cpp
        )
    endif()

    if(WGSL_BUILD_EGL_TESTS)
        find_package(OpenGL COMPONENTS EGL OpenGL)
        if(OpenGL_EGL_FOUND)
            list(APPEND TEST_SOURCES
                egl_compute_harness.cpp
                egl_compute_test.cpp
                egl_graphics_test.cpp
            )
        else()
            message(WARNING "WGSL_BUILD_EGL_TESTS enabled but EGL not found")
        endif()
    endif()

    add_executable(wgsl_tests ${TEST_SOURCES})

    target_link_libraries(wgsl_tests
        wgsl_compiler
        GTest::gtest
        GTest::gtest_main
        m
    )

    if(TARGET fft_stockham_gen)
        target_link_libraries(wgsl_tests
            fft_stockham_gen fft_fused_gen fft_2d_gen fft_fourstep_gen fft_optimal_gen)
        target_compile_definitions(wgsl_tests PRIVATE WGSL_HAS_FFT=1)
    endif()

    if(WGSL_BUILD_VULKAN_TESTS AND Vulkan_FOUND)
        target_link_libraries(wgsl_tests Vulkan::Vulkan)
        target_compile_definitions(wgsl_tests PRIVATE WGSL_HAS_VULKAN=1)
    endif()

    if(WGSL_BUILD_EGL_TESTS AND OpenGL_EGL_FOUND)
        target_link_libraries(wgsl_tests OpenGL::EGL OpenGL::OpenGL)
        target_compile_definitions(wgsl_tests PRIVATE WGSL_HAS_EGL=1)
    endif()

    target_include_directories(wgsl_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/cuvk_runtime/fft
    )

    if(WGSL_BUILD_EXPRESSION_TESTS)
        target_compile_definitions(wgsl_tests PRIVATE
            WGSL_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )
    endif()

    gtest_discover_tests(wgsl_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        PROPERTIES LABELS "unit"
    )

    add_test(NAME all_tests COMMAND wgsl_tests)
endif()

# --- CUVK runtime standalone tests ---
if(WGSL_BUILD_CUVK_TESTS AND Vulkan_FOUND)
    add_executable(cuvk_init_test cuvk_init_test.cpp)
    target_link_libraries(cuvk_init_test gtest_main cuvk_runtime_static)
    add_test(NAME cuvk_init_test COMMAND cuvk_init_test)

    add_executable(cuvk_stream_test cuvk_stream_test.cpp)
    target_link_libraries(cuvk_stream_test gtest_main cuvk_runtime_static)
    add_test(NAME cuvk_stream_test COMMAND cuvk_stream_test)

    add_executable(cuvk_memory_test cuvk_memory_test.cpp)
    target_link_libraries(cuvk_memory_test gtest_main cuvk_runtime_static)
    add_test(NAME cuvk_memory_test COMMAND cuvk_memory_test)

    add_executable(cuvk_launch_test cuvk_launch_test.cpp)
    target_link_libraries(cuvk_launch_test gtest_main cuvk_runtime_static)
    add_test(NAME cuvk_launch_test COMMAND cuvk_launch_test)

    add_executable(cuvk_bda_test cuvk_bda_test.cpp)
    target_link_libraries(cuvk_bda_test gtest_main cuvk_runtime_static)
    add_test(NAME cuvk_bda_test COMMAND cuvk_bda_test)

    add_executable(cuvk_integration_test cuvk_integration_test.cpp)
    target_link_libraries(cuvk_integration_test gtest_main cuvk_runtime_static)
    add_test(NAME cuvk_integration_test COMMAND cuvk_integration_test)

    add_executable(cuvk_graph_test cuvk_graph_test.cpp)
    target_link_libraries(cuvk_graph_test gtest_main cuvk_runtime_static)
    add_test(NAME cuvk_graph_test COMMAND cuvk_graph_test)

    add_executable(cuvk_complex_kernel_test cuvk_complex_kernel_test.cpp)
    target_link_libraries(cuvk_complex_kernel_test gtest_main cuvk_runtime_static)
    add_test(NAME cuvk_complex_kernel_test COMMAND cuvk_complex_kernel_test)
endif()

# --- FFT tests ---
if(WGSL_BUILD_FFT_TESTS)
    if(Vulkan_FOUND)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(FFTW3F QUIET fftw3f)
        endif()
        if(FFTW3F_FOUND)
            add_executable(cufft_test
                cufft_test.cpp
                ${CMAKE_SOURCE_DIR}/cuvk_runtime/fft/cuvk_cufft.c
            )
            target_link_libraries(cufft_test
                gtest_main
                cuvk_runtime_static
                wgsl_compiler
                fft_stockham_gen
                fft_fused_gen
                fft_fourstep_gen
                fft_2d_gen
                ${FFTW3F_LIBRARIES}
                m
            )
            target_include_directories(cufft_test PRIVATE
                ${CMAKE_SOURCE_DIR}/cuvk_runtime/fft
                ${CMAKE_SOURCE_DIR}/cuvk_runtime
                ${CMAKE_SOURCE_DIR}
                ${FFTW3F_INCLUDE_DIRS}
            )
            gtest_discover_tests(cufft_test
                PROPERTIES LABELS "cudart"
            )
        endif()
    endif()

    # FFT four-step generator tests (no Vulkan needed, but library built under cuvk_runtime)
    if(TARGET fft_fourstep_gen)
        add_executable(fourstep_gen_test fourstep_gen_test.cpp)
        target_link_libraries(fourstep_gen_test
            gtest_main
            fft_fourstep_gen
            wgsl_compiler
            m
        )
        target_include_directories(fourstep_gen_test PRIVATE
            ${CMAKE_SOURCE_DIR}/cuvk_runtime/fft
            ${CMAKE_SOURCE_DIR}
        )
        gtest_discover_tests(fourstep_gen_test)
    endif()

    # FFT 2D generator tests (no Vulkan needed, but library built under cuvk_runtime)
    if(TARGET fft_2d_gen)
        add_executable(fft_2d_gen_test fft_2d_gen_test.cpp)
        target_link_libraries(fft_2d_gen_test
            gtest_main
            fft_2d_gen
            wgsl_compiler
            m
        )
        target_include_directories(fft_2d_gen_test PRIVATE
            ${CMAKE_SOURCE_DIR}/cuvk_runtime/fft
            ${CMAKE_SOURCE_DIR}
        )
        gtest_discover_tests(fft_2d_gen_test)
    endif()
endif()

# --- E2E tests ---
if(WGSL_BUILD_E2E_TESTS AND Vulkan_FOUND)
    add_subdirectory(e2e)
endif()
