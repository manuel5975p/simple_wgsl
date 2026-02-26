# E2E tests: nvcc-compiled CUDA kernels running on our Vulkan backend
#
# Requires: nvcc (CUDA toolkit), Vulkan

find_program(NVCC_EXECUTABLE nvcc PATHS /opt/cuda/bin)
find_package(Vulkan)

if(NOT NVCC_EXECUTABLE)
    message(STATUS "nvcc not found - skipping e2e CUDA tests")
    return()
endif()

if(NOT Vulkan_FOUND)
    message(STATUS "Vulkan not found - skipping e2e CUDA tests")
    return()
endif()

message(STATUS "nvcc found: ${NVCC_EXECUTABLE}")
message(STATUS "Building e2e CUDA-on-Vulkan tests")

# Detect a supported nvcc architecture
execute_process(
    COMMAND ${NVCC_EXECUTABLE} --list-gpu-arch
    OUTPUT_VARIABLE NVCC_ARCHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
# Pick the first compute_* arch and map to sm_*
string(REGEX MATCH "compute_([0-9]+)" _match "${NVCC_ARCHS}")
if(_match)
    set(NVCC_SM "sm_${CMAKE_MATCH_1}")
else()
    set(NVCC_SM "sm_75")
endif()
message(STATUS "Using nvcc target: ${NVCC_SM}")

# --- Compile .cu kernels to .ptx ---
set(KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
set(PTX_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/ptx)
file(MAKE_DIRECTORY ${PTX_OUTPUT_DIR})

set(CUDA_KERNELS
    vector_add
    saxpy
    scale
    dot_product
)

foreach(kernel ${CUDA_KERNELS})
    set(CU_SRC ${KERNEL_DIR}/${kernel}.cu)
    set(PTX_OUT ${PTX_OUTPUT_DIR}/${kernel}.ptx)
    add_custom_command(
        OUTPUT ${PTX_OUT}
        COMMAND ${NVCC_EXECUTABLE} --ptx -o ${PTX_OUT} ${CU_SRC} -arch=${NVCC_SM}
        DEPENDS ${CU_SRC}
        COMMENT "Compiling ${kernel}.cu -> ${kernel}.ptx (${NVCC_SM})"
        VERBATIM
    )
    list(APPEND PTX_FILES ${PTX_OUT})
endforeach()

add_custom_target(e2e_ptx_kernels DEPENDS ${PTX_FILES})

# --- Build host test executables linked against cuvk_runtime ---
set(E2E_TESTS
    test_vector_add
    test_saxpy
    test_scale
    test_dot_product
)

# Map test name -> kernel name for PTX path
set(test_vector_add_KERNEL vector_add)
set(test_saxpy_KERNEL saxpy)
set(test_scale_KERNEL scale)
set(test_dot_product_KERNEL dot_product)

foreach(test ${E2E_TESTS})
    add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.c)
    target_include_directories(${test} PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(${test} cuvk_runtime_shared m)
    target_compile_options(${test} PRIVATE -std=c99)
    add_dependencies(${test} e2e_ptx_kernels)

    # Register as CTest with the PTX file as argument
    set(_kernel ${${test}_KERNEL})
    add_test(
        NAME e2e_${test}
        COMMAND ${test} ${PTX_OUTPUT_DIR}/${_kernel}.ptx
    )
    set_tests_properties(e2e_${test} PROPERTIES
        LABELS "e2e"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/cuvk_runtime:$ENV{LD_LIBRARY_PATH}"
    )
endforeach()

# --- LD_PRELOAD test: host binary linked against CUDA stub, intercepted at runtime ---
find_path(CUDA_STUBS_DIR libcuda.so PATHS /opt/cuda/lib64/stubs)
if(CUDA_STUBS_DIR)
    message(STATUS "CUDA stubs found: ${CUDA_STUBS_DIR} - building LD_PRELOAD test")

    add_executable(preload_test ${CMAKE_CURRENT_SOURCE_DIR}/preload_test.c)
    target_include_directories(preload_test PRIVATE /opt/cuda/include)
    target_link_directories(preload_test PRIVATE ${CUDA_STUBS_DIR})
    target_link_libraries(preload_test cuda m)
    target_compile_options(preload_test PRIVATE -std=c99)
    add_dependencies(preload_test e2e_ptx_kernels)

    # Collect all PTX files as arguments
    set(PTX_ARGS "")
    foreach(kernel ${CUDA_KERNELS})
        list(APPEND PTX_ARGS ${PTX_OUTPUT_DIR}/${kernel}.ptx)
    endforeach()

    add_test(
        NAME e2e_preload_test
        COMMAND ${CMAKE_COMMAND} -E env
            "LD_PRELOAD=${CMAKE_BINARY_DIR}/cuvk_runtime/libcuda.so.1"
            "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/cuvk_runtime:$ENV{LD_LIBRARY_PATH}"
            $<TARGET_FILE:preload_test> ${PTX_ARGS}
    )
    set_tests_properties(e2e_preload_test PROPERTIES
        LABELS "e2e;preload"
    )
endif()

# --- nvcc full-binary tests: compile .cu to executables, run against libcuda.so.1 ---
set(NVCC_TESTS
    01_vector_add
    02_device_clamp
    03_device_chain
    04_saxpy_fma
    05_polynomial
    06_mandelbrot
    07_multi_device_func
    08_multi_kernel
    09_streams
    10_matmul
    11_double_underscore_math
    12_rsqrt_rcp
    13_integer_intrinsics
    14_saturate_math
    15_double_precision
    16_half_float_basic
    17_half2_packed
    18_fast_trig
    19_warp_shuffle
    20_shared_memory
    21_atomic_ops
    22_type_conversions
    23_bitwise_ops
    24_ternary_select
    25_3d_grid
    26_float_special
    27_prefix_scan
    28_long_long_ops
    29_grid_stride_loop
    30_complex_math
    31_popc_brev
    32_clz_bfind
    33_cvt_rounding
    34_funnel_shift
    35_standard_math_heavy
    36_fast_intrinsics_stress
    37_compound_math_chains
    38_extended_math
)

set(NVCC_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/nvcc)
file(MAKE_DIRECTORY ${NVCC_BIN_DIR})

set(NVCC_BINARIES "")
foreach(test ${NVCC_TESTS})
    set(CU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/nvcc/${test}.cu)
    set(NVCC_OUT ${NVCC_BIN_DIR}/${test})
    add_custom_command(
        OUTPUT ${NVCC_OUT}
        COMMAND ${NVCC_EXECUTABLE} -o ${NVCC_OUT} ${CU_SRC} -arch=${NVCC_SM}
        DEPENDS ${CU_SRC}
        COMMENT "nvcc: ${test}.cu -> ${test}"
        VERBATIM
    )
    list(APPEND NVCC_BINARIES ${NVCC_OUT})

    add_test(
        NAME e2e_nvcc_${test}
        COMMAND ${CMAKE_COMMAND} -E env
            "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/cuvk_runtime:$ENV{LD_LIBRARY_PATH}"
            ${NVCC_OUT}
    )
    set_tests_properties(e2e_nvcc_${test} PROPERTIES
        LABELS "e2e;nvcc"
    )
endforeach()

add_custom_target(e2e_nvcc_binaries ALL DEPENDS ${NVCC_BINARIES})
