# Fuzz targets — requires Clang with -fsanitize=fuzzer support
if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "Fuzzing requires Clang (set CC=clang CXX=clang++)")
endif()

set(FUZZ_FLAGS -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
set(FUZZ_LINK  -fsanitize=fuzzer,address,undefined)

function(add_fuzz_target NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} wgsl_compiler)
    target_compile_options(${NAME} PRIVATE ${FUZZ_FLAGS})
    target_link_options(${NAME} PRIVATE ${FUZZ_LINK})
endfunction()

add_fuzz_target(fuzz_wgsl_parse    fuzz_wgsl_parse.c)
add_fuzz_target(fuzz_glsl_parse    fuzz_glsl_parse.c)
add_fuzz_target(fuzz_msl_parse     fuzz_msl_parse.c)
add_fuzz_target(fuzz_ptx_parse     fuzz_ptx_parse.c)
add_fuzz_target(fuzz_spirv_to_ssir fuzz_spirv_to_ssir.c)
add_fuzz_target(fuzz_wgsl_roundtrip fuzz_wgsl_roundtrip.c)
