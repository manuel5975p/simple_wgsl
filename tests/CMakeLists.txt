include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

find_package(Vulkan)

set(TEST_SOURCES
    parser_test.cpp
    resolver_test.cpp
    lower_test.cpp
    raise_test.cpp
    integration_test.cpp
    expression_test.cpp
)

if(Vulkan_FOUND)
    list(APPEND TEST_SOURCES
        vulkan_compute_harness.cpp
        vulkan_compute_test.cpp
        vulkan_graphics_harness.cpp
        vulkan_graphics_test.cpp
    )
endif()

add_executable(wgsl_tests ${TEST_SOURCES})

target_link_libraries(wgsl_tests
    wgsl_compiler
    GTest::gtest
    GTest::gtest_main
)

if(Vulkan_FOUND)
    target_link_libraries(wgsl_tests Vulkan::Vulkan)
    target_compile_definitions(wgsl_tests PRIVATE WGSL_HAS_VULKAN=1)
endif()

target_include_directories(wgsl_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Set source directory for expression tests to find test data
target_compile_definitions(wgsl_tests PRIVATE
    WGSL_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

include(GoogleTest)
gtest_discover_tests(wgsl_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES LABELS "unit"
)

add_test(NAME all_tests COMMAND wgsl_tests)
