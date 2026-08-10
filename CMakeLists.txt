cmake_minimum_required(VERSION 3.25)
project(llm_from_scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 

set(CMAKE_CXX_FLAGS_SANITIZE "-g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS_SANITIZE "-fsanitize=address,undefined")

find_package(Catch2 3 REQUIRED)

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
add_executable(tests ${TEST_SOURCES})

target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(tests)