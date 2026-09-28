cmake_minimum_required(VERSION 3.16)
project(gpt2cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)
option(GPT2_SANITIZE "Build with AddressSanitizer + UBSan" OFF)
if(GPT2_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(gpt2core src/model.cpp)
target_include_directories(gpt2core PUBLIC src)

add_executable(gpt2 src/main.cpp)
target_link_libraries(gpt2 gpt2core)

# Tests: generate a tiny random checkpoint with numpy, then load it from C++.
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_executable(test_loader tests/test_loader.cpp)
target_link_libraries(test_loader gpt2core)
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/tiny.bin ${CMAKE_BINARY_DIR}/tiny.bin.ref
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/make_tiny_checkpoint.py ${CMAKE_BINARY_DIR}/tiny.bin
  DEPENDS ${CMAKE_SOURCE_DIR}/scripts/make_tiny_checkpoint.py ${CMAKE_SOURCE_DIR}/scripts/gpt2bin.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_custom_target(tiny_checkpoint ALL DEPENDS ${CMAKE_BINARY_DIR}/tiny.bin)
add_test(NAME loader COMMAND test_loader ${CMAKE_BINARY_DIR}/tiny.bin
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
