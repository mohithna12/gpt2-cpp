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

add_library(gpt2core src/model.cpp src/tokenizer.cpp)
target_include_directories(gpt2core PUBLIC src)

add_executable(gpt2 src/main.cpp)
target_link_libraries(gpt2 gpt2core)

# Tests: generate a tiny random checkpoint with numpy, then load it from C++.
enable_testing()
# FindPython picks the newest interpreter on the system, which may lack numpy. Prefer the
# project venv (or an activated one); override with -DPython3_EXECUTABLE=... if needed.
if(NOT DEFINED Python3_EXECUTABLE AND EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
  set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
endif()
set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                RESULT_VARIABLE _numpy_rc OUTPUT_QUIET ERROR_QUIET)
if(NOT _numpy_rc EQUAL 0)
  message(FATAL_ERROR "numpy is not importable by ${Python3_EXECUTABLE}; tests need it to "
                      "generate the tiny checkpoint. Create .venv (see CLAUDE.md) or pass "
                      "-DPython3_EXECUTABLE=/path/to/python.")
endif()
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

# Tokenizer vs HF ids. Needs models/vocab.json + merges.txt from scripts/export_gpt2.py;
# skipped (exit 77) when they're absent.
add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer gpt2core)
add_test(NAME tokenizer
         COMMAND test_tokenizer ${CMAKE_SOURCE_DIR}/models ${CMAKE_SOURCE_DIR}/tests/tokenizer_cases.tsv
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(tokenizer PROPERTIES SKIP_RETURN_CODE 77)
