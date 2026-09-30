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
option(GPT2_TSAN "Build with ThreadSanitizer (not combinable with GPT2_SANITIZE)" OFF)
if(GPT2_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()
option(GPT2_NO_SIMD "Use the portable scalar kernels even where NEON is available" OFF)
if(GPT2_NO_SIMD)
  add_compile_definitions(GPT2_NO_SIMD)
endif()
if(GPT2_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(gpt2core src/model.cpp src/tokenizer.cpp src/ops.cpp src/forward.cpp
            src/threadpool.cpp src/sampler.cpp)
find_package(Threads REQUIRED)
target_link_libraries(gpt2core PUBLIC Threads::Threads)
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

# Forward pass: invariants on the tiny checkpoint, then logits vs HF on the real model.
# forward_hf: 6 prompts, every row (models/logits_ref.bin); forward_hf_decode: the same, fed one
# token at a time through the KV cache. diff_hf: the Phase 3 differential
# test, 94 prompts up to n_ctx (models/diff_ref.bin); slow until later phases speed up forward,
# so it's labelled: `ctest -LE slow` skips it. Both need scripts/make_logits_ref.py output and
# are skipped (exit 77) without it.
add_executable(test_forward tests/test_forward.cpp)
target_link_libraries(test_forward gpt2core)
add_test(NAME forward_tiny COMMAND test_forward tiny ${CMAKE_BINARY_DIR}/tiny.bin
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
add_test(NAME forward_hf
         COMMAND test_forward hf ${CMAKE_SOURCE_DIR}/models/gpt2.bin
                 ${CMAKE_SOURCE_DIR}/models/logits_ref.bin
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
add_test(NAME forward_threads COMMAND test_forward threads ${CMAKE_SOURCE_DIR}/models/gpt2.bin
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
add_test(NAME forward_hf_decode
         COMMAND test_forward decode ${CMAKE_SOURCE_DIR}/models/gpt2.bin
                 ${CMAKE_SOURCE_DIR}/models/logits_ref.bin
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
add_test(NAME diff_hf
         COMMAND test_forward hf ${CMAKE_SOURCE_DIR}/models/gpt2.bin
                 ${CMAKE_SOURCE_DIR}/models/diff_ref.bin
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(forward_hf forward_threads forward_hf_decode diff_hf PROPERTIES
                     SKIP_RETURN_CODE 77)
set_tests_properties(diff_hf PROPERTIES LABELS slow TIMEOUT 3600)

# Sampler: unit tests, and the filtered distributions vs HF's warpers on real logits
# (models/sampler_ref.bin from scripts/make_sampler_ref.py; skipped without it).
add_executable(test_sampler tests/test_sampler.cpp)
target_link_libraries(test_sampler gpt2core)
add_test(NAME sampler_unit COMMAND test_sampler unit)
add_test(NAME sampler_hf COMMAND test_sampler hf ${CMAKE_SOURCE_DIR}/models/sampler_ref.bin)
set_tests_properties(sampler_hf PROPERTIES SKIP_RETURN_CODE 77)
