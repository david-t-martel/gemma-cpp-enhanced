# Patch script for SentencePiece CMakeLists.txt compatibility
# Updates minimum CMake version requirement

set(SOURCE_DIR ${CMAKE_ARGV3})
if(NOT SOURCE_DIR OR NOT EXISTS "${SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "SentencePiece source directory not found: ${SOURCE_DIR}")
endif()

set(CMAKELIST_FILE "${SOURCE_DIR}/CMakeLists.txt")
file(READ "${CMAKELIST_FILE}" CONTENT)

# Replace CMake minimum version
string(REPLACE
    "cmake_minimum_required(VERSION 3.1 FATAL_ERROR)"
    "cmake_minimum_required(VERSION 3.5 FATAL_ERROR)"
    CONTENT "${CONTENT}")

file(WRITE "${CMAKELIST_FILE}" "${CONTENT}")
message(STATUS "Patched ${CMAKELIST_FILE} for CMake 3.5 compatibility")