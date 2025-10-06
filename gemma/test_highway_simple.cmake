cmake_minimum_required(VERSION 3.20)
project(highway_test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add Highway from GitHub directory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/highway-github/CMakeLists.txt")
    message(STATUS "Adding GitHub Highway from third_party/highway-github")
    add_subdirectory(third_party/highway-github)
else()
    message(FATAL_ERROR "Highway-github not found!")
endif()

# Simple test executable
add_executable(highway_test test_highway_github.cpp)
target_link_libraries(highway_test hwy)
target_include_directories(highway_test PRIVATE third_party/highway-github)