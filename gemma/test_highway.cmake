cmake_minimum_required(VERSION 3.20)
project(test_highway)

set(CMAKE_TOOLCHAIN_FILE "C:/codedev/vcpkg/scripts/buildsystems/vcpkg.cmake")

find_package(hwy CONFIG REQUIRED)

if(TARGET hwy::hwy)
    message(STATUS "SUCCESS: Found hwy::hwy target")
else()
    message(FATAL_ERROR "FAILED: hwy::hwy target not found")
endif()
