cmake_minimum_required(VERSION 3.16)
project(mcp_gemma_server VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Check if vcpkg is being used
if(DEFINED ENV{VCPKG_ROOT})
    message(STATUS "Using vcpkg from: $ENV{VCPKG_ROOT}")
endif()

# Find packages that might be available through vcpkg or system
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    # Fallback to find_path/find_library if package config is not available
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
    if(NOT NLOHMANN_JSON_INCLUDE_DIR)
        message(WARNING "nlohmann/json not found. Please install via vcpkg or system package manager")
    endif()
endif()

# WebSocket++ is header-only, find include directory
find_path(WEBSOCKETPP_INCLUDE_DIR websocketpp/server.hpp)
if(NOT WEBSOCKETPP_INCLUDE_DIR)
    message(WARNING "WebSocket++ not found. Please install via vcpkg or download headers")
    # Try common locations
    set(WEBSOCKETPP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/websocketpp" CACHE PATH "WebSocket++ include directory")
endif()

# Find Boost (required by WebSocket++)
find_package(Boost REQUIRED COMPONENTS system)

# Include directories
set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${WEBSOCKETPP_INCLUDE_DIR}
)

if(NLOHMANN_JSON_INCLUDE_DIR)
    list(APPEND INCLUDE_DIRS ${NLOHMANN_JSON_INCLUDE_DIR})
endif()

# Add Gemma.cpp path - adjust as needed
set(GEMMA_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../gemma" CACHE PATH "Path to Gemma.cpp root")
if(EXISTS "${GEMMA_CPP_ROOT}")
    list(APPEND INCLUDE_DIRS "${GEMMA_CPP_ROOT}")
    message(STATUS "Found Gemma.cpp at: ${GEMMA_CPP_ROOT}")
else()
    message(WARNING "Gemma.cpp not found at: ${GEMMA_CPP_ROOT}")
    message(WARNING "Please set GEMMA_CPP_ROOT to the correct path")
endif()

# Source files
set(SOURCES
    mcp_server.cpp
    inference_handler.cpp
    model_manager.cpp
    main.cpp
)

# Create executable
add_executable(mcp_gemma_server ${SOURCES})

# Set include directories
target_include_directories(mcp_gemma_server PRIVATE ${INCLUDE_DIRS})

# Link libraries
target_link_libraries(mcp_gemma_server
    PRIVATE
    Threads::Threads
    Boost::system
)

# Link nlohmann_json if found as a target
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(mcp_gemma_server PRIVATE nlohmann_json::nlohmann_json)
endif()

# Compiler-specific options
if(MSVC)
    target_compile_options(mcp_gemma_server PRIVATE
        /W4
        /WX-
        /EHsc
        /bigobj  # For large template instantiations
    )
    # Windows-specific definitions
    target_compile_definitions(mcp_gemma_server PRIVATE
        _WIN32_WINNT=0x0601  # Windows 7 minimum
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
else()
    target_compile_options(mcp_gemma_server PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -O3  # Optimization for release builds
    )
endif()

# Definitions
target_compile_definitions(mcp_gemma_server PRIVATE
    ASIO_STANDALONE  # Use standalone Asio with WebSocket++
    _WEBSOCKETPP_CPP11_STL_
)

# Platform-specific linking
if(WIN32)
    target_link_libraries(mcp_gemma_server PRIVATE ws2_32 wsock32)
elseif(UNIX)
    target_link_libraries(mcp_gemma_server PRIVATE pthread)
endif()

# Installation
install(TARGETS mcp_gemma_server
    RUNTIME DESTINATION bin
)

# Print build information
message(STATUS "Build configuration:")
message(STATUS "  CXX Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  WebSocket++ Include: ${WEBSOCKETPP_INCLUDE_DIR}")
if(nlohmann_json_FOUND)
    message(STATUS "  nlohmann/json: Found (package)")
elseif(NLOHMANN_JSON_INCLUDE_DIR)
    message(STATUS "  nlohmann/json: ${NLOHMANN_JSON_INCLUDE_DIR}")
endif()