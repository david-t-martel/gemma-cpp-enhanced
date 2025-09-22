# Session Management Library for Gemma.cpp
cmake_minimum_required(VERSION 3.16)

# Session management library
add_library(gemma_session
    Session.cpp
    Session.h
    SessionStorage.cpp
    SessionStorage.h
    SessionManager.cpp
    SessionManager.h
)

# Set target properties
set_target_properties(gemma_session PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Include directories
target_include_directories(gemma_session
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Find required dependencies
find_package(PkgConfig REQUIRED)

# Find SQLite3
find_package(SQLite3 REQUIRED)
if(NOT SQLite3_FOUND)
    # Fallback to pkg-config
    pkg_check_modules(SQLite3 REQUIRED sqlite3)
endif()

# Find nlohmann/json
find_package(nlohmann_json REQUIRED)
if(NOT nlohmann_json_FOUND)
    # Try to find it via FetchContent if not found
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Link libraries
target_link_libraries(gemma_session
    PUBLIC
        SQLite::SQLite3
        nlohmann_json::nlohmann_json
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
)

# Compiler-specific options
if(MSVC)
    target_compile_options(gemma_session PRIVATE 
        /W4 
        /permissive- 
        /Zc:__cplusplus
    )
    # Enable parallel compilation
    target_compile_options(gemma_session PRIVATE /MP)
else()
    target_compile_options(gemma_session PRIVATE 
        -Wall 
        -Wextra 
        -Wpedantic
        -Wno-unused-parameter
    )
    
    # Enable optimizations for Release builds
    target_compile_options(gemma_session PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-DNDEBUG>
    )
    
    # Enable debug symbols for Debug builds
    target_compile_options(gemma_session PRIVATE
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-O0>
    )
endif()

# Thread support
find_package(Threads REQUIRED)
target_link_libraries(gemma_session PRIVATE Threads::Threads)

# Optional: Build tests if testing is enabled
if(BUILD_TESTING OR GEMMA_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Build benchmark executable if requested
option(GEMMA_BUILD_BENCHMARKS "Build performance benchmarks" OFF)
if(GEMMA_BUILD_BENCHMARKS)
    add_executable(session_benchmark benchmark_session.cpp)
    target_link_libraries(session_benchmark
        PRIVATE
            gemma_session
            Threads::Threads
    )
    set_target_properties(session_benchmark PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
endif()

# Optional: Install targets
if(GEMMA_INSTALL_TARGETS)
    install(TARGETS gemma_session
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
    
    install(FILES
        Session.h
        SessionStorage.h
        SessionManager.h
        DESTINATION include/gemma/session
    )
endif()

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Platform-specific configurations
if(WIN32)
    # Windows-specific settings
    target_compile_definitions(gemma_session PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
    )
    
    # Link with additional Windows libraries if needed
    target_link_libraries(gemma_session PRIVATE ws2_32)
endif()

if(UNIX AND NOT APPLE)
    # Linux-specific settings
    target_compile_definitions(gemma_session PRIVATE
        _GNU_SOURCE
    )
endif()

if(APPLE)
    # macOS-specific settings
    target_compile_definitions(gemma_session PRIVATE
        _DARWIN_C_SOURCE
    )
endif()

# Add compile definitions for feature detection
target_compile_definitions(gemma_session PUBLIC
    GEMMA_HAS_SESSION_MANAGEMENT=1
)

# Debug build specific settings
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(gemma_session PRIVATE
        GEMMA_SESSION_DEBUG=1
    )
endif()

# Add version information
if(DEFINED PROJECT_VERSION)
    target_compile_definitions(gemma_session PRIVATE
        GEMMA_SESSION_VERSION="${PROJECT_VERSION}"
    )
endif()

# Create an alias for easier reference
add_library(Gemma::Session ALIAS gemma_session)

# Print configuration summary
message(STATUS "Gemma Session Management Library Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  SQLite3 Found: ${SQLite3_FOUND}")
message(STATUS "  nlohmann/json Found: ${nlohmann_json_FOUND}")
message(STATUS "  Testing Enabled: ${BUILD_TESTING}")
if(WIN32)
    message(STATUS "  Platform: Windows")
elseif(APPLE)
    message(STATUS "  Platform: macOS")
else()
    message(STATUS "  Platform: Linux/Unix")
endif()