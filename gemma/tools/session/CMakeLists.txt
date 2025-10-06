# Session Management Library for Gemma.cpp
cmake_minimum_required(VERSION 3.16)

# Session management library (monolithic initial version - slated for refactor per SESSION_REFACTORING_PLAN)
add_library(gemma_session
    Session.cpp
    Session.h
    session_serialization.cpp
    session_serialization.h
    conversation_types.h
    session_json.cpp
    i_session_storage.h
    i_session_cache.h
    SessionStorage.cpp
    SessionStorage.h
    SessionManager.cpp
    SessionManager.h
    sqlite_statement.cpp
    sqlite_statement.h
    lru_cache.cpp
    lru_cache.h
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

# Dependency resolution (with vendored fallbacks)
find_package(PkgConfig QUIET)

# --- SQLite3 ---
find_package(SQLite3 QUIET)
if(NOT SQLite3_FOUND)
    if(PkgConfig_FOUND)
        pkg_check_modules(SQLite3 QUIET sqlite3)
    endif()
endif()
if(NOT SQLite3_FOUND)
    set(_VENDORED_SQLITE_DIR "${PROJECT_SOURCE_DIR}/third_party/sqlite")
    if(EXISTS "${_VENDORED_SQLITE_DIR}/sqlite3.c")
        message(STATUS "Using vendored sqlite3 from third_party/sqlite")
        add_library(sqlite3_vendor STATIC "${_VENDORED_SQLITE_DIR}/sqlite3.c")
        target_include_directories(sqlite3_vendor PUBLIC "${_VENDORED_SQLITE_DIR}")
        # Normalize warnings for MSVC vs others
        if(MSVC)
            target_compile_definitions(sqlite3_vendor PRIVATE _CRT_SECURE_NO_WARNINGS)
            target_compile_options(sqlite3_vendor PRIVATE /W3)
        else()
            target_compile_options(sqlite3_vendor PRIVATE -w)
        endif()
        add_library(SQLite::SQLite3 ALIAS sqlite3_vendor)
        set(SQLite3_FOUND ON)
    else()
        message(FATAL_ERROR "SQLite3 not found and vendored source missing (expected sqlite3.c under third_party/sqlite)")
    endif()
endif()

# --- nlohmann_json ---
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    set(_VENDORED_JSON_DIR "${PROJECT_SOURCE_DIR}/third_party/nlohmann_json")
    if(EXISTS "${_VENDORED_JSON_DIR}/single_include/nlohmann/json.hpp")
        message(STATUS "Using vendored nlohmann_json from third_party/nlohmann_json")
        add_library(nlohmann_json_vendor INTERFACE)
        target_include_directories(nlohmann_json_vendor INTERFACE "${_VENDORED_JSON_DIR}/single_include")
        add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json_vendor)
        set(nlohmann_json_FOUND ON)
    else()
        message(FATAL_ERROR "nlohmann_json not found and vendored single_include missing (expected single_include/nlohmann/json.hpp)")
    endif()
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
        /WX
    )
    # Enable parallel compilation
    target_compile_options(gemma_session PRIVATE /MP)
else()
    target_compile_options(gemma_session PRIVATE 
        -Wall 
        -Wextra 
        -Wpedantic
        -Wno-unused-parameter
        -Werror
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
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    else()
        message(STATUS "Session tests directory not present; skipping session test targets")
    endif()
endif()

# Build benchmark executable if root requested
if(GEMMA_BUILD_SESSION_BENCH)
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
        conversation_types.h
        session_serialization.h
        i_session_storage.h
        i_session_cache.h
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