# Version.cmake - Versioning and Git Integration for Gemma.cpp
# Generates version information from Git and embeds it into the build

# Get Git information
find_package(Git QUIET)

if(GIT_FOUND)
    # Get commit hash
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short=8 HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Get full commit hash
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH_FULL
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Get branch name
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Check if working directory is clean
    execute_process(
        COMMAND ${GIT_EXECUTABLE} diff-index --quiet HEAD --
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE GIT_DIRTY_RESULT
        ERROR_QUIET
    )

    if(GIT_DIRTY_RESULT EQUAL 0)
        set(GIT_DIRTY "false")
        set(GIT_DIRTY_SUFFIX "")
    else()
        set(GIT_DIRTY "true")
        set(GIT_DIRTY_SUFFIX "-dirty")
    endif()

    # Get latest tag
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Get commit count since last tag
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-list --count ${GIT_TAG}..HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMITS_SINCE_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Get commit date
    execute_process(
        COMMAND ${GIT_EXECUTABLE} log -1 --format=%cd --date=iso-strict
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_DATE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Get committer name
    execute_process(
        COMMAND ${GIT_EXECUTABLE} log -1 --format=%cn
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_AUTHOR_NAME
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
else()
    set(GIT_COMMIT_HASH "unknown")
    set(GIT_COMMIT_HASH_FULL "unknown")
    set(GIT_BRANCH "unknown")
    set(GIT_DIRTY "false")
    set(GIT_DIRTY_SUFFIX "")
    set(GIT_TAG "v0.0.0")
    set(GIT_COMMITS_SINCE_TAG "0")
    set(GIT_COMMIT_DATE "unknown")
    set(GIT_AUTHOR_NAME "unknown")
endif()

# Parse version from tag (assuming format vX.Y.Z)
if(GIT_TAG MATCHES "^v?([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(VERSION_MAJOR ${CMAKE_MATCH_1})
    set(VERSION_MINOR ${CMAKE_MATCH_2})
    set(VERSION_PATCH ${CMAKE_MATCH_3})
else()
    set(VERSION_MAJOR 0)
    set(VERSION_MINOR 1)
    set(VERSION_PATCH 0)
endif()

# Build version string
if(GIT_COMMITS_SINCE_TAG EQUAL 0)
    # On a tag
    set(VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
else()
    # After a tag
    set(VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}+${GIT_COMMITS_SINCE_TAG}")
endif()

# Add pre-release suffix based on branch
if(GIT_BRANCH MATCHES "develop|dev")
    set(VERSION_STRING "${VERSION_STRING}-dev")
elseif(GIT_BRANCH MATCHES "alpha")
    set(VERSION_STRING "${VERSION_STRING}-alpha")
elseif(GIT_BRANCH MATCHES "beta")
    set(VERSION_STRING "${VERSION_STRING}-beta")
elseif(GIT_BRANCH MATCHES "rc")
    set(VERSION_STRING "${VERSION_STRING}-rc")
endif()

# Full version with commit hash
set(VERSION_FULL "${VERSION_STRING}.${GIT_COMMIT_HASH}${GIT_DIRTY_SUFFIX}")

# Build type suffix
string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
if(BUILD_TYPE_LOWER STREQUAL "debug")
    set(BUILD_VARIANT "debug")
elseif(BUILD_TYPE_LOWER STREQUAL "relwithdebinfo")
    set(BUILD_VARIANT "relwithdebinfo")
else()
    set(BUILD_VARIANT "release")
endif()

# Compiler identification
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(COMPILER_NAME "msvc")
    set(COMPILER_VERSION ${MSVC_VERSION})
elseif(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
    set(COMPILER_NAME "icx")
    set(COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(COMPILER_NAME "gcc")
    set(COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(COMPILER_NAME "clang")
    set(COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
else()
    set(COMPILER_NAME "unknown")
    set(COMPILER_VERSION "unknown")
endif()

# Build identifier (hash of build configuration)
string(MD5 BUILD_CONFIG_HASH
    "${CMAKE_BUILD_TYPE}:${CMAKE_CXX_COMPILER_ID}:${CMAKE_SYSTEM_NAME}:${CMAKE_SYSTEM_PROCESSOR}"
)
string(SUBSTRING ${BUILD_CONFIG_HASH} 0 8 BUILD_HASH)

# Create build identifier
set(BUILD_IDENTIFIER "${VERSION_FULL}-${BUILD_VARIANT}-${COMPILER_NAME}-${BUILD_HASH}")

# Build timestamp
string(TIMESTAMP BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ" UTC)

# Generate version header file
configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/version.h.in
    ${CMAKE_BINARY_DIR}/include/gemma/version.h
    @ONLY
)

# Print version information
message(STATUS "====================================")
message(STATUS "Gemma.cpp Version Information")
message(STATUS "====================================")
message(STATUS "Version: ${VERSION_STRING}")
message(STATUS "Full Version: ${VERSION_FULL}")
message(STATUS "Build Identifier: ${BUILD_IDENTIFIER}")
message(STATUS "Git Commit: ${GIT_COMMIT_HASH_FULL}")
message(STATUS "Git Branch: ${GIT_BRANCH}")
message(STATUS "Git Dirty: ${GIT_DIRTY}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${COMPILER_NAME} ${COMPILER_VERSION}")
message(STATUS "Build Hash: ${BUILD_HASH}")
message(STATUS "Build Timestamp: ${BUILD_TIMESTAMP}")
message(STATUS "====================================")

# Export variables for use in parent CMakeLists.txt
set(GEMMA_VERSION_MAJOR ${VERSION_MAJOR} PARENT_SCOPE)
set(GEMMA_VERSION_MINOR ${VERSION_MINOR} PARENT_SCOPE)
set(GEMMA_VERSION_PATCH ${VERSION_PATCH} PARENT_SCOPE)
set(GEMMA_VERSION_STRING ${VERSION_STRING} PARENT_SCOPE)
set(GEMMA_VERSION_FULL ${VERSION_FULL} PARENT_SCOPE)
set(GEMMA_BUILD_IDENTIFIER ${BUILD_IDENTIFIER} PARENT_SCOPE)
set(GEMMA_GIT_COMMIT_HASH ${GIT_COMMIT_HASH} PARENT_SCOPE)
set(GEMMA_GIT_COMMIT_HASH_FULL ${GIT_COMMIT_HASH_FULL} PARENT_SCOPE)
set(GEMMA_GIT_BRANCH ${GIT_BRANCH} PARENT_SCOPE)
set(GEMMA_BUILD_TIMESTAMP ${BUILD_TIMESTAMP} PARENT_SCOPE)
set(GEMMA_BUILD_HASH ${BUILD_HASH} PARENT_SCOPE)
