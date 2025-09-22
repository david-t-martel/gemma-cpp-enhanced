# Enhanced Gemma CLI Tool CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required dependencies
find_package(Threads REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../gemma.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../gemma.cpp/gemma)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../gemma.cpp/util)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../gemma.cpp/ops)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../gemma.cpp/io)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../gemma.cpp/compression)

# Optional: Add third-party include if highway is installed separately
# find_package(PkgConfig QUIET)
# if(PkgConfig_FOUND)
#     pkg_check_modules(HIGHWAY QUIET libhwy)
#     if(HIGHWAY_FOUND)
#         include_directories(${HIGHWAY_INCLUDE_DIRS})
#     endif()
# endif()

# CLI source files
set(CLI_SOURCES
    main.cpp
    CLIInterface.cpp
)

# CLI headers
set(CLI_HEADERS
    CLIInterface.h
)

# Create the executable
add_executable(gemma_cli ${CLI_SOURCES} ${CLI_HEADERS})

# Link against Gemma libraries
target_link_libraries(gemma_cli
    PRIVATE
    libgemma
    Threads::Threads
)

# Platform-specific settings
if(WIN32)
    # Windows-specific libraries and settings
    target_link_libraries(gemma_cli PRIVATE wsock32 ws2_32)
    
    # Enable Windows console features
    if(MSVC)
        target_compile_definitions(gemma_cli PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(gemma_cli PRIVATE /W3)
    endif()
else()
    # Unix-like systems
    target_link_libraries(gemma_cli PRIVATE dl)
    
    # Enable terminal features
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(gemma_cli PRIVATE -Wall -Wextra)
    endif()
endif()

# Compiler-specific optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(gemma_cli PRIVATE /O2 /DNDEBUG)
    else()
        target_compile_options(gemma_cli PRIVATE -O3 -DNDEBUG)
        if(NOT APPLE)
            target_compile_options(gemma_cli PRIVATE -march=native)
        endif()
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        target_compile_options(gemma_cli PRIVATE /Od /Zi)
    else()
        target_compile_options(gemma_cli PRIVATE -O0 -g3)
    endif()
endif()

# Install target
install(TARGETS gemma_cli
    RUNTIME DESTINATION bin
    COMPONENT cli
)

# Optional: Create convenience target in parent directory
if(TARGET gemma_all)
    add_dependencies(gemma_all gemma_cli)
endif()

# Add custom target for CLI only
add_custom_target(cli
    DEPENDS gemma_cli
    COMMENT "Build Enhanced Gemma CLI"
)

# Documentation
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building Enhanced Gemma CLI in Debug mode")
else()
    message(STATUS "Building Enhanced Gemma CLI in Release mode")
endif()

message(STATUS "Enhanced Gemma CLI will be built as: gemma_cli")
message(STATUS "CLI features: Interactive REPL, Batch mode, Command history, Tab completion")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")

# Add compile definitions for feature detection
target_compile_definitions(gemma_cli PRIVATE
    GEMMA_CLI_VERSION="1.0.0"
    GEMMA_CLI_BUILD_DATE="${CMAKE_BUILD_DATE}"
)

# Enable C++20 features we're using
target_compile_features(gemma_cli PRIVATE cxx_std_20)

# Ensure we can find the gemma headers
get_target_property(GEMMA_INCLUDE_DIRS libgemma INTERFACE_INCLUDE_DIRECTORIES)
if(GEMMA_INCLUDE_DIRS)
    target_include_directories(gemma_cli PRIVATE ${GEMMA_INCLUDE_DIRS})
endif()

# Optional: Add cxxopts for better argument parsing if available
find_package(cxxopts QUIET)
if(cxxopts_FOUND)
    target_link_libraries(gemma_cli PRIVATE cxxopts::cxxopts)
    target_compile_definitions(gemma_cli PRIVATE HAVE_CXXOPTS)
    message(STATUS "Using cxxopts for enhanced argument parsing")
else()
    message(STATUS "cxxopts not found, using basic argument parsing")
endif()

# Optional: Add readline/libedit for better line editing
if(NOT WIN32)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(READLINE QUIET readline)
        if(READLINE_FOUND)
            target_link_libraries(gemma_cli PRIVATE ${READLINE_LIBRARIES})
            target_include_directories(gemma_cli PRIVATE ${READLINE_INCLUDE_DIRS})
            target_compile_definitions(gemma_cli PRIVATE HAVE_READLINE)
            message(STATUS "Using readline for enhanced line editing")
        endif()
    endif()
endif()