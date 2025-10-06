# MCP Server CMake Configuration
# Builds the Gemma MCP (Model Context Protocol) server components as a static
# library plus optional executable wrapper (future extension).

cmake_minimum_required(VERSION 3.16)

# Avoid resetting project if included from root; guard with variable.
if(NOT DEFINED GEMMA_MCP_SUBPROJECT)
    set(GEMMA_MCP_SUBPROJECT ON)
    project(gemma_mcp LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to build standalone MCP server executable later
option(GEMMA_BUILD_MCP_EXECUTABLE "Build standalone MCP server executable" ON)

# Source files
set(GEMMA_MCP_SOURCES
    server/MCPServer.cpp
    server/MCPProtocol.cpp
    server/MCPTransport.cpp
    server/MCPTools.cpp
)

set(GEMMA_MCP_HEADERS
    server/MCPServer.h
    server/MCPProtocol.h
    server/MCPTransport.h
    server/MCPTools.h
)

add_library(gemma_mcp_lib STATIC ${GEMMA_MCP_SOURCES} ${GEMMA_MCP_HEADERS})
set_target_properties(gemma_mcp_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Include parent source directories for core gemma headers if build is nested
if(EXISTS ${CMAKE_SOURCE_DIR}/gemma.cpp/gemma)
    target_include_directories(gemma_mcp_lib PUBLIC
        ${CMAKE_SOURCE_DIR}/gemma.cpp
        ${CMAKE_SOURCE_DIR}/gemma.cpp/gemma
    )
endif()

# Public include for MCP consumers
target_include_directories(gemma_mcp_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/server)

# Link against core libgemma if available (when building within main tree)
if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json QUIET)
    if(NOT TARGET nlohmann_json::nlohmann_json)
        include(FetchContent)
        message(STATUS "nlohmann_json target not found; fetching locally for MCP module")
        FetchContent_Declare(nlohmann_json
            URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        )
        FetchContent_MakeAvailable(nlohmann_json)
    endif()
endif()
if(TARGET libgemma)
    target_link_libraries(gemma_mcp_lib PUBLIC libgemma)
else()
    message(WARNING "libgemma target not found; gemma_mcp_lib will build without core linkage")
endif()
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(gemma_mcp_lib PUBLIC nlohmann_json::nlohmann_json)
endif()

# Platform specific settings
if(WIN32)
    target_compile_definitions(gemma_mcp_lib PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

# Enable stricter warnings (non-fatal) for MCP code
if(MSVC)
    target_compile_options(gemma_mcp_lib PRIVATE /W4 /permissive- /Zc:__cplusplus)
else()
    target_compile_options(gemma_mcp_lib PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# Standalone executable (optional)
if(GEMMA_BUILD_MCP_EXECUTABLE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/server/main.cpp)
    add_executable(gemma_mcp_server server/main.cpp)
    target_link_libraries(gemma_mcp_server PRIVATE gemma_mcp_lib nlohmann_json::nlohmann_json)
    if(MSVC)
        target_compile_options(gemma_mcp_server PRIVATE /W4)
    else()
        target_compile_options(gemma_mcp_server PRIVATE -Wall -Wextra -Wpedantic)
    endif()
else()
    if(GEMMA_BUILD_MCP_EXECUTABLE)
        message(WARNING "GEMMA_BUILD_MCP_EXECUTABLE=ON but server/main.cpp missing; skipping executable")
    endif()
endif()

# Install rules
install(TARGETS gemma_mcp_lib
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

# Install executable if built
if(TARGET gemma_mcp_server)
    install(TARGETS gemma_mcp_server
            RUNTIME DESTINATION bin)
endif()

install(FILES ${GEMMA_MCP_HEADERS} DESTINATION include/gemma/mcp)

# Global property so root build can report MCP availability
set(GEMMA_MCP_AVAILABLE TRUE PARENT_SCOPE)
