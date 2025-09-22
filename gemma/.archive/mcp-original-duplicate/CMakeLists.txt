# MCP (Model Context Protocol) Server Component
# Provides MCP server interface for Gemma.cpp inference engine

cmake_minimum_required(VERSION 3.16)

# MCP Server sources
set(MCP_SOURCES
    server/mcp_server.cpp
    server/mcp_server.h
    server/inference_handler.cpp
    server/inference_handler.h
    server/model_manager.cpp
    server/model_manager.h
)

# Find required packages
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, using FetchContent")
    include(FetchContent)
    FetchContent_Declare(json 
        GIT_REPOSITORY https://github.com/nlohmann/json.git 
        GIT_TAG v3.11.2
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(json)
endif()

# Find WebSocket library
find_package(websocketpp QUIET)
if(NOT websocketpp_FOUND)
    message(STATUS "websocketpp not found, using FetchContent")
    include(FetchContent)
    FetchContent_Declare(websocketpp
        GIT_REPOSITORY https://github.com/zaphoyd/websocketpp.git
        GIT_TAG 0.8.2
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(websocketpp)
endif()

# Create MCP server library
add_library(gemma_mcp_lib ${MCP_SOURCES})
target_include_directories(gemma_mcp_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/server
    ${CMAKE_CURRENT_SOURCE_DIR}/client
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
)

# Link dependencies
target_link_libraries(gemma_mcp_lib 
    PUBLIC libgemma
    PRIVATE nlohmann_json::nlohmann_json
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(gemma_mcp_lib PRIVATE 
        _WIN32_WINNT=0x0601
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
    target_link_libraries(gemma_mcp_lib PRIVATE ws2_32 wsock32)
endif()

# Create MCP server executable (WebSocket version)
add_executable(gemma_mcp_server server/main.cpp)
target_link_libraries(gemma_mcp_server PRIVATE gemma_mcp_lib)

# Create MCP stdio server executable (for initial testing)
add_executable(gemma_mcp_stdio_server server/mcp_stdio_server.cpp)
target_link_libraries(gemma_mcp_stdio_server PRIVATE gemma_mcp_lib)

# Set C++ standard
set_property(TARGET gemma_mcp_lib PROPERTY CXX_STANDARD 20)
set_property(TARGET gemma_mcp_server PROPERTY CXX_STANDARD 20)
set_property(TARGET gemma_mcp_stdio_server PROPERTY CXX_STANDARD 20)

# Compiler-specific options
target_compile_options(gemma_mcp_lib PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
)

# Installation
install(TARGETS gemma_mcp_server gemma_mcp_stdio_server gemma_mcp_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    server/mcp_server.h
    server/inference_handler.h
    server/model_manager.h
    DESTINATION include/gemma/mcp
)

message(STATUS "MCP server component configured successfully")