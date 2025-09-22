# Tools directory CMakeLists.txt
# Contains CLI and other utility tools for Gemma.cpp

cmake_minimum_required(VERSION 3.20)

message(STATUS "Building Gemma.cpp tools")

# Add CLI tool
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cli)
    message(STATUS "Adding Enhanced CLI tool")
    add_subdirectory(cli)
else()
    message(WARNING "CLI directory not found")
endif()

# Add other tools if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
    message(STATUS "Adding benchmark tools")
    add_subdirectory(benchmarks)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/model_converter)
    message(STATUS "Adding model converter tools")
    add_subdirectory(model_converter)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mcp_server)
    message(STATUS "Adding MCP server tools")
    add_subdirectory(mcp_server)
endif()

# Create a convenience target for all tools
add_custom_target(tools
    COMMENT "Build all Gemma.cpp tools"
)

# Add dependencies for tools that exist
if(TARGET gemma_cli)
    add_dependencies(tools gemma_cli)
endif()

message(STATUS "Tools configuration complete")