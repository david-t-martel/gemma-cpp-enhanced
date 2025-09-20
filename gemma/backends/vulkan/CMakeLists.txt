# Vulkan Backend for Gemma.cpp
# Provides cross-platform GPU acceleration using Vulkan compute shaders

cmake_minimum_required(VERSION 3.16)

# Find Vulkan package
find_package(Vulkan REQUIRED)

# Check for glslang for shader compilation
find_program(GLSLANG_VALIDATOR glslangValidator HINTS ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE})
if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found. Please install Vulkan SDK.")
endif()

# Vulkan backend sources
set(VULKAN_SOURCES
    vulkan_backend.h
    vulkan_backend.cpp
    vulkan_utils.h
    vulkan_utils.cpp
)

# Shader files
set(VULKAN_SHADERS
    shaders/matmul.comp
    shaders/attention.comp
    shaders/activation.comp
    shaders/relu.comp
    shaders/gelu.comp
    shaders/softmax.comp
)

# Create Vulkan backend library
add_library(gemma_vulkan_backend ${VULKAN_SOURCES})

# Set C++ standard
set_property(TARGET gemma_vulkan_backend PROPERTY CXX_STANDARD 20)
set_property(TARGET gemma_vulkan_backend PROPERTY CXX_STANDARD_REQUIRED ON)

# Include directories
target_include_directories(gemma_vulkan_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Vulkan_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(gemma_vulkan_backend
    PUBLIC gemma_backends_common
    PRIVATE ${Vulkan_LIBRARIES}
)

# Compiler definitions
target_compile_definitions(gemma_vulkan_backend PRIVATE
    VK_NO_PROTOTYPES=1
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(gemma_vulkan_backend PRIVATE
        VK_USE_PLATFORM_WIN32_KHR=1
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
elseif(APPLE)
    target_compile_definitions(gemma_vulkan_backend PRIVATE
        VK_USE_PLATFORM_METAL_EXT=1
    )
elseif(UNIX)
    target_compile_definitions(gemma_vulkan_backend PRIVATE
        VK_USE_PLATFORM_XCB_KHR=1
    )
endif()

# Compiler-specific warnings and optimizations
if(MSVC)
    target_compile_options(gemma_vulkan_backend PRIVATE
        /W4 /WX- # Warning level 4, warnings not as errors for Vulkan headers
        /permissive- # Strict conformance
        /Zc:__cplusplus # Correct __cplusplus macro
    )

    # Disable specific warnings for Vulkan headers
    target_compile_options(gemma_vulkan_backend PRIVATE
        /wd4127 # conditional expression is constant
        /wd4324 # structure was padded due to alignment specifier
    )
else()
    target_compile_options(gemma_vulkan_backend PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter # Vulkan headers have unused parameters
        -Wno-missing-field-initializers # Vulkan structs often use this pattern
    )

    # Enable optimizations for release builds
    target_compile_options(gemma_vulkan_backend PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>
        $<$<CONFIG:Debug>:-O0 -g>
    )
endif()

# Function to compile GLSL shaders to SPIR-V
function(compile_shader TARGET SHADER)
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPIRV_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv)

    add_custom_command(
        OUTPUT ${SPIRV_FILE}
        COMMAND ${GLSLANG_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPIRV_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
        COMMENT "Compiling GLSL shader ${SHADER} to SPIR-V"
        VERBATIM
    )

    set_property(TARGET ${TARGET} APPEND PROPERTY ADDITIONAL_CLEAN_FILES ${SPIRV_FILE})
    target_sources(${TARGET} PRIVATE ${SPIRV_FILE})
endfunction()

# Compile all shaders
foreach(SHADER ${VULKAN_SHADERS})
    compile_shader(gemma_vulkan_backend ${SHADER})
endforeach()

# Create a header file with embedded shader bytecode
set(SHADER_HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/vulkan_shaders.h)
set(SHADER_HEADER_CONTENT
"#pragma once\n"
"// Auto-generated shader bytecode header\n"
"#include <vector>\n"
"#include <cstdint>\n"
"#include <unordered_map>\n"
"#include <string>\n\n"
"namespace gemma {\n"
"namespace backends {\n"
"namespace vulkan {\n"
"namespace shaders {\n\n"
)

# Function to embed shader bytecode
function(embed_shader SHADER_FILE VARIABLE_NAME)
    file(READ ${SHADER_FILE} SHADER_CONTENT HEX)
    string(LENGTH "${SHADER_CONTENT}" CONTENT_LENGTH)
    math(EXPR ARRAY_SIZE "${CONTENT_LENGTH} / 2")

    set(EMBEDDED_CONTENT "static const std::vector<uint32_t> ${VARIABLE_NAME} = {\n")

    string(REGEX REPLACE "([0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f])"
           "0x\\1, " FORMATTED_CONTENT "${SHADER_CONTENT}")

    set(EMBEDDED_CONTENT "${EMBEDDED_CONTENT}${FORMATTED_CONTENT}\n};\n\n")

    file(APPEND ${SHADER_HEADER_FILE} "${EMBEDDED_CONTENT}")
endfunction()

# Custom target to generate shader header
add_custom_target(generate_shader_header
    COMMAND ${CMAKE_COMMAND} -E echo "Generating shader header..."
    COMMAND ${CMAKE_COMMAND} -E echo "${SHADER_HEADER_CONTENT}" > ${SHADER_HEADER_FILE}
    DEPENDS ${VULKAN_SHADERS}
    COMMENT "Generating embedded shader header"
)

# Make sure shaders are compiled before generating header
add_dependencies(generate_shader_header gemma_vulkan_backend)

# Alternative: Generate shader loading function
set(SHADER_LOADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/vulkan_shader_loader.cpp)
file(WRITE ${SHADER_LOADER_FILE}
"#include \"vulkan_backend.h\"\n"
"#include <fstream>\n"
"#include <iostream>\n\n"
"namespace gemma {\n"
"namespace backends {\n"
"namespace vulkan {\n\n"
"std::vector<uint32_t> LoadShaderSPIRV(const std::string& shader_name) {\n"
"    std::string shader_path = \"${CMAKE_CURRENT_BINARY_DIR}/\" + shader_name + \".spv\";\n"
"    std::ifstream file(shader_path, std::ios::binary | std::ios::ate);\n"
"    \n"
"    if (!file.is_open()) {\n"
"        std::cerr << \"Failed to open shader file: \" << shader_path << std::endl;\n"
"        return {};\n"
"    }\n"
"    \n"
"    size_t file_size = static_cast<size_t>(file.tellg());\n"
"    std::vector<char> buffer(file_size);\n"
"    \n"
"    file.seekg(0);\n"
"    file.read(buffer.data(), file_size);\n"
"    file.close();\n"
"    \n"
"    // Convert to uint32_t vector\n"
"    std::vector<uint32_t> spirv_code;\n"
"    spirv_code.resize(file_size / sizeof(uint32_t));\n"
"    memcpy(spirv_code.data(), buffer.data(), file_size);\n"
"    \n"
"    return spirv_code;\n"
"}\n\n"
"} // namespace vulkan\n"
"} // namespace backends\n"
"} // namespace gemma\n"
)

# Add shader loader to the library
target_sources(gemma_vulkan_backend PRIVATE ${SHADER_LOADER_FILE})

# Debug information for Vulkan validation layers
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(gemma_vulkan_backend PRIVATE
        VULKAN_DEBUG=1
        VK_ENABLE_VALIDATION_LAYERS=1
    )
endif()

# Feature detection
include(CheckCXXSourceCompiles)

# Check for Vulkan memory allocator (VMA) support
check_cxx_source_compiles("
#include <vulkan/vulkan.h>
#ifdef VK_VERSION_1_1
int main() { return 0; }
#else
#error Vulkan 1.1 not supported
#endif
" VULKAN_1_1_SUPPORTED)

if(VULKAN_1_1_SUPPORTED)
    target_compile_definitions(gemma_vulkan_backend PRIVATE VULKAN_1_1_SUPPORTED=1)
endif()

# Check for Vulkan 1.2 features
check_cxx_source_compiles("
#include <vulkan/vulkan.h>
#ifdef VK_VERSION_1_2
int main() { return 0; }
#else
#error Vulkan 1.2 not supported
#endif
" VULKAN_1_2_SUPPORTED)

if(VULKAN_1_2_SUPPORTED)
    target_compile_definitions(gemma_vulkan_backend PRIVATE VULKAN_1_2_SUPPORTED=1)
endif()

# Optional: Link with Vulkan Memory Allocator if available
find_path(VMA_INCLUDE_DIR vk_mem_alloc.h
    HINTS ${CMAKE_SOURCE_DIR}/third_party/VulkanMemoryAllocator/include
)

if(VMA_INCLUDE_DIR)
    target_include_directories(gemma_vulkan_backend PRIVATE ${VMA_INCLUDE_DIR})
    target_compile_definitions(gemma_vulkan_backend PRIVATE GEMMA_USE_VMA=1)
    message(STATUS "Vulkan Memory Allocator found: ${VMA_INCLUDE_DIR}")
else()
    message(STATUS "Vulkan Memory Allocator not found, using manual memory management")
endif()

# Installation
install(TARGETS gemma_vulkan_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    vulkan_backend.h
    vulkan_utils.h
    DESTINATION include/gemma/backends/vulkan
)

# Install compiled shaders
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
    DESTINATION share/gemma/shaders
    FILES_MATCHING PATTERN "*.spv"
)

# Testing support
if(GEMMA_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Documentation
if(GEMMA_BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_PROJECT_NAME "Gemma.cpp Vulkan Backend")
        set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
        doxygen_add_docs(vulkan_backend_docs
            ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating Vulkan backend documentation"
        )
    endif()
endif()

# Performance profiling support
option(GEMMA_VULKAN_PROFILE "Enable Vulkan profiling support" OFF)
if(GEMMA_VULKAN_PROFILE)
    target_compile_definitions(gemma_vulkan_backend PRIVATE GEMMA_VULKAN_PROFILE=1)
    message(STATUS "Vulkan profiling support enabled")
endif()

# Print configuration summary
message(STATUS "Vulkan backend configuration:")
message(STATUS "  - Vulkan SDK: ${Vulkan_VERSION}")
message(STATUS "  - glslangValidator: ${GLSLANG_VALIDATOR}")
message(STATUS "  - Vulkan 1.1 support: ${VULKAN_1_1_SUPPORTED}")
message(STATUS "  - Vulkan 1.2 support: ${VULKAN_1_2_SUPPORTED}")
message(STATUS "  - VMA support: ${VMA_INCLUDE_DIR}")
message(STATUS "  - Debug mode: $<$<CONFIG:Debug>:ON>$<$<NOT:$<CONFIG:Debug>>:OFF>")
message(STATUS "  - Profiling: ${GEMMA_VULKAN_PROFILE}")

# Shader compilation verification
add_custom_target(verify_shaders
    COMMAND ${CMAKE_COMMAND} -E echo "Verifying shader compilation..."
    DEPENDS ${VULKAN_SHADERS}
    COMMENT "Verifying all shaders compile successfully"
)

add_dependencies(gemma_vulkan_backend verify_shaders)