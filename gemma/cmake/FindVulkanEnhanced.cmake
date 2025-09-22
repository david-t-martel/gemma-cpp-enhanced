# FindVulkanEnhanced.cmake
# Enhanced Vulkan detection with compute shader support and validation layers
#
# This module extends the standard FindVulkan with additional features
# specifically for compute workloads and ML inference
#
# This module sets the following variables:
#
# VulkanEnhanced_FOUND           - True if Vulkan is found
# Vulkan_VERSION                 - Version of Vulkan
# Vulkan_INCLUDE_DIRS            - Include directories
# Vulkan_LIBRARIES               - Vulkan libraries
# Vulkan_GLSLC_EXECUTABLE        - Path to glslc compiler
# Vulkan_DXC_EXECUTABLE          - Path to DirectX shader compiler
# VulkanEnhanced_HAS_VALIDATION  - True if validation layers are available
# VulkanEnhanced_HAS_COMPUTE     - True if compute shaders are supported
# VulkanEnhanced_HAS_SPIRV_TOOLS - True if SPIRV-Tools are available

cmake_minimum_required(VERSION 3.20)

# First, try to find the standard Vulkan SDK
find_package(Vulkan QUIET)

if(Vulkan_FOUND)
    set(VulkanEnhanced_FOUND TRUE)

    # Enhanced shader compiler detection
    find_program(Vulkan_GLSLC_EXECUTABLE
        NAMES glslc
        PATHS
            ${Vulkan_GLSLC_EXECUTABLE}
            $ENV{VULKAN_SDK}/bin
            $ENV{VULKAN_SDK}/Bin
            /usr/bin
            /usr/local/bin
        DOC "Path to glslc shader compiler"
    )

    find_program(Vulkan_DXC_EXECUTABLE
        NAMES dxc
        PATHS
            $ENV{VULKAN_SDK}/bin
            $ENV{VULKAN_SDK}/Bin
            /usr/bin
            /usr/local/bin
        DOC "Path to DirectX shader compiler"
    )

    find_program(Vulkan_SPIRV_AS_EXECUTABLE
        NAMES spirv-as
        PATHS
            $ENV{VULKAN_SDK}/bin
            $ENV{VULKAN_SDK}/Bin
            /usr/bin
            /usr/local/bin
        DOC "Path to SPIRV assembler"
    )

    find_program(Vulkan_SPIRV_DIS_EXECUTABLE
        NAMES spirv-dis
        PATHS
            $ENV{VULKAN_SDK}/bin
            $ENV{VULKAN_SDK}/Bin
            /usr/bin
            /usr/local/bin
        DOC "Path to SPIRV disassembler"
    )

    find_program(Vulkan_SPIRV_OPT_EXECUTABLE
        NAMES spirv-opt
        PATHS
            $ENV{VULKAN_SDK}/bin
            $ENV{VULKAN_SDK}/Bin
            /usr/bin
            /usr/local/bin
        DOC "Path to SPIRV optimizer"
    )

    # Check for SPIRV-Tools availability
    if(Vulkan_SPIRV_AS_EXECUTABLE AND Vulkan_SPIRV_DIS_EXECUTABLE AND Vulkan_SPIRV_OPT_EXECUTABLE)
        set(VulkanEnhanced_HAS_SPIRV_TOOLS TRUE)
        message(STATUS "Found SPIRV-Tools")
    else()
        set(VulkanEnhanced_HAS_SPIRV_TOOLS FALSE)
    endif()

    # Check for validation layers
    if(WIN32)
        set(VALIDATION_LAYER_PATHS
            $ENV{VULKAN_SDK}/Bin
            $ENV{VULKAN_SDK}/bin
            "C:/VulkanSDK/*/Bin"
        )
        set(VALIDATION_LAYER_NAME VkLayer_khronos_validation.dll)
    else()
        set(VALIDATION_LAYER_PATHS
            $ENV{VULKAN_SDK}/lib
            /usr/lib
            /usr/local/lib
            /usr/lib/x86_64-linux-gnu
        )
        set(VALIDATION_LAYER_NAME libVkLayer_khronos_validation.so)
    endif()

    find_file(VULKAN_VALIDATION_LAYER
        NAMES ${VALIDATION_LAYER_NAME}
        PATHS ${VALIDATION_LAYER_PATHS}
        DOC "Path to Vulkan validation layer"
    )

    if(VULKAN_VALIDATION_LAYER)
        set(VulkanEnhanced_HAS_VALIDATION TRUE)
        message(STATUS "Found Vulkan validation layers: ${VULKAN_VALIDATION_LAYER}")
    else()
        set(VulkanEnhanced_HAS_VALIDATION FALSE)
        message(STATUS "Vulkan validation layers not found")
    endif()

    # Check for compute shader support (assume yes if Vulkan 1.0+)
    if(Vulkan_VERSION VERSION_GREATER_EQUAL "1.0")
        set(VulkanEnhanced_HAS_COMPUTE TRUE)
    else()
        set(VulkanEnhanced_HAS_COMPUTE FALSE)
    endif()

    # Enhanced Vulkan target with compute features
    if(NOT TARGET Vulkan::Enhanced)
        add_library(Vulkan::Enhanced INTERFACE IMPORTED)

        # Link to standard Vulkan library
        set_target_properties(Vulkan::Enhanced PROPERTIES
            INTERFACE_LINK_LIBRARIES "Vulkan::Vulkan"
            INTERFACE_COMPILE_DEFINITIONS "VK_ENABLE_BETA_EXTENSIONS"
        )

        # Add validation layer support in debug builds
        if(VulkanEnhanced_HAS_VALIDATION)
            set_target_properties(Vulkan::Enhanced PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS "VK_ENABLE_BETA_EXTENSIONS;VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1"
            )
        endif()
    endif()

    # Utility functions for Vulkan compute development
    function(compile_vulkan_shader target shader_source output_name)
        if(NOT Vulkan_GLSLC_EXECUTABLE)
            message(FATAL_ERROR "glslc not found. Cannot compile Vulkan shaders.")
        endif()

        set(SHADER_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${output_name}.spv")

        add_custom_command(
            OUTPUT ${SHADER_OUTPUT}
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${shader_source} -o ${SHADER_OUTPUT}
            DEPENDS ${shader_source}
            COMMENT "Compiling Vulkan shader: ${shader_source}"
        )

        # Add to target dependencies
        target_sources(${target} PRIVATE ${SHADER_OUTPUT})
    endfunction()

    function(add_vulkan_compute_target target)
        set(sources ${ARGN})
        add_executable(${target} ${sources})

        target_link_libraries(${target} PRIVATE Vulkan::Enhanced)

        # Add compute-specific compile definitions
        target_compile_definitions(${target} PRIVATE
            VK_USE_PLATFORM_WIN32_KHR=$<BOOL:${WIN32}>
            VK_USE_PLATFORM_XLIB_KHR=$<BOOL:${UNIX}>
            VULKAN_COMPUTE_ENABLED
        )

        # Enable C++17 for Vulkan HPP
        target_compile_features(${target} PRIVATE cxx_std_17)
    endfunction()

    # Create a sample compute shader template
    function(create_vulkan_compute_shader_template output_dir)
        set(COMPUTE_SHADER_TEMPLATE "#version 450

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer InputBuffer {
    float input_data[];
};

layout(std430, binding = 1) buffer OutputBuffer {
    float output_data[];
};

layout(push_constant) uniform PushConstants {
    uint data_size;
    float scale_factor;
} pc;

void main() {
    uint index = gl_GlobalInvocationID.x;

    if (index >= pc.data_size) {
        return;
    }

    // Simple compute operation: scale input
    output_data[index] = input_data[index] * pc.scale_factor;
}")

        file(WRITE "${output_dir}/compute_template.comp" ${COMPUTE_SHADER_TEMPLATE})
        message(STATUS "Created Vulkan compute shader template: ${output_dir}/compute_template.comp")
    endfunction()

    # GPU device enumeration utility
    function(create_vulkan_device_info_utility output_dir)
        set(DEVICE_INFO_CODE "#include <vulkan/vulkan.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        vk::Instance instance;
        vk::InstanceCreateInfo createInfo{};
        instance = vk::createInstance(createInfo);

        auto devices = instance.enumeratePhysicalDevices();

        std::cout << \"Found \" << devices.size() << \" Vulkan device(s):\\n\";

        for (size_t i = 0; i < devices.size(); ++i) {
            auto properties = devices[i].getProperties();
            auto features = devices[i].getFeatures();
            auto memoryProperties = devices[i].getMemoryProperties();

            std::cout << \"\\nDevice \" << i << \":\\n\";
            std::cout << \"  Name: \" << properties.deviceName << \"\\n\";
            std::cout << \"  Type: \" << vk::to_string(properties.deviceType) << \"\\n\";
            std::cout << \"  API Version: \" << VK_VERSION_MAJOR(properties.apiVersion)
                      << \".\" << VK_VERSION_MINOR(properties.apiVersion)
                      << \".\" << VK_VERSION_PATCH(properties.apiVersion) << \"\\n\";
            std::cout << \"  Driver Version: \" << properties.driverVersion << \"\\n\";
            std::cout << \"  Compute Queues: \" << (features.geometryShader ? \"Yes\" : \"No\") << \"\\n\";

            uint64_t totalMemory = 0;
            for (uint32_t j = 0; j < memoryProperties.memoryHeapCount; ++j) {
                if (memoryProperties.memoryHeaps[j].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                    totalMemory += memoryProperties.memoryHeaps[j].size;
                }
            }
            std::cout << \"  Device Memory: \" << (totalMemory / 1024 / 1024) << \" MB\\n\";
        }

        instance.destroy();

    } catch (const std::exception& e) {
        std::cerr << \"Error: \" << e.what() << std::endl;
        return 1;
    }

    return 0;
}")

        file(WRITE "${output_dir}/vulkan_device_info.cpp" ${DEVICE_INFO_CODE})
        message(STATUS "Created Vulkan device info utility: ${output_dir}/vulkan_device_info.cpp")
    endfunction()

else()
    set(VulkanEnhanced_FOUND FALSE)

    # Provide helpful installation information
    message(STATUS "Vulkan SDK not found. Please install from:")
    message(STATUS "  https://vulkan.lunarg.com/sdk/home")

    if(WIN32)
        message(STATUS "  Windows: Download and run the installer")
    elseif(APPLE)
        message(STATUS "  macOS: Download the SDK or use 'brew install molten-vk'")
    else()
        message(STATUS "  Linux: Use package manager or download SDK")
        message(STATUS "    Ubuntu/Debian: sudo apt install vulkan-sdk")
        message(STATUS "    Fedora: sudo dnf install vulkan-devel")
        message(STATUS "    Arch: sudo pacman -S vulkan-devel")
    endif()
endif()

# Standard find_package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VulkanEnhanced
    FOUND_VAR VulkanEnhanced_FOUND
    REQUIRED_VARS Vulkan_FOUND
    VERSION_VAR Vulkan_VERSION
)

if(VulkanEnhanced_FOUND)
    message(STATUS "Vulkan Enhanced found: ${Vulkan_VERSION}")
    message(STATUS "Vulkan Include: ${Vulkan_INCLUDE_DIRS}")
    message(STATUS "Vulkan Library: ${Vulkan_LIBRARIES}")

    if(Vulkan_GLSLC_EXECUTABLE)
        message(STATUS "GLSL Compiler: ${Vulkan_GLSLC_EXECUTABLE}")
    endif()

    if(Vulkan_DXC_EXECUTABLE)
        message(STATUS "DXC Compiler: ${Vulkan_DXC_EXECUTABLE}")
    endif()

    message(STATUS "Features:")
    message(STATUS "  ✓ Compute Shaders: ${VulkanEnhanced_HAS_COMPUTE}")
    message(STATUS "  ${VulkanEnhanced_HAS_VALIDATION ? \"✓\" : \"✗\"} Validation Layers: ${VulkanEnhanced_HAS_VALIDATION}")
    message(STATUS "  ${VulkanEnhanced_HAS_SPIRV_TOOLS ? \"✓\" : \"✗\"} SPIRV Tools: ${VulkanEnhanced_HAS_SPIRV_TOOLS}")
endif()

# Mark advanced variables
mark_as_advanced(
    Vulkan_GLSLC_EXECUTABLE
    Vulkan_DXC_EXECUTABLE
    Vulkan_SPIRV_AS_EXECUTABLE
    Vulkan_SPIRV_DIS_EXECUTABLE
    Vulkan_SPIRV_OPT_EXECUTABLE
    VULKAN_VALIDATION_LAYER
)