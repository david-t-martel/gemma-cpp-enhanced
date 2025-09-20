#include "vulkan_utils.h"
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <sstream>

namespace gemma {
namespace backends {
namespace vulkan {
namespace utils {

bool IsVulkanAvailable() {
    // Try to create a minimal Vulkan instance to check availability
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Vulkan Availability Check";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "Gemma";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = 0;
    create_info.enabledLayerCount = 0;

    VkInstance test_instance;
    VkResult result = vkCreateInstance(&create_info, nullptr, &test_instance);

    if (result == VK_SUCCESS) {
        vkDestroyInstance(test_instance, nullptr);
        return true;
    }

    return false;
}

std::vector<const char*> GetRequiredInstanceExtensions() {
    std::vector<const char*> extensions;

    // Check which extensions are available
    uint32_t extension_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);

    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, available_extensions.data());

    // Helper to check if extension is available
    auto is_extension_available = [&](const char* name) {
        return std::any_of(available_extensions.begin(), available_extensions.end(),
                          [name](const VkExtensionProperties& ext) {
                              return strcmp(ext.extensionName, name) == 0;
                          });
    };

    // Add debug utils extension for validation layers if available
    if (is_extension_available(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // Add portability enumeration extension for macOS compatibility if available
    if (is_extension_available(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }

    return extensions;
}

bool CheckValidationLayerSupport(const std::vector<const char*>& layers) {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    for (const char* layer_name : layers) {
        bool layer_found = false;

        for (const auto& layer_properties : available_layers) {
            if (strcmp(layer_name, layer_properties.layerName) == 0) {
                layer_found = true;
                break;
            }
        }

        if (!layer_found) {
            return false;
        }
    }

    return true;
}

uint32_t ScorePhysicalDevice(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties device_properties;
    VkPhysicalDeviceFeatures device_features;
    VkPhysicalDeviceMemoryProperties memory_properties;

    vkGetPhysicalDeviceProperties(device, &device_properties);
    vkGetPhysicalDeviceFeatures(device, &device_features);
    vkGetPhysicalDeviceMemoryProperties(device, &memory_properties);

    uint32_t score = 0;

    // Device type scoring
    switch (device_properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            score += 1000;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            score += 500;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            score += 100;
            break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            score += 50;
            break;
        default:
            score += 10;
            break;
    }

    // Memory size scoring (prefer devices with more VRAM)
    uint64_t device_local_memory = 0;
    for (uint32_t i = 0; i < memory_properties.memoryHeapCount; i++) {
        if (memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            device_local_memory += memory_properties.memoryHeaps[i].size;
        }
    }

    // Score based on memory size (in GB, capped at 32GB bonus)
    score += std::min(static_cast<uint32_t>(device_local_memory / (1024 * 1024 * 1024)), 32u) * 10;

    // Check for compute support
    uint32_t compute_queue_family = FindComputeQueueFamily(device);
    if (compute_queue_family == UINT32_MAX) {
        return 0; // Device doesn't support compute operations
    }

    // Check API version (prefer newer Vulkan versions)
    if (device_properties.apiVersion >= VK_API_VERSION_1_3) {
        score += 50;
    } else if (device_properties.apiVersion >= VK_API_VERSION_1_2) {
        score += 30;
    } else if (device_properties.apiVersion >= VK_API_VERSION_1_1) {
        score += 10;
    }

    // Check for useful features
    if (device_features.shaderFloat64) {
        score += 10;
    }
    if (device_features.shaderInt64) {
        score += 5;
    }

    // Check compute limits
    score += std::min(device_properties.limits.maxComputeWorkGroupInvocations / 256u, 4u);
    score += std::min(device_properties.limits.maxComputeSharedMemorySize / 16384u, 4u);

    return score;
}

uint32_t FindComputeQueueFamily(VkPhysicalDevice device) {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

    // Look for a queue family that supports compute operations
    for (uint32_t i = 0; i < queue_families.size(); i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return i;
        }
    }

    return UINT32_MAX; // No compute queue family found
}

std::string VkResultToString(VkResult result) {
    switch (result) {
        case VK_SUCCESS:
            return "VK_SUCCESS";
        case VK_NOT_READY:
            return "VK_NOT_READY";
        case VK_TIMEOUT:
            return "VK_TIMEOUT";
        case VK_EVENT_SET:
            return "VK_EVENT_SET";
        case VK_EVENT_RESET:
            return "VK_EVENT_RESET";
        case VK_INCOMPLETE:
            return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:
            return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:
            return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED:
            return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT:
            return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:
            return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER:
            return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS:
            return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED:
            return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL:
            return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN:
            return "VK_ERROR_UNKNOWN";
        case VK_ERROR_OUT_OF_POOL_MEMORY:
            return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE:
            return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_FRAGMENTATION:
            return "VK_ERROR_FRAGMENTATION";
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS:
            return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
        case VK_PIPELINE_COMPILE_REQUIRED:
            return "VK_PIPELINE_COMPILE_REQUIRED";
        case VK_ERROR_SURFACE_LOST_KHR:
            return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
            return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR:
            return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR:
            return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
            return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
        case VK_ERROR_VALIDATION_FAILED_EXT:
            return "VK_ERROR_VALIDATION_FAILED_EXT";
        case VK_ERROR_INVALID_SHADER_NV:
            return "VK_ERROR_INVALID_SHADER_NV";
        case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT:
            return "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
        case VK_ERROR_NOT_PERMITTED_KHR:
            return "VK_ERROR_NOT_PERMITTED_KHR";
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
            return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
        case VK_THREAD_IDLE_KHR:
            return "VK_THREAD_IDLE_KHR";
        case VK_THREAD_DONE_KHR:
            return "VK_THREAD_DONE_KHR";
        case VK_OPERATION_DEFERRED_KHR:
            return "VK_OPERATION_DEFERRED_KHR";
        case VK_OPERATION_NOT_DEFERRED_KHR:
            return "VK_OPERATION_NOT_DEFERRED_KHR";
        default:
            return "VK_UNKNOWN_RESULT_CODE (" + std::to_string(static_cast<int>(result)) + ")";
    }
}

bool CheckDeviceExtensionSupport(VkPhysicalDevice device, const std::vector<const char*>& required_extensions) {
    uint32_t extension_count;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

    std::set<std::string> required_set(required_extensions.begin(), required_extensions.end());

    for (const auto& extension : available_extensions) {
        required_set.erase(extension.extensionName);
    }

    return required_set.empty();
}

VkFormat FindSupportedFormat(VkPhysicalDevice device, const std::vector<VkFormat>& candidates,
                           VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(device, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    return VK_FORMAT_UNDEFINED;
}

uint32_t FindMemoryType(VkPhysicalDevice device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    return UINT32_MAX;
}

bool IsDeviceSuitable(VkPhysicalDevice device, const std::vector<const char*>& required_extensions) {
    VkPhysicalDeviceProperties device_properties;
    VkPhysicalDeviceFeatures device_features;

    vkGetPhysicalDeviceProperties(device, &device_properties);
    vkGetPhysicalDeviceFeatures(device, &device_features);

    // Check if device supports compute operations
    uint32_t compute_queue_family = FindComputeQueueFamily(device);
    if (compute_queue_family == UINT32_MAX) {
        return false;
    }

    // Check device extension support
    if (!CheckDeviceExtensionSupport(device, required_extensions)) {
        return false;
    }

    // Check API version
    if (device_properties.apiVersion < VK_API_VERSION_1_1) {
        return false;
    }

    return true;
}

std::string GetDeviceTypeString(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_OTHER:
            return "Other";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return "Integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return "Discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return "Virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return "CPU";
        default:
            return "Unknown";
    }
}

void PrintDeviceInfo(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceFeatures features;
    VkPhysicalDeviceMemoryProperties memory_properties;

    vkGetPhysicalDeviceProperties(device, &properties);
    vkGetPhysicalDeviceFeatures(device, &features);
    vkGetPhysicalDeviceMemoryProperties(device, &memory_properties);

    std::cout << "Device: " << properties.deviceName << std::endl;
    std::cout << "  Type: " << GetDeviceTypeString(properties.deviceType) << std::endl;
    std::cout << "  API Version: " << VK_VERSION_MAJOR(properties.apiVersion) << "."
              << VK_VERSION_MINOR(properties.apiVersion) << "."
              << VK_VERSION_PATCH(properties.apiVersion) << std::endl;
    std::cout << "  Driver Version: " << properties.driverVersion << std::endl;
    std::cout << "  Vendor ID: 0x" << std::hex << properties.vendorID << std::dec << std::endl;
    std::cout << "  Device ID: 0x" << std::hex << properties.deviceID << std::dec << std::endl;

    // Memory information
    uint64_t total_device_memory = 0;
    uint64_t total_host_memory = 0;

    for (uint32_t i = 0; i < memory_properties.memoryHeapCount; i++) {
        if (memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total_device_memory += memory_properties.memoryHeaps[i].size;
        } else {
            total_host_memory += memory_properties.memoryHeaps[i].size;
        }
    }

    std::cout << "  Device Memory: " << (total_device_memory / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Host Memory: " << (total_host_memory / (1024 * 1024)) << " MB" << std::endl;

    // Compute capabilities
    std::cout << "  Max Compute Work Group Size: ["
              << properties.limits.maxComputeWorkGroupSize[0] << ", "
              << properties.limits.maxComputeWorkGroupSize[1] << ", "
              << properties.limits.maxComputeWorkGroupSize[2] << "]" << std::endl;
    std::cout << "  Max Compute Work Group Invocations: "
              << properties.limits.maxComputeWorkGroupInvocations << std::endl;
    std::cout << "  Max Compute Shared Memory Size: "
              << properties.limits.maxComputeSharedMemorySize << " bytes" << std::endl;

    // Queue families
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

    std::cout << "  Queue Families:" << std::endl;
    for (uint32_t i = 0; i < queue_families.size(); i++) {
        std::cout << "    Family " << i << ": ";
        if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) std::cout << "GRAPHICS ";
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) std::cout << "COMPUTE ";
        if (queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) std::cout << "TRANSFER ";
        if (queue_families[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) std::cout << "SPARSE ";
        std::cout << "(" << queue_families[i].queueCount << " queues)" << std::endl;
    }

    std::cout << "  Score: " << ScorePhysicalDevice(device) << std::endl;
    std::cout << std::endl;
}

VkPipelineShaderStageCreateInfo CreateShaderStageInfo(VkShaderStageFlagBits stage,
                                                     VkShaderModule shader_module,
                                                     const char* entry_point) {
    VkPipelineShaderStageCreateInfo shader_stage_info{};
    shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_info.stage = stage;
    shader_stage_info.module = shader_module;
    shader_stage_info.pName = entry_point;
    return shader_stage_info;
}

VkDescriptorSetLayoutBinding CreateDescriptorSetLayoutBinding(uint32_t binding,
                                                            VkDescriptorType descriptor_type,
                                                            uint32_t descriptor_count,
                                                            VkShaderStageFlags stage_flags) {
    VkDescriptorSetLayoutBinding layout_binding{};
    layout_binding.binding = binding;
    layout_binding.descriptorType = descriptor_type;
    layout_binding.descriptorCount = descriptor_count;
    layout_binding.stageFlags = stage_flags;
    layout_binding.pImmutableSamplers = nullptr;
    return layout_binding;
}

VkWriteDescriptorSet CreateWriteDescriptorSet(VkDescriptorSet descriptor_set,
                                            uint32_t binding,
                                            VkDescriptorType descriptor_type,
                                            uint32_t descriptor_count,
                                            const VkDescriptorBufferInfo* buffer_info) {
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = descriptor_type;
    write.descriptorCount = descriptor_count;
    write.pBufferInfo = buffer_info;
    return write;
}

bool CreateDescriptorPool(VkDevice device, uint32_t max_sets,
                         const std::vector<VkDescriptorPoolSize>& pool_sizes,
                         VkDescriptorPool* descriptor_pool) {
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = max_sets;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    return vkCreateDescriptorPool(device, &pool_info, nullptr, descriptor_pool) == VK_SUCCESS;
}

} // namespace utils
} // namespace vulkan
} // namespace backends
} // namespace gemma