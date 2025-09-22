#pragma once

/**
 * @file vulkan_utils.h
 * @brief Utility functions for Vulkan backend operations
 */

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

namespace gemma {
namespace backends {
namespace vulkan {
namespace utils {

/**
 * @brief Check if Vulkan is available on the system
 * @return true if Vulkan can be initialized
 */
bool IsVulkanAvailable();

/**
 * @brief Get required Vulkan instance extensions for the platform
 * @return Vector of extension names
 */
std::vector<const char*> GetRequiredInstanceExtensions();

/**
 * @brief Check if validation layers are available
 * @param layers Vector of layer names to check
 * @return true if all layers are available
 */
bool CheckValidationLayerSupport(const std::vector<const char*>& layers);

/**
 * @brief Score physical device suitability for compute operations
 * @param device Physical device to score
 * @return Suitability score (higher is better, 0 means unsuitable)
 */
uint32_t ScorePhysicalDevice(VkPhysicalDevice device);

/**
 * @brief Find compute queue family index
 * @param device Physical device to check
 * @return Queue family index, or UINT32_MAX if not found
 */
uint32_t FindComputeQueueFamily(VkPhysicalDevice device);

/**
 * @brief Convert Vulkan result to string
 * @param result VkResult to convert
 * @return String representation of the result
 */
std::string VkResultToString(VkResult result);

/**
 * @brief Check if device supports required extensions
 * @param device Physical device to check
 * @param required_extensions Vector of required extension names
 * @return true if all extensions are supported
 */
bool CheckDeviceExtensionSupport(VkPhysicalDevice device, const std::vector<const char*>& required_extensions);

/**
 * @brief Find supported format from candidates
 * @param device Physical device to check
 * @param candidates Vector of candidate formats
 * @param tiling Image tiling mode
 * @param features Required format features
 * @return Supported format, or VK_FORMAT_UNDEFINED if none found
 */
VkFormat FindSupportedFormat(VkPhysicalDevice device, const std::vector<VkFormat>& candidates,
                           VkImageTiling tiling, VkFormatFeatureFlags features);

/**
 * @brief Find memory type index
 * @param device Physical device to check
 * @param type_filter Memory type filter
 * @param properties Required memory properties
 * @return Memory type index, or UINT32_MAX if not found
 */
uint32_t FindMemoryType(VkPhysicalDevice device, uint32_t type_filter, VkMemoryPropertyFlags properties);

/**
 * @brief Check if device is suitable for our needs
 * @param device Physical device to check
 * @param required_extensions Vector of required extensions
 * @return true if device is suitable
 */
bool IsDeviceSuitable(VkPhysicalDevice device, const std::vector<const char*>& required_extensions = {});

/**
 * @brief Get string representation of device type
 * @param type Device type
 * @return String representation
 */
std::string GetDeviceTypeString(VkPhysicalDeviceType type);

/**
 * @brief Print detailed device information
 * @param device Physical device to print info for
 */
void PrintDeviceInfo(VkPhysicalDevice device);

/**
 * @brief Create shader stage info structure
 * @param stage Shader stage
 * @param shader_module Shader module
 * @param entry_point Entry point function name
 * @return Configured shader stage info
 */
VkPipelineShaderStageCreateInfo CreateShaderStageInfo(VkShaderStageFlagBits stage,
                                                     VkShaderModule shader_module,
                                                     const char* entry_point = "main");

/**
 * @brief Create descriptor set layout binding
 * @param binding Binding index
 * @param descriptor_type Type of descriptor
 * @param descriptor_count Number of descriptors
 * @param stage_flags Shader stages that use this binding
 * @return Configured layout binding
 */
VkDescriptorSetLayoutBinding CreateDescriptorSetLayoutBinding(uint32_t binding,
                                                            VkDescriptorType descriptor_type,
                                                            uint32_t descriptor_count,
                                                            VkShaderStageFlags stage_flags);

/**
 * @brief Create write descriptor set structure
 * @param descriptor_set Target descriptor set
 * @param binding Binding index
 * @param descriptor_type Type of descriptor
 * @param descriptor_count Number of descriptors
 * @param buffer_info Buffer info for storage buffers
 * @return Configured write descriptor set
 */
VkWriteDescriptorSet CreateWriteDescriptorSet(VkDescriptorSet descriptor_set,
                                            uint32_t binding,
                                            VkDescriptorType descriptor_type,
                                            uint32_t descriptor_count,
                                            const VkDescriptorBufferInfo* buffer_info);

/**
 * @brief Create descriptor pool
 * @param device Logical device
 * @param max_sets Maximum number of descriptor sets
 * @param pool_sizes Vector of pool sizes
 * @param descriptor_pool Output descriptor pool
 * @return true if creation successful
 */
bool CreateDescriptorPool(VkDevice device, uint32_t max_sets,
                         const std::vector<VkDescriptorPoolSize>& pool_sizes,
                         VkDescriptorPool* descriptor_pool);

/**
 * @brief Debug callback for validation layers
 */
VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data);

/**
 * @brief Helper macros for Vulkan error checking
 */
#define VK_CHECK(result) \
    do { \
        VkResult vk_result = (result); \
        if (vk_result != VK_SUCCESS) { \
            std::cerr << "Vulkan error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << utils::VkResultToString(vk_result) << std::endl; \
            abort(); \
        } \
    } while (0)

#define VK_CHECK_BOOL(result) \
    do { \
        VkResult vk_result = (result); \
        if (vk_result != VK_SUCCESS) { \
            std::cerr << "Vulkan error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << utils::VkResultToString(vk_result) << std::endl; \
            return false; \
        } \
    } while (0)

} // namespace utils
} // namespace vulkan
} // namespace backends
} // namespace gemma