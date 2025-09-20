#pragma once

/**
 * @file vulkan_backend.h
 * @brief Vulkan GPU backend for Gemma.cpp inference acceleration
 */

#include "../backend_interface.h"
#include <vulkan/vulkan.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <functional>

namespace gemma {
namespace backends {
namespace vulkan {

// Forward declarations
class VulkanDevice;
class VulkanBuffer;
class VulkanPipeline;
class VulkanCommandPool;
class VulkanMemoryAllocator;

/**
 * @brief Vulkan-specific buffer implementation
 */
class VulkanBuffer {
public:
    VulkanBuffer() = default;
    VulkanBuffer(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                 VkDeviceSize size, VkBufferUsageFlags usage);
    ~VulkanBuffer();

    // Move-only semantics
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;
    VulkanBuffer(VulkanBuffer&& other) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;

    bool IsValid() const { return buffer_ != VK_NULL_HANDLE; }
    VkBuffer GetBuffer() const { return buffer_; }
    VkDeviceMemory GetMemory() const { return memory_; }
    VkDeviceSize GetSize() const { return size_; }
    VkBufferUsageFlags GetUsage() const { return usage_; }

    // Map/unmap for host-visible buffers
    void* Map();
    void Unmap();
    bool IsMapped() const { return mapped_ptr_ != nullptr; }

    void Destroy();

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    VkBufferUsageFlags usage_ = 0;
    void* mapped_ptr_ = nullptr;
};

/**
 * @brief Vulkan compute pipeline wrapper
 */
class VulkanPipeline {
public:
    VulkanPipeline() = default;
    ~VulkanPipeline();

    // Move-only semantics
    VulkanPipeline(const VulkanPipeline&) = delete;
    VulkanPipeline& operator=(const VulkanPipeline&) = delete;
    VulkanPipeline(VulkanPipeline&& other) noexcept;
    VulkanPipeline& operator=(VulkanPipeline&& other) noexcept;

    bool Initialize(VkDevice device, const std::vector<uint32_t>& spirv_code,
                   const std::vector<VkDescriptorSetLayoutBinding>& bindings);

    bool IsValid() const { return pipeline_ != VK_NULL_HANDLE; }
    VkPipeline GetPipeline() const { return pipeline_; }
    VkPipelineLayout GetLayout() const { return layout_; }
    VkDescriptorSetLayout GetDescriptorSetLayout() const { return descriptor_set_layout_; }

    void Destroy();

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
};

/**
 * @brief Vulkan device abstraction
 */
class VulkanDevice {
public:
    VulkanDevice() = default;
    ~VulkanDevice();

    bool Initialize(VkPhysicalDevice physical_device, uint32_t queue_family_index);
    void Destroy();

    bool IsValid() const { return device_ != VK_NULL_HANDLE; }
    VkDevice GetDevice() const { return device_; }
    VkQueue GetQueue() const { return queue_; }
    uint32_t GetQueueFamilyIndex() const { return queue_family_index_; }
    VkPhysicalDevice GetPhysicalDevice() const { return physical_device_; }

    // Memory allocation
    std::unique_ptr<VulkanBuffer> CreateBuffer(VkDeviceSize size,
                                             VkBufferUsageFlags usage,
                                             VkMemoryPropertyFlags properties);

    // Shader compilation
    VkShaderModule CreateShaderModule(const std::vector<uint32_t>& spirv_code);

    // Synchronization
    VkSemaphore CreateSemaphore();
    VkFence CreateFence(bool signaled = false);

    // Memory properties
    uint32_t FindMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties);
    VkPhysicalDeviceProperties GetProperties() const { return device_properties_; }
    VkPhysicalDeviceMemoryProperties GetMemoryProperties() const { return memory_properties_; }

private:
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_index_ = UINT32_MAX;
    VkPhysicalDeviceProperties device_properties_{};
    VkPhysicalDeviceMemoryProperties memory_properties_{};
};

/**
 * @brief Vulkan command pool and buffer management
 */
class VulkanCommandPool {
public:
    VulkanCommandPool() = default;
    ~VulkanCommandPool();

    bool Initialize(VkDevice device, uint32_t queue_family_index);
    void Destroy();

    VkCommandBuffer AllocateCommandBuffer();
    void FreeCommandBuffer(VkCommandBuffer cmd_buffer);
    void Reset();

    bool IsValid() const { return command_pool_ != VK_NULL_HANDLE; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> allocated_buffers_;
    std::mutex pool_mutex_;
};

/**
 * @brief Vulkan backend implementation
 */
class VulkanBackend : public BackendInterface {
public:
    VulkanBackend();
    ~VulkanBackend() override;

    // BackendInterface implementation
    std::string GetName() const override { return "Vulkan"; }
    std::string GetVersion() const override;
    bool Initialize() override;
    void Shutdown() override;
    bool IsAvailable() const override;
    bool SupportsCapability(BackendCapability capability) const override;

    int GetDeviceCount() const override;
    bool SetDevice(int device_id) override;
    int GetCurrentDevice() const override { return current_device_id_; }

    BackendBuffer AllocateBuffer(size_t size, size_t alignment = 32) override;
    void FreeBuffer(const BackendBuffer& buffer) override;
    bool CopyToDevice(const BackendBuffer& dst, const void* src, size_t size) override;
    bool CopyFromDevice(void* dst, const BackendBuffer& src, size_t size) override;
    void Synchronize() override;

    BackendMetrics GetMetrics() const override;
    void ResetMetrics() override;

    // Matrix operations
    bool MatrixMultiply(const BackendBuffer& a, const BackendBuffer& b,
                       const BackendBuffer& c, int m, int n, int k,
                       float alpha = 1.0f, float beta = 0.0f) override;

    bool MatrixVectorMultiply(const BackendBuffer& a, const BackendBuffer& x,
                             const BackendBuffer& y, int m, int n) override;

    // Attention operations
    bool ComputeAttention(const BackendBuffer& queries, const BackendBuffer& keys,
                         const BackendBuffer& values, const BackendBuffer& output,
                         int batch_size, int seq_len, int head_dim, int num_heads) override;

    // Activation functions
    bool ApplyReLU(const BackendBuffer& input, const BackendBuffer& output, size_t size) override;
    bool ApplyGELU(const BackendBuffer& input, const BackendBuffer& output, size_t size) override;
    bool ApplySoftmax(const BackendBuffer& input, const BackendBuffer& output, size_t size) override;

    // Vulkan-specific methods
    VkInstance GetInstance() const { return instance_; }
    VulkanDevice* GetCurrentVulkanDevice() const;
    bool SubmitAndWait(VkCommandBuffer cmd_buffer);

private:
    // Vulkan instance and device management
    bool CreateInstance();
    bool EnumerateDevices();
    bool SelectBestDevice();
    void DestroyInstance();

    // Pipeline management
    bool InitializePipelines();
    void DestroyPipelines();

    // Shader loading and compilation
    std::vector<uint32_t> LoadShaderSPIRV(const std::string& shader_name);
    bool CompileGLSLToSPIRV(const std::string& glsl_source,
                           const std::string& entry_point,
                           std::vector<uint32_t>& spirv_code);

    // Memory management
    std::unique_ptr<VulkanBuffer> CreateVulkanBuffer(size_t size,
                                                   VkBufferUsageFlags usage,
                                                   VkMemoryPropertyFlags properties);

    // Command buffer helpers
    VkCommandBuffer BeginSingleTimeCommands();
    void EndSingleTimeCommands(VkCommandBuffer cmd_buffer);

    // Pipeline execution helpers
    bool ExecuteComputePipeline(VulkanPipeline* pipeline,
                               const std::vector<VkBuffer>& buffers,
                               const std::vector<uint32_t>& sizes,
                               uint32_t group_count_x,
                               uint32_t group_count_y = 1,
                               uint32_t group_count_z = 1);

    // Validation layers and debugging
    bool EnableValidationLayers() const;
    void SetupDebugMessenger();
    void DestroyDebugMessenger();

    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);

private:
    // Vulkan core objects
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;

    // Device management
    std::vector<VkPhysicalDevice> physical_devices_;
    std::vector<std::unique_ptr<VulkanDevice>> devices_;
    int current_device_id_ = -1;

    // Command and descriptor management
    std::unique_ptr<VulkanCommandPool> command_pool_;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    // Compute pipelines for different operations
    std::unordered_map<std::string, std::unique_ptr<VulkanPipeline>> pipelines_;

    // Performance tracking
    mutable BackendMetrics metrics_;
    mutable std::mutex metrics_mutex_;

    // Buffer management
    std::vector<std::unique_ptr<VulkanBuffer>> allocated_buffers_;
    std::mutex buffer_mutex_;

    // Configuration
    bool validation_layers_enabled_ = false;
    bool initialized_ = false;

    // Supported extensions and layers
    std::vector<const char*> required_extensions_;
    std::vector<const char*> validation_layers_;
};

// Utility functions
namespace utils {
    /**
     * @brief Check if Vulkan is available on the system
     */
    bool IsVulkanAvailable();

    /**
     * @brief Get Vulkan instance extensions required for the platform
     */
    std::vector<const char*> GetRequiredInstanceExtensions();

    /**
     * @brief Check if validation layers are available
     */
    bool CheckValidationLayerSupport(const std::vector<const char*>& layers);

    /**
     * @brief Score physical device suitability for compute operations
     */
    uint32_t ScorePhysicalDevice(VkPhysicalDevice device);

    /**
     * @brief Find compute queue family index
     */
    uint32_t FindComputeQueueFamily(VkPhysicalDevice device);

    /**
     * @brief Convert Vulkan result to string
     */
    std::string VkResultToString(VkResult result);
}

} // namespace vulkan
} // namespace backends
} // namespace gemma