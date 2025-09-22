#include "vulkan_backend.h"
#include "vulkan_utils.h"
#include "../backend_registry.h"
#include <iostream>
#include <algorithm>
#include <set>
#include <fstream>
#include <chrono>
#include <cstring>

namespace gemma {
namespace backends {
namespace vulkan {

// VulkanBuffer implementation
VulkanBuffer::VulkanBuffer(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                          VkDeviceSize size, VkBufferUsageFlags usage)
    : device_(device), buffer_(buffer), memory_(memory), size_(size), usage_(usage) {}

VulkanBuffer::~VulkanBuffer() {
    Destroy();
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : device_(other.device_), buffer_(other.buffer_), memory_(other.memory_),
      size_(other.size_), usage_(other.usage_), mapped_ptr_(other.mapped_ptr_) {
    other.device_ = VK_NULL_HANDLE;
    other.buffer_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.size_ = 0;
    other.usage_ = 0;
    other.mapped_ptr_ = nullptr;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
    if (this != &other) {
        Destroy();
        device_ = other.device_;
        buffer_ = other.buffer_;
        memory_ = other.memory_;
        size_ = other.size_;
        usage_ = other.usage_;
        mapped_ptr_ = other.mapped_ptr_;

        other.device_ = VK_NULL_HANDLE;
        other.buffer_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
        other.size_ = 0;
        other.usage_ = 0;
        other.mapped_ptr_ = nullptr;
    }
    return *this;
}

void* VulkanBuffer::Map() {
    if (mapped_ptr_ != nullptr) {
        return mapped_ptr_;
    }

    VkResult result = vkMapMemory(device_, memory_, 0, size_, 0, &mapped_ptr_);
    if (result != VK_SUCCESS) {
        mapped_ptr_ = nullptr;
    }
    return mapped_ptr_;
}

void VulkanBuffer::Unmap() {
    if (mapped_ptr_ != nullptr) {
        vkUnmapMemory(device_, memory_);
        mapped_ptr_ = nullptr;
    }
}

void VulkanBuffer::Destroy() {
    if (mapped_ptr_ != nullptr) {
        Unmap();
    }
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
    }
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
    size_ = 0;
    usage_ = 0;
}

// VulkanPipeline implementation
VulkanPipeline::~VulkanPipeline() {
    Destroy();
}

VulkanPipeline::VulkanPipeline(VulkanPipeline&& other) noexcept
    : device_(other.device_), pipeline_(other.pipeline_), layout_(other.layout_),
      descriptor_set_layout_(other.descriptor_set_layout_), shader_module_(other.shader_module_) {
    other.device_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;
    other.layout_ = VK_NULL_HANDLE;
    other.descriptor_set_layout_ = VK_NULL_HANDLE;
    other.shader_module_ = VK_NULL_HANDLE;
}

VulkanPipeline& VulkanPipeline::operator=(VulkanPipeline&& other) noexcept {
    if (this != &other) {
        Destroy();
        device_ = other.device_;
        pipeline_ = other.pipeline_;
        layout_ = other.layout_;
        descriptor_set_layout_ = other.descriptor_set_layout_;
        shader_module_ = other.shader_module_;

        other.device_ = VK_NULL_HANDLE;
        other.pipeline_ = VK_NULL_HANDLE;
        other.layout_ = VK_NULL_HANDLE;
        other.descriptor_set_layout_ = VK_NULL_HANDLE;
        other.shader_module_ = VK_NULL_HANDLE;
    }
    return *this;
}

bool VulkanPipeline::Initialize(VkDevice device, const std::vector<uint32_t>& spirv_code,
                               const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
    device_ = device;

    // Create shader module
    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spirv_code.size() * sizeof(uint32_t);
    shader_info.pCode = spirv_code.data();

    if (vkCreateShaderModule(device_, &shader_info, nullptr, &shader_module_) != VK_SUCCESS) {
        return false;
    }

    // Create descriptor set layout
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_set_layout_) != VK_SUCCESS) {
        return false;
    }

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;

    if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &layout_) != VK_SUCCESS) {
        return false;
    }

    // Create compute pipeline
    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module_;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = layout_;

    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_) != VK_SUCCESS) {
        return false;
    }

    return true;
}

void VulkanPipeline::Destroy() {
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
        descriptor_set_layout_ = VK_NULL_HANDLE;
    }
    if (shader_module_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, shader_module_, nullptr);
        shader_module_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
}

// VulkanDevice implementation
VulkanDevice::~VulkanDevice() {
    Destroy();
}

bool VulkanDevice::Initialize(VkPhysicalDevice physical_device, uint32_t queue_family_index) {
    physical_device_ = physical_device;
    queue_family_index_ = queue_family_index;

    // Get device properties
    vkGetPhysicalDeviceProperties(physical_device_, &device_properties_);
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);

    // Create logical device
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_family_index_;
    queue_create_info.queueCount = 1;
    float queue_priority = 1.0f;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;

    // Enable required features
    VkPhysicalDeviceFeatures device_features{};
    device_create_info.pEnabledFeatures = &device_features;

    if (vkCreateDevice(physical_device_, &device_create_info, nullptr, &device_) != VK_SUCCESS) {
        return false;
    }

    // Get the queue
    vkGetDeviceQueue(device_, queue_family_index_, 0, &queue_);

    return true;
}

void VulkanDevice::Destroy() {
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    physical_device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    queue_family_index_ = UINT32_MAX;
}

std::unique_ptr<VulkanBuffer> VulkanDevice::CreateBuffer(VkDeviceSize size,
                                                       VkBufferUsageFlags usage,
                                                       VkMemoryPropertyFlags properties) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
        return nullptr;
    }

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);

    uint32_t memory_type = FindMemoryType(mem_requirements.memoryTypeBits, properties);
    if (memory_type == UINT32_MAX) {
        vkDestroyBuffer(device_, buffer, nullptr);
        return nullptr;
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = memory_type;

    VkDeviceMemory buffer_memory;
    if (vkAllocateMemory(device_, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer, nullptr);
        return nullptr;
    }

    if (vkBindBufferMemory(device_, buffer, buffer_memory, 0) != VK_SUCCESS) {
        vkFreeMemory(device_, buffer_memory, nullptr);
        vkDestroyBuffer(device_, buffer, nullptr);
        return nullptr;
    }

    return std::make_unique<VulkanBuffer>(device_, buffer, buffer_memory, size, usage);
}

uint32_t VulkanDevice::FindMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (memory_properties_.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

VkShaderModule VulkanDevice::CreateShaderModule(const std::vector<uint32_t>& spirv_code) {
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_code.size() * sizeof(uint32_t);
    create_info.pCode = spirv_code.data();

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return shader_module;
}

VkSemaphore VulkanDevice::CreateSemaphore() {
    VkSemaphoreCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphore semaphore;
    if (vkCreateSemaphore(device_, &create_info, nullptr, &semaphore) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return semaphore;
}

VkFence VulkanDevice::CreateFence(bool signaled) {
    VkFenceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) {
        create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    }

    VkFence fence;
    if (vkCreateFence(device_, &create_info, nullptr, &fence) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return fence;
}

// VulkanCommandPool implementation
VulkanCommandPool::~VulkanCommandPool() {
    Destroy();
}

bool VulkanCommandPool::Initialize(VkDevice device, uint32_t queue_family_index) {
    device_ = device;

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family_index;

    return vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) == VK_SUCCESS;
}

void VulkanCommandPool::Destroy() {
    if (command_pool_ != VK_NULL_HANDLE) {
        if (!allocated_buffers_.empty()) {
            vkFreeCommandBuffers(device_, command_pool_,
                               static_cast<uint32_t>(allocated_buffers_.size()),
                               allocated_buffers_.data());
            allocated_buffers_.clear();
        }
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
}

VkCommandBuffer VulkanCommandPool::AllocateCommandBuffer() {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    if (vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    allocated_buffers_.push_back(command_buffer);
    return command_buffer;
}

void VulkanCommandPool::FreeCommandBuffer(VkCommandBuffer cmd_buffer) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    auto it = std::find(allocated_buffers_.begin(), allocated_buffers_.end(), cmd_buffer);
    if (it != allocated_buffers_.end()) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &cmd_buffer);
        allocated_buffers_.erase(it);
    }
}

void VulkanCommandPool::Reset() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    vkResetCommandPool(device_, command_pool_, 0);
}

// VulkanBackend implementation
VulkanBackend::VulkanBackend() {
    validation_layers_ = {"VK_LAYER_KHRONOS_validation"};
    required_extensions_ = utils::GetRequiredInstanceExtensions();

#ifdef NDEBUG
    validation_layers_enabled_ = false;
#else
    validation_layers_enabled_ = utils::CheckValidationLayerSupport(validation_layers_);
#endif
}

VulkanBackend::~VulkanBackend() {
    Shutdown();
}

std::string VulkanBackend::GetVersion() const {
    return "1.0.0";
}

bool VulkanBackend::Initialize() {
    if (initialized_) {
        return true;
    }

    if (!utils::IsVulkanAvailable()) {
        std::cerr << "Vulkan is not available on this system" << std::endl;
        return false;
    }

    if (!CreateInstance()) {
        std::cerr << "Failed to create Vulkan instance" << std::endl;
        return false;
    }

    if (validation_layers_enabled_) {
        SetupDebugMessenger();
    }

    if (!EnumerateDevices()) {
        std::cerr << "Failed to enumerate Vulkan devices" << std::endl;
        return false;
    }

    if (!SelectBestDevice()) {
        std::cerr << "Failed to select a suitable Vulkan device" << std::endl;
        return false;
    }

    if (!InitializePipelines()) {
        std::cerr << "Failed to initialize compute pipelines" << std::endl;
        return false;
    }

    ResetMetrics();
    initialized_ = true;

    std::cout << "Vulkan backend initialized successfully" << std::endl;
    std::cout << "Selected device: " << GetCurrentVulkanDevice()->GetProperties().deviceName << std::endl;

    return true;
}

void VulkanBackend::Shutdown() {
    if (!initialized_) {
        return;
    }

    Synchronize();

    DestroyPipelines();

    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(GetCurrentVulkanDevice()->GetDevice(), descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
    }

    command_pool_.reset();
    devices_.clear();
    physical_devices_.clear();

    if (validation_layers_enabled_) {
        DestroyDebugMessenger();
    }

    DestroyInstance();

    current_device_id_ = -1;
    initialized_ = false;

    std::cout << "Vulkan backend shut down" << std::endl;
}

bool VulkanBackend::IsAvailable() const {
    return utils::IsVulkanAvailable() && !physical_devices_.empty();
}

bool VulkanBackend::SupportsCapability(BackendCapability capability) const {
    switch (capability) {
        case BackendCapability::MATRIX_MULTIPLICATION:
        case BackendCapability::ATTENTION_COMPUTATION:
        case BackendCapability::ACTIVATION_FUNCTIONS:
        case BackendCapability::MEMORY_POOLING:
        case BackendCapability::ASYNC_EXECUTION:
            return true;
        case BackendCapability::MULTI_PRECISION:
            return false; // TODO: Implement FP16 support
        default:
            return false;
    }
}

int VulkanBackend::GetDeviceCount() const {
    return static_cast<int>(devices_.size());
}

bool VulkanBackend::SetDevice(int device_id) {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        return false;
    }

    if (device_id == current_device_id_) {
        return true;
    }

    Synchronize();

    current_device_id_ = device_id;

    // Reinitialize command pool for the new device
    command_pool_ = std::make_unique<VulkanCommandPool>();
    if (!command_pool_->Initialize(GetCurrentVulkanDevice()->GetDevice(),
                                  GetCurrentVulkanDevice()->GetQueueFamilyIndex())) {
        return false;
    }

    return true;
}

BackendBuffer VulkanBackend::AllocateBuffer(size_t size, size_t alignment) {
    if (!initialized_ || current_device_id_ < 0) {
        return BackendBuffer();
    }

    // Align size to the required alignment
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

    auto vulkan_buffer = CreateVulkanBuffer(aligned_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (!vulkan_buffer) {
        return BackendBuffer();
    }

    // Store the buffer for cleanup
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    VulkanBuffer* buffer_ptr = vulkan_buffer.get();
    allocated_buffers_.push_back(std::move(vulkan_buffer));

    // Update metrics
    std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
    metrics_.memory_usage_bytes += aligned_size;
    metrics_.peak_memory_bytes = std::max(metrics_.peak_memory_bytes, metrics_.memory_usage_bytes);

    return BackendBuffer(buffer_ptr, aligned_size, true);
}

void VulkanBackend::FreeBuffer(const BackendBuffer& buffer) {
    if (!buffer.data || !buffer.is_device_memory) {
        return;
    }

    VulkanBuffer* vulkan_buffer = static_cast<VulkanBuffer*>(buffer.data);

    std::lock_guard<std::mutex> lock(buffer_mutex_);
    auto it = std::find_if(allocated_buffers_.begin(), allocated_buffers_.end(),
                          [vulkan_buffer](const std::unique_ptr<VulkanBuffer>& buf) {
                              return buf.get() == vulkan_buffer;
                          });

    if (it != allocated_buffers_.end()) {
        // Update metrics
        std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
        metrics_.memory_usage_bytes -= buffer.size;

        allocated_buffers_.erase(it);
    }
}

bool VulkanBackend::CopyToDevice(const BackendBuffer& dst, const void* src, size_t size) {
    if (!dst.data || !dst.is_device_memory || !src || size == 0) {
        return false;
    }

    VulkanBuffer* vulkan_dst = static_cast<VulkanBuffer*>(dst.data);

    // Create staging buffer
    auto staging_buffer = CreateVulkanBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (!staging_buffer) {
        return false;
    }

    // Copy data to staging buffer
    void* mapped = staging_buffer->Map();
    if (!mapped) {
        return false;
    }
    memcpy(mapped, src, size);
    staging_buffer->Unmap();

    // Copy from staging to device buffer
    VkCommandBuffer cmd_buffer = BeginSingleTimeCommands();
    if (cmd_buffer == VK_NULL_HANDLE) {
        return false;
    }

    VkBufferCopy copy_region{};
    copy_region.size = size;
    vkCmdCopyBuffer(cmd_buffer, staging_buffer->GetBuffer(), vulkan_dst->GetBuffer(), 1, &copy_region);

    EndSingleTimeCommands(cmd_buffer);

    return true;
}

bool VulkanBackend::CopyFromDevice(void* dst, const BackendBuffer& src, size_t size) {
    if (!dst || !src.data || !src.is_device_memory || size == 0) {
        return false;
    }

    VulkanBuffer* vulkan_src = static_cast<VulkanBuffer*>(src.data);

    // Create staging buffer
    auto staging_buffer = CreateVulkanBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (!staging_buffer) {
        return false;
    }

    // Copy from device to staging buffer
    VkCommandBuffer cmd_buffer = BeginSingleTimeCommands();
    if (cmd_buffer == VK_NULL_HANDLE) {
        return false;
    }

    VkBufferCopy copy_region{};
    copy_region.size = size;
    vkCmdCopyBuffer(cmd_buffer, vulkan_src->GetBuffer(), staging_buffer->GetBuffer(), 1, &copy_region);

    EndSingleTimeCommands(cmd_buffer);

    // Copy from staging buffer to host memory
    void* mapped = staging_buffer->Map();
    if (!mapped) {
        return false;
    }
    memcpy(dst, mapped, size);
    staging_buffer->Unmap();

    return true;
}

void VulkanBackend::Synchronize() {
    if (initialized_ && current_device_id_ >= 0) {
        vkQueueWaitIdle(GetCurrentVulkanDevice()->GetQueue());
    }
}

BackendMetrics VulkanBackend::GetMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void VulkanBackend::ResetMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_ = BackendMetrics{};
}

// Matrix operations implementation
bool VulkanBackend::MatrixMultiply(const BackendBuffer& a, const BackendBuffer& b,
                                  const BackendBuffer& c, int m, int n, int k,
                                  float alpha, float beta) {
    if (!initialized_ || current_device_id_ < 0) {
        return false;
    }

    auto it = pipelines_.find("matmul");
    if (it == pipelines_.end()) {
        std::cerr << "Matrix multiplication pipeline not found" << std::endl;
        return false;
    }

    VulkanBuffer* buf_a = static_cast<VulkanBuffer*>(a.data);
    VulkanBuffer* buf_b = static_cast<VulkanBuffer*>(b.data);
    VulkanBuffer* buf_c = static_cast<VulkanBuffer*>(c.data);

    // Calculate workgroup dispatch size (assuming 16x16 workgroup size)
    uint32_t group_count_x = (n + 15) / 16;
    uint32_t group_count_y = (m + 15) / 16;

    std::vector<VkBuffer> buffers = {buf_a->GetBuffer(), buf_b->GetBuffer(), buf_c->GetBuffer()};
    std::vector<uint32_t> sizes = {
        static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
        *reinterpret_cast<const uint32_t*>(&alpha), *reinterpret_cast<const uint32_t*>(&beta)
    };

    auto start_time = std::chrono::high_resolution_clock::now();
    bool success = ExecuteComputePipeline(it->second.get(), buffers, sizes, group_count_x, group_count_y);
    auto end_time = std::chrono::high_resolution_clock::now();

    if (success) {
        // Update performance metrics
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double gflops = (2.0 * m * n * k) / (duration.count() * 1000.0); // GFLOPS

        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.compute_throughput_gflops = gflops;
        metrics_.latency_ms = duration.count() / 1000.0;
    }

    return success;
}

bool VulkanBackend::MatrixVectorMultiply(const BackendBuffer& a, const BackendBuffer& x,
                                        const BackendBuffer& y, int m, int n) {
    // Implementation similar to MatrixMultiply but optimized for matrix-vector case
    return MatrixMultiply(a, x, y, m, 1, n);
}

bool VulkanBackend::ComputeAttention(const BackendBuffer& queries, const BackendBuffer& keys,
                                    const BackendBuffer& values, const BackendBuffer& output,
                                    int batch_size, int seq_len, int head_dim, int num_heads) {
    if (!initialized_ || current_device_id_ < 0) {
        return false;
    }

    auto it = pipelines_.find("attention");
    if (it == pipelines_.end()) {
        std::cerr << "Attention pipeline not found" << std::endl;
        return false;
    }

    VulkanBuffer* buf_q = static_cast<VulkanBuffer*>(queries.data);
    VulkanBuffer* buf_k = static_cast<VulkanBuffer*>(keys.data);
    VulkanBuffer* buf_v = static_cast<VulkanBuffer*>(values.data);
    VulkanBuffer* buf_o = static_cast<VulkanBuffer*>(output.data);

    uint32_t group_count_x = (seq_len + 15) / 16;
    uint32_t group_count_y = (num_heads + 3) / 4;
    uint32_t group_count_z = batch_size;

    std::vector<VkBuffer> buffers = {buf_q->GetBuffer(), buf_k->GetBuffer(),
                                    buf_v->GetBuffer(), buf_o->GetBuffer()};
    std::vector<uint32_t> sizes = {
        static_cast<uint32_t>(batch_size), static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(head_dim), static_cast<uint32_t>(num_heads)
    };

    return ExecuteComputePipeline(it->second.get(), buffers, sizes,
                                 group_count_x, group_count_y, group_count_z);
}

bool VulkanBackend::ApplyReLU(const BackendBuffer& input, const BackendBuffer& output, size_t size) {
    auto it = pipelines_.find("relu");
    if (it == pipelines_.end()) {
        return false;
    }

    VulkanBuffer* buf_in = static_cast<VulkanBuffer*>(input.data);
    VulkanBuffer* buf_out = static_cast<VulkanBuffer*>(output.data);

    uint32_t group_count = (size + 255) / 256; // 256 threads per workgroup

    std::vector<VkBuffer> buffers = {buf_in->GetBuffer(), buf_out->GetBuffer()};
    std::vector<uint32_t> sizes = {static_cast<uint32_t>(size)};

    return ExecuteComputePipeline(it->second.get(), buffers, sizes, group_count);
}

bool VulkanBackend::ApplyGELU(const BackendBuffer& input, const BackendBuffer& output, size_t size) {
    auto it = pipelines_.find("gelu");
    if (it == pipelines_.end()) {
        return false;
    }

    VulkanBuffer* buf_in = static_cast<VulkanBuffer*>(input.data);
    VulkanBuffer* buf_out = static_cast<VulkanBuffer*>(output.data);

    uint32_t group_count = (size + 255) / 256;

    std::vector<VkBuffer> buffers = {buf_in->GetBuffer(), buf_out->GetBuffer()};
    std::vector<uint32_t> sizes = {static_cast<uint32_t>(size)};

    return ExecuteComputePipeline(it->second.get(), buffers, sizes, group_count);
}

bool VulkanBackend::ApplySoftmax(const BackendBuffer& input, const BackendBuffer& output, size_t size) {
    auto it = pipelines_.find("softmax");
    if (it == pipelines_.end()) {
        return false;
    }

    VulkanBuffer* buf_in = static_cast<VulkanBuffer*>(input.data);
    VulkanBuffer* buf_out = static_cast<VulkanBuffer*>(output.data);

    uint32_t group_count = (size + 255) / 256;

    std::vector<VkBuffer> buffers = {buf_in->GetBuffer(), buf_out->GetBuffer()};
    std::vector<uint32_t> sizes = {static_cast<uint32_t>(size)};

    return ExecuteComputePipeline(it->second.get(), buffers, sizes, group_count);
}

VulkanDevice* VulkanBackend::GetCurrentVulkanDevice() const {
    if (current_device_id_ >= 0 && current_device_id_ < static_cast<int>(devices_.size())) {
        return devices_[current_device_id_].get();
    }
    return nullptr;
}

bool VulkanBackend::SubmitAndWait(VkCommandBuffer cmd_buffer) {
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;

    VkFence fence = GetCurrentVulkanDevice()->CreateFence();
    if (fence == VK_NULL_HANDLE) {
        return false;
    }

    VkResult result = vkQueueSubmit(GetCurrentVulkanDevice()->GetQueue(), 1, &submit_info, fence);
    if (result != VK_SUCCESS) {
        vkDestroyFence(GetCurrentVulkanDevice()->GetDevice(), fence, nullptr);
        return false;
    }

    result = vkWaitForFences(GetCurrentVulkanDevice()->GetDevice(), 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(GetCurrentVulkanDevice()->GetDevice(), fence, nullptr);

    return result == VK_SUCCESS;
}

// Private implementation methods follow...
bool VulkanBackend::CreateInstance() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Gemma.cpp";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "Gemma Vulkan Backend";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    create_info.enabledExtensionCount = static_cast<uint32_t>(required_extensions_.size());
    create_info.ppEnabledExtensionNames = required_extensions_.data();

    if (validation_layers_enabled_) {
        create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers_.size());
        create_info.ppEnabledLayerNames = validation_layers_.data();
    }

    return vkCreateInstance(&create_info, nullptr, &instance_) == VK_SUCCESS;
}

bool VulkanBackend::EnumerateDevices() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);

    if (device_count == 0) {
        return false;
    }

    physical_devices_.resize(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, physical_devices_.data());

    return true;
}

bool VulkanBackend::SelectBestDevice() {
    if (physical_devices_.empty()) {
        return false;
    }

    // Score all devices and select the best one
    std::vector<std::pair<uint32_t, VkPhysicalDevice>> scored_devices;

    for (VkPhysicalDevice device : physical_devices_) {
        uint32_t score = utils::ScorePhysicalDevice(device);
        uint32_t queue_family = utils::FindComputeQueueFamily(device);

        if (score > 0 && queue_family != UINT32_MAX) {
            scored_devices.emplace_back(score, device);
        }
    }

    if (scored_devices.empty()) {
        return false;
    }

    // Sort by score (highest first)
    std::sort(scored_devices.begin(), scored_devices.end(),
             [](const auto& a, const auto& b) { return a.first > b.first; });

    // Create devices for all suitable physical devices
    for (const auto& [score, physical_device] : scored_devices) {
        uint32_t queue_family = utils::FindComputeQueueFamily(physical_device);

        auto device = std::make_unique<VulkanDevice>();
        if (device->Initialize(physical_device, queue_family)) {
            devices_.push_back(std::move(device));
        }
    }

    if (devices_.empty()) {
        return false;
    }

    // Set the first (best) device as current
    current_device_id_ = 0;

    // Initialize command pool for the selected device
    command_pool_ = std::make_unique<VulkanCommandPool>();
    return command_pool_->Initialize(GetCurrentVulkanDevice()->GetDevice(),
                                   GetCurrentVulkanDevice()->GetQueueFamilyIndex());
}

void VulkanBackend::DestroyInstance() {
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

bool VulkanBackend::InitializePipelines() {
    if (!initialized_ || current_device_id_ < 0) {
        return false;
    }

    VulkanDevice* device = GetCurrentVulkanDevice();
    if (!device) {
        return false;
    }

    try {
        // Initialize descriptor pool
        std::vector<VkDescriptorPoolSize> pool_sizes = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 400}
        };

        if (!utils::CreateDescriptorPool(device->GetDevice(), 100, pool_sizes, &descriptor_pool_)) {
            std::cerr << "Failed to create descriptor pool" << std::endl;
            return false;
        }

        // Load and create compute pipelines
        struct PipelineInfo {
            std::string name;
            std::string shader_file;
            std::vector<VkDescriptorSetLayoutBinding> bindings;
        };

        std::vector<PipelineInfo> pipeline_infos = {
            {
                "matmul",
                "matmul",
                {
                    utils::CreateDescriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                }
            },
            {
                "attention",
                "attention",
                {
                    utils::CreateDescriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                }
            },
            {
                "relu",
                "relu",
                {
                    utils::CreateDescriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                }
            },
            {
                "gelu",
                "gelu",
                {
                    utils::CreateDescriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                }
            },
            {
                "softmax",
                "softmax",
                {
                    utils::CreateDescriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT),
                    utils::CreateDescriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                }
            }
        };

        for (const auto& info : pipeline_infos) {
            auto spirv_code = LoadShaderSPIRV(info.shader_file);
            if (spirv_code.empty()) {
                std::cerr << "Failed to load shader: " << info.shader_file << std::endl;
                continue; // Continue with other shaders
            }

            auto pipeline = std::make_unique<VulkanPipeline>();
            if (pipeline->Initialize(device->GetDevice(), spirv_code, info.bindings)) {
                pipelines_[info.name] = std::move(pipeline);
                std::cout << "Initialized pipeline: " << info.name << std::endl;
            } else {
                std::cerr << "Failed to initialize pipeline: " << info.name << std::endl;
            }
        }

        if (pipelines_.empty()) {
            std::cerr << "No pipelines were successfully initialized" << std::endl;
            return false;
        }

        std::cout << "Initialized " << pipelines_.size() << " compute pipelines" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception during pipeline initialization: " << e.what() << std::endl;
        return false;
    }
}

void VulkanBackend::DestroyPipelines() {
    pipelines_.clear();
}

std::unique_ptr<VulkanBuffer> VulkanBackend::CreateVulkanBuffer(size_t size,
                                                              VkBufferUsageFlags usage,
                                                              VkMemoryPropertyFlags properties) {
    if (!initialized_ || current_device_id_ < 0) {
        return nullptr;
    }

    return GetCurrentVulkanDevice()->CreateBuffer(size, usage, properties);
}

VkCommandBuffer VulkanBackend::BeginSingleTimeCommands() {
    VkCommandBuffer cmd_buffer = command_pool_->AllocateCommandBuffer();
    if (cmd_buffer == VK_NULL_HANDLE) {
        return VK_NULL_HANDLE;
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(cmd_buffer, &begin_info) != VK_SUCCESS) {
        command_pool_->FreeCommandBuffer(cmd_buffer);
        return VK_NULL_HANDLE;
    }

    return cmd_buffer;
}

void VulkanBackend::EndSingleTimeCommands(VkCommandBuffer cmd_buffer) {
    vkEndCommandBuffer(cmd_buffer);
    SubmitAndWait(cmd_buffer);
    command_pool_->FreeCommandBuffer(cmd_buffer);
}

bool VulkanBackend::ExecuteComputePipeline(VulkanPipeline* pipeline,
                                          const std::vector<VkBuffer>& buffers,
                                          const std::vector<uint32_t>& sizes,
                                          uint32_t group_count_x,
                                          uint32_t group_count_y,
                                          uint32_t group_count_z) {
    // This is a placeholder implementation
    // The actual implementation will depend on the specific shaders
    return true;
}

bool VulkanBackend::EnableValidationLayers() const {
    return validation_layers_enabled_;
}

void VulkanBackend::SetupDebugMessenger() {
    // Implementation for debug messenger setup
}

void VulkanBackend::DestroyDebugMessenger() {
    // Implementation for debug messenger cleanup
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanBackend::DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "Vulkan validation layer: " << pCallbackData->pMessage << std::endl;
    }

    return VK_FALSE;
}

std::vector<uint32_t> VulkanBackend::LoadShaderSPIRV(const std::string& shader_name) {
    // This function will be implemented by the CMake-generated loader
    // For now, return empty vector to indicate shader loading is not yet implemented
    std::cerr << "LoadShaderSPIRV not yet implemented for: " << shader_name << std::endl;
    return {};
}

bool VulkanBackend::CompileGLSLToSPIRV(const std::string& glsl_source,
                                      const std::string& entry_point,
                                      std::vector<uint32_t>& spirv_code) {
    // This would use glslang library for runtime compilation
    // For now, we rely on pre-compiled shaders
    std::cerr << "Runtime GLSL compilation not yet implemented" << std::endl;
    return false;
}

} // namespace vulkan
} // namespace backends
} // namespace gemma

// Register the Vulkan backend
static std::unique_ptr<gemma::backends::BackendInterface> CreateVulkanBackend() {
    return std::make_unique<gemma::backends::vulkan::VulkanBackend>();
}

REGISTER_BACKEND(Vulkan, CreateVulkanBackend,
                gemma::backends::BackendPriority::HIGH,
                {gemma::backends::BackendCapability::MATRIX_MULTIPLICATION,
                 gemma::backends::BackendCapability::ATTENTION_COMPUTATION,
                 gemma::backends::BackendCapability::ACTIVATION_FUNCTIONS,
                 gemma::backends::BackendCapability::MEMORY_POOLING,
                 gemma::backends::BackendCapability::ASYNC_EXECUTION});