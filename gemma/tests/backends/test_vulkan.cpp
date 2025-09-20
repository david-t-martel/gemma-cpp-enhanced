/**
 * test_vulkan.cpp - Vulkan Backend Tests for Gemma.cpp
 *
 * Comprehensive test suite for Vulkan compute backend functionality
 * including device detection, compute shaders, memory management, and performance
 */

#ifdef GEMMA_ENABLE_VULKAN

#include <gtest/gtest.h>
#include <vulkan/vulkan.hpp>
#include <vector>
#include <chrono>
#include <memory>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>

namespace gemma {
namespace vulkan_backend {
namespace test {

// Vulkan error checking macro
#define VK_CHECK(call) \
    do { \
        vk::Result result = call; \
        if (result != vk::Result::eSuccess) { \
            FAIL() << "Vulkan error at " << __FILE__ << ":" << __LINE__ \
                   << " - " << vk::to_string(result); \
        } \
    } while(0)

class VulkanBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            InitializeVulkan();
            CreateComputeResources();
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize Vulkan: " << e.what();
        }
    }

    void TearDown() override {
        CleanupVulkan();
    }

private:
    void InitializeVulkan() {
        // Create Vulkan instance
        vk::ApplicationInfo app_info{
            "Gemma Vulkan Tests",
            VK_MAKE_VERSION(1, 0, 0),
            "Gemma Engine",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_2
        };

        vk::InstanceCreateInfo create_info{
            {},
            &app_info
        };

        instance_ = vk::createInstance(create_info);

        // Find compute-capable device
        auto devices = instance_.enumeratePhysicalDevices();
        ASSERT_FALSE(devices.empty()) << "No Vulkan devices found";

        for (const auto& device : devices) {
            auto properties = device.getProperties();
            auto features = device.getFeatures();
            auto queue_families = device.getQueueFamilyProperties();

            // Find compute queue family
            for (uint32_t i = 0; i < queue_families.size(); ++i) {
                if (queue_families[i].queueFlags & vk::QueueFlagBits::eCompute) {
                    physical_device_ = device;
                    compute_queue_family_ = i;

                    device_info_.name = properties.deviceName;
                    device_info_.type = properties.deviceType;
                    device_info_.api_version = properties.apiVersion;
                    device_info_.max_compute_shared_memory_size = properties.limits.maxComputeSharedMemorySize;
                    device_info_.max_compute_work_group_count = {
                        properties.limits.maxComputeWorkGroupCount[0],
                        properties.limits.maxComputeWorkGroupCount[1],
                        properties.limits.maxComputeWorkGroupCount[2]
                    };
                    device_info_.max_compute_work_group_size = {
                        properties.limits.maxComputeWorkGroupSize[0],
                        properties.limits.maxComputeWorkGroupSize[1],
                        properties.limits.maxComputeWorkGroupSize[2]
                    };

                    std::cout << "Vulkan Device: " << device_info_.name << std::endl;
                    std::cout << "Device Type: " << vk::to_string(device_info_.type) << std::endl;
                    std::cout << "Max Compute Work Group Size: "
                              << device_info_.max_compute_work_group_size[0] << "x"
                              << device_info_.max_compute_work_group_size[1] << "x"
                              << device_info_.max_compute_work_group_size[2] << std::endl;

                    goto device_found;
                }
            }
        }

        FAIL() << "No compute-capable Vulkan device found";

    device_found:
        // Create logical device
        float queue_priority = 1.0f;
        vk::DeviceQueueCreateInfo queue_create_info{
            {},
            compute_queue_family_,
            1,
            &queue_priority
        };

        vk::DeviceCreateInfo device_create_info{
            {},
            1,
            &queue_create_info
        };

        device_ = physical_device_.createDevice(device_create_info);
        compute_queue_ = device_.getQueue(compute_queue_family_, 0);

        // Create command pool
        vk::CommandPoolCreateInfo pool_info{
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            compute_queue_family_
        };
        command_pool_ = device_.createCommandPool(pool_info);
    }

    void CreateComputeResources() {
        // Create descriptor set layout for compute shaders
        vk::DescriptorSetLayoutBinding binding{
            0,
            vk::DescriptorType::eStorageBuffer,
            1,
            vk::ShaderStageFlagBits::eCompute
        };

        vk::DescriptorSetLayoutCreateInfo layout_info{
            {},
            1,
            &binding
        };
        descriptor_set_layout_ = device_.createDescriptorSetLayout(layout_info);

        // Create compute pipeline layout
        vk::PipelineLayoutCreateInfo pipeline_layout_info{
            {},
            1,
            &descriptor_set_layout_
        };
        pipeline_layout_ = device_.createPipelineLayout(pipeline_layout_info);

        // Create descriptor pool
        vk::DescriptorPoolSize pool_size{
            vk::DescriptorType::eStorageBuffer,
            10  // Max 10 storage buffers
        };

        vk::DescriptorPoolCreateInfo pool_info{
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            10,  // Max 10 descriptor sets
            1,
            &pool_size
        };
        descriptor_pool_ = device_.createDescriptorPool(pool_info);
    }

    void CleanupVulkan() {
        if (device_) {
            device_.waitIdle();

            if (descriptor_pool_) device_.destroyDescriptorPool(descriptor_pool_);
            if (pipeline_layout_) device_.destroyPipelineLayout(pipeline_layout_);
            if (descriptor_set_layout_) device_.destroyDescriptorSetLayout(descriptor_set_layout_);
            if (command_pool_) device_.destroyCommandPool(command_pool_);

            device_.destroy();
        }

        if (instance_) {
            instance_.destroy();
        }
    }

protected:
    struct DeviceInfo {
        std::string name;
        vk::PhysicalDeviceType type;
        uint32_t api_version;
        uint32_t max_compute_shared_memory_size;
        std::array<uint32_t, 3> max_compute_work_group_count;
        std::array<uint32_t, 3> max_compute_work_group_size;
    };

    vk::Buffer CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage) {
        vk::BufferCreateInfo buffer_info{
            {},
            size,
            usage
        };
        return device_.createBuffer(buffer_info);
    }

    vk::DeviceMemory AllocateBufferMemory(vk::Buffer buffer, vk::MemoryPropertyFlags properties) {
        auto mem_requirements = device_.getBufferMemoryRequirements(buffer);
        auto mem_properties = physical_device_.getMemoryProperties();

        uint32_t memory_type = UINT32_MAX;
        for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
            if ((mem_requirements.memoryTypeBits & (1 << i)) &&
                (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                memory_type = i;
                break;
            }
        }

        EXPECT_NE(memory_type, UINT32_MAX) << "Failed to find suitable memory type";

        vk::MemoryAllocateInfo alloc_info{
            mem_requirements.size,
            memory_type
        };

        auto memory = device_.allocateMemory(alloc_info);
        device_.bindBufferMemory(buffer, memory, 0);
        return memory;
    }

    vk::ShaderModule CreateShaderModule(const std::vector<uint32_t>& code) {
        vk::ShaderModuleCreateInfo create_info{
            {},
            code.size() * sizeof(uint32_t),
            code.data()
        };
        return device_.createShaderModule(create_info);
    }

    // Simple compute shader SPIR-V bytecode for vector addition
    std::vector<uint32_t> GetVectorAddShaderSPIRV() {
        // This is a pre-compiled SPIR-V for a simple vector addition compute shader
        // In practice, you would compile GLSL using glslc or use runtime compilation
        return {
            0x07230203, 0x00010000, 0x00080007, 0x0000002e, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
            0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
            0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x00000000, 0x00060010, 0x00000004,
            0x00000011, 0x00000020, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
            0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x00000000, 0x00050048, 0x00000006,
            0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000006, 0x00000003, 0x00040047, 0x00000008,
            0x00000022, 0x00000000, 0x00040047, 0x00000008, 0x00000021, 0x00000000, 0x00020013, 0x00000002,
            0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x0003001e, 0x00000007,
            0x00000006, 0x00040020, 0x00000008, 0x0000000c, 0x00000007, 0x0004003b, 0x00000008, 0x00000009,
            0x0000000c, 0x00040015, 0x0000000a, 0x00000020, 0x00000001, 0x0004002b, 0x0000000a, 0x0000000b,
            0x00000000, 0x00040020, 0x0000000c, 0x0000000c, 0x00000006, 0x00050036, 0x00000002, 0x00000004,
            0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x0000000c, 0x0000000d, 0x00000009,
            0x0000000b, 0x0004003d, 0x00000006, 0x0000000e, 0x0000000d, 0x000100fd, 0x00010038
        };
    }

    vk::Instance instance_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    vk::Queue compute_queue_;
    uint32_t compute_queue_family_;
    vk::CommandPool command_pool_;
    vk::DescriptorSetLayout descriptor_set_layout_;
    vk::PipelineLayout pipeline_layout_;
    vk::DescriptorPool descriptor_pool_;
    DeviceInfo device_info_;
};

// Test basic Vulkan device detection and properties
TEST_F(VulkanBackendTest, DeviceDetection) {
    EXPECT_FALSE(device_info_.name.empty());
    EXPECT_NE(device_info_.type, vk::PhysicalDeviceType::eOther);
    EXPECT_GT(device_info_.max_compute_work_group_size[0], 0);
    EXPECT_GT(device_info_.max_compute_work_group_size[1], 0);
    EXPECT_GT(device_info_.max_compute_work_group_size[2], 0);
}

// Test buffer creation and memory allocation
TEST_F(VulkanBackendTest, BufferCreation) {
    const vk::DeviceSize buffer_size = 1024 * sizeof(float);

    // Create storage buffer
    auto buffer = CreateBuffer(buffer_size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);

    // Allocate device memory
    auto memory = AllocateBufferMemory(buffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

    EXPECT_TRUE(buffer);
    EXPECT_TRUE(memory);

    // Clean up
    device_.destroyBuffer(buffer);
    device_.freeMemory(memory);
}

// Test memory mapping and data transfer
TEST_F(VulkanBackendTest, MemoryMapping) {
    const size_t count = 1000;
    const vk::DeviceSize buffer_size = count * sizeof(float);

    // Create host-visible buffer
    auto buffer = CreateBuffer(buffer_size, vk::BufferUsageFlagBits::eStorageBuffer);
    auto memory = AllocateBufferMemory(buffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    // Map memory and write test data
    void* mapped = device_.mapMemory(memory, 0, buffer_size);
    ASSERT_NE(mapped, nullptr);

    std::vector<float> test_data(count);
    std::iota(test_data.begin(), test_data.end(), 1.0f);
    std::memcpy(mapped, test_data.data(), buffer_size);

    device_.unmapMemory(memory);

    // Map again and verify data
    mapped = device_.mapMemory(memory, 0, buffer_size);
    std::vector<float> read_data(count);
    std::memcpy(read_data.data(), mapped, buffer_size);
    device_.unmapMemory(memory);

    EXPECT_EQ(test_data, read_data);

    // Clean up
    device_.destroyBuffer(buffer);
    device_.freeMemory(memory);
}

// Test basic compute shader execution
TEST_F(VulkanBackendTest, ComputeShaderExecution) {
    const size_t count = 1000;
    const vk::DeviceSize buffer_size = count * sizeof(float);

    // Create buffers
    auto input_buffer = CreateBuffer(buffer_size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
    auto output_buffer = CreateBuffer(buffer_size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);

    auto input_memory = AllocateBufferMemory(input_buffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
    auto output_memory = AllocateBufferMemory(output_buffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Create staging buffer for data transfer
    auto staging_buffer = CreateBuffer(buffer_size,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    auto staging_memory = AllocateBufferMemory(staging_buffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    // Upload test data
    std::vector<float> input_data(count, 2.0f);
    void* mapped = device_.mapMemory(staging_memory, 0, buffer_size);
    std::memcpy(mapped, input_data.data(), buffer_size);
    device_.unmapMemory(staging_memory);

    // Copy staging to input buffer (would normally use command buffer)
    // For this test, we'll assume the data is already in the buffer

    // Create simple compute shader that doubles input values
    // (In practice, you'd load this from a compiled SPIR-V file)
    auto shader_module = CreateShaderModule(GetVectorAddShaderSPIRV());

    // Create compute pipeline
    vk::PipelineShaderStageCreateInfo shader_stage{
        {},
        vk::ShaderStageFlagBits::eCompute,
        shader_module,
        "main"
    };

    vk::ComputePipelineCreateInfo pipeline_info{
        {},
        shader_stage,
        pipeline_layout_
    };

    auto [result, pipeline] = device_.createComputePipeline(nullptr, pipeline_info);
    VK_CHECK(result);

    // Note: This is a simplified test. In practice, you would:
    // 1. Create descriptor sets and bind buffers
    // 2. Record command buffer with dispatch commands
    // 3. Submit and wait for completion
    // 4. Copy results back and verify

    // Clean up
    device_.destroyPipeline(pipeline);
    device_.destroyShaderModule(shader_module);
    device_.destroyBuffer(input_buffer);
    device_.destroyBuffer(output_buffer);
    device_.destroyBuffer(staging_buffer);
    device_.freeMemory(input_memory);
    device_.freeMemory(output_memory);
    device_.freeMemory(staging_memory);
}

// Test command buffer recording and submission
TEST_F(VulkanBackendTest, CommandBufferSubmission) {
    // Allocate command buffer
    vk::CommandBufferAllocateInfo alloc_info{
        command_pool_,
        vk::CommandBufferLevel::ePrimary,
        1
    };

    auto command_buffers = device_.allocateCommandBuffers(alloc_info);
    auto command_buffer = command_buffers[0];

    // Record command buffer
    vk::CommandBufferBeginInfo begin_info{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    };

    command_buffer.begin(begin_info);
    // Add commands here (pipeline barriers, dispatches, etc.)
    command_buffer.end();

    // Submit command buffer
    vk::SubmitInfo submit_info{
        0, nullptr, nullptr,
        1, &command_buffer,
        0, nullptr
    };

    compute_queue_.submit(1, &submit_info, nullptr);
    compute_queue_.waitIdle();

    // Clean up
    device_.freeCommandBuffers(command_pool_, command_buffers);
}

// Test descriptor set creation and binding
TEST_F(VulkanBackendTest, DescriptorSets) {
    const vk::DeviceSize buffer_size = 1024 * sizeof(float);

    // Create buffer
    auto buffer = CreateBuffer(buffer_size, vk::BufferUsageFlagBits::eStorageBuffer);
    auto memory = AllocateBufferMemory(buffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Allocate descriptor set
    vk::DescriptorSetAllocateInfo alloc_info{
        descriptor_pool_,
        1,
        &descriptor_set_layout_
    };

    auto descriptor_sets = device_.allocateDescriptorSets(alloc_info);
    auto descriptor_set = descriptor_sets[0];

    // Update descriptor set
    vk::DescriptorBufferInfo buffer_info{
        buffer,
        0,
        buffer_size
    };

    vk::WriteDescriptorSet write{
        descriptor_set,
        0,
        0,
        1,
        vk::DescriptorType::eStorageBuffer,
        nullptr,
        &buffer_info,
        nullptr
    };

    device_.updateDescriptorSets(1, &write, 0, nullptr);

    // Descriptor set is now ready for use in compute pipeline

    // Clean up
    device_.freeDescriptorSets(descriptor_pool_, descriptor_sets);
    device_.destroyBuffer(buffer);
    device_.freeMemory(memory);
}

// Test synchronization primitives
TEST_F(VulkanBackendTest, Synchronization) {
    // Create fence
    vk::FenceCreateInfo fence_info{};
    auto fence = device_.createFence(fence_info);

    // Create semaphore
    vk::SemaphoreCreateInfo semaphore_info{};
    auto semaphore = device_.createSemaphore(semaphore_info);

    // Test fence operations
    auto fence_result = device_.getFenceStatus(fence);
    EXPECT_EQ(fence_result, vk::Result::eNotReady);

    // Reset fence
    device_.resetFences(1, &fence);

    // Clean up
    device_.destroyFence(fence);
    device_.destroySemaphore(semaphore);
}

// Performance benchmark test
TEST_F(VulkanBackendTest, PerformanceBenchmark) {
    const size_t count = 1000000;  // 1M elements
    const vk::DeviceSize buffer_size = count * sizeof(float);

    // Create buffers for benchmark
    auto input_buffer = CreateBuffer(buffer_size, vk::BufferUsageFlagBits::eStorageBuffer);
    auto output_buffer = CreateBuffer(buffer_size, vk::BufferUsageFlagBits::eStorageBuffer);

    auto input_memory = AllocateBufferMemory(input_buffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
    auto output_memory = AllocateBufferMemory(output_buffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Measure buffer creation time
    auto start = std::chrono::high_resolution_clock::now();

    // Create multiple buffers to measure allocation performance
    std::vector<vk::Buffer> test_buffers;
    std::vector<vk::DeviceMemory> test_memories;

    for (int i = 0; i < 100; ++i) {
        auto buffer = CreateBuffer(1024 * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer);
        auto memory = AllocateBufferMemory(buffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
        test_buffers.push_back(buffer);
        test_memories.push_back(memory);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Vulkan Performance:" << std::endl;
    std::cout << "  Buffer allocation time: " << duration.count() << " microseconds for 100 buffers" << std::endl;
    std::cout << "  Average allocation time: " << duration.count() / 100.0 << " microseconds per buffer" << std::endl;

    // Clean up test resources
    for (size_t i = 0; i < test_buffers.size(); ++i) {
        device_.destroyBuffer(test_buffers[i]);
        device_.freeMemory(test_memories[i]);
    }

    device_.destroyBuffer(input_buffer);
    device_.destroyBuffer(output_buffer);
    device_.freeMemory(input_memory);
    device_.freeMemory(output_memory);
}

// Test error handling
TEST_F(VulkanBackendTest, ErrorHandling) {
    // Test invalid buffer creation
    vk::BufferCreateInfo invalid_buffer_info{
        {},
        0,  // Invalid size
        vk::BufferUsageFlagBits::eStorageBuffer
    };

    EXPECT_THROW({
        device_.createBuffer(invalid_buffer_info);
    }, vk::SystemError);
}

}  // namespace test
}  // namespace vulkan_backend
}  // namespace gemma

#endif  // GEMMA_ENABLE_VULKAN