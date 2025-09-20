/**
 * @file sycl_backend.cpp
 * @brief Intel SYCL backend implementation for Gemma.cpp
 *
 * This implementation provides hardware acceleration using Intel oneAPI SYCL
 * with support for Intel GPUs, NPUs, and CPU fallback.
 */

#include "sycl_backend.h"
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstring>

namespace gemma {
namespace backends {
namespace sycl {

// SYCL Backend Implementation

SyclBackend::SyclBackend()
    : initialized_(false)
    , profiling_enabled_(false)
    , use_usm_device_(true)
    , current_device_id_(-1)
    , total_allocated_memory_(0)
    , peak_memory_usage_(0)
    , trans_n_(oneapi::mkl::transpose::nontrans)
    , trans_t_(oneapi::mkl::transpose::trans) {

    // Initialize metrics
    metrics_ = {};
}

SyclBackend::~SyclBackend() {
    if (initialized_) {
        Shutdown();
    }
}

std::string SyclBackend::GetVersion() const {
    std::stringstream ss;
    ss << "Intel SYCL " << SYCL_LANGUAGE_VERSION;

    std::string oneapi_version = GetOneAPIVersion();
    if (!oneapi_version.empty()) {
        ss << " (oneAPI " << oneapi_version << ")";
    }

    return ss.str();
}

bool SyclBackend::Initialize() {
    if (initialized_) {
        return true;
    }

    try {
        // Detect available devices
        if (!InitializeDevices()) {
            std::cerr << "SYCL Backend: No compatible devices found" << std::endl;
            return false;
        }

        // Initialize command queues
        if (!InitializeQueues()) {
            std::cerr << "SYCL Backend: Failed to initialize device queues" << std::endl;
            return false;
        }

        // Select best device automatically
        if (!SelectBestDevice()) {
            std::cerr << "SYCL Backend: Failed to select suitable device" << std::endl;
            return false;
        }

        initialized_ = true;
        std::cout << "SYCL Backend initialized with " << available_devices_.size()
                  << " devices, active device: " << GetCurrentDeviceInfo().name << std::endl;

        return true;

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "Initialize");
        return false;
    } catch (const std::exception& e) {
        std::cerr << "SYCL Backend initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void SyclBackend::Shutdown() {
    if (!initialized_) {
        return;
    }

    try {
        // Synchronize all queues
        for (auto& [device_id, queue] : device_queues_) {
            if (queue) {
                queue->wait();
            }
        }

        // Free all tracked memory
        std::lock_guard<std::mutex> lock(memory_mutex_);
        for (auto& [ptr, info] : memory_allocations_) {
            if (ptr) {
                ::sycl::free(ptr, *current_queue_);
            }
        }
        memory_allocations_.clear();
        total_allocated_memory_ = 0;

        // Clear device queues
        device_queues_.clear();
        current_queue_.reset();

        // Clear device info
        available_devices_.clear();
        current_device_id_ = -1;

        initialized_ = false;
        std::cout << "SYCL Backend shutdown complete" << std::endl;

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "Shutdown");
    }
}

bool SyclBackend::IsAvailable() const {
    return initialized_ && !available_devices_.empty() && current_queue_ != nullptr;
}

bool SyclBackend::SupportsCapability(BackendCapability capability) const {
    if (!IsAvailable()) {
        return false;
    }

    switch (capability) {
        case BackendCapability::MATRIX_MULTIPLICATION:
        case BackendCapability::ATTENTION_COMPUTATION:
        case BackendCapability::ACTIVATION_FUNCTIONS:
        case BackendCapability::MEMORY_POOLING:
            return true;
        case BackendCapability::ASYNC_EXECUTION:
            return true; // SYCL supports async execution
        case BackendCapability::MULTI_PRECISION:
            return GetCurrentDeviceInfo().supports_fp16;
        default:
            return false;
    }
}

int SyclBackend::GetDeviceCount() const {
    return static_cast<int>(available_devices_.size());
}

bool SyclBackend::SetDevice(int device_id) {
    if (!initialized_ || device_id < 0 || device_id >= GetDeviceCount()) {
        return false;
    }

    try {
        auto it = device_queues_.find(device_id);
        if (it == device_queues_.end()) {
            return false;
        }

        current_device_id_ = device_id;
        current_queue_ = std::make_unique<::sycl::queue>(*it->second);

        std::cout << "SYCL Backend: Switched to device " << device_id
                  << " (" << GetCurrentDeviceInfo().name << ")" << std::endl;
        return true;

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "SetDevice");
        return false;
    }
}

int SyclBackend::GetCurrentDevice() const {
    return current_device_id_;
}

BackendBuffer SyclBackend::AllocateBuffer(size_t size, size_t alignment) {
    if (!IsAvailable() || size == 0) {
        return BackendBuffer();
    }

    try {
        void* device_ptr = AllocateDeviceMemory(size, alignment);
        if (!device_ptr) {
            return BackendBuffer();
        }

        TrackMemoryAllocation(device_ptr, size, alignment);
        return BackendBuffer(device_ptr, size, true);

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "AllocateBuffer");
        return BackendBuffer();
    }
}

void SyclBackend::FreeBuffer(const BackendBuffer& buffer) {
    if (!IsAvailable() || !buffer.data || !buffer.is_device_memory) {
        return;
    }

    try {
        UntrackMemoryAllocation(buffer.data);
        FreeDeviceMemory(buffer.data);
    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "FreeBuffer");
    }
}

bool SyclBackend::CopyToDevice(const BackendBuffer& dst, const void* src, size_t size) {
    if (!IsAvailable() || !dst.data || !src || size == 0) {
        return false;
    }

    try {
        if (profiling_enabled_) {
            BeginProfiling("CopyToDevice");
        }

        current_queue_->memcpy(dst.data, src, size).wait();

        if (profiling_enabled_) {
            EndProfiling("CopyToDevice", size);
        }

        return CheckDeviceError(*current_queue_, "CopyToDevice");

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "CopyToDevice");
        return false;
    }
}

bool SyclBackend::CopyFromDevice(void* dst, const BackendBuffer& src, size_t size) {
    if (!IsAvailable() || !dst || !src.data || size == 0) {
        return false;
    }

    try {
        if (profiling_enabled_) {
            BeginProfiling("CopyFromDevice");
        }

        current_queue_->memcpy(dst, src.data, size).wait();

        if (profiling_enabled_) {
            EndProfiling("CopyFromDevice", size);
        }

        return CheckDeviceError(*current_queue_, "CopyFromDevice");

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "CopyFromDevice");
        return false;
    }
}

void SyclBackend::Synchronize() {
    if (!IsAvailable()) {
        return;
    }

    try {
        current_queue_->wait();
    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "Synchronize");
    }
}

BackendMetrics SyclBackend::GetMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void SyclBackend::ResetMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_ = {};
    profiling_data_.clear();
}

// Device initialization and management

bool SyclBackend::InitializeDevices() {
    try {
        available_devices_ = DetectDevices();
        return !available_devices_.empty();
    } catch (const std::exception& e) {
        std::cerr << "Failed to detect SYCL devices: " << e.what() << std::endl;
        return false;
    }
}

bool SyclBackend::InitializeQueues() {
    try {
        for (const auto& device_info : available_devices_) {
            auto queue = std::make_unique<::sycl::queue>(
                device_info.device,
                ::sycl::property::queue::enable_profiling{}
            );

            // Test basic queue functionality
            if (!TestDeviceCapabilities(device_info.device)) {
                std::cerr << "Device " << device_info.name << " failed capability test" << std::endl;
                continue;
            }

            device_queues_[device_info.device_id] = std::move(queue);
        }

        return !device_queues_.empty();
    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "InitializeQueues");
        return false;
    }
}

bool SyclBackend::TestDeviceCapabilities(const ::sycl::device& device) {
    try {
        ::sycl::queue test_queue(device);

        // Test basic memory allocation
        constexpr size_t test_size = 1024;
        void* test_ptr = ::sycl::malloc_device(test_size, test_queue);
        if (!test_ptr) {
            return false;
        }

        // Test basic kernel execution
        test_queue.submit([&](::sycl::handler& cgh) {
            cgh.parallel_for<class test_kernel>(
                ::sycl::range<1>(test_size/sizeof(float)),
                [=](::sycl::id<1> idx) {
                    static_cast<float*>(test_ptr)[idx] = static_cast<float>(idx);
                }
            );
        });

        test_queue.wait();
        ::sycl::free(test_ptr, test_queue);

        return true;
    } catch (const ::sycl::exception& e) {
        return false;
    }
}

bool SyclBackend::SelectBestDevice() {
    if (available_devices_.empty()) {
        return false;
    }

    // Prioritize devices: GPU > NPU > CPU
    int best_device_id = -1;
    SyclDeviceType best_type = SyclDeviceType::UNKNOWN;

    for (const auto& device_info : available_devices_) {
        if (device_queues_.find(device_info.device_id) == device_queues_.end()) {
            continue; // Skip devices without working queues
        }

        if (device_info.type == SyclDeviceType::GPU && best_type != SyclDeviceType::GPU) {
            best_device_id = device_info.device_id;
            best_type = device_info.type;
        } else if (device_info.type == SyclDeviceType::NPU &&
                   (best_type != SyclDeviceType::GPU && best_type != SyclDeviceType::NPU)) {
            best_device_id = device_info.device_id;
            best_type = device_info.type;
        } else if (device_info.type == SyclDeviceType::CPU && best_type == SyclDeviceType::UNKNOWN) {
            best_device_id = device_info.device_id;
            best_type = device_info.type;
        }
    }

    if (best_device_id >= 0) {
        return SetDevice(best_device_id);
    }

    return false;
}

SyclDeviceType SyclBackend::GetDeviceType(const ::sycl::device& device) const {
    if (device.is_gpu()) {
        return SyclDeviceType::GPU;
    } else if (device.is_cpu()) {
        return SyclDeviceType::CPU;
    } else {
        // Check for NPU by looking at device name/vendor
        std::string name = device.get_info<::sycl::info::device::name>();
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);

        if (name.find("npu") != std::string::npos ||
            name.find("neural") != std::string::npos) {
            return SyclDeviceType::NPU;
        }
    }

    return SyclDeviceType::UNKNOWN;
}

std::string SyclBackend::GetDeviceName(const ::sycl::device& device) const {
    try {
        return device.get_info<::sycl::info::device::name>();
    } catch (const ::sycl::exception&) {
        return "Unknown Device";
    }
}

const SyclDeviceInfo& SyclBackend::GetCurrentDeviceInfo() const {
    static SyclDeviceInfo invalid_device;

    if (current_device_id_ < 0 ||
        current_device_id_ >= static_cast<int>(available_devices_.size())) {
        return invalid_device;
    }

    return available_devices_[current_device_id_];
}

// Memory management

void* SyclBackend::AllocateDeviceMemory(size_t size, size_t alignment) {
    if (!current_queue_) {
        return nullptr;
    }

    try {
        void* ptr = nullptr;

        if (use_usm_device_) {
            // Use device USM for better performance
            ptr = ::sycl::aligned_alloc_device(alignment, size, *current_queue_);
        } else {
            // Use shared USM for easier access
            ptr = ::sycl::aligned_alloc_shared(alignment, size, *current_queue_);
        }

        return ptr;
    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "AllocateDeviceMemory");
        return nullptr;
    }
}

void SyclBackend::FreeDeviceMemory(void* ptr) {
    if (!current_queue_ || !ptr) {
        return;
    }

    try {
        ::sycl::free(ptr, *current_queue_);
    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "FreeDeviceMemory");
    }
}

void SyclBackend::TrackMemoryAllocation(void* ptr, size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(memory_mutex_);

    SyclMemoryInfo info;
    info.device_ptr = ptr;
    info.size = size;
    info.alignment = alignment;
    info.alloc_type = use_usm_device_ ? ::sycl::usm::alloc::device : ::sycl::usm::alloc::shared;
    info.device_id = current_device_id_;
    info.alloc_time = std::chrono::high_resolution_clock::now();

    memory_allocations_[ptr] = info;
    total_allocated_memory_ += size;
    peak_memory_usage_ = std::max(peak_memory_usage_, total_allocated_memory_);
}

void SyclBackend::UntrackMemoryAllocation(void* ptr) {
    std::lock_guard<std::mutex> lock(memory_mutex_);

    auto it = memory_allocations_.find(ptr);
    if (it != memory_allocations_.end()) {
        total_allocated_memory_ -= it->second.size;
        memory_allocations_.erase(it);
    }
}

// Performance tracking

void SyclBackend::BeginProfiling(const std::string& operation_name) {
    if (!profiling_enabled_) {
        return;
    }

    SyclProfileData profile;
    profile.operation_name = operation_name;
    profile.start_time = std::chrono::high_resolution_clock::now();
    profile.device_id = current_device_id_;

    std::lock_guard<std::mutex> lock(metrics_mutex_);
    profiling_data_.push_back(profile);
}

void SyclBackend::EndProfiling(const std::string& operation_name, size_t memory_transferred, size_t flops) {
    if (!profiling_enabled_) {
        return;
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(metrics_mutex_);

    // Find the matching profile entry
    auto it = std::find_if(profiling_data_.rbegin(), profiling_data_.rend(),
        [&operation_name](const SyclProfileData& data) {
            return data.operation_name == operation_name &&
                   data.end_time == std::chrono::high_resolution_clock::time_point{};
        });

    if (it != profiling_data_.rend()) {
        it->end_time = end_time;
        it->memory_transferred = memory_transferred;
        it->flops_performed = flops;

        // Update metrics
        double execution_time_ms = std::chrono::duration<double, std::milli>(
            end_time - it->start_time).count();
        UpdateMetrics(execution_time_ms, memory_transferred, flops);
    }
}

void SyclBackend::UpdateMetrics(double execution_time_ms, size_t memory_transferred, size_t flops) {
    metrics_.latency_ms = execution_time_ms;
    metrics_.memory_usage_bytes = total_allocated_memory_;
    metrics_.peak_memory_bytes = peak_memory_usage_;

    if (execution_time_ms > 0) {
        if (flops > 0) {
            metrics_.compute_throughput_gflops = (flops / 1e9) / (execution_time_ms / 1000.0);
        }
        if (memory_transferred > 0) {
            metrics_.memory_bandwidth_gbps = (memory_transferred / 1e9) / (execution_time_ms / 1000.0);
        }
    }
}

// Error handling

void SyclBackend::HandleSyclException(const ::sycl::exception& e, const std::string& operation) const {
    std::cerr << "SYCL Exception in " << operation << ": " << e.what() << std::endl;
    if (e.code() != ::sycl::errc::success) {
        std::cerr << "Error code: " << static_cast<int>(e.code().value()) << std::endl;
    }
}

bool SyclBackend::CheckDeviceError(const ::sycl::queue& queue, const std::string& operation) const {
    try {
        // Wait for any asynchronous errors
        queue.wait_and_throw();
        return true;
    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, operation);
        return false;
    }
}

// Static utility functions

std::vector<SyclDeviceInfo> SyclBackend::DetectDevices() {
    std::vector<SyclDeviceInfo> devices;
    int device_id = 0;

    try {
        // Get all available platforms
        auto platforms = ::sycl::platform::get_platforms();
        
        std::cout << "SYCL Backend: Scanning " << platforms.size() << " platforms for devices..." << std::endl;

        for (const auto& platform : platforms) {
            try {
                std::string platform_name = platform.get_info<::sycl::info::platform::name>();
                std::string platform_vendor = platform.get_info<::sycl::info::platform::vendor>();
                
                std::cout << "SYCL Backend: Platform - " << platform_name 
                          << " (Vendor: " << platform_vendor << ")" << std::endl;

                auto platform_devices = platform.get_devices();
                
                // Prioritize Intel platforms but don't exclude others entirely
                bool is_intel_platform = platform_vendor.find("Intel") != std::string::npos;

                for (const auto& device : platform_devices) {
                    if (!SupportsRequiredExtensions(device)) {
                        std::cout << "SYCL Backend: Device lacks required extensions, skipping" << std::endl;
                        continue;
                    }

                    SyclDeviceInfo info;
                    info.device_id = device_id++;
                    info.device = device;
                    info.name = device.get_info<::sycl::info::device::name>();
                    info.vendor = device.get_info<::sycl::info::device::vendor>();
                    info.max_memory_bytes = device.get_info<::sycl::info::device::global_mem_size>();
                    info.max_work_group_size = device.get_info<::sycl::info::device::max_work_group_size>();

                    // Check for Intel-specific features
                    info.supports_fp16 = device.has(::sycl::aspect::fp16);
                    info.supports_dp4a = device.has(::sycl::aspect::int64_base_atomics);
                    info.supports_unified_memory = device.has(::sycl::aspect::usm_device_allocations) ||
                                                   device.has(::sycl::aspect::usm_shared_allocations);

                    try {
                        info.driver_version = device.get_info<::sycl::info::device::driver_version>();
                    } catch (...) {
                        info.driver_version = "Unknown";
                    }

                    // Enhanced device type detection with Intel NPU support
                    std::string name_lower = info.name;
                    std::string vendor_lower = info.vendor;
                    std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
                    std::transform(vendor_lower.begin(), vendor_lower.end(), vendor_lower.begin(), ::tolower);

                    if (device.is_gpu()) {
                        info.type = SyclDeviceType::GPU;
                        std::cout << "SYCL Backend: Found GPU - " << info.name << std::endl;
                    } else if (device.is_cpu()) {
                        info.type = SyclDeviceType::CPU;
                        std::cout << "SYCL Backend: Found CPU - " << info.name << std::endl;
                    } else {
                        // Enhanced NPU detection for Intel Core Ultra processors
                        if (name_lower.find("npu") != std::string::npos ||
                            name_lower.find("neural") != std::string::npos ||
                            name_lower.find("ai boost") != std::string::npos ||
                            name_lower.find("intel(r) ai boost") != std::string::npos ||
                            (vendor_lower.find("intel") != std::string::npos && 
                             (name_lower.find("core ultra") != std::string::npos ||
                              name_lower.find("meteor lake") != std::string::npos ||
                              name_lower.find("arrow lake") != std::string::npos))) {
                            info.type = SyclDeviceType::NPU;
                            std::cout << "SYCL Backend: Found NPU - " << info.name << std::endl;
                        } else {
                            info.type = SyclDeviceType::UNKNOWN;
                            std::cout << "SYCL Backend: Unknown device type - " << info.name << std::endl;
                        }
                    }

                    // Additional Intel GPU detection improvements
                    if (info.type == SyclDeviceType::GPU && vendor_lower.find("intel") != std::string::npos) {
                        // Detect specific Intel GPU architectures
                        if (name_lower.find("arc") != std::string::npos) {
                            std::cout << "SYCL Backend: Intel Arc GPU detected" << std::endl;
                        } else if (name_lower.find("xe") != std::string::npos) {
                            std::cout << "SYCL Backend: Intel Xe GPU detected" << std::endl;
                        } else if (name_lower.find("iris") != std::string::npos) {
                            std::cout << "SYCL Backend: Intel Iris GPU detected" << std::endl;
                        }
                    }

                    // Skip non-Intel devices if we already have Intel devices, unless forced
                    if (!is_intel_platform && vendor_lower.find("intel") == std::string::npos) {
                        bool has_intel_devices = std::any_of(devices.begin(), devices.end(),
                            [](const SyclDeviceInfo& d) {
                                std::string v = d.vendor;
                                std::transform(v.begin(), v.end(), v.begin(), ::tolower);
                                return v.find("intel") != std::string::npos;
                            });
                        
                        if (has_intel_devices) {
                            std::cout << "SYCL Backend: Skipping non-Intel device (Intel devices already found): " 
                                      << info.name << std::endl;
                            continue;
                        }
                    }

                    devices.push_back(info);
                    std::cout << "SYCL Backend: Added device [" << info.device_id << "] " 
                              << info.name << " (" << info.vendor << ")" << std::endl;
                }
            } catch (const ::sycl::exception& e) {
                std::cerr << "SYCL Backend: Error processing platform: " << e.what() << std::endl;
                continue;
            }
        }
    } catch (const ::sycl::exception& e) {
        std::cerr << "SYCL Backend: Failed to detect SYCL devices: " << e.what() << std::endl;
    }

    std::cout << "SYCL Backend: Total compatible devices found: " << devices.size() << std::endl;
    return devices;
}

bool SyclBackend::IsIntelDevice(const ::sycl::device& device) {
    try {
        std::string vendor = device.get_info<::sycl::info::device::vendor>();
        std::string name = device.get_info<::sycl::info::device::name>();
        
        // Convert to lowercase for comparison
        std::transform(vendor.begin(), vendor.end(), vendor.begin(), ::tolower);
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        
        // Check vendor field
        if (vendor.find("intel") != std::string::npos) {
            return true;
        }
        
        // Check device name for Intel-specific markers
        if (name.find("intel") != std::string::npos ||
            name.find("arc") != std::string::npos ||
            name.find("iris") != std::string::npos ||
            name.find("xe") != std::string::npos ||
            name.find("ai boost") != std::string::npos) {
            return true;
        }
        
        return false;
    } catch (...) {
        return false;
    }
}

bool SyclBackend::SupportsRequiredExtensions(const ::sycl::device& device) {
    try {
        // Check for basic USM support (required for Gemma.cpp)
        bool has_usm = device.has(::sycl::aspect::usm_device_allocations) ||
                       device.has(::sycl::aspect::usm_shared_allocations);
        
        if (!has_usm) {
            std::cout << "SYCL Backend: Device lacks USM support: " 
                      << device.get_info<::sycl::info::device::name>() << std::endl;
            return false;
        }
        
        // Additional checks for optimal performance
        bool has_fp32 = true; // FP32 is mandatory in SYCL
        
        // Log device capabilities
        std::string device_name = device.get_info<::sycl::info::device::name>();
        std::cout << "SYCL Backend: Device capabilities for " << device_name << ":" << std::endl;
        std::cout << "  USM Device: " << (device.has(::sycl::aspect::usm_device_allocations) ? "Yes" : "No") << std::endl;
        std::cout << "  USM Shared: " << (device.has(::sycl::aspect::usm_shared_allocations) ? "Yes" : "No") << std::endl;
        std::cout << "  FP16: " << (device.has(::sycl::aspect::fp16) ? "Yes" : "No") << std::endl;
        std::cout << "  FP64: " << (device.has(::sycl::aspect::fp64) ? "Yes" : "No") << std::endl;
        
        // Check for Intel GPU specific features
        if (device.is_gpu() && IsIntelDevice(device)) {
            std::cout << "  Intel GPU optimizations available" << std::endl;
        }
        
        return has_usm && has_fp32;
        
    } catch (const ::sycl::exception& e) {
        std::cerr << "SYCL Backend: Error checking device extensions: " << e.what() << std::endl;
        return false;
    }
}

// Public utility functions

std::unique_ptr<BackendInterface> CreateSyclBackend() {
    return std::make_unique<SyclBackend>();
}

bool IsSyclBackendAvailable() {
    try {
        auto devices = SyclBackend::DetectDevices();
        return !devices.empty();
    } catch (...) {
        return false;
    }
}

std::string GetOneAPIVersion() {
    // This would typically query the oneAPI runtime
    // For now, return a placeholder
    return "2024.1";
}

} // namespace sycl
} // namespace backends
} // namespace gemma