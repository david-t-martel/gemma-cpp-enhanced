/**
 * @file cuda_backend.cpp
 * @brief NVIDIA CUDA backend implementation for Gemma.cpp
 */

#include "cuda_backend.h"
#include "cuda_kernels.h"
#include "cuda_memory.h"
#include "cuda_stream_manager.h"
#include "cuda_kernel_launcher.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <cmath>

namespace gemma {
namespace backends {
namespace cuda {

namespace {

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            return false; \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << GetCublasErrorString(status) << std::endl; \
            return false; \
        } \
    } while(0)

#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudnnGetErrorString(status) << std::endl; \
            return false; \
        } \
    } while(0)

const char* GetCublasErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "Unknown cuBLAS error";
    }
}

} // anonymous namespace

// Performance Monitor Implementation
void CudaPerformanceMonitor::RecordKernelTime(const std::string& kernel_name, double time_ms) {
    total_compute_time_ms += time_ms;
    total_kernel_launches++;

    std::lock_guard<std::mutex> lock(timing_mutex);
    operation_timings[kernel_name] += time_ms;
}

void CudaPerformanceMonitor::RecordMemoryOp(size_t bytes, double time_ms) {
    total_memory_transfer_time_ms += time_ms;
    memory_allocations++;
}

BackendMetrics CudaPerformanceMonitor::GetMetrics() const {
    BackendMetrics metrics;
    metrics.compute_throughput_gflops = 0.0; // Would need operation-specific calculation
    metrics.memory_bandwidth_gbps = current_memory_usage > 0 ?
        (current_memory_usage / 1024.0 / 1024.0 / 1024.0) / (total_memory_transfer_time_ms / 1000.0) : 0.0;
    metrics.latency_ms = total_compute_time_ms + total_memory_transfer_time_ms;
    metrics.memory_usage_bytes = current_memory_usage;
    metrics.peak_memory_bytes = peak_memory_usage;
    return metrics;
}

void CudaPerformanceMonitor::Reset() {
    total_compute_time_ms = 0.0;
    total_memory_transfer_time_ms = 0.0;
    memory_allocations = 0;
    current_memory_usage = 0;
    total_kernel_launches = 0;

    std::lock_guard<std::mutex> lock(timing_mutex);
    operation_timings.clear();
}

// CudaBackend Implementation
CudaBackend::CudaBackend(const CudaConfig& config)
    : config_(config), current_device_(0), initialized_(false) {

    // Set default device IDs if not specified
    if (config_.device_ids.empty()) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
            for (int i = 0; i < device_count; ++i) {
                config_.device_ids.push_back(i);
            }
        }
    }
}

CudaBackend::~CudaBackend() {
    if (initialized_) {
        Shutdown();
    }
}

std::string CudaBackend::GetVersion() const {
    int runtime_version = 0;
    int driver_version = 0;

    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);

    std::ostringstream oss;
    oss << "CUDA Runtime: " << (runtime_version / 1000) << "." << ((runtime_version % 100) / 10)
        << ", Driver: " << (driver_version / 1000) << "." << ((driver_version % 100) / 10);

    return oss.str();
}

bool CudaBackend::Initialize() {
    std::lock_guard<std::mutex> lock(backend_mutex_);

    if (initialized_) {
        return true;
    }

    // Check CUDA availability
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }

    // Initialize devices
    device_info_.clear();
    active_devices_.clear();

    for (int device_id : config_.device_ids) {
        if (device_id >= 0 && device_id < device_count) {
            if (InitializeDevice(device_id)) {
                active_devices_.push_back(device_id);
            }
        }
    }

    if (active_devices_.empty()) {
        std::cerr << "Failed to initialize any CUDA devices" << std::endl;
        return false;
    }

    // Set primary device
    current_device_ = config_.primary_device >= 0 &&
                     std::find(active_devices_.begin(), active_devices_.end(), config_.primary_device) != active_devices_.end()
                     ? config_.primary_device : active_devices_[0];

    CUDA_CHECK(cudaSetDevice(current_device_));

    // Enable peer access if requested and available
    if (config_.enable_peer_access && active_devices_.size() > 1) {
        EnablePeerAccess();
    }

    // Initialize memory pool
    if (config_.enable_memory_pool) {
        memory_pool_ = std::make_unique<CudaMemoryPool>(config_, active_devices_);
        if (!memory_pool_->Initialize()) {
            std::cerr << "Failed to initialize CUDA memory pool" << std::endl;
            return false;
        }
    }

    // Initialize stream manager
    stream_manager_ = std::make_unique<CudaStreamManager>(config_, active_devices_);
    if (!stream_manager_->Initialize()) {
        std::cerr << "Failed to initialize CUDA stream manager" << std::endl;
        return false;
    }

    // Initialize kernel launcher
    kernel_launcher_ = std::make_unique<CudaKernelLauncher>(config_);
    if (!kernel_launcher_->Initialize()) {
        std::cerr << "Failed to initialize CUDA kernel launcher" << std::endl;
        return false;
    }

    initialized_ = true;
    return true;
}

void CudaBackend::Shutdown() {
    std::lock_guard<std::mutex> lock(backend_mutex_);

    if (!initialized_) {
        return;
    }

    // Synchronize all devices
    for (int device_id : active_devices_) {
        cudaSetDevice(device_id);
        cudaDeviceSynchronize();
    }

    // Cleanup components
    kernel_launcher_.reset();
    stream_manager_.reset();
    memory_pool_.reset();

    // Cleanup devices
    for (int device_id : active_devices_) {
        CleanupDevice(device_id);
    }

    cublas_handles_.clear();
    cudnn_handles_.clear();
    device_info_.clear();
    active_devices_.clear();

    initialized_ = false;
}

bool CudaBackend::IsAvailable() const {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

bool CudaBackend::SupportsCapability(BackendCapability capability) const {
    switch (capability) {
        case BackendCapability::MATRIX_MULTIPLICATION:
        case BackendCapability::ATTENTION_COMPUTATION:
        case BackendCapability::ACTIVATION_FUNCTIONS:
        case BackendCapability::MEMORY_POOLING:
        case BackendCapability::ASYNC_EXECUTION:
        case BackendCapability::MULTI_PRECISION:
            return true;
        default:
            return false;
    }
}

int CudaBackend::GetDeviceCount() const {
    return static_cast<int>(active_devices_.size());
}

bool CudaBackend::SetDevice(int device_id) {
    std::lock_guard<std::mutex> lock(backend_mutex_);

    if (std::find(active_devices_.begin(), active_devices_.end(), device_id) == active_devices_.end()) {
        return false;
    }

    CUDA_CHECK(cudaSetDevice(device_id));
    current_device_ = device_id;
    return true;
}

int CudaBackend::GetCurrentDevice() const {
    return current_device_;
}

bool CudaBackend::SetMultiDevice(const std::vector<int>& device_ids) {
    std::lock_guard<std::mutex> lock(backend_mutex_);

    // Validate all device IDs
    for (int device_id : device_ids) {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_id < 0 || device_id >= device_count) {
            return false;
        }
    }

    // Update configuration
    config_.device_ids = device_ids;

    // Reinitialize if already initialized
    if (initialized_) {
        Shutdown();
        return Initialize();
    }

    return true;
}

std::vector<int> CudaBackend::GetActiveDevices() const {
    return active_devices_;
}

bool CudaBackend::EnablePeerAccess() {
    for (size_t i = 0; i < active_devices_.size(); ++i) {
        for (size_t j = 0; j < active_devices_.size(); ++j) {
            if (i != j) {
                int can_access = 0;
                cudaDeviceCanAccessPeer(&can_access, active_devices_[i], active_devices_[j]);

                if (can_access) {
                    cudaSetDevice(active_devices_[i]);
                    cudaError_t error = cudaDeviceEnablePeerAccess(active_devices_[j], 0);
                    if (error != cudaSuccess && error != cudaErrorPeerAccessAlreadyEnabled) {
                        std::cerr << "Failed to enable peer access from device "
                                  << active_devices_[i] << " to " << active_devices_[j] << std::endl;
                    }
                }
            }
        }
    }
    return true;
}

BackendBuffer CudaBackend::AllocateBuffer(size_t size, size_t alignment) {
    return AllocateCudaBuffer(size, config_.default_precision, -1, alignment);
}

CudaBuffer CudaBackend::AllocateCudaBuffer(size_t size, CudaPrecision precision,
                                          int device_id, size_t alignment) {
    if (device_id < 0) {
        device_id = current_device_;
    }

    // Use memory pool if available
    if (memory_pool_) {
        void* ptr = memory_pool_->Allocate(size, alignment, device_id);
        if (ptr) {
            CudaBuffer buffer(ptr, size, device_id, precision);
            performance_monitor_.current_memory_usage += size;
            performance_monitor_.peak_memory_usage = std::max(
                performance_monitor_.peak_memory_usage.load(),
                performance_monitor_.current_memory_usage.load());
            return buffer;
        }
    }

    // Fallback to direct allocation
    cudaSetDevice(device_id);
    void* ptr = nullptr;

    if (config_.enable_unified_memory) {
        CUDA_CHECK(cudaMallocManaged(&ptr, size));
    } else {
        CUDA_CHECK(cudaMalloc(&ptr, size));
    }

    CudaBuffer buffer(ptr, size, device_id, precision);
    buffer.is_managed_memory = config_.enable_unified_memory;

    performance_monitor_.current_memory_usage += size;
    performance_monitor_.peak_memory_usage = std::max(
        performance_monitor_.peak_memory_usage.load(),
        performance_monitor_.current_memory_usage.load());

    return buffer;
}

void CudaBackend::FreeBuffer(const BackendBuffer& buffer) {
    if (!buffer.data) return;

    const CudaBuffer& cuda_buffer = static_cast<const CudaBuffer&>(buffer);

    if (memory_pool_) {
        memory_pool_->Free(buffer.data, cuda_buffer.device_id);
    } else {
        cudaSetDevice(cuda_buffer.device_id);
        cudaFree(buffer.data);
    }

    performance_monitor_.current_memory_usage -= buffer.size;
}

bool CudaBackend::CopyToDevice(const BackendBuffer& dst, const void* src, size_t size) {
    const CudaBuffer& cuda_dst = static_cast<const CudaBuffer&>(dst);
    cudaSetDevice(cuda_dst.device_id);

    auto start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(dst.data, src, size, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    performance_monitor_.RecordMemoryOp(size, time_ms);

    return true;
}

bool CudaBackend::CopyFromDevice(void* dst, const BackendBuffer& src, size_t size) {
    const CudaBuffer& cuda_src = static_cast<const CudaBuffer&>(src);
    cudaSetDevice(cuda_src.device_id);

    auto start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(dst, src.data, size, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    performance_monitor_.RecordMemoryOp(size, time_ms);

    return true;
}

bool CudaBackend::CopyDeviceToDevice(const BackendBuffer& dst, const BackendBuffer& src, size_t size) {
    const CudaBuffer& cuda_dst = static_cast<const CudaBuffer&>(dst);
    const CudaBuffer& cuda_src = static_cast<const CudaBuffer&>(src);

    auto start = std::chrono::high_resolution_clock::now();

    if (cuda_dst.device_id == cuda_src.device_id) {
        // Same device copy
        cudaSetDevice(cuda_dst.device_id);
        CUDA_CHECK(cudaMemcpy(dst.data, src.data, size, cudaMemcpyDeviceToDevice));
    } else {
        // Cross-device copy
        CUDA_CHECK(cudaMemcpyPeer(dst.data, cuda_dst.device_id, src.data, cuda_src.device_id, size));
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    performance_monitor_.RecordMemoryOp(size, time_ms);

    return true;
}

void CudaBackend::Synchronize() {
    for (int device_id : active_devices_) {
        SynchronizeDevice(device_id);
    }
}

void CudaBackend::SynchronizeDevice(int device_id) {
    cudaSetDevice(device_id);
    cudaDeviceSynchronize();
}

BackendMetrics CudaBackend::GetMetrics() const {
    return performance_monitor_.GetMetrics();
}

void CudaBackend::ResetMetrics() {
    performance_monitor_.Reset();
}

bool CudaBackend::MatrixMultiply(const BackendBuffer& a, const BackendBuffer& b, const BackendBuffer& c,
                                int m, int n, int k, float alpha, float beta) {
    const CudaBuffer& cuda_a = static_cast<const CudaBuffer&>(a);
    const CudaBuffer& cuda_b = static_cast<const CudaBuffer&>(b);
    const CudaBuffer& cuda_c = static_cast<const CudaBuffer&>(c);

    // Ensure all buffers are on the same device
    if (cuda_a.device_id != cuda_b.device_id || cuda_b.device_id != cuda_c.device_id) {
        std::cerr << "All buffers must be on the same device for matrix multiplication" << std::endl;
        return false;
    }

    cudaSetDevice(cuda_a.device_id);
    cublasHandle_t handle = cublas_handles_[cuda_a.device_id];

    auto start = std::chrono::high_resolution_clock::now();

    // Perform GEMM based on precision
    if (cuda_a.precision == CudaPrecision::FP32) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                static_cast<const float*>(b.data), n,
                                static_cast<const float*>(a.data), k,
                                &beta,
                                static_cast<float*>(c.data), n));
    } else if (cuda_a.precision == CudaPrecision::FP16) {
        __half h_alpha = __float2half(alpha);
        __half h_beta = __float2half(beta);

        CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &h_alpha,
                                static_cast<const __half*>(b.data), n,
                                static_cast<const __half*>(a.data), k,
                                &h_beta,
                                static_cast<__half*>(c.data), n));
    } else {
        std::cerr << "Unsupported precision for matrix multiplication" << std::endl;
        return false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    performance_monitor_.RecordKernelTime("MatrixMultiply", time_ms);

    return true;
}

bool CudaBackend::MatrixVectorMultiply(const BackendBuffer& a, const BackendBuffer& x, const BackendBuffer& y,
                                      int m, int n) {
    const CudaBuffer& cuda_a = static_cast<const CudaBuffer&>(a);
    const CudaBuffer& cuda_x = static_cast<const CudaBuffer&>(x);
    const CudaBuffer& cuda_y = static_cast<const CudaBuffer&>(y);

    cudaSetDevice(cuda_a.device_id);
    cublasHandle_t handle = cublas_handles_[cuda_a.device_id];

    auto start = std::chrono::high_resolution_clock::now();

    float alpha = 1.0f, beta = 0.0f;

    if (cuda_a.precision == CudaPrecision::FP32) {
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
                                m, n,
                                &alpha,
                                static_cast<const float*>(a.data), m,
                                static_cast<const float*>(x.data), 1,
                                &beta,
                                static_cast<float*>(y.data), 1));
    } else if (cuda_a.precision == CudaPrecision::FP16) {
        __half h_alpha = __float2half(alpha);
        __half h_beta = __float2half(beta);

        // Note: cuBLAS doesn't have Hgemv, so we use Hgemm with n=1
        CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                m, 1, n,
                                &h_alpha,
                                static_cast<const __half*>(a.data), m,
                                static_cast<const __half*>(x.data), n,
                                &h_beta,
                                static_cast<__half*>(y.data), m));
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    performance_monitor_.RecordKernelTime("MatrixVectorMultiply", time_ms);

    return true;
}

bool CudaBackend::ComputeAttention(const BackendBuffer& queries, const BackendBuffer& keys,
                                  const BackendBuffer& values, const BackendBuffer& output,
                                  int batch_size, int seq_len, int head_dim, int num_heads) {
    if (config_.enable_flash_attention) {
        return ComputeFlashAttention(queries, keys, values, output,
                                   batch_size, seq_len, head_dim, num_heads);
    }

    // Fallback to standard attention implementation
    return kernel_launcher_->LaunchStandardAttention(
        queries, keys, values, output, batch_size, seq_len, head_dim, num_heads);
}

bool CudaBackend::ApplyReLU(const BackendBuffer& input, const BackendBuffer& output, size_t size) {
    const CudaBuffer& cuda_input = static_cast<const CudaBuffer&>(input);
    cudaSetDevice(cuda_input.device_id);

    auto start = std::chrono::high_resolution_clock::now();
    bool success = kernel_launcher_->LaunchReLU(input, output, size);
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    performance_monitor_.RecordKernelTime("ReLU", time_ms);

    return success;
}

bool CudaBackend::ApplyGELU(const BackendBuffer& input, const BackendBuffer& output, size_t size) {
    const CudaBuffer& cuda_input = static_cast<const CudaBuffer&>(input);
    cudaSetDevice(cuda_input.device_id);

    auto start = std::chrono::high_resolution_clock::now();
    bool success = kernel_launcher_->LaunchGELU(input, output, size);
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    performance_monitor_.RecordKernelTime("GELU", time_ms);

    return success;
}

bool CudaBackend::ApplySoftmax(const BackendBuffer& input, const BackendBuffer& output, size_t size) {
    const CudaBuffer& cuda_input = static_cast<const CudaBuffer&>(input);
    cudaSetDevice(cuda_input.device_id);

    auto start = std::chrono::high_resolution_clock::now();
    bool success = kernel_launcher_->LaunchSoftmax(input, output, size);
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    performance_monitor_.RecordKernelTime("Softmax", time_ms);

    return success;
}

// Private helper methods
bool CudaBackend::InitializeDevice(int device_id) {
    cudaSetDevice(device_id);

    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    CudaDeviceInfo info;
    info.device_id = device_id;
    info.name = prop.name;
    info.total_memory = prop.totalGlobalMem;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    info.supports_tensor_cores = (prop.major >= 7); // Volta and newer
    info.supports_fp16 = (prop.major >= 5); // Maxwell and newer
    info.supports_bf16 = (prop.major >= 8); // Ampere and newer
    info.supports_int8 = (prop.major >= 6); // Pascal and newer
    info.supports_cooperative_launch = prop.cooperativeLaunch;
    info.shared_memory_per_block = prop.sharedMemPerBlock;
    info.l2_cache_size = prop.l2CacheSize;
    info.memory_bus_width = prop.memoryBusWidth;
    info.memory_clock_rate = prop.memoryClockRate;

    // Get current free memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    info.free_memory = free_mem;

    device_info_.push_back(info);

    // Initialize cuBLAS for this device
    if (config_.enable_cublas) {
        if (!InitializeCuBLAS(device_id)) {
            std::cerr << "Failed to initialize cuBLAS for device " << device_id << std::endl;
            return false;
        }
    }

    // Initialize cuDNN for this device
    if (config_.enable_cudnn) {
        if (!InitializeCuDNN(device_id)) {
            std::cerr << "Failed to initialize cuDNN for device " << device_id << std::endl;
            // Don't fail completely as cuDNN is optional
        }
    }

    return true;
}

bool CudaBackend::InitializeCuBLAS(int device_id) {
    cudaSetDevice(device_id);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Set compute type for Tensor Core usage
    if (config_.enable_tensor_cores && IsTensorCoreAvailable(device_id)) {
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    }

    cublas_handles_[device_id] = handle;
    return true;
}

bool CudaBackend::InitializeCuDNN(int device_id) {
    cudaSetDevice(device_id);

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnn_handles_[device_id] = handle;
    return true;
}

void CudaBackend::CleanupDevice(int device_id) {
    cudaSetDevice(device_id);

    // Cleanup cuBLAS
    auto cublas_it = cublas_handles_.find(device_id);
    if (cublas_it != cublas_handles_.end()) {
        cublasDestroy(cublas_it->second);
        cublas_handles_.erase(cublas_it);
    }

    // Cleanup cuDNN
    auto cudnn_it = cudnn_handles_.find(device_id);
    if (cudnn_it != cudnn_handles_.end()) {
        cudnnDestroy(cudnn_it->second);
        cudnn_handles_.erase(cudnn_it);
    }
}

bool CudaBackend::CheckCudaError(cudaError_t error, const char* operation) const {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

bool CudaBackend::CheckCublasError(cublasStatus_t status, const char* operation) const {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error in " << operation << ": " << GetCublasErrorString(status) << std::endl;
        return false;
    }
    return true;
}

bool CudaBackend::CheckCudnnError(cudnnStatus_t status, const char* operation) const {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN error in " << operation << ": " << cudnnGetErrorString(status) << std::endl;
        return false;
    }
    return true;
}

size_t CudaBackend::GetElementSize(CudaPrecision precision) const {
    switch (precision) {
        case CudaPrecision::FP32: return sizeof(float);
        case CudaPrecision::FP16: return sizeof(__half);
        case CudaPrecision::BF16: return sizeof(__nv_bfloat16);
        case CudaPrecision::INT8: return sizeof(int8_t);
        case CudaPrecision::INT4: return sizeof(int8_t) / 2; // Packed
        default: return sizeof(float);
    }
}

bool CudaBackend::IsDeviceCompatible(int device_id, CudaPrecision precision) const {
    const auto& info = device_info_[device_id];

    switch (precision) {
        case CudaPrecision::FP32: return true;
        case CudaPrecision::FP16: return info.supports_fp16;
        case CudaPrecision::BF16: return info.supports_bf16;
        case CudaPrecision::INT8: return info.supports_int8;
        case CudaPrecision::INT4: return info.supports_int8; // INT4 requires INT8 support
        default: return false;
    }
}

CudaDeviceInfo CudaBackend::GetDeviceInfo(int device_id) const {
    for (const auto& info : device_info_) {
        if (info.device_id == device_id) {
            return info;
        }
    }
    return CudaDeviceInfo{};
}

std::vector<CudaDeviceInfo> CudaBackend::GetAllDeviceInfo() const {
    return device_info_;
}

bool CudaBackend::IsTensorCoreAvailable(int device_id) const {
    const auto& info = GetDeviceInfo(device_id);
    return info.supports_tensor_cores;
}

size_t CudaBackend::GetAvailableMemory(int device_id) const {
    cudaSetDevice(device_id);
    size_t free_mem, total_mem;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        return free_mem;
    }
    return 0;
}

// Factory functions
std::unique_ptr<BackendInterface> CreateCudaBackend(const CudaConfig& config) {
    return std::make_unique<CudaBackend>(config);
}

bool IsCudaAvailable() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::string GetCudaVersion() {
    int runtime_version = 0;
    int driver_version = 0;

    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);

    std::ostringstream oss;
    oss << "Runtime: " << (runtime_version / 1000) << "." << ((runtime_version % 100) / 10)
        << ", Driver: " << (driver_version / 1000) << "." << ((driver_version % 100) / 10);

    return oss.str();
}

CudaConfig GetOptimalCudaConfig() {
    CudaConfig config;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        return config;
    }

    // Auto-detect optimal settings based on available hardware
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            config.device_ids.push_back(i);

            // Enable features based on compute capability
            if (prop.major >= 7) {
                config.enable_tensor_cores = true;
                config.default_precision = CudaPrecision::FP16;
            }

            if (prop.major >= 6) {
                config.enable_flash_attention = true;
            }
        }
    }

    // Set memory fraction based on total available memory
    size_t total_memory = 0;
    for (int device_id : config.device_ids) {
        cudaSetDevice(device_id);
        size_t free_mem, total_mem;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
            total_memory += total_mem;
        }
    }

    // Conservative memory usage for stability
    config.memory_fraction = total_memory > 8ULL * 1024 * 1024 * 1024 ? 0.9 : 0.8; // 90% for >8GB, 80% otherwise

    return config;
}

} // namespace cuda
} // namespace backends
} // namespace gemma