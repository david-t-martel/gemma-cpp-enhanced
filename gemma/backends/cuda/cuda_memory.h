#pragma once

/**
 * @file cuda_memory.h
 * @brief Advanced CUDA memory management for Gemma.cpp
 *
 * Provides optimized memory allocation, pooling, and management:
 * - Memory pooling with different allocation strategies
 * - NUMA-aware allocation for multi-GPU systems
 * - Unified memory management
 * - Memory prefetching and async operations
 * - Memory usage tracking and debugging
 */

#include "cuda_backend.h"
#include <cuda_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>

namespace gemma {
namespace backends {
namespace cuda {

/**
 * @brief Memory allocation strategies
 */
enum class AllocationStrategy {
    BEST_FIT,       // Best fit algorithm
    FIRST_FIT,      // First fit algorithm
    BUDDY_SYSTEM,   // Buddy system allocation
    SLAB_ALLOCATOR, // Slab allocator for fixed sizes
    POOL_ALLOCATOR  // Simple pool allocator
};

/**
 * @brief Memory allocation flags
 */
enum class AllocationFlags {
    NONE = 0,
    ZERO_MEMORY = 1 << 0,        // Zero out allocated memory
    PINNED_HOST = 1 << 1,        // Allocate pinned host memory
    UNIFIED_MEMORY = 1 << 2,     // Use CUDA unified memory
    PREFER_L2_RESIDENT = 1 << 3, // Prefer L2 cache resident
    STREAM_ORDERED = 1 << 4      // Use stream-ordered allocation
};

inline AllocationFlags operator|(AllocationFlags a, AllocationFlags b) {
    return static_cast<AllocationFlags>(static_cast<int>(a) | static_cast<int>(b));
}

inline AllocationFlags operator&(AllocationFlags a, AllocationFlags b) {
    return static_cast<AllocationFlags>(static_cast<int>(a) & static_cast<int>(b));
}

/**
 * @brief Memory block metadata
 */
struct MemoryBlock {
    void* ptr = nullptr;
    size_t size = 0;
    size_t alignment = 32;
    int device_id = -1;
    AllocationFlags flags = AllocationFlags::NONE;
    cudaStream_t stream = nullptr;
    std::chrono::steady_clock::time_point allocation_time;
    bool is_free = false;
    bool is_pinned = false;

    // For buddy system
    int buddy_level = -1;
    MemoryBlock* buddy_ptr = nullptr;

    // For pool management
    MemoryBlock* next = nullptr;
    MemoryBlock* prev = nullptr;
};

/**
 * @brief Memory pool statistics
 */
struct MemoryPoolStats {
    size_t total_allocated_bytes = 0;
    size_t total_free_bytes = 0;
    size_t peak_allocated_bytes = 0;
    size_t num_allocations = 0;
    size_t num_deallocations = 0;
    size_t num_cache_hits = 0;
    size_t num_cache_misses = 0;
    double average_allocation_time_ms = 0.0;
    size_t fragmentation_bytes = 0;

    // Per-device statistics
    std::unordered_map<int, size_t> per_device_allocated;
    std::unordered_map<int, size_t> per_device_free;
};

/**
 * @brief Memory allocator interface
 */
class MemoryAllocator {
public:
    virtual ~MemoryAllocator() = default;

    virtual void* Allocate(size_t size, size_t alignment, int device_id,
                          AllocationFlags flags = AllocationFlags::NONE) = 0;
    virtual bool Free(void* ptr, int device_id) = 0;
    virtual bool Initialize(const CudaConfig& config, const std::vector<int>& devices) = 0;
    virtual void Cleanup() = 0;
    virtual MemoryPoolStats GetStats() const = 0;
    virtual void ResetStats() = 0;
    virtual size_t GetTotalAllocated() const = 0;
    virtual size_t GetTotalFree() const = 0;
};

/**
 * @brief Buddy system memory allocator
 */
class BuddyAllocator : public MemoryAllocator {
public:
    explicit BuddyAllocator(size_t pool_size, size_t min_block_size = 256);
    ~BuddyAllocator() override;

    void* Allocate(size_t size, size_t alignment, int device_id,
                  AllocationFlags flags = AllocationFlags::NONE) override;
    bool Free(void* ptr, int device_id) override;
    bool Initialize(const CudaConfig& config, const std::vector<int>& devices) override;
    void Cleanup() override;
    MemoryPoolStats GetStats() const override;
    void ResetStats() override;
    size_t GetTotalAllocated() const override;
    size_t GetTotalFree() const override;

private:
    struct DevicePool {
        void* base_ptr = nullptr;
        size_t pool_size = 0;
        std::vector<std::vector<MemoryBlock*>> free_lists; // One list per level
        std::unordered_map<void*, MemoryBlock*> allocated_blocks;
        mutable std::mutex mutex;
    };

    size_t pool_size_;
    size_t min_block_size_;
    int max_levels_;
    std::unordered_map<int, std::unique_ptr<DevicePool>> device_pools_;
    mutable std::mutex stats_mutex_;
    MemoryPoolStats stats_;

    int GetLevelForSize(size_t size) const;
    size_t GetSizeForLevel(int level) const;
    MemoryBlock* SplitBlock(MemoryBlock* block, int target_level, int device_id);
    void MergeBlock(MemoryBlock* block, int device_id);
    MemoryBlock* FindBuddy(MemoryBlock* block, int device_id) const;
};

/**
 * @brief Slab allocator for fixed-size allocations
 */
class SlabAllocator : public MemoryAllocator {
public:
    explicit SlabAllocator(const std::vector<size_t>& slab_sizes);
    ~SlabAllocator() override;

    void* Allocate(size_t size, size_t alignment, int device_id,
                  AllocationFlags flags = AllocationFlags::NONE) override;
    bool Free(void* ptr, int device_id) override;
    bool Initialize(const CudaConfig& config, const std::vector<int>& devices) override;
    void Cleanup() override;
    MemoryPoolStats GetStats() const override;
    void ResetStats() override;
    size_t GetTotalAllocated() const override;
    size_t GetTotalFree() const override;

private:
    struct Slab {
        void* base_ptr = nullptr;
        size_t object_size = 0;
        size_t num_objects = 0;
        std::vector<bool> free_map;
        std::vector<void*> free_list;
        mutable std::mutex mutex;
    };

    struct DeviceSlabs {
        std::unordered_map<size_t, std::vector<std::unique_ptr<Slab>>> slabs;
        mutable std::mutex mutex;
    };

    std::vector<size_t> slab_sizes_;
    std::unordered_map<int, std::unique_ptr<DeviceSlabs>> device_slabs_;
    std::unordered_map<void*, std::pair<int, size_t>> ptr_to_slab_; // ptr -> (device_id, slab_size)
    mutable std::mutex ptr_map_mutex_;
    mutable std::mutex stats_mutex_;
    MemoryPoolStats stats_;

    size_t FindBestSlabSize(size_t size) const;
    Slab* CreateSlab(size_t object_size, int device_id);
    void* AllocateFromSlab(Slab* slab);
    bool FreeToSlab(void* ptr, Slab* slab);
};

/**
 * @brief Main CUDA memory pool implementation
 */
class CudaMemoryPool {
public:
    explicit CudaMemoryPool(const CudaConfig& config, const std::vector<int>& devices);
    ~CudaMemoryPool();

    // Initialization and cleanup
    bool Initialize();
    void Cleanup();

    // Allocation interface
    void* Allocate(size_t size, size_t alignment = 32, int device_id = -1,
                  AllocationFlags flags = AllocationFlags::NONE);
    bool Free(void* ptr, int device_id = -1);

    // Advanced allocation
    void* AllocateAsync(size_t size, cudaStream_t stream, size_t alignment = 32,
                       int device_id = -1, AllocationFlags flags = AllocationFlags::NONE);
    bool FreeAsync(void* ptr, cudaStream_t stream, int device_id = -1);

    // Host memory allocation
    void* AllocateHost(size_t size, bool pinned = true, size_t alignment = 32);
    bool FreeHost(void* ptr);

    // Unified memory allocation
    void* AllocateUnified(size_t size, size_t alignment = 32, int device_id = -1);
    bool FreeUnified(void* ptr);

    // Memory operations
    bool Prefetch(void* ptr, size_t size, int device_id, cudaStream_t stream = nullptr);
    bool AdviseReadMostly(void* ptr, size_t size);
    bool AdvisePreferredLocation(void* ptr, size_t size, int device_id);

    // Memory mapping and hints
    bool SetAccessAdvice(void* ptr, size_t size, cudaMemoryAdvise advice, int device_id);
    bool QueryAttribute(void* ptr, cudaMemoryType* type, int* device_id = nullptr);

    // Statistics and monitoring
    MemoryPoolStats GetStats() const;
    void ResetStats();
    size_t GetTotalAllocated() const;
    size_t GetTotalFree() const;
    size_t GetAvailableMemory(int device_id) const;
    size_t GetMemoryPressure(int device_id) const; // 0-100, percentage of memory used

    // Pool management
    bool TrimPool(int device_id = -1, float target_utilization = 0.8f);
    bool ExpandPool(int device_id, size_t additional_size);
    void DefragmentPool(int device_id = -1);

    // Configuration
    void SetAllocationStrategy(AllocationStrategy strategy);
    AllocationStrategy GetAllocationStrategy() const;
    void SetMemoryFraction(double fraction);
    double GetMemoryFraction() const;

    // Debug and diagnostics
    void DumpMemoryLayout(int device_id = -1) const;
    std::vector<MemoryBlock> GetActiveAllocations(int device_id = -1) const;
    bool ValidatePool(int device_id = -1) const;

private:
    CudaConfig config_;
    std::vector<int> devices_;
    AllocationStrategy strategy_;

    // Allocators for different strategies
    std::unique_ptr<MemoryAllocator> primary_allocator_;
    std::unique_ptr<SlabAllocator> slab_allocator_;

    // Host memory management
    std::unordered_map<void*, size_t> host_allocations_;
    mutable std::mutex host_mutex_;

    // Unified memory management
    std::unordered_map<void*, std::pair<size_t, int>> unified_allocations_;
    mutable std::mutex unified_mutex_;

    // Stream-ordered memory pools
    std::unordered_map<int, cudaMemPool_t> memory_pools_;
    bool use_stream_ordered_;

    // Fallback allocation tracking
    std::unordered_map<void*, std::tuple<size_t, int, AllocationFlags>> fallback_allocations_;
    mutable std::mutex fallback_mutex_;

    // Performance monitoring
    mutable std::mutex stats_mutex_;
    MemoryPoolStats global_stats_;

    // Helper methods
    bool InitializeAllocator();
    bool InitializeStreamOrderedPools();
    void CleanupStreamOrderedPools();
    void* FallbackAllocate(size_t size, size_t alignment, int device_id, AllocationFlags flags);
    bool FallbackFree(void* ptr, int device_id);
    void UpdateStats(const MemoryPoolStats& allocator_stats);
    int SelectDevice(int requested_device) const;
    bool IsValidDevice(int device_id) const;
};

/**
 * @brief Memory usage tracker for debugging
 */
class MemoryTracker {
public:
    static MemoryTracker& Instance();

    void RecordAllocation(void* ptr, size_t size, const std::string& location);
    void RecordDeallocation(void* ptr);
    void PrintMemoryReport() const;
    void CheckForLeaks() const;
    size_t GetTotalAllocated() const;
    size_t GetCurrentAllocations() const;

private:
    struct AllocationInfo {
        size_t size;
        std::string location;
        std::chrono::steady_clock::time_point timestamp;
    };

    mutable std::mutex mutex_;
    std::unordered_map<void*, AllocationInfo> allocations_;
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> total_deallocated_{0};
};

// Utility macros for memory tracking
#ifdef GEMMA_ENABLE_MEMORY_TRACKING
#define TRACK_ALLOCATION(ptr, size) \
    MemoryTracker::Instance().RecordAllocation(ptr, size, __FILE__ ":" + std::to_string(__LINE__))
#define TRACK_DEALLOCATION(ptr) \
    MemoryTracker::Instance().RecordDeallocation(ptr)
#else
#define TRACK_ALLOCATION(ptr, size)
#define TRACK_DEALLOCATION(ptr)
#endif

// Convenience functions
size_t GetOptimalAlignment(size_t size);
size_t RoundUpToAlignment(size_t size, size_t alignment);
bool IsAligned(void* ptr, size_t alignment);
size_t GetPageSize();
size_t GetHugePageSize();

} // namespace cuda
} // namespace backends
} // namespace gemma