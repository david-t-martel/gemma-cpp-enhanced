/**
 * @file cuda_memory.cpp
 * @brief Advanced CUDA memory management implementation for Gemma.cpp
 */

#include "cuda_memory.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>

namespace gemma {
namespace backends {
namespace cuda {

// Utility functions implementation
size_t GetOptimalAlignment(size_t size) {
    if (size >= 512) return 512;
    if (size >= 256) return 256;
    if (size >= 128) return 128;
    if (size >= 64) return 64;
    if (size >= 32) return 32;
    return 16;
}

size_t RoundUpToAlignment(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

bool IsAligned(void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

size_t GetPageSize() {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
#else
    return sysconf(_SC_PAGESIZE);
#endif
}

size_t GetHugePageSize() {
#ifdef _WIN32
    return 2 * 1024 * 1024; // 2MB on Windows
#else
    return 2 * 1024 * 1024; // 2MB default
#endif
}

// BuddyAllocator implementation
BuddyAllocator::BuddyAllocator(size_t pool_size, size_t min_block_size)
    : pool_size_(pool_size), min_block_size_(min_block_size) {
    max_levels_ = static_cast<int>(std::log2(pool_size / min_block_size)) + 1;
}

BuddyAllocator::~BuddyAllocator() {
    Cleanup();
}

bool BuddyAllocator::Initialize(const CudaConfig& config, const std::vector<int>& devices) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    for (int device_id : devices) {
        cudaError_t error = cudaSetDevice(device_id);
        if (error != cudaSuccess) {
            std::cerr << "Failed to set CUDA device " << device_id << ": " 
                      << cudaGetErrorString(error) << std::endl;
            continue;
        }

        auto pool = std::make_unique<DevicePool>();
        pool->pool_size = pool_size_;
        pool->free_lists.resize(max_levels_);

        // Allocate the main pool
        error = cudaMalloc(&pool->base_ptr, pool_size_);
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate CUDA memory pool for device " << device_id 
                      << ": " << cudaGetErrorString(error) << std::endl;
            continue;
        }

        // Initialize the largest block in the free list
        auto* root_block = new MemoryBlock();
        root_block->ptr = pool->base_ptr;
        root_block->size = pool_size_;
        root_block->device_id = device_id;
        root_block->is_free = true;
        root_block->buddy_level = max_levels_ - 1;
        root_block->allocation_time = std::chrono::steady_clock::now();

        pool->free_lists[max_levels_ - 1].push_back(root_block);
        device_pools_[device_id] = std::move(pool);
    }

    stats_.total_free_bytes = pool_size_ * devices.size();
    return !device_pools_.empty();
}

void BuddyAllocator::Cleanup() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    for (auto& [device_id, pool] : device_pools_) {
        cudaSetDevice(device_id);
        
        if (pool->base_ptr) {
            cudaFree(pool->base_ptr);
            pool->base_ptr = nullptr;
        }

        for (auto& level_list : pool->free_lists) {
            for (auto* block : level_list) {
                delete block;
            }
            level_list.clear();
        }

        for (auto& [ptr, block] : pool->allocated_blocks) {
            delete block;
        }
        pool->allocated_blocks.clear();
    }
    
    device_pools_.clear();
}

void* BuddyAllocator::Allocate(size_t size, size_t alignment, int device_id,
                              AllocationFlags flags) {
    if (size == 0) return nullptr;

    size = RoundUpToAlignment(size, alignment);
    int required_level = GetLevelForSize(size);

    if (required_level < 0 || required_level >= max_levels_) {
        return nullptr;
    }

    auto it = device_pools_.find(device_id);
    if (it == device_pools_.end()) {
        return nullptr;
    }

    DevicePool* pool = it->second.get();
    std::lock_guard<std::mutex> lock(pool->mutex);

    // Find the smallest available block that can satisfy the request
    MemoryBlock* block = nullptr;
    for (int level = required_level; level < max_levels_; ++level) {
        if (!pool->free_lists[level].empty()) {
            block = pool->free_lists[level].front();
            pool->free_lists[level].erase(pool->free_lists[level].begin());
            break;
        }
    }

    if (!block) {
        return nullptr; // Out of memory
    }

    // Split the block down to the required level
    while (block->buddy_level > required_level) {
        block = SplitBlock(block, block->buddy_level - 1, device_id);
    }

    // Mark as allocated
    block->is_free = false;
    block->allocation_time = std::chrono::steady_clock::now();
    block->flags = flags;
    pool->allocated_blocks[block->ptr] = block;

    // Zero memory if requested
    if ((flags & AllocationFlags::ZERO_MEMORY) != AllocationFlags::NONE) {
        cudaSetDevice(device_id);
        cudaMemset(block->ptr, 0, block->size);
    }

    // Update statistics
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.total_allocated_bytes += block->size;
        stats_.total_free_bytes -= block->size;
        stats_.num_allocations++;
        stats_.peak_allocated_bytes = std::max(stats_.peak_allocated_bytes, 
                                             stats_.total_allocated_bytes);
    }

    TRACK_ALLOCATION(block->ptr, block->size);
    return block->ptr;
}

bool BuddyAllocator::Free(void* ptr, int device_id) {
    if (!ptr) return true;

    auto it = device_pools_.find(device_id);
    if (it == device_pools_.end()) {
        return false;
    }

    DevicePool* pool = it->second.get();
    std::lock_guard<std::mutex> lock(pool->mutex);

    auto block_it = pool->allocated_blocks.find(ptr);
    if (block_it == pool->allocated_blocks.end()) {
        return false;
    }

    MemoryBlock* block = block_it->second;
    pool->allocated_blocks.erase(block_it);

    // Update statistics
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.total_allocated_bytes -= block->size;
        stats_.total_free_bytes += block->size;
        stats_.num_deallocations++;
    }

    TRACK_DEALLOCATION(ptr);

    // Mark as free and attempt to merge with buddy
    block->is_free = true;
    MergeBlock(block, device_id);

    return true;
}

MemoryPoolStats BuddyAllocator::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void BuddyAllocator::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = MemoryPoolStats{};
}

size_t BuddyAllocator::GetTotalAllocated() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_.total_allocated_bytes;
}

size_t BuddyAllocator::GetTotalFree() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_.total_free_bytes;
}

int BuddyAllocator::GetLevelForSize(size_t size) const {
    size_t block_size = min_block_size_;
    for (int level = 0; level < max_levels_; ++level) {
        if (size <= block_size) {
            return level;
        }
        block_size *= 2;
    }
    return -1;
}

size_t BuddyAllocator::GetSizeForLevel(int level) const {
    return min_block_size_ << level;
}

MemoryBlock* BuddyAllocator::SplitBlock(MemoryBlock* block, int target_level, int device_id) {
    if (block->buddy_level <= target_level) {
        return block;
    }

    // Create buddy block
    auto* buddy = new MemoryBlock();
    size_t half_size = GetSizeForLevel(target_level);
    
    buddy->ptr = static_cast<char*>(block->ptr) + half_size;
    buddy->size = half_size;
    buddy->device_id = device_id;
    buddy->is_free = true;
    buddy->buddy_level = target_level;
    buddy->buddy_ptr = block;
    buddy->allocation_time = std::chrono::steady_clock::now();

    // Update original block
    block->size = half_size;
    block->buddy_level = target_level;
    block->buddy_ptr = buddy;

    // Add buddy to free list
    device_pools_[device_id]->free_lists[target_level].push_back(buddy);

    return block;
}

void BuddyAllocator::MergeBlock(MemoryBlock* block, int device_id) {
    DevicePool* pool = device_pools_[device_id].get();
    
    // Try to merge with buddy
    MemoryBlock* buddy = FindBuddy(block, device_id);
    if (buddy && buddy->is_free && buddy->buddy_level == block->buddy_level) {
        // Remove buddy from free list
        auto& free_list = pool->free_lists[buddy->buddy_level];
        auto it = std::find(free_list.begin(), free_list.end(), buddy);
        if (it != free_list.end()) {
            free_list.erase(it);
        }

        // Merge blocks
        if (block->ptr > buddy->ptr) {
            std::swap(block, buddy);
        }
        
        block->size *= 2;
        block->buddy_level++;
        delete buddy;

        // Recursively try to merge at higher level
        MergeBlock(block, device_id);
    } else {
        // Add to free list
        pool->free_lists[block->buddy_level].push_back(block);
    }
}

MemoryBlock* BuddyAllocator::FindBuddy(MemoryBlock* block, int device_id) const {
    size_t block_size = GetSizeForLevel(block->buddy_level);
    uintptr_t block_addr = reinterpret_cast<uintptr_t>(block->ptr);
    uintptr_t pool_base = reinterpret_cast<uintptr_t>(device_pools_.at(device_id)->base_ptr);
    
    size_t offset = block_addr - pool_base;
    size_t buddy_offset = offset ^ block_size;
    void* buddy_ptr = reinterpret_cast<void*>(pool_base + buddy_offset);

    // Search in free lists for the buddy
    DevicePool* pool = device_pools_.at(device_id).get();
    for (auto* candidate : pool->free_lists[block->buddy_level]) {
        if (candidate->ptr == buddy_ptr) {
            return candidate;
        }
    }

    return nullptr;
}

// SlabAllocator implementation
SlabAllocator::SlabAllocator(const std::vector<size_t>& slab_sizes)
    : slab_sizes_(slab_sizes) {
    std::sort(slab_sizes_.begin(), slab_sizes_.end());
}

SlabAllocator::~SlabAllocator() {
    Cleanup();
}

bool SlabAllocator::Initialize(const CudaConfig& config, const std::vector<int>& devices) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    for (int device_id : devices) {
        cudaError_t error = cudaSetDevice(device_id);
        if (error != cudaSuccess) {
            continue;
        }

        auto device_slabs = std::make_unique<DeviceSlabs>();
        device_slabs_[device_id] = std::move(device_slabs);
    }

    return !device_slabs_.empty();
}

void SlabAllocator::Cleanup() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    for (auto& [device_id, device_slabs] : device_slabs_) {
        cudaSetDevice(device_id);
        std::lock_guard<std::mutex> device_lock(device_slabs->mutex);
        
        for (auto& [size, slab_vector] : device_slabs->slabs) {
            for (auto& slab : slab_vector) {
                if (slab->base_ptr) {
                    cudaFree(slab->base_ptr);
                }
            }
        }
        device_slabs->slabs.clear();
    }
    
    device_slabs_.clear();
    
    std::lock_guard<std::mutex> ptr_lock(ptr_map_mutex_);
    ptr_to_slab_.clear();
}

void* SlabAllocator::Allocate(size_t size, size_t alignment, int device_id,
                             AllocationFlags flags) {
    if (size == 0) return nullptr;

    size_t slab_size = FindBestSlabSize(size);
    if (slab_size == 0) {
        return nullptr; // Size too large for slab allocation
    }

    auto it = device_slabs_.find(device_id);
    if (it == device_slabs_.end()) {
        return nullptr;
    }

    DeviceSlabs* device_slabs = it->second.get();
    std::lock_guard<std::mutex> lock(device_slabs->mutex);

    // Find or create a slab with free objects
    Slab* slab = nullptr;
    auto& slab_vector = device_slabs->slabs[slab_size];
    
    for (auto& candidate : slab_vector) {
        if (!candidate->free_list.empty()) {
            slab = candidate.get();
            break;
        }
    }

    // Create new slab if needed
    if (!slab) {
        slab = CreateSlab(slab_size, device_id);
        if (!slab) {
            return nullptr;
        }
        slab_vector.push_back(std::unique_ptr<Slab>(slab));
    }

    void* ptr = AllocateFromSlab(slab);
    if (ptr) {
        // Zero memory if requested
        if ((flags & AllocationFlags::ZERO_MEMORY) != AllocationFlags::NONE) {
            cudaSetDevice(device_id);
            cudaMemset(ptr, 0, slab_size);
        }

        // Track allocation
        {
            std::lock_guard<std::mutex> ptr_lock(ptr_map_mutex_);
            ptr_to_slab_[ptr] = {device_id, slab_size};
        }

        // Update statistics
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.total_allocated_bytes += slab_size;
            stats_.num_allocations++;
            stats_.peak_allocated_bytes = std::max(stats_.peak_allocated_bytes,
                                                 stats_.total_allocated_bytes);
        }

        TRACK_ALLOCATION(ptr, slab_size);
    }

    return ptr;
}

bool SlabAllocator::Free(void* ptr, int device_id) {
    if (!ptr) return true;

    std::lock_guard<std::mutex> ptr_lock(ptr_map_mutex_);
    auto it = ptr_to_slab_.find(ptr);
    if (it == ptr_to_slab_.end()) {
        return false;
    }

    int actual_device_id = it->second.first;
    size_t slab_size = it->second.second;
    ptr_to_slab_.erase(it);
    ptr_lock.~lock_guard();

    if (device_id != -1 && device_id != actual_device_id) {
        return false;
    }

    auto device_it = device_slabs_.find(actual_device_id);
    if (device_it == device_slabs_.end()) {
        return false;
    }

    DeviceSlabs* device_slabs = device_it->second.get();
    std::lock_guard<std::mutex> lock(device_slabs->mutex);

    // Find the slab containing this pointer
    auto& slab_vector = device_slabs->slabs[slab_size];
    for (auto& slab : slab_vector) {
        char* slab_start = static_cast<char*>(slab->base_ptr);
        char* slab_end = slab_start + (slab->num_objects * slab->object_size);
        
        if (ptr >= slab_start && ptr < slab_end) {
            bool success = FreeToSlab(ptr, slab.get());
            if (success) {
                // Update statistics
                std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                stats_.total_allocated_bytes -= slab_size;
                stats_.num_deallocations++;
                
                TRACK_DEALLOCATION(ptr);
            }
            return success;
        }
    }

    return false;
}

MemoryPoolStats SlabAllocator::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void SlabAllocator::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = MemoryPoolStats{};
}

size_t SlabAllocator::GetTotalAllocated() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_.total_allocated_bytes;
}

size_t SlabAllocator::GetTotalFree() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_.total_free_bytes;
}

size_t SlabAllocator::FindBestSlabSize(size_t size) const {
    auto it = std::lower_bound(slab_sizes_.begin(), slab_sizes_.end(), size);
    return (it != slab_sizes_.end()) ? *it : 0;
}

SlabAllocator::Slab* SlabAllocator::CreateSlab(size_t object_size, int device_id) {
    const size_t slab_memory = 1024 * 1024; // 1MB per slab
    size_t num_objects = slab_memory / object_size;
    if (num_objects == 0) num_objects = 1;

    auto* slab = new Slab();
    slab->object_size = object_size;
    slab->num_objects = num_objects;

    cudaError_t error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        delete slab;
        return nullptr;
    }

    error = cudaMalloc(&slab->base_ptr, num_objects * object_size);
    if (error != cudaSuccess) {
        delete slab;
        return nullptr;
    }

    // Initialize free list
    slab->free_map.resize(num_objects, true);
    slab->free_list.reserve(num_objects);
    
    for (size_t i = 0; i < num_objects; ++i) {
        void* obj_ptr = static_cast<char*>(slab->base_ptr) + (i * object_size);
        slab->free_list.push_back(obj_ptr);
    }

    return slab;
}

void* SlabAllocator::AllocateFromSlab(Slab* slab) {
    std::lock_guard<std::mutex> lock(slab->mutex);
    
    if (slab->free_list.empty()) {
        return nullptr;
    }

    void* ptr = slab->free_list.back();
    slab->free_list.pop_back();

    // Update free map
    size_t index = (static_cast<char*>(ptr) - static_cast<char*>(slab->base_ptr)) / slab->object_size;
    slab->free_map[index] = false;

    return ptr;
}

bool SlabAllocator::FreeToSlab(void* ptr, Slab* slab) {
    std::lock_guard<std::mutex> lock(slab->mutex);
    
    // Calculate object index
    size_t index = (static_cast<char*>(ptr) - static_cast<char*>(slab->base_ptr)) / slab->object_size;
    if (index >= slab->num_objects || slab->free_map[index]) {
        return false; // Invalid pointer or double free
    }

    slab->free_map[index] = true;
    slab->free_list.push_back(ptr);
    return true;
}

// CudaMemoryPool implementation
CudaMemoryPool::CudaMemoryPool(const CudaConfig& config, const std::vector<int>& devices)
    : config_(config), devices_(devices), strategy_(AllocationStrategy::BUDDY_SYSTEM),
      use_stream_ordered_(false) {
}

CudaMemoryPool::~CudaMemoryPool() {
    Cleanup();
}

bool CudaMemoryPool::Initialize() {
    if (!InitializeAllocator()) {
        return false;
    }

    if (config_.enable_unified_memory) {
        // Initialize stream-ordered memory pools if supported
        int driver_version;
        cudaDriverGetVersion(&driver_version);
        if (driver_version >= 11020) { // CUDA 11.2+
            use_stream_ordered_ = InitializeStreamOrderedPools();
        }
    }

    return true;
}

void CudaMemoryPool::Cleanup() {
    primary_allocator_.reset();
    slab_allocator_.reset();
    CleanupStreamOrderedPools();
    
    // Clean up host allocations
    {
        std::lock_guard<std::mutex> lock(host_mutex_);
        for (auto& [ptr, size] : host_allocations_) {
            cudaFreeHost(ptr);
        }
        host_allocations_.clear();
    }

    // Clean up unified allocations
    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        for (auto& [ptr, info] : unified_allocations_) {
            cudaFree(ptr);
        }
        unified_allocations_.clear();
    }

    // Clean up fallback allocations
    {
        std::lock_guard<std::mutex> lock(fallback_mutex_);
        for (auto& [ptr, info] : fallback_allocations_) {
            cudaFree(ptr);
        }
        fallback_allocations_.clear();
    }
}

void* CudaMemoryPool::Allocate(size_t size, size_t alignment, int device_id,
                              AllocationFlags flags) {
    device_id = SelectDevice(device_id);
    if (!IsValidDevice(device_id)) {
        return nullptr;
    }

    // Try primary allocator first
    void* ptr = primary_allocator_->Allocate(size, alignment, device_id, flags);
    if (ptr) {
        return ptr;
    }

    // Try slab allocator for small allocations
    if (slab_allocator_ && size <= 4096) {
        ptr = slab_allocator_->Allocate(size, alignment, device_id, flags);
        if (ptr) {
            return ptr;
        }
    }

    // Fallback to direct CUDA allocation
    return FallbackAllocate(size, alignment, device_id, flags);
}

bool CudaMemoryPool::Free(void* ptr, int device_id) {
    if (!ptr) return true;

    device_id = SelectDevice(device_id);

    // Try primary allocator
    if (primary_allocator_->Free(ptr, device_id)) {
        return true;
    }

    // Try slab allocator
    if (slab_allocator_ && slab_allocator_->Free(ptr, device_id)) {
        return true;
    }

    // Try fallback
    return FallbackFree(ptr, device_id);
}

void* CudaMemoryPool::AllocateHost(size_t size, bool pinned, size_t alignment) {
    void* ptr = nullptr;
    cudaError_t error;

    if (pinned) {
        error = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
    } else {
        error = cudaMallocHost(&ptr, size);
    }

    if (error != cudaSuccess) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(host_mutex_);
    host_allocations_[ptr] = size;
    return ptr;
}

bool CudaMemoryPool::FreeHost(void* ptr) {
    if (!ptr) return true;

    std::lock_guard<std::mutex> lock(host_mutex_);
    auto it = host_allocations_.find(ptr);
    if (it == host_allocations_.end()) {
        return false;
    }

    host_allocations_.erase(it);
    cudaError_t error = cudaFreeHost(ptr);
    return error == cudaSuccess;
}

void* CudaMemoryPool::AllocateUnified(size_t size, size_t alignment, int device_id) {
    device_id = SelectDevice(device_id);
    if (!IsValidDevice(device_id)) {
        return nullptr;
    }

    void* ptr = nullptr;
    cudaError_t error = cudaMallocManaged(&ptr, size);
    if (error != cudaSuccess) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(unified_mutex_);
    unified_allocations_[ptr] = {size, device_id};
    return ptr;
}

bool CudaMemoryPool::FreeUnified(void* ptr) {
    if (!ptr) return true;

    std::lock_guard<std::mutex> lock(unified_mutex_);
    auto it = unified_allocations_.find(ptr);
    if (it == unified_allocations_.end()) {
        return false;
    }

    unified_allocations_.erase(it);
    cudaError_t error = cudaFree(ptr);
    return error == cudaSuccess;
}

MemoryPoolStats CudaMemoryPool::GetStats() const {
    MemoryPoolStats stats;
    
    if (primary_allocator_) {
        stats = primary_allocator_->GetStats();
    }

    if (slab_allocator_) {
        auto slab_stats = slab_allocator_->GetStats();
        stats.total_allocated_bytes += slab_stats.total_allocated_bytes;
        stats.total_free_bytes += slab_stats.total_free_bytes;
        stats.num_allocations += slab_stats.num_allocations;
        stats.num_deallocations += slab_stats.num_deallocations;
    }

    return stats;
}

void CudaMemoryPool::ResetStats() {
    if (primary_allocator_) {
        primary_allocator_->ResetStats();
    }
    if (slab_allocator_) {
        slab_allocator_->ResetStats();
    }
}

size_t CudaMemoryPool::GetTotalAllocated() const {
    size_t total = 0;
    if (primary_allocator_) total += primary_allocator_->GetTotalAllocated();
    if (slab_allocator_) total += slab_allocator_->GetTotalAllocated();
    return total;
}

size_t CudaMemoryPool::GetTotalFree() const {
    size_t total = 0;
    if (primary_allocator_) total += primary_allocator_->GetTotalFree();
    if (slab_allocator_) total += slab_allocator_->GetTotalFree();
    return total;
}

bool CudaMemoryPool::InitializeAllocator() {
    switch (strategy_) {
        case AllocationStrategy::BUDDY_SYSTEM: {
            size_t pool_size = config_.memory_pool_size;
            if (pool_size == 0) {
                // Auto-calculate pool size based on available memory
                size_t free_mem, total_mem;
                cudaMemGetInfo(&free_mem, &total_mem);
                pool_size = static_cast<size_t>(free_mem * config_.memory_fraction);
            }
            primary_allocator_ = std::make_unique<BuddyAllocator>(pool_size, 256);
            break;
        }
        case AllocationStrategy::SLAB_ALLOCATOR: {
            std::vector<size_t> slab_sizes = {64, 128, 256, 512, 1024, 2048, 4096};
            primary_allocator_ = std::make_unique<SlabAllocator>(slab_sizes);
            break;
        }
        default:
            return false;
    }

    // Always create a slab allocator for small allocations
    std::vector<size_t> small_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    slab_allocator_ = std::make_unique<SlabAllocator>(small_sizes);

    bool primary_init = primary_allocator_->Initialize(config_, devices_);
    bool slab_init = slab_allocator_->Initialize(config_, devices_);

    return primary_init && slab_init;
}

bool CudaMemoryPool::InitializeStreamOrderedPools() {
    for (int device_id : devices_) {
        cudaError_t error = cudaSetDevice(device_id);
        if (error != cudaSuccess) continue;

        cudaMemPool_t pool;
        cudaMemPoolProps props = {};
        props.allocType = cudaMemAllocationTypePinned;
        props.handleTypes = cudaMemHandleTypeNone;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = device_id;

        error = cudaMemPoolCreate(&pool, &props);
        if (error == cudaSuccess) {
            memory_pools_[device_id] = pool;
        }
    }

    return !memory_pools_.empty();
}

void CudaMemoryPool::CleanupStreamOrderedPools() {
    for (auto& [device_id, pool] : memory_pools_) {
        cudaSetDevice(device_id);
        cudaMemPoolDestroy(pool);
    }
    memory_pools_.clear();
}

void* CudaMemoryPool::FallbackAllocate(size_t size, size_t alignment, int device_id,
                                      AllocationFlags flags) {
    void* ptr = nullptr;
    cudaError_t error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        return nullptr;
    }

    error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        return nullptr;
    }

    if ((flags & AllocationFlags::ZERO_MEMORY) != AllocationFlags::NONE) {
        cudaMemset(ptr, 0, size);
    }

    std::lock_guard<std::mutex> lock(fallback_mutex_);
    fallback_allocations_[ptr] = {size, device_id, flags};
    return ptr;
}

bool CudaMemoryPool::FallbackFree(void* ptr, int device_id) {
    std::lock_guard<std::mutex> lock(fallback_mutex_);
    auto it = fallback_allocations_.find(ptr);
    if (it == fallback_allocations_.end()) {
        return false;
    }

    fallback_allocations_.erase(it);
    cudaError_t error = cudaFree(ptr);
    return error == cudaSuccess;
}

int CudaMemoryPool::SelectDevice(int requested_device) const {
    if (requested_device == -1) {
        return devices_.empty() ? 0 : devices_[0];
    }
    return requested_device;
}

bool CudaMemoryPool::IsValidDevice(int device_id) const {
    return std::find(devices_.begin(), devices_.end(), device_id) != devices_.end();
}

// MemoryTracker implementation
MemoryTracker& MemoryTracker::Instance() {
    static MemoryTracker instance;
    return instance;
}

void MemoryTracker::RecordAllocation(void* ptr, size_t size, const std::string& location) {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[ptr] = {size, location, std::chrono::steady_clock::now()};
    total_allocated_ += size;
}

void MemoryTracker::RecordDeallocation(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        total_deallocated_ += it->second.size;
        allocations_.erase(it);
    }
}

void MemoryTracker::PrintMemoryReport() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "=== Memory Usage Report ===" << std::endl;
    std::cout << "Total allocated: " << total_allocated_.load() << " bytes" << std::endl;
    std::cout << "Total deallocated: " << total_deallocated_.load() << " bytes" << std::endl;
    std::cout << "Current allocations: " << allocations_.size() << std::endl;
    std::cout << "Leaked memory: " << (total_allocated_ - total_deallocated_) << " bytes" << std::endl;

    if (!allocations_.empty()) {
        std::cout << "\nActive allocations:" << std::endl;
        for (const auto& [ptr, info] : allocations_) {
            std::cout << "  " << ptr << ": " << info.size << " bytes at " << info.location << std::endl;
        }
    }
}

void MemoryTracker::CheckForLeaks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!allocations_.empty()) {
        std::cerr << "Memory leaks detected: " << allocations_.size() << " allocations not freed" << std::endl;
        PrintMemoryReport();
    }
}

size_t MemoryTracker::GetTotalAllocated() const {
    return total_allocated_.load();
}

size_t MemoryTracker::GetCurrentAllocations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocations_.size();
}

} // namespace cuda
} // namespace backends
} // namespace gemma