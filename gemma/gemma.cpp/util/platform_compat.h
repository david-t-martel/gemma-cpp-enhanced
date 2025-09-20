// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Platform compatibility layer for Windows/Linux differences

#ifndef GEMMA_UTIL_PLATFORM_COMPAT_H_
#define GEMMA_UTIL_PLATFORM_COMPAT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <memory>

#ifdef _WIN32
#define NOMINMAX  // Prevent Windows.h from defining min/max macros
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <direct.h>
#pragma warning(push)
#pragma warning(disable: 4996)  // Disable deprecation warnings
#else
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dlfcn.h>
#endif

namespace gcpp {
namespace platform {

// File path separator
#ifdef _WIN32
constexpr char kPathSeparator = '\\';
constexpr const char* kPathSeparatorStr = "\\";
#else
constexpr char kPathSeparator = '/';
constexpr const char* kPathSeparatorStr = "/";
#endif

// Dynamic library extension
#ifdef _WIN32
constexpr const char* kDynamicLibExt = ".dll";
#else
#ifdef __APPLE__
constexpr const char* kDynamicLibExt = ".dylib";
#else
constexpr const char* kDynamicLibExt = ".so";
#endif
#endif

// Platform-specific memory allocation with alignment
inline void* AlignedAlloc(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

inline void AlignedFree(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Platform-specific file operations
inline bool FileExists(const std::string& path) {
#ifdef _WIN32
    DWORD attrib = GetFileAttributesA(path.c_str());
    return (attrib != INVALID_FILE_ATTRIBUTES && 
            !(attrib & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat st;
    return (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
#endif
}

inline bool DirectoryExists(const std::string& path) {
#ifdef _WIN32
    DWORD attrib = GetFileAttributesA(path.c_str());
    return (attrib != INVALID_FILE_ATTRIBUTES && 
            (attrib & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat st;
    return (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
#endif
}

inline bool CreateDirectory(const std::string& path) {
#ifdef _WIN32
    return CreateDirectoryA(path.c_str(), nullptr) != 0 || 
           GetLastError() == ERROR_ALREADY_EXISTS;
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

inline std::string GetCurrentWorkingDirectory() {
#ifdef _WIN32
    char buffer[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, buffer);
    return std::string(buffer);
#else
    char buffer[PATH_MAX];
    getcwd(buffer, sizeof(buffer));
    return std::string(buffer);
#endif
}

// Platform-specific process/thread utilities
inline size_t GetProcessId() {
#ifdef _WIN32
    return static_cast<size_t>(GetCurrentProcessId());
#else
    return static_cast<size_t>(getpid());
#endif
}

inline size_t GetThreadId() {
#ifdef _WIN32
    return static_cast<size_t>(GetCurrentThreadId());
#else
    #ifdef __linux__
    return static_cast<size_t>(syscall(SYS_gettid));
    #else
    return static_cast<size_t>(pthread_self());
    #endif
#endif
}

// Platform-specific environment variable access
inline std::string GetEnvironmentVariable(const std::string& name) {
#ifdef _WIN32
    char buffer[32768];  // Max env var size on Windows
    DWORD size = GetEnvironmentVariableA(name.c_str(), buffer, sizeof(buffer));
    if (size == 0 || size >= sizeof(buffer)) {
        return "";
    }
    return std::string(buffer);
#else
    const char* value = getenv(name.c_str());
    return value ? std::string(value) : "";
#endif
}

inline bool SetEnvironmentVariable(const std::string& name, const std::string& value) {
#ifdef _WIN32
    return SetEnvironmentVariableA(name.c_str(), value.c_str()) != 0;
#else
    return setenv(name.c_str(), value.c_str(), 1) == 0;
#endif
}

// Platform-specific high-resolution timer
inline uint64_t GetHighResolutionTime() {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (counter.QuadPart * 1000000000ULL) / freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
#endif
}

// Memory mapping abstraction
class MemoryMappedFile {
public:
    MemoryMappedFile() : data_(nullptr), size_(0) {}
    
    ~MemoryMappedFile() {
        Close();
    }
    
    bool Open(const std::string& path, bool read_only = true) {
        Close();
        
#ifdef _WIN32
        DWORD access = read_only ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE);
        DWORD share = read_only ? FILE_SHARE_READ : 0;
        DWORD protect = read_only ? PAGE_READONLY : PAGE_READWRITE;
        DWORD map_access = read_only ? FILE_MAP_READ : FILE_MAP_ALL_ACCESS;
        
        file_handle_ = CreateFileA(path.c_str(), access, share, nullptr,
                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            return false;
        }
        
        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(file_handle_, &file_size)) {
            CloseHandle(file_handle_);
            return false;
        }
        size_ = static_cast<size_t>(file_size.QuadPart);
        
        map_handle_ = CreateFileMappingA(file_handle_, nullptr, protect, 
                                         file_size.HighPart, file_size.LowPart, nullptr);
        if (map_handle_ == nullptr) {
            CloseHandle(file_handle_);
            return false;
        }
        
        data_ = MapViewOfFile(map_handle_, map_access, 0, 0, 0);
        if (data_ == nullptr) {
            CloseHandle(map_handle_);
            CloseHandle(file_handle_);
            return false;
        }
#else
        int flags = read_only ? O_RDONLY : O_RDWR;
        fd_ = open(path.c_str(), flags);
        if (fd_ == -1) {
            return false;
        }
        
        struct stat st;
        if (fstat(fd_, &st) != 0) {
            close(fd_);
            return false;
        }
        size_ = static_cast<size_t>(st.st_size);
        
        int prot = read_only ? PROT_READ : (PROT_READ | PROT_WRITE);
        data_ = mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            close(fd_);
            data_ = nullptr;
            return false;
        }
#endif
        return true;
    }
    
    void Close() {
        if (data_) {
#ifdef _WIN32
            UnmapViewOfFile(data_);
            CloseHandle(map_handle_);
            CloseHandle(file_handle_);
#else
            munmap(data_, size_);
            close(fd_);
#endif
            data_ = nullptr;
            size_ = 0;
        }
    }
    
    void* Data() const { return data_; }
    size_t Size() const { return size_; }
    
private:
    void* data_;
    size_t size_;
#ifdef _WIN32
    HANDLE file_handle_;
    HANDLE map_handle_;
#else
    int fd_;
#endif
};

// Platform-specific compiler hints
#ifdef _WIN32
#define GEMMA_LIKELY(x) (x)
#define GEMMA_UNLIKELY(x) (x)
#define GEMMA_RESTRICT __restrict
#define GEMMA_ALIGNED(x) __declspec(align(x))
#define GEMMA_FORCE_INLINE __forceinline
#define GEMMA_NO_INLINE __declspec(noinline)
#else
#define GEMMA_LIKELY(x) __builtin_expect(!!(x), 1)
#define GEMMA_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define GEMMA_RESTRICT __restrict__
#define GEMMA_ALIGNED(x) __attribute__((aligned(x)))
#define GEMMA_FORCE_INLINE inline __attribute__((always_inline))
#define GEMMA_NO_INLINE __attribute__((noinline))
#endif

// Thread-local storage
#ifdef _WIN32
#define GEMMA_THREAD_LOCAL __declspec(thread)
#else
#define GEMMA_THREAD_LOCAL thread_local
#endif

// Export/Import for DLL support on Windows
#ifdef _WIN32
#ifdef GEMMA_DLL_EXPORTS
#define GEMMA_API __declspec(dllexport)
#else
#define GEMMA_API __declspec(dllimport)
#endif
#else
#define GEMMA_API __attribute__((visibility("default")))
#endif

// Endianness detection
inline bool IsLittleEndian() {
    uint32_t i = 1;
    char* c = reinterpret_cast<char*>(&i);
    return *c == 1;
}

// Path manipulation utilities
inline std::string JoinPath(const std::string& path1, const std::string& path2) {
    if (path1.empty()) return path2;
    if (path2.empty()) return path1;
    
    char last = path1.back();
    if (last == '/' || last == '\\') {
        return path1 + path2;
    }
    return path1 + kPathSeparatorStr + path2;
}

inline std::string GetFileExtension(const std::string& path) {
    size_t dot_pos = path.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return "";
    }
    return path.substr(dot_pos);
}

inline std::string GetBaseName(const std::string& path) {
    size_t sep_pos = path.find_last_of("/\\");
    if (sep_pos == std::string::npos) {
        return path;
    }
    return path.substr(sep_pos + 1);
}

inline std::string GetDirName(const std::string& path) {
    size_t sep_pos = path.find_last_of("/\\");
    if (sep_pos == std::string::npos) {
        return ".";
    }
    return path.substr(0, sep_pos);
}

#ifdef _WIN32
#pragma warning(pop)
#endif

}  // namespace platform
}  // namespace gcpp

#endif  // GEMMA_UTIL_PLATFORM_COMPAT_H_