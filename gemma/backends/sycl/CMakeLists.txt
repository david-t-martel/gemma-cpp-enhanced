# Intel SYCL Backend for Gemma.cpp
# Provides Intel GPU and NPU acceleration using oneAPI SYCL
# Requires Intel oneAPI Toolkit 2024.1 or later

cmake_minimum_required(VERSION 3.20)  # Required for SYCL support

# Check for Intel oneAPI toolkit
find_package(IntelSYCL QUIET)

if(NOT IntelSYCL_FOUND)
    # Try alternative detection methods with Windows-specific paths
    if(WIN32)
        # Windows-specific Intel oneAPI paths
        find_program(ICPX_COMPILER 
            NAMES icpx icpx.exe
            PATHS
                "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/bin"
                "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin"
                "C:/Program Files/Intel/oneAPI/compiler/latest/windows/bin"
                "C:/Program Files/Intel/oneAPI/compiler/latest/bin"
                "$ENV{ONEAPI_ROOT}/compiler/latest/windows/bin"
                "$ENV{ICPX_HOME}/bin"
        )
        find_program(DPCPP_COMPILER 
            NAMES dpcpp dpcpp.exe
            PATHS
                "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/bin"
                "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin"
                "C:/Program Files/Intel/oneAPI/compiler/latest/windows/bin"
                "C:/Program Files/Intel/oneAPI/compiler/latest/bin"
                "$ENV{ONEAPI_ROOT}/compiler/latest/windows/bin"
                "$ENV{DPCPP_HOME}/bin"
        )
    else()
        # Linux/macOS paths
        find_program(ICPX_COMPILER icpx)
        find_program(DPCPP_COMPILER dpcpp)
    endif()

    if(ICPX_COMPILER OR DPCPP_COMPILER)
        message(STATUS "Intel SYCL compiler found: ${ICPX_COMPILER}${DPCPP_COMPILER}")
        set(SYCL_AVAILABLE TRUE)
        
        # Set Intel oneAPI root directory for later use
        if(ICPX_COMPILER)
            get_filename_component(ICPX_BIN_DIR ${ICPX_COMPILER} DIRECTORY)
            get_filename_component(INTEL_ONEAPI_ROOT ${ICPX_BIN_DIR}/../../../.. ABSOLUTE)
        elseif(DPCPP_COMPILER)
            get_filename_component(DPCPP_BIN_DIR ${DPCPP_COMPILER} DIRECTORY)
            get_filename_component(INTEL_ONEAPI_ROOT ${DPCPP_BIN_DIR}/../../../.. ABSOLUTE)
        endif()
        
        message(STATUS "Intel oneAPI root detected: ${INTEL_ONEAPI_ROOT}")
    else()
        message(WARNING "Intel SYCL compiler not found. SYCL backend will be disabled.")
        message(STATUS "To enable SYCL backend, install Intel oneAPI Toolkit:")
        message(STATUS "  https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html")
        message(STATUS "Expected paths (Windows):")
        message(STATUS "  C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/bin/icpx.exe")
        return()
    endif()
else()
    set(SYCL_AVAILABLE TRUE)
endif()

# Only proceed if SYCL is available
if(SYCL_AVAILABLE)

# Set SYCL compiler
if(ICPX_COMPILER)
    set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
elseif(DPCPP_COMPILER)
    set(CMAKE_CXX_COMPILER ${DPCPP_COMPILER})
endif()

# Enable SYCL language support
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SYCL backend sources
set(SYCL_SOURCES
    sycl_backend.h
    sycl_backend.cpp
    sycl_matmul.cpp
    sycl_attention.cpp
)

# Create SYCL backend library
add_library(gemma_sycl_backend ${SYCL_SOURCES})

# Set C++ standard for SYCL compilation
set_target_properties(gemma_sycl_backend PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# SYCL compile options
target_compile_options(gemma_sycl_backend PRIVATE
    -fsycl                          # Enable SYCL compilation
    -fsycl-targets=spir64           # Target SPIR-V for broad compatibility
    -O3                             # Optimization level
    -ffast-math                     # Fast math optimizations
    -fno-sycl-early-optimizations   # Prevent early optimizations that may break
    $<$<CONFIG:Debug>:-g -fsycl-instrument-device-code>  # Debug info
    $<$<CONFIG:Release>:-DNDEBUG>   # Release optimizations
)

# Intel GPU specific optimizations with 2025.1 architecture support
if(INTEL_GPU_TARGET)
    target_compile_options(gemma_sycl_backend PRIVATE
        -fsycl-targets=intel_gpu_${INTEL_GPU_TARGET}
    )
else()
    # Default Intel GPU targets for oneAPI 2025.1 - broader compatibility
    target_compile_options(gemma_sycl_backend PRIVATE
        # Latest Intel Arc and integrated GPUs
        -fsycl-targets=intel_gpu_pvc,intel_gpu_dg2,intel_gpu_dg1
        # Add support for newer architectures in oneAPI 2025.1
        $<$<BOOL:${WIN32}>:-fsycl-targets=intel_gpu_mtl_s,intel_gpu_arl_h,intel_gpu_lnl_m>
    )
endif()

# Windows-specific Intel GPU optimizations
if(WIN32 AND INTEL_ONEAPI_ROOT)
    target_compile_options(gemma_sycl_backend PRIVATE
        # Enable Intel GPU runtime optimizations
        -fiopenmp-simd                  # SIMD optimizations
        -qopenmp                        # OpenMP support for hybrid execution
        # Intel-specific optimizations
        -diag-disable=remark            # Disable verbose remarks
        -fp-model=fast                  # Fast floating point model
    )
    
    # Add Intel GPU libraries path
    target_link_directories(gemma_sycl_backend PRIVATE
        "${INTEL_ONEAPI_ROOT}/compiler/latest/windows/lib"
        "${INTEL_ONEAPI_ROOT}/compiler/latest/lib"
    )
endif()

# Link options for SYCL
target_link_options(gemma_sycl_backend PRIVATE
    -fsycl
    -fsycl-targets=spir64
    $<$<BOOL:${INTEL_GPU_TARGET}>:-fsycl-targets=intel_gpu_${INTEL_GPU_TARGET}>
    $<$<NOT:$<BOOL:${INTEL_GPU_TARGET}>>:-fsycl-targets=intel_gpu_pvc,intel_gpu_dg2,intel_gpu_dg1>
)

# Include directories
target_include_directories(gemma_sycl_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Find and link oneMKL with enhanced Windows detection
find_package(MKL QUIET)
if(MKL_FOUND)
    target_link_libraries(gemma_sycl_backend PRIVATE MKL::MKL_SYCL)
    target_compile_definitions(gemma_sycl_backend PRIVATE GEMMA_ENABLE_ONEMKL)
    message(STATUS "oneMKL found and enabled for SYCL backend")
else()
    # Enhanced oneMKL detection with Windows-specific paths
    if(WIN32 AND INTEL_ONEAPI_ROOT)
        # Windows Intel oneAPI 2025.1 paths
        find_library(ONEMKL_SYCL_LIBRARY
            NAMES mkl_sycl onemkl_sycl mkl_intel_lp64_dll mkl_intel_thread_dll mkl_core_dll
            HINTS
                "${INTEL_ONEAPI_ROOT}/mkl/latest/lib/intel64"
                "${INTEL_ONEAPI_ROOT}/mkl/latest/lib"
                "$ENV{MKLROOT}/lib/intel64"
                "$ENV{ONEAPI_ROOT}/mkl/latest/lib/intel64"
            PATH_SUFFIXES
                intel64
                x64
        )
        
        # Find oneMKL SYCL specific library
        find_library(ONEMKL_SYCL_INTERFACE_LIBRARY
            NAMES mkl_sycl_interface mkl_sycl_blas mkl_sycl_lapack
            HINTS
                "${INTEL_ONEAPI_ROOT}/mkl/latest/lib/intel64"
                "${INTEL_ONEAPI_ROOT}/mkl/latest/lib"
                "$ENV{MKLROOT}/lib/intel64"
        )
        
        # Add oneMKL include directories for Windows
        find_path(ONEMKL_INCLUDE_DIR oneapi/mkl.hpp
            HINTS
                "${INTEL_ONEAPI_ROOT}/mkl/latest/include"
                "$ENV{MKLROOT}/include"
                "$ENV{ONEAPI_ROOT}/mkl/latest/include"
        )
    else()
        # Linux/macOS paths
        find_library(ONEMKL_SYCL_LIBRARY
            NAMES mkl_sycl onemkl_sycl
            HINTS
                $ENV{MKLROOT}/lib/intel64
                $ENV{ONEAPI_ROOT}/mkl/latest/lib/intel64
                /opt/intel/oneapi/mkl/latest/lib/intel64
        )
        
        find_path(ONEMKL_INCLUDE_DIR oneapi/mkl.hpp
            HINTS
                $ENV{MKLROOT}/include
                $ENV{ONEAPI_ROOT}/mkl/latest/include
                /opt/intel/oneapi/mkl/latest/include
        )
    endif()

    if(ONEMKL_SYCL_LIBRARY OR ONEMKL_SYCL_INTERFACE_LIBRARY)
        # Link found oneMKL libraries
        if(ONEMKL_SYCL_LIBRARY)
            target_link_libraries(gemma_sycl_backend PRIVATE ${ONEMKL_SYCL_LIBRARY})
        endif()
        if(ONEMKL_SYCL_INTERFACE_LIBRARY)
            target_link_libraries(gemma_sycl_backend PRIVATE ${ONEMKL_SYCL_INTERFACE_LIBRARY})
        endif()
        
        target_compile_definitions(gemma_sycl_backend PRIVATE GEMMA_ENABLE_ONEMKL)
        message(STATUS "oneMKL libraries found:")
        if(ONEMKL_SYCL_LIBRARY)
            message(STATUS "  SYCL library: ${ONEMKL_SYCL_LIBRARY}")
        endif()
        if(ONEMKL_SYCL_INTERFACE_LIBRARY)
            message(STATUS "  SYCL interface: ${ONEMKL_SYCL_INTERFACE_LIBRARY}")
        endif()

        if(ONEMKL_INCLUDE_DIR)
            target_include_directories(gemma_sycl_backend PRIVATE ${ONEMKL_INCLUDE_DIR})
            message(STATUS "  Include directory: ${ONEMKL_INCLUDE_DIR}")
        endif()
        
        # Add Windows-specific MKL threading libraries
        if(WIN32)
            find_library(MKL_INTEL_THREAD_LIB mkl_intel_thread_dll
                HINTS "${INTEL_ONEAPI_ROOT}/mkl/latest/lib/intel64")
            find_library(MKL_CORE_LIB mkl_core_dll
                HINTS "${INTEL_ONEAPI_ROOT}/mkl/latest/lib/intel64")
            
            if(MKL_INTEL_THREAD_LIB)
                target_link_libraries(gemma_sycl_backend PRIVATE ${MKL_INTEL_THREAD_LIB})
            endif()
            if(MKL_CORE_LIB)
                target_link_libraries(gemma_sycl_backend PRIVATE ${MKL_CORE_LIB})
            endif()
        endif()
    else()
        message(WARNING "oneMKL not found. SYCL backend will use basic linear algebra operations.")
        message(STATUS "Searched paths:")
        if(WIN32 AND INTEL_ONEAPI_ROOT)
            message(STATUS "  ${INTEL_ONEAPI_ROOT}/mkl/latest/lib/intel64")
        endif()
        message(STATUS "  $ENV{MKLROOT}/lib/intel64")
        message(STATUS "  $ENV{ONEAPI_ROOT}/mkl/latest/lib/intel64")
        target_compile_definitions(gemma_sycl_backend PRIVATE GEMMA_NO_ONEMKL)
    endif()
endif()

# Link with backend interface
target_link_libraries(gemma_sycl_backend
    PUBLIC gemma_backends_common
)

# Compiler-specific settings
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel|IntelLLVM")
    target_compile_options(gemma_sycl_backend PRIVATE
        -Wall -Wextra                    # Enable warnings
        -Wno-unknown-pragmas             # Ignore unknown pragmas
        -diag-disable=2196               # Disable Intel-specific warning
    )

    # Intel GPU specific optimizations
    target_compile_options(gemma_sycl_backend PRIVATE
        -fiopenmp                        # Enable OpenMP for CPU fallback
        -xHost                           # Optimize for current CPU
    )
endif()

# Windows specific settings
if(WIN32)
    target_compile_definitions(gemma_sycl_backend PRIVATE
        _WIN32_WINNT=0x0A00              # Windows 10 minimum (required for Intel GPU drivers)
        NOMINMAX                         # Avoid min/max macro conflicts
        WIN32_LEAN_AND_MEAN             # Reduce Windows header size
        _CRT_SECURE_NO_WARNINGS         # Disable CRT security warnings
        INTEL_SYCL_BACKEND_ENABLED       # Mark Intel SYCL backend as available
    )

    # Link with Windows and Intel GPU libraries
    target_link_libraries(gemma_sycl_backend PRIVATE
        ole32                            # For COM support
        oleaut32                         # For variant support
        user32                           # For user interface functions
        gdi32                            # For graphics device interface
        kernel32                         # For system functions
        advapi32                         # For advanced API functions
    )
    
    # Add Intel GPU runtime libraries if available
    if(INTEL_ONEAPI_ROOT)
        find_library(INTEL_GPU_RUNTIME_LIB
            NAMES ze_loader level-zero
            HINTS
                "${INTEL_ONEAPI_ROOT}/compiler/latest/windows/lib"
                "${INTEL_ONEAPI_ROOT}/level-zero/latest/lib"
            PATH_SUFFIXES
                x64
                intel64
        )
        
        if(INTEL_GPU_RUNTIME_LIB)
            target_link_libraries(gemma_sycl_backend PRIVATE ${INTEL_GPU_RUNTIME_LIB})
            message(STATUS "Intel GPU runtime library found: ${INTEL_GPU_RUNTIME_LIB}")
        endif()
    endif()
endif()

# Linux specific settings
if(UNIX AND NOT APPLE)
    target_link_libraries(gemma_sycl_backend PRIVATE
        pthread                          # Threading support
        dl                               # Dynamic loading
    )
endif()

# Feature detection and configuration
include(CheckCXXSourceCompiles)

# Check for SYCL device selector support
check_cxx_source_compiles("
    #include <sycl/sycl.hpp>
    int main() {
        sycl::device device = sycl::device(sycl::gpu_selector_v);
        return 0;
    }
" SYCL_HAS_DEVICE_SELECTOR_V)

if(SYCL_HAS_DEVICE_SELECTOR_V)
    target_compile_definitions(gemma_sycl_backend PRIVATE SYCL_HAS_DEVICE_SELECTOR_V)
endif()

# Check for USM support
check_cxx_source_compiles("
    #include <sycl/sycl.hpp>
    int main() {
        sycl::queue q;
        void* ptr = sycl::malloc_device(1024, q);
        sycl::free(ptr, q);
        return 0;
    }
" SYCL_HAS_USM)

if(SYCL_HAS_USM)
    target_compile_definitions(gemma_sycl_backend PRIVATE SYCL_HAS_USM)
endif()

# Performance optimization flags
target_compile_definitions(gemma_sycl_backend PRIVATE
    SYCL_BACKEND_VERSION_MAJOR=1
    SYCL_BACKEND_VERSION_MINOR=0
    SYCL_BACKEND_VERSION_PATCH=0
)

# Build type specific settings
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(gemma_sycl_backend PRIVATE
        SYCL_DEBUG_ENABLED
        SYCL_PROFILE_ENABLED
    )
    target_compile_options(gemma_sycl_backend PRIVATE
        -O0                              # No optimization
        -g                               # Debug symbols
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(gemma_sycl_backend PRIVATE
        NDEBUG
    )
    target_compile_options(gemma_sycl_backend PRIVATE
        -O3                              # Maximum optimization
        -march=native                    # Optimize for current CPU
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(gemma_sycl_backend PRIVATE
        -O2                              # Moderate optimization
        -g                               # Debug symbols
    )
endif()

# Installation
install(TARGETS gemma_sycl_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    sycl_backend.h
    DESTINATION include/gemma/backends/sycl
)

# Register backend with the main backend registry
if(TARGET gemma_backends_registry)
    target_compile_definitions(gemma_backends_registry PRIVATE GEMMA_ENABLE_SYCL_BACKEND)
    target_link_libraries(gemma_backends_registry PRIVATE gemma_sycl_backend)
endif()

# Add backend to global backends list
set_property(GLOBAL APPEND PROPERTY GEMMA_AVAILABLE_BACKENDS "SYCL")

# Configuration summary
message(STATUS "========================================")
message(STATUS "Intel SYCL Backend Configuration:")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Intel oneAPI Root: ${INTEL_ONEAPI_ROOT}")
message(STATUS "  oneMKL: ${MKL_FOUND}")
message(STATUS "  Device Selector V: ${SYCL_HAS_DEVICE_SELECTOR_V}")
message(STATUS "  USM Support: ${SYCL_HAS_USM}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
if(WIN32)
    message(STATUS "  Windows GPU Driver Support: Enabled")
    if(INTEL_GPU_RUNTIME_LIB)
        message(STATUS "  Intel GPU Runtime: ${INTEL_GPU_RUNTIME_LIB}")
    endif()
endif()
message(STATUS "========================================")

# Create a test executable for SYCL backend
if(GEMMA_BUILD_TESTS OR GEMMA_BUILD_ENHANCED_TESTS)
    add_executable(test_sycl_backend test_sycl_backend.cpp)
    target_link_libraries(test_sycl_backend
        PRIVATE gemma_sycl_backend
        PRIVATE gemma_backends_common
    )
    target_compile_options(test_sycl_backend PRIVATE -fsycl)
    target_link_options(test_sycl_backend PRIVATE -fsycl)
    
    # Windows specific settings for test
    if(WIN32)
        target_compile_definitions(test_sycl_backend PRIVATE
            _WIN32_WINNT=0x0A00
            NOMINMAX
            WIN32_LEAN_AND_MEAN
        )
        
        # Set PATH for Intel GPU runtime
        if(INTEL_ONEAPI_ROOT)
            set_target_properties(test_sycl_backend PROPERTIES
                VS_DEBUGGER_ENVIRONMENT "PATH=${INTEL_ONEAPI_ROOT}/compiler/latest/windows/bin;$ENV{PATH}"
            )
        endif()
    endif()

    # Add to CTest with multiple device filters
    add_test(NAME SyclBackendTest COMMAND test_sycl_backend)
    set_tests_properties(SyclBackendTest PROPERTIES
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:gpu;opencl:gpu"
        TIMEOUT 300
    )
    
    # Add Intel GPU specific test
    add_test(NAME SyclBackendIntelGPUTest COMMAND test_sycl_backend)
    set_tests_properties(SyclBackendIntelGPUTest PROPERTIES
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:*"
        TIMEOUT 300
    )
    
    # Install test executable
    install(TARGETS test_sycl_backend
        RUNTIME DESTINATION bin/tests
    )
endif()

# Add example application
if(GEMMA_BUILD_EXAMPLES)
    add_executable(sycl_backend_example example_sycl_backend.cpp)
    target_link_libraries(sycl_backend_example
        PRIVATE gemma_sycl_backend
        PRIVATE gemma_backends_common
    )
    target_compile_options(sycl_backend_example PRIVATE -fsycl)
    target_link_options(sycl_backend_example PRIVATE -fsycl)

    install(TARGETS sycl_backend_example
        RUNTIME DESTINATION bin
    )
endif()

# Documentation
if(GEMMA_BUILD_DOCS)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/README.md.in
        ${CMAKE_CURRENT_BINARY_DIR}/README.md
        @ONLY
    )
endif()

else()
    message(STATUS "SYCL backend disabled - Intel oneAPI toolkit not found")
endif()

# Export configuration for parent projects
set(GEMMA_SYCL_BACKEND_AVAILABLE ${SYCL_AVAILABLE} PARENT_SCOPE)
if(SYCL_AVAILABLE)
    set(GEMMA_SYCL_BACKEND_TARGET gemma_sycl_backend PARENT_SCOPE)
endif()