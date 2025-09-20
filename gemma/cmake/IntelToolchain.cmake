# Intel oneAPI C++ Toolchain Configuration for Gemma.cpp
# This toolchain file configures Intel ICX compiler with aggressive optimizations
# for maximum inference performance on Intel CPUs

# Prevent in-source builds (but allow build subdirectories)
string(FIND "${CMAKE_BINARY_DIR}" "${CMAKE_SOURCE_DIR}" source_in_binary)
if(source_in_binary EQUAL 0 AND CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed with Intel toolchain")
endif()

# Set Intel oneAPI paths
if(WIN32)
    # Windows Intel oneAPI installation
    set(INTEL_ONEAPI_ROOT "C:/Program Files (x86)/Intel/oneAPI/2025.1" CACHE PATH "Intel oneAPI root directory")
    set(INTEL_COMPILER_ROOT "${INTEL_ONEAPI_ROOT}/compiler/2025.1")
    set(INTEL_MKL_ROOT "${INTEL_ONEAPI_ROOT}/mkl/2025.1")
    set(INTEL_IPP_ROOT "${INTEL_ONEAPI_ROOT}/ipp/2025.1")
    set(INTEL_TBB_ROOT "${INTEL_ONEAPI_ROOT}/tbb/2025.1")
    set(INTEL_VTUNE_ROOT "${INTEL_ONEAPI_ROOT}/vtune/2025.1")
else()
    # Linux Intel oneAPI installation
    set(INTEL_ONEAPI_ROOT "/opt/intel/oneapi" CACHE PATH "Intel oneAPI root directory")
    set(INTEL_COMPILER_ROOT "${INTEL_ONEAPI_ROOT}/compiler/latest")
    set(INTEL_MKL_ROOT "${INTEL_ONEAPI_ROOT}/mkl/latest")
    set(INTEL_IPP_ROOT "${INTEL_ONEAPI_ROOT}/ipp/latest")
    set(INTEL_TBB_ROOT "${INTEL_ONEAPI_ROOT}/tbb/latest")
    set(INTEL_VTUNE_ROOT "${INTEL_ONEAPI_ROOT}/vtune/latest")
endif()

# Verify Intel compiler availability
if(WIN32)
    find_program(CMAKE_C_COMPILER
        NAMES icx.exe clang.exe
        PATHS "${INTEL_COMPILER_ROOT}/bin/compiler"
        NO_DEFAULT_PATH
    )
    find_program(CMAKE_CXX_COMPILER
        NAMES icx.exe clang++.exe
        PATHS "${INTEL_COMPILER_ROOT}/bin/compiler"
        NO_DEFAULT_PATH
    )
else()
    find_program(CMAKE_C_COMPILER
        NAMES icx clang
        PATHS "${INTEL_COMPILER_ROOT}/bin/compiler"
        NO_DEFAULT_PATH
    )
    find_program(CMAKE_CXX_COMPILER
        NAMES icpx clang++
        PATHS "${INTEL_COMPILER_ROOT}/bin/compiler"
        NO_DEFAULT_PATH
    )
endif()

# Fallback to wrapper scripts if available
if(NOT CMAKE_C_COMPILER OR NOT CMAKE_CXX_COMPILER)
    if(WIN32)
        find_program(CMAKE_C_COMPILER
            NAMES intel-icx.cmd
            PATHS "C:/users/david/.local/bin"
            NO_DEFAULT_PATH
        )
        find_program(CMAKE_CXX_COMPILER
            NAMES intel-icpx.cmd
            PATHS "C:/users/david/.local/bin"
            NO_DEFAULT_PATH
        )
    endif()
endif()

if(NOT CMAKE_C_COMPILER OR NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "Intel oneAPI compiler not found. Please install Intel oneAPI toolkit or check paths.")
endif()

message(STATUS "Using Intel C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "Using Intel C++ compiler: ${CMAKE_CXX_COMPILER}")

# Set compiler ID for CMake detection
set(CMAKE_C_COMPILER_ID "IntelLLVM")
set(CMAKE_CXX_COMPILER_ID "IntelLLVM")

# Configure Intel compiler for cross-compilation
set(CMAKE_SYSTEM_NAME ${CMAKE_HOST_SYSTEM_NAME})
set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_HOST_SYSTEM_PROCESSOR})

# Intel-specific optimization flags
set(INTEL_BASE_FLAGS
    "-O3"                               # Maximum optimization
    "-xHost"                            # Auto-detect CPU and optimize
    "-march=native"                     # Use native instruction set
    "-mtune=native"                     # Tune for native CPU
    "-ffast-math"                       # Aggressive math optimizations
    "-finline-functions"                # Aggressive function inlining
    "-funroll-loops"                    # Loop unrolling
    "-fno-alias"                        # Assume no pointer aliasing
)

# Intel-specific SIMD optimization flags
set(INTEL_SIMD_FLAGS
    "-msse4.2"                          # SSE 4.2 support
    "-mavx2"                            # AVX2 support
    "-mfma"                             # Fused multiply-add
)

# Detect AVX-512 support and enable if available
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # Check for AVX-512 support
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -march=native -dM -E - < /dev/null
        OUTPUT_VARIABLE COMPILER_DEFINES
        ERROR_QUIET
    )

    if(COMPILER_DEFINES MATCHES "__AVX512F__")
        list(APPEND INTEL_SIMD_FLAGS "-mavx512f" "-mavx512cd" "-mavx512bw" "-mavx512dq" "-mavx512vl")
        message(STATUS "Intel toolchain: AVX-512 support detected and enabled")
    else()
        message(STATUS "Intel toolchain: AVX-512 not available, using AVX2")
    endif()
endif()

# Intel parallel execution flags
set(INTEL_PARALLEL_FLAGS
    "-fopenmp"                          # OpenMP support
    "-parallel"                         # Intel parallel optimization
)

# Intel Math Kernel Library (MKL) flags
set(INTEL_MKL_FLAGS
    "-mkl=parallel"                     # Use parallel MKL
    "-DMKL_ILP64"                       # 64-bit integer interface
)

# Intel Integrated Performance Primitives (IPP) flags
set(INTEL_IPP_FLAGS
    "-ipp=parallel"                     # Use parallel IPP
)

# VTune profiling support flags
set(INTEL_VTUNE_FLAGS
    "-g"                                # Debug info for profiling
    "-gdwarf-4"                         # DWARF 4 debug format
    "-fno-omit-frame-pointer"           # Keep frame pointers for profiling
)

# Security and performance flags
set(INTEL_SECURITY_FLAGS
    "-fstack-protector-strong"          # Stack protection
    "-D_FORTIFY_SOURCE=2"               # Buffer overflow protection
    "-Wformat"                          # Format string warnings
    "-Wformat-security"                 # Format security warnings
)

# Combine all Intel optimization flags
set(INTEL_OPTIMIZATION_FLAGS
    ${INTEL_BASE_FLAGS}
    ${INTEL_SIMD_FLAGS}
    ${INTEL_PARALLEL_FLAGS}
    ${INTEL_MKL_FLAGS}
    ${INTEL_IPP_FLAGS}
    ${INTEL_VTUNE_FLAGS}
    ${INTEL_SECURITY_FLAGS}
)

# Apply Intel optimization flags
string(REPLACE ";" " " INTEL_FLAGS_STRING "${INTEL_OPTIMIZATION_FLAGS}")
set(CMAKE_C_FLAGS_INIT "${INTEL_FLAGS_STRING}")
set(CMAKE_CXX_FLAGS_INIT "${INTEL_FLAGS_STRING}")

# Build type specific flags
set(CMAKE_C_FLAGS_RELEASE_INIT "-O3 -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -DNDEBUG -flto")
set(CMAKE_C_FLAGS_DEBUG_INIT "-O0 -g3 -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-O0 -g3 -DDEBUG")

# Intel-specific linker flags
set(CMAKE_EXE_LINKER_FLAGS_INIT "-flto -Wl,--as-needed")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-flto -Wl,--as-needed")

# Set up Intel library paths
if(EXISTS "${INTEL_MKL_ROOT}")
    set(ENV{MKLROOT} "${INTEL_MKL_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH "${INTEL_MKL_ROOT}")
    message(STATUS "Intel MKL root: ${INTEL_MKL_ROOT}")
endif()

if(EXISTS "${INTEL_IPP_ROOT}")
    set(ENV{IPPROOT} "${INTEL_IPP_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH "${INTEL_IPP_ROOT}")
    message(STATUS "Intel IPP root: ${INTEL_IPP_ROOT}")
endif()

if(EXISTS "${INTEL_TBB_ROOT}")
    set(ENV{TBBROOT} "${INTEL_TBB_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH "${INTEL_TBB_ROOT}")
    message(STATUS "Intel TBB root: ${INTEL_TBB_ROOT}")
endif()

if(EXISTS "${INTEL_VTUNE_ROOT}")
    set(ENV{VTUNE_PROFILER_DIR} "${INTEL_VTUNE_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH "${INTEL_VTUNE_ROOT}")
    message(STATUS "Intel VTune root: ${INTEL_VTUNE_ROOT}")
endif()

# Configure Intel-specific environment
if(WIN32)
    # Windows-specific Intel environment setup
    set(ENV{INTEL_TARGET_ARCH} "intel64")
    set(ENV{INTEL_TARGET_PLATFORM} "windows")
else()
    # Linux-specific Intel environment setup
    set(ENV{INTEL_TARGET_ARCH} "intel64")
    set(ENV{INTEL_TARGET_PLATFORM} "linux")
endif()

# Intel compiler feature detection
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

# Define Intel-specific macros
add_definitions(-DINTEL_OPTIMIZED_BUILD)
add_definitions(-DUSE_INTEL_MKL)
add_definitions(-DUSE_INTEL_IPP)
add_definitions(-DHWY_INTEL_OPTIMIZED)

message(STATUS "=== Intel oneAPI Toolchain Configuration ===")
message(STATUS "Compiler: Intel ICX with oneAPI ${INTEL_ONEAPI_ROOT}")
message(STATUS "Optimization Level: -O3 -xHost -march=native")
message(STATUS "SIMD Extensions: SSE4.2, AVX2" ${INTEL_SIMD_FLAGS})
message(STATUS "Intel Libraries: MKL (parallel), IPP (parallel), TBB")
message(STATUS "Profiling: VTune support enabled")
message(STATUS "================================================")