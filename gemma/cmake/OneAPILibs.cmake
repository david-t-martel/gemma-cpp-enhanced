# OneAPILibs.cmake
# Optional Intel oneAPI framework library integration
# Provides modular, opt-in integration of TBB, IPP, DNNL, DPL

# Only proceed if user explicitly enabled oneAPI libs
if(NOT GEMMA_USE_ONEAPI_LIBS)
    return()
endif()

# ============================================================================
# Locate oneAPI Root
# ============================================================================

if(NOT DEFINED ONEAPI_ROOT)
    # Try environment variable first
    if(DEFINED ENV{ONEAPI_ROOT})
        set(ONEAPI_ROOT "$ENV{ONEAPI_ROOT}")
    elseif(DEFINED ENV{INTEL_ONEAPI_ROOT})
        set(ONEAPI_ROOT "$ENV{INTEL_ONEAPI_ROOT}")
    # Windows default installation paths
    elseif(WIN32)
        if(EXISTS "C:/Program Files (x86)/Intel/oneAPI")
            set(ONEAPI_ROOT "C:/Program Files (x86)/Intel/oneAPI")
        elseif(EXISTS "C:/Program Files/Intel/oneAPI")
            set(ONEAPI_ROOT "C:/Program Files/Intel/oneAPI")
        endif()
    # Linux default installation paths
    elseif(UNIX AND NOT APPLE)
        if(EXISTS "/opt/intel/oneapi")
            set(ONEAPI_ROOT "/opt/intel/oneapi")
        endif()
    endif()
endif()

if(NOT ONEAPI_ROOT OR NOT EXISTS "${ONEAPI_ROOT}")
    message(WARNING "GEMMA_USE_ONEAPI_LIBS=ON but oneAPI root not found")
    message(STATUS "  Set ONEAPI_ROOT environment variable or install oneAPI toolkit")
    return()
endif()

message(STATUS "")
message(STATUS "╔════════════════════════════════════════════════════════╗")
message(STATUS "║  Intel oneAPI Framework Libraries Integration         ║")
message(STATUS "╚════════════════════════════════════════════════════════╝")
message(STATUS "oneAPI Root: ${ONEAPI_ROOT}")

# Set for use by submodules
set(GEMMA_ONEAPI_ROOT "${ONEAPI_ROOT}" CACHE PATH "Intel oneAPI root directory" FORCE)

# ============================================================================
# Helper Functions
# ============================================================================

# Find and link an oneAPI library
function(gemma_add_oneapi_library TARGET_NAME OPTION_NAME LIB_NAME LIB_SUBDIR)
    if(NOT ${OPTION_NAME})
        message(STATUS "  ${LIB_NAME}: DISABLED (${OPTION_NAME}=OFF)")
        return()
    endif()
    
    # Platform-specific library patterns
    if(WIN32)
        set(LIB_PATTERNS "${LIB_NAME}.lib" "lib${LIB_NAME}.lib" "${LIB_NAME}12.lib")
        set(LIB_PATH_SUFFIXES lib lib/intel64 intel64)
    else()
        set(LIB_PATTERNS "lib${LIB_NAME}.so" "lib${LIB_NAME}.a")
        set(LIB_PATH_SUFFIXES lib lib/intel64)
    endif()
    
    # Find library
    find_library(${LIB_NAME}_LIBRARY
        NAMES ${LIB_PATTERNS}
        PATHS "${ONEAPI_ROOT}/${LIB_SUBDIR}/latest"
        PATH_SUFFIXES ${LIB_PATH_SUFFIXES}
        NO_DEFAULT_PATH
    )
    
    # Find includes
    set(INCLUDE_NAMES "${LIB_NAME}.h" "${LIB_NAME}/${LIB_NAME}.h")
    find_path(${LIB_NAME}_INCLUDE
        NAMES ${INCLUDE_NAMES}
        PATHS "${ONEAPI_ROOT}/${LIB_SUBDIR}/latest/include"
        NO_DEFAULT_PATH
    )
    
    if(${LIB_NAME}_LIBRARY AND ${LIB_NAME}_INCLUDE)
        target_link_libraries(${TARGET_NAME} PRIVATE ${${LIB_NAME}_LIBRARY})
        target_include_directories(${TARGET_NAME} PRIVATE ${${LIB_NAME}_INCLUDE})
        
        # Define preprocessor macro
        string(TOUPPER "${LIB_NAME}" LIB_NAME_UPPER)
        target_compile_definitions(${TARGET_NAME} PRIVATE GEMMA_USE_${LIB_NAME_UPPER})
        
        message(STATUS "  ${LIB_NAME}: ✅ ENABLED")
        message(STATUS "    Library: ${${LIB_NAME}_LIBRARY}")
        message(STATUS "    Include: ${${LIB_NAME}_INCLUDE}")
        
        # Store for later use (executable naming, deployment)
        set(GEMMA_ONEAPI_${LIB_NAME_UPPER}_ENABLED TRUE PARENT_SCOPE)
        set(GEMMA_ONEAPI_${LIB_NAME_UPPER}_LIBRARY "${${LIB_NAME}_LIBRARY}" PARENT_SCOPE)
        set(GEMMA_ONEAPI_${LIB_NAME_UPPER}_INCLUDE "${${LIB_NAME}_INCLUDE}" PARENT_SCOPE)
    else()
        message(WARNING "  ${LIB_NAME}: ❌ REQUESTED but not found")
        if(NOT ${LIB_NAME}_LIBRARY)
            message(STATUS "    Library search paths: ${ONEAPI_ROOT}/${LIB_SUBDIR}/latest/{lib,lib/intel64}")
        endif()
        if(NOT ${LIB_NAME}_INCLUDE)
            message(STATUS "    Include search path: ${ONEAPI_ROOT}/${LIB_SUBDIR}/latest/include")
        endif()
    endif()
endfunction()

# ============================================================================
# Integrate Libraries
# ============================================================================

if(NOT TARGET libgemma)
    message(WARNING "libgemma target not found, skipping oneAPI library integration")
    return()
endif()

# Threading Building Blocks
gemma_add_oneapi_library(libgemma GEMMA_USE_TBB "tbb" "tbb")

# Integrated Performance Primitives
gemma_add_oneapi_library(libgemma GEMMA_USE_IPP "ipp" "ipp")

# Deep Neural Network Library
gemma_add_oneapi_library(libgemma GEMMA_USE_DNNL "dnnl" "dnnl")

# Data Parallel C++ Library (header-only)
if(GEMMA_USE_DPL)
    find_path(DPL_INCLUDE
        NAMES oneapi/dpl/algorithm
        PATHS "${ONEAPI_ROOT}/dpl/latest/include"
        NO_DEFAULT_PATH
    )
    
    if(DPL_INCLUDE)
        target_include_directories(libgemma PRIVATE "${DPL_INCLUDE}")
        target_compile_definitions(libgemma PRIVATE GEMMA_USE_DPL)
        message(STATUS "  DPL: ✅ ENABLED (header-only)")
        message(STATUS "    Include: ${DPL_INCLUDE}")
        set(GEMMA_ONEAPI_DPL_ENABLED TRUE PARENT_SCOPE)
    else()
        message(WARNING "  DPL: ❌ REQUESTED but not found")
        message(STATUS "    Include search path: ${ONEAPI_ROOT}/dpl/latest/include")
    endif()
endif()

# ============================================================================
# Validation and Warnings
# ============================================================================

# Warn if DNNL is used with GPU backends (potential conflicts)
if(GEMMA_USE_DNNL AND GEMMA_ONEAPI_DNNL_ENABLED)
    if(GEMMA_ENABLE_CUDA OR GEMMA_ENABLE_SYCL)
        message(WARNING "")
        message(WARNING "╔═══════════════════════════════════════════════════════╗")
        message(WARNING "║  DNNL + GPU Backend Detected                         ║")
        message(WARNING "║  DNNL may conflict with cuBLAS/oneMKL                ║")
        message(WARNING "║  Monitor for linking errors or performance issues    ║")
        message(WARNING "╚═══════════════════════════════════════════════════════╝")
    endif()
endif()

message(STATUS "╚════════════════════════════════════════════════════════╝")
message(STATUS "")

# ============================================================================
# Export Variables for Deployment and Versioning
# ============================================================================

# Build a list of enabled oneAPI libraries for executable naming
set(GEMMA_ONEAPI_ENABLED_LIBS "")
if(GEMMA_ONEAPI_TBB_ENABLED)
    list(APPEND GEMMA_ONEAPI_ENABLED_LIBS "tbb")
endif()
if(GEMMA_ONEAPI_IPP_ENABLED)
    list(APPEND GEMMA_ONEAPI_ENABLED_LIBS "ipp")
endif()
if(GEMMA_ONEAPI_DNNL_ENABLED)
    list(APPEND GEMMA_ONEAPI_ENABLED_LIBS "dnnl")
endif()
if(GEMMA_ONEAPI_DPL_ENABLED)
    list(APPEND GEMMA_ONEAPI_ENABLED_LIBS "dpl")
endif()

set(GEMMA_ONEAPI_ENABLED_LIBS "${GEMMA_ONEAPI_ENABLED_LIBS}" CACHE INTERNAL "List of enabled oneAPI libraries")
