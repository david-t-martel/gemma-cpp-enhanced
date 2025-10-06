# FindIntelSYCL.cmake
# Find Intel oneAPI SYCL compiler and libraries
#
# This module finds Intel oneAPI DPC++ compiler and SYCL runtime
# and sets the following variables:
#
# IntelSYCL_FOUND         - True if Intel SYCL is found
# IntelSYCL_COMPILER      - Path to the Intel SYCL compiler
# IntelSYCL_VERSION       - Version of Intel SYCL
# IntelSYCL_INCLUDE_DIRS  - Include directories for SYCL
# IntelSYCL_LIBRARIES     - SYCL libraries
# IntelSYCL_FLAGS         - Compilation flags for SYCL
#
# This module also creates the following targets:
# Intel::SYCL - The Intel SYCL library target
# (Removed module-level cmake_minimum_required to avoid overriding top-level)
# Check for oneAPI environment variables
set(ONEAPI_ROOT_HINTS
    $ENV{ONEAPI_ROOT}
    $ENV{INTEL_ONEAPI_ROOT}
    $ENV{MKLROOT}/../../..
    $ENV{SETVARS_COMPLETED}
)

# Prepend explicitly provided roots so they take precedence
if(DEFINED GEMMA_ONEAPI_ROOT AND NOT "${GEMMA_ONEAPI_ROOT}" STREQUAL "")
    list(INSERT ONEAPI_ROOT_HINTS 0 "${GEMMA_ONEAPI_ROOT}")
endif()
if(DEFINED oneapi_root AND NOT "${oneapi_root}" STREQUAL "")
    list(INSERT ONEAPI_ROOT_HINTS 0 "${oneapi_root}")
endif()

# Default installation paths
if(WIN32)
    list(APPEND ONEAPI_ROOT_HINTS
        "C:/Program Files (x86)/Intel/oneAPI"
        "C:/Intel/oneAPI"
    )
elseif(APPLE)
    list(APPEND ONEAPI_ROOT_HINTS
        "/opt/intel/oneapi"
        "$ENV{HOME}/intel/oneapi"
    )
else()
    list(APPEND ONEAPI_ROOT_HINTS
        "/opt/intel/oneapi"
        "$ENV{HOME}/intel/oneapi"
    )
endif()

# Find oneAPI root directory
find_path(IntelSYCL_ROOT
    NAMES
        compiler/latest/env/vars.sh
        compiler/latest/env/vars.bat
    PATHS ${ONEAPI_ROOT_HINTS}
    DOC "Intel oneAPI root directory"
)

if(IntelSYCL_ROOT)
    # Set compiler search paths
    if(WIN32)
        set(SYCL_COMPILER_PATHS
            "${IntelSYCL_ROOT}/compiler/latest/windows/bin"
            "${IntelSYCL_ROOT}/compiler/latest/windows/bin-llvm"
        )
        set(SYCL_COMPILER_NAMES icx.exe clang++.exe)
        set(SYCL_INCLUDE_PATHS "${IntelSYCL_ROOT}/compiler/latest/windows/include")
        set(SYCL_LIBRARY_PATHS "${IntelSYCL_ROOT}/compiler/latest/windows/lib")
    else()
        set(SYCL_COMPILER_PATHS
            "${IntelSYCL_ROOT}/compiler/latest/linux/bin"
            "${IntelSYCL_ROOT}/compiler/latest/linux/bin-llvm"
        )
        set(SYCL_COMPILER_NAMES icpx clang++)
        set(SYCL_INCLUDE_PATHS "${IntelSYCL_ROOT}/compiler/latest/linux/include")
        set(SYCL_LIBRARY_PATHS "${IntelSYCL_ROOT}/compiler/latest/linux/lib")
    endif()

    # Find SYCL compiler
    find_program(IntelSYCL_COMPILER
        NAMES ${SYCL_COMPILER_NAMES}
        PATHS ${SYCL_COMPILER_PATHS}
        NO_DEFAULT_PATH
    )

    # Find SYCL include directory
    find_path(IntelSYCL_INCLUDE_DIR
        NAMES sycl/sycl.hpp CL/sycl.hpp
        PATHS ${SYCL_INCLUDE_PATHS}
        PATH_SUFFIXES sycl
        NO_DEFAULT_PATH
    )

    # Find SYCL library
    if(WIN32)
        set(SYCL_LIB_NAMES sycl8.lib sycl7.lib sycl.lib)
    else()
        set(SYCL_LIB_NAMES libsycl.so libsycl.a)
    endif()

    find_library(IntelSYCL_LIBRARY
        NAMES ${SYCL_LIB_NAMES}
        PATHS ${SYCL_LIBRARY_PATHS}
        NO_DEFAULT_PATH
    )

    # Get version information
    if(IntelSYCL_COMPILER)
        execute_process(
            COMMAND ${IntelSYCL_COMPILER} --version
            OUTPUT_VARIABLE SYCL_VERSION_OUTPUT
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # Parse version from output
        string(REGEX MATCH "DPC\\+\\+/C\\+\\+ Compiler ([0-9]+\\.[0-9]+\\.[0-9]+)"
               VERSION_MATCH "${SYCL_VERSION_OUTPUT}")
        if(CMAKE_MATCH_1)
            set(IntelSYCL_VERSION ${CMAKE_MATCH_1})
        endif()
    endif()

    # Set compilation flags
    set(IntelSYCL_FLAGS "-fsycl")
    if(WIN32)
        list(APPEND IntelSYCL_FLAGS "/EHsc")
    endif()

    # Set variables
    set(IntelSYCL_INCLUDE_DIRS ${IntelSYCL_INCLUDE_DIR})
    set(IntelSYCL_LIBRARIES ${IntelSYCL_LIBRARY})

    # Find additional required libraries
    if(NOT WIN32)
        find_library(IntelSYCL_OPENCL_LIBRARY
            NAMES OpenCL
            PATHS ${SYCL_LIBRARY_PATHS}
            NO_DEFAULT_PATH
        )
        if(IntelSYCL_OPENCL_LIBRARY)
            list(APPEND IntelSYCL_LIBRARIES ${IntelSYCL_OPENCL_LIBRARY})
        endif()
    endif()
endif()

# Handle standard find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(IntelSYCL
    FOUND_VAR IntelSYCL_FOUND
    REQUIRED_VARS
        IntelSYCL_COMPILER
        IntelSYCL_INCLUDE_DIR
        IntelSYCL_LIBRARY
    VERSION_VAR IntelSYCL_VERSION
)

# Create imported target
if(IntelSYCL_FOUND AND NOT TARGET Intel::SYCL)
    add_library(Intel::SYCL INTERFACE IMPORTED)

    set_target_properties(Intel::SYCL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${IntelSYCL_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${IntelSYCL_LIBRARIES}"
        INTERFACE_COMPILE_OPTIONS "${IntelSYCL_FLAGS}"
    )

    # Set compiler for SYCL files
    set_target_properties(Intel::SYCL PROPERTIES
        INTERFACE_COMPILE_FEATURES cxx_std_17
    )
endif()

# Mark variables as advanced
mark_as_advanced(
    IntelSYCL_ROOT
    IntelSYCL_COMPILER
    IntelSYCL_INCLUDE_DIR
    IntelSYCL_LIBRARY
)

# Provide information about the found installation
if(IntelSYCL_FOUND)
    message(STATUS "Found Intel SYCL: ${IntelSYCL_COMPILER}")
    if(IntelSYCL_VERSION)
        message(STATUS "Intel SYCL version: ${IntelSYCL_VERSION}")
    endif()
    message(STATUS "Intel SYCL include dir: ${IntelSYCL_INCLUDE_DIR}")
    message(STATUS "Intel SYCL library: ${IntelSYCL_LIBRARY}")
endif()

# Function to add SYCL compilation
function(add_sycl_to_target target)
    if(IntelSYCL_FOUND)
        target_link_libraries(${target} PRIVATE Intel::SYCL)

        # Set the compiler for .cpp files to use SYCL compiler
        get_target_property(target_sources ${target} SOURCES)
        foreach(source ${target_sources})
            if(source MATCHES "\\.(cpp|cc|cxx)$")
                set_source_files_properties(${source} PROPERTIES
                    COMPILE_FLAGS "${IntelSYCL_FLAGS}"
                )
            endif()
        endforeach()
    endif()
endfunction()