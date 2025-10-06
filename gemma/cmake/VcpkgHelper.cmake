# VcpkgHelper.cmake
# Helper to ensure proper vcpkg configuration

# Ensure vcpkg triplet is set correctly for the platform
if(NOT VCPKG_TARGET_TRIPLET)
    if(WIN32)
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg target triplet")
        else()
            set(VCPKG_TARGET_TRIPLET "x86-windows" CACHE STRING "vcpkg target triplet")
        endif()
    elseif(APPLE)
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(VCPKG_TARGET_TRIPLET "x64-osx" CACHE STRING "vcpkg target triplet")
        else()
            set(VCPKG_TARGET_TRIPLET "arm64-osx" CACHE STRING "vcpkg target triplet")
        endif()
    else()
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(VCPKG_TARGET_TRIPLET "x64-linux" CACHE STRING "vcpkg target triplet")
        else()
            set(VCPKG_TARGET_TRIPLET "arm64-linux" CACHE STRING "vcpkg target triplet")
        endif()
    endif()
    message(STATUS "[vcpkg] Auto-set triplet: ${VCPKG_TARGET_TRIPLET}")
endif()

# Ensure CMAKE_PREFIX_PATH includes vcpkg paths for better package discovery
if(CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg.cmake$")
    get_filename_component(_VCPKG_ROOT "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
    get_filename_component(_VCPKG_ROOT "${_VCPKG_ROOT}" DIRECTORY)
    get_filename_component(_VCPKG_ROOT "${_VCPKG_ROOT}" DIRECTORY)

    set(_VCPKG_INSTALLED_DIR "${_VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}")

    if(EXISTS "${_VCPKG_INSTALLED_DIR}")
        list(APPEND CMAKE_PREFIX_PATH "${_VCPKG_INSTALLED_DIR}")
        list(APPEND CMAKE_PREFIX_PATH "${_VCPKG_INSTALLED_DIR}/share")
        message(STATUS "[vcpkg] Added to CMAKE_PREFIX_PATH: ${_VCPKG_INSTALLED_DIR}")
    endif()
endif()

# Helper function to print package detection results
function(vcpkg_package_status PACKAGE_NAME TARGET_NAME)
    if(TARGET ${TARGET_NAME})
        message(STATUS "[vcpkg] ✓ ${PACKAGE_NAME} found as target '${TARGET_NAME}'")
    else()
        message(STATUS "[vcpkg] ✗ ${PACKAGE_NAME} not found (target '${TARGET_NAME}' not available)")
    endif()
endfunction()