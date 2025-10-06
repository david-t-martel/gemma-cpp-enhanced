# FindVcpkgPackages.cmake
# Helper module to find vcpkg packages with fallbacks

# Set vcpkg variables if they're not set but CMAKE_TOOLCHAIN_FILE points to vcpkg
if(CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg.cmake$" AND NOT VCPKG_INSTALLED_DIR)
    get_filename_component(VCPKG_ROOT "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
    get_filename_component(VCPKG_ROOT "${VCPKG_ROOT}" DIRECTORY)
    get_filename_component(VCPKG_ROOT "${VCPKG_ROOT}" DIRECTORY)

    if(NOT VCPKG_TARGET_TRIPLET)
        if(WIN32)
            if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                set(VCPKG_TARGET_TRIPLET "x64-windows")
            else()
                set(VCPKG_TARGET_TRIPLET "x86-windows")
            endif()
        elseif(APPLE)
            set(VCPKG_TARGET_TRIPLET "x64-osx")
        else()
            set(VCPKG_TARGET_TRIPLET "x64-linux")
        endif()
    endif()

    set(VCPKG_INSTALLED_DIR "${VCPKG_ROOT}/installed")
    message(STATUS "[vcpkg] Auto-detected: ROOT=${VCPKG_ROOT}, TRIPLET=${VCPKG_TARGET_TRIPLET}")
endif()

# Function to find vcpkg library with proper debug/release handling
function(find_vcpkg_library LIB_VAR LIB_NAME)
    if(VCPKG_INSTALLED_DIR AND VCPKG_TARGET_TRIPLET)
        # Find release library
        find_library(${LIB_VAR}_RELEASE
            NAMES ${LIB_NAME}
            HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib"
            NO_DEFAULT_PATH
        )

        # Find debug library
        find_library(${LIB_VAR}_DEBUG
            NAMES ${LIB_NAME}
            HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/lib"
            NO_DEFAULT_PATH
        )

        # Set the main variable
        if(${LIB_VAR}_RELEASE)
            set(${LIB_VAR} "${${LIB_VAR}_RELEASE}" PARENT_SCOPE)
            set(${LIB_VAR}_RELEASE "${${LIB_VAR}_RELEASE}" PARENT_SCOPE)
            if(${LIB_VAR}_DEBUG)
                set(${LIB_VAR}_DEBUG "${${LIB_VAR}_DEBUG}" PARENT_SCOPE)
            endif()
        endif()
    endif()
endfunction()

# Function to find vcpkg include directory
function(find_vcpkg_include INCLUDE_VAR HEADER_NAME)
    if(VCPKG_INSTALLED_DIR AND VCPKG_TARGET_TRIPLET)
        find_path(${INCLUDE_VAR}
            NAMES ${HEADER_NAME}
            HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include"
            NO_DEFAULT_PATH
        )
        set(${INCLUDE_VAR} "${${INCLUDE_VAR}}" PARENT_SCOPE)
    endif()
endfunction()

# Function to create imported target from vcpkg library
function(create_vcpkg_target TARGET_NAME LIB_VAR INCLUDE_VAR)
    if(${LIB_VAR} AND ${INCLUDE_VAR})
        add_library(${TARGET_NAME} STATIC IMPORTED)
        set_target_properties(${TARGET_NAME} PROPERTIES
            IMPORTED_LOCATION "${${LIB_VAR}}"
            INTERFACE_INCLUDE_DIRECTORIES "${${INCLUDE_VAR}}"
        )

        # Set debug version if available
        if(${LIB_VAR}_DEBUG)
            set_target_properties(${TARGET_NAME} PROPERTIES
                IMPORTED_LOCATION_DEBUG "${${LIB_VAR}_DEBUG}"
            )
        endif()

        message(STATUS "Created vcpkg target ${TARGET_NAME}: ${${LIB_VAR}}")
    endif()
endfunction()