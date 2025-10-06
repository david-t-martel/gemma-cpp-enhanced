# CMakeLists.txt for SentencePiece third-party integration
# This provides the sentencepiece target using locally copied vcpkg libraries

cmake_minimum_required(VERSION 3.5)

# Create the SentencePiece interface library
add_library(sentencepiece STATIC IMPORTED)

# Set the include directories for SentencePiece
target_include_directories(sentencepiece INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set the library paths based on build configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(sentencepiece PROPERTIES
        IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/debug/lib/sentencepiece.lib
    )
else()
    set_target_properties(sentencepiece PROPERTIES
        IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/sentencepiece.lib
    )
endif()

# Ensure the target directory exists
get_target_property(SENTENCEPIECE_LIB_LOCATION sentencepiece IMPORTED_LOCATION)
if(NOT EXISTS ${SENTENCEPIECE_LIB_LOCATION})
    message(FATAL_ERROR "SentencePiece library not found at: ${SENTENCEPIECE_LIB_LOCATION}")
endif()

# SentencePiece typically requires C++11 or higher
target_compile_features(sentencepiece INTERFACE cxx_std_11)

# Add system dependencies that SentencePiece needs on Windows
if(WIN32)
    target_link_libraries(sentencepiece INTERFACE
        # Add system libraries that SentencePiece depends on
        # These might be needed depending on the build
    )
endif()

# Add any necessary compile definitions
target_compile_definitions(sentencepiece INTERFACE
    # Add any SentencePiece-specific definitions if needed
)

message(STATUS "SentencePiece configured from third_party with library: ${SENTENCEPIECE_LIB_LOCATION}")

# Optional: Create an alias for compatibility
add_library(sentencepiece::sentencepiece ALIAS sentencepiece)