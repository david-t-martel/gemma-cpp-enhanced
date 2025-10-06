function(gemma_validate_environment)
  message(STATUS "[Env] Validating key environment variables and paths...")

  if(CMAKE_ROOT AND EXISTS "${CMAKE_ROOT}")
    message(STATUS "[Env] CMAKE_ROOT: ${CMAKE_ROOT}")
  else()
    message(WARNING "[Env] CMAKE_ROOT not set or path missing: '${CMAKE_ROOT}'")
  endif()

  # oneAPI roots
  set(_oneapi_any FALSE)
  foreach(_cand IN ITEMS ${GEMMA_ONEAPI_ROOT} $ENV{ONEAPI_ROOT} $ENV{INTEL_ONEAPI_ROOT})
    if(_cand AND EXISTS "${_cand}")
      message(STATUS "[Env] oneAPI root candidate present: ${_cand}")
      set(_oneapi_any TRUE)
    endif()
  endforeach()
  if(NOT _oneapi_any)
    message(STATUS "[Env] No oneAPI root detected (SYCL backend may remain disabled)")
  endif()

  # vcpkg toolchain detection summary
  if(GEMMA_USING_VCPKG)
    message(STATUS "[Env] vcpkg toolchain active: ${CMAKE_TOOLCHAIN_FILE}")
  else()
    message(STATUS "[Env] vcpkg toolchain not active")
  endif()

  # Compiler check
  if(CMAKE_CXX_COMPILER AND EXISTS "${CMAKE_CXX_COMPILER}")
    message(STATUS "[Env] C++ Compiler: ${CMAKE_CXX_COMPILER}")
  else()
    message(WARNING "[Env] C++ compiler not resolved yet at configure stage")
  endif()
endfunction()