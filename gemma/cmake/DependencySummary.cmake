# DependencySummary.cmake - central dependency provenance reporting for Gemma
# Provides a function gemma_print_dependency_summary() that emits a consistent
# summary of how core third-party deps were resolved (system/vcpkg vs FetchContent).

function(gemma_print_dependency_summary)
  message(STATUS "Gemma Dependency Provenance Summary:")
  message(STATUS "  Using vcpkg toolchain: ${GEMMA_USING_VCPKG}")

  # Helper macro to print status of a target (FOUND / MISSING)
  macro(_gemma_dep_status label target)
    if(TARGET ${target})
      # Try to distinguish IMPORTED (likely system / package manager) vs built-from-source.
      get_target_property(_imported ${target} IMPORTED)
      if(_imported)
        set(_src "system/packaged")
      else()
        set(_src "built")
      endif()
      message(STATUS "  ${label}: FOUND (${_src})")
    else()
      message(STATUS "  ${label}: MISSING")
    endif()
  endmacro()

  _gemma_dep_status("highway core" hwy)
  _gemma_dep_status("highway contrib" hwy_contrib)
  if(TARGET sentencepiece-static)
    _gemma_dep_status("sentencepiece" sentencepiece-static)
  else()
    _gemma_dep_status("sentencepiece" sentencepiece)
  endif()
  _gemma_dep_status("nlohmann_json" nlohmann_json::nlohmann_json)
  _gemma_dep_status("benchmark" benchmark::benchmark)
  message(STATUS "-------------------------------------------------")
endfunction()
