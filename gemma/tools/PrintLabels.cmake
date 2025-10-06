# Simple script placeholder for printing test labels (extend as needed)
file(READ ${CMAKE_BINARY_DIR}/CTestTestfile.cmake _ctest_raw)
string(REGEX MATCHALL "add_test\(NAME ([^ ]+)" _matches "${_ctest_raw}")
message(STATUS "Registered tests (labels can be viewed with ctest -N -V | findstr LABELS)")
foreach(m IN LISTS _matches)
  # Just echo test names; labels require running ctest -N -V for full display
  string(REGEX REPLACE "add_test\(NAME ([^ ]+)" "\\1" _tname "${m}")
  message(STATUS "  ${_tname}")
endforeach()
