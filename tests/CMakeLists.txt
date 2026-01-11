file(GLOB_RECURSE HAHAHA_TEST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# ---- GoogleTest dependency ----
set(_gtest_main_target "")

find_package(GTest CONFIG QUIET) # vcpkg/homebrew usually provide CONFIG packages
if(NOT GTest_FOUND)
  find_package(GTest QUIET)      # fallback to CMake's FindGTest module (system installs)
endif()
if(TARGET GTest::gtest_main)
  set(_gtest_main_target GTest::gtest_main)
elseif(TARGET GTest::gtest)
  # Some distros don't ship gtest_main as an imported target; fall back.
  set(_gtest_main_target GTest::gtest)
endif()

if(NOT _gtest_main_target)
  message(FATAL_ERROR "GTest dependency not found.  Install GTest via repo-local vcpkg (preferred) or system GTest.")
endif()

add_executable(hahaha_tests ${HAHAHA_TEST_SOURCES})
target_link_libraries(hahaha_tests PRIVATE hahaha::hahaha ${_gtest_main_target})

# Optional coverage flags (gcc/clang).
if(HAHAHA_ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hahaha_tests PRIVATE -O0 -g --coverage)
    target_link_options(hahaha_tests PRIVATE --coverage)
  endif()
endif()

# MSVC: large translation units from typed tests may require /bigobj.
if(MSVC)
  target_compile_options(hahaha_tests PRIVATE /bigobj)
endif()

add_test(NAME hahaha_unit_tests COMMAND hahaha_tests)
