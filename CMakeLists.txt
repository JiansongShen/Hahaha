cmake_minimum_required(VERSION 3.20)

project(hahaha
  VERSION 0.0.1
  LANGUAGES CXX
)

include(GNUInstallDirs)

option(HAHAHA_DISPLAY "Build visualization/display components (GLFW/OpenGL/ImGui). Disable for headless CI." ON)
option(HAHAHA_BUILD_TESTS "Build unit tests (GoogleTest)." ON)
option(HAHAHA_BUILD_EXAMPLES "Build example programs." ON)
option(HAHAHA_ENABLE_COVERAGE "Enable coverage flags for tests (gcc/clang)." OFF)

set(HAHAHA_IMGUI_DIR "" CACHE PATH "Path to ImGui source directory (e.g. a git submodule at subprojects/imgui). If empty, CMake will auto-detect.")
set(HAHAHA_GOOGLETEST_DIR "" CACHE PATH "Path to GoogleTest source directory (e.g. a git submodule at subprojects/googletest). If empty, CMake will auto-detect.")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- core library ----

file(GLOB_RECURSE HAHAHA_SOURCES CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/core/src/*.cpp"
)

if(NOT HAHAHA_DISPLAY)
  # Headless build: drop display sources that depend on GLFW/OpenGL/ImGui.
  set(_filtered_sources "")
  foreach(_src IN LISTS HAHAHA_SOURCES)
    string(REPLACE "\\" "/" _src_norm "${_src}")
    if(NOT _src_norm MATCHES "/core/src/display/")
      list(APPEND _filtered_sources "${_src}")
    endif()
  endforeach()
  set(HAHAHA_SOURCES "${_filtered_sources}")
endif()

add_library(hahaha STATIC ${HAHAHA_SOURCES})
add_library(hahaha::hahaha ALIAS hahaha)

if(HAHAHA_ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Keep coverage builds debuggable and accurate.
    target_compile_options(hahaha PRIVATE -O0 -g --coverage)
    target_link_options(hahaha PRIVATE --coverage)
  else()
    message(WARNING "HAHAHA_ENABLE_COVERAGE=ON is only supported for GCC/Clang in this CMake setup.")
  endif()
endif()

target_include_directories(hahaha
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Meson used this flag; keep it for compatibility (even if currently unused).
target_compile_definitions(hahaha PRIVATE BUILDING_MESON_LIBRARY)

# dl (Linux/Unix) - harmless elsewhere.
if(CMAKE_DL_LIBS)
  target_link_libraries(hahaha PUBLIC ${CMAKE_DL_LIBS})
endif()

# std::stacktrace may require extra libs on some gcc/libstdc++ toolchains.
foreach(_libname IN ITEMS stdc++_libbacktrace stdc++exp backtrace)
  find_library(HAHAHA_STACKTRACE_${_libname} NAMES ${_libname})
  if(HAHAHA_STACKTRACE_${_libname})
    target_link_libraries(hahaha PUBLIC ${HAHAHA_STACKTRACE_${_libname}})
  endif()
endforeach()

if(HAHAHA_DISPLAY)
  find_package(OpenGL REQUIRED)

  # ---- GLFW (try config package, then pkg-config) ----
  set(_glfw_target "")
  find_package(glfw3 CONFIG QUIET)
  if(TARGET glfw)
    set(_glfw_target glfw)
  elseif(TARGET glfw3::glfw)
    set(_glfw_target glfw3::glfw)
  endif()

  if(NOT _glfw_target)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(GLFW3 QUIET glfw3)
    endif()
    if(GLFW3_FOUND)
      add_library(glfw3_pkg INTERFACE)
      target_include_directories(glfw3_pkg INTERFACE ${GLFW3_INCLUDE_DIRS})
      target_link_directories(glfw3_pkg INTERFACE ${GLFW3_LIBRARY_DIRS})
      target_link_libraries(glfw3_pkg INTERFACE ${GLFW3_LIBRARIES})
      target_compile_options(glfw3_pkg INTERFACE ${GLFW3_CFLAGS_OTHER})
      set(_glfw_target glfw3_pkg)
    endif()
  endif()

  if(NOT _glfw_target)
    message(FATAL_ERROR "GLFW3 not found. Install glfw3 dev package or provide a CMake glfw3 config package, or configure with -DHAHAHA_DISPLAY=OFF.")
  endif()

  # ---- ImGui (vendored as Meson subproject) ----
  if(NOT HAHAHA_IMGUI_DIR)
    if(EXISTS "${PROJECT_SOURCE_DIR}/subprojects/imgui/imgui.h")
      set(HAHAHA_IMGUI_DIR "${PROJECT_SOURCE_DIR}/subprojects/imgui")
    else()
      file(GLOB _imgui_candidates LIST_DIRECTORIES true "${PROJECT_SOURCE_DIR}/subprojects/imgui-*")
      foreach(_cand IN LISTS _imgui_candidates)
        if(EXISTS "${_cand}/imgui.h")
          set(HAHAHA_IMGUI_DIR "${_cand}")
          break()
        endif()
      endforeach()
    endif()
  endif()
  if(NOT HAHAHA_IMGUI_DIR OR NOT EXISTS "${HAHAHA_IMGUI_DIR}/imgui.h")
    message(FATAL_ERROR "ImGui not found. Provide -DHAHAHA_IMGUI_DIR=/path/to/imgui (recommended: git submodule at subprojects/imgui), or configure with -DHAHAHA_DISPLAY=OFF.")
  endif()

  set(_imgui_root "${HAHAHA_IMGUI_DIR}")
  # Build ImGui directly into libhahaha to keep installation/export simple.
  target_sources(hahaha PRIVATE
    "${_imgui_root}/imgui.cpp"
    "${_imgui_root}/imgui_demo.cpp"
    "${_imgui_root}/imgui_draw.cpp"
    "${_imgui_root}/imgui_tables.cpp"
    "${_imgui_root}/imgui_widgets.cpp"
    "${_imgui_root}/backends/imgui_impl_glfw.cpp"
    "${_imgui_root}/backends/imgui_impl_opengl3.cpp"
    "${_imgui_root}/misc/cpp/imgui_stdlib.cpp"
  )
  target_include_directories(hahaha PRIVATE
    "${_imgui_root}"
    "${_imgui_root}/backends"
    "${_imgui_root}/misc/cpp"
  )

  # Public (build-tree only): examples include <imgui.h>. Keep install interface clean.
  target_include_directories(hahaha PUBLIC
    $<BUILD_INTERFACE:${_imgui_root}>
    $<BUILD_INTERFACE:${_imgui_root}/backends>
  )

  # This mirrors Meson's declare_dependency() behavior: downstream targets that
  # link to hahaha get the needed libs as well.
  target_link_libraries(hahaha PUBLIC ${_glfw_target} OpenGL::GL)
endif()

# ---- install ----

install(TARGETS hahaha
  EXPORT hahahaTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/core/include/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT hahahaTargets
  NAMESPACE hahaha::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hahaha"
)

# ---- subprojects ----

if(HAHAHA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(HAHAHA_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ---- coverage report (gcovr) ----
if(HAHAHA_ENABLE_COVERAGE)
  find_program(GCOVR_EXECUTABLE gcovr)
  if(GCOVR_EXECUTABLE)
    set(HAHAHA_GCOVR_COMMON_ARGS
      -r .
      --merge-mode-functions=merge-use-line-min
      --exclude-unreachable-branches
      --exclude-noncode-lines
      --filter "core/."
      --exclude "subprojects/."
      --exclude "examples/."
      --exclude "core/src/display/."
      --exclude "core/include/display/."
    )

    add_custom_target(coverage
      COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
      COMMAND ${GCOVR_EXECUTABLE} ${HAHAHA_GCOVR_COMMON_ARGS} --txt-metric line --fail-under-line 80
      COMMAND ${GCOVR_EXECUTABLE} ${HAHAHA_GCOVR_COMMON_ARGS} --txt-metric branch --fail-under-branch 35
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMENT "Running tests and generating coverage report via gcovr (core only; exclude display/examples/subprojects)"
      VERBATIM
    )

    if(TARGET hahaha_tests)
      add_dependencies(coverage hahaha_tests)
    endif()

  else()
    message(STATUS "gcovr not found; coverage target will not be available. Install gcovr (e.g., pacman -S gcovr or pipx install gcovr).")
  endif()
endif()
