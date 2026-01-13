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
option(HAHAHA_USE_CUDA "Enable CUDA support." OFF)

set(HAHAHA_IMGUI_DIR "" CACHE PATH "Path to ImGui source directory (e.g. a git submodule at subprojects/imgui). If empty, CMake will auto-detect.")
set(HAHAHA_GOOGLETEST_DIR "" CACHE PATH "Path to GoogleTest source directory (e.g. a git submodule at subprojects/googletest). If empty, CMake will auto-detect.")

# ---- vcpkg (repo-local) helper ----
# Prefer passing the vcpkg toolchain on the CLI:
#   -DCMAKE_TOOLCHAIN_FILE=<repo>/vcpkg/vcpkg_root/scripts/buildsystems/vcpkg.cmake
# But for convenience (and to mirror Meson's behavior), we also auto-hint CMAKE_PREFIX_PATH.
set(HAHAHA_VCPKG_TRIPLET "auto" CACHE STRING "vcpkg triplet under vcpkg/vcpkg_root/installed/<triplet> (auto/x64-linux/arm64-linux/x64-osx/arm64-osx/x64-windows/...).")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC: Use static runtime library to match vcpkg x64-windows-static triplet
# This avoids LNK2038 "RuntimeLibrary mismatch" errors
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set(_HAHAHA_VCPKG_ROOT "${PROJECT_SOURCE_DIR}/vcpkg/vcpkg_root")
set(_HAHAHA_VCPKG_INSTALLED "${_HAHAHA_VCPKG_ROOT}/installed")
if(EXISTS "${_HAHAHA_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" AND EXISTS "${_HAHAHA_VCPKG_INSTALLED}")
  set(_HAHAHA_VCPKG_TRIPLET_RESOLVED "${HAHAHA_VCPKG_TRIPLET}")
  if(_HAHAHA_VCPKG_TRIPLET_RESOLVED STREQUAL "auto" OR _HAHAHA_VCPKG_TRIPLET_RESOLVED STREQUAL "")
    # Host-based heuristic (works for native CI runners).
    # Windows: prefer static triplet to avoid DLL PATH issues in CI/tests
    if(WIN32)
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64|arm64")
        set(_HAHAHA_VCPKG_TRIPLET_RESOLVED "arm64-windows-static")
      else()
        set(_HAHAHA_VCPKG_TRIPLET_RESOLVED "x64-windows-static")
      endif()
    elseif(APPLE)
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(_HAHAHA_VCPKG_TRIPLET_RESOLVED "arm64-osx")
      else()
        set(_HAHAHA_VCPKG_TRIPLET_RESOLVED "x64-osx")
      endif()
    else()
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64|arm64")
        set(_HAHAHA_VCPKG_TRIPLET_RESOLVED "arm64-linux")
      else()
        set(_HAHAHA_VCPKG_TRIPLET_RESOLVED "x64-linux")
      endif()
    endif()
  endif()

  if(EXISTS "${_HAHAHA_VCPKG_INSTALLED}/${_HAHAHA_VCPKG_TRIPLET_RESOLVED}")
    list(PREPEND CMAKE_PREFIX_PATH "${_HAHAHA_VCPKG_INSTALLED}/${_HAHAHA_VCPKG_TRIPLET_RESOLVED}")
    message(STATUS "Using repo-local vcpkg prefix: ${_HAHAHA_VCPKG_INSTALLED}/${_HAHAHA_VCPKG_TRIPLET_RESOLVED}")
  else()
    message(STATUS "Repo-local vcpkg detected but triplet '${_HAHAHA_VCPKG_TRIPLET_RESOLVED}' not found under ${_HAHAHA_VCPKG_INSTALLED}. Pass -DHAHAHA_VCPKG_TRIPLET=<triplet> or install it via vcpkg.")
  endif()
endif()

# ---- core library ----

if(HAHAHA_USE_CUDA)
  message("Using cuda to compile")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-Bdynamic")

  find_package(CUDAToolkit REQUIRED)

  file(GLOB_RECURSE HAHAHA_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/core/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/core/src/*.cu"
  )
else()
  file(GLOB_RECURSE HAHAHA_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/core/src/*.cpp"
  )
endif()

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

target_compile_definitions(hahaha PUBLIC)

if(HAHAHA_USE_CUDA)
  target_compile_definitions(hahaha PUBLIC HAHAHA_USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # Enable separable compilation if you use device linking or dynamic parallelism
  set_target_properties(hahaha PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # Link against common CUDA libraries if needed (e.g. cuBLAS, cuFFT)
   target_link_libraries(hahaha PRIVATE CUDA::cublas CUDA::cudart)
endif()

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

  # Workaround: Remove problematic absolute paths from GLFW's link interface on Linux
  # Modern glibc includes rt and m functions, and hardcoded absolute paths may not exist
  if(TARGET ${_glfw_target} AND UNIX AND NOT APPLE)
    get_target_property(_glfw_link_libs ${_glfw_target} INTERFACE_LINK_LIBRARIES)
    if(_glfw_link_libs)
      set(_filtered_libs "")
      foreach(_lib IN LISTS _glfw_link_libs)
        # Keep the library unless it's a problematic absolute path
        if(NOT _lib MATCHES "^/usr/lib.*/(librt\\.a|libm\\.so)$")
          list(APPEND _filtered_libs "${_lib}")
        else()
          message(STATUS "Filtering out problematic library: ${_lib}")
        endif()
      endforeach()
      set_target_properties(${_glfw_target} PROPERTIES INTERFACE_LINK_LIBRARIES "${_filtered_libs}")
    endif()
  endif()

  # ---- ImGui ----
  set(_imgui_target "")

  # Try finding via Config (e.g. vcpkg/Homebrew). Repo-local vcpkg prefix is hinted above.
  find_package(imgui CONFIG REQUIRED)
  target_link_libraries(hahaha PUBLIC imgui::imgui)

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
