cmake_minimum_required(VERSION 3.20)
if (MSVC)
    add_compile_options("/execution_charset:utf-8" "/source_charset:utf-8")
    set(ENV{VSLANG} "1033")
endif ()

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE OR CMAKE_TOOLCHAIN_FILE STREQUAL "")
    message(FATAL_ERROR
            "vcpkg toolchain not set. Please either:\n"
            "  1) Use a CMake preset: cmake --preset <preset-name>\n"
            "     (set VCPKG_ROOT to your vcpkg root, e.g. <repo>/vcpkg/vcpkg_root)\n"
            "  2) Pass the toolchain: -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake"
    )
endif ()

# Use static triplet on Windows so vcpkg builds GTest (and deps) with static CRT.
# This avoids heap corruption (debug_heap _acrt_first_block == header) when the exe
# uses /MTd and GTest would otherwise be a DLL with /MDd (separate heaps).
if (WIN32 AND (NOT DEFINED VCPKG_TARGET_TRIPLET OR VCPKG_TARGET_TRIPLET STREQUAL ""))
    set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING "vcpkg triplet (static CRT to match this project)")
endif ()

#=========================== Project Configuration ===================================
project(hahaha
        VERSION 0.0.1
        LANGUAGES CXX
)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC: Use static runtime library to match vcpkg x64-windows-static triplet
# This avoids LNK2038 "RuntimeLibrary mismatch" errors
if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif ()

#=========================== Dependencies Management ===================================
# ---- core library ----
# Full configuration: include all source files
file(GLOB_RECURSE HAHAHA_CPP_SOURCES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/core/src/*.cpp"
)

file(
        GLOB_RECURSE HAHAHA_CUDA_SOURCES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/core/src/*.cu"
)

# Enable CUDA language before any target uses .cu sources (required for CMAKE_CUDA_COMPILE_OBJECT etc.)
if (HAHAHA_USE_CUDA)
    if (MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    endif ()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

if (HAHAHA_USE_CUDA)
    add_library(hahaha STATIC ${HAHAHA_CPP_SOURCES} ${HAHAHA_CUDA_SOURCES})
    add_compile_definitions(HAHAHA_USE_CUDA)
else ()
    add_library(hahaha STATIC ${HAHAHA_CPP_SOURCES} )
endif ()
add_library(hahaha::hahaha ALIAS hahaha)

target_include_directories(hahaha
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(hahaha PUBLIC)

# dl (Linux/Unix) - harmless elsewhere.
if (CMAKE_DL_LIBS)
    target_link_libraries(hahaha PUBLIC ${CMAKE_DL_LIBS})
endif ()

# std::stacktrace may require extra libs on some gcc/libstdc++ toolchains.
foreach (_libname IN ITEMS stdc++_libbacktrace stdc++exp backtrace)
    find_library(HAHAHA_STACKTRACE_${_libname} NAMES ${_libname})
    if (HAHAHA_STACKTRACE_${_libname})
        target_link_libraries(hahaha PUBLIC ${HAHAHA_STACKTRACE_${_libname}})
    endif ()
endforeach ()

# ---- Dependencies (Full Configuration) ----
# All dependencies are managed via vcpkg.json and will be automatically installed

# CUDA: when enabled, add toolkit include path so __has_include(<driver_types.h>) works in .cpp
if (HAHAHA_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(hahaha PUBLIC CUDA::cudart)
endif ()

# OpenGL
find_package(OpenGL REQUIRED)

# GLFW3 (via vcpkg)
find_package(glfw3 CONFIG REQUIRED)
if (TARGET glfw3::glfw)
    set(_glfw_target glfw3::glfw)
elseif (TARGET glfw)
    set(_glfw_target glfw)
else ()
    message(FATAL_ERROR "GLFW3 not found. Please ensure glfw3 is installed via vcpkg.")
endif ()

# ImGui (via vcpkg)
find_package(imgui CONFIG REQUIRED)

# Link all display dependencies
target_link_libraries(hahaha PUBLIC
        ${_glfw_target}
        imgui::imgui
        OpenGL::GL
)

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
# Full configuration: build tests and examples
enable_testing()
add_subdirectory(tests)
add_subdirectory(examples)
