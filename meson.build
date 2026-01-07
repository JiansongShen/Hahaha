    project('hahaha', 'cpp',
        version : '0.0.1',
        default_options : [
                'warning_level=3',
                'cpp_std=c++23'])

# Include directories
include_dirs = [
    'core/include',
]
include_dir = include_directories(include_dirs)
fs = import('fs')
py = import('python').find_installation(required: true)

# Dependencies
glfw_dep = dependency('glfw3', required : false)
gl_dep = dependency('gl', required : false)
dl_dep = meson.get_compiler('cpp').find_library('dl', required : false)
cpp_compiler = meson.get_compiler('cpp')
dl_dep = cpp_compiler.find_library('dl', required : false)

stacktrace_dep = []
if ['gcc', 'clang'].contains(cpp_compiler.get_id())
    # Try common libraries that might be needed for std::stacktrace.
    #
    # Note:
    # - On libstdc++ (GCC toolchain), std::stacktrace implementation may live in
    #   a separate library (often 'stdc++exp' or 'stdc++_libbacktrace').
    # - This can also affect Clang builds that link against libstdc++.
    foreach lib : ['stdc++_libbacktrace', 'stdc++exp', 'backtrace']
        l = cpp_compiler.find_library(lib, required : false)
        if l.found()
            stacktrace_dep += l
        endif
    endforeach
endif

# These arguments are only used to build the shared library
lib_args = ['-DBUILDING_MESON_LIBRARY']

sources = run_command(py, '-c',
                      'import glob; print("\\n".join(glob.glob("core/src/**/*.cpp", recursive=True)))',
                      check: true).stdout().strip().split('\n')

display_enabled = get_option('display')

if not display_enabled
  # Headless build: drop display sources that depend on GLFW/OpenGL/X11.
  filtered_sources = []
  foreach s : sources
    if s != '' and not s.startswith('core/src/display/') and not s.startswith('core/include/display/')
      filtered_sources += s
    endif
  endforeach
  sources = filtered_sources
endif



imgui_dep = []
if display_enabled
  # Use Meson wrap subproject for ImGui (no vendored extern/ tree).
  #
  # IMPORTANT:
  # The WrapDB ImGui subproject defaults many backends to 'auto'. If Vulkan is
  # detected (even partially), it may try to compile imgui_impl_vulkan.cpp,
  # which requires <vulkan/vulkan.h>. To keep display builds lightweight and
  # CI-friendly, we only enable the backends we actually use (GLFW + OpenGL3).
  imgui_sp = subproject('imgui',
                        default_options: [
                          'opengl=enabled',
                          'glfw=enabled',

                          'vulkan=disabled',
                          'webgpu=disabled',

                          'dx9=disabled',
                          'dx10=disabled',
                          'dx11=disabled',
                          'dx12=disabled',
                          'metal=disabled',

                          'sdl2_renderer=disabled',
                          'sdl3_renderer=disabled',
                          'sdl2=disabled',
                          'sdl3=disabled',

                          'osx=disabled',
                          'win=disabled',
                          'allegro5=disabled',
                        ])
  imgui_dep = imgui_sp.get_variable('imgui_dep')
endif

hahaha_deps = [dl_dep] + stacktrace_dep
if display_enabled
  hahaha_deps += [glfw_dep, gl_dep, imgui_dep]
endif

hahaha_lib = static_library('hahaha', sources,
                            install : true,
                            include_directories : include_dir,
                            cpp_args : lib_args,
                            dependencies : hahaha_deps
)

hahaha = declare_dependency(link_with: hahaha_lib,
                            include_directories: include_dir,
                            dependencies: hahaha_deps)

subdir('tests')
subdir('examples')
