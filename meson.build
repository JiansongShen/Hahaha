    project('hahaha', 'cpp',
        version : '0.0.1',
        default_options : [
                'warning_level=3',
                'cpp_std=c++23'])

# Include directories
include_dirs = [
    'core/include',
    'extern/externlibs/imgui',
    'extern/externlibs/imgui/backends',
    'extern/externlibs/imgui/examples/libs/glfw/include' # Fallback for GLFW
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

# Conditionally add ImGui backends (only when display is enabled)
imgui_backend_sources = []
if display_enabled
  # Even if dependency is not found by pkg-config, we try to compile if headers are present
  # On Arch, the user might have them but pkg-config might not be set up in this environment
  imgui_backend_sources += 'extern/externlibs/imgui/backends/imgui_impl_glfw.cpp'
  imgui_backend_sources += 'extern/externlibs/imgui/backends/imgui_impl_opengl3.cpp'
endif

extern_sources = []
if display_enabled
  # Exclude backends from general extern_sources to avoid duplicates.
  # We use Python glob here for cross-platform source discovery (Windows doesn't have GNU find).
  extern_glob_code = '''
import glob
paths = glob.glob("extern/**/*.cpp", recursive=True)
def keep(p):
  p = p.replace("\\\\", "/")
  if "/examples/" in p: return False
  if "/backends/" in p: return False
  if "/misc/freetype/" in p: return False
  if "/misc/fonts/" in p: return False
  return True
print("\\n".join([p for p in paths if keep(p)]))
'''
  extern_sources = run_command(py, '-c', extern_glob_code,
                               check: true).stdout().strip().split('\n')
endif

hahaha_deps = [dl_dep] + stacktrace_dep
if display_enabled
  hahaha_deps += [glfw_dep, gl_dep]
endif

hahaha_lib = static_library('hahaha', sources + extern_sources + imgui_backend_sources,
                            install : true,
                            include_directories : include_dir,
                            cpp_args : lib_args,
                            dependencies : hahaha_deps
)

hahaha = declare_dependency(link_with: hahaha_lib,
                            include_directories: include_dir)

subdir('tests')
subdir('examples')
