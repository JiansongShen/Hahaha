project('hahaha', 'cpp',
        version : '0.0.1',
        default_options : [
            'warning_level=3',
            'cpp_std=c++23'
        ])

# Include directories
include_dirs = [
    'core/include',
]

include_dir = include_directories(include_dirs)
fs = import('fs')
py = import('python').find_installation(required : true)

# vcpkg (vendored in this repo)
vcpkg_triplet_opt = get_option('vcpkg_triplet')
vcpkg_installed_abs = meson.project_source_root() / 'vcpkg/vcpkg_root/installed'

vcpkg_triplet = vcpkg_triplet_opt
if vcpkg_triplet_opt == 'auto' or vcpkg_triplet_opt == ''
    vcpkg_triplet = ''

    sys = host_machine.system()
    cpu = host_machine.cpu_family()

    # Prefer likely vcpkg triplets for the current host.
    candidates = []
    if sys == 'linux'
        candidates += ['x64-linux', 'x64-linux-dynamic', 'arm64-linux', 'arm64-linux-dynamic']
    elif sys == 'windows'
        candidates += ['x64-windows', 'x64-windows-static', 'x64-windows-static-md', 'arm64-windows', 'arm64-windows-static', 'arm64-windows-static-md']
    elif sys == 'darwin'
        candidates += ['arm64-osx', 'x64-osx']
    endif

    # Heuristic: if CPU suggests 64-bit x86, keep x64 first; if arm64, prefer arm64.
    if cpu.contains('aarch64') or cpu.contains('arm')
        candidates = ['arm64-osx', 'arm64-linux', 'arm64-windows'] + candidates
    endif

    foreach c : candidates
        if fs.is_dir(vcpkg_installed_abs / c)
            vcpkg_triplet = c
            break
        endif
    endforeach

    # Fallback: if installed/ contains exactly one triplet directory *compatible with this host*, use it.
    # (Repo may carry artifacts for other OSes; never pick those.)
    if vcpkg_triplet == '' and fs.is_dir(vcpkg_installed_abs)
        triplet_guess = run_command(py, '-c',
            'import os,sys\n' +
            'p=sys.argv[1]\n' +
            'sysname=sys.argv[2]\n' +
            'c=[d for d in os.listdir(p) if os.path.isdir(os.path.join(p,d))]\n' +
            'def ok(d):\n' +
            '  if sysname=="linux": return "-linux" in d\n' +
            '  if sysname=="windows": return "-windows" in d\n' +
            '  if sysname=="darwin": return "-osx" in d\n' +
            '  return False\n' +
            'm=[d for d in c if ok(d)]\n' +
            'print(m[0] if len(m)==1 else "")\n',
            vcpkg_installed_abs,
            sys,
            check : false).stdout().strip()
        if triplet_guess != ''
            vcpkg_triplet = triplet_guess
        endif
    endif
endif

vcpkg_prefix_rel = join_paths('vcpkg', 'vcpkg_root', 'installed', vcpkg_triplet)
vcpkg_prefix_abs = meson.project_source_root() / vcpkg_prefix_rel
vcpkg_prefix_found = (vcpkg_triplet != '') and fs.is_dir(vcpkg_prefix_abs)
force_vcpkg_gtest = get_option('force_vcpkg_gtest')

# Dependencies
glfw_dep = dependency('glfw3', required : false)
gl_dep = dependency('gl', required : false)
dl_dep = meson.get_compiler('cpp').find_library('dl', required : false)
cpp_compiler = meson.get_compiler('cpp')

# Select the best C++ standard string for the current compiler
# Meson 1.1.0+ handles c++23, but clang-cl on older Meson needs c++latest
hahaha_cpp_std = 'c++23'
if cpp_compiler.get_id() == 'clang-cl'
    hahaha_cpp_std = 'c++latest'
endif
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
                      'import glob; import os; print("\\n".join(os.path.normpath(f).replace("\\\\", "/") for f in glob.glob("core/src/**/*.cpp", recursive=True)))',
                      check : true).stdout().strip().split('\n')

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


hahaha_deps = stacktrace_dep

imgui_dep = []
if display_enabled
    # Try finding ImGui via system/vcpkg first
    # Note: Meson finds vcpkg packages automatically if you use the vcpkg toolchain file
    # or if pkg-config/cmake can find them.

    # Check for local vcpkg path to help CMake find it without env vars
    imgui_cmake_args = []
    if vcpkg_prefix_found
        imgui_cmake_args += ['-DCMAKE_PREFIX_PATH=' + vcpkg_prefix_abs]
    endif

    # First try: explicit CMake lookup with local vcpkg path if available
    imgui_dep = dependency('imgui',
        required : true,
        method : 'cmake',
        modules : ['imgui::imgui'],
        cmake_args : imgui_cmake_args
    )

    # Second try: generic lookup (system paths, pkg-config, or standard cmake search)
    if not imgui_dep.found()
        imgui_dep = dependency('imgui', required : false)
    endif

    hahaha_deps += imgui_dep
endif


# GTest dependency (needed for tests)
gtest_dep = disabler()

if vcpkg_prefix_found
    # vcpkg puts gtest_main under lib/manual-link (see gtest_main.pc)
    vcpkg_inc = include_directories(join_paths(vcpkg_prefix_rel, 'include'))
    buildtype = get_option('buildtype')
    vcpkg_libdirs = []
    if ['debug', 'debugoptimized'].contains(buildtype)
        vcpkg_libdirs += [vcpkg_prefix_abs / 'debug/lib', vcpkg_prefix_abs / 'debug/lib/manual-link']
    endif
    vcpkg_libdirs += [vcpkg_prefix_abs / 'lib', vcpkg_prefix_abs / 'lib/manual-link']

    lib_gtest = disabler()
    foreach n : ['gtest', 'gtestd']
        lib_gtest = cpp_compiler.find_library(n, dirs : vcpkg_libdirs, required : false)
        if lib_gtest.found()
            break
        endif
    endforeach

    lib_gtest_main = disabler()
    foreach n : ['gtest_main', 'gtest_maind']
        lib_gtest_main = cpp_compiler.find_library(n, dirs : vcpkg_libdirs, required : false)
        if lib_gtest_main.found()
            break
        endif
    endforeach

    if lib_gtest.found() and lib_gtest_main.found()
        gtest_dep = declare_dependency(
            dependencies : [lib_gtest, lib_gtest_main, dependency('threads')],
            include_directories : vcpkg_inc
        )
    endif
endif

if not gtest_dep.found() and not force_vcpkg_gtest
    # CI compatibility:
    # - Linux: system GTest (libgtest-dev, then build/install libs)
    # - Other OSes: may be absent, so we also allow wrap fallback below.
    gtest_dep = dependency('gtest', main : true, required : false)
endif

if not gtest_dep.found()
    # Pure Meson fallback (subproject)
    if fs.exists(join_paths(meson.project_source_root(), 'subprojects', 'gtest.wrap')) or
       fs.is_dir(join_paths(meson.project_source_root(), 'subprojects', 'gtest'))
        gtest_dep = dependency('gtest_',
                               fallback : ['gtest', 'gtest_main_dep'],
                               default_options : ['default_library=static'])
    endif
endif

hahaha_lib = static_library('hahaha', sources,
                            install : true,
                            include_directories : include_dir,
                            cpp_args : lib_args,
                            dependencies : hahaha_deps
)

hahaha = declare_dependency(link_with : hahaha_lib,
                            include_directories : include_dir,
                            dependencies : hahaha_deps)

subdir('tests')
subdir('examples')
