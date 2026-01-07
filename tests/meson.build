# tests/meson.build
test_include_dir = include_directories('../core/include')

py = import('python').find_installation(required: true)
test_sources = run_command(py,
                           '../dev/meson/list_sources.py',
                           '--base',
                           meson.current_source_dir(),
                           '--glob',
                           '**/*.cpp',
                           check: true).stdout().strip().split('\n')

gtest_dep = dependency('gtest', main : true, required : false)
if not gtest_dep.found()
    # Pure Meson fallback (no CMake):
    # If CI installed gtest via vcpkg, it usually sets CMAKE_PREFIX_PATH to:
    #   <vcpkg_root>/installed/<triplet>
    # We can use that prefix to locate headers and .lib/.a via find_library().
    cpp = meson.get_compiler('cpp')
    prefix = meson.get_env('CMAKE_PREFIX_PATH')
    gtest_found = false

    if prefix != ''
        inc_dir = include_directories(join_paths(prefix, 'include'))
        lib_dir = join_paths(prefix, 'lib')

        gtest_main_lib = cpp.find_library('gtest_main', dirs : [lib_dir], required : false)
        gtest_lib = cpp.find_library('gtest', dirs : [lib_dir], required : false)

        if gtest_main_lib.found()
            # Prefer gtest_main if available; it pulls gtest too on most toolchains.
            gtest_dep = declare_dependency(include_directories : inc_dir,
                                           dependencies : [gtest_main_lib, gtest_lib])
            gtest_found = true
        elif gtest_lib.found()
            gtest_dep = declare_dependency(include_directories : inc_dir,
                                           dependencies : [gtest_lib])
            gtest_found = true
        endif
    endif

    if not gtest_found
        # Optional fallback to subproject if the repo includes subprojects/.
        fs = import('fs')
        if fs.is_dir(join_paths(meson.project_source_root(), 'subprojects'))
            gtest_dep = dependency('gtest', fallback : ['googletest', 'gtest_main_dep'])
            gtest_found = true
        endif
    endif

    if not gtest_found
        error('GTest dependency not found. Install gtest and/or set CMAKE_PREFIX_PATH to a prefix that contains include/gtest and lib/{gtest,gtest_main}.')
    endif
endif

# Optional coverage support:
# Enable with: meson setup builddir -Db_coverage=true
cov_cpp_args = []
cov_link_args = []
cpp_compiler = meson.get_compiler('cpp')
if get_option('b_coverage')
    if cpp_compiler.has_link_argument('--coverage')
        cov_cpp_args += ['--coverage']
        cov_link_args += ['--coverage']
    else
        # Fallback for toolchains that don't accept --coverage
        cov_cpp_args += ['-fprofile-arcs', '-ftest-coverage']
        cov_link_args += ['-lgcov']
    endif
endif

test_executable = executable('hahaha_tests', test_sources,
                           include_directories: [test_include_dir, include_dir],
                           cpp_args: cov_cpp_args,
                           link_args: cov_link_args,
                           link_with: hahaha_lib,
                           dependencies: [gtest_dep])

test('hahaha_unit_tests', test_executable)
