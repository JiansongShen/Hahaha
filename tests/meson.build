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
    # Pure Meson fallback:
    # Prefer a Meson subproject. In CI, install it via:
    #   meson wrap install gtest
    fs = import('fs')
    if fs.is_dir(join_paths(meson.project_source_root(), 'subprojects'))
        # WrapDB project name is "gtest" (directory may be googletest-<ver>).
        gtest_dep = dependency('gtest', fallback : ['gtest', 'gtest_main_dep'])
    endif

    if not gtest_dep.found()
        error('GTest dependency not found. Install system gtest, or run `meson wrap install gtest` to enable the Meson subproject fallback.')
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
