# tests/meson.build
test_include_dir = include_directories('../core/include')

py = import('python').find_installation(required : true)
test_sources = run_command(py,
                           '../dev/meson/list_sources.py',
                           '--base',
                           meson.current_source_dir(),
                           '--glob',
                           '**/*.cpp',
                           check : true).stdout().strip().split('\n')


if not gtest_dep.found()
    error('GTest dependency not found. Supported providers:\n' +
          '- repo-local vcpkg: vcpkg/vcpkg_root/installed/<triplet>\n' +
          '- system GTest (commonly used on Linux CI)\n' +
          '- Meson wrap: run `meson subprojects download gtest` (commonly used on macOS/Windows CI)\n' +
          'If multiple vcpkg triplets exist, pass -Dvcpkg_triplet=<triplet>.')
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

# MSVC requires /bigobj for large object files (e.g., from TYPED_TEST templates)
msvc_cpp_args = []
if cpp_compiler.get_id() == 'msvc'
    msvc_cpp_args += ['/bigobj']
endif

test_executable = executable('hahaha_tests', test_sources,
                             include_directories : [test_include_dir, include_dir],
                             cpp_args : cov_cpp_args + msvc_cpp_args,
                             link_args : cov_link_args,
                             dependencies : [gtest_dep, hahaha])

test('hahaha_unit_tests', test_executable)
