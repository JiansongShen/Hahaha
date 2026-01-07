# tests/meson.build
test_include_dir = include_directories('../core/include')

test_sources = run_command('find', '.', '-name', '*.cpp', '-type', 'f', check: true).stdout().strip().split('\n')

gtest_dep = dependency('gtest', main : true, required : false)
if not gtest_dep.found()
    gtest_dep = dependency('gtest', fallback : ['googletest', 'gtest_main_dep'])
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
