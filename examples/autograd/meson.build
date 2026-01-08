# examples/autograd/meson.build
example_include_dir = include_directories('../../core/include')

py = import('python').find_installation(required: true)
example_sources = run_command(py,
                              '../../dev/meson/list_sources.py',
                              '--base',
                              meson.current_source_dir(),
                              '--glob',
                              '**/*.cpp',
                              check: true).stdout().strip().split('\n')


example_executable = executable('hahaha_example_autograd', example_sources,
                           include_directories: [example_include_dir, include_dir],
                           dependencies: [hahaha])
