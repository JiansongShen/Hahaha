# examples/meson.build
example_include_dir = include_directories('../core/include')

example_sources = run_command('find', '.', '-name', '*.cpp', '-type', 'f', check: true).stdout().strip().split('\n')

subdir('autograd')
subdir('basic_usage')
subdir('ml_basic_usage')

# Visualization example depends on GLFW/OpenGL/ImGui. Disable it in headless builds.
if get_option('display')
  subdir('ml_visualizer')
endif
