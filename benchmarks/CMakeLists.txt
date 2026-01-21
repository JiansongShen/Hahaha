file(GLOB_RECURSE HAHAHA_BENCHMARK_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

if (HAHAHA_USE_CUDA)
add_executable(hahaha_benchmark ${HAHAHA_BENCHMARK_SOURCES})
target_link_libraries(hahaha_benchmark PRIVATE hahaha::hahaha)

# Include CUDA directories if CUDA is enabled
if(HAHAHA_USE_CUDA AND TARGET CUDA::cudart)
  target_include_directories(hahaha_benchmark PRIVATE
    $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

# MSVC: large translation units may require /bigobj.
if(MSVC)
  target_compile_options(hahaha_benchmark PRIVATE /bigobj)
endif()
endif ()
