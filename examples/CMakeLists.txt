file(GLOB_RECURSE HAHAHA_EXAMPLE_AUTOGRAD_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/autograd/*.cpp"
)
add_executable(hahaha_example_autograd ${HAHAHA_EXAMPLE_AUTOGRAD_SOURCES})
target_link_libraries(hahaha_example_autograd PRIVATE hahaha::hahaha)

file(GLOB_RECURSE HAHAHA_EXAMPLE_BASIC_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/basic_usage/*.cpp"
)
add_executable(hahaha_example_tensor_basic_usage ${HAHAHA_EXAMPLE_BASIC_SOURCES})
target_link_libraries(hahaha_example_tensor_basic_usage PRIVATE hahaha::hahaha)

file(GLOB_RECURSE HAHAHA_EXAMPLE_ML_BASIC_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/ml_basic_usage/*.cpp"
)
add_executable(hahaha_example_ml_basic_usage ${HAHAHA_EXAMPLE_ML_BASIC_SOURCES}
        ml_basic_usage/basic_adam_optimizer_example.h
        ml_basic_usage/basic_adam_optimizer_example.h)
target_link_libraries(hahaha_example_ml_basic_usage PRIVATE hahaha::hahaha)

if(HAHAHA_DISPLAY)
  add_executable(hahaha_example_ml_visualizer
    "${CMAKE_CURRENT_SOURCE_DIR}/ml_visualizer/ml_visualizer_main.cpp"
  )
  target_link_libraries(hahaha_example_ml_visualizer PRIVATE hahaha::hahaha)
endif()
