cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
set (CMAKE_CXX_STANDARD 14)

find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIR})
add_definitions(${HDF5_CPPFLAGS})

add_executable(00_simple_model 00_simple_model.cxx)
add_executable(01_simple_model_double 01_simple_model_double.cxx)
add_executable(02_simple_model_min_sum 02_simple_model_min_sum.cxx)
target_link_libraries(01_simple_model_double ${HDF5_LIBRARIES})
target_link_libraries(02_simple_model_min_sum ${HDF5_LIBRARIES})
