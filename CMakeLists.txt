cmake_minimum_required(VERSION 3.12)
project(tiny_mpi_cxx LANGUAGES CXX)

find_package(MPI REQUIRED)

add_library(tiny_mpi_lib src/tiny_mpi.cpp)
target_include_directories(tiny_mpi_lib PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_compile_features(tiny_mpi_lib PUBLIC cxx_std_20)
target_link_libraries(tiny_mpi_lib PUBLIC MPI::MPI_CXX)

add_library(tiny_mpi::tiny_mpi ALIAS tiny_mpi_lib)
