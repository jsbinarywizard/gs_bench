cmake_minimum_required(VERSION 3.21)
project(gs_bench LANGUAGES CXX)

include(FetchContent)

find_package(MPI REQUIRED COMPONENTS CXX)

set(GS_GPU_BACKEND "CPU" CACHE STRING "Kokkos backend: CPU, CUDA, or HIP")
set_property(CACHE GS_GPU_BACKEND PROPERTY STRINGS CPU CUDA HIP)
option(GS_FETCH_KOKKOS "Fetch Kokkos when it is not installed" ON)
set(GS_KOKKOS_TAG "4.6.01" CACHE STRING "Kokkos git tag used by FetchContent")

if(GS_GPU_BACKEND STREQUAL "CPU")
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
elseif(GS_GPU_BACKEND STREQUAL "CUDA")
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
elseif(GS_GPU_BACKEND STREQUAL "HIP")
  set(Kokkos_ENABLE_HIP ON CACHE BOOL "" FORCE)
else()
  message(FATAL_ERROR "GS_GPU_BACKEND must be CPU, CUDA, or HIP")
endif()

find_package(Kokkos CONFIG QUIET)
if(NOT Kokkos_FOUND)
  if(NOT GS_FETCH_KOKKOS)
    message(FATAL_ERROR "Kokkos was not found and GS_FETCH_KOKKOS is OFF")
  endif()
  FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG ${GS_KOKKOS_TAG}
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(kokkos)
endif()

if(GS_GPU_BACKEND STREQUAL "CUDA")
  find_package(CUDAToolkit REQUIRED)
  find_path(NCCL_INCLUDE_DIR nccl.h REQUIRED)
  find_library(NCCL_LIBRARY nccl REQUIRED)
  set(GS_COMM_LIBRARY ${NCCL_LIBRARY})
  set(GS_COMM_INCLUDE_DIR ${NCCL_INCLUDE_DIR})
elseif(GS_GPU_BACKEND STREQUAL "HIP")
  find_path(RCCL_INCLUDE_DIR rccl/rccl.h REQUIRED)
  find_library(RCCL_LIBRARY rccl REQUIRED)
  set(GS_COMM_LIBRARY ${RCCL_LIBRARY})
  set(GS_COMM_INCLUDE_DIR ${RCCL_INCLUDE_DIR})
endif()

add_library(gs_bench_options INTERFACE)
target_include_directories(gs_bench_options INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(GS_COMM_INCLUDE_DIR)
  target_include_directories(gs_bench_options SYSTEM INTERFACE ${GS_COMM_INCLUDE_DIR})
endif()
target_link_libraries(gs_bench_options INTERFACE MPI::MPI_CXX Kokkos::kokkos ${GS_COMM_LIBRARY})
target_compile_features(gs_bench_options INTERFACE cxx_std_17)
target_compile_options(gs_bench_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

foreach(name IN ITEMS mpi_isend_irecv mpi_persistent)
  add_executable(gs_${name} benchmarks/${name}.cpp)
  target_link_libraries(gs_${name} PRIVATE gs_bench_options)
endforeach()

if(NOT GS_GPU_BACKEND STREQUAL "CPU")
  add_executable(gs_nccl_rccl benchmarks/nccl_rccl.cpp)
  target_link_libraries(gs_nccl_rccl PRIVATE gs_bench_options)
endif()
