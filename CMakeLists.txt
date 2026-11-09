cmake_minimum_required(VERSION 3.21)
project(gs_bench LANGUAGES CXX)

include(FetchContent)

# ---------------------------------------------------------------------------
# Build settings
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GS_GPU_BACKEND "CPU" CACHE STRING "Kokkos backend: CPU, CUDA, or HIP")
set_property(CACHE GS_GPU_BACKEND PROPERTY STRINGS CPU CUDA HIP)
option(GS_FETCH_KOKKOS "Fetch Kokkos when it is not installed" ON)
set(GS_KOKKOS_TAG "5.2.2" CACHE STRING "Kokkos git tag used by FetchContent")
set(GS_KOKKOS_ARCH "" CACHE STRING
    "Kokkos architecture suffix for a fetched Kokkos, e.g. AMD_GFX90A (MI250), AMPERE80, HOPPER90")
set(GS_MPI_EXTRA_LINK_FLAGS "" CACHE STRING
    "Extra link items for GPU-aware MPI, ';'-separated (e.g. Cray MPICH GTL: -L<dir>;-lmpi_gtl_hsa)")

if(NOT GS_GPU_BACKEND MATCHES "^(CPU|CUDA|HIP)$")
  message(FATAL_ERROR "GS_GPU_BACKEND must be CPU, CUDA, or HIP (got '${GS_GPU_BACKEND}')")
endif()

# Kokkos device name that has to be present for each backend
set(_gs_device_CPU OPENMP)
set(_gs_device_CUDA CUDA)
set(_gs_device_HIP HIP)
set(GS_REQUIRED_DEVICE ${_gs_device_${GS_GPU_BACKEND}})

if(GS_GPU_BACKEND STREQUAL "HIP" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(WARNING "HIP backend: use a ROCm compiler, e.g. -DCMAKE_CXX_COMPILER=hipcc (or amdclang++)")
endif()

# ---------------------------------------------------------------------------
# MPI
# ---------------------------------------------------------------------------
find_package(MPI REQUIRED COMPONENTS CXX)

# ---------------------------------------------------------------------------
# Kokkos: use an installed one if it provides the requested backend, else fetch
# ---------------------------------------------------------------------------
find_package(Kokkos 4.0 CONFIG QUIET)
if(Kokkos_FOUND)
  if(NOT GS_REQUIRED_DEVICE IN_LIST Kokkos_DEVICES)
    message(FATAL_ERROR
      "Installed Kokkos (${Kokkos_DIR}) has devices '${Kokkos_DEVICES}' but backend "
      "${GS_GPU_BACKEND} needs ${GS_REQUIRED_DEVICE}. Point Kokkos_ROOT to a matching install "
      "or hide this one (e.g. -DCMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON to fetch instead).")
  endif()
  message(STATUS "Using installed Kokkos ${Kokkos_VERSION} from ${Kokkos_DIR} (devices: ${Kokkos_DEVICES})")
else()
  if(NOT GS_FETCH_KOKKOS)
    message(FATAL_ERROR "Kokkos was not found and GS_FETCH_KOKKOS is OFF")
  endif()
  # Backend options only matter for a fetched Kokkos, so they are set here.
  if(GS_GPU_BACKEND STREQUAL "CPU")
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
  elseif(GS_GPU_BACKEND STREQUAL "CUDA")
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
  else()
    set(Kokkos_ENABLE_HIP ON CACHE BOOL "" FORCE)
  endif()
  if(GS_KOKKOS_ARCH)
    set(Kokkos_ARCH_${GS_KOKKOS_ARCH} ON CACHE BOOL "" FORCE)
  elseif(NOT GS_GPU_BACKEND STREQUAL "CPU")
    message(WARNING "GS_KOKKOS_ARCH is empty: Kokkos will try to auto-detect the GPU architecture")
  endif()
  message(STATUS "Fetching Kokkos ${GS_KOKKOS_TAG} (backend ${GS_GPU_BACKEND})")
  FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG ${GS_KOKKOS_TAG}
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(kokkos)
endif()

# ---------------------------------------------------------------------------
# NCCL / RCCL (GPU backends only)
# ---------------------------------------------------------------------------
add_library(gs_comm INTERFACE)
if(GS_GPU_BACKEND STREQUAL "CUDA")
  find_path(NCCL_INCLUDE_DIR nccl.h
            HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{NCCL_HOME} $ENV{NCCL_DIR}
            PATH_SUFFIXES include REQUIRED)
  find_library(NCCL_LIBRARY nccl
               HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{NCCL_HOME} $ENV{NCCL_DIR}
               PATH_SUFFIXES lib lib64 REQUIRED)
  target_include_directories(gs_comm SYSTEM INTERFACE ${NCCL_INCLUDE_DIR})
  target_link_libraries(gs_comm INTERFACE ${NCCL_LIBRARY})
  message(STATUS "NCCL: ${NCCL_LIBRARY}")
elseif(GS_GPU_BACKEND STREQUAL "HIP")
  # Code includes <rccl/rccl.h>, so the include dir is the one that contains rccl/.
  find_path(RCCL_INCLUDE_DIR rccl/rccl.h
            HINTS ${ROCM_PATH} $ENV{ROCM_PATH} $ENV{ROCM_HOME} /opt/rocm
            PATH_SUFFIXES include REQUIRED)
  find_library(RCCL_LIBRARY rccl
               HINTS ${ROCM_PATH} $ENV{ROCM_PATH} $ENV{ROCM_HOME} /opt/rocm
               PATH_SUFFIXES lib lib64 REQUIRED)
  target_include_directories(gs_comm SYSTEM INTERFACE ${RCCL_INCLUDE_DIR})
  target_link_libraries(gs_comm INTERFACE ${RCCL_LIBRARY})
  message(STATUS "RCCL: ${RCCL_LIBRARY}")
endif()

# ---------------------------------------------------------------------------
# Common options target
# ---------------------------------------------------------------------------
add_library(gs_bench_options INTERFACE)
target_include_directories(gs_bench_options INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gs_bench_options INTERFACE MPI::MPI_CXX Kokkos::kokkos gs_comm)
if(GS_MPI_EXTRA_LINK_FLAGS)
  target_link_libraries(gs_bench_options INTERFACE ${GS_MPI_EXTRA_LINK_FLAGS})
endif()
target_compile_features(gs_bench_options INTERFACE cxx_std_17)
target_compile_options(gs_bench_options INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)

function(gs_add_benchmark target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE gs_bench_options)
endfunction()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
foreach(name IN ITEMS mpi_isend_irecv mpi_persistent)
  gs_add_benchmark(gs_${name} benchmarks/${name}.cpp)
endforeach()

if(NOT GS_GPU_BACKEND STREQUAL "CPU")
  foreach(name IN ITEMS nccl_rccl_1com nccl_rccl_2com)
    gs_add_benchmark(gs_${name} benchmarks/${name}.cpp)
  endforeach()
endif()

# Self-contained MPI-vs-RCCL halo benchmark (HIP only, built if the file is present).
if(GS_GPU_BACKEND STREQUAL "HIP" AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/gs_halo_bench.cpp)
  gs_add_benchmark(gs_halo_bench benchmarks/gs_halo_bench.cpp)
endif()