# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

##############################
# List of all source files
# c++ source files
set(_cc_sources
    LBVH.cc
    LBVHTraverser.cc
    UniformGrid.cc
    UniformGridTraverser.cc
    )
# cuda source files
set(_cu_sources
    LBVH.cu
    LBVHTraverser.cu
    UniformGrid.cu
    UniformGridTraverser.cu
    )
# headers
set(_h_sources
    BoundingVolumes.h
    InsertOps.h
    LBVH.cuh
    LBVH.h
    LBVHTraverser.cuh
    LBVHTraverser.h
    OutputOps.h
    QueryOps.h
    TransformOps.h
    UniformGrid.cuh
    UniformGrid.h
    UniformGridTraverser.cuh
    UniformGridTraverser.h
    )
###########################

# compile cuda sources
if(THRUST_FIX)
  list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX THRUST_CUB_WRAPPED_NAMESPACE)
  list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX HOOMD_THRUST)
  list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX HOOMD_CUB)
  list(APPEND CUDA_NVCC_FLAGS "-DTHRUST_CUB_WRAPPED_NAMESPACE=THRUST_CUB_NEIGHBOR")
  list(APPEND CUDA_NVCC_FLAGS "-DHOOMD_THRUST=THRUST_CUB_NEIGHBOR::thrust")
  list(APPEND CUDA_NVCC_FLAGS "-DHOOMD_CUB=THRUST_CUB_NEIGHBOR::cub")
endif()
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)

# setup library with cc sources
add_library(neighbor SHARED ${_cc_sources} ${_CUDA_GENERATED_FILES})
target_link_libraries(neighbor PRIVATE ${HOOMD_LIBRARIES})
# if we are compiling with MPI support built in, set appropriate compiler/linker flags
if (ENABLE_MPI)
   if(MPI_CXX_COMPILE_FLAGS)
       set_target_properties(neighbor PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_CXX_COMPILE_FLAGS)
   if(MPI_CXX_LINK_FLAGS)
       set_target_properties(neighbor PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_CXX_LINK_FLAGS)
endif(ENABLE_MPI)
fix_cudart_rpath(neighbor)

# install the library
if (HOOMD_INTERNAL)
    install(TARGETS neighbor
            LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR})
else()
    set_target_properties(neighbor PROPERTIES PUBLIC_HEADER "${_h_sources}")
    install(TARGETS neighbor
            LIBRARY DESTINATION lib
            PUBLIC_HEADER DESTINATION include/neighbor)
endif()

if(BUILD_TESTING)
    add_subdirectory(test)
endif(BUILD_TESTING)
