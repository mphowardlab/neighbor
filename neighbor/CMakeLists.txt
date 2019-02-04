# Copyright (c) 2018, Michael P. Howard.
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
    LBVH.cuh
    LBVH.h
    LBVHTraverser.cuh
    LBVHTraverser.h
    OutputOps.h
    UniformGrid.cuh
    UniformGrid.h
    UniformGridTraverser.cuh
    UniformGridTraverser.h
    )
###########################

# compile cuda sources
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)

# setup library with cc sources
add_library(${PROJECT_NAME} SHARED ${_cc_sources} ${_CUDA_GENERATED_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${HOOMD_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${_h_sources}")
# if we are compiling with MPI support built in, set appropriate compiler/linker flags
if (ENABLE_MPI)
   if(MPI_CXX_COMPILE_FLAGS)
       set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_CXX_COMPILE_FLAGS)
   if(MPI_CXX_LINK_FLAGS)
       set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_CXX_LINK_FLAGS)
endif(ENABLE_MPI)
fix_cudart_rpath(${PROJECT_NAME})

# install the library
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/${PROJECT_NAME})

if(BUILD_TESTING)
    add_subdirectory(test)
endif(BUILD_TESTING)
