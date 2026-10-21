# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2026, Auburn University
# This file is released under the Modified BSD License.

if(NEIGHBOR_HIP)
    message(FATAL_ERROR "Benchmarks cannot be built with HIP.")
endif()

add_library(gsd STATIC ${CMAKE_SOURCE_DIR}/extern/gsd/gsd/gsd.c)
target_include_directories(gsd PUBLIC ${CMAKE_SOURCE_DIR}/extern/gsd)

find_package(CUDAToolkit REQUIRED)

add_executable(lbvh_benchmark lbvh_benchmark.cc lbvh_benchmark.cu)
set_source_files_properties(lbvh_benchmark.cc PROPERTIES LANGUAGE CUDA)
target_compile_definitions(lbvh_benchmark PRIVATE HIPPER_CUDA)
target_link_libraries(lbvh_benchmark PRIVATE neighbor::neighbor gsd CUDA::cudart)

install(TARGETS lbvh_benchmark
        DESTINATION ${CMAKE_INSTALL_BINDIR})
