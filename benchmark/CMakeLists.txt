# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021, Auburn University
# This file is released under the Modified BSD License.

if(NEIGHBOR_HIP)
    message(FATAL_ERROR "Benchmarks cannot be built with HIP.")
else()
    enable_language(CUDA)
endif()

find_package(HOOMD REQUIRED)

add_executable(lbvh_benchmark lbvh_benchmark.cc lbvh_benchmark.cu)
target_link_libraries(lbvh_benchmark PRIVATE neighbor::neighbor HOOMD::_hoomd pybind11::embed)

install(TARGETS lbvh_benchmark
        DESTINATION ${CMAKE_INSTALL_BINDIR})
