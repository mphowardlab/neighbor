# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

set(BENCH_LIST
    lbvh
    uniform_grid
    )

macro(compile_benchmark BENCH_EXE BENCH_SRC)
    add_executable(${BENCH_EXE} ${BENCH_SRC})
    target_link_libraries(${BENCH_EXE} ${HOOMD_LIBRARIES} ${PROJECT_NAME} ${PYTHON_LIBRARIES})
    fix_cudart_rpath(${BENCH_EXE})

    if(ENABLE_MPI)
        # set appropriate compiler/linker flags
        if(MPI_CXX_COMPILE_FLAGS)
            set_target_properties(${BENCH_EXE} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
        endif(MPI_CXX_COMPILE_FLAGS)
        if(MPI_CXX_LINK_FLAGS)
            set_target_properties(${BENCH_EXE} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
        endif(MPI_CXX_LINK_FLAGS)
    endif(ENABLE_MPI)

    install(TARGETS ${BENCH_EXE} DESTINATION bin)
endmacro(compile_benchmark)

foreach(CUR_BENCH ${BENCH_LIST})
    set(CUR_BENCH_SRC ${CUR_BENCH}.cc)
    set(CUR_BENCH_EXE benchmark_${CUR_BENCH})

    compile_benchmark(${CUR_BENCH_EXE} ${CUR_BENCH_SRC})
endforeach(CUR_BENCH)
