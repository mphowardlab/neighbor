# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

set(BENCH_LIST
    lbvh
    uniform_grid
    )

macro(compile_benchmark BENCH_EXE BENCH_SRC)
    add_executable(${BENCH_EXE} ${BENCH_SRC})
    target_link_libraries(${BENCH_EXE} PUBLIC HOOMD::_hoomd _${PROJECT_NAME} pybind11::embed)
    fix_cudart_rpath(${BENCH_EXE})

    install(TARGETS ${BENCH_EXE} DESTINATION bin)
endmacro(compile_benchmark)

foreach(CUR_BENCH ${BENCH_LIST})
    set(CUR_BENCH_SRC ${CUR_BENCH}.cc)
    set(CUR_BENCH_EXE benchmark_${CUR_BENCH})

    compile_benchmark(${CUR_BENCH_EXE} ${CUR_BENCH_SRC})
endforeach(CUR_BENCH)
