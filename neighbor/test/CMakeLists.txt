# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

###################################
## Setup all of the test executables
#
# Example for foo_bar_test.cc:
#   foo_bar
set(TEST_LIST
    lbvh
    uniform_grid
    )
###################################

macro(compile_test TEST_EXE TEST_SRC)
    # add and link the unit test executable
    add_executable(${TEST_EXE} EXCLUDE_FROM_ALL ${TEST_SRC})
    add_dependencies(test_all ${TEST_EXE})
    target_link_libraries(${TEST_EXE} PUBLIC _neighbor HOOMD::_hoomd pybind11::embed)

    fix_cudart_rpath(${TEST_EXE})
endmacro(compile_test)

# add tests
foreach(CUR_TEST ${TEST_LIST})
    set(CUR_TEST_SRC ${CUR_TEST}_test.cc)
    set(CUR_TEST_EXE ${CUR_TEST}_test)
    set(CUR_TEST_NAME ${CUR_TEST})

    compile_test(${CUR_TEST_EXE} ${CUR_TEST_SRC})
    if(ENABLE_MPI)
        add_test(NAME neighbor-${CUR_TEST_NAME} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_POSTFLAGS} $<TARGET_FILE:${CUR_TEST_EXE}>)
    else()
        add_test(NAME neighbor-${CUR_TEST_NAME} COMMAND $<TARGET_FILE:${CUR_TEST_EXE}>)
    endif()
endforeach(CUR_TEST)
