# neighbor

A header-only library of GPU neighbor search algorithms based on bounding
volume hierarchies. For additional details, please refer to the following publications:

* M.P. Howard et al. "Efficient neighbor list calculation for molecular
  simulation of colloidal systems using graphics processing units".
  [Comput. Phys. Commun. 203, 45-52 (2016)](https://doi.org/10.1016/j.cpc.2016.02.003).
* M.P. Howard et al. "Quantized bounding volume hierarchies for neighbor
  search in molecular simulations on graphics processing units".
  [Comput. Mater. Sci. 164, 139-146 (2019)](https://doi.org/10.1016/j.cpc.2016.02.003).

## Dependencies

* CMake >= 3.9
* CUDA toolkit and an NVIDIA GPU with compute capability >= 3.5.
* [CUB](https://github.com/NVlabs/cub).

## Compiling

The neighbor library is header-only. It can either be installed via CMake
```
cmake ..
make install
```
or it can be included as part of an existing project.

If the library is being built directly by CMake, benchmarks and tests
can optionally be built.

## Testing

You can validate your installation using the included test suite. To build
the test suite, you must point CMake to the [upp11](https://github.com/DronMDF/upp11)
header-only test library. Then run the tests using `ctest`.

## Versioning

This library is currently under development, so APIs may change between
all minor versions in 0.x. Stability of the API will be guaranteed between
minor versions beginning with 1.x.
