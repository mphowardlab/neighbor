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

For installation:

* CMake >= 3.21

To use with NVIDIA GPUs:

* [CUDA](https://docs.nvidia.com/cuda) toolkit.

Optional:

* [upp11](https://github.com/DronMDF/upp11) header-only test library. If
  not found, it will be downloaded automatically to the build directory when
  `NEIGHBOR_TEST=ON`.

## Compiling

The neighbor library is header-only. It can be installed via CMake:

```bash
cmake ..
make install
```

(It can be included as part of an existing project, or simply downloaded.)
CMake projects can discover the installation using `find_package` in `CONFIG`
mode and should link against the `neighbor::neighbor` target.

## Testing

If neighbor is being built directly by CMake, tests can optionally be built with
a working CUDA installation by setting `NEIGHBOR_TEST=ON`. Run the included test
suite with `ctest`.

## Versioning

This library is currently under development, so APIs may change between
all minor versions in 0.x. Stability of the API will be guaranteed between
minor versions beginning with 1.x.
