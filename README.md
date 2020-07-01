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

* CMake >= 3.9

To use with NVIDIA GPUs:

* [CUDA](https://docs.nvidia.com/cuda) toolkit.
* [CUB](https://github.com/NVlabs/cub).

To use with AMD GPUs:

* [HIP](https://github.com/ROCm-Developer-Tools/HIP).
* [hipCUB](https://github.com/ROCmSoftwarePlatform/hipCUB).

Optionally:

* [hipper](https://github.com/mphowardlab/hipper) header-only GPU runtime
  interoperability layer. If not found, it will be added as a git submodule.
* [upp11](https://github.com/DronMDF/upp11) header-only test library. If
  not found, it will be downloaded automatically to the build directory.

## Compiling

The neighbor library is header-only. It can be installed via CMake:
```
cmake ..
make install
```
(It can be included as part of an existing project, or simply downloaded.)
CMake projects can discover the installation using `find_package` in `CONFIG`
mode and should link against the `neighbor::neighbor` target.

neighbor will select between a CUDA or HIP build using the `NEIGHBOR_HIP` define.
If `NEIGHBOR_HIP=ON`, neighbor will use HIP for the GPU runtime, and both
NVIDIA and AMD GPUs are supported. If `NEIGHBOR_HIP=OFF`, neighbor will use
CUDA for the GPU runtime. If only NVIDIA GPUs are targeted, performance may
be slightly improved by setting `NEIGHBOR_HIP=OFF`.

## Testing

If neighbor is being built directly by CMake, tests can optionally be built
with a working CUDA or HIP installation by setting `NEIGHBOR_TEST=ON`. Run the
included test suite with `ctest`.

## Versioning

This library is currently under development, so APIs may change between
all minor versions in 0.x. Stability of the API will be guaranteed between
minor versions beginning with 1.x.
