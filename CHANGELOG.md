# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2020-10-16
### Changed
- C++14 standard is now required.
- Use hipper 0.2.0 runtime.
- Support CMake 3.18 syntax for setting CUDA architecture.
### Removed
- SM 3.5 devices from default list of CUDA architectures.

## [0.3.0] - 2020-07-01
### Added
- Use modern CMake build system.
- Support AMD GPUs using the HIP runtime. To ensure compatability with CUDA-only builds,
  the hipper runtime acts as an interoperability layer.
- Support approximate math without CUDA intrinsics.
- Support user-defined translation operations. The self-image is no longer included by default.

### Changed
- neighbor is now a header-only library.
- HOOMD-blue is no longer a dependency of neighbor, except for benchmarking. Benchmarks
  are disabled by default. All external data should now be managed by the user and can
  be passed as a pointer. Use `neighbor::shared_array` as a replacement for HOOMD
  GPUArray or GlobalArray if you do not have your own data container. It is a wrapper around
  a GPU managed array.
- Autotuning is deferred outside all objects, to ensure fully asynchronous execution in streams.
  Use the `getTunableParameters` method to access the valid parameters, and manage the tuning
  using preferred method.

### Removed
- UniformGrid methods. Their API was too restrictive compared to the LBVH.

## [0.2.0] - 2019-11-21
### Added
- The LBVH build and traversal are now templated for flexible programs.
- The UniformGrid has been simplified and given a unified API with the LBVH.
- CUDA streams are supported for building and traversing. The build may
  not be fully asynchronous, as there is a CUDA stream synchronization step.
- neighbor now builds internally to HOOMD-blue. External builds should be unaffected.
  In future, the dependencies on HOOMD-blue may be removed for greater flexibility.

## [0.1.0] - 2018-11-28
### Added
- Initial release of code.

[Unreleased]: https://github.com/mphowardlab/neighbor/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/mphowardlab/neighbor/releases/tag/v0.3.1
[0.3.0]: https://github.com/mphowardlab/neighbor/releases/tag/v0.3.0
[0.2.0]: https://github.com/mphowardlab/neighbor/releases/tag/v0.2.0
[0.1.0]: https://github.com/mphowardlab/neighbor/releases/tag/v0.1.0
