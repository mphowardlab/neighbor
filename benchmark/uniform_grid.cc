// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "neighbor/UniformGrid.h"
#include "neighbor/UniformGridTraverser.h"
#include "neighbor/OutputOps.h"
#include "neighbor/QueryOps.h"

#include "hoomd/ClockSource.h"
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/GSDReader.h"
#include "hoomd/SystemDefinition.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <iostream>
#include <fstream>
#include <iomanip>

//! Convenience wrapper to return an exit code in MPI-compiled HOOMD.
/*!
 * \param val Exit code
 *
 * In MPI builds, this function first calls MPI_Finalize before
 * returning the exit code.
 */
int safe_exit(int val)
    {
    #ifdef ENABLE_MPI
    MPI_Finalize();
    #endif // ENABLE_MPI
    return val;
    }

//! Profile a function call
/*!
 * \param f Function to profile.
 * \param samples Number of samples to take.
 * \returns Average time per call to \a f in milliseconds.
 *
 * The GPU is synchronized before profiling. \a f is then called \a samples times,
 * and then the GPU is synchronized again. The total elapsed time in milliseconds is
 * computed. It is recommended to warmup and then freeze any autotuner parameters
 * before profiling.
 */
double profile(const std::function <void ()>& f, unsigned int samples)
    {
    ClockSource t;

    cudaDeviceSynchronize();
    uint64_t start = t.getTime();
    for (unsigned int i=0; i < samples; ++i)
        {
        f();
        }
    cudaDeviceSynchronize();
    uint64_t elapsed = t.getTime() - start;

    return double(elapsed)/1e6/double(samples);
    }

//! Performs a simple benchmark of the UniformGrid.
/*!
 * UniformGrid build and traversal time are profiled for a set of simulation snapshots stored
 * in a GSD file. For each frame, a new UniformGrid and traverser are constructed. Their
 * autotuners are warmed up for 200 calls before freezing the parameters and profiling
 * for 500 calls. The results are stored for each snapshot in an output tabulated file.
 * The benchmark is to determine the number of particles within a distance \a rcut of
 * each particle (i.e., computing the number of neighbors).
 *
 * The command line parameters are:
 *
 *      ./benchmark_grid <input> <Nframes> <rcut> <maxthreads> <output>
 *
 * - <input>: GSD file to benchmark
 * - <Nframes>: Number of frames to benchmark in GSD file, starting from 0.
 * - <rcut>: Cutoff radius for computing overlaps.
 * - <maxthreads>: Maximum number of threads per traversal sphere.
 * - <output>: Name of tabulated file with output.
 *
 * The traversal benchmark is performed for all valid number of threads per sphere (1,2,...)
 * up to \a maxthreads. For faster benchmarks, set this to 1 when your system is large
 * enough that it does not get any performance benefit from using multiple threads per sphere.
 */
int main(int argc, char * argv[])
    {
    #ifdef ENABLE_MPI
    MPI_Init(&argc, &argv);
    #endif

    // parse the filename
    std::string filename, outf;
    uint64_t num_frames(0);
    double rcut;
    unsigned int max_threads;
    if (argc != 6)
        {
        std::cerr << "Usage: benchmark_grid <input> <Nframes> <rcut> <maxthreads> <output>" << std::endl;
        return safe_exit(1);
        }
    else
        {
        filename = std::string(argv[1]);
        num_frames = std::stoull(argv[2]);
        rcut = std::stod(argv[3]);
        max_threads = std::stoi(argv[4]);
        outf = std::string(argv[5]);
        }

    if (max_threads == 0 || max_threads > 32 || (max_threads & (max_threads-1)))
        {
        std::cerr << "Maximum number of threads must be a power of 2 between 1 and 32" << std::endl;
        return safe_exit(1);
        }

    // try to execute the benchmark
    try
        {
        std::cout << "Benchmark for " << filename << " using UniformGrid with rcut = " << rcut << std::endl;
        std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
        #ifdef ENABLE_MPI
        if (exec_conf->getNRanks() > 1)
            {
            exec_conf->msg->error() << "Benchmarks cannot be run in MPI" << std::endl;
            throw std::runtime_error("Benchmarks cannot be run in MPI");
            }
        #endif

        // write benchmark results to file
        std::ofstream output;
        output.open(outf.c_str());
        output << "# Benchmark for " << filename << " using UniformGrid with rcut = " << rcut << std::endl;
        output << "# Traversal times are for w threads per particle" << std::endl;
        output << "#" << std::endl;
        output << "# " << std::setw(6) << "frame" << std::setw(16) << "build (ms)";
        for (unsigned int num_threads=1; num_threads <= max_threads; num_threads *= 2)
            output << std::setw(16) << "w=" << num_threads << " (ms)";
        output << std::endl;

        for (uint64_t frame=0; frame < num_frames; ++frame)
            {
            std::cout << "Frame " << frame << std::endl;
            std::cout << "------------" << std::endl;
            auto reader = std::make_shared<GSDReader>(exec_conf, filename, frame, false);

            // setup system and neighborlist
            auto sysdef = std::make_shared<SystemDefinition>(reader->getSnapshot(), exec_conf);
            auto pdata = sysdef->getParticleData();

            // uniform grid
            const BoxDim& box = pdata->getBox();
            auto grid = std::make_shared<neighbor::UniformGrid>(exec_conf, rcut);
            // warmup the lbvh autotuners
            ArrayHandle<Scalar4> d_postype(pdata->getPositions(), access_location::device, access_mode::read);
            for (unsigned int i=0; i < 200; ++i)
                {
                grid->build(neighbor::GridPointOp(d_postype.data, pdata->getN()), box.getLo(), box.getHi());
                }
            grid->setAutotunerParams(false, 100000);

            // profile grid build times
            std::vector<double> times(5);
            for (size_t i=0; i < times.size(); ++i)
                {
                times[i] = profile([&]{grid->build(neighbor::GridPointOp(d_postype.data, pdata->getN()), box.getLo(), box.getHi());}, 500);
                }
            std::sort(times.begin(), times.end());
            std::cout << "Median grid build time: " << times[times.size()/2]<< " ms / build" << std::endl;
                {
                ArrayHandle<unsigned int> h_sizes(grid->getSizes(), access_location::host, access_mode::read);
                unsigned int min = *std::min_element(h_sizes.data, h_sizes.data + grid->getIndexer().getNumElements());
                unsigned int max = *std::max_element(h_sizes.data, h_sizes.data + grid->getIndexer().getNumElements());
                double mean = std::accumulate(h_sizes.data, h_sizes.data + grid->getIndexer().getNumElements(), 0.0) / grid->getIndexer().getNumElements();
                std::cout << "min: " << min << ", max: " << max << ", mean: " << mean << std::endl;
                }
            output << std::setw(8) << frame << " " << std::setw(16) << std::fixed << std::setprecision(5) << times[times.size()/2];

            GlobalArray<Scalar4> spheres(pdata->getN(), exec_conf);
            GlobalArray<Scalar3> images(26, exec_conf);
            GlobalArray<unsigned int> hits(pdata->getN(), exec_conf);
                {
                ArrayHandle<Scalar4> h_spheres(spheres, access_location::host, access_mode::overwrite);
                ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
                ArrayHandle<unsigned int> h_primitives(grid->getPrimitives(), access_location::host, access_mode::read);
                for (unsigned int i=0; i < pdata->getN(); ++i)
                    {
                    unsigned int tag = h_primitives.data[i];
                    const Scalar4 postype = h_pos.data[tag];
                    const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
                    h_spheres.data[i] = make_scalar4(pos.x, pos.y, pos.z, rcut);
                    }

                // 26 periodic image vectors
                ArrayHandle<Scalar3> h_images(images, access_location::host, access_mode::overwrite);
                Scalar3 L = pdata->getBox().getL();
                unsigned int idx=0;
                for (int ix=-1; ix <= 1; ++ix)
                    {
                    for (int iy=-1; iy <= 1; ++iy)
                        {
                        for (int iz=-1; iz <= 1; ++iz)
                            {
                            if (ix == 0 && iy == 0 && iz == 0) continue;
                            h_images.data[idx++] = make_scalar3(L.x*ix, L.y*iy, L.z*iz);
                            }
                        }
                    }
                }

            // profile traversal
                {
                neighbor::UniformGridTraverser traverser(exec_conf);
                    {
                    ArrayHandle<unsigned int> d_hits(hits, access_location::device, access_mode::overwrite);
                    neighbor::CountNeighborsOp count(d_hits.data);

                    ArrayHandle<Scalar4> d_spheres(spheres, access_location::device, access_mode::read);
                    neighbor::SphereQueryOp query(d_spheres.data, pdata->getN());

                    // warmup the autotuners
                    for (unsigned int i=0; i < 200; ++i)
                        {
                        traverser.traverse(count, query, *grid, images);
                        }
                    traverser.setAutotunerParams(false, 100000);

                    for (size_t i=0; i < times.size(); ++ i)
                        {
                        times[i] = profile([&]{traverser.traverse(count, query, *grid, images);},500);
                        }
                    }
                std::sort(times.begin(), times.end());
                std::cout << "Median grid traverser time : " << times[times.size()/2]<< " ms / traversal" << std::endl;

                ArrayHandle<unsigned int> h_hits(hits, access_location::host, access_mode::read);
                unsigned int min = *std::min_element(h_hits.data, h_hits.data + pdata->getN());
                unsigned int max = *std::max_element(h_hits.data, h_hits.data + pdata->getN());
                double mean = std::accumulate(h_hits.data, h_hits.data + pdata->getN(), 0.0) / pdata->getN();
                std::cout << "min: " << min << ", max: " << max << ", mean: " << mean << std::endl;
                output << " " << std::setw(16) << std::fixed << std::setprecision(5) << times[times.size()/2];
                }
            output << std::endl;
            std::cout << std::endl;
            }
        }
    catch(...)
        {
        std::cerr << "**error** Program terminated due to exception." << std::endl;
        return safe_exit(1);
        }

    return safe_exit(0);
    }
