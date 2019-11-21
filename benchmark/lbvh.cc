// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "neighbor/LBVH.h"
#include "neighbor/LBVHTraverser.h"
#include "neighbor/OutputOps.h"
#include "neighbor/QueryOps.h"
#include "neighbor/InsertOps.h"

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

//! Performs a simple benchmark of the LBVH.
/*!
 * LBVH build and traversal time are profiled for a set of simulation snapshots stored
 * in a GSD file. For each frame, a new LBVH and LBVH traverser are constructed. Their
 * autotuners are warmed up for 200 calls before freezing the parameters and profiling
 * for 500 calls. The results are stored for each snapshot in an output tabulated file.
 * The benchmark is to determine the number of particles within a distance \a rcut of
 * each particle (i.e., computing the number of neighbors).
 *
 * The command line parameters are:
 *
 *      ./benchmark_lbvh <input> <Nframes> <rcut> <output>
 *
 * - <input>: GSD file to benchmark
 * - <Nframes>: Number of frames to benchmark in GSD file, starting from 0.
 * - <rcut>: Cutoff radius for computing overlaps.
 * - <output>: Name of tabulated file with output.
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
    if (argc != 5)
        {
        std::cout << "Usage: benchmark_lbvh <input> <Nframes> <rcut> <output>" << std::endl;
        return safe_exit(1);
        }
    else
        {
        filename = std::string(argv[1]);
        num_frames = std::stoull(argv[2]);
        rcut = std::stod(argv[3]);
        outf = std::string(argv[4]);
        }

    // try to execute the benchmark
    try
        {
        std::cout << "Benchmark for " << filename << " using LBVH with rcut = " << rcut << std::endl;
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
        output << "# Benchmark for " << filename << " using LBVH with rcut = " << rcut << std::endl;
        output << "#" << std::endl;
        output << "# " << std::setw(6) << "frame" << std::setw(16) << "build (ms)" << std::setw(16) << "traverse (ms)" << std::endl;

        for (uint64_t frame=0; frame < num_frames; ++frame)
            {
            std::cout << "Frame " << frame << std::endl;
            std::cout << "------------" << std::endl;
            // read snapshot
            auto reader = std::make_shared<GSDReader>(exec_conf, filename, frame, false);

            // setup system and neighborlist
            auto sysdef = std::make_shared<SystemDefinition>(reader->getSnapshot(), exec_conf);
            auto pdata = sysdef->getParticleData();

            // build the lbvh
            auto lbvh = std::make_shared<neighbor::LBVH>(exec_conf);
            const BoxDim& box = pdata->getBox();

            // warmup the lbvh autotuners
            ArrayHandle<Scalar4> d_postype(pdata->getPositions(), access_location::device, access_mode::read);

            for (unsigned int i=0; i < 200; ++i)
                {
                // warmup the lbvh autotuners
                for (unsigned int i=0; i < 200; ++i)
                    {
                    lbvh->build(neighbor::PointInsertOp(d_postype.data, pdata->getN()), box.getLo(), box.getHi());
                    }
                }
            lbvh->setAutotunerParams(false, 100000);

            // profile lbvh build times
            std::vector<double> times(5);
                {
                ArrayHandle<Scalar4> d_postype(pdata->getPositions(), access_location::device, access_mode::read);

                for (size_t i=0; i < times.size(); ++i)
                    {
                    times[i] = profile([&]{lbvh->build(neighbor::PointInsertOp(d_postype.data, pdata->getN()), box.getLo(), box.getHi());}, 500);
                    }
                }
            std::sort(times.begin(), times.end());
            std::cout << "Median LBVH build time: " << times[times.size()/2]<< " ms / build" << std::endl;
            output << std::setw(8) << frame << " " << std::setw(16) << std::fixed << std::setprecision(5) << times[times.size()/2];

            // make traversal volumes
            GlobalArray<Scalar4> spheres(pdata->getN(), exec_conf);
            GlobalArray<Scalar3> images(26, exec_conf);
            GlobalArray<unsigned int> hits(pdata->getN(), exec_conf);
                {
                ArrayHandle<Scalar4> h_spheres(spheres, access_location::host, access_mode::overwrite);
                ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
                ArrayHandle<unsigned int> h_primitives(lbvh->getPrimitives(), access_location::host, access_mode::read);
                for (unsigned int i=0; i < pdata->getN(); ++i)
                    {
                    unsigned int tag = h_primitives.data[i];
                    const Scalar4 postype = h_pos.data[tag];
                    const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
                    h_spheres.data[i] = make_scalar4(pos.x, pos.y, pos.z, rcut);
                    }

                // 26 periodic image vectors
                ArrayHandle<Scalar3> h_images(images, access_location::host, access_mode::overwrite);
                const Scalar3 L = pdata->getBox().getL();
                unsigned int idx = 0;
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

            // try with rope traversal
                {
                neighbor::LBVHTraverser traverser(exec_conf);
                    {
                    ArrayHandle<unsigned int> d_hits(hits, access_location::device, access_mode::overwrite);
                    neighbor::CountNeighborsOp count(d_hits.data);

                    ArrayHandle<Scalar4> d_spheres(spheres, access_location::device, access_mode::read);
                    neighbor::SphereQueryOp query(d_spheres.data, pdata->getN());

                    // warmup the autotuners
                    for (unsigned int i=0; i < 200; ++i)
                        {
                        traverser.traverse(count, query, *lbvh, images);
                        }
                    traverser.setAutotunerParams(false, 100000);

                    for (size_t i=0; i < times.size(); ++ i)
                        {
                        times[i] = profile([&]{traverser.traverse(count, query, *lbvh, images);},500);
                        }
                    }
                std::sort(times.begin(), times.end());
                std::cout << "Median LBVH rope time: " << times[times.size()/2]<< " ms / traversal" << std::endl;
                output << " " << std::setw(16) << std::fixed << std::setprecision(5) << times[times.size()/2] << std::endl;

                ArrayHandle<unsigned int> h_hits(hits, access_location::host, access_mode::read);
                unsigned int min = *std::min_element(h_hits.data, h_hits.data + pdata->getN());
                unsigned int max = *std::max_element(h_hits.data, h_hits.data + pdata->getN());
                double mean = std::accumulate(h_hits.data, h_hits.data + pdata->getN(), 0.0) / pdata->getN();
                std::cout << "min: " << min << ", max: " << max << ", mean: " << mean << std::endl;
                }
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
