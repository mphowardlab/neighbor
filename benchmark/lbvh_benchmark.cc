// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is released under the Modified BSD License.

#include "lbvh_benchmark.cuh"

#include <gsd/gsd.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>
#include <iostream>
#include <fstream>
#include <iomanip>

//! Profile a function call
/*!
 * \param f Function to profile.
 * \param samples Number of samples to take.
 * \returns Average time per call to \a f in milliseconds.
 *
 * The GPU is synchronized before profiling. \a f is then called \a samples times,
 * and then the GPU is synchronized again. The total elapsed time in milliseconds is
 * computed. It is recommended to tune the launch parameters before profiling.
 */
double profile(const std::function <void ()>& f, unsigned int samples)
    {
    cudaDeviceSynchronize();
    const auto start = std::chrono::steady_clock::now();
    for (unsigned int i=0; i < samples; ++i)
        {
        f();
        }
    cudaDeviceSynchronize();
    const auto end = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / double(samples);
    }

//! Snapshot read from a GSD file
struct Snapshot
    {
    unsigned int N;                             //!< Number of particles
    float3 L;                                   //!< Simulation box lengths
    neighbor::shared_array<float3> positions;   //!< Particle positions
    };

//! Find the data chunk of the given name at the given frame
/*!
 * \param handle Open GSD handle.
 * \param frame Frame index to read from.
 * \param name Name of the data chunk.
 * \returns The index entry for the data chunk.
 *
 * Attempts to find the data chunk of the given name at the given frame. If it is not present at
 * this frame, attempt to read from frame 0.
 */
static const gsd_index_entry* read_chunk(gsd_handle& handle, uint64_t frame, const char* name)
    {
    const gsd_index_entry* entry = gsd_find_chunk(&handle, frame, name);
    if (entry == NULL && frame != 0)
        {
        entry = gsd_find_chunk(&handle, 0, name);
        }
    if (entry == NULL)
        {
        throw std::runtime_error(std::string("GSD chunk not found ") + name);
        }
    return entry;
    }

//! Read a frame from a GSD file
/*!
 * \param filename GSD file.
 * \param frame Frame index.
 * \returns The snapshot.
 */
static Snapshot read_snapshot(const std::string& filename, uint64_t frame)
    {
    gsd_handle handle;
    if (gsd_open(&handle, filename.c_str(), GSD_OPEN_READONLY) != GSD_SUCCESS)
        {
        throw std::runtime_error("Unable to open GSD file " + filename);
        }

    Snapshot snap;
    try
        {
        // simulation box
        const gsd_index_entry* box_entry = read_chunk(handle, frame, "configuration/box");
        float box[6];
        gsd_read_chunk(&handle, box, box_entry);
        snap.L = make_float3(box[0], box[1], box[2]);
        
        // number of particles and its positions
        const gsd_index_entry* pos_entry = read_chunk(handle, frame, "particles/position");
        snap.N = static_cast<unsigned int>(pos_entry->N);
        snap.positions = neighbor::shared_array<float3>(snap.N);
        gsd_read_chunk(&handle, snap.positions.get(), pos_entry);
        }
    catch (...)
        {
        gsd_close(&handle);
        throw;
        }  
    gsd_close(&handle);

    return snap;
    }

//! Select the fastest tunable parameter
/*!
 * \param params Valid values for the tunable parameter (e.g., block sizes).
 * \param f Function that launches the kernel with a given parameter.
 * \param samples Number of time samples to take at each parameter.
 * \returns The parameter with the smallest median time.
 *
 * Every parameter is timed \a samples times with CUDA events, and the one with
 * the smallest median is chosen.
 */
unsigned int tune(const std::vector<unsigned int>& params,
                  const std::function<void (unsigned int)>& f,
                  unsigned int samples)
    {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned int best_param = params.at(0);
    float best_time = std::numeric_limits<float>::max();
    std::vector<float> times(samples);
    for (unsigned int p : params)
        {
        // warmup
        f(p);

        // time each call
        for (unsigned int i=0; i < samples; ++i)
            {
            cudaEventRecord(start, 0);
            f(p);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&times[i], start, stop);
            }

        // median
        std::nth_element(times.begin(), times.begin() + samples/2, times.end());
        const float median = times[samples/2];
        if (median < best_time)
            {
            best_time = median;
            best_param = p;
            }
        }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return best_param;
    }

//! Performs a simple benchmark of the LBVH.
/*!
 * LBVH build and traversal time are profiled for a set of simulation snapshots stored
 * in a GSD file. For each frame, a new LBVH and LBVH traverser are constructed. Their
 * kernel block sizes are tuned by scanning all valid values before profiling for 500
 * calls. The results are stored for each snapshot in an output tabulated file. The 
 * benchmark is to determine the number of particles within a distance \a rcut of each 
 * particle (i.e., computing the number of neighbors).
 *
 * The command line parameters are:
 *
 *      ./lbvh_benchmark <input> <Nframes> <rcut> <output>
 *
 * - <input>: GSD file to benchmark
 * - <Nframes>: Number of frames to benchmark in GSD file, starting from 0.
 * - <rcut>: Cutoff radius for computing overlaps.
 * - <output>: Name of tabulated file with output.
 */
int main(int argc, char * argv[])
    {
    // parse the filename
    std::string filename, outf;
    uint64_t num_frames(0);
    double rcut;
    if (argc != 5)
        {
        std::cout << "Usage: lbvh_benchmark <input> <Nframes> <rcut> <output>" << std::endl;
        return 1;
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
        cudaSetDevice(0);

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
            const Snapshot snap = read_snapshot(filename, frame);
            const unsigned int N = snap.N;

            const float3 lo = make_float3(-0.5f*snap.L.x, -0.5f*snap.L.y, -0.5f*snap.L.z);
            const float3 hi = make_float3(0.5f*snap.L.x, 0.5f*snap.L.y, 0.5f*snap.L.z);

            // build the lbvh
            LBVHWrapper lbvh;

            // tune the lbvh block size
            const unsigned int lbvh_param = tune(lbvh.getTunableParameters(),
                                                 [&](unsigned int param){lbvh.build(snap.positions.get(), N, lo, hi, param);},
                                                 5);
            std::cout << "LBVH tuner: " << lbvh_param << std::endl;

            // profile lbvh build times
            std::vector<double> times(5);
            
            for (size_t i=0; i < times.size(); ++i)
                {
                times[i] = profile([&]{lbvh.build(snap.positions.get(), N, lo, hi, lbvh_param);}, 500);
                }
            
            std::sort(times.begin(), times.end());
            std::cout << "Median LBVH build time: " << times[times.size()/2]<< " ms / build" << std::endl;
            output << std::setw(8) << frame << " " << std::setw(16) << std::fixed << std::setprecision(5) << times[times.size()/2];
            cudaDeviceSynchronize();

            // make traversal volumes
            neighbor::shared_array<float4> spheres(N);
            neighbor::shared_array<float3> images(27);
            neighbor::shared_array<unsigned int> hits(N);
                
            auto primitives = lbvh.getPrimitives();
            for (unsigned int i=0; i < N; ++i)
                {
                unsigned int tag = primitives[i];
                const float3 pos = snap.positions[tag];
                spheres[i] = make_float4(pos.x, pos.y, pos.z, float(rcut));
                }

            // 26 periodic image vectors
            const float3 L = snap.L;
            images[0] = make_float3(0,0,0);
            unsigned int idx = 1;
            for (int ix=-1; ix <= 1; ++ix)
                {
                for (int iy=-1; iy <= 1; ++iy)
                    {
                    for (int iz=-1; iz <= 1; ++iz)
                        {
                        if (ix == 0 && iy == 0 && iz == 0) continue;
                        images[idx++] = make_float3(L.x*ix, L.y*iy, L.z*iz);
                        }
                    }
                }

            // traversal
            LBVHTraverserWrapper traverser;

            // tune the traverser block size
            const unsigned int traverser_param = tune(traverser.getTunableParameters(),
                    [&](unsigned int param){ traverser.traverse(hits.get(), spheres.get(), N, lbvh.get(), images.get(), 27, param);},
                    5);
            std::cout << "Traverser tuner: " << traverser_param << std::endl;

            for (size_t i=0; i < times.size(); ++ i)
                {
                times[i] = profile([&]{traverser.traverse(hits.get(), spheres.get(), N, lbvh.get(), images.get(), 27, traverser_param);},500);
                }

            std::sort(times.begin(), times.end());
            std::cout << "Median LBVH rope time: " << times[times.size()/2]<< " ms / traversal" << std::endl;
            output << " " << std::setw(16) << std::fixed << std::setprecision(5) << times[times.size()/2] << std::endl;

            cudaDeviceSynchronize();
            unsigned int min = *std::min_element(hits.get(), hits.get() + N);
            unsigned int max = *std::max_element(hits.get(), hits.get() + N);
            double mean = std::accumulate(hits.get(), hits.get() + N, 0.0) / N;
            std::cout << "min: " << min << ", max: " << max << ", mean: " << mean << std::endl;
            std::cout << std::endl;
            }
        }
    catch(...)
        {
        std::cerr << "**error** Program terminated due to exception." << std::endl;
        return 1;
        }

    return 0;
    }
