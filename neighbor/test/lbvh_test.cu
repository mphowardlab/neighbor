// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include <cuda_runtime.h>

#include "neighbor/LBVH.h"
#include "neighbor/LBVHTraverser.h"
#include "neighbor/OutputOps.h"
#include "neighbor/QueryOps.h"
#include "neighbor/TransformOps.h"
#include "neighbor/InsertOps.h"

#include <random>

#include "upp11_config.h"
UP_MAIN();

// Test of basic LBVH build and traverse functionalities
UP_TEST( lbvh_test )
    {
    std::shared_ptr<neighbor::LBVH> lbvh;

    // make some cuda streams to test with
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]); // different stream
    streams[1] = 0; // default stream

    // points for tree
    thrust::device_vector<float3> points(3);
        {
        points[0] = make_float3(2.5, 0., 0.);
        points[1] = make_float3(1.5, 0., 0.);
        points[2] = make_float3(0.5, 0., 0.);
        }

    // query spheres for tree
    thrust::device_vector<float4> spheres(7);
        {
        // p2
        spheres[0] = make_float4(0.5, 0., 0., 0.5);
        // p1
        spheres[1] = make_float4(1.5, 0., 0., 0.5);
        // p0
        spheres[2] = make_float4(2.5, 0., 0., 0.5);
        // p2, p1
        spheres[3] = make_float4(1.0, 0., 0., 1.0);
        // p1, p0
        spheres[4] = make_float4(2.0, 0., 0., 1.0);
        // p2, p1, p0
        spheres[5] = make_float4(1.5, 0., 0., 1.5);
        // miss
        spheres[6] = make_float4(-0.5, 0., 0., 0.5);
        }

    std::cout << "Testing LBVH build..." << std::endl;
    /*
     *           0
     *          / \
     *         1   p0
     *        / \
     *       p2 p1
     */
    const float3 max = make_float3(1024, 1024, 1024);
    const float3 min = make_float3(0, 0, 0);

    for (unsigned int i=0; i < 2; ++i)
        {
        lbvh = std::make_shared<neighbor::LBVH>();
            {
            lbvh->build(neighbor::PointInsertOp(thrust::raw_pointer_cast(points.data()), 3), min, max, streams[i]);
            cudaStreamSynchronize(streams[i]);
            }

        UP_ASSERT_EQUAL(lbvh->getN(), 3);
        UP_ASSERT_EQUAL(lbvh->getRoot(), 0);

        // parents of each node
        UP_ASSERT_EQUAL(lbvh->getParents().size(), 5);
        thrust::host_vector<int> h_parent(lbvh->getParents());
        UP_ASSERT_EQUAL(h_parent[0], neighbor::gpu::LBVHSentinel);
        UP_ASSERT_EQUAL(h_parent[1], 0);
        UP_ASSERT_EQUAL(h_parent[2], 1);
        UP_ASSERT_EQUAL(h_parent[3], 1);
        UP_ASSERT_EQUAL(h_parent[4], 0);

        UP_ASSERT_EQUAL(lbvh->getLeftChildren().size(), 2);
        thrust::host_vector<int> h_left(lbvh->getLeftChildren());
        UP_ASSERT_EQUAL(h_left[0], 1);
        UP_ASSERT_EQUAL(h_left[1], 2);

        UP_ASSERT_EQUAL(lbvh->getRightChildren().size(), 2);
        thrust::host_vector<int> h_right(lbvh->getRightChildren());
        UP_ASSERT_EQUAL(h_right[0], 4);
        UP_ASSERT_EQUAL(h_right[1], 3);

        UP_ASSERT_EQUAL(lbvh->getPrimitives().size(), 3);
        thrust::host_vector<unsigned int> h_data(lbvh->getPrimitives());
        UP_ASSERT_EQUAL(h_data[0], 2);
        UP_ASSERT_EQUAL(h_data[1], 1);
        UP_ASSERT_EQUAL(h_data[2], 0);

        UP_ASSERT_EQUAL(lbvh->getLowerBounds().size(), 5);
        UP_ASSERT_EQUAL(lbvh->getUpperBounds().size(), 5);
        thrust::host_vector<float3> h_lo(lbvh->getLowerBounds());
        thrust::host_vector<float3> h_hi(lbvh->getUpperBounds());

        // check leafs first
        UP_ASSERT_CLOSE(h_lo[2].x, 0.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_hi[2].x, 0.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_lo[3].x, 1.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_hi[3].x, 1.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_lo[4].x, 2.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_hi[4].x, 2.5f, 1.e-6f);
        // check internal node wrapping 2/3
        UP_ASSERT_CLOSE(h_lo[1].x, 0.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_hi[1].x, 1.5f, 1.e-6f);
        // check root node wrapping 2/3/4
        UP_ASSERT_CLOSE(h_lo[0].x, 0.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_hi[0].x, 2.5f, 1.e-6f);
        }

    // test traverser
    std::cout << "Testing traverser basics..." << std::endl;
        {
        neighbor::LBVHTraverser traverser;
        thrust::device_vector<unsigned int> hits(spheres.size());
            {
            neighbor::CountNeighborsOp count(thrust::raw_pointer_cast(hits.data()));
            neighbor::SphereQueryOp query(thrust::raw_pointer_cast(spheres.data()), spheres.size());
            traverser.traverse(count, query, *lbvh);
            }

        thrust::host_vector<int4> h_data(traverser.getData());
        // Node 0
            {
            int4 node = h_data[0];
            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(node.z, 1);
            UP_ASSERT_EQUAL(node.w, neighbor::gpu::LBVHSentinel);
            }
        // Node 1
            {
            int4 node = h_data[1];

            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 511);
            UP_ASSERT_EQUAL(node.z, 2);
            UP_ASSERT_EQUAL(node.w, 4);
            }
        // Node 2
            {
            int4 node = h_data[2];

            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 1023);
            UP_ASSERT_EQUAL(node.z, ~2);
            UP_ASSERT_EQUAL(node.w, 3);
            }
        // Node 3
            {
            int4 node = h_data[3];

            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 511);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 511);
            UP_ASSERT_EQUAL(node.z, ~1);
            UP_ASSERT_EQUAL(node.w, 4);
            }
        // Node 4
            {
            int4 node = h_data[4];

            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 1023);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(node.z, ~0);
            UP_ASSERT_EQUAL(node.w, neighbor::gpu::LBVHSentinel);
            }

        // each node should have the correct number of hits
        thrust::host_vector<unsigned int> h_hits(hits);
        UP_ASSERT_EQUAL(h_hits[0], 1);
        UP_ASSERT_EQUAL(h_hits[1], 1);
        UP_ASSERT_EQUAL(h_hits[2], 1);
        UP_ASSERT_EQUAL(h_hits[3], 2);
        UP_ASSERT_EQUAL(h_hits[4], 2);
        UP_ASSERT_EQUAL(h_hits[5], 3);
        UP_ASSERT_EQUAL(h_hits[6], 0);
        }

    // test traverser neigh list op
    std::cout << "Testing traverser neighbor list..." << std::endl;
    for (unsigned int i=0; i < 2; ++i)
        {
        neighbor::LBVHTraverser traverser;
        // setup nlist data structures
        const unsigned int max_neigh = 2;
        thrust::device_vector<unsigned int> neigh_list(max_neigh*spheres.size());
        thrust::device_vector<unsigned int> nneigh(spheres.size());
        // generate list on gpu
            {
            neighbor::NeighborListOp nl_op(thrust::raw_pointer_cast(neigh_list.data()),
                                           thrust::raw_pointer_cast(nneigh.data()),
                                           max_neigh);

            neighbor::SphereQueryOp query(thrust::raw_pointer_cast(spheres.data()),
                                          spheres.size());

            traverser.traverse(nl_op, query, *lbvh, thrust::device_vector<float3>(), streams[i]);
            cudaStreamSynchronize(streams[i]);
            }
        // check output
            {
            thrust::host_vector<unsigned int> h_neigh_list(neigh_list);
            thrust::host_vector<unsigned int> h_nneigh(nneigh);

            //
            UP_ASSERT_EQUAL(h_nneigh[0], 1);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*0+0], 2);

            UP_ASSERT_EQUAL(h_nneigh[1], 1);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*1+0], 1);

            UP_ASSERT_EQUAL(h_nneigh[2], 1);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*2+0], 0);

            UP_ASSERT_EQUAL(h_nneigh[3], 2);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*3+0], 2);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*3+1], 1);

            UP_ASSERT_EQUAL(h_nneigh[4], 2);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*4+0], 1);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*4+1], 0);

            UP_ASSERT_EQUAL(h_nneigh[5], 3);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*5+0], 2);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*5+1], 1);
            // this neighbor should be left off because it exceeds max neigh
            //UP_ASSERT_EQUAL(h_neigh_list[max_neigh*5+2], 0);

            UP_ASSERT_EQUAL(h_nneigh[6], 0);
            }
        }

    // test traverser with transform op
    std::cout << "Testing traverser with map transform op..." << std::endl;
        {
        neighbor::LBVHTraverser traverser;

        // remap the particle tags into the order I'm expecting them
        thrust::device_vector<unsigned int> map(spheres.size());
            {
            map[0] = 2;
            map[1] = 1;
            map[2] = 0;
            }

        // setup nlist data structures
        const unsigned int max_neigh = 2;
        thrust::device_vector<unsigned int> neigh_list(max_neigh*spheres.size());
        thrust::device_vector<unsigned int> nneigh(spheres.size());
        // generate list on gpu
            {
            neighbor::NeighborListOp nl_op(thrust::raw_pointer_cast(neigh_list.data()),
                                           thrust::raw_pointer_cast(nneigh.data()),
                                           max_neigh);

            neighbor::SphereQueryOp query(thrust::raw_pointer_cast(spheres.data()),
                                          spheres.size());

            neighbor::MapTransformOp transform(thrust::raw_pointer_cast(map.data()));

            traverser.traverse(nl_op, query, transform, *lbvh);
            }
        // check output
            {
            thrust::host_vector<unsigned int> h_neigh_list(neigh_list);
            thrust::host_vector<unsigned int> h_nneigh(nneigh);

            UP_ASSERT_EQUAL(h_nneigh[0], 1);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*0+0], 0);

            UP_ASSERT_EQUAL(h_nneigh[1], 1);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*1+0], 1);

            UP_ASSERT_EQUAL(h_nneigh[2], 1);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*2+0], 2);

            UP_ASSERT_EQUAL(h_nneigh[3], 2);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*3+0], 0);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*3+1], 1);

            UP_ASSERT_EQUAL(h_nneigh[4], 2);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*4+0], 1);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*4+1], 2);

            UP_ASSERT_EQUAL(h_nneigh[5], 3);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*5+0], 0);
            UP_ASSERT_EQUAL(h_neigh_list[max_neigh*5+1], 1);
            // this neighbor should be left off because it exceeds max neigh
            //UP_ASSERT_EQUAL(h_neigh_list[max_neigh*5+2], 2);

            UP_ASSERT_EQUAL(h_nneigh[6], 0);
            }
        }

    cudaStreamDestroy(streams[0]);
    }

// Test that LBVH traverser handles images correctly
UP_TEST( lbvh_periodic_test )
    {
    auto lbvh = std::make_shared<neighbor::LBVH>();

    // points for tree
    thrust::device_vector<float3> points(3);
        {
        points[0] = make_float3( 1.9, 1.9, 1.9);
        points[1] = make_float3(  0., 0.,  0.);
        points[2] = make_float3(-1.9,-1.9,-1.9);
        }

    const float3 max = make_float3( 2., 2., 2.);
    const float3 min = make_float3(-2.,-2.,-2.);
    lbvh->build(neighbor::PointInsertOp(thrust::raw_pointer_cast(points.data()), points.size()), min, max);

    // query spheres for tree that intersect through boundaries
    thrust::device_vector<float4> spheres(2);
    thrust::device_vector<float3> images(26);
        {
        // p2
        spheres[0] = make_float4(-1.9, 1.9, 1.9, 0.5);
        // p1
        spheres[1] = make_float4( 1.9,-1.9,-1.9, 0.5);

        unsigned int idx=0;
        for (int ix=-1; ix <= 1; ++ix)
            {
            for (int iy=-1; iy <= 1; ++iy)
                {
                for (int iz=-1; iz <= 1; ++iz)
                    {
                    if (ix == 0 && iy == 0 && iz == 0) continue;

                    images[idx++] = make_float3(4*ix, 4*iy, 4*iz);
                    }
                }
            }
        }

    // no hits without images
    neighbor::LBVHTraverser traverser;
    thrust::device_vector<unsigned int> hits(spheres.size());
        {
        neighbor::CountNeighborsOp count(thrust::raw_pointer_cast(hits.data()));

        neighbor::SphereQueryOp query(thrust::raw_pointer_cast(spheres.data()),
                                      spheres.size());

        traverser.traverse(count, query, *lbvh);

        thrust::host_vector<unsigned int> h_hits(hits);
        UP_ASSERT_EQUAL(h_hits[0], 0);
        UP_ASSERT_EQUAL(h_hits[1], 0);
        }

    // 2 hits with images
        {
        neighbor::CountNeighborsOp count(thrust::raw_pointer_cast(hits.data()));

        neighbor::SphereQueryOp query(thrust::raw_pointer_cast(spheres.data()),
                                      spheres.size());

        traverser.traverse(count, query, *lbvh, images);

        thrust::host_vector<unsigned int> h_hits(hits);
        UP_ASSERT_EQUAL(h_hits[0], 2);
        UP_ASSERT_EQUAL(h_hits[1], 2);
        }
    }

// Test that LBVH counts at least the same number of neighbors in an ideal gas as brute force
UP_TEST( lbvh_validate )
    {
    auto lbvh = std::make_shared<neighbor::LBVH>();

    // N particles in orthorhombic box
    const float3 L = make_float3(20,15,25);
    const unsigned int N = static_cast<unsigned int>(1.0*L.x*L.y*L.z);
    const float rcut = 1.0;

    // generate random points in the box
    thrust::host_vector<float3> h_points(N);
        {
        std::mt19937 mt(42);
        std::uniform_real_distribution<float> U(-0.5, 0.5);
        for (unsigned int i=0; i < N; ++i)
            {
            h_points[i] = make_float3(L.x*U(mt), L.y*U(mt), L.z*U(mt));
            }
        }
    thrust::device_vector<float3> points(h_points);

    const float3 lo = make_float3(-0.5*L.x, -0.5*L.y, -0.5*L.z);
    const float3 hi = make_float3( 0.5*L.x,  0.5*L.y,  0.5*L.z);
    lbvh->build(neighbor::PointInsertOp(thrust::raw_pointer_cast(points.data()), N), lo, hi);

    // query spheres for tree
    thrust::device_vector<float4> spheres(N);
        {
        thrust::host_vector<float4> h_spheres(spheres);
        for (unsigned int i=0; i < N; ++i)
            {
            const float3 point = h_points[i];
            h_spheres[i] = make_float4(point.x, point.y, point.z, rcut);
            }
        spheres = h_spheres;
        }

    // traversal images
    thrust::device_vector<float3> images(26);
        {
        unsigned int idx=0;
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
        }

    // build hit list
    neighbor::LBVHTraverser traverser;
    thrust::device_vector<unsigned int> hits(N);
        {
        neighbor::CountNeighborsOp count(thrust::raw_pointer_cast(hits.data()));

        neighbor::SphereQueryOp query(thrust::raw_pointer_cast(spheres.data()),
                                      spheres.size());

        traverser.traverse(count, query, *lbvh, images);
        }

    // generate list of reference collisions
    std::vector<unsigned int> ref_hits(N);
        {
        const float rcut2 = rcut*rcut;
        std::fill(ref_hits.begin(), ref_hits.end(), 0);
        for (unsigned int i=0; i < N; ++i)
            {
            const float3 ri = h_points[i];
            for (unsigned int j=i; j < N; ++j)
                {
                const float3 rj = h_points[j];
                float3 dr = make_float3(rj.x-ri.x, rj.y-ri.y, rj.z-ri.z);

                // minimum image
                if (dr.x >= hi.x) dr.x -= L.x;
                else if (dr.x < lo.x) dr.x += L.x;

                if (dr.y >= hi.y) dr.y -= L.y;
                else if (dr.y < lo.y) dr.y += L.y;

                if (dr.z >= hi.z) dr.z -= L.z;
                else if (dr.z < lo.z) dr.z += L.z;

                const float dr2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
                if (dr2 <= rcut2)
                    {
                    ++ref_hits[i];
                    if (j != i)
                        ++ref_hits[j];
                    }
                }
            }
        }

    // check that tree always has at least as many hits as the reference
        {
        thrust::host_vector<unsigned int> h_hits(hits);
        for (unsigned int i=0; i < N; ++i)
            {
            if (h_hits[i] < ref_hits[i])
                {
                std::cout << "Particle " << i << std::endl;
                }
            UP_ASSERT_GREATER_EQUAL(h_hits[i], ref_hits[i]);
            }
        }
    }

// Test of basic LBVH build and traverse functionalities
UP_TEST( lbvh_small_test )
    {
    auto lbvh = std::make_shared<neighbor::LBVH>();

    // one point for tree
    thrust::device_vector<float3> points(1);
        {
        points[0] = make_float3(2.5, 0., 0.);
        }
    // query spheres for tree
    thrust::device_vector<float4> spheres(2);
        {
        // p0
        spheres[0] = make_float4(2.5, 0., 0., 0.5);
        // miss
        spheres[1] = make_float4(-0.5, 0., 0., 0.5);
        }

    std::cout << "Testing small LBVH build..." << std::endl;
    const float3 max = make_float3(1024, 1024, 1024);
    const float3 min = make_float3(0, 0, 0);
    lbvh->build(neighbor::PointInsertOp(thrust::raw_pointer_cast(points.data()), 1), min, max);
        {
        UP_ASSERT_EQUAL(lbvh->getN(), 1);
        UP_ASSERT_EQUAL(lbvh->getRoot(), 0);

        // parents of each node
        UP_ASSERT_EQUAL(lbvh->getParents().size(), 1);
        thrust::host_vector<int> h_parent(lbvh->getParents());
        UP_ASSERT_EQUAL(h_parent[0], neighbor::gpu::LBVHSentinel);

        UP_ASSERT_EQUAL(lbvh->getLeftChildren().size(), 0);
        UP_ASSERT_EQUAL(lbvh->getRightChildren().size(), 0);

        UP_ASSERT_EQUAL(lbvh->getPrimitives().size(), 1);
        thrust::host_vector<unsigned int> h_data(lbvh->getPrimitives());
        UP_ASSERT_EQUAL(h_data[0], 0);

        UP_ASSERT_EQUAL(lbvh->getLowerBounds().size(), 1);
        UP_ASSERT_EQUAL(lbvh->getUpperBounds().size(), 1);
        thrust::host_vector<float3> h_lo(lbvh->getLowerBounds());
        thrust::host_vector<float3> h_hi(lbvh->getUpperBounds());

        // check leafs first
        UP_ASSERT_CLOSE(h_lo[0].x, 2.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_hi[0].x, 2.5f, 1.e-6f);
        }

    // test traverser
    std::cout << "Testing small traverser..." << std::endl;
        {
        neighbor::LBVHTraverser traverser;
        thrust::device_vector<unsigned int> hits(spheres.size());
            {
            neighbor::CountNeighborsOp count(thrust::raw_pointer_cast(hits.data()));

            neighbor::SphereQueryOp query(thrust::raw_pointer_cast(spheres.data()),
                                          spheres.size());

            traverser.traverse(count, query, *lbvh);
            }

        thrust::host_vector<int4> h_data(traverser.getData());
        // only one node, just check its index contents
            {
            int4 node = h_data[0];
            UP_ASSERT_EQUAL(node.z, ~0);
            UP_ASSERT_EQUAL(node.w, neighbor::gpu::LBVHSentinel);
            }

        // each node should have the correct number of hits
        thrust::host_vector<unsigned int> h_hits(hits);
        UP_ASSERT_EQUAL(h_hits[0], 1);
        UP_ASSERT_EQUAL(h_hits[1], 0);
        }
    }
