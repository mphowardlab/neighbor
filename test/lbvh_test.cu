// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include <cuda_runtime.h>

#include "neighbor/neighbor.h"

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
    neighbor::shared_array<float3> points(3);
        {
        points[0] = make_float3(2.5, 0., 0.);
        points[1] = make_float3(1.5, 0., 0.);
        points[2] = make_float3(0.5, 0., 0.);
        }

    // query spheres for tree
    neighbor::shared_array<float4> spheres(7);
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
            lbvh->build(streams[i], neighbor::PointInsertOp(points.get(), 3), min, max);
            cudaStreamSynchronize(streams[i]);
            }

        UP_ASSERT_EQUAL(lbvh->getN(), 3);
        UP_ASSERT_EQUAL(lbvh->getRoot(), 0);

        // parents of each node
        UP_ASSERT_EQUAL(lbvh->getParents().size(), 5);
        auto parents = lbvh->getParents();
        UP_ASSERT_EQUAL(parents[0], neighbor::LBVHSentinel);
        UP_ASSERT_EQUAL(parents[1], 0);
        UP_ASSERT_EQUAL(parents[2], 1);
        UP_ASSERT_EQUAL(parents[3], 1);
        UP_ASSERT_EQUAL(parents[4], 0);

        UP_ASSERT_EQUAL(lbvh->getLeftChildren().size(), 2);
        auto left = lbvh->getLeftChildren();
        UP_ASSERT_EQUAL(left[0], 1);
        UP_ASSERT_EQUAL(left[1], 2);

        UP_ASSERT_EQUAL(lbvh->getRightChildren().size(), 2);
        auto right = lbvh->getRightChildren();
        UP_ASSERT_EQUAL(right[0], 4);
        UP_ASSERT_EQUAL(right[1], 3);

        UP_ASSERT_EQUAL(lbvh->getPrimitives().size(), 3);
        auto data = lbvh->getPrimitives();
        UP_ASSERT_EQUAL(data[0], 2);
        UP_ASSERT_EQUAL(data[1], 1);
        UP_ASSERT_EQUAL(data[2], 0);

        UP_ASSERT_EQUAL(lbvh->getLowerBounds().size(), 5);
        UP_ASSERT_EQUAL(lbvh->getUpperBounds().size(), 5);
        auto lo = lbvh->getLowerBounds();
        auto hi = lbvh->getUpperBounds();

        // check leafs first
        UP_ASSERT_CLOSE(lo[2].x, 0.5f, 1.e-6f);
        UP_ASSERT_CLOSE(hi[2].x, 0.5f, 1.e-6f);
        UP_ASSERT_CLOSE(lo[3].x, 1.5f, 1.e-6f);
        UP_ASSERT_CLOSE(hi[3].x, 1.5f, 1.e-6f);
        UP_ASSERT_CLOSE(lo[4].x, 2.5f, 1.e-6f);
        UP_ASSERT_CLOSE(hi[4].x, 2.5f, 1.e-6f);
        // check internal node wrapping 2/3
        UP_ASSERT_CLOSE(lo[1].x, 0.5f, 1.e-6f);
        UP_ASSERT_CLOSE(hi[1].x, 1.5f, 1.e-6f);
        // check root node wrapping 2/3/4
        UP_ASSERT_CLOSE(lo[0].x, 0.5f, 1.e-6f);
        UP_ASSERT_CLOSE(hi[0].x, 2.5f, 1.e-6f);
        }

    // test traverser
    std::cout << "Testing traverser basics..." << std::endl;
        {
        neighbor::LBVHTraverser traverser;
        neighbor::shared_array<unsigned int> hits(spheres.size());
            {
            neighbor::CountNeighborsOp count(hits.get());
            neighbor::SphereQueryOp query(spheres.get(), spheres.size());
            traverser.traverse(*lbvh, query, count);
            cudaDeviceSynchronize();
            }

        auto data = traverser.getData();
        // Node 0
            {
            int4 node = data[0];
            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(node.z, 1);
            UP_ASSERT_EQUAL(node.w, neighbor::LBVHSentinel);
            }
        // Node 1
            {
            int4 node = data[1];

            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 511);
            UP_ASSERT_EQUAL(node.z, 2);
            UP_ASSERT_EQUAL(node.w, 4);
            }
        // Node 2
            {
            int4 node = data[2];

            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 1023);
            UP_ASSERT_EQUAL(node.z, ~2);
            UP_ASSERT_EQUAL(node.w, 3);
            }
        // Node 3
            {
            int4 node = data[3];

            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 511);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 511);
            UP_ASSERT_EQUAL(node.z, ~1);
            UP_ASSERT_EQUAL(node.w, 4);
            }
        // Node 4
            {
            int4 node = data[4];

            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 1023);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(node.z, ~0);
            UP_ASSERT_EQUAL(node.w, neighbor::LBVHSentinel);
            }

        // each node should have the correct number of hits
        UP_ASSERT_EQUAL(hits[0], 1);
        UP_ASSERT_EQUAL(hits[1], 1);
        UP_ASSERT_EQUAL(hits[2], 1);
        UP_ASSERT_EQUAL(hits[3], 2);
        UP_ASSERT_EQUAL(hits[4], 2);
        UP_ASSERT_EQUAL(hits[5], 3);
        UP_ASSERT_EQUAL(hits[6], 0);
        }

    // test traverser neigh list op
    std::cout << "Testing traverser neighbor list..." << std::endl;
    for (unsigned int i=0; i < 2; ++i)
        {
        neighbor::LBVHTraverser traverser;
        // setup nlist data structures
        const unsigned int max_neigh = 2;
        neighbor::shared_array<unsigned int> neigh_list(max_neigh*spheres.size());
        neighbor::shared_array<unsigned int> nneigh(spheres.size());
        // generate list on gpu
            {
            neighbor::NeighborListOp nl_op(neigh_list.get(),
                                           nneigh.get(),
                                           max_neigh);

            neighbor::SphereQueryOp query(spheres.get(),
                                          spheres.size());

            traverser.traverse(streams[i], *lbvh, query, nl_op);
            cudaStreamSynchronize(streams[i]);
            }

        // check output
            {
            UP_ASSERT_EQUAL(nneigh[0], 1);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*0+0], 2);

            UP_ASSERT_EQUAL(nneigh[1], 1);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*1+0], 1);

            UP_ASSERT_EQUAL(nneigh[2], 1);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*2+0], 0);

            UP_ASSERT_EQUAL(nneigh[3], 2);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*3+0], 2);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*3+1], 1);

            UP_ASSERT_EQUAL(nneigh[4], 2);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*4+0], 1);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*4+1], 0);

            UP_ASSERT_EQUAL(nneigh[5], 3);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*5+0], 2);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*5+1], 1);
            // this neighbor should be left off because it exceeds max neigh
            //UP_ASSERT_EQUAL(neigh_list[max_neigh*5+2], 0);

            UP_ASSERT_EQUAL(nneigh[6], 0);
            }
        }

    // test traverser with transform op
    std::cout << "Testing traverser with map transform op..." << std::endl;
        {
        neighbor::LBVHTraverser traverser;

        // remap the particle tags into the order I'm expecting them
        neighbor::shared_array<unsigned int> map(spheres.size());
            {
            map[0] = 2;
            map[1] = 1;
            map[2] = 0;
            }

        // setup nlist data structures
        const unsigned int max_neigh = 2;
        neighbor::shared_array<unsigned int> neigh_list(max_neigh*spheres.size());
        neighbor::shared_array<unsigned int> nneigh(spheres.size());
        // generate list on gpu
            {
            neighbor::NeighborListOp nl_op(neigh_list.get(),
                                           nneigh.get(),
                                           max_neigh);

            neighbor::SphereQueryOp query(spheres.get(),
                                          spheres.size());

            neighbor::MapTransformOp transform(map.get());

            traverser.traverse(*lbvh, query, nl_op, neighbor::SelfOp(), transform);
            cudaDeviceSynchronize();
            }

        // check output
            {
            UP_ASSERT_EQUAL(nneigh[0], 1);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*0+0], 0);

            UP_ASSERT_EQUAL(nneigh[1], 1);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*1+0], 1);

            UP_ASSERT_EQUAL(nneigh[2], 1);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*2+0], 2);

            UP_ASSERT_EQUAL(nneigh[3], 2);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*3+0], 0);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*3+1], 1);

            UP_ASSERT_EQUAL(nneigh[4], 2);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*4+0], 1);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*4+1], 2);

            UP_ASSERT_EQUAL(nneigh[5], 3);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*5+0], 0);
            UP_ASSERT_EQUAL(neigh_list[max_neigh*5+1], 1);
            // this neighbor should be left off because it exceeds max neigh
            //UP_ASSERT_EQUAL(neigh_list[max_neigh*5+2], 2);

            UP_ASSERT_EQUAL(nneigh[6], 0);
            }
        }

    cudaStreamDestroy(streams[0]);
    }

// Test that LBVH traverser handles images correctly
UP_TEST( lbvh_periodic_test )
    {
    auto lbvh = std::make_shared<neighbor::LBVH>();

    // points for tree
    neighbor::shared_array<float3> points(3);
        {
        points[0] = make_float3( 1.9, 1.9, 1.9);
        points[1] = make_float3(  0., 0.,  0.);
        points[2] = make_float3(-1.9,-1.9,-1.9);
        }

    const float3 max = make_float3( 2., 2., 2.);
    const float3 min = make_float3(-2.,-2.,-2.);
    lbvh->build(neighbor::PointInsertOp(points.get(), points.size()), min, max);
    cudaDeviceSynchronize();

    // query spheres for tree that intersect through boundaries
    neighbor::shared_array<float4> spheres(2);
    neighbor::shared_array<float3> images(27);
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
                    images[idx++] = make_float3(4*ix, 4*iy, 4*iz);
                    }
                }
            }
        }

    // no hits without images
    neighbor::LBVHTraverser traverser;
    neighbor::shared_array<unsigned int> hits(spheres.size());
        {
        neighbor::CountNeighborsOp count(hits.get());

        neighbor::SphereQueryOp query(spheres.get(),
                                      spheres.size());

        traverser.traverse(*lbvh, query, count);
        cudaDeviceSynchronize();

        UP_ASSERT_EQUAL(hits[0], 0);
        UP_ASSERT_EQUAL(hits[1], 0);
        }

    // 2 hits with images
        {
        neighbor::CountNeighborsOp count(hits.get());

        neighbor::SphereQueryOp query(spheres.get(),
                                      spheres.size());

        neighbor::ImageListOp<float3> translate(images.get(), images.size());

        traverser.traverse(*lbvh, query, count, translate);
        cudaDeviceSynchronize();

        UP_ASSERT_EQUAL(hits[0], 2);
        UP_ASSERT_EQUAL(hits[1], 2);
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
    neighbor::shared_array<float3> points(N);
        {
        std::mt19937 mt(42);
        std::uniform_real_distribution<float> U(-0.5, 0.5);
        for (unsigned int i=0; i < N; ++i)
            {
            points[i] = make_float3(L.x*U(mt), L.y*U(mt), L.z*U(mt));
            }
        }

    const float3 lo = make_float3(-0.5*L.x, -0.5*L.y, -0.5*L.z);
    const float3 hi = make_float3( 0.5*L.x,  0.5*L.y,  0.5*L.z);
    lbvh->build(neighbor::PointInsertOp(points.get(), N), lo, hi);
    cudaDeviceSynchronize();

    // query spheres for tree
    neighbor::shared_array<float4> spheres(N);
        {
        for (unsigned int i=0; i < N; ++i)
            {
            const float3 point = points[i];
            spheres[i] = make_float4(point.x, point.y, point.z, rcut);
            }
        }

    // traversal images
    neighbor::shared_array<float3> images(27);
        {
        unsigned int idx=0;
        for (int ix=-1; ix <= 1; ++ix)
            {
            for (int iy=-1; iy <= 1; ++iy)
                {
                for (int iz=-1; iz <= 1; ++iz)
                    {
                    images[idx++] = make_float3(L.x*ix, L.y*iy, L.z*iz);
                    }
                }
            }
        }

    // build hit list
    neighbor::LBVHTraverser traverser;
    neighbor::shared_array<unsigned int> hits(N);
        {
        neighbor::CountNeighborsOp count(hits.get());

        neighbor::SphereQueryOp query(spheres.get(),
                                      spheres.size());

        neighbor::ImageListOp<float3> translate(images.get(),
                                                images.size());

        traverser.traverse(*lbvh, query, count, translate);
        cudaDeviceSynchronize();
        }

    // generate list of reference collisions
    std::vector<unsigned int> ref_hits(N);
        {
        const float rcut2 = rcut*rcut;
        std::fill(ref_hits.begin(), ref_hits.end(), 0);
        for (unsigned int i=0; i < N; ++i)
            {
            const float3 ri = points[i];
            for (unsigned int j=i; j < N; ++j)
                {
                const float3 rj = points[j];
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
        for (unsigned int i=0; i < N; ++i)
            {
            if (hits[i] < ref_hits[i])
                {
                std::cout << "Particle " << i << std::endl;
                }
            UP_ASSERT_GREATER_EQUAL(hits[i], ref_hits[i]);
            }
        }
    }

// Test of basic LBVH build and traverse functionalities
UP_TEST( lbvh_small_test )
    {
    auto lbvh = std::make_shared<neighbor::LBVH>();

    // one point for tree
    neighbor::shared_array<float3> points(1);
        {
        points[0] = make_float3(2.5, 0., 0.);
        }
    // query spheres for tree
    neighbor::shared_array<float4> spheres(2);
        {
        // p0
        spheres[0] = make_float4(2.5, 0., 0., 0.5);
        // miss
        spheres[1] = make_float4(-0.5, 0., 0., 0.5);
        }

    std::cout << "Testing small LBVH build..." << std::endl;
    const float3 max = make_float3(1024, 1024, 1024);
    const float3 min = make_float3(0, 0, 0);
    lbvh->build(neighbor::PointInsertOp(points.get(), 1), min, max);
    cudaDeviceSynchronize();
        {
        UP_ASSERT_EQUAL(lbvh->getN(), 1);
        UP_ASSERT_EQUAL(lbvh->getRoot(), 0);

        // parents of each node
        UP_ASSERT_EQUAL(lbvh->getParents().size(), 1);
        auto parent = lbvh->getParents();
        UP_ASSERT_EQUAL(parent[0], neighbor::LBVHSentinel);

        UP_ASSERT_EQUAL(lbvh->getLeftChildren().size(), 0);
        UP_ASSERT_EQUAL(lbvh->getRightChildren().size(), 0);

        UP_ASSERT_EQUAL(lbvh->getPrimitives().size(), 1);
        auto data = lbvh->getPrimitives();
        UP_ASSERT_EQUAL(data[0], 0);

        UP_ASSERT_EQUAL(lbvh->getLowerBounds().size(), 1);
        UP_ASSERT_EQUAL(lbvh->getUpperBounds().size(), 1);
        auto lo = lbvh->getLowerBounds();
        auto hi = lbvh->getUpperBounds();

        // check leafs only
        UP_ASSERT_CLOSE(lo[0].x, 2.5f, 1.e-6f);
        UP_ASSERT_CLOSE(hi[0].x, 2.5f, 1.e-6f);
        }

    // test traverser
    std::cout << "Testing small traverser..." << std::endl;
        {
        neighbor::LBVHTraverser traverser;
        neighbor::shared_array<unsigned int> hits(spheres.size());
            {
            neighbor::CountNeighborsOp count(hits.get());

            neighbor::SphereQueryOp query(spheres.get(),
                                          spheres.size());

            traverser.traverse(*lbvh, query, count);
            cudaDeviceSynchronize();
            }

        auto data = traverser.getData();
        // only one node, just check its index contents
            {
            int4 node = data[0];
            UP_ASSERT_EQUAL(node.z, ~0);
            UP_ASSERT_EQUAL(node.w, neighbor::LBVHSentinel);
            }

        // each node should have the correct number of hits
        UP_ASSERT_EQUAL(hits[0], 1);
        UP_ASSERT_EQUAL(hits[1], 0);
        }
    }
