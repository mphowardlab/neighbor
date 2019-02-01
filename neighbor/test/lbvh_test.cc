// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "neighbor/LBVH.h"
#include "neighbor/LBVH.cuh"
#include "neighbor/LBVHRopeTraverser.h"

#include "hoomd/BoxDim.h"
#include <random>

#include "upp11_config.h"
HOOMD_UP_MAIN()

// Test of basic LBVH build and traverse functionalities
UP_TEST( lbvh_test )
    {
    auto exec_conf = std::make_shared<const ExecutionConfiguration>(ExecutionConfiguration::GPU);
    auto lbvh = std::make_shared<neighbor::LBVH>(exec_conf);

    // points for tree
    GlobalArray<Scalar4> points(3, exec_conf);
        {
        ArrayHandle<Scalar4> h_points(points, access_location::host, access_mode::overwrite);
        h_points.data[0] = make_scalar4(2.5, 0., 0., 0.);
        h_points.data[1] = make_scalar4(1.5, 0., 0.,  0.);
        h_points.data[2] = make_scalar4(0.5, 0., 0., 0.);
        }

    // query spheres for tree
    GlobalArray<Scalar4> spheres(7, exec_conf);
        {
        ArrayHandle<Scalar4> h_spheres(spheres, access_location::host, access_mode::overwrite);
        // p2
        h_spheres.data[0] = make_scalar4(0.5, 0., 0., 0.5);
        // p1
        h_spheres.data[1] = make_scalar4(1.5, 0., 0., 0.5);
        // p0
        h_spheres.data[2] = make_scalar4(2.5, 0., 0., 0.5);
        // p2, p1
        h_spheres.data[3] = make_scalar4(1.0, 0., 0., 1.0);
        // p1, p0
        h_spheres.data[4] = make_scalar4(2.0, 0., 0., 1.0);
        // p2, p1, p0
        h_spheres.data[5] = make_scalar4(1.5, 0., 0., 1.5);
        // miss
        h_spheres.data[6] = make_scalar4(-0.5, 0., 0., 0.5);
        }

    exec_conf->msg->notice(0) << "Testing LBVH build..." << std::endl;
    /*
     *           0
     *          / \
     *         1   p0
     *        / \
     *       p2 p1
     */
    const Scalar3 max = make_scalar3(1024, 1024, 1024);
    const Scalar3 min = make_scalar3(0, 0, 0);
    lbvh->build(points, 3, min, max);
        {
        UP_ASSERT_EQUAL(lbvh->getN(), 3);
        UP_ASSERT_EQUAL(lbvh->getRoot(), 0);

        // parents of each node
        UP_ASSERT_EQUAL(lbvh->getParents().getNumElements(), 5);
        ArrayHandle<int> h_parent(lbvh->getParents(), access_location::host, access_mode::read);
        UP_ASSERT_EQUAL(h_parent.data[0], neighbor::gpu::LBVHSentinel);
        UP_ASSERT_EQUAL(h_parent.data[1], 0);
        UP_ASSERT_EQUAL(h_parent.data[2], 1);
        UP_ASSERT_EQUAL(h_parent.data[3], 1);
        UP_ASSERT_EQUAL(h_parent.data[4], 0);

        UP_ASSERT_EQUAL(lbvh->getLeftChildren().getNumElements(), 2);
        ArrayHandle<int> h_left(lbvh->getLeftChildren(), access_location::host, access_mode::read);
        UP_ASSERT_EQUAL(h_left.data[0], 1);
        UP_ASSERT_EQUAL(h_left.data[1], 2);

        UP_ASSERT_EQUAL(lbvh->getRightChildren().getNumElements(), 2);
        ArrayHandle<int> h_right(lbvh->getRightChildren(), access_location::host, access_mode::read);
        UP_ASSERT_EQUAL(h_right.data[0], 4);
        UP_ASSERT_EQUAL(h_right.data[1], 3);

        UP_ASSERT_EQUAL(lbvh->getPrimitives().getNumElements(), 3);
        ArrayHandle<unsigned int> h_data(lbvh->getPrimitives(), access_location::host, access_mode::read);
        UP_ASSERT_EQUAL(h_data.data[0], 2);
        UP_ASSERT_EQUAL(h_data.data[1], 1);
        UP_ASSERT_EQUAL(h_data.data[2], 0);

        UP_ASSERT_EQUAL(lbvh->getLowerBounds().getNumElements(), 5);
        UP_ASSERT_EQUAL(lbvh->getUpperBounds().getNumElements(), 5);
        ArrayHandle<float3> h_lo(lbvh->getLowerBounds(), access_location::host, access_mode::read);
        ArrayHandle<float3> h_hi(lbvh->getUpperBounds(), access_location::host, access_mode::read);

        // check leafs first
        UP_ASSERT_CLOSE(h_lo.data[2].x, 0.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_hi.data[2].x, 0.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_lo.data[3].x, 1.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_hi.data[3].x, 1.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_lo.data[4].x, 2.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_hi.data[4].x, 2.5f, 1.e-6f);
        // check internal node wrapping 2/3
        UP_ASSERT_CLOSE(h_lo.data[1].x, 0.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_hi.data[1].x, 1.5f, 1.e-6f);
        // check root node wrapping 2/3/4
        UP_ASSERT_CLOSE(h_lo.data[0].x, 0.5f, 1.e-6f);
        UP_ASSERT_CLOSE(h_hi.data[0].x, 2.5f, 1.e-6f);
        }

    // test rope traverser
    exec_conf->msg->notice(0) << "Testing rope traverser..." << std::endl;
        {
        neighbor::LBVHRopeTraverser traverser(exec_conf);
        GlobalArray<unsigned int> hits(spheres.getNumElements(), exec_conf);
        traverser.traverse(hits, spheres, spheres.getNumElements(), *lbvh);

        ArrayHandle<int4> h_data(traverser.getData(), access_location::host, access_mode::read);
        // Node 0
            {
            int4 node = h_data.data[0];
            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(node.z, 1);
            UP_ASSERT_EQUAL(node.w, neighbor::gpu::LBVHSentinel);
            }
        // Node 1
            {
            int4 node = h_data.data[1];

            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 511);
            UP_ASSERT_EQUAL(node.z, 2);
            UP_ASSERT_EQUAL(node.w, 4);
            }
        // Node 2
            {
            int4 node = h_data.data[2];

            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 1023);
            UP_ASSERT_EQUAL(node.z, ~2);
            UP_ASSERT_EQUAL(node.w, 3);
            }
        // Node 3
            {
            int4 node = h_data.data[3];

            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 511);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 511);
            UP_ASSERT_EQUAL(node.z, ~1);
            UP_ASSERT_EQUAL(node.w, 4);
            }
        // Node 4
            {
            int4 node = h_data.data[4];

            UP_ASSERT_EQUAL(((unsigned int)node.x >> 20) & 0x3ffu, 1023);
            UP_ASSERT_EQUAL(((unsigned int)node.y >> 20) & 0x3ffu, 0);
            UP_ASSERT_EQUAL(node.z, ~0);
            UP_ASSERT_EQUAL(node.w, neighbor::gpu::LBVHSentinel);
            }

        // each node should have the correct number of hits
        ArrayHandle<unsigned int> h_hits(hits, access_location::host, access_mode::read);
        UP_ASSERT_EQUAL(h_hits.data[0], 1);
        UP_ASSERT_EQUAL(h_hits.data[1], 1);
        UP_ASSERT_EQUAL(h_hits.data[2], 1);
        UP_ASSERT_EQUAL(h_hits.data[3], 2);
        UP_ASSERT_EQUAL(h_hits.data[4], 2);
        UP_ASSERT_EQUAL(h_hits.data[5], 3);
        UP_ASSERT_EQUAL(h_hits.data[6], 0);
        }
    }

// Test that LBVH traverser handles images correctly
UP_TEST( lbvh_periodic_test )
    {
    auto exec_conf = std::make_shared<const ExecutionConfiguration>(ExecutionConfiguration::GPU);
    auto lbvh = std::make_shared<neighbor::LBVH>(exec_conf);

    // points for tree
    GlobalArray<Scalar4> points(3, exec_conf);
        {
        ArrayHandle<Scalar4> h_points(points, access_location::host, access_mode::overwrite);
        h_points.data[0] = make_scalar4( 1.9, 1.9, 1.9, 0.);
        h_points.data[1] = make_scalar4(  0., 0.,  0., 0.);
        h_points.data[2] = make_scalar4(-1.9,-1.9,-1.9, 0.);
        }
    const Scalar3 max = make_scalar3( 2., 2., 2.);
    const Scalar3 min = make_scalar3(-2.,-2.,-2.);
    lbvh->build(points, points.getNumElements(), min, max);

    // query spheres for tree that intersect through boundaries
    GlobalArray<Scalar4> spheres(2, exec_conf);
    GlobalArray<Scalar3> images(26, exec_conf);
        {
        ArrayHandle<Scalar4> h_spheres(spheres, access_location::host, access_mode::overwrite);
        // p2
        h_spheres.data[0] = make_scalar4(-1.9, 1.9, 1.9, 0.5);
        // p1
        h_spheres.data[1] = make_scalar4( 1.9,-1.9,-1.9, 0.5);

        ArrayHandle<Scalar3> h_images(images, access_location::host, access_mode::overwrite);
        unsigned int idx=0;
        for (int ix=-1; ix <= 1; ++ix)
            {
            for (int iy=-1; iy <= 1; ++iy)
                {
                for (int iz=-1; iz <= 1; ++iz)
                    {
                    if (ix == 0 && iy == 0 && iz == 0) continue;

                    h_images.data[idx++] = make_scalar3(4*ix, 4*iy, 4*iz);
                    }
                }
            }
        }

    // no hits without images
    neighbor::LBVHRopeTraverser traverser(exec_conf);
    GlobalArray<unsigned int> hits(spheres.getNumElements(), exec_conf);
    traverser.traverse(hits, spheres, spheres.getNumElements(), *lbvh);
        {
        ArrayHandle<unsigned int> h_hits(hits, access_location::host, access_mode::read);
        UP_ASSERT_EQUAL(h_hits.data[0], 0);
        UP_ASSERT_EQUAL(h_hits.data[1], 0);
        }

    // 2 hits with images
    traverser.traverse(hits, spheres, spheres.getNumElements(), *lbvh, images);
        {
        ArrayHandle<unsigned int> h_hits(hits, access_location::host, access_mode::read);
        UP_ASSERT_EQUAL(h_hits.data[0], 2);
        UP_ASSERT_EQUAL(h_hits.data[1], 2);
        }
    }

// Test that LBVH counts at least the same number of neighbors in an ideal gas as brute force
UP_TEST( lbvh_validate )
    {
    auto exec_conf = std::make_shared<const ExecutionConfiguration>(ExecutionConfiguration::GPU);
    auto lbvh = std::make_shared<neighbor::LBVH>(exec_conf);

    // N particles in orthorhombic box
    const BoxDim box(20,15,25);
    const Scalar3 L = box.getL();
    const unsigned int N = static_cast<unsigned int>(1.0*L.x*L.y*L.z);
    const Scalar rcut = 1.0;

    // generate random points in the box
    GlobalArray<Scalar4> points(N, exec_conf);
        {
        ArrayHandle<Scalar4> h_points(points, access_location::host, access_mode::overwrite);
        std::mt19937 mt(42);
        std::uniform_real_distribution<Scalar> U(-0.5, 0.5);
        for (unsigned int i=0; i < N; ++i)
            {
            h_points.data[i] = make_scalar4(L.x*U(mt), L.y*U(mt), L.z*U(mt), __int_as_scalar(0));
            }
        }
    lbvh->build(points, N, box.getLo(), box.getHi());

    // query spheres for tree
    GlobalArray<Scalar4> spheres(N, exec_conf);
        {
        ArrayHandle<Scalar4> h_points(points, access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_spheres(spheres, access_location::host, access_mode::overwrite);
        for (unsigned int i=0; i < N; ++i)
            {
            const Scalar4 point = h_points.data[i];
            h_spheres.data[i] = make_scalar4(point.x, point.y, point.z, rcut);
            }
        }

    // traversal images
    GlobalArray<Scalar3> images(26, exec_conf);
        {
        ArrayHandle<Scalar3> h_images(images, access_location::host, access_mode::overwrite);
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

    // build hit list
    GlobalArray<unsigned int> hits(N, exec_conf);
    neighbor::LBVHRopeTraverser traverser(exec_conf);
    traverser.traverse(hits, spheres, spheres.getNumElements(), *lbvh, images);

    // generate list of reference collisions
    GlobalArray<unsigned int> ref_hits(N, exec_conf);
        {
        const Scalar rcut2 = rcut*rcut;
        ArrayHandle<unsigned int> h_ref_hits(ref_hits, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_points(points, access_location::host, access_mode::read);
        std::fill(h_ref_hits.data, h_ref_hits.data + N, 0);
        for (unsigned int i=0; i < N; ++i)
            {
            const Scalar4 ri = h_points.data[i];
            for (unsigned int j=i; j < N; ++j)
                {
                const Scalar4 rj = h_points.data[j];
                Scalar3 dr = make_scalar3(rj.x-ri.x, rj.y-ri.y, rj.z-ri.z);
                dr = box.minImage(dr);
                const Scalar dr2 = dot(dr,dr);
                if (dr2 <= rcut2)
                    {
                    ++h_ref_hits.data[i];
                    if (j != i)
                        ++h_ref_hits.data[j];
                    }
                }
            }
        }

    // check that tree always has at least as many hits as the reference
        {
        ArrayHandle<unsigned int> h_hits(hits, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_ref_hits(ref_hits, access_location::host, access_mode::read);
        for (unsigned int i=0; i < N; ++i)
            {
            if (h_hits.data[i] < h_ref_hits.data[i])
                {
                std::cout << "Particle " << i << std::endl;
                }
            UP_ASSERT_GREATER_EQUAL(h_hits.data[i], h_ref_hits.data[i]);
            }
        }
    }
