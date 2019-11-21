// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "neighbor/UniformGrid.h"
#include "neighbor/UniformGrid.cuh"
#include "neighbor/UniformGridTraverser.h"
#include "neighbor/OutputOps.h"
#include "neighbor/QueryOps.h"
#include "neighbor/TransformOps.h"
#include "neighbor/InsertOps.h"

#include "hoomd/GlobalArray.h"

#include "upp11_config.h"
HOOMD_UP_MAIN()

// Test of basic UniformGrid construction
UP_TEST( uniform_grid_test )
    {
    auto exec_conf = std::make_shared<const ExecutionConfiguration>(ExecutionConfiguration::GPU);

    // points for grid
    const Scalar3 lo = make_scalar3(-2., -3., -4.);
    const Scalar3 hi = make_scalar3(2., 3., 4.);
    GlobalArray<Scalar4> points(4, exec_conf);
        {
        ArrayHandle<Scalar4> h_points(points, access_location::host, access_mode::overwrite);
        h_points.data[0] = make_scalar4( 1.,  2.,  3., 0.);
        h_points.data[1] = make_scalar4(-1., -2., -1., 0.);
        h_points.data[2] = make_scalar4( 1.,  0.,  1., 0.);
        h_points.data[3] = make_scalar4( 1.1,  0.1,  1.1, 0.);
        }

    // width of 1.9 will be rounded up to 2
    auto grid = std::make_shared<neighbor::UniformGrid>(exec_conf, 1.9);
    grid->setup(4, lo, hi);

    // check read in
        {
        const Scalar3 glo = grid->getLo();
        UP_ASSERT_CLOSE(glo.x, -2.0, 1.e-6);
        UP_ASSERT_CLOSE(glo.y, -3.0, 1.e-6);
        UP_ASSERT_CLOSE(glo.z, -4.0, 1.e-6);

        const Scalar3 gL = grid->getL();
        UP_ASSERT_CLOSE(gL.x, 4.0, 1.e-6);
        UP_ASSERT_CLOSE(gL.y, 6.0, 1.e-6);
        UP_ASSERT_CLOSE(gL.z, 8.0, 1.e-6);
        }

    // check widths
        {
        const Scalar3 width = grid->getWidth();
        UP_ASSERT_CLOSE(width.x, 2.0, 1.e-6);
        UP_ASSERT_CLOSE(width.y, 2.0, 1.e-6);
        UP_ASSERT_CLOSE(width.z, 2.0, 1.e-6);

        const uint3 dim = grid->getDimensions();
        UP_ASSERT_EQUAL(dim.x, 2);
        UP_ASSERT_EQUAL(dim.y, 3);
        UP_ASSERT_EQUAL(dim.z, 4);

        const Index3D ci = grid->getIndexer();
        UP_ASSERT_EQUAL(ci.getW(), 2);
        UP_ASSERT_EQUAL(ci.getH(), 3);
        UP_ASSERT_EQUAL(ci.getD(), 4);

        UP_ASSERT_EQUAL(grid->getFirsts().getNumElements(), 2*3*4);
        UP_ASSERT_EQUAL(grid->getSizes().getNumElements(), 2*3*4);
        }

    // check allocation
        {
        UP_ASSERT_EQUAL(grid->getN(), 4);
        UP_ASSERT_EQUAL(grid->getCells().getNumElements(), 4);
        UP_ASSERT_EQUAL(grid->getPrimitives().getNumElements(), 4);
        }

    // build grid from points
        {
        ArrayHandle<Scalar4> d_points(points, access_location::device, access_mode::read);
        neighbor::GridPointOp op(d_points.data, 4);
        grid->build(op, lo, hi);
        }

    /* Check values
     *
     * Particle 0 is in (1, 2, 3)
     * Particle 1 is in (0, 0, 1)
     * Particle 2 is in (1, 1, 2)
     * Particle 3 is in (1, 1, 2)
     *
     * sorted order is 1, (2,3), 0
     */
        {
        // check that the expected cells have the right number of particles and right indexes into primitives
        const Index3D ci = grid->getIndexer();
        ArrayHandle<unsigned int> h_first(grid->getFirsts(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_sizes(grid->getSizes(), access_location::host, access_mode::read);
        UP_ASSERT_EQUAL(h_first.data[ci(0,0,1)], 0);
        UP_ASSERT_EQUAL(h_sizes.data[ci(0,0,1)], 1);
        UP_ASSERT_EQUAL(h_first.data[ci(1,1,2)], 1);
        UP_ASSERT_EQUAL(h_sizes.data[ci(1,1,2)], 2);
        UP_ASSERT_EQUAL(h_first.data[ci(1,2,3)], 3);
        UP_ASSERT_EQUAL(h_sizes.data[ci(1,2,3)], 1);

        // pick a random cell and ensure all flags are right
        UP_ASSERT_EQUAL(h_first.data[ci(0,0,0)], neighbor::gpu::UniformGridSentinel);
        UP_ASSERT_EQUAL(h_sizes.data[ci(0,0,0)], 0);

        // check that primitives and cells hold the right content, in sorted order
        ArrayHandle<unsigned int> h_primitives(grid->getPrimitives(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_cells(grid->getCells(), access_location::host, access_mode::read);
        UP_ASSERT_EQUAL(h_primitives.data[0], 1);
        UP_ASSERT_EQUAL(h_primitives.data[1], 2);
        UP_ASSERT_EQUAL(h_primitives.data[2], 3);
        UP_ASSERT_EQUAL(h_primitives.data[3], 0);

        UP_ASSERT_EQUAL(h_cells.data[0], ci(0,0,1));
        UP_ASSERT_EQUAL(h_cells.data[1], ci(1,1,2));
        UP_ASSERT_EQUAL(h_cells.data[2], ci(1,1,2));
        UP_ASSERT_EQUAL(h_cells.data[3], ci(1,2,3));
        }
    }

// Test of UniformGridTraverser
UP_TEST( uniform_grid_traverser_test )
    {
    auto exec_conf = std::make_shared<const ExecutionConfiguration>(ExecutionConfiguration::GPU);

    // points for grid
    const Scalar3 lo = make_scalar3(-2., -2., -2.);
    const Scalar3 hi = make_scalar3( 2.,  2.,  2.);
    GlobalArray<Scalar4> points(9, exec_conf);
        {
        ArrayHandle<Scalar4> h_points(points, access_location::host, access_mode::overwrite);
        h_points.data[0] = make_scalar4(-1.6,-1.6,-1.6, 0.);
        h_points.data[7] = make_scalar4(-1.6,-1.6, 1.6, 0.);
        h_points.data[1] = make_scalar4(-1.6, 1.6,-1.6, 0.);
        h_points.data[6] = make_scalar4(-1.6, 1.6, 1.6, 0.);
        h_points.data[2] = make_scalar4( 1.6,-1.6,-1.6, 0.);
        h_points.data[5] = make_scalar4( 1.6,-1.6, 1.6, 0.);
        h_points.data[3] = make_scalar4( 1.6, 1.6,-1.6, 0.);
        // two points in the last cell
        h_points.data[4] = make_scalar4( 1.6, 1.6, 1.6, 0.);
        h_points.data[8] = make_scalar4( 1.6, 1.6, 1.6, 0.);
        }

    const Scalar rcut = 1.0;
    auto grid = std::make_shared<neighbor::UniformGrid>(exec_conf, rcut);
        {
        ArrayHandle<Scalar4> d_points(points, access_location::device, access_mode::read);
        neighbor::GridPointOp op(d_points.data, 9);
        grid->build(op, lo, hi);
        }

    GlobalArray<Scalar4> spheres(4, exec_conf);
    GlobalArray<Scalar3> images(26, exec_conf);
    GlobalArray<unsigned int> hits(spheres.getNumElements(), exec_conf);
        {
        ArrayHandle<Scalar4> h_spheres(spheres, access_location::host, access_mode::overwrite);
        h_spheres.data[0] = make_scalar4(-1.6,-1.6,-1.6, 0.1);
        h_spheres.data[1] = make_scalar4( 1.6, 1.6, 1.6, 0.1);
        h_spheres.data[2] = make_scalar4(-2.,-2.,-2., rcut);
        h_spheres.data[3] = make_scalar4( 2., 2., 2., rcut);

        // generate 3d image list
        ArrayHandle<Scalar3> h_images(images, access_location::host, access_mode::overwrite);
        Scalar3 L = hi-lo;
        unsigned int idx=0;
        for (int k=-1; k <= 1; ++k)
            {
            for (int j=-1; j <= 1; ++j)
                {
                for (int i=-1; i <= 1; ++i)
                    {
                    // self-image is always included
                    if (i == 0 && j == 0 && k == 0) continue;
                    h_images.data[idx++] = make_scalar3(i*L.x, j*L.y, k*L.z);
                    }
                }
            }
        }

    // check that correct numbers of neighbors are obtained with different thread configs
    neighbor::UniformGridTraverser traverser(exec_conf);
        {
        ArrayHandle<unsigned int> d_hits(hits, access_location::device, access_mode::overwrite);
        neighbor::CountNeighborsOp count(d_hits.data);

        ArrayHandle<Scalar4> d_spheres(spheres, access_location::device, access_mode::read);
        neighbor::SphereQueryOp query(d_spheres.data, spheres.getNumElements());

        traverser.traverse(count, query, *grid, images);
        }

    ArrayHandle<unsigned int> h_hits(hits, access_location::host, access_mode::read);
    UP_ASSERT_EQUAL(h_hits.data[0], 1);
    UP_ASSERT_EQUAL(h_hits.data[1], 2);
    UP_ASSERT_EQUAL(h_hits.data[2], 9);
    UP_ASSERT_EQUAL(h_hits.data[3], 9);
    }

// Test that UniformGrid counts at least the same number of neighbors in an ideal gas as brute force
UP_TEST( uniform_grid_validate )
    {
    auto exec_conf = std::make_shared<const ExecutionConfiguration>(ExecutionConfiguration::GPU);

    // N particles in orthorhombic box
    const BoxDim box(20,15,25);
    const Scalar3 L = box.getL();
    const unsigned int N = static_cast<unsigned int>(1.0*L.x*L.y*L.z);
    const Scalar rcut = 1.0;
    auto grid = std::make_shared<neighbor::UniformGrid>(exec_conf, rcut);

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
        {
        ArrayHandle<Scalar4> d_points(points, access_location::device, access_mode::read);
        grid->build(neighbor::GridPointOp(d_points.data, N), box.getLo(), box.getHi());
        }

    // query spheres for grid
    GlobalArray<Scalar4> spheres(N, exec_conf);
    GlobalArray<Scalar3> images(26, exec_conf);
        {
        ArrayHandle<Scalar4> h_points(points, access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_spheres(spheres, access_location::host, access_mode::overwrite);
        for (unsigned int i=0; i < N; ++i)
            {
            const Scalar4 point = h_points.data[i];
            h_spheres.data[i] = make_scalar4(point.x, point.y, point.z, rcut);
            }

        // generate 3d image list
        ArrayHandle<Scalar3> h_images(images, access_location::host, access_mode::overwrite);
        unsigned int idx=0;
        for (int k=-1; k <= 1; ++k)
            {
            for (int j=-1; j <= 1; ++j)
                {
                for (int i=-1; i <= 1; ++i)
                    {
                    // self-image is always included
                    if (i == 0 && j == 0 && k == 0) continue;
                    h_images.data[idx++] = make_scalar3(i*L.x, j*L.y, k*L.z);
                    }
                }
            }
        }

    // build hit list
    neighbor::UniformGridTraverser traverser(exec_conf);
    GlobalArray<unsigned int> hits(N, exec_conf);
        {
        ArrayHandle<unsigned int> d_hits(hits, access_location::device, access_mode::overwrite);
        neighbor::CountNeighborsOp count(d_hits.data);

        ArrayHandle<Scalar4> d_spheres(spheres, access_location::device, access_mode::read);
        neighbor::SphereQueryOp query(d_spheres.data, spheres.getNumElements());

        traverser.traverse(count, query, *grid, images);
        }

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

    // check that grid always has the same hits as the reference
        {
        ArrayHandle<unsigned int> h_hits(hits, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_ref_hits(ref_hits, access_location::host, access_mode::read);
        for (unsigned int i=0; i < N; ++i)
            {
            if (h_hits.data[i] != h_ref_hits.data[i])
                {
                std::cout << "Particle " << i << std::endl;
                }
            UP_ASSERT_EQUAL(h_hits.data[i], h_ref_hits.data[i]);
            }
        }
    }
