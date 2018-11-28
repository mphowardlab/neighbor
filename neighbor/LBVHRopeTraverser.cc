// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "LBVHRopeTraverser.h"
#include "LBVHRopeTraverser.cuh"

namespace neighbor
{

/*!
 * \param exec_conf HOOMD-blue execution configuration.
 */
LBVHRopeTraverser::LBVHRopeTraverser(std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : LBVHTraverser(exec_conf), m_lbvh_lo(exec_conf), m_lbvh_hi(exec_conf), m_bins(exec_conf)
    {
    m_exec_conf->msg->notice(4) << "Constructing LBVHRopeTraverser" << std::endl;

    m_tune_traverse.reset(new Autotuner(32, 1024, 32, 5, 100000, "lbvh_rope_traverse", m_exec_conf));
    m_tune_compress.reset(new Autotuner(32, 1024, 32, 5, 100000, "lbvh_rope_compress", m_exec_conf));
    }

LBVHRopeTraverser::~LBVHRopeTraverser()
    {
    m_exec_conf->msg->notice(4) << "Destroying LBVHRopeTraverser" << std::endl;
    }

/*!
 * \param out Number of overlaps per sphere.
 * \param spheres Test spheres.
 * \param N Number of test spheres.
 * \param lbvh LBVH to traverse.
 * \param images Additional images of \a spheres to test.
 *
 * The format for a \a sphere is (x,y,z,R), where R is the radius of the sphere.
 *
 * A maximum of 32 \a images are allowed due to the internal representation of the image list
 * in the traversal CUDA kernel. This is more than enough to perform traversal in 3D periodic
 * boundary conditions (26 additional images). Multiple calls to ::traverse are required if
 * more images are needed, but \a out will be overwritten each time.
 *
 * If a query sphere overlaps an internal node, the traversal should descend to the left child.
 * If the query sphere does not overlap OR it has reached a leaf node, the traversal should proceed
 * along the rope. Traversal terminates when the LBVHSentinel is reached for the rope.
 */
void LBVHRopeTraverser::traverse(const GPUArray<unsigned int>& out,
                                 const GPUArray<Scalar4>& spheres,
                                 unsigned int N,
                                 const LBVH& lbvh,
                                 const GPUArray<Scalar3>& images)
    {
    // kernel uses int32 bitflags for the images, so limit to 32 images
    const unsigned int Nimages = images.getNumElements();
    if (Nimages > 32)
        {
        m_exec_conf->msg->error() << "A maximum of 32 image vectors are supported by LBVH traversers." << std::endl;
        throw std::runtime_error("Too many images (>32) in LBVH traverser.");
        }

    // compress the tree
    compress(lbvh);

    // traverse the tree
    ArrayHandle<unsigned int> d_out(out, access_location::device, access_mode::overwrite);
    ArrayHandle<int4> d_data(m_data, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_spheres(spheres, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_images(images, access_location::device, access_mode::read);

    m_tune_traverse->begin();
    gpu::lbvh_traverse_ropes(d_out.data,
                             lbvh.getRoot(),
                             d_data.data,
                             m_lbvh_lo.getDeviceFlags(),
                             m_lbvh_hi.getDeviceFlags(),
                             m_bins.getDeviceFlags(),
                             d_spheres.data,
                             d_images.data,
                             Nimages,
                             N,
                             m_tune_traverse->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tune_traverse->end();
    }

/*!
 * \param lbvh LBVH to compress
 *
 * The nodes are compressed according to the scheme described previously. The storage
 * requirements are 16B / node (int4). The components of the int4 are:
 *
 *  - x: bits = 00lo.x[0-9]lo.y[0-9]lo.z[0-9]
 *  - y: bits = 00hi.x[0-9]hi.y[0-9]hi.z[0-9]
 *  - z: left child node (if >= 0) or primitive (if < 0)
 *  - w: rope
 *
 * The bits for the bounding box can be decompressed using:
 *      lo.x = ((unsigned int)node.x >> 20) & 0x3ffu;
 *      lo.y = ((unsigned int)node.x >> 10) & 0x3ffu;
 *      lo.z = ((unsigned int)node.x      ) & 0x3ffu;
 * which simply shifts and masks the low 10 bits. These integer bins should then be scaled by
 * the compressed bin size, which is stored internally.
 *
 * If node.z >= 0, then the current node is an internal node, and traversal should descend
 * to the child (node.z). If node.z < 0, the current node is actually a leaf node. In this case,
 * there is no left child. Instead, ~node.z gives the original index of the intersected primitive.
 */
void LBVHRopeTraverser::compress(const LBVH& lbvh)
    {
    // resize the internal data array
    const unsigned int num_data = lbvh.getNNodes();
    if (num_data > m_data.getNumElements())
        {
        GPUArray<int4> tmp(num_data, m_exec_conf);
        m_data.swap(tmp);
        }
    ArrayHandle<int4> d_data(m_data, access_location::device, access_mode::overwrite);

    // acquire current tree data for reading
    ArrayHandle<int> d_parent(lbvh.getParents(), access_location::device, access_mode::read);
    ArrayHandle<int> d_left(lbvh.getLeftChildren(), access_location::device, access_mode::read);
    ArrayHandle<int> d_right(lbvh.getRightChildren(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_sorted_indexes(lbvh.getPrimitives(), access_location::device, access_mode::read);
    ArrayHandle<float3> d_lo(lbvh.getLowerBounds(), access_location::device, access_mode::read);
    ArrayHandle<float3> d_hi(lbvh.getUpperBounds(), access_location::device, access_mode::read);

    gpu::LBVHData tree;
    tree.parent = d_parent.data;
    tree.left = d_left.data;
    tree.right = d_right.data;
    tree.primitive = d_sorted_indexes.data;
    tree.lo = d_lo.data;
    tree.hi = d_hi.data;
    tree.root = lbvh.getRoot();

    // compress the data
    m_tune_compress->begin();
    gpu::lbvh_compress_ropes(d_data.data,
                             m_lbvh_lo.getDeviceFlags(),
                             m_lbvh_hi.getDeviceFlags(),
                             m_bins.getDeviceFlags(),
                             tree,
                             lbvh.getNInternal(),
                             lbvh.getNNodes(),
                             m_tune_compress->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tune_compress->end();
    }

} // end namespace neighbor
