// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "LBVHTraverser.h"
#include "OutputOps.h"
#include "QueryOps.h"

namespace neighbor
{
/*!
 * \param exec_conf HOOMD-blue execution configuration.
 */
LBVHTraverser::LBVHTraverser(std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_exec_conf(exec_conf), m_lbvh_lo(exec_conf), m_lbvh_hi(exec_conf), m_bins(exec_conf)
    {
    m_exec_conf->msg->notice(4) << "Constructing LBVHTraverser" << std::endl;

    m_tune_traverse.reset(new Autotuner(32, 1024, 32, 5, 100000, "lbvh_rope_traverse", m_exec_conf));
    m_tune_compress.reset(new Autotuner(32, 1024, 32, 5, 100000, "lbvh_rope_compress", m_exec_conf));
    }

LBVHTraverser::~LBVHTraverser()
    {
    m_exec_conf->msg->notice(4) << "Destroying LBVHTraverser" << std::endl;
    }

template void LBVHTraverser::traverse(CountNeighborsOp& out,
                                      const SphereQueryOp& query,
                                      const LBVH& lbvh,
                                      const GlobalArray<Scalar3>& images);

template void LBVHTraverser::traverse(NeighborListOp& out,
                                      const SphereQueryOp& query,
                                      const LBVH& lbvh,
                                      const GlobalArray<Scalar3>& images);

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
void LBVHTraverser::compress(const LBVH& lbvh)
    {
    // resize the internal data array
    const unsigned int num_data = lbvh.getNNodes();
    if (num_data > m_data.getNumElements())
        {
        GlobalArray<int4> tmp(num_data, m_exec_conf);
        m_data.swap(tmp);
        }

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

    // acquire compressed tree data for writing
    gpu::LBVHCompressedData ctree;
    ArrayHandle<int4> d_data(m_data, access_location::device, access_mode::overwrite);
    ctree.root = lbvh.getRoot();
    ctree.data = d_data.data;
    ctree.lo = m_lbvh_lo.getDeviceFlags();
    ctree.hi = m_lbvh_hi.getDeviceFlags();
    ctree.bins = m_bins.getDeviceFlags();

    // compress the data
    m_tune_compress->begin();
    gpu::lbvh_compress_ropes(ctree,
                             tree,
                             lbvh.getNInternal(),
                             lbvh.getNNodes(),
                             m_tune_compress->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tune_compress->end();
    }

} // end namespace neighbor
