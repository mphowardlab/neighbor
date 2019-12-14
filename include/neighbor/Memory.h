// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_MEMORY_H_
#define NEIGHBOR_MEMORY_H_

#include <cuda_runtime.h>
#include <memory>

namespace neighbor
{

//! Smart pointer for device array.
/*!
 * This object is a thin wrapper around a std::shared_ptr. The underlying raw pointer can be acquired using the ::get()
 * method. The allocation is CUDA managed memory, so the pointer can also be accessed on the host (e.g., using the
 * [] operator). However, it is the responsibility of the caller to synchronize the GPU before accessing it if this is
 * required by their hardware.
 *
 * This array functions like a std::shared_ptr so that copies of the array point to the same underlying memory.
 * As such, the array cannot be resized after it is constructed. The memory will only be freed after all copies
 * have been destroyed.
 *
 * \tparam T Data type to allocate.
 */
template<typename T>
class shared_array
    {
    public:
        //! Empty constructor.
        shared_array()
            {
            allocate(0);
            }

        //! Constructor for fixed count.
        /*!
         * \param count Number of elements to allocate.
         */
        explicit shared_array(size_t size)
            {
            allocate(size);
            }

        //! Copy constructor.
        /*!
         * \param other An existing shared_array.
         *
         * A copy of the array is created that shares ownership of the data with \a other.
         */
        shared_array(const shared_array& other)
            : data_(other.data_), size_(other.size_)
            {}

        //! Copy assignment.
        /*!
         * \param other An existing shared_array.
         * \returns A reference to the new shared_array.
         *
         * A copy of the array is created that shares ownership of the data with \a other.
         */
        shared_array& operator=(const shared_array& other)
            {
            if (this != &other)
                {
                data_ = other.data_;
                size_= other.size_;
                }
            return *this;
            }

        //! Move constructor.
        /*!
         * \param other An existing shared_array.
         *
         * The data from \a other is moved into the new shared_array.
         */
        shared_array(shared_array&& other)
            : data_(std::move(other.data_)), size_(std::move(other.size_))
            {}

        //! Move assignment.
        /*!
         * \param other An existing shared_array.
         * \returns A reference to the new shared_array.
         *
         * The data from \a other is moved into the new shared_array.
         */
        shared_array& operator=(shared_array&& other)
            {
            if (this != &other)
                {
                data_ = std::move(other.data_);
                size_ = std::move(other.size_);
                }
            return *this;
            }

        //! Swap operation.
        /*!
         * \param other An existing shared_array.
         *
         * The data in this shared_array is swapped with the data in \a other.
         */
        void swap(shared_array& other)
            {
            std::swap(data_, other.data_);
            std::swap(size_, other.size_);
            }

        //! Index operator.
        /*!
         * \param idx Index to access.
         * \returns A reference to the data at the index.
         */
        T& operator[](size_t idx)
            {
            return data_.get()[idx];
            }

        //! Index operator.
        /*!
         * \param idx Index to access.
         * \returns A constant reference to the data at the index.
         */
        T const& operator[](size_t idx) const
            {
            return data_.get()[idx];
            }

        //! Get the raw pointer for the data.
        T* get()
            {
            return data_.get();
            }

        //! Get a constant raw pointer to the data.
        T const* get() const
            {
            return data_.get();
            }

        //! Get the number of elements in the array.
        size_t size() const
            {
            return size_;
            }

    private:
        std::shared_ptr<T> data_;   //!< Underlying data
        size_t size_;               //!< Number of elements in array

        //! Custom deleter for CUDA memory
        struct deleter
            {
            void operator()(T* ptr)
                {
                if(ptr) cudaFree(ptr);
                }
            };

        //! Allocate memory.
        /*!
         * \param size Number of elements to allocate.
         *
         * The requested memory is allocated using cudaMallocManaged. If an error occurs, no memory allocation occurs
         * and an exception is raised. If \a size is 0, then the memory is freed.
         */
        void allocate(size_t size)
            {
            if (size > 0)
                {
                T* data = nullptr;
                cudaError_t code = cudaMallocManaged(&data, size*sizeof(T));
                if (code == cudaSuccess)
                    {
                    data_ = std::shared_ptr<T>(data, deleter());
                    size_ = size;
                    }
                else
                    {
                    deallocate();
                    throw std::runtime_error("Error allocating managed memory.");
                    }
                }
            else
                {
                deallocate();
                }
            }

        //! Deallocate memory.
        void deallocate()
            {
            data_.reset();
            size_ = 0;
            }
    };

} // end namespace neighbor

#endif // NEIGHBOR_MEMORY_H_
