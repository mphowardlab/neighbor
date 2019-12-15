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

//! Double-buffered device array.
/*!
 * This object holds two shared_array objects that can be used in a double-buffered style, e.g., in cub::DeviceRadixSort.
 * There is a nominal "current" array and nominal "alternate" array. A bit can be toggled using ::flip() to change which
 * shared_array corresponds to which. Hence, no pointers ever need to be exchanged, and code needing to toggle between
 * the buffers can be simplified.
 *
 * \tparam T Data type to allocate.
 */
template<typename T>
class buffered_array
    {
    public:
        //! Empty constructor.
        buffered_array()
            : x_(0), y_(0), selector_(0)
            {}

        //! Constructor for fixed count.
        /*!
         * \param count Number of elements to allocate.
         */
        explicit buffered_array(size_t size)
            : x_(size), y_(size), selector_(0)
            {}

        //! Swap operation.
        /*!
         * \param other An existing shared_array.
         *
         * The data in this shared_array is swapped with the data in \a other.
         */
        void swap(buffered_array& other)
            {
            std::swap(x_, other.x_);
            std::swap(y_, other.y_);
            std::swap(selector_, other.selector_);
            }

        //! Get the current (active) array.
        /*!
         * The x_ member array is current when the selector_ is 0.
         */
        shared_array<T>& current()
            {
            return (selector_ == 0) ? x_ : y_;
            }

        //! Get the currently (active) array.
        /*!
         * The x_ member array is current when the selector_ is 0.
         */
        shared_array<T> const& current() const
            {
            return (selector_ == 0) ? x_ : y_;
            }

        //! Get the alternate array.
        /*!
         * The y_ member array is alternate when the selector_ is 0.
         */
        shared_array<T>& alternate()
            {
            return (selector_ == 0) ? y_ : x_;
            }

        //! Get the alternate array.
        /*!
         * The y_ member array is alternate when the selector_ is 0.
         */
        shared_array<T> const& alternate() const
            {
            return (selector_ == 0) ? y_ : x_;
            }

        //! Flip the current and alternate arrays.
        /*!
         * This procedure is a very quick bit operation and does \b not change the arrays themselves.
         */
        void flip()
            {
            selector_ ^= 1;
            }

        //! Get the number of elements in the array.
        size_t size() const
            {
            return x_.size();
            }

    private:
        shared_array<T> x_; //!< First array
        shared_array<T> y_; //!< Second array
        int selector_;      //!< Bit setting if x_ (0) or y_ (1) is current
    };

} // end namespace neighbor

#endif // NEIGHBOR_MEMORY_H_
