// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_TUNER_H_
#define NEIGHBOR_TUNER_H_

#include <hipper/hipper_runtime.h>
#include <set>
#include <stdexcept>
#include <vector>

namespace neighbor
{

//! Base for a class with a tunable method.
/*!
 * The Tunable class species base methods for supplying a single tunable parameter, e.g., a kernel block size.
 * The range of valid launch parameters can be set by the constructor or using a list. This list is accessible via
 * ::getTunableParameters, which can be used in a runtime autotuner. The class also exposes a LaunchParameters object
 * that can be used to pack this parameter with other kernel launch parameters like the CUDA stream. Last, the tunable
 * class supplies a method for checking if a set of LaunchParameters is valid. This can be used to validate parameters
 * before a kernel launch.
 */
template<typename T>
class Tunable
    {
    public:
        //! Constructor from range.
        /*!
         * \param begin First value for tunable parameter.
         * \param end Last value for tunable parameter.
         * \param step Step size for tunable parameter.
         *
         * The parameter list is filled with values in the range [begin,end] in steps of \a step.
         */
        Tunable(T begin, T end, T step)
            {
            for (T param=begin; param <= end; param += step)
                {
                m_params.insert(param);
                }
            }

        //! Constructor from vector.
        /*!
         * \param params Vector of valid tunable parameters.
         */
        Tunable(const std::vector<T>& params)
            {
            setTunableParameters(params);
            }

        //! Get the list of valid tuner parameters.
        /*!
         * \returns The vector of valid tunable parameters.
         */
        std::vector<T> getTunableParameters() const
            {
            return std::vector<T>(m_params.begin(), m_params.end());
            }

        //! Set the list of valid tuner parameters.
        /*!
         * \param params Vector of valid tunable parameters.
         */
        void setTunableParameters(const std::vector<T>& params)
            {
            m_params = std::set<T>(params.begin(), params.end());
            }

        //! Structure holding the kernel launch parameters.
        /*!
         * This object is useful for specifying both a tunable parameter and an execution stream.
         */
        struct LaunchParameters
            {
            typedef T type;

            //! Constructor with default stream.
            /*!
             * \param tunable_ Tunable parameter.
             */
            explicit LaunchParameters(T tunable_)
                : tunable(tunable_), stream(0)
                {}

            //! Constructor with stream.
            /*!
             * \param tunable_ Tunable parameter.
             * \param stream_ CUDA stream for execution.
             */
            LaunchParameters(T tunable_, hipper::stream_t stream_)
                : tunable(tunable_), stream(stream_)
                {}

            T tunable;                  //!< Tunable parameter (e.g., block size)
            hipper::stream_t stream;    //!< Stream for execution
            };

        //! Check if a parameter is valid.
        /*!
         * \param params Launch parameters, including tuning parameter.
         * \param returns The tunable parameter.
         *
         * \raises An error if the tuning parameter is not in the set.
         */
        T checkParameter(const LaunchParameters& params) const
            {
            bool valid = inParameterSet(params.tunable);
            if (!valid)
                {
                throw std::runtime_error("Invalid tuning parameter.");
                }
            return params.tunable;
            }

    private:
        std::set<T> m_params;   //!< Set of tunable parameters

        //! Check if a parameter is in the tuning set.
        /*!
         * \param param Tuning parameter.
         * \returns True if the tuning parameter if it is in the set.
         */
        bool inParameterSet(T param) const
            {
            auto result = m_params.find(param);
            return (result != m_params.end());
            }
    };
}

#endif // NEIGHBOR_TUNER_H_
