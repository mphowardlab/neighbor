// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

// Modified source code for the Autotuner is used under the following license:
// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.
/*
BSD 3-Clause License for HOOMD-blue

Copyright (c) 2009-2019 The Regents of the University of Michigan All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef NEIGHBOR_AUTOTUNER_H_
#define NEIGHBOR_AUTOTUNER_H_

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace neighbor
{

class Autotuner
    {
    public:
        //! Constructor with implicit range
        Autotuner(unsigned int start,
                  unsigned int end,
                  unsigned int step,
                  unsigned int nsamples,
                  unsigned int period);

        //! Destructor
        ~Autotuner();

        //! Call before kernel launch
        void begin();

        //! Call after kernel launch
        void end();

        //! Get the parameter to set for the kernel launch
        /*! \returns the current parameter that should be set for the kernel launch

        While sampling, the value returned by this function will sweep though all valid parameters. Otherwise, it will
        return the fastest performing parameter.
        */
        unsigned int getParam()
            {
            return m_current_param;
            }

        //! Enable/disable sampling
        /*! \param enabled true to enable sampling, false to disable it
        */
        void setEnabled(bool enabled)
            {
            m_enabled = enabled;

            if (!enabled && isComplete())
                {
                // ensure that we are in the idle state and have an up to date optimal parameter
                m_current_element = 0;
                m_state = IDLE;
                m_current_param = computeOptimalParameter();
                }
            }

        //! Test if initial sampling is complete
        /*! \returns true if the initial sampling run is complete
        */
        bool isComplete()
            {
            return (m_state != STARTUP);
            }

        //! Change the sampling period
        /*! \param period New period to set
        */
        void setPeriod(unsigned int period)
            {
            m_period = period;
            }

    protected:
        unsigned int computeOptimalParameter();

        //! State names
        enum State
           {
           STARTUP,
           IDLE,
           SCANNING
           };

        // parameters
        unsigned int m_nsamples;    //!< Number of samples to take for each parameter
        unsigned int m_period;      //!< Number of calls before sampling occurs again
        bool m_enabled;             //!< True if enabled
        std::vector<unsigned int> m_parameters;  //!< valid parameters

        // state info
        State m_state;                  //!< Current state
        unsigned int m_current_sample;  //!< Current sample taken
        unsigned int m_current_element; //!< Index of current parameter sampled
        unsigned int m_calls;           //!< Count of the number of calls since the last sample
        unsigned int m_current_param;   //!< Value of the current parameter

        std::vector< std::vector< float > > m_samples;  //!< Raw sample data for each element
        std::vector< float > m_sample_median;           //!< Current sample median for each element

        cudaEvent_t m_start;      //!< CUDA event for recording start times
        cudaEvent_t m_stop;       //!< CUDA event for recording end times
    };

/*! \param start first valid parameter
    \param end last valid parameter
    \param step spacing between valid parameters
    \param nsamples Number of time samples to take at each parameter
    \param period Number of calls to begin() before sampling is redone
    \param name Descriptive name (used in messenger output)
    \param exec_conf Execution configuration

    \post Valid parameters will be generated with a spacing of \a step in the range [start,end] inclusive.
*/
Autotuner::Autotuner(unsigned int start,
                     unsigned int end,
                     unsigned int step,
                     unsigned int nsamples,
                     unsigned int period)
    : m_nsamples(nsamples), m_period(period), m_enabled(true),
      m_state(STARTUP), m_current_sample(0), m_current_element(0), m_calls(0), m_current_param(0)
    {
    // initialize the parameters
    m_parameters.resize((end - start) / step + 1);
    unsigned int cur_param = start;
    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        m_parameters[i] = cur_param;
        cur_param += step;
        }
    m_current_param = m_parameters[m_current_element];

    // ensure that m_nsamples is odd (so the median is easy to get). This also ensures that m_nsamples > 0.
    if ((m_nsamples & 1) == 0) m_nsamples += 1;
    m_samples.resize(m_parameters.size());
    m_sample_median.resize(m_parameters.size());
    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        m_samples[i].resize(m_nsamples);
        }

    // create CUDA events
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
    }

Autotuner::~Autotuner()
    {
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);
    }

void Autotuner::begin()
    {
    // skip if disabled
    if (!m_enabled)
        return;

    if (m_state == STARTUP || m_state == SCANNING)
        {
        cudaEventRecord(m_start, 0);
        }
    }

void Autotuner::end()
    {
    // skip if disabled
    if (!m_enabled)
        return;

    // handle timing updates if scanning
    if (m_state == STARTUP || m_state == SCANNING)
        {
        cudaEventRecord(m_stop, 0);
        cudaEventSynchronize(m_stop);
        cudaEventElapsedTime(&m_samples[m_current_element][m_current_sample], m_start, m_stop);
        }

    // handle state data updates and transitions
    if (m_state == STARTUP)
        {
        // move on to the next sample
        m_current_sample++;

        // if we hit the end of the samples, reset and move on to the next element
        if (m_current_sample >= m_nsamples)
            {
            m_current_sample = 0;
            m_current_element++;

            // if we hit the end of the elements, transition to the IDLE state and compute the optimal parameter
            if (m_current_element >= m_parameters.size())
                {
                m_current_element = 0;
                m_state = IDLE;
                m_current_param = computeOptimalParameter();
                }
            else
                {
                // if moving on to the next element, update the cached parameter to set
                m_current_param = m_parameters[m_current_element];
                }
            }
        }
    else if (m_state == SCANNING)
        {
        // move on to the next element
        m_current_element++;

        // if we hit the end of the elements, transition to the IDLE state and compute the optimal parameter, and move
        // on to the next sample for next time
        if (m_current_element >= m_parameters.size())
            {
            m_current_element = 0;
            m_state = IDLE;
            m_current_param = computeOptimalParameter();
            m_current_sample = (m_current_sample + 1) % m_nsamples;
            }
        else
            {
            // if moving on to the next element, update the cached parameter to set
            m_current_param = m_parameters[m_current_element];
            }
        }
    else if (m_state == IDLE)
        {
        // increment the calls counter and see if we should transition to the scanning state
        m_calls++;

        if (m_calls > m_period)
            {
            // reset state for the next time
            m_calls = 0;

            // initialize a scan
            m_current_param = m_parameters[m_current_element];
            m_state = SCANNING;
            }
        }
    }

/*! \returns The optimal parameter given the current data in m_samples

    computeOptimalParameter computes the median time among all samples for a given element. It then chooses the
    fastest time (with the lowest index breaking a tie) and returns the parameter that resulted in that time.
*/
unsigned int Autotuner::computeOptimalParameter()
    {
    // start by computing the median for each element
    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        std::vector<float> v = m_samples[i];
        size_t n = v.size() / 2;
        std::nth_element(v.begin(), v.begin()+n, v.end());
        m_sample_median[i] = v[n];
        }

    // now find the minimum time in the medians
    float min = m_sample_median[0];
    unsigned int min_idx = 0;
    for (unsigned int i = 1; i < m_parameters.size(); i++)
        {
        if (m_sample_median[i] < min)
            {
            min = m_sample_median[i];
            min_idx = i;
            }
        }

    // get the optimal param
    return m_parameters[min_idx];
    }

} // end namespace neighbor

#endif // NEIGHBOR_AUTOTUNER_H_
