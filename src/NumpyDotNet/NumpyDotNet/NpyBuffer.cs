/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018-2019
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{
    /// <summary>
    /// Extended buffer protocol.   It does not implement IBufferProtocol for now
    /// because several methods on that interface are unnecessary for the time being.
    /// </summary>
    public interface IExtBufferProtocol
    {
        /// <summary>
        /// Number of items in the buffer
        /// </summary>
        long ItemCount {
            get;
        }

        string Format {
            get;
        }

        /// <summary>
        /// Size of each element in bytes.
        /// </summary>
        int ItemSize {
            get;
        }

        /// <summary>
        /// Number of dimensions in each array.
        /// </summary>
        int NumberDimensions {
            get;
        }

        /// <summary>
        /// True if array can not be written to, false if data is writable.
        /// </summary>
        bool ReadOnly {
            get;
        }

        /// <summary>
        /// Size of each dimension in array elements.
        /// </summary>
        /// <returns>List of each dimension size</returns>
        IList<npy_intp> Shape {
            get;
        }

        /// <summary>
        /// Number of bytes to skip to get to the next element in each dimension.
        /// </summary>
        npy_intp[] Strides {
            get;
        }

        long[] SubOffsets {
            get;
        }
  

        /// <summary>
        /// Total size of the buffer in bytes.
        /// </summary>
        long Size {
            get;
        }
    }


    /// <summary>
    /// Indicates that a given object can provide an adapter that implements the buffer
    /// protocol.
    /// </summary>
    public interface IBufferProvider
    {
        IExtBufferProtocol GetBuffer(NpyBuffer.PyBuf flags);
    }



    // Temporary until real IPythonBufferable is exposed.
    public interface IPythonBufferable
    {
          int Size {
            get;
        }
    }


    /// <summary>
    /// Provides utilities and adapters that mimick the CPython PEP 3118 buffer protocol
    /// as far as currently needed.
    /// </summary>
    public static class NpyBuffer
    {
        [Flags]
        public enum PyBuf
        {
            SIMPLE = 0x00,
            WRITABLE = 0x01,
            FORMAT = 0x04,
            ND = 0x08,
            STRIDES = 0x18,         // Implies ND
            C_CONTIGUOUS = 0x38,    // Implies STRIDES
            F_CONTIGUOUS = 0x58,    // Implies STRIDES
            ANY_CONTIGUOUS = 0x98,  // Implies STRIDES
            INDIRECT = 0x118,       // Implies STRIDES

            // Composite sets
            CONTIG = 0x41,          // Multidimensional ( ND | WRITABLE )
            CONTIG_RO = 0x40,       // ND
            STRIDED = 0x0D,         // Multidimensional, aligned ( STRIDES | WRITABLE )
            STRIDED_RO = 0x0C,      // STRIDES
            RECORDS = 0x0F,         // Multidimensional, unaligned (STRIDEs | WRITABLE | FORMAT )
            RECORDS_RO = 0x0E,      // STRIDES | FORMAT
            FULL = 0x8F,            // Multidimensional using sub-offsets
            FULL_RO = 0x8E          //
        }



    }
}
