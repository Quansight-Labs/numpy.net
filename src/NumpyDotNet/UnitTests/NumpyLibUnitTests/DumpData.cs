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

using NumpyLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyLibTests
{
    static class DumpData
    {

        public static string DumpArray(NpyArray arr, bool repr)
        {
            // Equivalent to array_repr_builtin (arrayobject.c)
            StringBuilder sb = new StringBuilder();
            if (repr)
                sb.Append("array(");

            DumpArray(arr, sb, arr.dimensions, arr.strides, 0, 0);

            if (repr)
            {
                if (false /*NpyDefs.IsExtended(arr.Dtype.TypeNum)*/)
                {
                    sb.AppendFormat(", '{0}{1}')", arr.ItemType, arr.ItemType);
                }
                else
                {
                    sb.AppendFormat(", '{0}')", arr.ItemType);
                }
            }
            return sb.ToString();
        }

        /// <summary>
        /// Recursively walks the array and appends a representation of each element
        /// to the passed string builder.  Square brackets delimit each array dimension.
        /// </summary>
        /// <param name="sb">StringBuilder instance to append to</param>
        /// <param name="dimensions">Array of size of each dimension</param>
        /// <param name="strides">Offset in bytes to reach next element in each dimension</param>
        /// <param name="dimIdx">Index of the current dimension (starts at 0, recursively counts up)</param>
        /// <param name="offset">Byte offset into data array, starts at 0</param>
        private static void DumpArray(NpyArray arr, StringBuilder sb, npy_intp[] dimensions, npy_intp[] strides, int dimIdx, long offset)
        {

            if (dimIdx == arr.nd)
            {
                Object value = arr.descr.f.getitem(offset, arr);
                if (value == null)
                {
                    sb.Append("None");
                }
                else
                {
                    sb.Append(value);
                }
            }
            else
            {
                sb.Append('[');
                for (int i = 0; i < dimensions[dimIdx]; i++)
                {
                    DumpArray(arr, sb, dimensions, strides, dimIdx + 1,  offset + strides[dimIdx] * i);
                    if (i < dimensions[dimIdx] - 1)
                    {
                        sb.Append(", ");
                    }
                }
                sb.Append("]\n");
            }
        }

    }
}
