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
using System.Text;
using System.Threading.Tasks;
using size_t = System.UInt64;


namespace NumpyLib
{
    internal partial class numpyinternal
    {
        internal static int npy_array_getsegcount(NpyArray self, ref size_t lenp)
        {
            lenp = (size_t)NpyArray_NBYTES(self);
           
            if (NpyArray_ISONESEGMENT(self))
            {
                return 1;
            }
            lenp = 0;
            return 0;
        }

        internal static int npy_array_getreadbuf(NpyArray self, size_t segment, ref VoidPtr ptrptr)
        {
            if (segment != 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "accessing non-existing array segment");
                return -1;
            }
            if (NpyArray_ISONESEGMENT(self))
            {
                ptrptr = NpyArray_BYTES(self);
                return (int)NpyArray_NBYTES(self);
            }
            NpyErr_SetString(npyexc_type.NpyExc_ValueError, "array is not a single segment");
            ptrptr = null;
            return -1;
        }

        internal static int npy_array_getwritebuf(NpyArray self, size_t segment, ref VoidPtr ptrptr)
        {
            if (NpyArray_CHKFLAGS(self, NPYARRAYFLAGS.NPY_WRITEABLE))
            {
                return npy_array_getreadbuf(self, segment, ref ptrptr);
            }
            else
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "array cannot be accessed as a writeable buffer");
                return -1;
            }
        }


        internal static int npy_append_char(npy_tmp_string_t s, char c)
        {
            if (s.s == null)
            {
                s.s = c.ToString();
                s.pos = 0;
                s.allocated = s.s.Length;
            }
            else
            {
                s.s += c.ToString();
                s.pos = 0;
                s.allocated = s.s.Length;
            }
  
            return 0;
        }

        internal static int npy_append_str(npy_tmp_string_t s, string c)
        {
            if (s.s == null)
            {
                s.s = c.ToString();
                s.pos = 0;
                s.allocated = s.s.Length;
            }
            else
            {
                s.s += c.ToString();
                s.pos = 0;
                s.allocated = s.s.Length;
            }
            return 0;
        }

        /*
         * Global information about all active buffers
         *
         * Note: because for backward compatibility we cannot define bf_releasebuffer,
         * we must manually keep track of the additional data required by the buffers.
         */



        /* Fill in the info structure */
        internal static npy_buffer_info_t npy_buffer_info_new(NpyArray arr)
        {
            npy_buffer_info_t info;
            npy_tmp_string_t fmt = new npy_tmp_string_t()  {allocated = 0, pos = 0, s = null };
            int k;

            info = new npy_buffer_info_t();

            /* Fill in format */
            size_t offset =0;
            char active_byteorder = '0';
            if (npy_buffer_format_string(NpyArray_DESCR(arr), fmt, arr, ref offset, ref active_byteorder) != 0)
            {
                return null;
            }
            npy_append_char(fmt, '\0');
            info.format = fmt.s;

            /* Fill in shape and strides */
            info.ndim = NpyArray_NDIM(arr);

            if (info.ndim == 0)
            {
                info.shape = null;
                info.strides = null;
            }
            else
            {
                info.shape = new size_t[NpyArray_NDIM(arr)];
                info.strides = new size_t[NpyArray_NDIM(arr)];
                for (k = 0; k < NpyArray_NDIM(arr); ++k)
                {
                    info.shape[k] = (ulong)NpyArray_DIM(arr, k);
                    info.strides[k] = (ulong)NpyArray_STRIDE(arr, k);
                }
            }

            return info;
        }

        internal static int npy_buffer_format_string(NpyArray_Descr descr, npy_tmp_string_t str, NpyArray arr, ref size_t offset, ref char active_byteorder)
        {
            // todo:
            return 0;
        }


        /* Compare two info structures */
        internal static size_t npy_buffer_info_cmp(npy_buffer_info_t a, npy_buffer_info_t b)
        {
            size_t c;

            c = (size_t)a.format.CompareTo(b.format);
            if (c != 0) return c;

            c = (size_t)(a.ndim - b.ndim);
            if (c != 0) return c;

            for (int k = 0; k < a.ndim; ++k)
            {
                c = a.shape[k] - b.shape[k];
                if (c != 0) return c;
                c = a.strides[k] - b.strides[k];
                if (c != 0) return c;
            }

            return 0;
        }

        internal static void npy_buffer_info_free(npy_buffer_info_t info)
        {
            if (info.format != null)
            {
                npy_free(info.format);
            }
            if (info.shape != null)
            {
                npy_free(info.shape);
            }
            npy_free(info);
        }

    }
}
