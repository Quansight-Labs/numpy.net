/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018-2021
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

#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyLib
{
    internal partial class numpyinternal
    {
        internal static bool Npy_IsAligned(NpyArray ap)
        {
            return true;
        }

        internal static bool Npy_IsWriteable(NpyArray ap)
        {
            NpyArray base_arr = ap.base_arr;
            object base_obj = ap.base_obj;

            /* If we own our own data, then no-problem */
            if ((base_arr == null && null == base_obj) || ((ap.flags & NPYARRAYFLAGS.NPY_OWNDATA) > 0))
            {
                return true;
            }
            /*
             * Get to the final base object
             * If it is a writeable array, then return TRUE
             * If we can find an array object
             * or a writeable buffer object as the final base object
             * or a string object (for pickling support memory savings).
             * - this last could be removed if a proper pickleable
             * buffer was added to Python.
             */

            while (null != base_arr)
            {
                if (NpyArray_CHKFLAGS(base_arr, NPYARRAYFLAGS.NPY_OWNDATA))
                {
                    return NpyArray_ISWRITEABLE(base_arr);
                }
                base_arr = base_arr.base_arr;
                base_obj = base_arr.base_obj;
            }

            /*
             * here so pickle support works seamlessly
             * and unpickled array can be set and reset writeable
             * -- could be abused --
             */

            return true;
        }

        internal static VoidPtr NpyArray_Index2Ptr(NpyArray mp, npy_intp i)
        {
            npy_intp dim0;

            if (NpyArray_NDIM(mp) == 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_IndexError, "0-d arrays can't be indexed");
                return null;
            }
            dim0 = NpyArray_DIM(mp, 0);
            if (i < 0)
            {
                i += dim0;
            }
            if (i == 0 && dim0 > 0)
            {
                return NpyArray_BYTES(mp);
            }
            if (i > 0 && i < dim0)
            {
                return NpyArray_BYTES(mp) + i * NpyArray_STRIDE(mp, 0);
            }
            NpyErr_SetString(npyexc_type.NpyExc_IndexError, "index out of bounds");
            return null;
        }


    }
}
