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
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif
using size_t = System.UInt64;


namespace NumpyLib
{
    internal partial class numpyinternal
    {
        /* See array_assign.h for parameter documentation */
        private static int broadcast_strides(int ndim, npy_intp[] shape,
                        int strides_ndim, npy_intp[] strides_shape, npy_intp[] strides,
                        string strides_name,
                        npy_intp[] out_strides)
        {
            int idim, idim_start = ndim - strides_ndim;

            /* Can't broadcast to fewer dimensions */
            if (idim_start < 0)
            {
                goto broadcast_error;
            }

            /*
             * Process from the end to the start, so that 'strides' and 'out_strides'
             * can point to the same memory.
             */
            for (idim = ndim - 1; idim >= idim_start; --idim)
            {
                npy_intp strides_shape_value = strides_shape[idim - idim_start];
                /* If it doesn't have dimension one, it must match */
                if (strides_shape_value == 1)
                {
                    out_strides[idim] = 0;
                }
                else if (strides_shape_value != shape[idim])
                {
                    goto broadcast_error;
                }
                else
                {
                    out_strides[idim] = strides[idim - idim_start];
                }
            }

            /* New dimensions get a zero stride */
            for (idim = 0; idim < idim_start; ++idim)
            {
                out_strides[idim] = 0;
            }

            return 0;

            broadcast_error:

            {
                string errmsg = string.Format("could not broadcast %s from shape {0} {1} into shape {2}", 
                        strides_name, build_shape_string(strides_ndim, strides_shape), build_shape_string(ndim, shape));


                NpyErr_SetString(npyexc_type.NpyExc_ValueError, errmsg);

                return -1;
            }
        }

        /* See array_assign.h for parameter documentation */
        private static bool raw_array_is_aligned(int ndim, VoidPtr data, npy_intp[] strides, int alignment)
        {
            return true;

            //if (alignment > 1)
            //{
            //    npy_intp []align_check = data.datap as npy_intp[];
            //    int idim;

            //    for (idim = 0; idim < ndim; ++idim)
            //    {
            //        align_check[0] |= strides[idim];
            //    }

            //    return true; // npy_is_aligned(align_check[0], alignment);
            //}
            //else
            //{
            //    return true;
            //}
        }


        /* Returns 1 if the arrays have overlapping data, 0 otherwise */
        private static bool arrays_overlap(NpyArray arr1, NpyArray arr2)
        {
            return (Object.ReferenceEquals(arr1.data.datap, arr2.data.datap));
        }
        /* Returns 1 if the arrays have overlapping data, 0 otherwise */
        private static bool arrays_overlap(VoidPtr arr1, VoidPtr arr2)
        {
            return (Object.ReferenceEquals(arr1.datap, arr2.datap));
        }

    }
}
