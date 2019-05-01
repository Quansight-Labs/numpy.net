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

        /*
         * This function pointer is for unary operations that input an
         * arbitrarily strided one-dimensional array segment and output
         * an arbitrarily strided array segment of the same size.
         * It may be a fully general function, or a specialized function
         * when the strides or item size have particular known values.
         *
         * Examples of unary operations are a straight copy, a byte-swap,
         * and a casting operation,
         *
         * The 'transferdata' parameter is slightly special, following a
         * generic auxiliary data pattern defined in ndarraytypes.h
         * Use NPY_AUXDATA_CLONE and NPY_AUXDATA_FREE to deal with this data.
         *
         */

         delegate void PyArray_StridedUnaryOp(VoidPtr dst, npy_intp dst_stride,
                                             VoidPtr src, npy_intp src_stride,
                                             npy_intp N, npy_intp src_itemsize,
                                             NpyAuxData transferdata);

        /*
         * This is for pointers to functions which behave exactly as
         * for PyArray_StridedUnaryOp, but with an additional mask controlling
         * which values are transformed.
         *
         * In particular, the 'i'-th element is operated on if and only if
         * mask[i*mask_stride] is true.
         */
        delegate void PyArray_MaskedStridedUnaryOp(VoidPtr dst, npy_intp dst_stride,
                                             VoidPtr src, npy_intp src_stride,
                                             bool[] mask, npy_intp mask_stride,
                                             npy_intp N, npy_intp src_itemsize,
                                             NpyAuxData transferdata);

        private static void NPY_RAW_ITER_START(int idim, int ndim, long[] coord, long[] shape_it)
        {
            memset(new VoidPtr(coord), 0, ndim * sizeof(long));
        }


        private static bool NPY_RAW_ITER_TWO_NEXT(int idim, int ndim, long[] coord, long[] shape, VoidPtr dataA, long[] stridesA, VoidPtr dataB, long[] stridesB)
        {
            for (idim = 1; idim < ndim; ++idim)
            {
                if (++coord[idim] == shape[idim])
                {
                    coord[idim] = 0;
                    dataA.data_offset -= (shape[idim] - 1) * stridesA[idim];
                    dataB.data_offset -= (shape[idim] - 1) * stridesB[idim];
                }
                else
                {
                    dataA.data_offset += stridesA[idim];
                    dataB.data_offset += stridesB[idim];
                    break;
                }
            }
            return (idim < ndim);
        }


        private static bool NPY_RAW_ITER_THREE_NEXT(int idim, int ndim, long[] coord, long[] shape, 
                                                VoidPtr dataA, long[] stridesA, 
                                                VoidPtr dataB, long[] stridesB, 
                                                VoidPtr dataC, long[] stridesC)
        {
            for (idim = 1; idim < ndim; ++idim)
            { 
                if (++coord[idim] == shape[idim])
                { 
                    coord[idim] = 0; 
                    dataA.data_offset -= (shape[idim] - 1) * stridesA[idim]; 
                    dataB.data_offset -= (shape[idim] - 1) * stridesB[idim]; 
                    dataC.data_offset -= (shape[idim] - 1) * stridesC[idim]; 
                } 
                else
                { 
                    dataA.data_offset += (stridesA)[idim]; 
                    dataB.data_offset += (stridesB)[idim]; 
                    dataC.data_offset += (stridesC)[idim]; 
                    break; 
                } 
            } 
            return (idim < ndim);
        }



        static void _strided_to_strided(VoidPtr dst, npy_intp dst_stride,
                                VoidPtr src, npy_intp src_stride,
                                npy_intp N, npy_intp src_itemsize,
                                NpyAuxData data)
        {
            while (N > 0)
            {
                memmove(dst, src, src_itemsize);
                dst += dst_stride;
                src += src_stride;
                --N;
            }
        }


    }
}
