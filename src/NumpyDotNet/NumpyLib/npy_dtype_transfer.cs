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
         * Prepares shape and strides for a simple raw array iteration.
         * This sorts the strides into FORTRAN order, reverses any negative
         * strides, then coalesces axes where possible. The results are
         * filled in the output parameters.
         *
         * This is intended for simple, lightweight iteration over arrays
         * where no buffering of any kind is needed, and the array may
         * not be stored as a PyArrayObject.
         *
         * The arrays shape, out_shape, strides, and out_strides must all
         * point to different data.
         *
         * Returns 0 on success, -1 on failure.
         */
        internal static int PyArray_PrepareOneRawArrayIter(int ndim, npy_intp[] shape,
                                    VoidPtr data, npy_intp[] strides,
                                    ref int out_ndim, npy_intp[] out_shape,
                                    ref VoidPtr out_data, npy_intp[] out_strides)
        {
            npy_stride_sort_item []strideperm = new npy_stride_sort_item[npy_defs.NPY_MAXDIMS];
            int i, j;

            /* Special case 0 and 1 dimensions */
            if (ndim == 0)
            {
                out_ndim = 1;
                out_data = data;
                out_shape[0] = 1;
                out_strides[0] = 0;
                return 0;
            }
            else if (ndim == 1)
            {
                npy_intp stride_entry = strides[0], shape_entry = shape[0];
                out_ndim = 1;
                out_shape[0] = shape[0];
                /* Always make a positive stride */
                if (stride_entry >= 0)
                {
                    out_data = data;
                    out_strides[0] = stride_entry;
                }
                else
                {
                    out_data = data + stride_entry * (shape_entry - 1);
                    out_strides[0] = -stride_entry;
                }
                return 0;
            }

            /* Sort the axes based on the destination strides */
            PyArray_CreateSortedStridePerm(ndim, strides, strideperm);
            for (i = 0; i < ndim; ++i)
            {
                npy_intp iperm = strideperm[ndim - i - 1].perm;
                out_shape[i] = shape[iperm];
                out_strides[i] = strides[iperm];
            }

            /* Reverse any negative strides */
            for (i = 0; i < ndim; ++i)
            {
                npy_intp stride_entry = out_strides[i], shape_entry = out_shape[i];

                if (stride_entry < 0)
                {
                    data += stride_entry * (shape_entry - 1);
                    out_strides[i] = -stride_entry;
                }
                /* Detect 0-size arrays here */
                if (shape_entry == 0)
                {
                    out_ndim = 1;
                    out_data = data;
                    out_shape[0] = 0;
                    out_strides[0] = 0;
                    return 0;
                }
            }

            /* Coalesce any dimensions where possible */
            i = 0;
            for (j = 1; j < ndim; ++j)
            {
                if (out_shape[i] == 1)
                {
                    /* Drop axis i */
                    out_shape[i] = out_shape[j];
                    out_strides[i] = out_strides[j];
                }
                else if (out_shape[j] == 1)
                {
                    /* Drop axis j */
                }
                else if (out_strides[i] * out_shape[i] == out_strides[j])
                {
                    /* Coalesce axes i and j */
                    out_shape[i] *= out_shape[j];
                }
                else
                {
                    /* Can't coalesce, go to next i */
                    ++i;
                    out_shape[i] = out_shape[j];
                    out_strides[i] = out_strides[j];
                }
            }
            ndim = i + 1;

            #if false
            /* DEBUG */
            {
                printf("raw iter ndim %d\n", ndim);
                printf("shape: ");
                for (i = 0; i < ndim; ++i) {
                    printf("%d ", (int)out_shape[i]);
                }
                printf("\n");
                printf("strides: ");
                for (i = 0; i < ndim; ++i) {
                    printf("%d ", (int)out_strides[i]);
                }
                printf("\n");
            }
            #endif

            out_data = data;
            out_ndim = ndim;
            return 0;
        }

        /*
         * The same as PyArray_PrepareOneRawArrayIter, but for two
         * operands instead of one. Any broadcasting of the two operands
         * should have already been done before calling this function,
         * as the ndim and shape is only specified once for both operands.
         *
         * Only the strides of the first operand are used to reorder
         * the dimensions, no attempt to consider all the strides together
         * is made, as is done in the NpyIter object.
         *
         * You can use this together with NPY_RAW_ITER_START and
         * NPY_RAW_ITER_TWO_NEXT to handle the looping boilerplate of everything
         * but the innermost loop (which is for idim == 0).
         *
         * Returns 0 on success, -1 on failure.
         */
        internal static int PyArray_PrepareTwoRawArrayIter(int ndim, npy_intp[] shape,
                                    VoidPtr dataA, npy_intp[] stridesA,
                                    VoidPtr dataB, npy_intp[] stridesB,
                                    ref int out_ndim, npy_intp[] out_shape,
                                    ref VoidPtr out_dataA, npy_intp[] out_stridesA,
                                    ref VoidPtr out_dataB, npy_intp[] out_stridesB)
        {
            npy_stride_sort_item []strideperm = new npy_stride_sort_item[npy_defs.NPY_MAXDIMS];
            int i, j;

            /* Special case 0 and 1 dimensions */
            if (ndim == 0)
            {
                out_ndim = 1;
                out_dataA = dataA;
                out_dataB = dataB;
                out_shape[0] = 1;
                out_stridesA[0] = 0;
                out_stridesB[0] = 0;
                return 0;
            }
            else if (ndim == 1)
            {
                npy_intp stride_entryA = stridesA[0], stride_entryB = stridesB[0];
                npy_intp shape_entry = shape[0];
                out_ndim = 1;
                out_shape[0] = shape[0];
                /* Always make a positive stride for the first operand */
                if (stride_entryA >= 0)
                {
                    out_dataA = dataA;
                    out_dataB = dataB;
                    out_stridesA[0] = stride_entryA;
                    out_stridesB[0] = stride_entryB;
                }
                else
                {
                    out_dataA = dataA + stride_entryA * (shape_entry - 1);
                    out_dataB = dataB + stride_entryB * (shape_entry - 1);
                    out_stridesA[0] = -stride_entryA;
                    out_stridesB[0] = -stride_entryB;
                }
                return 0;
            }

            /* Sort the axes based on the destination strides */
            PyArray_CreateSortedStridePerm(ndim, stridesA, strideperm);
            for (i = 0; i < ndim; ++i)
            {
                npy_intp iperm = strideperm[ndim - i - 1].perm;
                out_shape[i] = shape[iperm];
                out_stridesA[i] = stridesA[iperm];
                out_stridesB[i] = stridesB[iperm];
            }

            /* Reverse any negative strides of operand A */
            for (i = 0; i < ndim; ++i)
            {
                npy_intp stride_entryA = out_stridesA[i];
                npy_intp stride_entryB = out_stridesB[i];
                npy_intp shape_entry = out_shape[i];

                if (stride_entryA < 0)
                {
                    dataA += stride_entryA * (shape_entry - 1);
                    dataB += stride_entryB * (shape_entry - 1);
                    out_stridesA[i] = -stride_entryA;
                    out_stridesB[i] = -stride_entryB;
                }
                /* Detect 0-size arrays here */
                if (shape_entry == 0)
                {
                    out_ndim = 1;
                    out_dataA = dataA;
                    out_dataB = dataB;
                    out_shape[0] = 0;
                    out_stridesA[0] = 0;
                    out_stridesB[0] = 0;
                    return 0;
                }
            }

            /* Coalesce any dimensions where possible */
            i = 0;
            for (j = 1; j < ndim; ++j)
            {
                if (out_shape[i] == 1)
                {
                    /* Drop axis i */
                    out_shape[i] = out_shape[j];
                    out_stridesA[i] = out_stridesA[j];
                    out_stridesB[i] = out_stridesB[j];
                }
                else if (out_shape[j] == 1)
                {
                    /* Drop axis j */
                }
                else if (out_stridesA[i] * out_shape[i] == out_stridesA[j] &&
                            out_stridesB[i] * out_shape[i] == out_stridesB[j])
                {
                    /* Coalesce axes i and j */
                    out_shape[i] *= out_shape[j];
                }
                else
                {
                    /* Can't coalesce, go to next i */
                    ++i;
                    out_shape[i] = out_shape[j];
                    out_stridesA[i] = out_stridesA[j];
                    out_stridesB[i] = out_stridesB[j];
                }
            }
            ndim = i + 1;

            out_dataA = dataA;
            out_dataB = dataB;
            out_ndim = ndim;
            return 0;
        }

        /*
         * The same as PyArray_PrepareOneRawArrayIter, but for three
         * operands instead of one. Any broadcasting of the three operands
         * should have already been done before calling this function,
         * as the ndim and shape is only specified once for all operands.
         *
         * Only the strides of the first operand are used to reorder
         * the dimensions, no attempt to consider all the strides together
         * is made, as is done in the NpyIter object.
         *
         * You can use this together with NPY_RAW_ITER_START and
         * NPY_RAW_ITER_THREE_NEXT to handle the looping boilerplate of everything
         * but the innermost loop (which is for idim == 0).
         *
         * Returns 0 on success, -1 on failure.
         */
        internal static int PyArray_PrepareThreeRawArrayIter(int ndim, npy_intp[] shape,
                                    VoidPtr dataA, npy_intp[] stridesA,
                                    VoidPtr dataB, npy_intp[] stridesB,
                                    VoidPtr dataC, npy_intp[] stridesC,
                                    ref int out_ndim, npy_intp[] out_shape,
                                    ref VoidPtr out_dataA, npy_intp[] out_stridesA,
                                    ref VoidPtr out_dataB, npy_intp[] out_stridesB,
                                    ref VoidPtr out_dataC, npy_intp[] out_stridesC)
        {
            npy_stride_sort_item []strideperm = new npy_stride_sort_item[npy_defs.NPY_MAXDIMS];
            int i, j;

            /* Special case 0 and 1 dimensions */
            if (ndim == 0)
            {
                out_ndim = 1;
                out_dataA = dataA;
                out_dataB = dataB;
                out_dataC = dataC;
                out_shape[0] = 1;
                out_stridesA[0] = 0;
                out_stridesB[0] = 0;
                out_stridesC[0] = 0;
                return 0;
            }
            else if (ndim == 1)
            {
                npy_intp stride_entryA = stridesA[0];
                npy_intp stride_entryB = stridesB[0];
                npy_intp stride_entryC = stridesC[0];
                npy_intp shape_entry = shape[0];
                out_ndim = 1;
                out_shape[0] = shape[0];
                /* Always make a positive stride for the first operand */
                if (stride_entryA >= 0)
                {
                    out_dataA = dataA;
                    out_dataB = dataB;
                    out_dataC = dataC;
                    out_stridesA[0] = stride_entryA;
                    out_stridesB[0] = stride_entryB;
                    out_stridesC[0] = stride_entryC;
                }
                else
                {
                    out_dataA = dataA + stride_entryA * (shape_entry - 1);
                    out_dataB = dataB + stride_entryB * (shape_entry - 1);
                    out_dataC = dataC + stride_entryC * (shape_entry - 1);
                    out_stridesA[0] = -stride_entryA;
                    out_stridesB[0] = -stride_entryB;
                    out_stridesC[0] = -stride_entryC;
                }
                return 0;
            }

            /* Sort the axes based on the destination strides */
            PyArray_CreateSortedStridePerm(ndim, stridesA, strideperm);
            for (i = 0; i < ndim; ++i)
            {
                npy_intp iperm = strideperm[ndim - i - 1].perm;
                out_shape[i] = shape[iperm];
                out_stridesA[i] = stridesA[iperm];
                out_stridesB[i] = stridesB[iperm];
                out_stridesC[i] = stridesC[iperm];
            }

            /* Reverse any negative strides of operand A */
            for (i = 0; i < ndim; ++i)
            {
                npy_intp stride_entryA = out_stridesA[i];
                npy_intp stride_entryB = out_stridesB[i];
                npy_intp stride_entryC = out_stridesC[i];
                npy_intp shape_entry = out_shape[i];

                if (stride_entryA < 0)
                {
                    dataA += stride_entryA * (shape_entry - 1);
                    dataB += stride_entryB * (shape_entry - 1);
                    dataC += stride_entryC * (shape_entry - 1);
                    out_stridesA[i] = -stride_entryA;
                    out_stridesB[i] = -stride_entryB;
                    out_stridesC[i] = -stride_entryC;
                }
                /* Detect 0-size arrays here */
                if (shape_entry == 0)
                {
                    out_ndim = 1;
                    out_dataA = dataA;
                    out_dataB = dataB;
                    out_dataC = dataC;
                    out_shape[0] = 0;
                    out_stridesA[0] = 0;
                    out_stridesB[0] = 0;
                    out_stridesC[0] = 0;
                    return 0;
                }
            }

            /* Coalesce any dimensions where possible */
            i = 0;
            for (j = 1; j < ndim; ++j)
            {
                if (out_shape[i] == 1)
                {
                    /* Drop axis i */
                    out_shape[i] = out_shape[j];
                    out_stridesA[i] = out_stridesA[j];
                    out_stridesB[i] = out_stridesB[j];
                    out_stridesC[i] = out_stridesC[j];
                }
                else if (out_shape[j] == 1)
                {
                    /* Drop axis j */
                }
                else if (out_stridesA[i] * out_shape[i] == out_stridesA[j] &&
                            out_stridesB[i] * out_shape[i] == out_stridesB[j] &&
                            out_stridesC[i] * out_shape[i] == out_stridesC[j])
                {
                    /* Coalesce axes i and j */
                    out_shape[i] *= out_shape[j];
                }
                else
                {
                    /* Can't coalesce, go to next i */
                    ++i;
                    out_shape[i] = out_shape[j];
                    out_stridesA[i] = out_stridesA[j];
                    out_stridesB[i] = out_stridesB[j];
                    out_stridesC[i] = out_stridesC[j];
                }
            }
            ndim = i + 1;

            out_dataA = dataA;
            out_dataB = dataB;
            out_dataC = dataC;
            out_ndim = ndim;
            return 0;
        }

        private static int PyArray_GetDTypeTransferFunction(bool aligned,
                            npy_intp src_stride, npy_intp dst_stride,
                            NpyArray_Descr src_dtype, NpyArray_Descr dst_dtype,
                            int move_references,
                            ref PyArray_StridedUnaryOp out_stransfer,
                            ref NpyAuxData out_transferdata,
                            ref bool out_needs_api)
        {
            return npy_defs.NPY_SUCCEED;
        }

        private static int PyArray_GetMaskedDTypeTransferFunction(bool aligned,
                            npy_intp src_stride,
                            npy_intp dst_stride,
                            npy_intp mask_stride,
                            NpyArray_Descr src_dtype,
                            NpyArray_Descr dst_dtype,
                            NpyArray_Descr mask_dtype,
                            int move_references,
                            ref PyArray_MaskedStridedUnaryOp out_stransfer,
                            ref NpyAuxData out_transferdata,
                            ref bool out_needs_api)
        {
            return npy_defs.NPY_SUCCEED;
        }

    }
}
