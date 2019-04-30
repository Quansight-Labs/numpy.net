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

namespace NumpyLib
{
    internal partial class numpyinternal
    {

        /*
         * Assigns the array from 'src' to 'dst'. The strides must already have
         * been broadcast.
         *
         * Returns 0 on success, -1 on failure.
         */
        private static int raw_array_assign_array(int ndim, npy_intp[] shape,
                NpyArray_Descr dst_dtype, VoidPtr dst_data, npy_intp[] dst_strides,
                NpyArray_Descr src_dtype, VoidPtr src_data, npy_intp[] src_strides)
        {
            int idim;
            npy_intp [] shape_it = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp [] dst_strides_it = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp []src_strides_it = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp []coord = new npy_intp[npy_defs.NPY_MAXDIMS];

            PyArray_StridedUnaryOp stransfer = null;
            NpyAuxData transferdata = null;
            bool aligned, needs_api = false;
            npy_intp src_itemsize = src_dtype.elsize;


            /* Check alignment */
            aligned = raw_array_is_aligned(ndim,
                                dst_data, dst_strides, dst_dtype.alignment) &&
                      raw_array_is_aligned(ndim,
                                src_data, src_strides, src_dtype.alignment);

            /* Use raw iteration with no heap allocation */
            if (PyArray_PrepareTwoRawArrayIter(
                            ndim, shape,
                            dst_data, dst_strides,
                            src_data, src_strides,
                            ref ndim, shape_it,
                            ref dst_data, dst_strides_it,
                            ref src_data, src_strides_it) < 0)
            {
                return -1;
            }

            /*
             * Overlap check for the 1D case. Higher dimensional arrays and
             * opposite strides cause a temporary copy before getting here.
             */
            if (ndim == 1 && arrays_overlap(src_data, dst_data))
            {
                src_data += (shape_it[0] - 1) * src_strides_it[0];
                dst_data += (shape_it[0] - 1) * dst_strides_it[0];
                src_strides_it[0] = -src_strides_it[0];
                dst_strides_it[0] = -dst_strides_it[0];
            }

            /* Get the function to do the casting */
            if (PyArray_GetDTypeTransferFunction(aligned,
                                src_strides_it[0], dst_strides_it[0],
                                src_dtype, dst_dtype,
                                0,
                                &stransfer, &transferdata,
                                &needs_api) != NPY_SUCCEED)
            {
                return -1;
            }

            if (!needs_api)
            {
                NPY_BEGIN_THREADS;
            }

            NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
                /* Process the innermost dimension */
                stransfer(dst_data, dst_strides_it[0], src_data, src_strides_it[0],
                            shape_it[0], src_itemsize, transferdata);
            }
            NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                                  dst_data, dst_strides_it,
                                  src_data, src_strides_it);

            NPY_END_THREADS;

            NPY_AUXDATA_FREE(transferdata);

            return (needs_api && NpyErr_Occurred()) ? -1 : 0;
        }

        /*
         * Assigns the array from 'src' to 'dst, wherever the 'wheremask'
         * value is True. The strides must already have been broadcast.
         *
         * Returns 0 on success, -1 on failure.
         */
        private static int raw_array_wheremasked_assign_array(int ndim, npy_intp[] shape,
                NpyArray_Descr dst_dtype, VoidPtr dst_data, npy_intp[] dst_strides,
                NpyArray_Descr src_dtype, VoidPtr src_data, npy_intp[] src_strides,
                NpyArray_Descr wheremask_dtype, VoidPtr wheremask_data,
                npy_intp[] wheremask_strides)
        {
            int idim;
            npy_intp []shape_it = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp []dst_strides_it = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp []src_strides_it = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp []wheremask_strides_it = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp []coord = new npy_intp[npy_defs.NPY_MAXDIMS];

            PyArray_MaskedStridedUnaryOp* stransfer = null;
            NpyAuxData* transferdata = null;
            bool aligned, needs_api = false;
            npy_intp src_itemsize = src_dtype.elsize;

            NPY_BEGIN_THREADS_DEF;

            /* Check alignment */
            aligned = raw_array_is_aligned(ndim,
                                dst_data, dst_strides, dst_dtype.alignment) &&
                      raw_array_is_aligned(ndim,
                                src_data, src_strides, src_dtype.alignment);

            /* Use raw iteration with no heap allocation */
            if (PyArray_PrepareThreeRawArrayIter(
                            ndim, shape,
                            dst_data, dst_strides,
                            src_data, src_strides,
                            wheremask_data, wheremask_strides,
                            ref ndim, shape_it,
                            ref dst_data, dst_strides_it,
                            ref src_data, src_strides_it,
                            ref wheremask_data, wheremask_strides_it) < 0)
            {
                return -1;
            }

            /*
             * Overlap check for the 1D case. Higher dimensional arrays cause
             * a temporary copy before getting here.
             */
            if (ndim == 1 && arrays_overlap(src_data, dst_data))
            {
                src_data += (shape_it[0] - 1) * src_strides_it[0];
                dst_data += (shape_it[0] - 1) * dst_strides_it[0];
                wheremask_data += (shape_it[0] - 1) * wheremask_strides_it[0];
                src_strides_it[0] = -src_strides_it[0];
                dst_strides_it[0] = -dst_strides_it[0];
                wheremask_strides_it[0] = -wheremask_strides_it[0];
            }

            /* Get the function to do the casting */
            if (PyArray_GetMaskedDTypeTransferFunction(aligned,
                                src_strides_it[0],
                                dst_strides_it[0],
                                wheremask_strides_it[0],
                                src_dtype, dst_dtype, wheremask_dtype,
                                0,
                                &stransfer, &transferdata,
                                &needs_api) != NPY_SUCCEED)
            {
                return -1;
            }

            if (!needs_api)
            {
                NPY_BEGIN_THREADS;
            }

            NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
                /* Process the innermost dimension */
                stransfer(dst_data, dst_strides_it[0], src_data, src_strides_it[0],
                            (npy_bool*)wheremask_data, wheremask_strides_it[0],
                            shape_it[0], src_itemsize, transferdata);
            }
            NPY_RAW_ITER_THREE_NEXT(idim, ndim, coord, shape_it,
                                  dst_data, dst_strides_it,
                                  src_data, src_strides_it,
                                  wheremask_data, wheremask_strides_it);

            NPY_END_THREADS;

            NPY_AUXDATA_FREE(transferdata);

            return (needs_api && NpyErr_Occurred()) ? -1 : 0;
        }

        /*
         * An array assignment function for copying arrays, broadcasting 'src' into
         * 'dst'. This function makes a temporary copy of 'src' if 'src' and
         * 'dst' overlap, to be able to handle views of the same data with
         * different strides.
         *
         * dst: The destination array.
         * src: The source array.
         * wheremask: If non-null, a boolean mask specifying where to copy.
         * casting: An exception is raised if the copy violates this
         *          casting rule.
         *
         * Returns 0 on success, -1 on failure.
         */

        private static int NpyArray_AssignArray(NpyArray dst, NpyArray src, NpyArray wheremask, NPY_CASTING casting)
        {
            bool copied_src = false;

            npy_intp []src_strides = new npy_intp[npy_defs.NPY_MAXDIMS];

            /* Use array_assign_scalar if 'src' NDIM is 0 */
            if (NpyArray_NDIM(src) == 0)
            {
                return NpyArray_AssignRawScalar(dst, NpyArray_DESCR(src), NpyArray_DATA(src), wheremask, casting);
            }

            /*
             * Performance fix for expressions like "a[1000:6000] += x".  In this
             * case, first an in-place add is done, followed by an assignment,
             * equivalently expressed like this:
             *
             *   tmp = a[1000:6000]   # Calls array_subscript in mapping.c
             *   np.add(tmp, x, tmp)
             *   a[1000:6000] = tmp   # Calls array_assign_subscript in mapping.c
             *
             * In the assignment the underlying data type, shape, strides, and
             * data pointers are identical, but src != dst because they are separately
             * generated slices.  By detecting this and skipping the redundant
             * copy of values to themselves, we potentially give a big speed boost.
             *
             * Note that we don't call EquivTypes, because usually the exact same
             * dtype object will appear, and we don't want to slow things down
             * with a complicated comparison.  The comparisons are ordered to
             * try and reject this with as little work as possible.
             */
            if (NpyArray_DATA(src) == NpyArray_DATA(dst) &&
                                NpyArray_DESCR(src) == NpyArray_DESCR(dst) &&
                                NpyArray_NDIM(src) == NpyArray_NDIM(dst) &&
                                NpyArray_CompareLists(NpyArray_DIMS(src),
                                                     NpyArray_DIMS(dst),
                                                     NpyArray_NDIM(src)) &&
                                NpyArray_CompareLists(NpyArray_STRIDES(src),
                                                     NpyArray_STRIDES(dst),
                                                     NpyArray_NDIM(src)))
            {
                /*printf("Redundant copy operation detected\n");*/
                return 0;
            }

            if (NpyArray_FailUnlessWriteable(dst, "assignment destination") < 0)
            {
                goto fail;
            }

            /* Check the casting rule */

            if (!NpyArray_CanCastTypeTo(NpyArray_DESCR(src),
                                        NpyArray_DESCR(dst), casting))
            {
                string errmsg = string.Format("Cannot cast scalar from {0} to {1} according to the rule {2}", src.GetType(), dst.GetType(), npy_casting_to_string(casting)); 
    
                goto fail;
            }

            /*
             * When ndim is 1 and the strides point in the same direction,
             * the lower-level inner loop handles copying
             * of overlapping data. For bigger ndim and opposite-strided 1D
             * data, we make a temporary copy of 'src' if 'src' and 'dst' overlap.'
             */
            if (((NpyArray_NDIM(dst) == 1 && NpyArray_NDIM(src) >= 1 &&
                            NpyArray_STRIDES(dst)[0] *
                                    NpyArray_STRIDES(src)[NpyArray_NDIM(src) - 1] < 0) ||
                            NpyArray_NDIM(dst) > 1 || NpyArray_HASFIELDS(dst)) &&
                            arrays_overlap(src, dst))
            {
                NpyArray tmp;

                /*
                 * Allocate a temporary copy array.
                 */
                tmp = NpyArray_NewLikeArray(dst, NPY_KEEPORDER, null, 0);
                if (tmp == null)
                {
                    goto fail;
                }

                if (NpyArray_AssignArray(tmp, src, null, NPY_CASTING.NPY_UNSAFE_CASTING) < 0)
                {
                    Npy_DECREF(tmp);
                    goto fail;
                }

                src = tmp;
                copied_src = true;
            }

            /* Broadcast 'src' to 'dst' for raw iteration */
            if (NpyArray_NDIM(src) > NpyArray_NDIM(dst))
            {
                int ndim_tmp = NpyArray_NDIM(src);
                npy_intp []src_shape_tmp = NpyArray_DIMS(src);
                npy_intp []src_strides_tmp = NpyArray_STRIDES(src);
                /*
                 * As a special case for backwards compatibility, strip
                 * away unit dimensions from the left of 'src'
                 */
                while (ndim_tmp > NpyArray_NDIM(dst) && src_shape_tmp[0] == 1)
                {
                    --ndim_tmp;
                    ++src_shape_tmp;
                    ++src_strides_tmp;
                }

                if (broadcast_strides(NpyArray_NDIM(dst), NpyArray_DIMS(dst),
                            ndim_tmp, src_shape_tmp,
                            src_strides_tmp, "input array",
                            src_strides) < 0)
                {
                    goto fail;
                }
            }
            else
            {
                if (broadcast_strides(NpyArray_NDIM(dst), NpyArray_DIMS(dst),
                            NpyArray_NDIM(src), NpyArray_DIMS(src),
                            NpyArray_STRIDES(src), "input array",
                            src_strides) < 0)
                {
                    goto fail;
                }
            }

            /* optimization: scalar boolean mask */
            if (wheremask != null &&
                    NpyArray_NDIM(wheremask) == 0 &&
                    NpyArray_DESCR(wheremask).type_num == NPY_TYPES.NPY_BOOL)
            {
                bool[] values = NpyArray_DATA(wheremask).datap as bool[];
                bool value = values[0];
                if (value)
                {
                    /* where=True is the same as no where at all */
                    wheremask = null;
                }
                else
                {
                    /* where=False copies nothing */
                    return 0;
                }
            }

            if (wheremask == null)
            {
                /* A straightforward value assignment */
                /* Do the assignment with raw array iteration */
                if (raw_array_assign_array(NpyArray_NDIM(dst), NpyArray_DIMS(dst),
                        NpyArray_DESCR(dst), NpyArray_DATA(dst), NpyArray_STRIDES(dst),
                        NpyArray_DESCR(src), NpyArray_DATA(src), src_strides) < 0)
                {
                    goto fail;
                }
            }
            else
            {
                npy_intp []wheremask_strides = new npy_intp[npy_defs.NPY_MAXDIMS];

                /* Broadcast the wheremask to 'dst' for raw iteration */
                if (broadcast_strides(NpyArray_NDIM(dst), NpyArray_DIMS(dst),
                            NpyArray_NDIM(wheremask), NpyArray_DIMS(wheremask),
                            NpyArray_STRIDES(wheremask), "where mask",
                            wheremask_strides) < 0)
                {
                    goto fail;
                }

                /* A straightforward where-masked assignment */
                /* Do the masked assignment with raw array iteration */
                if (raw_array_wheremasked_assign_array(
                        NpyArray_NDIM(dst), NpyArray_DIMS(dst),
                        NpyArray_DESCR(dst), NpyArray_DATA(dst), NpyArray_STRIDES(dst),
                        NpyArray_DESCR(src), NpyArray_DATA(src), src_strides,
                        NpyArray_DESCR(wheremask), NpyArray_DATA(wheremask),
                                wheremask_strides) < 0)
                {
                    goto fail;
                }
            }

            if (copied_src)
            {
                Npy_DECREF(src);
            }
            return 0;

            fail:
            if (copied_src)
            {
                Npy_DECREF(src);
            }
            return -1;
        }


}
}
