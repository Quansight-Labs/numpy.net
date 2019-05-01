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
using System.Diagnostics;
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
        static npy_tp_error_set NpyErr_SetString_callback;
        static npy_tp_error_occurred NpyErr_Occurred_callback;
        static npy_tp_error_clear NpyErr_Clear_callback;
        static npy_tp_cmp_priority Npy_CmpPriority;

        static enable_threads npy_enable_threads;
        static disable_threads npy_disable_threads;

        static object Npy_RefCntLock = null;


        internal static void npy_initlib(NpyArray_FunctionDefs functionDefs, NpyInterface_WrapperFuncs wrapperFuncs,
          npy_tp_error_set error_set,
          npy_tp_error_occurred error_occurred,
          npy_tp_error_clear error_clear,
          npy_tp_cmp_priority cmp_priority,
          npy_interface_incref incref, npy_interface_decref decref,
          enable_threads et, disable_threads dt)
        {
            Npy_RefCntLock = new object();

            _NpyArrayWrapperFuncs = wrapperFuncs;

            npy_enable_threads = et;
            npy_disable_threads = dt;

            // Verify that the structure definition is correct and has the memory layout
            // that we expect. 
            
            Debug.Assert(null == functionDefs || npy_defs.NPY_VALID_MAGIC == functionDefs.sentinel);

            if (null != wrapperFuncs)
            {
                // Store the passed in set of wrapper funcs.  However, the CPython interface has the ufunc
                // code in a separate module which may have initialized the ufunc method prior to this so
                // we don't want to overwrite it.
                //        npy_interface_ufunc_new_wrapper x = _NpyArrayWrapperFuncs.ufunc_new_wrapper;
                //        memmove(&_NpyArrayWrapperFuncs, wrapperFuncs, sizeof(struct NpyInterface_WrapperFuncs));
                //if (null == wrapperFuncs.ufunc_new_wrapper) {
                //    _NpyArrayWrapperFuncs.ufunc_new_wrapper = x;
            }

            //NpyArrayIter_Type.ntp_interface_alloc = (npy_wrapper_construct)wrapperFuncs.iter_new_wrapper;
            //NpyArrayMultiIter_Type.ntp_interface_alloc = (npy_wrapper_construct)wrapperFuncs.multi_iter_new_wrapper;
            //NpyArrayNeighborhoodIter_Type = wrapperFuncs.neighbor_iter_new_wrapper;

            NpyErr_SetString_callback = error_set;
            NpyErr_Occurred_callback = error_occurred;
            NpyErr_Clear_callback = error_clear;
            Npy_CmpPriority = cmp_priority;

            //_NpyInterface_Incref = incref;
            //_NpyInterface_Decref = decref;

            /* Must be last because it uses some of the above functions. */
            if (null != functionDefs)
            {
                _init_type_functions(functionDefs);
            }
            _init_builtin_descr_wrappers();
        }

        internal static void npy_set_ufunc_wrapper_func(npy_interface_ufunc_new_wrapper wrapperFunc)
        {
            _NpyArrayWrapperFuncs.ufunc_new_wrapper = wrapperFunc;
        }

        internal static npy_intp NpyArray_MultiplyIntList(npy_intp[] l1, int n)
        {
            npy_intp s = 1;

            int i = 0;
            while (n-- > 0)
            {
                s *= (l1[i++]);
            }
            return s;
        }

        internal static npy_intp NpyArray_MultiplyList(npy_intp[] l1, int n)
        {
            npy_intp s = 1;

            int i = 0;
            while (n-- > 0)
            {
                s *= (l1[i++]);
            }
            return s;
        }

        /*
        * Multiply a List of Non-negative numbers with over-flow detection.
        */
        internal static npy_intp NpyArray_OverflowMultiplyList(npy_intp[] l1, int n)
        {
            npy_intp prod = 1;
            npy_intp imax = Int32.MaxValue;
            int i;

            for (i = 0; i < n; i++)
            {
                npy_intp dim = l1[i];

                if (dim == 0)
                {
                    return 0;
                }
                if (dim > imax)
                {
                    return -1;
                }
                imax /= dim;
                prod *= dim;
            }
            return prod;
        }


        internal static VoidPtr NpyArray_GetPtr(NpyArray arr, npy_intp[] indexes)
        {
            int n = arr.nd;
            npy_intp[] strides = arr.strides;
            object dptr = arr.data;

            int strides_index = 0;
            int ind_index = 0;
            npy_intp dptr_index = 0;

            while (n-- > 0)
            {
                dptr_index += (strides[strides_index++]) * (indexes[ind_index++]);
            }

            VoidPtr vp = new VoidPtr(arr, dptr_index);
            return vp;
        }


        internal static bool NpyArray_CompareLists(npy_intp[] l1, npy_intp[] l2, npy_intp n)
        {
            int i;

            for (i = 0; i < n; i++)
            {
                if (l1[i] != l2[i])
                {
                    return false;
                }
            }
            return true;
        }

        internal static int NpyArray_AsCArray(ref NpyArray apIn, ref byte[] ptr, npy_intp[] dims, int nd, NpyArray_Descr typedescr)
        {
            NpyArray ap;
            npy_intp n, m, i, j;
            byte[] ptr2;
            byte[] ptr3;

            if ((nd < 1) || (nd > 3))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                 "C arrays of only 1-3 dimensions available");
                Npy_XDECREF(typedescr);
                return -1;
            }
            if ((ap = NpyArray_FromArray(apIn, typedescr, NPYARRAYFLAGS.NPY_CARRAY)) == null)
            {
                return -1;
            }

            byte[] data = ap.data.datap as byte[];
            switch (nd)
            {
                case 1:
                    ptr = ap.data.datap as byte[];
                    break;
                case 2:
                    n = ap.dimensions[0];
                    ptr2 = new byte[n * 4];
                    if (ptr2 == null)
                    {
                        goto fail;
                    }
                    for (i = 0; i < n; i++)
                    {
                        ptr2[i] = data[i * ap.strides[0]];
                    }
                    ptr = ptr2;
                    break;
                case 3:
                    n = ap.dimensions[0];
                    m = ap.dimensions[1];
                    ptr3 = new byte[n * (m + 1) * 4];
                    if (ptr3 == null)
                    {
                        goto fail;
                    }
                    for (i = 0; i < n; i++)
                    {
                        ptr3[i] = ptr3[n + (m - 1) * i];
                        for (j = 0; j < m; j++)
                        {
                            ptr3[i*j] = data[i * ap.strides[0] + j * ap.strides[1]];
                        }
                    }
                    ptr = ptr3;
                    break;
            }
            memcpy(dims, ap.dimensions, nd * sizeof(npy_intp));
            apIn = ap;
            return 0;

            fail:
            NpyErr_MEMORY();
            return -1;
        }

        internal static int NpyArray_Free(NpyArray ap, object ptr)
        {
            if ((ap.nd < 1) || (ap.nd > 3))
            {
                return -1;
            }
            if (ap.nd >= 2)
            {
                /* TODO: Notice lower case 'f' - points to define that translate to
                         free or something. */
                NpyArray_free(ptr);
            }
            Npy_DECREF(ap);
            return 0;
        }

        internal static NpyArray NpyArray_Concatenate(NpyArray [] op, int? axis, NpyArray ret)
        {
            return NpyArray_ConcatenateInto(op, axis, ret);
        }

        private static NpyArray NpyArray_ConcatenateInto(NpyArray[] op, int? axis, NpyArray ret)
        {
            int iarrays, narrays;
            NpyArray [] arrays;

            /* Convert the input list into arrays */
            narrays = op.Length;
            if (narrays < 0)
            {
                return null;
            }
            arrays = new NpyArray[narrays];
            if (arrays == null)
            {
                NpyErr_NoMemory();
                return null;
            }

            for (iarrays = 0; iarrays < narrays; ++iarrays)
            {
                NpyArray item = NpyArray_FromArray(op[iarrays], null, 0);
                if (item == null)
                {
                    narrays = iarrays;
                    goto fail;
                }
                arrays[iarrays] = item;
                Npy_DECREF(item);
                if (arrays[iarrays] == null)
                {
                    narrays = iarrays;
                    goto fail;
                }
            }


            if (axis == null)
            {
                ret = PyArray_ConcatenateFlattenedArrays(narrays, arrays, NPY_ORDER.NPY_CORDER, ret);
            }
            else
            {
                ret = PyArray_ConcatenateArrays(narrays, arrays, axis.Value, ret);
            }

            for (iarrays = 0; iarrays < narrays; ++iarrays)
            {
                Npy_DECREF(arrays[iarrays]);
            }
            NpyArray_free(arrays);

            return ret;

            fail:
            /* 'narrays' was set to how far we got in the conversion */
            for (iarrays = 0; iarrays < narrays; ++iarrays)
            {
                Npy_DECREF(arrays[iarrays]);
            }
            NpyArray_free(arrays);

            return null;

        }

        private static NpyArray PyArray_ConcatenateArrays(int narrays, NpyArray[] arrays, int axis, NpyArray ret)
        {
            int iarrays, idim, ndim;
            npy_intp []shape = new npy_intp[npy_defs.NPY_MAXDIMS];
            NpyArray sliding_view = null;

            if (narrays <= 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_TypeError, "need at least one array to concatenate");
                return null;
            }

            /* All the arrays must have the same 'ndim' */
            ndim = NpyArray_NDIM(arrays[0]);

            if (ndim == 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "zero-dimensional arrays cannot be concatenated");
                return null;
            }

            /* Handle standard Python negative indexing */
            if (check_and_adjust_axis(ref axis, ndim) == false)
            {
                return null;
            }

            /*
            * Figure out the final concatenated shape starting from the first
            * array's shape.
            */
            memcpy(shape, arrays[0].dimensions, ndim * sizeof(npy_intp));
            for (iarrays = 1; iarrays < narrays; ++iarrays)
            {
                npy_intp[] arr_shape;

                if (NpyArray_NDIM(arrays[iarrays]) != ndim)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "all the input arrays must have same number of dimensions");
                    return null;
                }
                arr_shape = arrays[iarrays].dimensions;

                for (idim = 0; idim < ndim; ++idim)
                {
                    /* Build up the size of the concatenation axis */
                    if (idim == axis)
                    {
                        shape[idim] += arr_shape[idim];
                    }
                    /* Validate that the rest of the dimensions match */
                    else if (shape[idim] != arr_shape[idim])
                    {
                        NpyErr_SetString(npyexc_type.NpyExc_ValueError, "all the input array dimensions except for the concatenation axis must match exactly");
                        return null;
                    }
                }
            }


            if (ret != null)
            {
                if (NpyArray_NDIM(ret) != ndim)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "Output array has wrong dimensionality");
                    return null;
                }
                if (!NpyArray_CompareLists(shape, ret.dimensions, ndim))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "Output array is the wrong shape");
                    return null;
                }
                Npy_INCREF(ret);
            }
            else
            {
                npy_intp s;
                npy_intp []strides = new npy_intp[npy_defs.NPY_MAXDIMS];
                int []strideperm = new int[npy_defs.NPY_MAXDIMS];

                /* Get the priority subtype for the array */
                object subtype = NpyArray_GetSubType(narrays, arrays);


                /* Get the resulting dtype from combining all the arrays */
                NpyArray_Descr dtype = NpyArray_ResultType(narrays, arrays, 0, null);
                if (dtype == null)
                {
                    return null;
                }

                /*
                 * Figure out the permutation to apply to the strides to match
                 * the memory layout of the input arrays, using ambiguity
                 * resolution rules matching that of the NpyIter.
                */

                NpyArray_CreateMultiSortedStridePerm(narrays, arrays, ndim, strideperm);
                s = dtype.elsize;
                for (idim = ndim - 1; idim >= 0; --idim)
                {
                    int iperm = strideperm[idim];
                    strides[iperm] = s;
                    s *= shape[iperm];
                }

                /* Allocate the array for the result. This steals the 'dtype' reference. */
                ret = NpyArray_NewFromDescr(dtype, ndim,  shape,  strides, null, 0, false, subtype, null);

 
                if (ret == null)
                {
                    return null;
                }
            }

            /*
            * Create a view which slides through ret for assigning the
            * successive input arrays.
            */

            sliding_view = NpyArray_View(ret, null, PyArray_Type);
            if (sliding_view == null)
            {
                Npy_DECREF(ret);
                return null;
            }

            for (iarrays = 0; iarrays < narrays; ++iarrays)
            {
                /* Set the dimension to match the input array's */
                sliding_view.dimensions[axis] = arrays[iarrays].dimensions[axis];

                ///* Copy the data for this array */
                //if (NpyArray_StridedCopyInto(sliding_view, arrays[iarrays]) < 0)
                //{
                //    Npy_DECREF(sliding_view);
                //    Npy_DECREF(ret);
                //    return null;
                //}

                /* Copy the data for this array */
                if (NpyArray_AssignArray(sliding_view, arrays[iarrays],
                                    null, NPY_CASTING.NPY_SAME_KIND_CASTING) < 0)
                {
                    Npy_DECREF(sliding_view);
                    Npy_DECREF(ret);
                    return null;
                }


                /* Slide to the start of the next window */
                sliding_view.data += sliding_view.dimensions[axis] * sliding_view.strides[axis];
            }


            Npy_DECREF(sliding_view);
            return ret;

        }

        private static NpyArray PyArray_ConcatenateFlattenedArrays(int narrays, NpyArray[] arrays, NPY_ORDER order, NpyArray ret)
        {
            int iarrays;
            npy_intp shape = 0;
            NpyArray sliding_view = null;

            if (narrays <= 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "need at least one array to concatenate");
                return null;
            }

            /*
             * Figure out the final concatenated shape starting from the first
             * array's shape.
             */
            for (iarrays = 0; iarrays < narrays; ++iarrays)
            {
                shape += NpyArray_SIZE(arrays[iarrays]);
                /* Check for overflow */
                if (shape < 0)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "total number of elements too large to concatenate");
                    return null;
                }
            }

            if (ret != null)
            {
                if (NpyArray_NDIM(ret) != 1)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "Output array must be 1D");
                    return null;
                }
                if (shape != NpyArray_SIZE(ret))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "Output array is the wrong size");
                    return null;
                }
                Npy_INCREF(ret);
            }
            else
            {
                npy_intp stride;

                /* Get the priority subtype for the array */
                object subtype = NpyArray_GetSubType(narrays, arrays);

                /* Get the resulting dtype from combining all the arrays */
                NpyArray_Descr dtype = NpyArray_ResultType(narrays, arrays, 0, null);
                if (dtype == null)
                {
                    return null;
                }

                stride = dtype.elsize;

                /* Allocate the array for the result. This steals the 'dtype' reference. */
                ret = NpyArray_NewFromDescr(dtype,
                                           1,
                                           new npy_intp[] { shape },
                                           new npy_intp[] { stride },
                                           null,
                                           0,
                                           false,
                                           null, null);

                if (ret == null)
                {
                    return null;
                }
            }

            /*
            * Create a view which slides through ret for assigning the
            * successive input arrays.
            */
            sliding_view =  NpyArray_View(ret, null, PyArray_Type);
            if (sliding_view == null)
            {
                Npy_DECREF(ret);
                return null;
            }

            for (iarrays = 0; iarrays < narrays; ++iarrays)
            {
                /* Adjust the window dimensions for this array */
                sliding_view.dimensions[0] = NpyArray_SIZE(arrays[iarrays]);

                /* Copy the data for this array */
                if (PyArray_CopyAsFlat(sliding_view, arrays[iarrays],
                                    order) < 0)
                {
                    Npy_DECREF(sliding_view);
                    Npy_DECREF(ret);
                    return null;
                }

                /* Slide to the start of the next window */
                sliding_view.data.data_offset += sliding_view.strides[0] * NpyArray_SIZE(arrays[iarrays]);
            }

            Npy_DECREF(sliding_view);
            return ret;


        }

        internal static NPY_SCALARKIND NpyArray_ScalarKind(NPY_TYPES typenum, NpyArray arr)
        {
            if (NpyTypeNum_ISSIGNED(typenum))
            {
                if (arr != null && _signbit_set(arr))
                {
                    return NPY_SCALARKIND.NPY_INTNEG_SCALAR;
                }
                else
                {
                    return NPY_SCALARKIND.NPY_INTPOS_SCALAR;
                }
            }
            if (NpyTypeNum_ISFLOAT(typenum))
            {
                return NPY_SCALARKIND.NPY_FLOAT_SCALAR;
            }
            if (NpyTypeNum_ISUNSIGNED(typenum))
            {
                return NPY_SCALARKIND.NPY_INTPOS_SCALAR;
            }
            if (NpyTypeNum_ISCOMPLEX(typenum))
            {
                return NPY_SCALARKIND.NPY_COMPLEX_SCALAR;
            }
            if (NpyTypeNum_ISBOOL(typenum))
            {
                return NPY_SCALARKIND.NPY_BOOL_SCALAR;
            }

            if (NpyTypeNum_ISUSERDEF(typenum))
            {
                NPY_SCALARKIND retval;
                NpyArray_Descr descr = NpyArray_DescrFromType(typenum);

                if (descr.f.scalarkind != null)
                {
                    retval = descr.f.scalarkind(arr);
                }
                else
                {
                    retval = NPY_SCALARKIND.NPY_NOSCALAR;
                }
                Npy_DECREF(descr);
                return retval;
            }
            return NPY_SCALARKIND.NPY_OBJECT_SCALAR;
        }

        internal static bool NpyArray_CanCoerceScalar(NPY_TYPES thistype, NPY_TYPES neededtype, NPY_SCALARKIND scalar)
        {
            NpyArray_Descr from;
            object castlist;

            if (scalar == NPY_SCALARKIND.NPY_NOSCALAR)
            {
                return NpyArray_CanCastSafely(thistype, neededtype);
            }

            from = NpyArray_DescrFromType(thistype);
            if (from.f.cancastscalarkindto != null)
            {
                castlist = from.f.cancastscalarkindto[scalar];
                if (castlist != null)
                {
                    NPY_TYPES casttype = (NPY_TYPES)castlist;

                    while (casttype != NPY_TYPES.NPY_NOTYPE)
                    {
                        if (casttype++ == neededtype)
                        {
                            Npy_DECREF(from);
                            return true;
                        }
                    }
                }
     
            }
            Npy_DECREF(from);

            switch (scalar)
            {
                case NPY_SCALARKIND.NPY_BOOL_SCALAR:
                case NPY_SCALARKIND.NPY_OBJECT_SCALAR:
                    return NpyArray_CanCastSafely(thistype, neededtype);
                default:
                    if (NpyTypeNum_ISUSERDEF(neededtype))
                    {
                        return false;
                    }
                    switch (scalar)
                    {
                        case NPY_SCALARKIND.NPY_INTPOS_SCALAR:
                            return (neededtype >= NPY_TYPES.NPY_BYTE);
                        case NPY_SCALARKIND.NPY_INTNEG_SCALAR:
                            return (neededtype >= NPY_TYPES.NPY_BYTE) && !(NpyTypeNum_ISUNSIGNED(neededtype));
                        case NPY_SCALARKIND.NPY_FLOAT_SCALAR:
                            return (neededtype >= NPY_TYPES.NPY_FLOAT);
                        case NPY_SCALARKIND.NPY_COMPLEX_SCALAR:
                            return (neededtype >= NPY_TYPES.NPY_CFLOAT);
                        default:
                            /* should never get here... */
                            return true;
                    }
            }
        }


        internal static NpyArray NpyArray_InnerProduct(NpyArray ap1, NpyArray ap2, NPY_TYPES typenum)
        {
            NpyArray ret = null;
            NpyArrayIterObject it1, it2;
            npy_intp i, j, l;
            int nd, axis;
            npy_intp is1, is2, os;
            VoidPtr op;
            npy_intp []dimensions = new npy_intp[npy_defs.NPY_MAXDIMS];
            NpyArray_DotFunc dot;

            l = ap1.dimensions[ap1.nd - 1];
            if (ap2.dimensions[ap2.nd - 1] != l)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "matrices are not aligned");
                return null;
            }

            nd = ap1.nd + ap2.nd - 2;
            j = 0;
            for (i = 0; i < ap1.nd - 1; i++)
            {
                dimensions[j++] = ap1.dimensions[i];
            }
            for (i = 0; i < ap2.nd - 1; i++)
            {
                dimensions[j++] = ap2.dimensions[i];
            }

            /*
             * Need to choose an output array that can hold a sum
             * -- use priority to determine which subtype.
             */
            ret = new_array_for_sum(ap1, ap2, nd, dimensions, typenum);
            if (ret == null)
            {
                return null;
            }
            dot = ret.descr.f.dotfunc;
            if (dot == null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "dot not available for this type");
                goto fail;
            }
            is1 = ap1.strides[ap1.nd - 1];
            is2 = ap2.strides[ap2.nd - 1];
            op = new VoidPtr(ret);
            os = ret.descr.elsize;
            axis = ap1.nd - 1;
            it1 = NpyArray_IterAllButAxis(ap1, ref axis);
            axis = ap2.nd - 1;
            it2 = NpyArray_IterAllButAxis(ap2, ref axis);
            while (true)
            {
                while (it2.index < it2.size)
                {
                    dot(it1.dataptr, is1, it2.dataptr, is2, op, l, ret);
                    op.data_offset += os;
                    NpyArray_ITER_NEXT(it2);
                }
                NpyArray_ITER_NEXT(it1);
                if (it1.index >= it1.size)
                {
                    break;
                }
                NpyArray_ITER_RESET(it2);
            }
            Npy_DECREF(it1);
            Npy_DECREF(it2);
            if (NpyErr_Occurred())
            {
                goto fail;
            }
            return ret;

            fail:
            Npy_DECREF(ret);
            return null;
        }

        internal static NpyArray NpyArray_MatrixProduct(NpyArray ap1, NpyArray ap2, NPY_TYPES typenum)
        {
            NpyArray ret = null;
            NpyArrayIterObject it1, it2;
            npy_intp i, j, l, is1, is2, os;
            npy_intp []dimensions = new  npy_intp[npy_defs.NPY_MAXDIMS];
            int nd, axis, matchDim;
            VoidPtr op;
            NpyArray_DotFunc dot;

            if (ap2.nd > 1)
            {
                matchDim = ap2.nd - 2;
            }
            else
            {
                matchDim = 0;
            }
            l = ap1.dimensions[ap1.nd - 1];
            if (ap2.dimensions[matchDim] != l)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "objects are not aligned");
                return null;
            }
            nd = ap1.nd + ap2.nd - 2;
            if (nd > npy_defs.NPY_MAXDIMS)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "dot: too many dimensions in result");
                return null;
            }
            j = 0;
            for (i = 0; i < ap1.nd - 1; i++)
            {
                dimensions[j++] = ap1.dimensions[i];
            }
            for (i = 0; i < ap2.nd - 2; i++)
            {
                dimensions[j++] = ap2.dimensions[i];
            }
            if (ap2.nd > 1)
            {
                dimensions[j++] = ap2.dimensions[ap2.nd - 1];
            }
            is1 = ap1.strides[ap1.nd - 1];
            is2 = ap2.strides[matchDim];
            /* Choose which subtype to return */
            ret = new_array_for_sum(ap1, ap2, nd, dimensions, typenum);
            if (ret == null)
            {
                return null;
            }
            /* Ensure that multiarray.dot(<Nx0>,<0xM>) . zeros((N,M)) */
            if (NpyArray_SIZE(ap1) == 0 && NpyArray_SIZE(ap2) == 0)
            {
                memset(NpyArray_DATA(ret), 0, (int)NpyArray_NBYTES(ret));
            }
            else
            {
                /* Ensure that multiarray.dot([],[]) . 0 */
                memset(NpyArray_DATA(ret), 0, (int)NpyArray_ITEMSIZE(ret));
            }

            dot = ret.descr.f.dotfunc;
            if (dot == null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "dot not available for this type");
                goto fail;
            }

            op = new VoidPtr(ret);
            os = NpyArray_ITEMSIZE(ret);
            axis = ap1.nd - 1;
            it1 = NpyArray_IterAllButAxis(ap1, ref axis);
            it2 = NpyArray_IterAllButAxis(ap2, ref matchDim);
            while (true)
            {
                while (it2.index < it2.size)
                {
                    dot(it1.dataptr, is1, it2.dataptr, is2, op, l, ret);
                    op.data_offset += os;
                    NpyArray_ITER_NEXT(it2);
                }
                NpyArray_ITER_NEXT(it1);
                if (it1.index >= it1.size)
                {
                    break;
                }
                NpyArray_ITER_RESET(it2);
            }
            Npy_DECREF(it1);
            Npy_DECREF(it2);
            if (NpyErr_Occurred())
            {
                goto fail;
            }
            return ret;

            fail:
            Npy_XDECREF(ret);
            return null;
        }

        internal static NpyArray NpyArray_ContiguousFromArray(NpyArray op, NPY_TYPES type)
        {
            return NpyArray_FromArray(op, NpyArray_DescrFromType(type), NPYARRAYFLAGS.NPY_DEFAULT);
        }

        internal static NpyArray NpyArray_CopyAndTranspose(NpyArray arr)
        {
            NpyArray ret, tmp;
            int nd, eltsize;
            npy_intp stride2;
            npy_intp[] dims = new npy_intp[2];
            npy_intp i, j;
            VoidPtr iptr; VoidPtr optr;

            /* make sure it is well-behaved */
            tmp = NpyArray_ContiguousFromArray(arr, NpyArray_TYPE(arr));
            if (tmp == null)
            {
                return null;
            }
            arr = tmp;

            nd = NpyArray_NDIM(arr);
            if (nd == 1)
            {
                /* we will give in to old behavior */
                Npy_DECREF(tmp);
                return arr;
            }
            else if (nd != 2)
            {
                Npy_DECREF(tmp);
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "only 2-d arrays are allowed");
                return null;
            }

            /* Now construct output array */
            dims[0] = NpyArray_DIM(arr, 1);
            dims[1] = NpyArray_DIM(arr, 0);
            eltsize = NpyArray_ITEMSIZE(arr);
            Npy_INCREF(arr.descr);
            ret = NpyArray_Alloc(arr.descr, 2, dims, false, null);
            if (ret == null)
            {
                Npy_DECREF(tmp);
                return null;
            }

            /* do 2-d loop */
            optr = new VoidPtr(ret);
            stride2 = eltsize * dims[0];
            for (i = 0; i < dims[0]; i++)
            {
                iptr = new VoidPtr(arr);
                iptr.data_offset += i * eltsize;
                for (j = 0; j < dims[1]; j++)
                {
                    /* optr[i,j] = iptr[j,i] */
                    memcpy(optr, iptr, eltsize);
                    optr.data_offset += eltsize;
                    iptr.data_offset += stride2;
                }
            }

            Npy_DECREF(tmp);
            return ret;
        }

        internal static NpyArray NpyArray_Correlate2(NpyArray ap1, NpyArray ap2, NPY_TYPES typenum, NPY_CONVOLE_MODE mode)
        {
            NpyArray ret = null;
            NpyArray cap2 = null;
            int inverted = 0;
            int status;

            if (NpyArray_ISCOMPLEX(ap2))
            {
                /* FIXME: PyArray_Conjugate need to be replaced by NpyArray_Conjugate,
                          once NpyArray_Conjugate is created, which can be done once
                          the ufunc stuff is in the core.
                 */
                ap2 = NpyArray_Conjugate(ap2, null);
                if (null == ap2)
                {
                    return null;
                }
                cap2 = ap2;
            }

            ret = _npyarray_correlate(ap1, ap2, typenum, mode, ref inverted);
            if (ret == null)
            {
                goto done;
            }

            /* If we inverted input orders, we need to reverse the output array (i.e.
               ret = ret[::-1]) */
            if (inverted != 0)
            {
                status = _npyarray_revert(ret);
                if (status != 0)
                {
                    Npy_DECREF(ret);
                    ret = null;
                    goto done;
                }
            }
            done:
            Npy_XDECREF(cap2);
            return ret;
        }

        internal static NpyArray NpyArray_Correlate(NpyArray ap1, NpyArray ap2, NPY_TYPES typenum, NPY_CONVOLE_MODE mode)
        {
            int unused = 0;

            return _npyarray_correlate(ap1, ap2, typenum, mode, ref unused);
        }

        static NpyArray _npyarray_correlate(NpyArray ap1, NpyArray ap2, NPY_TYPES typenum, NPY_CONVOLE_MODE mode, ref int inverted)
        {
            NpyArray ret;
            npy_intp[] length = new npy_intp[1];
            npy_intp i, n1, n2, n, n_left, n_right, is1, is2, os;
            VoidPtr ip1;
            VoidPtr ip2;
            VoidPtr op;
            NpyArray_DotFunc dot;

            n1 = NpyArray_DIM(ap1, 0);
            n2 = NpyArray_DIM(ap2, 0);
            if (n1 < n2)
            {
                ret = ap1;
                ap1 = ap2;
                ap2 = ret;
                ret = null;
                i = n1;
                n1 = n2;
                n2 = i;
                inverted = 1;
            }
            else
            {
                inverted = 0;
            }

            length[0] = n1;
            n = n2;
            switch (mode)
            {
                case NPY_CONVOLE_MODE.NPY_CONVOLVE_VALID:
                    length[0] = length[0] - n + 1;
                    n_left = n_right = 0;
                    break;
                case NPY_CONVOLE_MODE.NPY_CONVOLVE_SAME:
                    n_left = (npy_intp)(n / 2);
                    n_right = n - n_left - 1;
                    break;
                case NPY_CONVOLE_MODE.NPY_CONVOLVE_FULL:
                    n_right = n - 1;
                    n_left = n - 1;
                    length[0] = length[0] + n - 1;
                    break;
                default:
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "mode must be 0, 1, or 2");
                    return null;
            }

            /*
             * Need to choose an output array that can hold a sum
             * -- use priority to determine which subtype.
             */
            ret = new_array_for_sum(ap1, ap2, 1, length, typenum);
            if (ret == null)
            {
                return null;
            }
            dot = NpyArray_DESCR(ret).f.dotfunc;
            if (dot == null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                 "function not available for this data type");
                goto clean_ret;
            }

            is1 = NpyArray_STRIDE(ap1, 0);
            is2 = NpyArray_STRIDE(ap2, 0);
            op = new VoidPtr(ret);
            os = NpyArray_ITEMSIZE(ret);
            ip1 = new VoidPtr(ap1);
            ip2 = new VoidPtr(ap2);
            ip2.data_offset += n_left * is2;
            n -= n_left;

            for (i = 0; i < n_left; i++)
            {
                dot(ip1, is1, ip2, is2, op, n, ret);
                n++;
                ip2.data_offset -= is2;
                op.data_offset += os;
            }

            for (i = 0; i < (n1 - n2 + 1); i++)
            {
                dot(ip1, is1, ip2, is2, op, n, ret);
                ip1.data_offset += is1;
                op.data_offset += os;
            }
            for (i = 0; i < n_right; i++)
            {
                n--;
                dot(ip1, is1, ip2, is2, op, n, ret);
                ip1.data_offset += is1;
                op.data_offset += os;
            }

            if (NpyErr_Occurred())
            {
                goto clean_ret;
            }

            return ret;

            clean_ret:
            Npy_DECREF(ret);
            return null;
        }

        /*
         * Revert a one dimensional array in-place
         *
         * Return 0 on success, other value on failure
         */

        static int _npyarray_revert(NpyArray ret)
        {
            npy_intp length, i, os;
            NpyArray_CopySwapFunc copyswap;
            VoidPtr tmp = null;
            VoidPtr sw1;
            VoidPtr sw2;
            VoidPtr op;

            length = NpyArray_DIM(ret, 0);
            copyswap = NpyArray_DESCR(ret).f.copyswap;

            tmp = NpyDataMem_NEW(ret.descr.type_num, (ulong)NpyArray_ITEMSIZE(ret));
            if (tmp == null)
            {
                return -1;
            }

            os = NpyArray_ITEMSIZE(ret);
            op = new VoidPtr(ret);
            sw1 = new VoidPtr(op);
            sw2 = new VoidPtr(op);
            sw2.data_offset += (length - 1) * os;

            if (NpyArray_ISFLEXIBLE(ret) || NpyArray_ISOBJECT(ret))
            {
                for (i = 0; i < length / 2; i++)
                {
                    memmove(tmp,0, sw1, 0, os);
                    copyswap(tmp, null,  false, null);


                    memmove(sw1, 0, sw2, 0, os);
                    copyswap(sw1, null, false, null);


                    memmove(sw2, 0, tmp, 0, os);
                    copyswap(sw2, null, false, null);
                    copyswap(sw2, null, false, null);

                    sw1.data_offset += os;
                    sw2.data_offset -= os;
                }
            }
            else
            {
                for (i = 0; i < length / 2; i++)
                {
                    memcpy(tmp, sw1, os);
                    memcpy(sw1, sw2, os);
                    memcpy(sw2, tmp, os);
                    sw1.data_offset += os;
                    sw2.data_offset -= os;
                }
            }

            return 0;
        }

        internal static NpyArray new_array_for_sum(NpyArray ap1, NpyArray ap2,  int nd, npy_intp []dimensions, NPY_TYPES typenum)
        {
            int tmp;

            /*
             * Need to choose an output array that can hold a sum
             */
            tmp = Npy_CmpPriority(Npy_INTERFACE(ap1), Npy_INTERFACE(ap2));

            return NpyArray_New(null, nd, dimensions,
                                typenum, null, null, 0, 0,
                                Npy_INTERFACE((tmp != 0) ? ap2 : ap1));
        }


        internal static bool NpyArray_EquivTypenums(NPY_TYPES typenum1, NPY_TYPES typenum2)
        {
            NpyArray_Descr d1, d2;
            bool ret;

            d1 = NpyArray_DescrFromType(typenum1);
            d2 = NpyArray_DescrFromType(typenum2);
            ret = NpyArray_EquivTypes(d1, d2);
            Npy_DECREF(d1);
            Npy_DECREF(d2);
            return ret;
        }

        enum Endianess
        {
            NPY_CPU_UNKNOWN_ENDIAN,
            NPY_CPU_LITTLE,
            NPY_CPU_BIG,
        }

        internal static int NpyArray_GetEndianness()
        {
            return (int)((BitConverter.IsLittleEndian == true) ? Endianess.NPY_CPU_LITTLE : Endianess.NPY_CPU_BIG);
        }


        private static void _init_builtin_descr_wrappers()
        {
            return;
        }

        private static void _init_type_functions(NpyArray_FunctionDefs functionDefs)
        {
            return;
        }


   

        internal static bool NpyArray_EquivArrTypes(NpyArray a1, NpyArray a2)
        {
            return NpyArray_EquivTypes(NpyArray_DESCR(a1), NpyArray_DESCR(a2));
        }

     

        internal static bool NpyArray_EquivTypes(NpyArray_Descr typ1, NpyArray_Descr typ2)
        {
            NPY_TYPES typenum1 = typ1.type_num;
            NPY_TYPES typenum2 = typ2.type_num;
            int size1 = typ1.elsize;
            int size2 = typ2.elsize;

            if (size1 != size2)
            {
                return false;
            }
            if (NpyArray_ISNBO(typ1) != NpyArray_ISNBO(typ2))
            {
                return false;
            }
            if (null != typ1.subarray || null != typ2.subarray)
            {
                return typenum1 == typenum2 && _equivalent_subarrays(typ1.subarray, typ2.subarray);
            }

            if (typenum1 == NPY_TYPES.NPY_VOID || typenum2 == NPY_TYPES.NPY_VOID)
            {
                return ((typenum1 == typenum2)
                        && _equivalent_fields(typ1.fields, typ2.fields));
            }
            if (typenum1 == NPY_TYPES.NPY_DATETIME
                || typenum1 == NPY_TYPES.NPY_DATETIME
                || typenum2 == NPY_TYPES.NPY_TIMEDELTA
                || typenum2 == NPY_TYPES.NPY_TIMEDELTA)
            {
                return ((typenum1 == typenum2)
                        && _equivalent_units(typ1.dtinfo, typ2.dtinfo));
            }
            return typ1.kind == typ2.kind;
        }

        internal static void NpyArray_CopyTo(NpyArray destArray, NpyArray srcArray, NPY_CASTING casting, NpyArray whereArray)
        {
            var destSize = NpyArray_Size(destArray);

            NpyArrayIterObject SrcIter = NpyArray_BroadcastToShape(srcArray, destArray.dimensions, destArray.nd);
            NpyArrayIterObject DestIter = NpyArray_BroadcastToShape(destArray, destArray.dimensions, destArray.nd);
            NpyArrayIterObject WhereIter = null;
            if (whereArray != null)
            {
                WhereIter = NpyArray_BroadcastToShape(whereArray, destArray.dimensions, destArray.nd);
            }

            bool whereValue = true;

            for (long i = 0; i < destSize; i++)
            {
                var srcValue = srcArray.descr.f.getitem(SrcIter.dataptr.data_offset - srcArray.data.data_offset, srcArray);

                if (WhereIter != null)
                {
                    whereValue = (bool)whereArray.descr.f.getitem(WhereIter.dataptr.data_offset - whereArray.data.data_offset, whereArray);
                }

                if (whereValue)
                {
                    destArray.descr.f.setitem(DestIter.dataptr.data_offset - destArray.data.data_offset, srcValue, destArray);
                }

                NpyArray_ITER_NEXT(SrcIter);
                NpyArray_ITER_NEXT(DestIter);
                if (WhereIter != null)
                {
                    NpyArray_ITER_NEXT(WhereIter);
                }
            }

            return;
        }

        internal static void NpyArray_Place(NpyArray arr,  NpyArray mask, NpyArray vals)
        {
            var arrSize = NpyArray_Size(arr);
            var maskSize = NpyArray_Size(mask);

            if (arrSize != maskSize)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "size of mask must be same size as src arr");
                return;
            }

            NpyArrayIterObject valsIter = NpyArray_IterNew(vals);
            NpyArrayIterObject arrIter = NpyArray_IterNew(arr);
            NpyArrayIterObject maskIter = NpyArray_IterNew(mask);
      
            for (long i = 0; i < arrSize; i++)
            {
                bool whereValue = (bool)mask.descr.f.getitem(maskIter.dataptr.data_offset - mask.data.data_offset, mask);

                if (whereValue)
                {
                    var valValue = vals.descr.f.getitem(valsIter.dataptr.data_offset - vals.data.data_offset, vals);
                    arr.descr.f.setitem(arrIter.dataptr.data_offset - arr.data.data_offset, valValue, arr);
                    NpyArray_ITER_NEXT(valsIter);
                }

                if (!NpyArray_ITER_NOTDONE(valsIter))
                {
                    NpyArray_ITER_RESET(valsIter);
                }

                NpyArray_ITER_NEXT(arrIter);
                NpyArray_ITER_NEXT(maskIter);
            }

            return;
        }


        internal static void NpyArray_CopyTo2(NpyArray dst, NpyArray src, NPY_CASTING casting, NpyArray wheremask_in)
        {
            NpyArray wheremask = null;

            if (wheremask_in != null)
            {
                /* Get the boolean where mask */
                NpyArray_Descr dtype = NpyArray_DescrFromType(NPY_TYPES.NPY_BOOL);
                if (dtype == null)
                {
                    goto fail;
                }
                wheremask = NpyArray_FromArray(wheremask_in, dtype, NPYARRAYFLAGS.NPY_DEFAULT);
                if (wheremask == null)
                {
                    goto fail;
                }
            }

            if (NpyArray_AssignArray(dst, src, wheremask, casting) < 0)
            {
                goto fail;
            }

            Npy_XDECREF(src);
            Npy_XDECREF(wheremask);

            return;

            fail:
            Npy_XDECREF(src);
            Npy_XDECREF(wheremask);
            return;
        }

        static bool _equivalent_fields(NpyDict field1, NpyDict field2)
        {
            NpyDict_Iter pos = new NpyDict_Iter();
            NpyArray_DescrField value1 = null;
            NpyArray_DescrField value2 = null;
            object key = null;
            bool same = true;
            NpyDict_KVPair KVPair = new NpyDict_KVPair();

            if (field1 == field2)
            {
                return true;
            }
            if (field1 == null || field2 == null)
            {
                return false;
            }
            if (NpyDict_Size(field1) != NpyDict_Size(field2))
            {
                same = false;
            }

            NpyDict_IterInit(pos);
            while (same && NpyDict_IterNext(field1, pos,  KVPair))
            {
                value2 = (NpyArray_DescrField)NpyDict_Get(field2, key);
                if (null == value2 || value1.offset != value2.offset ||
                    ((null == value1.title && null != value2.title) ||
                     (null != value1.title && null == value2.title) ||
                     (null != value1.title && null != value2.title &&
                      value1.title == value2.title)))
                {
                    same = false;
                }
                else if (!NpyArray_EquivTypes(value1.descr, value2.descr))
                {
                    same = false;
                }
            }
            return same;
        }

        static bool _equivalent_subarrays(NpyArray_ArrayDescr sub1, NpyArray_ArrayDescr sub2)
        {
            if (sub1 == sub2)
            {
                return true;

            }
            if (sub1 == null || sub2 == null)
            {
                return false;
            }

            if (sub1.shape_num_dims != sub2.shape_num_dims)
            {
                return false;
            }
            for (int i = 0; i < sub1.shape_num_dims; i++)
            {
                if (sub1.shape_dims[i] != sub2.shape_dims[i])
                {
                    return false;
                }
            }

            return NpyArray_EquivTypes(sub1._base, sub2._base);
        }

        /* compare the metadata for two date-times
         * return 1 if they are the same, or 0 if not
         */
        static bool _equivalent_units(NpyArray_DateTimeInfo info1, NpyArray_DateTimeInfo info2)
        {
            /* Same meta object */
            return ((info1 == info2)
                    || ((info1._base == info2._base)
                    && (info1.num == info2.num)
                    && (info1.den == info2.den)
                    && (info1.events == info2.events)));

        }

        static bool _signbit_set(NpyArray arr)
        {
            char bitmask = (char)0x80;
            VoidPtr ptr;  /* points to the byte to test */
            int ptr_index = 0;
            char byteorder;
            int elsize;

            elsize = arr.descr.elsize;
            byteorder = arr.descr.byteorder;
            ptr = new VoidPtr(arr);
            if (elsize > 1 &&
                (byteorder == NPY_LITTLE ||
                (byteorder == NPY_NATIVE && NpyArray_ISNBO(NPY_LITTLE))))
            {
                ptr_index += elsize - 1;
            }

            dynamic data = GetIndex(ptr, ptr_index);
            return ((data & bitmask) != 0);
        }


        private static NpyArray_Descr PyArray_Type = null;
        private static object NpyArray_GetSubType(int narrays, NpyArray[] arrays)
        {
            return null;
        }


    }
}
