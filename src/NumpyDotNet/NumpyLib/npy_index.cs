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
using System.Diagnostics;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif


namespace NumpyLib
{
    public enum NpyIndexType
    {
        NPY_INDEX_INTP,
        NPY_INDEX_BOOL,
        NPY_INDEX_SLICE_NOSTOP,
        NPY_INDEX_SLICE,
        NPY_INDEX_STRING,
        NPY_INDEX_BOOL_ARRAY,
        NPY_INDEX_INTP_ARRAY,
        NPY_INDEX_ELLIPSIS,
        NPY_INDEX_NEWAXIS,
    };

    /*
    * Structure for describing a slice without a stop.
    */
    public class NpyIndexSliceNoStop
    {
        public npy_intp start;
        public npy_intp step;
    }


    /*
    * Structure for describing a slice.
    */
    public class NpyIndexSlice
    {
        public npy_intp start;
        public npy_intp step;
        public npy_intp stop;
    }

    public class NpyIndex
    {
        public NpyIndex()
        {
            slice = new NpyIndexSlice();
            slice_nostop = new NpyIndexSliceNoStop();
        }
        public NpyIndexType type;
        public npy_intp intp;
        public bool boolean;
        public NpyIndexSlice slice;
        public NpyIndexSliceNoStop slice_nostop;
        public string _string;
        public NpyArray bool_array;
        public NpyArray intp_array;
    }

    internal partial class numpyinternal
    {
        internal static void NpyArray_IndexDealloc(NpyIndex[] indexes, int n)
        {
            int i;
            NpyIndex[] index = indexes;

            for (i = 0; i < n; i++)
            {
                switch (index[i].type)
                {
                    case NpyIndexType.NPY_INDEX_INTP_ARRAY:
                        Npy_DECREF(index[i].intp_array);
                        index[i].intp_array = null;
                        break;
                    case NpyIndexType.NPY_INDEX_BOOL_ARRAY:
                        Npy_DECREF(index[i].bool_array);
                        index[i].bool_array = null;
                        break;
                    default:
                        break;
                }
            }
        }

        /*
         * Returns the number of non-new indices.  Boolean arrays are
         * counted as if they are expanded.
         */
        static int count_non_new(NpyIndex[] indexes, int index_offset,  int n)
        {
            int i;
            int result = 0;

            for (i = index_offset; i < n; i++)
            {
                switch (indexes[i].type)
                {
                    case NpyIndexType.NPY_INDEX_NEWAXIS:
                        break;
                    case NpyIndexType.NPY_INDEX_BOOL_ARRAY:
                        result += indexes[i].bool_array.nd;
                        break;
                    default:
                        result++;
                        break;
                }
            }
            return result;
        }


        internal static int NpyArray_IndexExpandBool(NpyIndex[] indexes, int n, NpyIndex[] out_indexes)
        {
            int i;
            int result = 0;

            for (i = 0; i < n; i++)
            {
                switch (indexes[i].type)
                {
                    case NpyIndexType.NPY_INDEX_BOOL_ARRAY:
                        {
                            /* Convert to intp array on non-zero indexes. */
                            NpyArray []index_arrays = new NpyArray[npy_defs.NPY_MAXDIMS];
                            NpyArray bool_array = indexes[i].bool_array;
                            int j;

                            if (NpyArray_NonZero(bool_array, index_arrays, null) < 0)
                            {
                                NpyArray_IndexDealloc(out_indexes, result);
                                return -1;
                            }
                            for (j = 0; j < bool_array.nd; j++)
                            {
                                out_indexes[result].type = NpyIndexType.NPY_INDEX_INTP_ARRAY;
                                out_indexes[result].intp_array = index_arrays[j];
                                result++;
                            }
                        }
                        break;
                    case NpyIndexType.NPY_INDEX_INTP_ARRAY:
                        out_indexes[result++] = indexes[i];
                        Npy_INCREF(indexes[i].intp_array);
                        break;
                    case NpyIndexType.NPY_INDEX_BOOL:
                        out_indexes[result].type = NpyIndexType.NPY_INDEX_INTP;
                        out_indexes[result].intp = (npy_intp)(indexes[i].boolean == true ? 1 : 0);
                        result++;
                        break;
                    default:
                        /* Copy anything else. */
                        out_indexes[result++] = indexes[i];
                        break;
                }
            }

            return result;
        }


        /*
        * Converts indexes int out_indexes appropriate for an array by:
        *
        * 1. Expanding any ellipses.
        * 2. Setting slice start/stop/step appropriately for the array dims.
        * 3. Handling any negative indexes.
        * 4. Expanding any boolean arrays to intp arrays of non-zero indices.
        * 5. Convert any booleans to intp.
        *
        * Returns the number of indices in out_indexes, or -1 on error.
        */
        internal static int NpyArray_IndexBind(NpyIndex[] indexes, int n, npy_intp[] dimensions, int nd, NpyIndex[] out_indexes)
        {
            int i;
            int result = 0;
            int n_new = 0;

            for (i = 0; i < n; i++)
            {
                switch (indexes[i].type)
                {

                    case NpyIndexType.NPY_INDEX_STRING:
                        NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                         "String index not allowed.");
                        return -1;

                    case NpyIndexType.NPY_INDEX_ELLIPSIS:
                        {
                            /* Expand the ellipsis. */
                            int j, n2;
                            n2 = nd + n_new - count_non_new(indexes, i + 1, n - i - 1) - result;
                            if (n2 < 0)
                            {
                                NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                                 "too many indices");
                                NpyArray_IndexDealloc(out_indexes, result);
                                return -1;
                            }
                            /* Fill with full slices. */
                            for (j = 0; j < n2; j++)
                            {
                                NpyIndex _out = out_indexes[result];
                                _out.type = NpyIndexType.NPY_INDEX_SLICE;
                                _out.slice.start = 0;
                                _out.slice.stop = dimensions[result - n_new];
                                _out.slice.step = 1;
                                result++;
                            }
                        }
                        break;

                    case NpyIndexType.NPY_INDEX_BOOL_ARRAY:
                        {
                            /* Convert to intp array on non-zero indexes. */
                            NpyArray[] index_arrays = new NpyArray[npy_defs.NPY_MAXDIMS];
                            NpyArray bool_array = indexes[i].bool_array;
                            int j;

                            if (result + bool_array.nd >= nd + n_new)
                            {
                                NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                                 "too many indices");
                                NpyArray_IndexDealloc(out_indexes, result);
                                return -1;
                            }
                            if (NpyArray_NonZero(bool_array, index_arrays, null) < 0)
                            {
                                NpyArray_IndexDealloc(out_indexes, result);
                                return -1;
                            }
                            for (j = 0; j < bool_array.nd; j++)
                            {
                                out_indexes[result].type = NpyIndexType.NPY_INDEX_INTP_ARRAY;
                                out_indexes[result].intp_array = index_arrays[j];
                                result++;
                            }
                        }
                        break;

                    case NpyIndexType.NPY_INDEX_SLICE:
                        {
                            /* Sets the slice values based on the array. */
                            npy_intp dim;
                            NpyIndexSlice slice;

                            if (result >= nd + n_new)
                            {
                                NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                                 "too many indices");
                                NpyArray_IndexDealloc(out_indexes, result);
                                return -1;
                            }

                            dim = dimensions[result - n_new];
                            out_indexes[result].type = NpyIndexType.NPY_INDEX_SLICE;
                            out_indexes[result].slice = indexes[i].slice;
                            slice = out_indexes[result].slice;

                            if (slice.start < 0)
                            {
                                slice.start += dim;
                            }
                            if (slice.start < 0)
                            {
                                slice.start = 0;
                                if (slice.step < 0)
                                {
                                    slice.start -= 1;
                                }
                            }
                            if (slice.start >= dim)
                            {
                                slice.start = dim;
                                if (slice.step < 0)
                                {
                                    slice.start -= 1;
                                }
                            }

                            if (slice.stop < 0)
                            {
                                slice.stop += dim;
                            }

                            if (slice.stop < 0)
                            {
                                slice.stop = -1;
                            }
                            if (slice.stop > dim)
                            {
                                slice.stop = dim;
                            }

                            result++;
                        }
                        break;

                    case NpyIndexType.NPY_INDEX_SLICE_NOSTOP:
                        {
                            /* Sets the slice values based on the array. */
                            npy_intp dim;
                            NpyIndexSlice oslice;
                            NpyIndexSliceNoStop islice;

                            if (result >= nd + n_new)
                            {
                                NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                                 "too many indices");
                                NpyArray_IndexDealloc(out_indexes, result);
                                return -1;
                            }

                            dim = dimensions[result - n_new];
                            out_indexes[result].type = NpyIndexType.NPY_INDEX_SLICE;
                            oslice = out_indexes[result].slice;
                            islice = indexes[i].slice_nostop;

                            oslice.step = islice.step;

                            if (islice.start < 0)
                            {
                                oslice.start = islice.start + dim;
                            }
                            else
                            {
                                oslice.start = islice.start;
                            }
                            if (oslice.start < 0)
                            {
                                oslice.start = 0;
                                if (oslice.step < 0)
                                {
                                    oslice.start -= 1;
                                }
                            }
                            if (oslice.start >= dim)
                            {
                                oslice.start = dim;
                                if (oslice.step < 0)
                                {
                                    oslice.start -= 1;
                                }
                            }

                            if (oslice.step > 0)
                            {
                                oslice.stop = dim;
                            }
                            else
                            {
                                oslice.stop = -1;
                            }

                            result++;
                        }
                        break;

                    case NpyIndexType.NPY_INDEX_INTP:
                    case NpyIndexType.NPY_INDEX_BOOL:
                        {
                            npy_intp val, dim;

                            if (result >= nd + n_new)
                            {
                                NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                                 "too many indices");
                                NpyArray_IndexDealloc(out_indexes, result);
                                return -1;
                            }

                            if (indexes[i].type == NpyIndexType.NPY_INDEX_INTP)
                            {
                                val = indexes[i].intp;
                            }
                            else
                            {
                                val = (npy_intp)(indexes[i].boolean == true ? 1 : 0);
                            }
                            dim = dimensions[result - n_new];

                            if (val < 0)
                            {
                                val += dim;
                            }
                            if (val < 0 || val >= dim)
                            {
                                NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                                 "Invalid index.");
                                return -1;
                            }

                            out_indexes[result].type = NpyIndexType.NPY_INDEX_INTP;
                            out_indexes[result].intp = val;
                            result++;
                        }
                        break;


                    case NpyIndexType.NPY_INDEX_INTP_ARRAY:
                        if (result >= nd + n_new)
                        {
                            NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                             "too many indices");
                            NpyArray_IndexDealloc(out_indexes, result);
                            return -1;
                        }

                        out_indexes[result++] = indexes[i];
                        Npy_INCREF(indexes[i].intp_array);
                        break;

                    case NpyIndexType.NPY_INDEX_NEWAXIS:
                        n_new++;
                        out_indexes[result++] = indexes[i];
                        break;

                    default:
                        /* Copy anything else. */
                        out_indexes[result++] = indexes[i];
                        break;
                }
            }

            return result;
        }

        internal static int NpyArray_IndexToDimsEtc(NpyArray array, NpyIndex[] indexes, int n,
                            npy_intp[] dimensions, npy_intp[] strides, ref npy_intp offset_ptr, bool allow_arrays)
        {
            int i;
            int iDim = 0;
            int nd_new = 0;
            npy_intp offset = 0;

            for (i = 0; i < n; i++)
            {
                switch (indexes[i].type)
                {
                    case NpyIndexType.NPY_INDEX_INTP:
                        if (iDim >= array.nd)
                        {
                            NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                             "too many indices");
                            return -1;
                        }
                        offset += array.strides[iDim] * indexes[i].intp;
                        iDim++;
                        break;
                    case NpyIndexType.NPY_INDEX_SLICE:
                        {
                            NpyIndexSlice slice = indexes[i].slice;

                            if (iDim >= array.nd)
                            {
                                NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                                 "too many indices");
                                return -1;
                            }
                            dimensions[nd_new] = NpyArray_SliceSteps(slice);
                            strides[nd_new] = slice.step * array.strides[iDim];
                            offset += array.strides[iDim] * slice.start;
                            iDim++;
                            nd_new++;
                        }
                        break;

                    case NpyIndexType.NPY_INDEX_INTP_ARRAY:
                        if (allow_arrays)
                        {
                            /* Treat arrays as a 0 index to get the subspace. */
                            if (iDim >= array.nd)
                            {
                                NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                                 "too many indices");
                                return -1;
                            }
                            iDim++;
                            break;
                        }
                        else
                        {
                            NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                             "Array indices are not allowed.");
                            return -1;
                        }

                    case NpyIndexType.NPY_INDEX_NEWAXIS:
                        dimensions[nd_new] = 1;
                        strides[nd_new] = 0;
                        nd_new++;
                        break;

                    case NpyIndexType.NPY_INDEX_SLICE_NOSTOP:
                    case NpyIndexType.NPY_INDEX_BOOL_ARRAY:
                    case NpyIndexType.NPY_INDEX_ELLIPSIS:
                        NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                         "Index is not bound to an array.");
                        return -1;
                    case NpyIndexType.NPY_INDEX_STRING:
                        NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                         "String indices not allowed.");
                        return -1;

                    default:
                        Debug.Assert(false);
                        NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                         "Illegal index type.");
                        return -1;
                }
            }

            /* Add full slices for the rest of the array indices. */
            for (; iDim < array.nd; iDim++)
            {
                dimensions[nd_new] = array.dimensions[iDim];
                strides[nd_new] = array.strides[iDim];
                nd_new++;
            }

            offset_ptr = offset;
            return nd_new;
        }

        internal static npy_intp NpyArray_SliceSteps(NpyIndexSlice slice)
        {
            if ((slice.step < 0 && slice.stop >= slice.start) || (slice.step > 0 && slice.start >= slice.stop))
            {
                return 0;
            }
            else if (slice.step < 0)
            {
                return ((slice.stop - slice.start + 1) / slice.step) + 1;
            }
            else
            {
                return ((slice.stop - slice.start - 1) / slice.step) + 1;
            }
        }

    }




}
