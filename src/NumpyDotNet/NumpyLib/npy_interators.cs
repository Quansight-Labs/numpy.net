﻿/*
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
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
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
        internal static NpyTypeObject NpyArrayMultiIter_Type = new NpyTypeObject()
        {
            ntp_dealloc = arraymultiter_dealloc,
            ntp_interface_alloc = null,
        };

        internal static NpyTypeObject NpyArrayIter_Type = new NpyTypeObject()
        {
            ntp_dealloc = arraymultiter_dealloc,
            ntp_interface_alloc = null,
        };

        //internal static NpyTypeObject NpyArrayNeighborhoodIter_Type = new NpyTypeObject()
        //{
        //    ntp_dealloc = null,
        //    ntp_interface_alloc = null,
        //};


        /* get the dataptr from its current coordinates for simple iterator */
        static VoidPtr get_ptr_simple(NpyArrayIterObject iter, npy_intp[] coordinates)
        {
            VoidPtr ret = new VoidPtr(iter.ao);

            for (int i = 0; i < iter.ao.nd; ++i)
            {
                ret.data_offset += coordinates[i] * iter.strides[i];
            }

            return ret;
        }

        static void arraymultiter_dealloc(object obj)
        {
            NpyArrayMultiIterObject multi = obj as NpyArrayMultiIterObject;

            Debug.Assert(0 == multi.nob_refcnt);

            for (int i = 0; i < multi.numiter; i++)
            {
                Npy_XDECREF(multi.iters[i]);
            }
            multi.nob_magic_number = npy_defs.NPY_INVALID_MAGIC;
            NpyArray_free(multi);
        }


        internal static VoidPtr NpyArray_IterNext(NpyArrayIterObject it)
        {
            VoidPtr result = null;

            if (it.index < it.size-1)
            {
                result = it.dataptr;
                NpyArray_ITER_NEXT(it);
            }
            return result;
        }

        // Resets the iterator to the first element in the array.

        internal static void NpyArray_IterReset(NpyArrayIterObject it)
        {
            NpyArray_ITER_RESET(it);
        }

        /*
        * Get Iterator.
        */
        internal static NpyArrayIterObject NpyArray_IterNew(NpyArray ao)
        {
            NpyArrayIterObject it = new NpyArrayIterObject();

            NpyObject_Init(it, new NpyTypeObject());
            if (it == null)
            {
                return null;
            }

            array_iter_base_init(it, ao);
            /* Defer creation of the wrapper - will be handled by Npy_INTERFACE. */
            return it;
        }
 

        internal static NpyArrayIterObject NpyArray_BroadcastToShape(NpyArray ao, npy_intp[] dims, int nd)
        {
            NpyArrayIterObject it;
            int i, diff, j, k;
            bool compat;

  

            // if this is a scalar with no dimensions, let's fake 1 dimension
            if (dims == null || nd <= 0)
            {
                dims = new npy_intp[1] { 1 };
                nd = 1;
            }

            if (ao.nd > nd)
            {
                goto err;
            }
            compat = true;
            diff = j = nd - ao.nd;
            for (i = 0; i < ao.nd; i++, j++)
            {
                if (ao.dimensions[i] == 1)
                {
                    continue;
                }
                if (ao.dimensions[i] != dims[j])
                {
                    compat = false;
                    break;
                }
            }
            if (!compat)
            {
                goto err;
            }
            it = new NpyArrayIterObject();
            if (it == null)
            {
                return null;
            }
            NpyObject_Init(it, NpyArrayIter_Type);
  
            NpyArray_UpdateFlags(ao, NPYARRAYFLAGS.NPY_CONTIGUOUS);
            if (NpyArray_ISCONTIGUOUS(ao))
            {
                it.contiguous = true;
            }
            else
            {
                it.contiguous = false;
            }
            Npy_INCREF(ao);
            it.ao = ao;
            it.elsize = ao.descr.elsize;
            it.size = NpyArray_MultiplyList(dims, nd);
            it.nd_m1 = nd - 1;
            it.factors[nd - 1] = 1;
            for (i = 0; i < nd; i++)
            {
                it.dims_m1[i] = dims[i] - 1;
                k = i - diff;
                if ((k < 0) || ao.dimensions[k] != dims[i])
                {
                    it.contiguous = false;
                    it.strides[i] = 0;
                }
                else
                {
                    it.strides[i] = ao.strides[k];
                }
                it.backstrides[i] = it.strides[i] * it.dims_m1[i];
                if (i > 0)
                {
                    it.factors[nd - i - 1] = it.factors[nd - i] * dims[nd - i];
                }
            }
            NpyArray_ITER_RESET(it);
            return it;

            err:
            NpyErr_SetString(npyexc_type.NpyExc_ValueError, "array is not broadcastable to correct shape");
            return null;

        }

        internal static NpyArray NpyArray_IterSubscript(NpyArrayIterObject self, NpyIndex []indexes, int n)
        {
            if (n == 0 || (n == 1 && indexes[0].type == NpyIndexType.NPY_INDEX_ELLIPSIS))
            {
                Npy_INCREF(self.ao);
                return self.ao;
            }

            if (n > 1)
            {
                NpyErr_SetString(npyexc_type.NpyExc_IndexError, "unsupported iterator index.");
                return null;
            }

            switch (indexes[0].type)
            {

                case NpyIndexType.NPY_INDEX_BOOL:
                    return NpyArray_IterSubscriptBool(self, indexes[0].boolean);

                case NpyIndexType.NPY_INDEX_INTP:
                    /* Return a 0-d array with the value at the index. */
                    return NpyArray_IterSubscriptIntp(self, indexes[0].intp);

                case NpyIndexType.NPY_INDEX_SLICE_NOSTOP:
                case NpyIndexType.NPY_INDEX_SLICE:
                    {
                        NpyIndex []new_index = new NpyIndex[1] { new NpyIndex() };

                        npy_intp[] newSize = new npy_intp[1] { self.size };

                        /* Bind the slice. */
                        if (NpyArray_IndexBind(indexes, 1,
                                               newSize, 1,
                                               new_index) < 0)
                        {
                            return null;
                        }

                        Debug.Assert(new_index[0].type == NpyIndexType.NPY_INDEX_SLICE);

                        self.size = newSize[0];

                        return NpyArray_IterSubscriptSlice(self, new_index[0].slice);
                    }

                case NpyIndexType.NPY_INDEX_BOOL_ARRAY:
                    return NpyArray_IterSubscriptBoolArray(self, indexes[0].bool_array);

                case NpyIndexType.NPY_INDEX_INTP_ARRAY:
                    return NpyArray_IterSubscriptIntpArray(self, indexes[0].intp_array);

                case NpyIndexType.NPY_INDEX_NEWAXIS:
                case NpyIndexType.NPY_INDEX_ELLIPSIS:
                    NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                     "cannot use Ellipsis or newaxes here");
                    return null;

                default:
                    NpyErr_SetString(npyexc_type.NpyExc_IndexError, "unsupported iterator index");
                    return null;
            }
        }

        internal static int NpyArray_IterSubscriptAssign(NpyArrayIterObject self, NpyIndex []indexes, int n, NpyArray value)
        {
            NpyIndex index;

            if (n > 1)
            {
                NpyErr_SetString(npyexc_type.NpyExc_IndexError, "unsupported iterator index.");
                return -1;
            }

            if (n == 0 || (n == 1 && indexes[0].type == NpyIndexType.NPY_INDEX_ELLIPSIS))
            {
                /* Assign to the whole iter using a slice. */
                NpyIndexSlice slice = new NpyIndexSlice();

                slice.start = 0;
                slice.stop = self.size;
                slice.step = 1;
                return NpyArray_IterSubscriptAssignSlice(self, slice, value);
            }

            index = indexes[0];

            switch (index.type)
            {

                case NpyIndexType.NPY_INDEX_BOOL:
                    if (index.boolean)
                    {
                        return NpyArray_IterSubscriptAssignIntp(self, 0, value);
                    }
                    else
                    {
                        return 0;
                    }

                case NpyIndexType.NPY_INDEX_INTP:
                    return NpyArray_IterSubscriptAssignIntp(self, index.intp, value);

                case NpyIndexType.NPY_INDEX_SLICE:
                case NpyIndexType.NPY_INDEX_SLICE_NOSTOP:
                    {
                        npy_intp[] new_size = new npy_intp[1] { self.size }; 
                        NpyIndex []new_index = new NpyIndex[1] { new NpyIndex() };

                        /* Bind the slice. */
                        if (NpyArray_IndexBind(indexes, 1,
                                               new_size, 1,
                                               new_index) < 0)
                        {
                            return -1;
                        }
                        Debug.Assert(new_index[0].type == NpyIndexType.NPY_INDEX_SLICE);

                        self.size = new_size[0];
                        return NpyArray_IterSubscriptAssignSlice(self, new_index[0].slice, value);
                    }

                case NpyIndexType.NPY_INDEX_BOOL_ARRAY:
                    return NpyArray_IterSubscriptAssignBoolArray(self,
                                                                 index.bool_array,
                                                                 value);

                case NpyIndexType.NPY_INDEX_INTP_ARRAY:
                    return NpyArray_IterSubscriptAssignIntpArray(self,
                                                                 index.intp_array,
                                                                 value);

                case NpyIndexType.NPY_INDEX_NEWAXIS:
                case NpyIndexType.NPY_INDEX_ELLIPSIS:
                    NpyErr_SetString(npyexc_type.NpyExc_IndexError,
                                     "cannot use Ellipsis or newaxes here");
                    return -1;

                default:
                    NpyErr_SetString(npyexc_type.NpyExc_IndexError, "unsupported iterator index");
                    return -1;
            }

        }

        internal static int NpyArray_Broadcast(NpyArrayMultiIterObject mit)
        {
            int i, nd, k, j;
            npy_intp tmp;
            NpyArrayIterObject it;

            /* Discover the broadcast number of dimensions */
            for (i = 0, nd = 0; i < mit.numiter; i++)
            {
                nd = Math.Max(nd, mit.iters[i].ao.nd);
            }
            mit.nd = nd;

            /* Discover the broadcast shape in each dimension */
            for (i = 0; i < nd; i++)
            {
                mit.dimensions[i] = 1;
                for (j = 0; j < mit.numiter; j++)
                {
                    it = mit.iters[j];
                    /* This prepends 1 to shapes not already equal to nd */
                    k = i + it.ao.nd - nd;
                    if (k >= 0)
                    {
                        tmp = it.ao.dimensions[k];
                        if (tmp == 1)
                        {
                            continue;
                        }
                        if (mit.dimensions[i] == 1)
                        {
                            mit.dimensions[i] = tmp;
                        }
                        else if (mit.dimensions[i] != tmp)
                        {
                            NpyErr_SetString(npyexc_type.NpyExc_ValueError, "shape mismatch: objects cannot be broadcast to a single shape");
                            return -1;
                        }
                    }
                }
            }

            /*
             * Reset the iterator dimensions and strides of each iterator
             * object -- using 0 valued strides for broadcasting
             * Need to check for overflow
             */
            tmp = NpyArray_OverflowMultiplyList(mit.dimensions, mit.nd);
            if (tmp == npy_intp.MaxValue)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "broadcast dimensions too large.");
                return -1;
            }
            mit.size = tmp;
            for (i = 0; i < mit.numiter; i++)
            {
                it = mit.iters[i];
                it.nd_m1 = mit.nd - 1;
                it.size = tmp;
                nd = it.ao.nd;

                if (mit.nd != 0)
                {
                    it.factors[mit.nd - 1] = 1;
                }
                for (j = 0; j < mit.nd; j++)
                {
                    it.dims_m1[j] = mit.dimensions[j] - 1;
                    k = j + nd - mit.nd;
                    /*
                     * If this dimension was added or shape of
                     * underlying array was 1
                     */
                    if ((k < 0) ||
                        it.ao.dimensions[k] != mit.dimensions[j])
                    {
                        it.contiguous = false;
                        it.strides[j] = 0;
                    }
                    else
                    {
                        it.strides[j] = it.ao.strides[k];
                    }
                    it.backstrides[j] = it.strides[j] * it.dims_m1[j];
                    if (j > 0)
                        it.factors[mit.nd - j - 1] =
                            it.factors[mit.nd - j] * mit.dimensions[mit.nd - j];
                }
                NpyArray_ITER_RESET(it);
            }
            return 0;
        }

        internal static NpyArrayMultiIterObject NpyArray_MultiIterNew()
        {
            NpyArrayMultiIterObject ret;

            ret = new NpyArrayMultiIterObject();
            if (null == ret)
            {
                NpyErr_MEMORY();
                return null;
            }
            NpyObject_Init(ret, NpyArrayMultiIter_Type);
            /* Defer creation of the wrapper - will be handled by Npy_INTERFACE. */
            return ret;
        }




        internal static void NpyArray_ITER_CACHE(NpyArrayIterObject it, npy_intp cacheSize)
        {
            if (cacheSize == 0)
                return;

            it.internalCacheLength = Math.Min(cacheSize, numpyinternal.maxIterOffsetCacheSize);
            if (it.internalCache == null || it.internalCache.Length < it.internalCacheLength)
            {
                it.internalCache = new npy_intp[it.internalCacheLength];
            }
            it.internalCacheIndex = 0;

            if (it.nd_m1 == 0)
            {
                npy_intp it_index = it.index;
                npy_intp it_dataptr_data_offset = it.dataptr.data_offset;
                npy_intp it_strides_0 = it.strides[0];
                npy_intp it_coordinates_0 = it.coordinates[0];

                for (int i = 0; i < it.internalCacheLength; i++)
                {
                    it.internalCache[i] = it_dataptr_data_offset;

                    it_index++;

                    it_dataptr_data_offset += it_strides_0;
                    it_coordinates_0++;
                }

                it.index = it_index;
                it.dataptr.data_offset = it_dataptr_data_offset;
                it.strides[0] = it_strides_0;
                it.coordinates[0] = it_coordinates_0;

                return;
            }

            if (it.contiguous)
            {
                npy_intp it_index = it.index;
                npy_intp it_dataptr_data_offset = it.dataptr.data_offset;
                int elsize = it.elsize;

                for (int i = 0; i < it.internalCacheLength; i++)
                {
                    it.internalCache[i] = it_dataptr_data_offset;

                    it_index++;

                    it_dataptr_data_offset += elsize;
                }

                it.index = it_index;
                it.dataptr.data_offset = it_dataptr_data_offset;
                return;
            }

            if (it.nd_m1 == 1)
            {
                npy_intp it_coordinates_1 = it.coordinates[1];
                npy_intp it_dims_m1_1 = it.dims_m1[1];
                npy_intp it_strides_0 = it.strides[0];
                npy_intp it_strides_1 = it.strides[1];
                npy_intp it_coordinates_0 = it.coordinates[0];
                npy_intp it_backstrides_1 = it.backstrides[1];
                npy_intp it_dataptr_data_offset = it.dataptr.data_offset;
                npy_intp it_index = it.index;

                for (int j = 0; j < it.internalCacheLength; j++)
                {
                    it.internalCache[j] = it_dataptr_data_offset;

                    it_index++;

                    if (it_coordinates_1 < it.dims_m1[1])
                    {
                        it_coordinates_1++;
                        it_dataptr_data_offset += it_strides_1;
                    }
                    else
                    {
                        it_coordinates_1 = 0;
                        it_coordinates_0++;
                        it_dataptr_data_offset += it_strides_0 - it_backstrides_1;
                    }
                }

                it.coordinates[1] = it_coordinates_1;
                it.dims_m1[1] = it_dims_m1_1;
                it.strides[0] = it_strides_0;
                it.strides[1] = it_strides_1;
                it.coordinates[0] = it_coordinates_0;
                it.backstrides[1] = it_backstrides_1;
                it.dataptr.data_offset = it_dataptr_data_offset;
                it.index = it_index;

                return;
            }
            else
            {
                npy_intp it_dataptr_data_offset = it.dataptr.data_offset;
                npy_intp it_index = it.index;

                for (int j = 0; j < it.internalCacheLength; j++)
                {
                    it.internalCache[j] = it_dataptr_data_offset;

                    it_index++;

                    for (int i = it.nd_m1; i >= 0; i--)
                    {
                        if (it.coordinates[i] < it.dims_m1[i])
                        {
                            it.coordinates[i]++;
                            it_dataptr_data_offset += it.strides[i];
                            break;
                        }
                        else
                        {
                            it.coordinates[i] = 0;
                            it_dataptr_data_offset -= it.backstrides[i];
                        }
                    }
                }

                it.dataptr.data_offset = it_dataptr_data_offset;
                it.index = it_index;

                return;

            }
        }


        internal static NpyArrayIterObject NpyArray_ITER_ConvertToIndex(NpyArrayIterObject it, int ItemDiv)
        {
            it.elsize >>= ItemDiv;
            it.dataptr.data_offset >>= ItemDiv;
            it.ao_offset = it.ao.data.data_offset >> ItemDiv;

            for (int i = 0; i <= it.nd_m1; i++)
            {
                it.strides[i] >>= ItemDiv;
                it.backstrides[i] >>= ItemDiv;
            }
  
            return it;
        }


        internal static NpyArrayMapIterObject NpyArray_ITER_ConvertToIndex(NpyArrayMapIterObject mit)
        {
            for (int i = 0; i < mit.numiter; i++)
            {
                var it = mit.iters[i];
                mit.iters[i] = NpyArray_ITER_ConvertToIndex(it, it.ao.ItemDiv);
            }
            if (mit.subspace != null)
            {
                mit.subspace = NpyArray_ITER_ConvertToIndex(mit.subspace, mit.subspace.ao.ItemDiv);
            }

            return mit;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_NEXT(NpyArrayIterObject it)
        {
            //Debug.Assert(Validate(it));

            it.index++;
            if (it.nd_m1 == 0)
            {
                it.dataptr.data_offset += it.strides[0];
                it.coordinates[0]++;
            }
            else if (it.contiguous)
            {
                it.dataptr.data_offset += it.elsize;
            }
            else if (it.nd_m1 == 1)
            {
                if (it.coordinates[1] < it.dims_m1[1])
                {
                    it.coordinates[1]++;
                    it.dataptr.data_offset += it.strides[1];
                }
                else
                {
                    it.coordinates[1] = 0;
                    it.coordinates[0]++;
                    it.dataptr.data_offset += it.strides[0] - it.backstrides[1];
                }
            }
            else
            {
                int i;
                for (i = it.nd_m1; i >= 0; i--)
                {
                    if (it.coordinates[i] < it.dims_m1[i])
                    {
                        it.coordinates[i]++;
                        it.dataptr.data_offset += it.strides[i];
                        break;
                    }
                    else
                    {
                        it.coordinates[i] = 0;
                        it.dataptr.data_offset -= it.backstrides[i];
                    }
                }
            }
        }

        internal static void NpyArray_ITER_WALK(NpyArrayIterObject it, npy_intp walkCount)
        {
            //Debug.Assert(Validate(it));

            it.index += walkCount;

            if (it.nd_m1 == 0)
            {
                it.dataptr.data_offset += walkCount * it.strides[0];
                it.coordinates[0] += walkCount;

                return;
            }

            if (it.contiguous)
            {
                it.dataptr.data_offset += walkCount * it.elsize;
                return;
            }

            if (it.nd_m1 == 1)
            {
                npy_intp it_coordinates_1 = it.coordinates[1];
                npy_intp it_dims_m1_1 = it.dims_m1[1];
                npy_intp it_strides_0 = it.strides[0];
                npy_intp it_strides_1 = it.strides[1];
                npy_intp it_coordinates_0 = it.coordinates[0];
                npy_intp it_backstrides_1 = it.backstrides[1];
                npy_intp it_dataptr_data_offset = it.dataptr.data_offset;

                while (walkCount-- > 0)
                {
                    if (it_coordinates_1 < it_dims_m1_1)
                    {
                        it_coordinates_1++;
                        it_dataptr_data_offset += it_strides_1;
                    }
                    else
                    {
                        it_coordinates_1 = 0;
                        it_coordinates_0++;
                        it_dataptr_data_offset += it_strides_0 - it_backstrides_1;
                    }
                }

                it.coordinates[1] = it_coordinates_1;
                it.dims_m1[1] = it_dims_m1_1;
                it.strides[0] = it_strides_0;
                it.strides[1] = it_strides_1;
                it.coordinates[0] = it_coordinates_0;
                it.backstrides[1] = it_backstrides_1;
                it.dataptr.data_offset = it_dataptr_data_offset;

            }
            else
            {
                npy_intp it_dataptr_data_offset = it.dataptr.data_offset;

                while (walkCount-- > 0)
                {
                    for (int i = it.nd_m1; i >= 0; i--)
                    {
                        if (it.coordinates[i] < it.dims_m1[i])
                        {
                            it.coordinates[i]++;
                            it_dataptr_data_offset += it.strides[i];
                            break;
                        }
                        else
                        {
                            it.coordinates[i] = 0;
                            it_dataptr_data_offset -= it.backstrides[i];
                        }
                    }
                }

                it.dataptr.data_offset = it_dataptr_data_offset;

            }

            return;
        }

        internal static (IEnumerable<NpyArrayIterObject>, IEnumerable<NpyArrayIterObject>) NpyArray_ITER_ParallelSplit(NpyArrayIterObject destIter, NpyArrayIterObject srcIter)
        {
            npy_intp TotalSize = destIter.size - destIter.index;
            NpyArrayIterObject[] DestIters = null;
            NpyArrayIterObject[] SrcIters = null;

            if (TotalSize < 2 || maxParallelIterators == 1)
            {
                DestIters = new NpyArrayIterObject[1];
                SrcIters = new NpyArrayIterObject[1];
            }
            else
            if (TotalSize < 4 || maxParallelIterators == 2)
            {
                DestIters = new NpyArrayIterObject[2];
                SrcIters = new NpyArrayIterObject[2];
            }
            else
            if (TotalSize < 8 || maxParallelIterators == 4)
            {
                DestIters = new NpyArrayIterObject[4];
                SrcIters = new NpyArrayIterObject[4];
            }
            else
            if (TotalSize < 16 || maxParallelIterators == 8)
            {
                DestIters = new NpyArrayIterObject[8];
                SrcIters = new NpyArrayIterObject[8];
            }
            else
            {
                DestIters = new NpyArrayIterObject[16];
                SrcIters = new NpyArrayIterObject[16];
            }


            var taskSize = destIter.size / DestIters.Length;
            DestIters[0] = destIter.copy();
            DestIters[0].size = taskSize;
            SrcIters[0] = srcIter.copy();
 
            for (int Index = 1; Index < DestIters.Length; Index++)
            {
                DestIters[Index] = DestIters[Index-1].copy();
                DestIters[Index].size += taskSize;
                SrcIters[Index] = SrcIters[Index-1].copy();

                npy_intp walkCount = DestIters[Index - 1].size - DestIters[Index].index;
                numpyinternal.NpyArray_ITER_WALK(DestIters[Index], walkCount);
                numpyinternal.NpyArray_ITER_WALK(SrcIters[Index], walkCount);
            }
            DestIters[DestIters.Length - 1].size = destIter.size;

            return (DestIters, SrcIters);
        }

        internal static IEnumerable<NpyArrayIterObject> NpyArray_ITER_ParallelSplit(NpyArrayIterObject it, npy_intp SingleIterSize = -1)
        {
            npy_intp TotalSize = it.size - it.index;
            NpyArrayIterObject[] DestIters = null;
   

            if (TotalSize <= SingleIterSize || maxParallelIterators == 1)
            {
                DestIters = new NpyArrayIterObject[1];
            }
            else
            if (TotalSize <= (SingleIterSize * 2) || maxParallelIterators == 2)
            {
                DestIters = new NpyArrayIterObject[2];
            }
            else
            if (TotalSize <= (SingleIterSize * 4) || maxParallelIterators == 4)
            {
                DestIters = new NpyArrayIterObject[4];
            }
            else
            if (TotalSize <= (SingleIterSize * 8) || maxParallelIterators == 8)
            {
                DestIters = new NpyArrayIterObject[8];
            }
            else
            {
                DestIters = new NpyArrayIterObject[16];
            }


            var taskSize = TotalSize / DestIters.Length;
            if (taskSize == 0)
            {
                DestIters = new NpyArrayIterObject[1];
                taskSize = TotalSize / DestIters.Length;
            }

            DestIters[0] = it.copy();
            DestIters[0].size = taskSize;

            for (int Index = 1; Index < DestIters.Length; Index++)
            {
                DestIters[Index] = DestIters[Index - 1].copy();
                DestIters[Index].size += taskSize;

                numpyinternal.NpyArray_ITER_WALK(DestIters[Index], DestIters[Index - 1].size - DestIters[Index].index);
            }
            DestIters[DestIters.Length - 1].size = it.size;

            return DestIters;
        }

        internal class LoopSegment
        {
            public npy_intp start;
            public npy_intp end;
        }

        internal static IEnumerable<LoopSegment> NpyArray_SEGMENT_ParallelSplit(npy_intp loopCount, npy_intp SingleIterSize = -1)
        {
            npy_intp TotalSize = loopCount;
            LoopSegment[] segments = null;

            if (TotalSize <= SingleIterSize || maxParallelIterators == 1)
            {
                segments = new LoopSegment[1];
            }
            else
            if (TotalSize <= (SingleIterSize * 2) || maxParallelIterators == 2)
            {
                segments = new LoopSegment[2];
            }
            else
            if (TotalSize <= (SingleIterSize * 4) || maxParallelIterators == 4)
            {
                segments = new LoopSegment[4];
            }
            else
            if (TotalSize <= (SingleIterSize * 8) || maxParallelIterators == 8)
            {
                segments = new LoopSegment[8];
            }
            else
            {
                segments = new LoopSegment[16];
            }

            for (int i = 0; i < segments.Length; i++)
            {
                segments[i] = new LoopSegment();
            }

            var taskSize = TotalSize / segments.Length;
            segments[0].start = 0;
            segments[0].end = taskSize;

            for (int Index = 1; Index < segments.Length; Index++)
            {
                segments[Index].start = segments[Index - 1].end;
                segments[Index].end = segments[Index].start + taskSize;

            }
            segments[segments.Length - 1].end = TotalSize;

            return segments;
        }

 
        internal static NpyArrayIterObject NpyArray_IterAllButAxis(NpyArray obj, ref int inaxis)
        {
            NpyArrayIterObject it;
            int axis;
            it = NpyArray_IterNew(obj);
            if (it == null)
            {
                return null;
            }
            if (NpyArray_NDIM(obj) == 0)
            {
                return it;
            }
            if (inaxis < 0)
            {
                int i, minaxis = 0;
                npy_intp minstride = 0;
                i = 0;
                while (minstride == 0 && i < NpyArray_NDIM(obj))
                {
                    minstride = NpyArray_STRIDE(obj, i);
                    i++;
                }
                for (i = 1; i < NpyArray_NDIM(obj); i++)
                {
                    if (NpyArray_STRIDE(obj, i) > 0 &&
                        NpyArray_STRIDE(obj, i) < minstride)
                    {
                        minaxis = i;
                        minstride = NpyArray_STRIDE(obj, i);
                    }
                }
                inaxis = minaxis;
            }
            axis = inaxis;
            /* adjust so that will not iterate over axis */
            it.contiguous = false;
            if (it.size != 0)
            {
                it.size /= NpyArray_DIM(obj, axis);
            }
            it.dims_m1[axis] = 0;
            it.backstrides[axis] = 0;

            /*
             * (won't fix factors so don't use
             * NpyArray_ITER_GOTO1D with this iterator)
             */
            return it;
        }

        /*
         * Get MultiIterator from array of Python objects and any additional
         *
         * NpyArray **mps -- array of NpyArrays
         * int n - number of NpyArrays in the array
         * int nadd - number of additional arrays to include in the iterator.
         *
         * Returns a multi-iterator object.
         */
        internal static NpyArrayMultiIterObject NpyArray_MultiIterFromArrays(NpyArray[] mps, int n, int nadd, params object[] va)
        {
            var result = NpyArray_vMultiIterFromArrays(mps, n, nadd, va);
            return result;
        }

        internal static NpyArrayMultiIterObject NpyArray_vMultiIterFromArrays(NpyArray[] mps, int n, int nadd, params object[] va)
        {
            NpyArrayMultiIterObject multi;
            NpyArray current;
            int i, ntot;
            bool err = false;

            ntot = n + nadd;
            if (ntot < 2 || ntot > npy_defs.NPY_MAXARGS)
            {
                var msg = string.Format("Need between 2 and({0}) array objects(inclusive).", npy_defs.NPY_MAXARGS);
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                return null;
            }
            multi = new NpyArrayMultiIterObject();
            if (multi == null)
            {
                NpyErr_MEMORY();
                return null;
            }
            NpyObject_Init(multi, NpyArrayMultiIter_Type);

            for (i = 0; i < ntot; i++)
            {
                multi.iters[i] = null;
            }
            multi.numiter = ntot;
            multi.index = 0;

            for (i = 0; i < ntot; i++)
            {
                if (i < n)
                {
                    current = mps[i];
                }
                else
                {
                    current = (NpyArray)va[i-n];
                }
                multi.iters[i] = NpyArray_IterNew(current);
            }

            if (!err && NpyArray_Broadcast(multi) < 0)
            {
                err = true;
            }
            if (err)
            {
                Npy_DECREF(multi);
                return null;
            }
            NpyArray_MultiIter_RESET(multi);
            /* Defer creation of the wrapper - will be handled by Npy_INTERFACE. */
            return multi;
        }

        internal static int NpyArray_RemoveSmallest(NpyArrayMultiIterObject multi)
        {
            NpyArrayIterObject it;
            int i, j;
            int axis;
            npy_intp smallest;
            npy_intp []sumstrides = new npy_intp[npy_defs.NPY_MAXDIMS];

            if (multi.nd == 0)
            {
                return -1;
            }
            for (i = 0; i < multi.nd; i++)
            {
                sumstrides[i] = 0;
                for (j = 0; j < multi.numiter; j++)
                {
                    sumstrides[i] += multi.iters[j].strides[i];
                }
            }
            axis = 0;
            smallest = sumstrides[0];
            /* Find longest dimension */
            for (i = 1; i < multi.nd; i++)
            {
                if (sumstrides[i] < smallest)
                {
                    axis = i;
                    smallest = sumstrides[i];
                }
            }
            for (i = 0; i < multi.numiter; i++)
            {
                it = multi.iters[i];
                it.contiguous = false;
                if (it.size != 0)
                {
                    it.size /= (it.dims_m1[axis] + 1);
                }
                it.dims_m1[axis] = 0;
                it.backstrides[axis] = 0;
            }
            multi.size = multi.iters[0].size;
            return axis;
        }

 

        internal static void NpyArray_MultiIter_NEXT(NpyArrayMultiIterObject multi)
        {
            Debug.Assert(Validate(multi));
            multi.index++;

            for (int i = 0; i < multi.numiter; i++)
            {
                NpyArray_ITER_NEXT(multi.iters[i]);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_RESET(NpyArrayIterObject it, int ItemDiv = 0)
        {
            Debug.Assert(Validate(it));
            it.index = 0;
            if (it.dataptr.datap != null)
            {
                it.dataptr.data_offset = it.ao.data.data_offset;
            }
            else
            {
                it.dataptr = new VoidPtr(it.ao);
            }

            it.dataptr.data_offset >>= ItemDiv;
            Array.Clear(it.coordinates, 0, it.coordinates.Length);
        }

        static int array_iter_base_init(NpyArrayIterObject it, NpyArray ao)
        {
            int nd, i;

            it.nob_interface = null;
            it.nob_magic_number = npy_defs.NPY_VALID_MAGIC;
            nd = ao.nd;
            NpyArray_UpdateFlags(ao, NPYARRAYFLAGS.NPY_CONTIGUOUS);
            if (NpyArray_ISCONTIGUOUS(ao))
            {
                it.contiguous = true;
            }
            else
            {
                it.contiguous = false;
            }
            Npy_INCREF(ao);
            it.ao = ao;
            it.elsize = ao.descr.elsize;
            it.size = NpyArray_SIZE(ao);
            it.nd_m1 = nd - 1;

            if (nd != 0)
            {
                it.factors[nd - 1] = 1;
            }
            for (i = 0; i < nd; i++)
            {
                it.dims_m1[i] = ao.dimensions[i] - 1;
                it.strides[i] = ao.strides[i];
                it.backstrides[i] = it.strides[i] * it.dims_m1[i];
                if (i > 0)
                {
                    it.factors[nd - i - 1] = it.factors[nd - i] * ao.dimensions[nd - i];
                }
                //it.bounds[i,0] = 0;
                //it.bounds[i,1] = ao.dimensions[i] - 1;
                //it.limits[i,0] = 0;
                //it.limits[i,1] = ao.dimensions[i] - 1;
                //it.limits_sizes[i] = it.limits[i,1] - it.limits[i,0] + 1;
            }

            NpyArray_ITER_RESET(it);

            return 0;
        }

        static NpyArray NpyArray_IterSubscriptBool(NpyArrayIterObject self, bool index)
        {
            NpyArray result;
            bool swap;


            NpyArray_ITER_RESET(self);

            if (index)
            {
                /* Returns a 0-d array with the value. */
                Npy_INCREF(self.ao.descr);
                result = NpyArray_Alloc(self.ao.descr, 0, null, false, Npy_INTERFACE(self.ao));
                if (result == null)
                {
                    return null;
                }

                swap = (NpyArray_ISNOTSWAPPED(self.ao) != NpyArray_ISNOTSWAPPED(result));

                MemCopy.GetMemcopyHelper(result.data).copyswap(result.data, self.dataptr, swap);
                return result;
            }
            else
            {
                /* Make an empty array. */
                npy_intp []ii = new npy_intp[1] { 0 };
                Npy_INCREF(self.ao.descr);
                result = NpyArray_Alloc(self.ao.descr, 7, ii, false, Npy_INTERFACE(self.ao));
                return result;
            }
        }

        internal static NpyArray NpyArray_IterSubscriptSlice(NpyArrayIterObject self, NpyIndexSlice slice)
        {
            NpyArray result;
            npy_intp[] steps = new npy_intp[1] { 0 };
            npy_intp start, step_size;
            bool swap;
            VoidPtr dptr;
 
            /* Build the result. */
            steps[0] = NpyArray_SliceSteps(slice);

            Npy_INCREF(self.ao.descr);
            result = NpyArray_Alloc(self.ao.descr, 1, steps,
                                    false, Npy_INTERFACE(self.ao));
            if (result == null)
            {
                return result;
            }

            /* Copy in the data. */
            start = slice.start;
            step_size = slice.step;
            swap = (NpyArray_ISNOTSWAPPED(self.ao) != NpyArray_ISNOTSWAPPED(result));
            dptr = new VoidPtr(result);

            NpyArray_ITER_RESET(self);

            var helper = MemCopy.GetMemcopyHelper(dptr);
            helper.IterSubscriptSlice(steps, self, dptr, start, step_size, swap);

            NpyArray_ITER_RESET(self);

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_GOTO1D(NpyArrayIterObject it, npy_intp indices)
        {
            Debug.Assert(Validate(it));

            if (indices < 0)
                indices += it.size;

            it.index = indices;
            if (it.nd_m1 == 0)
            {
                it.dataptr.data_offset = it.ao.data.data_offset + (indices * it.strides[0]);
            }
            else if (it.contiguous)
            {
                it.dataptr.data_offset = it.ao.data.data_offset + (indices * it.elsize);
            }
            else
            {
                it.dataptr.data_offset = it.ao.data.data_offset;
                for (int index = 0; index <= it.nd_m1; index++)
                {
                    it.dataptr.data_offset += (indices / it.factors[index]) * it.strides[index];
                    indices %= it.factors[index];
                }
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_GOTO1D_INDEX(NpyArrayIterObject it, npy_intp indices)
        {
            Debug.Assert(Validate(it));

            if (indices < 0)
                indices += it.size;

            it.index = indices;
            if (it.nd_m1 == 0)
            {
                it.dataptr.data_offset = it.ao_offset + (indices * it.strides[0]);
            }
            else if (it.contiguous)
            {
                it.dataptr.data_offset = it.ao_offset + (indices * it.elsize);
            }
            else
            {
                it.dataptr.data_offset = it.ao_offset;
                for (int index = 0; index <= it.nd_m1; index++)
                {
                    it.dataptr.data_offset += (indices / it.factors[index]) * it.strides[index];
                    indices %= it.factors[index];
                }
            }
        }

        internal static void NpyArray_ITER_GOTO1D_CACHE(NpyArrayIterObject it, npy_intp indices, npy_intp step, npy_intp cacheSize)
        {
            if (cacheSize == 0)
                return;

            it.internalCacheLength = Math.Min(cacheSize, numpyinternal.maxIterOffsetCacheSize);
            if (it.internalCache == null || it.internalCache.Length < it.internalCacheLength)
            {
                it.internalCache = new npy_intp[it.internalCacheLength];
            }
            it.internalCacheIndex = 0;

            if (indices < 0)
                indices += it.size;

            if (it.nd_m1 == 0)
            {
    
                npy_intp it_strides_0 = it.strides[0];
                npy_intp it_ao_data_data_offset = it.ao.data.data_offset;

                for (int i = 0; i < it.internalCacheLength; i++)
                {
                    it.internalCache[i] = it_ao_data_data_offset + (indices * it_strides_0);
                    indices += step;
                }

 
                it.strides[0] = it_strides_0;
                it.ao.data.data_offset = it_ao_data_data_offset;

                return;
            }

            if (it.contiguous)
            {
                int elsize = it.elsize;
                npy_intp it_ao_data_data_offset = it.ao.data.data_offset;

                for (int i = 0; i < it.internalCacheLength; i++)
                {
                    it.internalCache[i] = it_ao_data_data_offset + (indices * elsize);
                    indices += step;
                }

                it.ao.data.data_offset = it_ao_data_data_offset;
                return;
            }

            for (int i = 0; i < it.internalCacheLength; i++)
            {
                it.internalCache[i] = it.ao.data.data_offset;
                npy_intp _indices = indices;
                for (int index = 0; index <= it.nd_m1; index++)
                {
                    it.internalCache[i] += (_indices / it.factors[index]) * it.strides[index];
                    _indices %= it.factors[index];
                }

                indices += step;
            }

            return;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_GOTO(NpyArrayIterObject it, npy_intp[] destination)
        {
            Debug.Assert(Validate(it));
            int i;
            it.index = 0;
            it.dataptr.data_offset = it.ao.data.data_offset;
            for (i = it.nd_m1; i >= 0; i--)
            {
                if (destination[i] < 0)
                {
                    destination[i] += it.dims_m1[i] + 1;
                }
                it.dataptr.data_offset += destination[i] * it.strides[i];
                it.coordinates[i] =destination[i];
                it.index += destination[i] * (i == it.nd_m1 ? 1 : it.dims_m1[i + 1] + 1);
            }
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_GOTO_INDEX(NpyArrayIterObject it, npy_intp[] destination)
        {
            Debug.Assert(Validate(it));
            int i;
            it.index = 0;
            it.dataptr.data_offset = it.ao_offset;
            for (i = it.nd_m1; i >= 0; i--)
            {
                if (destination[i] < 0)
                {
                    destination[i] += it.dims_m1[i] + 1;
                }
                it.dataptr.data_offset += ((destination[i] * it.strides[i]) >> it.ao.ItemDiv);
                it.coordinates[i] = destination[i];
                it.index += destination[i] * (i == it.nd_m1 ? 1 : it.dims_m1[i + 1] + 1);
            }
        }

        internal static NpyArray NpyArray_IterSubscriptBoolArray(NpyArrayIterObject self, NpyArray index)
        {
            NpyArray result;
            npy_intp bool_size, i;
            npy_intp result_size;
            npy_intp stride;
            VoidPtr dptr;
            VoidPtr optr;
            bool swap;

            if (index.nd != 1)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                 "boolean index array should have 1 dimension");
                return null;
            }

            bool_size = index.dimensions[0];
            if (bool_size > self.size)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                "too many boolean indices");
                return null;
            }

            /* Get the size of the result by counting the Trues in the index. */
            stride = index.strides[0];
            dptr = new VoidPtr(index);
            Debug.Assert(index.descr.elsize == 1);

            i = bool_size;
            result_size = 0;

            bool[] data = dptr.datap as bool[];
            npy_intp dptr_index = 0;
            while (i-- > 0)
            {
                if (data[dptr_index])
                {
                    ++result_size;
                }
                dptr_index += stride;
            }

            /* Build the result. */
            Npy_INCREF(self.ao.descr);
            result = NpyArray_Alloc(self.ao.descr, 1, new npy_intp[] { result_size }, false, Npy_INTERFACE(self.ao));
            if (result == null)
            {
                return null;
            }

            /* Copy in the data. */
            swap = (NpyArray_ISNOTSWAPPED(self.ao) != NpyArray_ISNOTSWAPPED(result));
            optr = new VoidPtr(result);
            dptr = new VoidPtr(index);
            NpyArray_ITER_RESET(self);

            var helper = MemCopy.GetMemcopyHelper(optr);
            helper.IterSubscriptBoolArray(self, optr, data, stride, bool_size, swap);
        
            NpyArray_ITER_RESET(self);

            return result;
        }
   

        internal static NpyArray NpyArray_IterSubscriptIntpArray(NpyArrayIterObject self, NpyArray Intp_Array)
        {
            NpyArray result;
            NpyArrayIterObject index_iter;

            /* Build the result in the same shape as the index. */
            Npy_INCREF(self.ao.descr);
            result = NpyArray_Alloc(self.ao.descr,
                                    Intp_Array.nd, Intp_Array.dimensions,
                                    false, Npy_INTERFACE(self.ao));
            if (result == null)
            {
                return null;
            }

            /* Copy in the data. */
            index_iter = NpyArray_IterNew(Intp_Array);
            if (index_iter == null)
            {
                Npy_DECREF(result);
                return null;
            }

            bool swap = (NpyArray_ISNOTSWAPPED(self.ao) != NpyArray_ISNOTSWAPPED(result));
            VoidPtr optr = new VoidPtr(result);

            NpyArray_ITER_RESET(self);

            var helper = MemCopy.GetMemcopyHelper(optr);
            npy_intp? num = helper.IterSubscriptIntpArray(self, index_iter, optr, swap);
            if (num.HasValue)
            {
                string msg = string.Format("index {0} out of bounds 0<=index<{1}", num.Value, self.size);
                NpyErr_SetString(npyexc_type.NpyExc_IndexError, msg);
                Npy_DECREF(index_iter);
                Npy_DECREF(result);
                NpyArray_ITER_RESET(self);
                return null;
            }

            Npy_DECREF(index_iter);
            NpyArray_ITER_RESET(self);

            return result;
        }

        internal static NpyArray NpyArray_IterSubscriptIntp(NpyArrayIterObject self, npy_intp index)
        {
            NpyArray result;
            bool swap;

 
            if (index < 0)
            {
                index += self.size;
            }
            if (index < 0 || index >= self.size)
            {
                string msg = string.Format("index {0} out of bounds 0<=index<{1}", index, self.size);
                NpyErr_SetString(npyexc_type.NpyExc_IndexError, msg);

                return null;
            }

            Npy_INCREF(self.ao.descr);
            result = NpyArray_Alloc(self.ao.descr, 0, null,
                                    false, Npy_INTERFACE(self.ao));
            if (result == null)
            {
                return null;
            }

            swap = (NpyArray_ISNOTSWAPPED(self.ao) != NpyArray_ISNOTSWAPPED(result));
            NpyArray_ITER_RESET(self);
            NpyArray_ITER_GOTO1D(self, index);

            MemCopy.GetMemcopyHelper(result.data).copyswap(result.data, self.dataptr, swap);
            NpyArray_ITER_RESET(self);
            return result;
        }

        static int NpyArray_IterSubscriptAssignSlice(NpyArrayIterObject self, NpyIndexSlice slice, NpyArray value)
        {
            NpyArray converted_value;
            npy_intp steps, start, step_size;
            bool swap;
            NpyArrayIterObject value_iter = null;

 
            Npy_INCREF(self.ao.descr);
            converted_value = NpyArray_FromArray(value, self.ao.descr, 0);
            if (converted_value == null)
            {
                return -1;
            }

            /* Copy in the data. */
            value_iter = NpyArray_IterNew(converted_value);
            if (value_iter == null)
            {
                Npy_DECREF(converted_value);
                return -1;
            }

            if (value_iter.size > 0)
            {
                steps = NpyArray_SliceSteps(slice);
                start = slice.start;
                step_size = slice.step;
                swap = (NpyArray_ISNOTSWAPPED(self.ao) !=
                        NpyArray_ISNOTSWAPPED(converted_value));

                NpyArray_ITER_RESET(self);
                var helper = MemCopy.GetMemcopyHelper(self.dataptr);
                helper.IterSubscriptAssignSlice(self, value_iter, steps, start, step_size, swap);
    
                NpyArray_ITER_RESET(self);
            }

            Npy_DECREF(value_iter);
            Npy_DECREF(converted_value);

            return 0;
        }


        internal static int NpyArray_IterSubscriptAssignIntp(NpyArrayIterObject self, npy_intp index, NpyArray value)
        {
            NpyArray converted_value;
            bool swap;

            if (NpyArray_SIZE(value) == 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                 "Error setting single item of array");
                return -1;
            }

            Npy_INCREF(self.ao.descr);
            converted_value = NpyArray_FromArray(value, self.ao.descr, 0);
            if (converted_value == null)
            {
                return -1;
            }

            swap = (NpyArray_ISNOTSWAPPED(self.ao) != NpyArray_ISNOTSWAPPED(converted_value));

            NpyArray_ITER_RESET(self);
            NpyArray_ITER_GOTO1D(self, index);

            MemCopy.GetMemcopyHelper(self.dataptr).copyswap(self.dataptr, converted_value.data, swap);
            NpyArray_ITER_RESET(self);

            Npy_DECREF(converted_value);
            return 0;
        }

        internal static int NpyArray_IterSubscriptAssignBoolArray(NpyArrayIterObject self, NpyArray index, NpyArray value)
        {
            NpyArray converted_value;
            npy_intp bool_size;
            npy_intp stride;
            bool[] dptr;
            NpyArrayIterObject value_iter;
            bool swap;

            if (index.nd != 1)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                 "boolean index array should have 1 dimension");
                return -1;
            }

            bool_size = index.dimensions[0];
            if (bool_size > self.size)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                "too many boolean indices");
                return -1;
            }

            Npy_INCREF(self.ao.descr);
            converted_value = NpyArray_FromArray(value, self.ao.descr, 0);
            if (converted_value == null)
            {
                return -1;
            }

            value_iter = NpyArray_IterNew(converted_value);
            if (value_iter == null)
            {
                Npy_DECREF(converted_value);
                return -1;
            }

            if (value_iter.size > 0)
            {
                /* Copy in the data. */
                stride = index.strides[0];

                dptr = index.data.datap as bool[];

                Debug.Assert(index.descr.elsize == 1);
                swap = (NpyArray_ISNOTSWAPPED(self.ao) != NpyArray_ISNOTSWAPPED(converted_value));

                NpyArray_ITER_RESET(self);

                var helper = MemCopy.GetMemcopyHelper(self.dataptr);
                helper.IterSubscriptAssignBoolArray(self, value_iter, bool_size, dptr, stride, swap);
         
                NpyArray_ITER_RESET(self);
            }

            Npy_DECREF(value_iter);
            Npy_DECREF(converted_value);

            return 0;
        }

        static int NpyArray_IterSubscriptAssignIntpArray(NpyArrayIterObject self, NpyArray index, NpyArray value)
        {
            Npy_INCREF(self.ao.descr);
            NpyArray converted_value = NpyArray_FromArray(value, self.ao.descr, 0);
            if (converted_value == null)
            {
                return -1;
            }

            NpyArrayIterObject index_iter = NpyArray_IterNew(index);
            if (index_iter == null)
            {
                Npy_DECREF(converted_value);
                return -1;
            }

            NpyArrayIterObject value_iter = NpyArray_IterNew(converted_value);
            if (value_iter == null)
            {
                Npy_DECREF(index_iter);
                Npy_DECREF(converted_value);
                return -1;
            }
            Npy_DECREF(converted_value);

            if (value_iter.size > 0)
            {
                bool swap = (NpyArray_ISNOTSWAPPED(self.ao) != NpyArray_ISNOTSWAPPED(converted_value));

                NpyArray_ITER_RESET(self);

                var helper = MemCopy.GetMemcopyHelper(self.dataptr);
                npy_intp? num = helper.IterSubscriptAssignIntpArray(self, index_iter, value_iter, swap);
                if (num.HasValue)
                {
                    string msg = string.Format("index {0} out of bounds 0<=index<{1}", num.Value, self.size);
                    NpyErr_SetString(npyexc_type.NpyExc_IndexError, msg);

                    Npy_DECREF(index_iter);
                    Npy_DECREF(value_iter);
                    NpyArray_ITER_RESET(self);
                    return -1;
                }
    
                NpyArray_ITER_RESET(self);
            }
            Npy_DECREF(index_iter);
            Npy_DECREF(value_iter);

            return 0;
        }

        internal static bool NpyArray_ITER_NOTDONE(NpyArrayIterObject it)
        {
            return (it.index < it.size);
        }

        internal static void NpyArray_MultiIter_RESET(NpyArrayMultiIterObject multi)
        {
            Debug.Assert(Validate(multi));

            multi.index = 0;
            for (int i = 0; i < multi.numiter; i++)
            {
                NpyArray_ITER_RESET(multi.iters[i]);
            }
        }

        internal static VoidPtr NpyArray_MultiIter_DATA(NpyArrayMultiIterObject multi, npy_intp index)
        {
            return multi.iters[index].dataptr;
        }

  
        internal static bool NpyArray_MultiIter_NOTDONE(NpyArrayMultiIterObject multi)
        {
            return (multi.index < multi.size);
        }

        

    }
}
