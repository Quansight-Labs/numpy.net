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
        internal static NpyArrayMapIterObject NpyArray_MapIterNew(NpyIndex []indexes, int n)
        {
            NpyArrayMapIterObject mit;
            int i, j;

            /* Allocates the Python object wrapper around the map iterator. */
            mit = new NpyArrayMapIterObject();
             
            NpyObject_Init(mit, NpyArrayMapIter_Type);
            if (mit == null)
            {
                return null;
            }
            for (i = 0; i < npy_defs.NPY_MAXDIMS; i++)
            {
                mit.iters[i] = null;
            }
            mit.index = 0;
            mit.ait = null;
            mit.subspace = null;
            mit.numiter = 0;
            mit.consec = 1;
            mit.n_indexes = 0;
            mit.nob_interface = null;

            /* Expand the boolean arrays in indexes. */
            mit.n_indexes = NpyArray_IndexExpandBool(indexes, n,    mit.indexes);
            if (mit.n_indexes < 0)
            {
                Npy_DECREF(mit);
                return null;
            }

            /* Make iterators from any intp arrays and intp in the index. */
            j = 0;
            for (i = 0; i < mit.n_indexes; i++)
            {
                NpyIndex index = mit.indexes[i];

                if (index.type ==  NpyIndexType.NPY_INDEX_INTP_ARRAY)
                {
                    mit.iters[j] = NpyArray_IterNew(index.intp_array);
                    if (mit.iters[j] == null)
                    {
                        mit.numiter = j - 1;
                        Npy_DECREF(mit);
                        return null;
                    }
                    j++;
                }
                else if (index.type == NpyIndexType.NPY_INDEX_INTP)
                {
                    NpyArray_Descr indtype;
                    NpyArray indarray;

                    /* Make a 0-d array for the index. */
                    indtype = NpyArray_DescrFromType(NPY_TYPES.NPY_INTP);
                    indarray = NpyArray_Alloc(indtype, 0, null, false, null);
                    if (indarray == null)
                    {
                        mit.numiter = j - 1;
                        Npy_DECREF(mit);
                        return null;
                    }

                    byte[] src = BitConverter.GetBytes(index.intp);

                    var srcvp = new VoidPtr(src);
                    var helper = MemCopy.GetMemcopyHelper(indarray.data);
                    helper.memmove_init(indarray.data, srcvp);
                    helper.memcpy(indarray.data.data_offset, srcvp.data_offset, sizeof(npy_intp));
                    mit.iters[j] = NpyArray_IterNew(indarray);
                    Npy_DECREF(indarray);
                    if (mit.iters[j] == null)
                    {
                        mit.numiter = j - 1;
                        Npy_DECREF(mit);
                        return null;
                    }
                    j++;
                }
            }
            mit.numiter = j;
 
            /* Broadcast the index iterators. */
            if (NpyArray_Broadcast(mit) < 0)
            {
                Npy_DECREF(mit);
                return null;
            }

            return mit;
        }


        internal static int NpyArray_MapIterBind(NpyArrayMapIterObject mit, NpyArray arr, NpyArray true_array)
        {
            NpyArrayIterObject it;
            int subnd;
            npy_intp i, j;
            int n;
            npy_intp dimsize;

            NpyIndex []bound_indexes = AllocateNpyIndexes(npy_defs.NPY_MAXDIMS);
            int nbound = 0;

            subnd = arr.nd - mit.numiter;
            if (subnd < 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "too many indices for array");
                return -1;
            }

            mit.ait = NpyArray_IterNew(arr);
            if (mit.ait == null)
            {
                return -1;
            }
            /* no subspace iteration needed.  Finish up and Return */
            if (subnd == 0)
            {
                n = arr.nd;
                for (i = 0; i < n; i++)
                {
                    mit.iteraxes[i] = i;
                }
                goto finish;
            }


            /* Bind the indexes to the array. */
            nbound = NpyArray_IndexBind(mit.indexes, mit.n_indexes,
                                        arr.dimensions, arr.nd,
                                        bound_indexes);
            if (nbound < 0)
            {
                nbound = 0;
                goto fail;
            }

            /* Fill in iteraxes and bscoord from the bound indexes. */
            j = 0;
            for (i = 0; i < nbound; i++)
            {
                NpyIndex index = bound_indexes[i];

                switch (index.type)
                {
                    case NpyIndexType.NPY_INDEX_INTP_ARRAY:
                        mit.iteraxes[j++] = i;
                        mit.bscoord[i] = 0;
                        break;
                    case NpyIndexType.NPY_INDEX_INTP:
                        mit.bscoord[i] = index.intp;
                        break;
                    case NpyIndexType.NPY_INDEX_SLICE:
                        mit.bscoord[i] = index.slice.start;
                        break;
                    default:
                        mit.bscoord[i] = 0;
                        break;
                }
            }

            /* Check for non-consecutive axes. */
            mit.consec = 1;
            j = mit.iteraxes[0];
            for (i = 1; i < mit.numiter; i++)
            {
                if (mit.iteraxes[i] != j + i)
                {
                    mit.consec = 0;
                    break;
                }
            }

            /*
             * Make the subspace iterator.
             */
            {
                npy_intp []dimensions = new npy_intp[npy_defs.NPY_MAXDIMS];
                npy_intp [] strides = new npy_intp[npy_defs.NPY_MAXDIMS];
                npy_intp offset = 0;
                int n2;
                NpyArray view;

                /* Convert to dimensions and strides. */
                n2 = NpyArray_IndexToDimsEtc(arr, bound_indexes, nbound,
                                             dimensions, strides, ref offset,
                                             true);
                if (n2 < 0)
                {
                    goto fail;
                }

                Npy_INCREF(arr.descr);
                view = NpyArray_NewView(arr.descr, n2,
                                        dimensions, strides,
                                        arr, offset, true);
                if (view == null)
                {
                    goto fail;
                }
                mit.subspace = NpyArray_IterNew(view);
                Npy_DECREF(view);
                if (mit.subspace == null)
                {
                    goto fail;
                }
            }

            /* Expand dimensions of result */
            n = mit.subspace.ao.nd;
            for (i = 0; i < n; i++)
            {
                mit.dimensions[mit.nd + i] = mit.subspace.ao.dimensions[i];
                mit.bscoord[mit.nd + i] = 0;
            }
            mit.nd += n;

            /* Free the indexes. */
            NpyArray_IndexDealloc(bound_indexes, nbound);
            nbound = 0;

            finish:
            /* Here check the indexes (now that we have iteraxes) */
            mit.size = NpyArray_OverflowMultiplyList(mit.dimensions, mit.nd);
            if (mit.size < 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,"dimensions too large in fancy indexing");
                goto fail;
            }
            if (mit.ait.size == 0 && mit.size != 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,"invalid index into a 0-size array");
                goto fail;
            }


            for (i = 0; i < mit.numiter; i++)
            {
                npy_intp indval;
                it = mit.iters[i];
                NpyArray_ITER_RESET(it);
                dimsize = NpyArray_DIM(arr, mit.iteraxes[i]);
                npy_intp[] dataptr = it.dataptr.datap as npy_intp[];
                while (it.index < it.size)
                {
                    indval =  dataptr[it.dataptr.data_offset >> IntpDivSize];
                    if (indval < 0)
                    {
                        indval += dimsize;
                    }
                    if (indval < 0 || indval >= dimsize)
                    {
                        string msg = string.Format("index ({0}) out of range (0<=index<{1}) in dimension {2}", indval, (dimsize - 1), mit.iteraxes[i]);
                        NpyErr_SetString(npyexc_type.NpyExc_IndexError, msg);
                        goto fail;
                    }
                    NpyArray_ITER_NEXT(it);
                }
                NpyArray_ITER_RESET(it);
            }
            return 0;

            fail:
            NpyArray_IndexDealloc(bound_indexes, nbound);
            Npy_XDECREF(mit.subspace);
            Npy_XDECREF(mit.ait);
            mit.subspace = null;
            mit.ait = null;
            return -1;
        }

        internal static void NpyArray_MapIterReset(NpyArrayMapIterObject mit)
        {
            NpyArrayIterObject it;
            npy_intp i, j;
            npy_intp []coord = new npy_intp[npy_defs.NPY_MAXDIMS];

            mit.index = 0;

            if (mit.subspace != null)
            {
                copydims(coord, mit.bscoord, mit.ait.ao.nd);
                NpyArray_ITER_RESET(mit.subspace);
                for (i = 0; i < mit.numiter; i++)
                {
                    it = mit.iters[i];
                    NpyArray_ITER_RESET(it);
                    j = mit.iteraxes[i];

                    npy_intp[] s = it.dataptr.datap as npy_intp[];
                    coord[j] = s[it.dataptr.data_offset >> IntpDivSize];
                    //if (!NpyArray_ISNOTSWAPPED(it.ao))
                    //{
                    //    // not sure I need to do anything here.
                    //}

                }
                NpyArray_ITER_GOTO(mit.ait, coord);
                mit.subspace.dataptr = new VoidPtr(mit.ait.dataptr);
                mit.dataptr = new VoidPtr(mit.subspace.dataptr);
            }
            else
            {
                for (i = 0; i < mit.numiter; i++)
                {
                    it = mit.iters[i];
                    if (it.size != 0)
                    {
                        NpyArray_ITER_RESET(it);

                        npy_intp[] s = it.dataptr.datap as npy_intp[];
                        coord[i] = s[it.dataptr.data_offset >> IntpDivSize];
                        //if (!NpyArray_ISNOTSWAPPED(it.ao))
                        //{
                        //    // not sure I need to do anything here.
                        //}
                    }
                    else
                    {
                        coord[i] = 0;
                    }
                }
                NpyArray_ITER_GOTO(mit.ait, coord);
                mit.dataptr = new VoidPtr(mit.ait.dataptr);
            }
            return;
        }

 

        internal static void NpyArray_MapIterNext(NpyArrayMapIterObject mit)
        {
            NpyArrayIterObject it;
            npy_intp i, j;
            npy_intp []coord = new npy_intp[npy_defs.NPY_MAXDIMS];

            mit.index += 1;
            if (mit.index >= mit.size)
            {
                return;
            }
            /* Sub-space iteration */
            if (mit.subspace != null)
            {
                NpyArray_ITER_NEXT(mit.subspace);
                if (mit.subspace.index >= mit.subspace.size)
                {
                    /* reset coord to coordinates of beginning of the subspace */
                    copydims(coord, mit.bscoord, mit.ait.ao.nd);
                    NpyArray_ITER_RESET(mit.subspace);
                    for (i = 0; i < mit.numiter; i++)
                    {
                        it = mit.iters[i];
                        NpyArray_ITER_NEXT(it);
                        j = mit.iteraxes[i];

                        npy_intp[] s = it.dataptr.datap as npy_intp[];
                        coord[j] = s[it.dataptr.data_offset >> IntpDivSize];
                        //if (!NpyArray_ISNOTSWAPPED(it.ao))
                        //{
                        //    // not sure I need to do anything here.
                        //}
                    }
                    NpyArray_ITER_GOTO(mit.ait, coord);
                    mit.subspace.dataptr = new VoidPtr(mit.ait.dataptr);
                }
                mit.dataptr = new VoidPtr(mit.subspace.dataptr);
            }
            else
            {
                for (i = 0; i < mit.numiter; i++)
                {
                    it = mit.iters[i];
                    NpyArray_ITER_NEXT(it);

                    npy_intp[] s = it.dataptr.datap as npy_intp[];
                    coord[i] = s[it.dataptr.data_offset >> IntpDivSize];
                    //if (!NpyArray_ISNOTSWAPPED(it.ao))
                    //{
                    //    // not sure I need to do anything here.
                    //}
                }
                NpyArray_ITER_GOTO(mit.ait, coord);
                mit.dataptr.data_offset = mit.ait.dataptr.data_offset;
            }
            return;
        }

     
        internal static void NpyArray_MapIterNext_SubSpace(NpyArrayMapIterObject mit, npy_intp[] offsets, npy_intp offsetsLength, npy_intp offsetsIndex)
        {
            NpyArrayIterObject it;
            npy_intp i, j;
            npy_intp[] coord = new npy_intp[npy_defs.NPY_MAXDIMS];

            if (mit.subspace == null)
            {
                throw new Exception("NpyArray_MapIterNext called without a subspace");
            }

            while (offsetsIndex < offsetsLength)
            {
                mit.index += 1;
                if (mit.index >= mit.size)
                {
                    return;
                }

                NpyArray_ITER_NEXT(mit.subspace);
                if (mit.subspace.index >= mit.subspace.size)
                {
                    /* reset coord to coordinates of beginning of the subspace */
                    copydims(coord, mit.bscoord, mit.ait.ao.nd);
                    NpyArray_ITER_RESET(mit.subspace, mit.subspace.ao.ItemDiv);
                    for (i = 0; i < mit.numiter; i++)
                    {
                        it = mit.iters[i];
                        NpyArray_ITER_NEXT(it);
                        j = mit.iteraxes[i];

                        npy_intp[] s = it.dataptr.datap as npy_intp[];
                        coord[j] = s[it.dataptr.data_offset];
                        //if (!NpyArray_ISNOTSWAPPED(it.ao))
                        //{
                        //    // not sure I need to do anything here.
                        //}
                    }
                    NpyArray_ITER_GOTO_INDEX(mit.ait, coord);
                    mit.subspace.dataptr = new VoidPtr(mit.ait.dataptr);
                }
                offsets[offsetsIndex++] = mit.subspace.dataptr.data_offset;

            }


            return;
        }

        internal static void NpyArray_MapIterNext(NpyArrayMapIterObject mit, npy_intp[] offsets, npy_intp offsetsLength, npy_intp offsetsIndex)
        {
            NpyArrayIterObject it;
            npy_intp i, j;
            npy_intp[] coord = new npy_intp[npy_defs.NPY_MAXDIMS];

            while (offsetsIndex < offsetsLength)
            {
                mit.index += 1;
                if (mit.index >= mit.size)
                {
                    return;
                }

                for (i = 0; i < mit.numiter; i++)
                {
                    it = mit.iters[i];
                    NpyArray_ITER_NEXT(it);

                    npy_intp[] s = it.dataptr.datap as npy_intp[];
                    coord[i] = s[it.dataptr.data_offset];
                    //if (!NpyArray_ISNOTSWAPPED(it.ao))
                    //{
                    //    // not sure I need to do anything here.
                    //}
                }
                NpyArray_ITER_GOTO_INDEX(mit.ait, coord);
                offsets[offsetsIndex++] = mit.ait.dataptr.data_offset;
            }

            return;
        }

        internal static NpyArray NpyArray_GetMap(NpyArrayMapIterObject mit)
        {
            NpyArrayIterObject it;

            /* Unbound map iterator --- Bind should have been called */
            if (mit.ait == null)
            {
                return null;
            }

            /* This relies on the map iterator object telling us the shape
               of the new array in nd and dimensions.
            */
            NpyArray temp = mit.ait.ao;
            Npy_INCREF(temp.descr);
            NpyArray ret = NpyArray_Alloc(temp.descr, mit.nd, mit.dimensions,
                                 NpyArray_ISFORTRAN(temp), Npy_INTERFACE(temp));
            if (ret == null)
            {
                return null;
            }

            /*
             * Now just iterate through the new array filling it in
             * with the next object from the original array as
             * defined by the mapping iterator
             */

            if ((it = NpyArray_IterNew(ret)) == null)
            {
                Npy_DECREF(ret);
                return null;
            }

            bool swap = (NpyArray_ISNOTSWAPPED(temp) != NpyArray_ISNOTSWAPPED(ret));
            NpyArray_MapIterReset(mit);

            bool EnableGetMapHelper = false;
            if (EnableGetMapHelper)
            {
                var helper = MemCopy.GetMemcopyHelper(it.dataptr);
                helper.GetMap(it, mit, swap);
            }
            else
            {
                var helper = MemCopy.GetMemcopyHelper(it.dataptr);
                npy_intp index = it.size;

                while (index-- > 0)
                {
                    helper.copyswap(it.dataptr, mit.dataptr, swap);
                    NpyArray_MapIterNext(mit);
                    NpyArray_ITER_NEXT(it);
                }
            }
  


            Npy_DECREF(it);

            /* check for consecutive axes */
            if ((mit.subspace != null) && (mit.consec != 0))
            {
                if (mit.iteraxes[0] > 0)
                {  /* then we need to swap */
                    _swap_axes(mit, ref ret, true);
                }
            }
            return ret;
        }

        internal static int NpyArray_SetMap(NpyArrayMapIterObject mit, NpyArray arr)
        {
            NpyArrayIterObject it;

            /* Unbound Map Iterator */
            if (mit.ait == null)
            {
                return -1;
            }
            Npy_INCREF(arr);
            if ((mit.subspace != null) && (mit.consec != 0))
            {
                if (mit.iteraxes[0] > 0)
                {  /* then we need to swap */
                    _swap_axes(mit, ref arr, false);
                    if (arr == null)
                    {
                        return -1;
                    }
                }
            }

            /* Be sure values array is "broadcastable"
               to shape of mit.dimensions, mit.nd */

            if ((it = NpyArray_BroadcastToShape(arr, mit.dimensions, mit.nd)) == null)
            {
                Npy_DECREF(arr);
                return -1;
            }

            bool swap = (NpyArray_ISNOTSWAPPED(mit.ait.ao) != NpyArray_ISNOTSWAPPED(arr));
            NpyArray_MapIterReset(mit);

            bool EnableSetMapHelper = false;
            if (EnableSetMapHelper)
            {
                var helper = MemCopy.GetMemcopyHelper(mit.dataptr);
                helper.SetMap(mit, it, swap);
            }
            else
            {
                npy_intp index = mit.size;
                var helper = MemCopy.GetMemcopyHelper(mit.dataptr);
                helper.memmove_init(mit.dataptr, it.dataptr);
                while (index-- > 0)
                {
                    helper.memmove(mit.dataptr.data_offset, it.dataptr.data_offset, NpyArray_ITEMSIZE(arr));
                    if (swap)
                    {
                        helper.copyswap(mit.dataptr, null, swap);
                    }
                    NpyArray_MapIterNext(mit);
                    NpyArray_ITER_NEXT(it);
                }
            }

            Npy_DECREF(arr);
            Npy_DECREF(it);
            return 0;
        }


        internal static NpyArray NpyArray_ArrayItem(NpyArray self, npy_intp i)
        {
            NpyArray newView;

            if (NpyArray_NDIM(self) == 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_IndexError,  "0-d arrays can't be indexed");
                return null;
            }

            VoidPtr item = NpyArray_Index2Ptr(self, i);
            if (item == null)
            {
                return null;
            }

            int NewDimSize = NpyArray_NDIM(self) - 1;

            npy_intp[] NewDims = new npy_intp[NewDimSize];
            Array.Copy(NpyArray_DIMS(self), 1, NewDims, 0, NewDimSize);

            npy_intp[] NewStrides = new npy_intp[NewDimSize];
            Array.Copy(NpyArray_STRIDES(self), 1, NewStrides, 0, NewDimSize);

            Npy_INCREF(NpyArray_DESCR(self));
            newView = NpyArray_NewView(NpyArray_DESCR(self),
                                 NewDimSize,
                                 NewDims,
                                 NewStrides,
                                 self, item.data_offset - self.data.data_offset,
                                 false);
            return newView;

        }

        internal static NpyArray NpyArray_IndexSimple(NpyArray self, NpyIndex []indexes, int n)
        {
            NpyIndex []new_indexes = AllocateNpyIndexes(npy_defs.NPY_MAXDIMS);
            npy_intp []dimensions = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp []strides = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp offset = 0;
            int n2, n_new;
            NpyArray result;



            /* Bind the index to the array. */
            n_new = NpyArray_IndexBind(indexes, n, self.dimensions, self.nd, new_indexes);
            if (n_new < 0)
            {
                return null;
            }

            /* Convert to dimensions and strides. */
            n2 = NpyArray_IndexToDimsEtc(self, new_indexes, n_new,
                                         dimensions, strides, ref offset, false);
            NpyArray_IndexDealloc(new_indexes, n_new);
            if (n2 < 0)
            {
                return null;
            }

            /* Make the result. */
            Npy_INCREF(self.descr);
            result = NpyArray_NewView(self.descr, n2, dimensions, strides,
                                      self, offset, false);

            return result;
        }

        internal static int NpyArray_IndexFancyAssign(NpyArray self, NpyIndex []indexes, int n, NpyArray value)
        {
            int result;

            if (self.nd == 1 && n == 1)
            {
                /* Special case for 1-d arrays. */
                NpyArrayIterObject iter = NpyArray_IterNew(self);
                if (iter == null)
                {
                    return -1;
                }
                result = NpyArray_IterSubscriptAssign(iter, indexes, n, value);
                Npy_DECREF(iter);
                return result;
            }
            else
            {
                NpyArrayMapIterObject mit = NpyArray_MapIterNew(indexes, n);
                if (mit == null)
                {
                    return -1;
                }
                if (NpyArray_MapIterBind(mit, self, null) < 0)
                {
                    Npy_DECREF(mit);
                    return -1;
                }

                result = NpyArray_SetMap(mit, value);
                Npy_DECREF(mit);
                return result;
            }
        }

        static void arraymapiter_dealloc(object o1)
        {
            int i;

            NpyArrayMapIterObject mit = o1 as NpyArrayMapIterObject;

            Debug.Assert(0 == mit.nob_refcnt);

            mit.nob_interface = null;
            Npy_XDECREF(mit.ait);
            Npy_XDECREF(mit.subspace);
            for (i = 0; i < mit.numiter; i++)
            {
                Npy_XDECREF(mit.iters[i]);
            }
            NpyArray_IndexDealloc(mit.indexes, mit.n_indexes);
            NpyArray_free(mit);
        }

        internal static NpyArray NpyArray_Subscript(NpyArray self, NpyIndex []indexes, int n)
        {
            /* Handle cases where we just return this array. */
            if (n == 0 || (n == 1 && indexes[0].type == NpyIndexType.NPY_INDEX_ELLIPSIS))
            {
                Npy_INCREF(self);
                return self;
            }

            /* Handle returning a single field. */
            if (n == 1 && indexes[0].type == NpyIndexType.NPY_INDEX_STRING)
            {
                return NpyArray_SubscriptField(self, indexes[0]._string);
            }

            /* Handle the simple item case. */
            if (n == 1 && indexes[0].type == NpyIndexType.NPY_INDEX_INTP)
            {
                return NpyArray_ArrayItem(self, indexes[0].intp);
            }

            /* Treat 0-d indexes as a special case. */
            if (self.nd == 0)
            {
                return NpyArray_Subscript0d(self, indexes, n);
            }

            /* Either do simple or fancy indexing. */
            if (is_simple(indexes, n))
            {
                return NpyArray_IndexSimple(self, indexes, n);
            }
            else
            {
                return NpyArray_IndexFancy(self, indexes, n);
            }
        }

        static NpyArray NpyArray_Subscript0d(NpyArray self, NpyIndex []indexes, int n)
        {
            NpyArray result;
            npy_intp []dimensions = new npy_intp[npy_defs.NPY_MAXDIMS];
            bool has_ellipsis = false;
            int nd_new = 0;
            int i;

            for (i = 0; i < n; i++)
            {
                switch (indexes[i].type)
                {
                    case NpyIndexType.NPY_INDEX_NEWAXIS:
                        dimensions[nd_new++] = 1;
                        break;
                    case NpyIndexType.NPY_INDEX_ELLIPSIS:
                        if (has_ellipsis)
                        {
                            goto err;
                        }
                        has_ellipsis = true;
                        break;
                    default:
                        goto err;
                }
            }

            Npy_INCREF(self.descr);
            result = NpyArray_NewView(self.descr, nd_new, dimensions, null, self, 0, false);
            return result;

            err:
            NpyErr_SetString(npyexc_type.NpyExc_IndexError, "0-d arrays can only use a single () or a list of newaxes (and a single ...) as an index");
            return null;
        }

        static NpyArray NpyArray_IndexFancy(NpyArray self, NpyIndex []indexes, int n)
        {
            NpyArray result;

            if (self.nd == 1 && n == 1)
            {
                /* Special case for 1-d arrays. */
                NpyArrayIterObject iter = NpyArray_IterNew(self);
                if (iter == null)
                {
                    return null;
                }
                result = NpyArray_IterSubscript(iter, indexes, n);
                Npy_DECREF(iter);
                return result;
            }
            else
            {
                NpyArrayMapIterObject mit = NpyArray_MapIterNew(indexes, n);
                if (mit == null)
                {
                    return null;
                }
                if (NpyArray_MapIterBind(mit, self, null) < 0)
                {
                    Npy_DECREF(mit);
                    return null;
                }

                result = NpyArray_GetMap(mit);
                Npy_DECREF(mit);
                return result;
            }
        }


        internal static int NpyArray_SubscriptAssign(NpyArray self, NpyIndex []indexes, int n, NpyArray value)
        {
            NpyArray view;
            int result;

            if (!NpyArray_ISWRITEABLE(self))
            {
                NpyErr_SetString(npyexc_type.NpyExc_RuntimeError, "array is not writeable");
                return -1;
            }

            /* Handle cases where we have the whole array */
            if (n == 0 || (n == 1 && indexes[0].type == NpyIndexType.NPY_INDEX_ELLIPSIS))
            {
                return NpyArray_MoveInto(self, value);
            }

            /* Handle returning a single field. */
            if (n == 1 && indexes[0].type == NpyIndexType.NPY_INDEX_STRING)
            {
                return NpyArray_SubscriptAssignField(self, indexes[0]._string, value);
            }

            /* Handle the simple item case. */
            if (n == 1 && indexes[0].type == NpyIndexType.NPY_INDEX_INTP)
            {
                view = NpyArray_ArrayItem(self, indexes[0].intp);
                if (view == null)
                {
                    return -1;
                }
                result = NpyArray_MoveInto(view, value);
                Npy_DECREF(view);
                return result;
            }

            /* Either do simple or fancy indexing. */
            if (is_simple(indexes, n))
            {
                view = NpyArray_IndexSimple(self, indexes, n);
                if (view == null)
                {
                    return -1;
                }
                result = NpyArray_MoveInto(view, value);
                Npy_DECREF(view);
                return result;
            }
            else
            {
                return NpyArray_IndexFancyAssign(self, indexes, n, value);
            }
        }

        static NpyArray NpyArray_SubscriptField(NpyArray self, string field)
        {
            NpyArray_DescrField value = null;

            if (self.descr.names != null)
            {
                value = (NpyArray_DescrField)NpyDict_Get(self.descr.fields, field);
            }

            if (value != null)
            {
                Npy_INCREF(value.descr);
                return NpyArray_GetField(self, value.descr, value.offset);
            }
            else
            {
                string msg = string.Format("field named {0} not found.", field);
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                return null;
            }
        }

        static int NpyArray_SubscriptAssignField(NpyArray self, string field,  NpyArray v)
        {
            NpyArray_DescrField value = null;

            if (self.descr.names != null)
            {
                value = (NpyArray_DescrField)NpyDict_Get(self.descr.fields, field);
            }

            if (value != null)
            {
                Npy_INCREF(value.descr);
                return NpyArray_SetField(self, value.descr, value.offset, v);
            }
            else
            {
                string msg = string.Format("field named {0} not found.", field);
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                return -1;
            }
        }

        /*
         * Determine if this is a simple index.
         */
        static bool is_simple(NpyIndex []indexes, int n)
        {
            int i;

            for (i = 0; i < n; i++)
            {
                switch (indexes[i].type)
                {
                    case NpyIndexType.NPY_INDEX_INTP_ARRAY:
                    case NpyIndexType.NPY_INDEX_BOOL_ARRAY:
                    case NpyIndexType.NPY_INDEX_STRING:
                        return false;
                    default:
                        break;
                }
            }

            return true;
        }

        static void _swap_axes(NpyArrayMapIterObject mit, ref NpyArray ret, bool getmap)
        {
            NpyArray _new;
            npy_intp n1, n2, n3, val, bnd;
            int i;
            NpyArray_Dims permute = new NpyArray_Dims();
            npy_intp []d = new npy_intp[npy_defs.NPY_MAXDIMS];
            NpyArray arr;

            permute.ptr = d;
            permute.len = mit.nd;

            /*
             * arr might not have the right number of dimensions
             * and need to be reshaped first by pre-pending ones
             */
            arr = ret;
            if (arr.nd != mit.nd)
            {
                for (i = 1; i <= arr.nd; i++)
                {
                    permute.ptr[mit.nd - i] = arr.dimensions[arr.nd - i];
                }
                for (i = 0; i < mit.nd - arr.nd; i++)
                {
                    permute.ptr[i] = 1;
                }
                _new = NpyArray_Newshape(arr, permute, NPY_ORDER.NPY_ANYORDER);
                Npy_DECREF(arr);
                ret = _new;
                if (_new == null)
                {
                    return;
                }
            }

            /*
             * Setting and getting need to have different permutations.
             * On the get we are permuting the returned object, but on
             * setting we are permuting the object-to-be-set.
             * The set permutation is the inverse of the get permutation.
             */

            /*
             * For getting the array the tuple for transpose is
             * (n1,...,n1+n2-1,0,...,n1-1,n1+n2,...,n3-1)
             * n1 is the number of dimensions of the broadcast index array
             * n2 is the number of dimensions skipped at the start
             * n3 is the number of dimensions of the result
             */

            /*
             * For setting the array the tuple for transpose is
             * (n2,...,n1+n2-1,0,...,n2-1,n1+n2,...n3-1)
             */
            n1 = mit.iters[0].nd_m1 + 1;
            n2 = mit.iteraxes[0];
            n3 = mit.nd;

            /* use n1 as the boundary if getting but n2 if setting */
            bnd = getmap ? n1 : n2;
            val = bnd;
            i = 0;
            while (val < n1 + n2)
            {
                permute.ptr[i++] = val++;
            }
            val = 0;
            while (val < bnd)
            {
                permute.ptr[i++] = val++;
            }
            val = n1 + n2;
            while (val < n3)
            {
                permute.ptr[i++] = val++;
            }
            _new = NpyArray_Transpose(ret, permute);
            Npy_DECREF(ret);
            ret = _new;
        }


        private static NpyIndex[] AllocateNpyIndexes(int NumIndices)
        {
            NpyIndex[] new_indexes = new NpyIndex[NumIndices];

            for (int i = 0; i < NumIndices; i++)
            {
                new_indexes[i] = new NpyIndex();
            }

            return new_indexes;
        }


    }
}
