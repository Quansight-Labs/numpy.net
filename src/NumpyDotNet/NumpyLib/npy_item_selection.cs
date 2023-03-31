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
using System.Threading;
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
  
        static int qsortCompare(object a, object b)
        {
            throw new NotImplementedException();
            // return global_obj.descr.f.compare(a, b, global_obj);
            return 0;
        }


        internal static NpyArray NpyArray_TakeFrom(NpyArray self0, NpyArray indices0, int axis, NpyArray ret, NPY_CLIPMODE clipmode)
        {
            NpyArray self;
            NpyArray indices;
            int nd;
            npy_intp max_item, nelem;
            npy_intp n, m, chunk;
            npy_intp []shape = new npy_intp[npy_defs.NPY_MAXDIMS];
            VoidPtr src;
            VoidPtr dest;
            bool copyret = false;
            int err;

            indices = null;
            self = NpyArray_CheckAxis(self0, ref axis, NPYARRAYFLAGS.NPY_CARRAY);
            if (self == null)
            {
                return null;
            }
            indices = NpyArray_ContiguousFromArray(indices0, NPY_TYPES.NPY_INTP);
            if (indices == null)
            {
                Npy_XDECREF(self);
                return null;
            }
            n = m = chunk = 1;
            nd = self.nd + indices.nd -1;
            for (int i = 0; i < nd; i++)
            {
                if (i < axis)
                {
                    shape[i] = self.dimensions[i];
                    n *= shape[i];
                }
                else
                {
                    if (i < axis + indices.nd)
                    {
                        shape[i] = indices.dimensions[i - axis];
                        m *= shape[i];
                    }
                    else
                    {
                        shape[i] = self.dimensions[i - indices.nd + 1];
                        chunk *= shape[i];
                    }
                }
            }
            Npy_INCREF(self.descr);
            if (ret == null)
            {
                ret = NpyArray_Alloc(self.descr, nd, shape, false, Npy_INTERFACE(self));
                if (ret == null)
                {
                    goto fail;
                }
            }
            else
            {
                NpyArray obj;
                NPYARRAYFLAGS flags = NPYARRAYFLAGS.NPY_CARRAY | NPYARRAYFLAGS.NPY_UPDATEIFCOPY;

                if ((ret.nd != nd) ||
                    !NpyArray_CompareLists(ret.dimensions, shape, nd))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "bad shape in output array");
                    ret = null;
                    Npy_DECREF(self.descr);
                    goto fail;
                }

                if (clipmode == NPY_CLIPMODE.NPY_RAISE)
                {
                    /*
                     * we need to make sure and get a copy
                     * so the input array is not changed
                     * before the error is called
                     */
                    flags |= NPYARRAYFLAGS.NPY_ENSURECOPY;
                }
                obj = NpyArray_FromArray(ret, self.descr, flags);
                if (obj != ret)
                {
                    copyret = true;
                }
                ret = obj;
                if (ret == null)
                {
                    goto fail;
                }
            }

            max_item = self.dimensions[axis];
            nelem = chunk;
            chunk = chunk * ret.descr.elsize;
            src = new VoidPtr(self);
            npy_intp src_index = 0;
            dest = new VoidPtr(ret);
            npy_intp dest_index = 0;

            npy_intp[] indicesData = indices.data.datap as npy_intp[];

            var helper = MemCopy.GetMemcopyHelper(dest);
            helper.memmove_init(dest, src);

            switch (clipmode)
            {
                case NPY_CLIPMODE.NPY_RAISE:
                    for (int i = 0; i < n; i++)
                    {
                        bool out_of_range = false;
                        Parallel.For(0, m, j =>
                        {
                            var tmp = indicesData[j];
                            if (tmp < 0)
                            {
                                tmp = tmp + max_item;
                            }
                            if ((tmp < 0) || (tmp >= max_item))
                            {
                                out_of_range = true;
                            }

                            helper.memmove(dest.data_offset + (dest_index + (j * chunk)), src.data_offset + (src_index + (tmp * chunk)), chunk);
                        });

                        if (out_of_range)
                        {
                            NpyErr_SetString(npyexc_type.NpyExc_IndexError, "index out of range for array");
                            goto fail;
                        }

                        dest_index += m * chunk;
                        src_index += chunk * max_item;
                    }
                    break;
                case NPY_CLIPMODE.NPY_WRAP:
                    for (int i = 0; i < n; i++)
                    {
                        Parallel.For(0, m, j =>
                        {
                            var tmp = indicesData[j];
                            if (tmp < 0)
                            {
                                while (tmp < 0)
                                {
                                    tmp += max_item;
                                }
                            }
                            else if (tmp >= max_item)
                            {
                                while (tmp >= max_item)
                                {
                                    tmp -= max_item;
                                }
                            }

                            helper.memmove(dest.data_offset + (dest_index + (j * chunk)), src.data_offset + (src_index + (tmp * chunk)), chunk);
                        });

                        dest_index += m * chunk;
                        src_index += chunk * max_item;
                    }
                    break;
                case NPY_CLIPMODE.NPY_CLIP:
                    for (int i = 0; i < n; i++)
                    {
                        Parallel.For(0, m, j =>
                        {
                            var tmp = indicesData[j];
                            if (tmp < 0)
                            {
                                tmp = 0;
                            }
                            else if (tmp >= max_item)
                            {
                                tmp = max_item - 1;
                            }

                            helper.memmove(dest.data_offset + (dest_index + (j * chunk)), src.data_offset + (src_index + (tmp * chunk)), chunk);
                        });

                        dest_index += (m * chunk);
                        src_index += chunk * max_item;
                    }
                    break;
            }


            NpyArray_INCREF(ret);
            Npy_XDECREF(indices);
            Npy_XDECREF(self);
            if (copyret)
            {
                NpyArray obj;
                obj = ret.base_arr;
                Npy_INCREF(obj);
                NpyArray_ForceUpdate(ret);
                Npy_DECREF(ret);
                ret = obj;
            }
            return ret;

            fail:
            NpyArray_XDECREF_ERR(ret);
            Npy_XDECREF(indices);
            Npy_XDECREF(self);
            return null;
        }

        internal static int NpyArray_PutTo(NpyArray self, NpyArray values0, NpyArray indices0, NPY_CLIPMODE clipmode)
        {
            NpyArray indices; 
            NpyArray values;
            npy_intp i, chunk, ni, max_item, nv, tmp;
            VoidPtr src;
            VoidPtr dest;
            bool copied = false;
            VoidPtr buf = null;

            indices = null;
            values = null;
            if (!NpyArray_ISCONTIGUOUS(self))
            {
                NpyArray obj;
                NPYARRAYFLAGS flags = NPYARRAYFLAGS.NPY_CARRAY | NPYARRAYFLAGS.NPY_UPDATEIFCOPY;

                if (clipmode == NPY_CLIPMODE.NPY_RAISE)
                {
                    flags |= NPYARRAYFLAGS.NPY_ENSURECOPY;
                }
                Npy_INCREF(self.descr);
                obj = NpyArray_FromArray(self, self.descr, flags);
                if (obj != self)
                {
                    copied = true;
                }
                self = obj;
            }


            max_item = NpyArray_SIZE(self);
            dest = new VoidPtr(self);
            chunk = self.descr.elsize;
            indices = NpyArray_ContiguousFromArray(indices0, NPY_TYPES.NPY_INTP);
            if (indices == null)
            {
                goto fail;
            }
            ni = NpyArray_SIZE(indices);
            Npy_INCREF(self.descr);
            values = NpyArray_FromArray(values0, self.descr, NPYARRAYFLAGS.NPY_DEFAULT | NPYARRAYFLAGS.NPY_FORCECAST);
            if (values == null)
            {
                goto fail;
            }
            nv = NpyArray_SIZE(values);
            if (nv <= 0)
            {
                goto finish;
            }

            npy_intp[] indicesData = indices.data.datap as npy_intp[];

            var helper = MemCopy.GetMemcopyHelper(dest);
            helper.memmove_init(dest, values.data);

            switch (clipmode)
            {
                case NPY_CLIPMODE.NPY_RAISE:
                    for (i = 0; i < ni; i++)
                    {
                        src = new VoidPtr(values.data, chunk * (i % nv));
                        tmp = indicesData[i];
                        if (tmp < 0)
                        {
                            tmp = tmp + max_item;
                        }
                        if ((tmp < 0) || (tmp >= max_item))
                        {
                            NpyErr_SetString(npyexc_type.NpyExc_IndexError, "index out of range for array");
                            goto fail;
                        }

                        helper.memmove(dest.data_offset + (tmp * chunk), src.data_offset, chunk);
                    }
                    break;
                case NPY_CLIPMODE.NPY_WRAP:
                    for (i = 0; i < ni; i++)
                    {
                        src = new VoidPtr(values.data, chunk * (i % nv));
                        tmp = indicesData[i];
                        if (tmp < 0)
                        {
                            while (tmp < 0)
                            {
                                tmp += max_item;
                            }
                        }
                        else if (tmp >= max_item)
                        {
                            while (tmp >= max_item)
                            {
                                tmp -= max_item;
                            }
                        }

                        helper.memmove(dest.data_offset + (tmp * chunk), src.data_offset, chunk);
                    }
                    break;
                case NPY_CLIPMODE.NPY_CLIP:
                    for (i = 0; i < ni; i++)
                    {
                        src = new VoidPtr(values.data, chunk * (i % nv));
                        tmp = indicesData[i];
                        if (tmp < 0)
                        {
                            tmp = 0;
                        }
                        else if (tmp >= max_item)
                        {
                            tmp = max_item - 1;
                        }
                        helper.memmove(dest.data_offset + (tmp * chunk), src.data_offset, chunk);
                    }
                    break;
            }


            finish:
            if (buf != null)
            {
                npy_free(buf);
            }
            Npy_XDECREF(values);
            Npy_XDECREF(indices);
            if (copied)
            {
                NpyArray_ForceUpdate(self);
                Npy_DECREF(self);
            }
            return 0;

            fail:
            if (buf != null)
            {
                npy_free(buf);
            }
            Npy_XDECREF(indices);
            Npy_XDECREF(values);
            if (copied)
            {
                NpyArray_XDECREF_ERR(self);
            }
            return -1;
        }

        internal static int NpyArray_PutMask(NpyArray self, NpyArray values0, NpyArray mask0)
        {
            NpyArray mask;
            NpyArray values;
            npy_intp i, chunk, ni, max_item, nv;
            VoidPtr src;
            VoidPtr dest;
            bool copied = false;

            mask = null;
            values = null;

            if (!NpyArray_ISCONTIGUOUS(self))
            {
                NpyArray obj;
                NPYARRAYFLAGS flags = NPYARRAYFLAGS.NPY_CARRAY | NPYARRAYFLAGS.NPY_UPDATEIFCOPY;

                Npy_INCREF(self.descr);
                obj = NpyArray_FromArray(self, self.descr, flags);
                if (obj != self)
                {
                    copied = true;
                }
                self = obj;
            }

            max_item = NpyArray_SIZE(self);
            dest = new VoidPtr(self);
            chunk = self.descr.elsize;
            mask = NpyArray_FromArray(mask0, NpyArray_DescrFromType(NPY_TYPES.NPY_BOOL), NPYARRAYFLAGS.NPY_CARRAY | NPYARRAYFLAGS.NPY_FORCECAST);
            if (mask == null)
            {
                goto fail;
            }
            ni = NpyArray_SIZE(mask);
            if (ni != max_item)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "putmask: mask and data must be the same size");
                goto fail;
            }
            Npy_INCREF(self.descr);
            values = NpyArray_FromArray(values0, self.descr, NPYARRAYFLAGS.NPY_CARRAY);
            if (values == null)
            {
                goto fail;
            }
            nv = NpyArray_SIZE(values); /* zero if null array */
            if (nv <= 0)
            {
                Npy_XDECREF(values);
                Npy_XDECREF(mask);
                return 0;
            }

            bool[] maskData = mask.data.datap as bool[];

            npy_intp j;

            if (nv == 1)
            {
                double s_val = (double)GetIndex(values.data, 0);
                for (i = 0; i < ni; i++)
                {
                    if ((bool)GetIndex(mask.data, i))
                    {
                        SetIndex(dest, i, s_val);
                    }
                }
            }
            else
            {
                for (i = 0, j = 0; i < ni; i++, j++)
                {
                    if (j >= nv)
                    {
                        j = 0;
                    }
                    if ((bool)GetIndex(mask.data, i))
                    {
                        SetIndex(dest, i, GetIndex(values.data, j));
                    }
                }
            }

            Npy_XDECREF(values);
            Npy_XDECREF(mask);
            if (copied)
            {
                NpyArray_ForceUpdate(self);
                Npy_DECREF(self);
            }
            return 0;

            fail:
            Npy_XDECREF(mask);
            Npy_XDECREF(values);
            if (copied)
            {
                NpyArray_XDECREF_ERR(self);
            }
            return -1;
        }

        internal static NpyArray NpyArray_Repeat(NpyArray aop, NpyArray op, int axis)
        {
            npy_intp []counts;
            npy_intp n, n_outer, i, j, k, chunk, total;
            npy_intp tmp;
            bool broadcast = false;
            NpyArray repeats = null;
            NpyArray ret = null;
            VoidPtr new_data = null;
            VoidPtr old_data = null;

            repeats = NpyArray_ContiguousFromArray(op, NPY_TYPES.NPY_INTP);
            if (repeats == null)
            {
                return null;
            }

            /*
            * Scalar and size 1 'repeat' arrays broadcast to any shape, for all
            * other inputs the dimension must match exactly.
            */
            if (NpyArray_NDIM(repeats) == 0 || NpyArray_SIZE(repeats) == 1)
            {
                broadcast = true;
            }

            //nd = repeats.nd;
            counts = ConvertToIntP(repeats.data);

            aop = NpyArray_CheckAxis(aop, ref axis, NPYARRAYFLAGS.NPY_CARRAY);
            if (aop == null)
            {
                Npy_DECREF(repeats);
                return null;
            }

            n = NpyArray_DIM(aop, axis);

            if (!broadcast && aop.dimensions[axis] != n)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "a.shape[axis] != len(repeats)");
                goto fail;
            }

            if (broadcast)
            {
                total = counts[0] * n;
            }
            else
            {

                total = 0;
                for (j = 0; j < n; j++)
                {
                    if (counts[j] < 0)
                    {
                        NpyErr_SetString(npyexc_type.NpyExc_ValueError, "count < 0");
                        goto fail;
                    }
                    total += counts[j];
                }
            }


            /* Construct new array */
            aop.dimensions[axis] = total;
            Npy_INCREF(aop.descr);
            ret = NpyArray_Alloc(aop.descr, aop.nd, aop.dimensions, false, Npy_INTERFACE(aop));
            aop.dimensions[axis] = n;
            if (ret == null)
            {
                goto fail;
            }
            new_data = new VoidPtr(ret);
            old_data = new VoidPtr(aop);

            chunk = aop.descr.elsize;
            for (i = axis + 1; i < aop.nd; i++)
            {
                chunk *= aop.dimensions[i];
            }

            n_outer = 1;
            for (i = 0; i < axis; i++)
            {
                n_outer *= aop.dimensions[i];
            }

            var helper = MemCopy.GetMemcopyHelper(new_data);
            helper.memmove_init(new_data, old_data);
            for (i = 0; i < n_outer; i++)
            {
                for (j = 0; j < n; j++)
                {
                    tmp = broadcast ? counts[0] : counts[j];
                    for (k = 0; k < tmp; k++)
                    {
                        helper.memcpy(new_data.data_offset, old_data.data_offset, chunk);
                        new_data.data_offset += chunk;
                    }
                    old_data.data_offset += chunk;
                }
            }

            Npy_DECREF(repeats);
            NpyArray_INCREF(ret);
            Npy_XDECREF(aop);
            return ret;

            fail:
            Npy_DECREF(repeats);
            Npy_XDECREF(aop);
            Npy_XDECREF(ret);
            return null;
        }

        internal static NpyArray NpyArray_Choose(NpyArray ip, NpyArray[] mps, int n, NpyArray ret, NPY_CLIPMODE clipmode)
        {
            int elsize;
            VoidPtr ret_data;
            NpyArray ap;
            NpyArrayMultiIterObject multi = null;
            npy_intp mi;
            bool copyret = false;
            ap = null;


            ap = NpyArray_FromArray(ip, NpyArray_DescrFromType(NPY_TYPES.NPY_INTP), 0);
            if (ap == null)
            {
                goto fail;
            }

            /* Broadcast all arrays to each other, index array at the end. */
            multi = NpyArray_MultiIterFromArrays(mps, n, 1, ap);
            if (multi == null)
            {
                goto fail;
            }
            /* Set-up return array */
            if (ret == null)
            {
                Npy_INCREF(mps[0].descr);
                ret = NpyArray_Alloc(mps[0].descr,
                                     multi.nd, multi.dimensions,
                                     false, Npy_INTERFACE(ap));
            }
            else
            {
                NpyArray obj;
                NPYARRAYFLAGS flags = NPYARRAYFLAGS.NPY_CARRAY | NPYARRAYFLAGS.NPY_UPDATEIFCOPY | NPYARRAYFLAGS.NPY_FORCECAST;

                if ((NpyArray_NDIM(ret) != multi.nd)
                        || !NpyArray_CompareLists(NpyArray_DIMS(ret), multi.dimensions, multi.nd))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_TypeError, "invalid shape for output array.");
                    ret = null;
                    goto fail;
                }
                if (clipmode == NPY_CLIPMODE.NPY_RAISE)
                {
                    /*
                     * we need to make sure and get a copy
                     * so the input array is not changed
                     * before the error is called
                     */
                    flags |= NPYARRAYFLAGS.NPY_ENSURECOPY;
                }
                Npy_INCREF(mps[0].descr);
                obj = NpyArray_FromArray(ret, mps[0].descr, flags);
                if (obj != ret)
                {
                    copyret = true;
                }
                ret = obj;
            }

            if (ret == null)
            {
                goto fail;
            }
            elsize = ret.descr.elsize;
            ret_data = new VoidPtr(ret);
            int ret_data_index = 0;

            var helper = MemCopy.GetMemcopyHelper(ret_data);

            while (NpyArray_MultiIter_NOTDONE(multi))
            {
                VoidPtr data = NpyArray_MultiIter_DATA(multi, n);
                mi = (npy_intp)GetIndex(data, data.data_offset >> IntpDivSize);
                if (mi < 0 || mi >= n)
                {
                    switch (clipmode)
                    {
                        case NPY_CLIPMODE.NPY_RAISE:
                            NpyErr_SetString(npyexc_type.NpyExc_ValueError, "invalid entry in choice array");
                            goto fail;
                        case NPY_CLIPMODE.NPY_WRAP:
                            if (mi < 0)
                            {
                                while (mi < 0)
                                {
                                    mi += n;
                                }
                            }
                            else
                            {
                                while (mi >= n)
                                {
                                    mi -= n;
                                }
                            }
                            break;
                        case NPY_CLIPMODE.NPY_CLIP:
                            if (mi < 0)
                            {
                                mi = 0;
                            }
                            else if (mi >= n)
                            {
                                mi = n - 1;
                            }
                            break;
                    }
                }


                var srcVP = new VoidPtr(NpyArray_MultiIter_DATA(multi, mi));
                helper.memmove_init(ret_data, srcVP);
                helper.memmove(ret_data.data_offset + ret_data_index, srcVP.data_offset, elsize);

                ret_data_index += elsize;

                NpyArray_MultiIter_NEXT(multi);
            }

            NpyArray_INCREF(ret);
            Npy_DECREF(multi);
            Npy_DECREF(ap);
            if (copyret)
            {
                NpyArray obj;
                obj = ret.base_arr;
                Npy_INCREF(obj);
                NpyArray_ForceUpdate(ret);
                Npy_DECREF(ret);
                ret = obj;
            }
            return ret;

        fail:
            Npy_XDECREF(multi);
            Npy_XDECREF(ap);
            NpyArray_XDECREF_ERR(ret);
            return null;
        }

        private static int _new_sortlike(NpyArray op, int axis, NpyArray_SortFunc sort, NpyArray_PartitionFunc part, npy_intp[] kth, npy_intp nkth, NPY_SORTKIND kind)
        {
            npy_intp N = NpyArray_DIM(op, axis);
            NpyArrayIterObject it;
            npy_intp size;


            /* Check if there is any sorting to do */
            if (N <= 1 || NpyArray_SIZE(op) == 0)
            {
                return 0;
            }

            it = NpyArray_IterAllButAxis(op, ref axis);
            if (it == null)
            {
                return -1;
            }
            size = it.size;

            int errorsDetected = 0;

            var itParallelIters = NpyArray_ITER_ParallelSplit(it);

            Parallel.For(0, itParallelIters.Count(), index =>
            {
                var litIter = itParallelIters.ElementAt(index);

                while (litIter.index < litIter.size)
                {
                    _new_sortlike(litIter.dataptr, ref errorsDetected, op, axis, sort, part, kth, nkth, kind);
                    NpyArray_ITER_NEXT(litIter);
                }
            });
   

            Npy_DECREF(it);

            if (errorsDetected > 0)
                return -1;
            return 0;
        }

        private static void _new_sortlike(VoidPtr iterList, ref int errorsDetected, NpyArray op, int axis, NpyArray_SortFunc sort, NpyArray_PartitionFunc part, npy_intp[] kth, npy_intp nkth, NPY_SORTKIND kind)
        {
            npy_intp N = NpyArray_DIM(op, axis);
            int elsize = NpyArray_ITEMSIZE(op);
            npy_intp astride = NpyArray_STRIDE(op, axis);
            bool swap = NpyArray_ISBYTESWAPPED(op);
            bool needcopy = !NpyArray_ISALIGNED(op) || swap || astride != elsize;

            int ret = 0;

            bool has_failed = false;


            VoidPtr buffer = null;
            VoidPtr bufptr = new VoidPtr(iterList);

            if (needcopy)
            {
                buffer = NpyDataMem_NEW(op.ItemType, (ulong)(N * elsize));
                _default_copyswap(buffer, elsize, iterList, astride, N, swap, op);
                bufptr = new VoidPtr(buffer);
            }
            /*
             * TODO: If the input array is byte-swapped but contiguous and
             * aligned, it could be swapped (and later unswapped) in-place
             * rather than after copying to the buffer. Care would have to
             * be taken to ensure that, if there is an error in the call to
             * sort or part, the unswapping is still done before returning.
             */

            if (part == null)
            {
                ret = sort(bufptr, N, op, kind);
                if (ret < 0)
                {
                    has_failed = true;
                }
            }
            else
            {
                npy_intp[] pivots = new npy_intp[npy_defs.NPY_MAX_PIVOT_STACK];
                npy_intp? npiv = 0;
                for (npy_intp i = 0; i < nkth; ++i)
                {
                    ret = part(bufptr, N, kth[i], pivots, ref npiv, op);
                    if (ret < 0)
                    {
                        has_failed = true;
                    }
                }
            }

            if (needcopy)
            {
                _default_copyswap(iterList, astride, buffer, elsize, N, swap, op);
            }


            if (has_failed)
            {
                Interlocked.Increment(ref errorsDetected);
            }
        }

        private static NpyArray _new_argsortlike(NpyArray op, int axis, 
                    NpyArray_ArgSortFunc argsort, NpyArray_ArgPartitionFunc argpart,  
                    npy_intp[] kth, npy_intp nkth)
        {
            npy_intp N = NpyArray_DIM(op, axis);
            int elsize = NpyArray_ITEMSIZE(op);
            npy_intp astride = NpyArray_STRIDE(op, axis);
            bool swap = NpyArray_ISBYTESWAPPED(op);
            bool needcopy = !NpyArray_ISALIGNED(op) || swap || astride != elsize;
            bool needidxbuffer;


            NpyArray rop;
            npy_intp rstride;

            NpyArrayIterObject it, rit;
            npy_intp size;

            int ret = 0;

            rop = NpyArray_New(null, NpyArray_NDIM(op),
                                               NpyArray_DIMS(op), NPY_TYPES.NPY_INTP,
                                               null, null, 0, 0, op);
            if (rop == null)
            {
                return null;
            }
            rstride = NpyArray_STRIDE(rop, axis);
            needidxbuffer = rstride != sizeof(npy_intp);

            /* Check if there is any argsorting to do */
            if (N <= 1 || NpyArray_SIZE(op) == 0)
            {
                memclr(NpyArray_DATA(rop), NpyArray_SIZE(rop));
                return rop;
            }

            it = NpyArray_IterAllButAxis(op, ref axis);
            rit = NpyArray_IterAllButAxis(rop, ref axis);
    
            size = it.size;


            int errorsDetected = 0;

            var itParallelIters = NpyArray_ITER_ParallelSplit(it);
            var ritParallelIters = NpyArray_ITER_ParallelSplit(rit);

            Parallel.For(0, ritParallelIters.Count(), index =>
            {
                var lritIter = ritParallelIters.ElementAt(index);
                var litIter = itParallelIters.ElementAt(index);

                while (litIter.index < litIter.size)
                {
                    _new_argsortlike(litIter.dataptr, lritIter.dataptr, ref errorsDetected,
                                op, axis, argsort, argpart, kth, nkth, needidxbuffer, rstride);

                    NpyArray_ITER_NEXT(litIter);
                    NpyArray_ITER_NEXT(lritIter);
                }

            });
   

            if (errorsDetected > 0)
            {
                if (!NpyErr_Occurred())
                {
                    /* Out of memory during sorting or buffer creation */
                    NpyErr_NoMemory();
                }
                Npy_XDECREF(rop);
                rop = null;
            }
            Npy_XDECREF(it);
            Npy_XDECREF(rit);

            return rop;
        }

        private static void _new_argsortlike(VoidPtr opIterList, VoidPtr ropIterList, ref int errorsDetected, 
                    NpyArray op, int axis, NpyArray_ArgSortFunc argsort, NpyArray_ArgPartitionFunc argpart,
                    npy_intp[] kth, npy_intp nkth, bool needidxbuffer, npy_intp rstride)
        {
            npy_intp N = NpyArray_DIM(op, axis);
            int elsize = NpyArray_ITEMSIZE(op);
            npy_intp astride = NpyArray_STRIDE(op, axis);
            bool swap = NpyArray_ISBYTESWAPPED(op);
            bool needcopy = !NpyArray_ISALIGNED(op) || swap || astride != elsize;

            int ret = 0;

            bool has_failed = false;

            
            {
                VoidPtr idxbuffer = null;

                VoidPtr valptr = opIterList;
                VoidPtr idxptr = ropIterList;
                VoidPtr iptr;
                int i;

                if (needcopy)
                {
                    var valbuffer = NpyDataMem_NEW(op.ItemType, (ulong)(N * elsize));
                    _default_copyswap(valbuffer, elsize, opIterList, astride, N, swap, op);
                    valptr = new VoidPtr(valbuffer);
                }

                if (needidxbuffer)
                {
                    idxbuffer = NpyDataMem_NEW(NPY_TYPES.NPY_INTP, (ulong)(N * sizeof(npy_intp)));
                    idxptr = new VoidPtr(idxbuffer);
                }

                iptr = new VoidPtr(idxptr);
                iptr.data_offset >>= GetDivSize(GetTypeSize(iptr));

                for (i = 0; i < N; ++i)
                {
                    SetIndex(iptr, iptr.data_offset, i);
                    iptr.data_offset += 1;
                }

                if (argpart == null)
                {
                    ret = argsort(valptr, idxptr, N, op, NPY_SORTKIND.NPY_QUICKSORT);

                    if (ret < 0)
                    {
                        has_failed = true;
                    }
                }
                else
                {
                    npy_intp[] pivots = new npy_intp[npy_defs.NPY_MAX_PIVOT_STACK];
                    npy_intp? npiv = 0;

                    for (i = 0; i < nkth; ++i)
                    {
                        ret = argpart(valptr, idxptr, N, kth[i], pivots, ref npiv, op);

                        if (ret < 0)
                        {
                            has_failed = true;
                        }
                    }
                }

                if (needidxbuffer)
                {
                    VoidPtr rptr = new VoidPtr(ropIterList);
                    var rptrSize = GetTypeSize(rptr);

                    iptr = new VoidPtr(idxbuffer);
                    iptr.data_offset /= GetTypeSize(iptr);

                    for (i = 0; i < N; ++i)
                    {
                        SetIndex(rptr, rptr.data_offset / rptrSize, GetIndex(iptr, iptr.data_offset));
                        iptr.data_offset += 1;
                        rptr.data_offset += rstride;
                    }
                }

            };

            if (has_failed)
            {
                Interlocked.Increment(ref errorsDetected);
            }
        }



        internal static int NpyArray_Sort(NpyArray op, int axis, NPY_SORTKIND which)
        {
            NpyArray ap = null;
            VoidPtr ip;
            int i,  elsize, orign = 0;
            int m, n;

            n = op.nd;
            if ((n == 0) || (NpyArray_SIZE(op) == 1))
            {
                return 0;
            }
            if (axis < 0)
            {
                axis += n;
            }
            if ((axis < 0) || (axis >= n))
            {
                string msg = string.Format("axis(={0}) out of bounds", axis);
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                return -1;
            }
            if (!NpyArray_ISWRITEABLE(op))
            {
                NpyErr_SetString(npyexc_type.NpyExc_RuntimeError,"attempted sort on unwriteable array.");
                return -1;
            }

            /* Determine if we should use type-specific algorithm or not */
            if (op.descr.f.sort[(int)which] != null)
            {
                return _new_sort(op, axis, which);
            }
            if ((which != NPY_SORTKIND.NPY_QUICKSORT) || op.descr.f.compare == null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_TypeError, "desired sort not supported for this type");
                return -1;
            }

            SWAPAXES2(op, ref axis, ref orign);

            ap = NpyArray_FromArray(op, null, NPYARRAYFLAGS.NPY_DEFAULT);
            if (ap == null)
            {
                goto fail;
            }
            elsize = ap.descr.elsize;
            m = (int)ap.dimensions[ap.nd - 1];
            if (m == 0)
            {
                goto finish;
            }
            n = (int)NpyArray_SIZE(ap) / m;

            npy_intp ip_index = 0;
            for (ip = new VoidPtr(ap), i = 0; i < n; i++, ip_index += elsize * m)
            {
                qsort(ip, ip_index, m, elsize, qsortCompare);
            }

            if (NpyErr_Occurred())
            {
                goto fail;
            }

            finish:
            if (ap != op)
            {
                // Copy back
                if (NpyArray_CopyAnyInto(op, ap) < 0)
                {
                    goto fail;
                }
            }
            Npy_DECREF(ap);  /* Should update op if needed */
            SWAPBACK2(op, ref axis, ref orign);
            return 0;

            fail:
            Npy_XDECREF(ap);
            SWAPBACK2(op, ref axis, ref orign);
            return -1;
        }


        /*
        * make kth array positive, ravel and sort it
        */
        private static NpyArray partition_prep_kth_array(NpyArray ktharray, NpyArray op, int axis)
        {
            npy_intp[] shape = NpyArray_DIMS(op);
            NpyArray kthrvl;
            npy_intp[] kth;
            npy_intp nkth, i;

            if (!NpyArray_CanCastSafely(NpyArray_TYPE(ktharray), NPY_TYPES.NPY_INTP))
            {
                NpyErr_SetString(npyexc_type.NpyExc_TypeError, "Partition index must be integer");
                return null;
            }

            if (NpyArray_NDIM(ktharray) > 1)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "kth array must have dimension <= 1");
                return null;
            }
            kthrvl = NpyArray_CastToType(ktharray, NpyArray_DescrFromType(NPY_TYPES.NPY_INTP), false);

            if (kthrvl == null)
                return null;

            kth = NpyArray_DATA(kthrvl).datap as npy_intp[];
            nkth = NpyArray_SIZE(kthrvl);

            for (i = 0; i < nkth; i++)
            {
                if (kth[i] < 0)
                {
                    kth[i] += shape[axis];
                }
                if (NpyArray_SIZE(op) != 0 &&
                            (kth[i] < 0 || kth[i] >= shape[axis]))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, string.Format("kth(={0}) out of bounds ({1})", kth[i], shape[axis]));
                    Npy_XDECREF(kthrvl);
                    return null;
                }
            }


            /*
             * sort the array of kths so the partitions will
             * not trample on each other
             */
            if (NpyArray_SIZE(kthrvl) > 1)
            {
                NpyArray_Sort(kthrvl, -1, NPY_SORTKIND.NPY_QUICKSORT);
            }

            return kthrvl;
        }

        private static bool check_and_adjust_axis(ref int axis, int ndim)
        {
            if (axis < -ndim || axis >= ndim)
            {
                NpyErr_SetString(npyexc_type.NpyExc_TypeError, "specified axis outside range for this array");
                return false;
            }

            if (axis < 0)
                axis += ndim;

            return true;
        }

 

        internal static int NpyArray_Partition(NpyArray op, NpyArray ktharray, int axis,  NPY_SELECTKIND kind)
        {
            NpyArray kthrvl;
            NpyArray_PartitionFunc part;
            NpyArray_SortFunc sort = null;

            int n = NpyArray_NDIM(op);
            int ret;

            if (check_and_adjust_axis(ref axis, n) == false)
            {
                return -1;
            }

            if (!NpyArray_ISWRITEABLE(op))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "partition array is read only");
                return -1;
            }
       
            part = get_partition_func(NpyArray_TYPE(op), kind);
            if (part == null)
            {
                /* Use sorting, slower but equivalent */
                if (NpyArray_DESCR(op).f.compare != null)
                {
                    sort = NpyArray_SortFunc;
                }
                else
                {
                    NpyErr_SetString( npyexc_type.NpyExc_TypeError,  "type does not have compare function");
                    return -1;
                }
            }

            /* Process ktharray even if using sorting to do bounds checking */
            kthrvl = partition_prep_kth_array(ktharray, op, axis);
            if (kthrvl == null)
            {
                return -1;
            }

            ret = _new_sortlike(op, axis, sort, part, NpyArray_DATA(kthrvl).datap as npy_intp[], NpyArray_SIZE(kthrvl), NPY_SORTKIND.NPY_QUICKSORT);

            Npy_DECREF(kthrvl);

            return ret;

        }

  
        internal static NpyArray NpyArray_ArgSort(NpyArray op, int axis, NPY_SORTKIND kind)
        {
            NpyArray ap = null, ret = null, op2;
            VoidPtr ip;
            npy_intp i, j, n, m;
            int orign = 0;
            int argsort_elsize;

            n = op.nd;
            if ((n == 0) || (NpyArray_SIZE(op) == 1))
            {
                ret = NpyArray_New(null, op.nd,
                                   op.dimensions,
                                   NPY_TYPES.NPY_INTP,
                                   null, null, 0, 0,
                                   Npy_INTERFACE(op));
                if (ret == null)
                {
                    return null;
                }
                npy_intp[] i1 = ConvertToIntP(ret.data);
                i1[0] = 0;
                return ret;
            }

            /* Creates new reference op2 */
            if ((op2 = NpyArray_CheckAxis(op, ref axis, 0)) == null)
            {
                return null;
            }

            ///* Determine if we should use new algorithm or not */
            if (op2.descr.f.argsort[(int)kind] != null)
            {
                ret = _new_argsort(op2, axis, kind);
                Npy_DECREF(op2);
                return ret;
            }

            if ((kind != NPY_SORTKIND.NPY_QUICKSORT) || op2.descr.f.compare == null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_TypeError, "requested sort not available for type");
                Npy_DECREF(op2);
                op = null;
                goto fail;
            }

            /* ap will contain the reference to op2 */
            if (!SWAPAXES(ref ap, ref op2, ref axis, ref orign))
                return null;

            op = NpyArray_ContiguousFromArray(ap, NPY_TYPES.NPY_NOTYPE);
            Npy_DECREF(ap);
            if (op == null)
            {
                return null;
            }
            ret = NpyArray_New(null, op.nd,
                               op.dimensions, NPY_TYPES.NPY_INTP,
                               null, null, 0, 0, Npy_INTERFACE(op));
            if (ret == null)
            {
                goto fail;
            }
            ip = new VoidPtr(ret.data);
            argsort_elsize = op.descr.elsize;
            m = op.dimensions[op.nd - 1];
            if (m == 0)
            {
                goto finish;
            }

            n = NpyArray_SIZE(op) / m;
            VoidPtr sortData = new VoidPtr(op.data);

            for (i = 0; i < n; i++, ip.data_offset += m * sizeof(npy_intp), sortData += m * argsort_elsize)
            {
                ArgSortIndexes(ip, m, sortData, 0, kind);
            }


            finish:
            Npy_DECREF(op);
            if (!SWAPBACK(ref op, ref ret, ref axis, ref orign))
                return null;

            return op;

            fail:
            Npy_XDECREF(op);
            Npy_XDECREF(ret);
            return null;

        }

        /*NUMPY_API
         * ArgPartition an array
         */
        internal static NpyArray NpyArray_ArgPartition(NpyArray op, NpyArray ktharray, int axis, NPY_SELECTKIND which)
        {
            NpyArray op2, kthrvl;
            NpyArray_ArgPartitionFunc argpart;
            NpyArray_ArgSortFunc argsort = null;
            NpyArray ret;

  
            argpart = get_argpartition_func(NpyArray_TYPE(op), which);
            if (argpart == null)
            {
                /* Use sorting, slower but equivalent */
                if (NpyArray_DESCR(op).f.compare != null)
                {
                    argsort = NpyArray_ArgSortFunc;
                }
                else
                {
                    NpyErr_SetString(npyexc_type.NpyExc_TypeError, "type does not have compare function");
                    return null;
                }
            }

            op2 = NpyArray_CheckAxis(op, ref axis, 0);
            if (op2 == null)
            {
                return null;
            }

            /* Process ktharray even if using sorting to do bounds checking */
            kthrvl = partition_prep_kth_array(ktharray, op2, axis);
            if (kthrvl == null)
            {
                Npy_DECREF(op2);
                return null;
            }

            ret = _new_argsortlike(op2, axis, argsort, argpart,
                                   NpyArray_DATA(kthrvl).datap as npy_intp[], NpyArray_SIZE(kthrvl));

            Npy_DECREF(kthrvl);
            Npy_DECREF(op2);

            return ret;
        }

  
        private static void ArgSortIndexes(VoidPtr ip, npy_intp m, VoidPtr sortData, npy_intp startingIndex, NPY_SORTKIND kind)
        {
            var DivSize = GetDivSize(GetTypeSize(sortData.type_num));

            switch (sortData.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    var sortBool = new Sort_BOOL();
                    sortBool.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_BYTE:
                    var sortByte = new Sort_BYTE();
                    sortByte.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_UBYTE:
                    var sortUByte = new Sort_UBYTE();
                    sortUByte.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_INT16:
                    var sortInt16 = new Sort_INT16();
                    sortInt16.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_UINT16:
                    var sortUInt16 = new Sort_UINT16();
                    sortUInt16.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_INT32:
                    var sortInt32 = new Sort_INT32();
                    sortInt32.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_UINT32:
                    var sortUInt32 = new Sort_UINT32();
                    sortUInt32.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_INT64:
                    var sortInt64 = new Sort_INT64();
                    sortInt64.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_UINT64:
                    var sortUInt64 = new Sort_UINT64();
                    sortUInt64.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_FLOAT:
                    var sortFloat = new Sort_FLOAT();
                    sortFloat.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_DOUBLE:
                    var sortDouble = new Sort_DOUBLE();
                    sortDouble.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_DECIMAL:
                    var sortDecimal = new Sort_DECIMAL();
                    sortDecimal.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_COMPLEX:
                    var sortCOMPLEX = new Sort_COMPLEX();
                    sortCOMPLEX.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_BIGINT:
                    var sortBIGINT = new Sort_BIGINT();
                    sortBIGINT.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_OBJECT:
                    var sortOBJECT = new Sort_OBJECT();
                    sortOBJECT.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                case NPY_TYPES.NPY_STRING:
                    var sortSTRING = new Sort_STRING();
                    sortSTRING.argSortIndexes(ip, m, sortData, startingIndex, kind, DivSize, IntpDivSize);
                    return;
                default:
                    throw new Exception("ArgSortIndexes does not support this data type");

            }

        }

        internal static NpyArray NpyArray_LexSort(NpyArray []mps, int n, int axis)
        {
            throw new NotImplementedException();
        }

        internal static NpyArray NpyArray_SearchSorted(NpyArray op1, NpyArray op2, NPY_SEARCHSIDE side)
        {
            NpyArray ap1 = null;
            NpyArray ap2 = null;
            NpyArray ret = null;
            NpyArray_Descr dtype;

            dtype = NpyArray_DescrFromArray(op2, op1.descr);
            /* need ap1 as contiguous array and of right type */
            Npy_INCREF(dtype);
            ap1 = NpyArray_FromArray(op1, dtype, NPYARRAYFLAGS.NPY_DEFAULT);
            if (ap1 == null)
            {
                Npy_DECREF(dtype);
                return null;
            }

            /* need ap2 as contiguous array and of right type */
            ap2 = NpyArray_FromArray(op2, dtype, NPYARRAYFLAGS.NPY_DEFAULT);
            if (ap2 == null)
            {
                goto fail;
            }
            /* ret is a contiguous array of intp type to hold returned indices */
            ret = NpyArray_New(null, ap2.nd,
                               ap2.dimensions, NPY_TYPES.NPY_INTP,
                               null, null, 0, 0, Npy_INTERFACE(ap2));
            if (ret == null)
            {
                goto fail;
            }
            /* check that comparison function exists */
            if (ap2.descr.f.compare == null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_TypeError, "compare not supported for type");
                goto fail;
            }

            if (side == NPY_SEARCHSIDE.NPY_SEARCHLEFT)
            {
                local_search_left(ap1, ap2, ret);
            }
            else if (side == NPY_SEARCHSIDE.NPY_SEARCHRIGHT)
            {
                local_search_right(ap1, ap2, ret);
            }
            Npy_DECREF(ap1);
            Npy_DECREF(ap2);
            return ret;

            fail:
            Npy_XDECREF(ap1);
            Npy_XDECREF(ap2);
            Npy_XDECREF(ret);
            return null;
        }

        internal static int NpyArray_NonZeroBool(NpyArray self, NpyArray[] index_arrays, object obj)
        {
            int n = self.nd, j;
            npy_intp count = 0;
            npy_intp i, size;
            NpyArrayIterObject it = null;
            NpyArray item;
            VoidPtr[] dptr = new VoidPtr[npy_defs.NPY_MAXDIMS];

            if (self.ItemType != NPY_TYPES.NPY_BOOL)
            {
                throw new Exception("NpyArray_NonZeroBool called for non-bool array");
            }

            bool[] bdata = self.data.datap as bool[];

            for (i = 0; i < n; i++)
            {
                index_arrays[i] = null;
            }

            it = NpyArray_IterNew(self);
            if (it == null)
            {
                return -1;
            }
            size = it.size;
            for (i = 0; i < size; i++)
            {
                if (bdata[it.dataptr.data_offset])
                {
                    count++;
                }
                NpyArray_ITER_NEXT(it);
            }

            NpyArray_ITER_RESET(it);
            for (j = 0; j < n; j++)
            {
                item = NpyArray_New(null, 1, new npy_intp[] { count }, NPY_TYPES.NPY_INTP, null, null, 0, 0, obj);
                if (item == null)
                {
                    goto fail;
                }
                index_arrays[j] = item;
                dptr[j] = NpyArray_DATA(item);
            }
            if (n == 1)
            {
                npy_intp[] dp = dptr[0].datap as npy_intp[];
                npy_intp dp_offset = 0;
                for (i = 0; i < size; i++)
                {
                    if (bdata[it.dataptr.data_offset])
                    {
                        dp[dp_offset] = i;
                        dp_offset += 1;
                    }
                    NpyArray_ITER_NEXT(it);
                }
            }
            else
            {
                npy_intp[] dp_offsets = new npy_intp[npy_defs.NPY_MAXDIMS];

                /* reset contiguous so that coordinates gets updated */
                it.contiguous = false;
                for (i = 0; i < size; i++)
                {
                    if (bdata[it.dataptr.data_offset])
                    {
                        for (j = 0; j < n; j++)
                        {
                            npy_intp[] dp = dptr[j].datap as npy_intp[];
                            dp[dptr[j].data_offset + dp_offsets[j]] = it.coordinates[j];
                            dp_offsets[j] += 1;
                        }
                    }
                    NpyArray_ITER_NEXT(it);
                }
            }

            Npy_DECREF(it);
            return 0;

            fail:
            for (i = 0; i < n; i++)
            {
                Npy_XDECREF(index_arrays[i]);
            }
            Npy_XDECREF(it);
            return -1;
        }

        internal static int NpyArray_NonZero(NpyArray self, NpyArray[] index_arrays, object obj)
        {
            if (self.ItemType == NPY_TYPES.NPY_BOOL)
            {
                return NpyArray_NonZeroBool(self, index_arrays, obj);
            }

            int n = self.nd, j;
            npy_intp count = 0;
            npy_intp i, size;
            NpyArrayIterObject it = null;
            NpyArray item;
            VoidPtr[] dptr = new VoidPtr[npy_defs.NPY_MAXDIMS];
            NpyArray_NonzeroFunc nonzero = self.descr.f.nonzero;

            for (i = 0; i < n; i++)
            {
                index_arrays[i] = null;
            }

            it = NpyArray_IterNew(self);
            if (it == null)
            {
                return -1;
            }

            it = NpyArray_ITER_ConvertToIndex(it, self.ItemDiv);
            size = it.size;
            for (i = 0; i < size; i++)
            {
                if (nonzero(it.dataptr, it.dataptr.data_offset))
                {
                    count++;
                }
                NpyArray_ITER_NEXT(it);
            }

            NpyArray_ITER_RESET(it, self.ItemDiv);
            for (j = 0; j < n; j++)
            {
                item = NpyArray_New(null, 1, new npy_intp[] { count }, NPY_TYPES.NPY_INTP, null, null, 0, 0, obj);
                if (item == null)
                {
                    goto fail;
                }
                index_arrays[j] = item;
                dptr[j] = NpyArray_DATA(item);
            }
            if (n == 1)
            {
                npy_intp[] dp = dptr[0].datap as npy_intp[];
                npy_intp dp_offset = 0;
                for (i = 0; i < size; i++)
                {
                    if (nonzero(it.dataptr, it.dataptr.data_offset))
                    {
                        dp[dp_offset] = i;
                        dp_offset += 1;
                    }
                    NpyArray_ITER_NEXT(it);
                }
            }
            else
            {
                npy_intp[] dp_offsets = new npy_intp[npy_defs.NPY_MAXDIMS];
                
                /* reset contiguous so that coordinates gets updated */
                it.contiguous = false;
                for (i = 0; i < size; i++)
                {
                    if (nonzero(it.dataptr, it.dataptr.data_offset))
                    {
                        for (j = 0; j < n; j++)
                        {
                            npy_intp[] dp = dptr[j].datap as npy_intp[];
                            dp[dptr[j].data_offset + dp_offsets[j]] = it.coordinates[j];
                            dp_offsets[j] += 1;
                        }
                    }
                    NpyArray_ITER_NEXT(it);
                }
            }

            Npy_DECREF(it);
            return 0;

            fail:
            for (i = 0; i < n; i++)
            {
                Npy_XDECREF(index_arrays[i]);
            }
            Npy_XDECREF(it);
            return -1;
        }

        internal static NpyArray NpyArray_Subarray(NpyArray self, VoidPtr dataptr)
        {
            NpyArray result;
            NpyArray_ArrayDescr subarray = self.descr.subarray;

   
            Npy_INCREF(subarray._base);
            result = NpyArray_NewFromDescr(subarray._base,
                                           subarray.shape_num_dims, subarray.shape_dims, null, 
                                           dataptr, self.flags, true,
                                           null, null);
            if (result == null)
            {
                return null;
            }
            result.SetBase(self);
            Npy_INCREF(self);
            NpyArray_UpdateFlags(result, NPYARRAYFLAGS.NPY_UPDATE_ALL);
            return result;
        }

        internal static IList<npy_intp> NpyArray_IndexesFromAxis(NpyArray op, int axis)
        {
            NpyArray ap = null;
            List<npy_intp> indexes = new List<npy_intp>();
            npy_intp elCount, stride;
            int elsize;

            if ((ap = NpyArray_CheckAxis(op, ref axis, 0)) == null)
            {
                return null;
            }

            /* Will get native-byte order contiguous copy. */
            ap = NpyArray_ContiguousFromArray(op, op.descr.type_num);
            Npy_DECREF(op);
            if (ap == null)
            {
                return null;
            }
   
            elsize = ap.descr.elsize;
            stride = ap.strides[axis];
            stride = stride / elsize;
            if (stride == 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "attempt to get indices of an empty sequence");
                goto fail;
            }

            elCount = ap.dimensions[axis];

            npy_intp data_offset = ap.data.data_offset;
            for (int i = 0; i < elCount; i++, data_offset += elsize * stride)
            {
                indexes.Add(data_offset / elsize);
            }

            Npy_DECREF(ap);
  
            return indexes;

            fail:
            Npy_DECREF(ap);
            return null;
        }

        private static npy_intp[] ConvertToIntP(VoidPtr data)
        {
            try
            {
                #if NPY_INTP_64
                return ConvertToInt64(data);
                #else
                return ConvertToInt32(data);
                #endif
            }
            catch
            {
                throw new Exception(string.Format("{0} can't be converted to integers", data.type_num.ToString()));
            }
   
        }

        private static Int32[] ConvertToInt32(VoidPtr data)
        {
            switch (data.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    var dbool = data.datap as bool[];
                    return Array.ConvertAll<bool, Int32>(dbool, Convert.ToInt32);
                case NPY_TYPES.NPY_BYTE:
                    var dsbyte = data.datap as sbyte[];
                    return Array.ConvertAll<sbyte, Int32>(dsbyte, Convert.ToInt32);
                case NPY_TYPES.NPY_UBYTE:
                    var dbyte = data.datap as byte[];
                    return Array.ConvertAll<byte, Int32>(dbyte, Convert.ToInt32);
                case NPY_TYPES.NPY_UINT16:
                    var duint16 = data.datap as UInt16[];
                    return Array.ConvertAll<UInt16, Int32>(duint16, Convert.ToInt32);
                case NPY_TYPES.NPY_INT16:
                    var dint16 = data.datap as Int16[];
                    return Array.ConvertAll<Int16, Int32>(dint16, Convert.ToInt32);
                case NPY_TYPES.NPY_UINT32:
                    var duint32 = data.datap as UInt32[];
                    return Array.ConvertAll<UInt32, Int32>(duint32, Convert.ToInt32);
                case NPY_TYPES.NPY_INT32:
                    var dint32 = data.datap as Int32[];
                    return Array.ConvertAll<Int32, Int32>(dint32, Convert.ToInt32);
                case NPY_TYPES.NPY_INT64:
                    var dint64 = data.datap as Int64[];
                    return Array.ConvertAll<Int64, Int32>(dint64, Convert.ToInt32);
                case NPY_TYPES.NPY_UINT64:
                    var duint64 = data.datap as UInt64[];
                    return Array.ConvertAll<UInt64, Int32>(duint64, Convert.ToInt32);
                case NPY_TYPES.NPY_FLOAT:
                    var float1 = data.datap as float[];
                    return Array.ConvertAll<float, Int32>(float1, Convert.ToInt32);
                case NPY_TYPES.NPY_DOUBLE:
                    var double1 = data.datap as double[];
                    return Array.ConvertAll<double, Int32>(double1, Convert.ToInt32);
                case NPY_TYPES.NPY_DECIMAL:
                    var decimal1 = data.datap as decimal[];
                    return Array.ConvertAll<decimal, Int32>(decimal1, Convert.ToInt32);
                default:
                    throw new Exception("Unsupported data type");
            }

        }

        private static Int64[] ConvertToInt64(VoidPtr data)
        {
            switch (data.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    var dbool = data.datap as bool[];
                    return Array.ConvertAll<bool, Int64>(dbool, Convert.ToInt64);
                case NPY_TYPES.NPY_BYTE:
                    var dbyte = data.datap as byte[];
                    return Array.ConvertAll<byte, Int64>(dbyte, Convert.ToInt64);
                case NPY_TYPES.NPY_UBYTE:
                    var dsbyte = data.datap as sbyte[];
                    return Array.ConvertAll<sbyte, Int64>(dsbyte, Convert.ToInt64);
                case NPY_TYPES.NPY_UINT16:
                    var duint16 = data.datap as UInt16[];
                    return Array.ConvertAll<UInt16, Int64>(duint16, Convert.ToInt64);
                case NPY_TYPES.NPY_INT16:
                    var dint16 = data.datap as Int16[];
                    return Array.ConvertAll<Int16, Int64>(dint16, Convert.ToInt64);
                case NPY_TYPES.NPY_UINT32:
                    var duint32 = data.datap as UInt32[];
                    return Array.ConvertAll<UInt32, Int64>(duint32, Convert.ToInt64);
                case NPY_TYPES.NPY_INT32:
                    var dint32 = data.datap as Int32[];
                    return Array.ConvertAll<Int32, Int64>(dint32, Convert.ToInt64);
                case NPY_TYPES.NPY_INT64:
                    var dint64 = data.datap as Int64[];
                    return Array.ConvertAll<Int64, Int64>(dint64, Convert.ToInt64);
                case NPY_TYPES.NPY_UINT64:
                    var duint64 = data.datap as UInt64[];
                    return Array.ConvertAll<UInt64, Int64>(duint64, Convert.ToInt64);
                case NPY_TYPES.NPY_FLOAT:
                    var float1 = data.datap as float[];
                    return Array.ConvertAll<float, Int64>(float1, Convert.ToInt64);
                case NPY_TYPES.NPY_DOUBLE:
                    var double1 = data.datap as double[];
                    return Array.ConvertAll<double, Int64>(double1, Convert.ToInt64);
                case NPY_TYPES.NPY_DECIMAL:
                    var decimal1 = data.datap as decimal[];
                    return Array.ConvertAll<decimal, Int64>(decimal1, Convert.ToInt64);
                default:
                    throw new Exception("Unsupported data type");
            }

        }

        private static byte[] ConvertToByte(Int32[] data)
        {
            var AdjustedArray = Array.ConvertAll<Int32, byte>(data, Convert.ToByte);
            return AdjustedArray;
        }


        #region sorting algorithms
        private static void qsort(VoidPtr ip, npy_intp ip_index, npy_intp length, int elsize, Func<object, object, int> qsortCompare)
        {
            NpyArray_SortFunc(new VoidPtr(ip, ip_index), length, null, NPY_SORTKIND.NPY_QUICKSORT);
        }

        /*
        * These algorithms use special sorting.  They are not called unless the
        * underlying sort function for the type is available.  Note that axis is
        * already valid. The sort functions require 1-d contiguous and well-behaved
        * data.  Therefore, a copy will be made of the data if needed before handing
        * it to the sorting routine.  An iterator is constructed and adjusted to walk
        * over all but the desired sorting axis.
        */
   
        static int _new_sort(NpyArray op, int axis, NPY_SORTKIND kind)
        {
            NpyArrayIterObject it;
            bool needcopy = false;
            bool swap;
            npy_intp N;
            npy_intp size;
            int elsize, eldiv;
            npy_intp astride;
            NpyArray_SortFunc sort;

            it = NpyArray_IterAllButAxis(op, ref axis);
            swap = !NpyArray_ISNOTSWAPPED(op);
            if (it == null)
            {
                return -1;
            }

            sort = op.descr.f.sort[(int)kind];
            size = it.size;
            N = op.dimensions[axis];
            elsize = op.descr.elsize;
            eldiv = GetDivSize(elsize);
            astride = op.strides[axis];

            needcopy = !((op.flags & NPYARRAYFLAGS.NPY_ALIGNED) != 0) || (astride != (npy_intp)elsize) || swap;

            if (needcopy)
            {
                var parallelIters = NpyArray_ITER_ParallelSplit(it, numpyinternal.maxSortOperationParallelSize);

                Parallel.For(0, parallelIters.Count(), index =>
                //for (int index = 0; index < parallelIters.Count(); index++) // 
                {
                    var paraIter = parallelIters.ElementAt(index);

                    VoidPtr buffer = NpyDataMem_NEW(op.descr.type_num, (ulong)(N * elsize));

                    var helper1 = MemCopy.GetMemcopyHelper(buffer);
                    helper1.strided_byte_copy_init(buffer, elsize, paraIter.dataptr, astride, elsize, eldiv);

                    var helper2 = MemCopy.GetMemcopyHelper(paraIter.dataptr);
                    helper2.strided_byte_copy_init(paraIter.dataptr, astride, buffer, elsize, elsize, eldiv);

                    while (paraIter.index < paraIter.size)
                    {
                        helper1.strided_byte_copy(buffer.data_offset, paraIter.dataptr.data_offset, N);
                        if (swap)
                        {
                            _strided_byte_swap(buffer, (npy_intp)elsize, N, elsize);
                        }
                        if (sort(buffer, N, op, kind) < 0)
                        {
                            NpyDataMem_FREE(buffer);
                            break;
                        }
                        if (swap)
                        {
                            _strided_byte_swap(buffer, (npy_intp)elsize, N, elsize);
                        }
                        helper2.strided_byte_copy(paraIter.dataptr.data_offset, buffer.data_offset, N);

                        NpyArray_ITER_NEXT(paraIter);
                    }
                    NpyDataMem_FREE(buffer);
                });


            }
            else
            {

                // NpyArray_ITER_ParallelSplit does not work if run in parallel
                while (size-- > 0)
                {
                    if (sort(it.dataptr, N, op, kind) < 0)
                    {
                        goto fail;
                    }
                    NpyArray_ITER_NEXT(it);
                }
            }
            Npy_DECREF(it);
            return 0;

        fail:
            Npy_DECREF(it);
            return 0;
        }


        static NpyArray _new_argsort(NpyArray op, int axis, NPY_SORTKIND kind)
        {

            NpyArrayIterObject it = null;
            NpyArrayIterObject rit = null;
            NpyArray ret;
            bool needcopy = false;
            int i;
            npy_intp N, size;
            int elsize, eldiv;
            bool swap;
            npy_intp astride, rstride; 
            VoidPtr iptr;
            NpyArray_ArgSortFunc argsort;

            ret = NpyArray_New(null, op.nd,
                               op.dimensions, NPY_TYPES.NPY_INTP,
                               null, null, 0, 0, Npy_INTERFACE(op));
            if (ret == null)
            {
                return null;
            }
            it = NpyArray_IterAllButAxis(op, ref axis);
            rit = NpyArray_IterAllButAxis(ret, ref axis);
            if (rit == null || it == null)
            {
                goto fail;
            }
            swap = !NpyArray_ISNOTSWAPPED(op);

            argsort = op.descr.f.argsort[(int)kind];
            size = it.size;
            N = op.dimensions[axis];
            elsize = op.descr.elsize;
            eldiv = GetDivSize(elsize);
            astride = op.strides[axis];
            rstride = NpyArray_STRIDE(ret, axis);

            needcopy = swap || !((op.flags &  NPYARRAYFLAGS.NPY_ALIGNED) != 0) ||
                (astride != (npy_intp)elsize) || (rstride != sizeof(npy_intp));
            if (needcopy)
            {
                VoidPtr valbuffer, indbuffer;

                valbuffer = NpyDataMem_NEW(op.ItemType, (ulong)(N * elsize));
                indbuffer = NpyDataMem_NEW(NPY_TYPES.NPY_INTP, (ulong)(N * sizeof(npy_intp)));

                var helper1 = MemCopy.GetMemcopyHelper(valbuffer);
                helper1.strided_byte_copy_init(valbuffer, elsize, it.dataptr, astride, elsize, eldiv);

                var helper2 = MemCopy.GetMemcopyHelper(rit.dataptr);
                helper2.strided_byte_copy_init(rit.dataptr, rstride, indbuffer, sizeof(npy_intp), sizeof(npy_intp), GetDivSize(sizeof(npy_intp)));

                while (size-- > 0)
                {
                    helper1.strided_byte_copy(valbuffer.data_offset, it.dataptr.data_offset, N);
                    if (swap)
                    {
                        _strided_byte_swap(valbuffer, (npy_intp)elsize, N, elsize);
                    }
                    iptr = indbuffer;
                    BuildINTPArray(iptr, N);

                    if (argsort(valbuffer, iptr, N, op, kind) < 0)
                    {
                        NpyDataMem_FREE(valbuffer);
                        NpyDataMem_FREE(indbuffer);
                        goto fail;
                    }
                    helper2.strided_byte_copy(rit.dataptr.data_offset, indbuffer.data_offset, N);
                    NpyArray_ITER_NEXT(it);
                    NpyArray_ITER_NEXT(rit);
                }
                NpyDataMem_FREE(valbuffer);
                NpyDataMem_FREE(indbuffer);
            }
            else
            {
                if (size <= 1)
                {
                    if (argsort(it.dataptr, rit.dataptr, N, op, kind) < 0)
                    {
                        goto fail;
                    }
                }
                else
                {
                    int errorsDetected = 0;

                    var itParallelIters = NpyArray_ITER_ParallelSplit(it);
                    var ritParallelIters = NpyArray_ITER_ParallelSplit(rit);

                    Parallel.For(0, itParallelIters.Count(), index =>
                    {
                        var litIter = itParallelIters.ElementAt(index);
                        var lritIter = ritParallelIters.ElementAt(index);

                        while (litIter.index < litIter.size)
                        {
                            var intpArray = lritIter.dataptr;

                            BuildINTPArray(intpArray, N);

                            if (argsort(litIter.dataptr, intpArray, N, op, kind) < 0)
                            {
                                errorsDetected++;
                            }

                            NpyArray_ITER_NEXT(litIter);
                            NpyArray_ITER_NEXT(lritIter);
                        }

                    });

                    if (errorsDetected > 0)
                        goto fail;
                }

            }


            Npy_DECREF(it);
            Npy_DECREF(rit);
            return ret;

            fail:
            Npy_DECREF(ret);
            Npy_XDECREF(it);
            Npy_XDECREF(rit);
            return null;
        }

        internal static void BuildINTPArray(VoidPtr iptr, npy_intp N)
        {
            var intpArr = iptr.datap as npy_intp[];

            // lets do the divide only once
            npy_intp data_offset = iptr.data_offset >> IntpDivSize;

            if (iptr.data_offset >= 0)
            {
                for (int i = 0; i < N; i++)
                {
                    var index = data_offset + i;
                    intpArr[index] = i;
                }
            }
            else
            {
                for (int i = 0; i < N; i++)
                {
                    var index = data_offset + i;
                    index = AdjustNegativeIndex<npy_intp>(iptr, index);

                    intpArr[index] = i;
                }
            }

  
        }


        /** @brief Use bisection of sorted array to find first entries >= keys.
        *
        * For each key use bisection to find the first index i s.t. key <= arr[i].
        * When there is no such index i, set i = len(arr). Return the results in ret.
        * All arrays are assumed contiguous on entry and both arr and key must be of
        * the same comparable type.
        *
        * @param arr contiguous sorted array to be searched.
        * @param key contiguous array of keys.
        * @param ret contiguous array of intp for returned indices.
        * @return void
        */
        static void local_search_left(NpyArray arr, NpyArray key, NpyArray ret)
        {
            NpyArray_CompareFunc compare = key.descr.f.compare;
            npy_intp nelts = arr.dimensions[arr.nd - 1];
            npy_intp nkeys = NpyArray_SIZE(key);
            VoidPtr parr = new VoidPtr(arr);
            VoidPtr pkey = new VoidPtr(key);
            npy_intp[] pret = ret.data.datap as npy_intp[];
            int pret_index = 0;
            int elsize = arr.descr.elsize;
            npy_intp i;

            for (i = 0; i < nkeys; ++i)
            {
                npy_intp imin = 0;
                npy_intp imax = nelts;
                while (imin < imax)
                {
                    npy_intp imid = imin + ((imax - imin) >> 1);

                    if (compare(parr + elsize * imid, pkey, elsize, key) < 0)
                    {
                        imin = imid + 1;
                    }
                    else
                    {
                        imax = imid;
                    }
                }
                pret[pret_index] = imin;
                pret_index += 1;
                pkey.data_offset += elsize;
            }
        }


        /** @brief Use bisection of sorted array to find first entries > keys.
         *
         * For each key use bisection to find the first index i s.t. key < arr[i].
         * When there is no such index i, set i = len(arr). Return the results in ret.
         * All arrays are assumed contiguous on entry and both arr and key must be of
         * the same comparable type.
         *
         * @param arr contiguous sorted array to be searched.
         * @param key contiguous array of keys.
         * @param ret contiguous array of intp for returned indices.
         * @return void
         */
        static void  local_search_right(NpyArray arr, NpyArray key, NpyArray ret)
        {
            NpyArray_CompareFunc compare = key.descr.f.compare;
            npy_intp nelts = arr.dimensions[arr.nd - 1];
            npy_intp nkeys = NpyArray_SIZE(key);
            VoidPtr parr = new VoidPtr(arr);
            VoidPtr pkey = new VoidPtr(key);
            npy_intp[] pret = ret.data.datap as npy_intp[];
            int pret_index = 0;
            int elsize = arr.descr.elsize;
            npy_intp i;

            for (i = 0; i < nkeys; ++i)
            {
                npy_intp imin = 0;
                npy_intp imax = nelts;
                while (imin < imax)
                {
                    npy_intp imid = imin + ((imax - imin) >> 1);

                    if (compare(parr + elsize * imid, pkey, elsize, key) <= 0)
                    {
                        imin = imid + 1;
                    }
                    else
                    {
                        imax = imid;
                    }
                }
                pret[pret_index] = imin;
                pret_index += 1;
                pkey.data_offset += elsize;
            }
        }


        #endregion

#region SWAP functions

        /*
         * Consumes reference to ap (op gets it) op contains a version of
         * the array with axes swapped if local variable axis is not the
         * last dimension.  Origin must be defined locally.
         */
        private static bool SWAPAXES(ref NpyArray op, ref NpyArray ap, ref int axis, ref int orign)
        {
            orign = ap.nd - 1;
            if (axis != orign)
            {
                op = NpyArray_SwapAxes(ap, axis, orign);
                Npy_DECREF(ap);
                if (op == null)
                    return false;
            }
            else
            {
                op = ap;
            }
            return true;
        }

        private static bool SWAPBACK(ref NpyArray op, ref NpyArray ap, ref int axis, ref int orign)
        {
            if (axis != orign)
            {
                op = NpyArray_SwapAxes(ap, axis, orign);
                Npy_DECREF(ap);
                if (op == null)
                    return false;
            }
            else
            {
                op = ap;
            }

            return true;
        }


        private static void SWAPAXES2(NpyArray ap, ref int axis, ref int orign)
        {
            orign = ap.nd - 1;
            if (axis != orign)
            {
                SWAPINTP(ap.dimensions[axis], ap.dimensions[orign]);
                SWAPINTP(ap.strides[axis], ap.strides[orign]);
                NpyArray_UpdateFlags(ap, NPYARRAYFLAGS.NPY_CONTIGUOUS | NPYARRAYFLAGS.NPY_FORTRAN);
            }
        }

        private static void SWAPBACK2(NpyArray ap, ref int axis, ref int orign)
        {
            if (axis != orign)
            {
                SWAPINTP(ap.dimensions[axis], ap.dimensions[orign]);
                SWAPINTP(ap.strides[axis], ap.strides[orign]);
                NpyArray_UpdateFlags(ap, NPYARRAYFLAGS.NPY_CONTIGUOUS | NPYARRAYFLAGS.NPY_FORTRAN);
            }
        }

        private static void SWAPINTP(npy_intp a, npy_intp b)
        {
            npy_intp c;
            c = a; a = b; b = c;
        }

   
#endregion

    }
}
