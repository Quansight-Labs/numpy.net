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

#define USE_REFCHECK
//#define ENABLELOCKING
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security;
using System.Text;
using System.Runtime.InteropServices;
using System.Threading;
using NumpyLib;

#if NPY_INTP_64
using npy_intp = System.Int64;
using npy_ucs4 = System.Int64;
#else
using npy_intp = System.Int32;
using npy_ucs4 = System.Int32;
#endif

using size_t = System.UInt64;
using npy_datetime = System.Int64;
using npy_timedelta = System.Int64;
using System.IO;

namespace NumpyDotNet {
    /// <summary>
    /// NpyCoreApi class wraps the interactions with the libndarray core library. It
    /// also makes use of NpyAccessLib.dll for a few functions that must be
    /// implemented in native code.
    ///
    /// TODO: This class is going to get very large.  Not sure if it's better to
    /// try to break it up or just use partial classes and split it across
    /// multiple files.
    /// </summary>
    [SuppressUnmanagedCodeSecurity]
    internal static class NpyCoreApi {

        /// <summary>
        /// Stupid hack to allow us to pass an already-allocated wrapper instance
        /// through the interfaceData argument and tell the wrapper creation functions
        /// like ArrayNewWrapper to use an existing instance instead of creating a new
        /// one.  This is necessary because CPython does construction as an allocator
        /// but .NET only triggers code after allocation.
        /// </summary>
        internal struct UseExistingWrapper
        {
            internal object Wrapper;
        }

        #region API Wrappers

        /// <summary>
        /// Returns a new descriptor object for internal types or user defined
        /// types.
        /// </summary>
        internal static dtype DescrFromType(NPY_TYPES type) {
            // NOTE: No GIL wrapping here, function is re-entrant and includes locking.
            NpyArray_Descr descr = numpyAPI.NpyArray_DescrFromType(type);
            CheckError();
            return new dtype(descr);
        }

        internal static bool IsAligned(ndarray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.Npy_IsAligned(arr.Array);
            }
        }

        internal static bool IsWriteable(ndarray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.Npy_IsWriteable(arr.Array);
            }
        }

        internal static byte OppositeByteOrder
        {
            get { return oppositeByteOrder; }
        }

        internal static byte NativeByteOrder
        {
            get { return (oppositeByteOrder == '<') ? (byte)'>' : (byte)'<'; }
        }

        internal static dtype SmallType(dtype t1, dtype t2)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new dtype(numpyAPI.NpyArray_SmallType(t1.Descr, t2.Descr));
            }
        }


        /// <summary>
        /// Moves the contents of src into dest.  Arrays are assumed to have the
        /// same number of elements, but can be different sizes and different types.
        /// </summary>
        /// <param name="dest">Destination array</param>
        /// <param name="src">Source array</param>
        internal static void MoveInto(ndarray dest, ndarray src)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                if (numpyAPI.NpyArray_MoveInto(dest.Array, src.Array) == -1)
                {
                    CheckError();
                }
            }
        }

        /// <summary>
        /// combines two ndarrays together
        /// </summary>
        /// <param name="dest"></param>
        /// <param name="src"></param>
        /// <returns></returns>
        internal static ndarray Combine(ndarray dest, ndarray src)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                NpyArray newArray = numpyAPI.NpyArray_Combine(dest.Array, src.Array);
                if (newArray == null)
                {
                    CheckError();
                    return null;
                }

                return new ndarray(newArray);
            }
        }

        /// <summary>
        /// appends multiple ndarrays together
        /// </summary>
        /// <param name="dest"></param>
        /// <param name="src"></param>
        /// <returns></returns>
        internal static int CombineInto(ndarray dest, IEnumerable<ndarray> ndarrays)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                List<NpyArray> arrays = new List<NpyArray>();
                foreach (var ndarray in ndarrays)
                {
                    arrays.Add(ndarray.Array);
                }

                int result = numpyAPI.NpyArray_CombineInto(dest.Array, arrays);
                if (result < 0)
                {
                    CheckError();
                    return result;
                }

                return result;
            }
        }



        /// <summary>
        /// Allocates a new array and returns the ndarray wrapper
        /// </summary>
        /// <param name="descr">Type descriptor</param>
        /// <param name="numdim">Num of dimensions</param>
        /// <param name="dimensions">Size of each dimension</param>
        /// <param name="fortran">True if Fortran layout, false for C layout</param>
        /// <returns>Newly allocated array</returns>
        internal static ndarray AllocArray(dtype descr, int numdim, npy_intp[] dimensions, bool fortran)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Alloc(descr.Descr, numdim, dimensions, fortran, null));
            }
        }

        /// <summary>
        /// Constructs a new array from an input array and descriptor type.  The
        /// Underlying array may or may not be copied depending on the requirements.
        /// </summary>
        /// <param name="src">Source array</param>
        /// <param name="descr">Desired type</param>
        /// <param name="flags">New array flags</param>
        /// <returns>New array (may be source array)</returns>
        internal static ndarray FromArray(ndarray src, dtype descr, NPYARRAYFLAGS flags)
        {
            if (descr == null && flags == 0)
                return src;
            if (descr == null)
                descr = src.Dtype;
            if (descr != null)
                NpyCoreApi.Incref(descr.Descr);


#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_FromArray(src.Array, descr.Descr, flags));
            }
        }


        /// <summary>
        /// Returns an array with the size or stride of each dimension in the given array.
        /// </summary>
        /// <param name="arr">The array</param>
        /// <param name="getDims">True returns size of each dimension, false returns stride of each dimension</param>
        /// <returns>Array w/ an array size or stride for each dimension</returns>
        internal static npy_intp[] GetArrayDimsOrStrides(ndarray arr, bool getDims)
        {
            npy_intp[] retArr;
            npy_intp[] srcPtr;
            if (getDims)
            {
                srcPtr = arr.Array.dimensions;
            }
            else
            {
                srcPtr = arr.Array.strides;
            }

            retArr = new npy_intp[arr.ndim];

            for (int i = 0; i < arr.ndim; i++)
            {
                retArr[i] = srcPtr[i];
            }
            return retArr;
        }

        /// <summary>
        /// sets an array with the size or stride of each dimension in the given array.
        /// </summary>
        /// <param name="arr">The array</param>
        /// <param name="setDims">True returns size of each dimension, false returns stride of each dimension</param>
        /// <returns>Array w/ an array size or stride for each dimension</returns>
        internal static void SetArrayDimsOrStrides(ndarray arr, npy_intp[] dim_or_strides, int len, bool setDims)
        {
            if (setDims)
            {
                arr.Array.dimensions = dim_or_strides;
                arr.Array.nd = len;
            }
            else
            {
                arr.Array.strides = dim_or_strides;
                arr.Array.nd = len;
            }

        }
        [Conditional("USE_REFCHECK")]
        internal static void Incref(NpyArray_Descr obj)
        {
            numpyAPI.NpyArrayAccess_Incref(obj);
        }
        [Conditional("USE_REFCHECK")]
        internal static void Decref(NpyArray_Descr obj)
        {
            numpyAPI.NpyArrayAccess_Decref(obj);
        }

        [Conditional("USE_REFCHECK")]
        internal static void Incref(NpyArray obj)
        {
            numpyAPI.NpyArrayAccess_Incref(obj);
        }

        [Conditional("USE_REFCHECK")]
        internal static void Decref(NpyArray obj)
        {
            numpyAPI.NpyArrayAccess_Decref(obj);
        }

        internal static ndarray NewFromDescr(dtype descr, npy_intp[] dims, npy_intp[] strides, NPYARRAYFLAGS flags, object interfaceData)
        {
            Incref(descr.Descr);
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_NewFromDescr(descr.Descr, dims.Length, dims, strides, null, flags, false, null, interfaceData));
            }
        }

        internal static ndarray NewFromDescr(dtype descr, npy_intp[] dims, npy_intp[] strides, VoidPtr data, NPYARRAYFLAGS flags, object interfaceData) {

            Incref(descr.Descr);
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_NewFromDescr(descr.Descr, dims.Length, dims, strides, data, flags, false, null, interfaceData));
            }
        }

        internal static flatiter IterNew(ndarray ao)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new flatiter(numpyAPI.NpyArray_IterNew(ao.core));
            }
        }


        internal static NpyArrayIterObject BroadcastToShape(ndarray ao, npy_intp[] dims, int nd)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_BroadcastToShape(ao.core, dims, nd);
            }
        }

        internal static ndarray IterSubscript(flatiter iter, NpyIndexes indexes)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_IterSubscript(iter.Iter, indexes.Indexes, indexes.NumIndexes));
            }
        }

        internal static void IterSubscriptAssign(flatiter iter, NpyIndexes indexes, ndarray val)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                if (numpyAPI.NpyArray_IterSubscriptAssign(iter.Iter, indexes.Indexes, indexes.NumIndexes, val.Array) < 0)
                {
                    CheckError();
                }
            }
        }

        internal static void GetItems(ndarray a, object [] buffer, npy_intp index, npy_intp length)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.GetItems(a.Array, buffer, index, length);
            }
 
        }

        internal static ndarray FlatView(ndarray a)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_FlatView(a.Array));
            }
        }


        /// <summary>
        /// Creates a multiterator
        /// </summary>
        /// <param name="objs">Sequence of objects to iterate over</param>
        /// <returns>Pointer to core multi-iterator structure</returns>
        internal static NpyArrayMultiIterObject MultiIterFromObjects(IEnumerable<object> objs)
        {
            return MultiIterFromArrays(objs.Select(x => np.FromAny(null, (dtype)x)));
        }

        internal static NpyArrayMultiIterObject MultiIterFromArrays(IEnumerable<ndarray> arrays)
        {
            NpyArray[] coreArrays = arrays.Select(x => { Incref(x.Array); return x.Array; }).ToArray();
            NpyArrayMultiIterObject result;

#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                result = numpyAPI.NpyArrayAccess_MultiIterFromArrays(coreArrays, coreArrays.Length);
            }
            CheckError();
            return result;
        }


        internal static void MultiIterReset(NpyArrayMultiIterObject multi)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_Reset(multi);
            }
        }

        internal static void MultiIterNext(NpyArrayMultiIterObject multi)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_Next(multi);
            }
        }

        internal static bool MultiIterDone(NpyArrayMultiIterObject multi)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_NotDone(multi);
            }
        }

        internal static VoidPtr MultiIterData(NpyArrayMultiIterObject multi, int index)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_MultiIter_DATA(multi, index);
            }
        }
        internal static int MultiIterBroadcast(NpyArrayMultiIterObject multi)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_Broadcast(multi);
            }
        }

        internal static ndarray PerformNumericOp(ndarray a, UFuncOperation ops, object operand, bool UseSrcAsDest = false)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                ndarray @out = null;
                if (UseSrcAsDest)
                    @out = a;

                var b = np.asanyarray(operand);

                return new ndarray(numpyAPI.NpyArray_PerformNumericOperation(ops, a.Array, b != null ? b.Array : null, @out != null ? @out.Array : null, null));
            }

        }

        internal static ndarray PerformNumericOp(ndarray a, UFuncOperation ops, ndarray b, bool UseSrcAsDest = false)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                ndarray @out = null;
                if (UseSrcAsDest)
                    @out = a;

                return new ndarray(numpyAPI.NpyArray_PerformNumericOperation(ops, a.Array, b != null ? b.Array : null, @out != null ? @out.Array : null, null));
            }

        }

        internal static ndarray PerformUFUNC(UFuncOperation ops, ndarray x1, ndarray x2, ndarray @out, ndarray where)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_PerformNumericOperation(ops, x1.Array, x2 != null ? x2.Array : null, @out != null ? @out.Array : null, where != null ? where.Array : null));
            }

        }


        internal static ndarray PerformOuterOp(ndarray a, ndarray b, ndarray dest, UFuncOperation ops)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_PerformOuterOp(a.Array, b.Array, dest.Array, ops));
            }

        }

        internal static ndarray PerformReduceOp(ndarray a, int axis, UFuncOperation ops, NPY_TYPES rtype, ndarray @out = null, bool keepdims = false)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_PerformReduceOp(a.Array, axis, ops, rtype, @out == null ? null : @out.Array, keepdims));
            }

        }

        internal static ndarray PerformReduceAtOp(ndarray a, ndarray indices, int axis, UFuncOperation ops, NPY_TYPES rtype, ndarray @out = null)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_PerformReduceAtOp(a.Array, indices.Array, axis, ops, rtype, @out == null ? null : @out.Array));
            }

        }

        internal static ndarray PerformAccumulateOp(ndarray a, int axis, UFuncOperation ops, NPY_TYPES rtype, ndarray @out = null)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_PerformAccumulateOp(a.Array, axis, ops, rtype, @out == null ? null : @out.Array));
            }

        }

        internal static IArrayHandlers GetArrayHandler(NPY_TYPES ItemType)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.GetArrayHandler(ItemType);
            }
        }


        internal static void SetArrayHandler(NPY_TYPES ItemType, IArrayHandlers Handlers)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.SetArrayHandler(ItemType, Handlers);
            }
        }

        internal static NpyUFuncObject GetNumericOp(UFuncOperation op)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_GetNumericOp(op);
            }
        }


        internal static ndarray Byteswap(ndarray arr, bool inplace)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Byteswap(arr.Array, inplace));
            }
        }

        internal static ndarray CastToType(ndarray arr, dtype d, bool fortran)
        {
            Incref(d.Descr);
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_CastToType(arr.Array, d.Descr, fortran));
            }
        }

        internal static ndarray CheckAxis(ndarray arr, ref int axis, NPYARRAYFLAGS flags)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_CheckAxis(arr.Array, ref axis, flags));
            }
        }

        internal static void CopyAnyInto(ndarray dest, ndarray src)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                if (numpyAPI.NpyArray_CopyAnyInto(dest.Array, src.Array) < 0)
                {
                    CheckError();
                }
            }
        }

        internal static void DescrDestroyFields(NpyDict fields)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyDict_Destroy(fields);
            }
        }


        internal static ndarray GetField(ndarray arr, dtype d, int offset)
        {
            Incref(d.Descr);
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_GetField(arr.Array, d.Descr, offset));
            }
        }

        internal static ndarray GetImag(ndarray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_GetImag(arr.Array));
            }
        }

        internal static ndarray GetReal(ndarray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_GetReal(arr.Array));
            }
        }

        internal static ndarray GetField(ndarray arr, string name)
        {
            NpyArray_DescrField field = GetDescrField(arr.Dtype, name);
            dtype field_dtype = new dtype(field.descr);
            return GetField(arr, field_dtype, field.offset);
        }

        internal static ndarray Newshape(ndarray arr, NpyArray_Dims dims, NPY_ORDER order)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Newshape(arr.Array, dims, order));
            }
        }

        internal static ndarray Newshape(ndarray arr, npy_intp[] dims, NPY_ORDER order)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                NpyArray_Dims newDims = new NpyArray_Dims()
                {
                    ptr = dims,
                    len = dims.Length,
                };
                return new ndarray(numpyAPI.NpyArray_Newshape(arr.Array, newDims, order));
            }
        }

        internal static void SetShape(ndarray arr, NpyArray_Dims dims)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                if (numpyAPI.NpyArray_SetShape(arr.Array, dims) < 0)
                {
                    CheckError();
                }
            }
        }

        internal static void SetState(ndarray arr, npy_intp[] dims, NPY_ORDER order, string rawdata)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArrayAccess_SetState(arr.Array, dims.Length, dims, order, rawdata, (rawdata != null) ? rawdata.Length : 0);
            }
            CheckError();
        }


        internal static ndarray NewView(dtype d, int nd, npy_intp[] dims, npy_intp[] strides,
            ndarray arr, npy_intp offset, bool ensure_array)
        {
            Incref(d.Descr);
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_NewView(d.Descr, nd, dims, strides, arr.Array, offset, ensure_array));
            }
        }

        /// <summary>
        /// Returns a copy of the passed array in the specified order (C, Fortran)
        /// </summary>
        /// <param name="arr">Array to copy</param>
        /// <param name="order">Desired order</param>
        /// <returns>New array</returns>
        internal static ndarray NewCopy(ndarray arr, NPY_ORDER order)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_NewCopy(arr.Array, order));
            }
        }

        internal static void AddField(NpyDict fields, List<string> names, int i, string name, dtype fieldType, int offset, string title)
        {
            Incref(fieldType.Descr);
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                if (numpyAPI.NpyArrayAccess_AddField(fields, names, i, name, fieldType.Descr, offset, title) < 0)
                {
                    CheckError();
                }
            }
        }

        internal static NpyArray_DescrField GetDescrField(dtype d, string name)
        {
            NpyArray_DescrField result = null;
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                if (numpyAPI.NpyArrayAccess_GetDescrField(d.Descr, name, ref result) < 0)
                {
                    throw new ArgumentException(String.Format("Field {0} does not exist", name));
                }
            }
            return result;
        }

        internal static dtype DescrNewSubarray(dtype basetype, npy_intp[] shape)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new dtype(numpyAPI.NpyArray_DescrNewSubarray(basetype.Descr, shape.Length, shape));
            }
        }

        internal static dtype DescrNew(dtype d)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new dtype(numpyAPI.NpyArray_DescrNew(d.Descr));
            }
        }

        internal static void GetBytes(ndarray arr, byte[] bytes, NPY_ORDER order)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {

                if (arr.IsCArray && (order == NPY_ORDER.NPY_CORDER || order == NPY_ORDER.NPY_ANYORDER))
                {
                    numpyAPI.MemCpy(new VoidPtr(bytes), 0, arr.Array.data, 0, bytes.LongLength);
                }
                else
                {
                    ndarray arrcopy = arr.Copy(order);
                    numpyAPI.MemCpy(new VoidPtr(bytes), 0, arrcopy.Array.data, 0, bytes.LongLength);
                }

            }
        }

        internal static void FillWithScalar(ndarray arr, ndarray zero_d_array)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                if (numpyAPI.NpyArray_FillWithScalar(arr.Array, zero_d_array.Array) < 0)
                {
                    CheckError();
                }
            }
        }

        internal static ndarray View(ndarray arr, dtype d, object subtype)
        {
            NpyArray_Descr descr = (d == null ? null : d.Descr);
            if (descr != null)
            {
                Incref(descr);
            }

#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                if (subtype != null)
                {
                    return new ndarray(numpyAPI.NpyArray_View(arr.Array, descr, subtype));
                }
                else
                {
                    return new ndarray(numpyAPI.NpyArray_View(arr.Array, descr, null));
                }
            }

        }

        internal static ndarray ViewLike(ndarray arr, ndarray proto)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArrayAccess_ViewLike(arr.Array, proto.Array));
            }
        }

        internal static ndarray Subarray(ndarray self, VoidPtr dataptr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Subarray(self.Array, dataptr));
            }
        }

        internal static IList<npy_intp> IndicesFromAxis(ndarray self, int axis)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_IndexesFromAxis(self.Array, axis);
            }
        }

        internal static dtype DescrNewByteorder(dtype d, char order)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new dtype(numpyAPI.NpyArray_DescrNewByteorder(d.Descr, order));
            }
        }

        internal static void UpdateFlags(ndarray arr, NPYARRAYFLAGS flagmask)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_UpdateFlags(arr.Array, flagmask);
            }
        }

        /// <summary>
        /// Calls the fill function on the array dtype.  This takes the first 2 values in the array and fills the array
        /// so the difference between each pair of elements is the same.
        /// </summary>
        /// <param name="arr"></param>
        internal static void Fill(ndarray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                if (numpyAPI.NpyArrayAccess_Fill(arr.Array) < 0)
                {
                    CheckError();
                }
            }
        }

        internal static dtype InheritDescriptor(dtype t1, dtype other)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new dtype(numpyAPI.NpyArrayAccess_InheritDescriptor(t1.Descr, other.Descr));
            }
        }

        internal static bool EquivTypes(dtype d1, dtype d2)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_EquivTypes(d1.Descr, d2.Descr);
            }
        }

        internal static void CopyTo(ndarray dst, ndarray src, NPY_CASTING casting, ndarray wheremask_in = null)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_CopyTo(dst.Array, src.Array, casting, wheremask_in != null ? wheremask_in.Array : null);
            }
        }


        internal static void Place(ndarray arr, ndarray mask, ndarray vals)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_Place(arr.Array, mask.Array, vals.Array);
            }

        }


        internal static bool CanCastTo(dtype d1, dtype d2)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_CanCastTo(d1.Descr, d2.Descr);
            }
        }

        /// <summary>
        /// Reads the specified text or binary file and produces an array from the content.  Currently only
        /// the file name is allowed and not a PythonFile or Stream type due to limitations in the core
        /// (assumes FILE *).
        /// </summary>
        /// <param name="fileName">File to read</param>
        /// <param name="type">Type descriptor for the resulting array</param>
        /// <param name="count">Number of elements to read, less than zero reads all available</param>
        /// <param name="sep">Element separator string for text files, null for binary files</param>
        /// <returns>Array of file contents</returns>
        internal static ndarray ArrayFromFile(string fileName, dtype type, int count, string sep)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArrayAccess_FromFile(fileName, (type != null) ? type.Descr : null, count, sep));
            }
        }

        /// <summary>
        /// Reads the specified text or binary file and produces an array from the content.  Currently only
        /// the file name is allowed and not a PythonFile or Stream type due to limitations in the core
        /// (assumes FILE *).
        /// </summary>
        /// <param name="fileStream">File to read</param>
        /// <param name="type">Type descriptor for the resulting array</param>
        /// <param name="count">Number of elements to read, less than zero reads all available</param>
        /// <param name="sep">Element separator string for text files, null for binary files</param>
        /// <returns>Array of file contents</returns>
        internal static ndarray ArrayFromStream(Stream fileStream, dtype type, int count, string sep)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArrayAccess_FromStream(fileStream, (type != null) ? type.Descr : null, count, sep));
            }
        }

        /// <summary>
        /// writes the array contents to the specified file name
        /// </summary>
        /// <param name="arr">array with data to write to file</param>
        /// <param name="fileName">file name to write to</param>
        /// <param name="sep">Element separator string for text files, null for binary files</param>
        /// <param name="format">.NET format string to use for writing values</param>
        internal static void ArrayToFile(ndarray arr, string fileName, string sep, string format)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArrayAccess_ToFile(arr.Array, fileName, sep, format);
            }
        }

        /// <summary>
        /// writes the array contents to the specified file name
        /// </summary>
        /// <param name="arr">array with data to write to file</param>
        /// <param name="fileStream">file stream to write to</param>
        /// <param name="sep">Element separator string for text files, null for binary files</param>
        /// <param name="format">.NET format string to use for writing values</param>
        internal static void ArrayToStream(ndarray arr, Stream fileStream, string sep, string format)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArrayAccess_ToStream(arr.Array, fileStream, sep, format);
            }
        }


        internal static ndarray ArrayFromString(string data, dtype type, int count, string sep)
        {
            if (type != null) Incref(type.Descr);
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_FromString(data, data.Length, (type != null) ? type.Descr : null, count, sep));
            }
        }

        //internal static ndarray ArrayFromBytes(Bytes data, dtype type, int count, string sep) {
        //    if (type != null) Incref(type.Descr);
        //    lock (GlobalIterpLock) {
        //        return new ndarray(numpyAPI.NpyArray_FromString(data, data.Length, (type != null) ? type.Descr : null, count, sep));
        //    }
        //}

        internal static ndarray CompareStringArrays(ndarray a1, ndarray a2, NpyDefs.NPY_COMPARE_OP op,
                                                    bool rstrip = false)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_CompareStringArrays(a1.Array, a2.Array, (int)op, rstrip ? 1 : 0));
            }
        }

        // API Defintions: every native call is private and must currently be wrapped by a function
        // that at least holds the global interpreter lock (GlobalInterpLock).
        internal static int ElementStrides(ndarray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_ElementStrides(arr.Array);
            }
        }

        internal static npy_intp[] GetViewOffsets(ndarray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.GetViewOffsets(arr.Array);
            }
        }


        internal static npy_intp[] GetViewOffsets(NpyArrayIterObject iter, npy_intp count)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.GetViewOffsets(iter, count);
            }
        }

        internal static NpyArray ArraySubscript(ndarray arr, NpyIndexes indexes)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_Subscript(arr.Array, indexes.Indexes, indexes.NumIndexes);
            }
        }

        internal static void IndexDealloc(NpyIndexes indexes)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_IndexDealloc(indexes.Indexes, indexes.NumIndexes);
            }
        }

        internal static npy_intp ArraySize(ndarray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_Size(arr.Array);
            }
        }

        /// <summary>
        /// Indexes an array by a single long and returns the sub-array.
        /// </summary>
        /// <param name="index">The index into the array.</param>
        /// <returns>The sub-array.</returns>
        internal static ndarray ArrayItem(ndarray arr, npy_intp index)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_ArrayItem(arr.Array, index));
            }
        }

        internal static ndarray IndexSimple(ndarray arr, NpyIndexes indexes)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                var array = numpyAPI.NpyArray_IndexSimple(arr.Array, indexes.Indexes, indexes.NumIndexes);
                return array == null ? null : new ndarray(array);
            }
        }

        internal static int IndexFancyAssign(ndarray dest, NpyIndexes indexes, ndarray values)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_IndexFancyAssign(dest.Array, indexes.Indexes, indexes.NumIndexes, values.Array);
            }
        }

        internal static int SetField(ndarray arr, NpyArray_Descr dtype, int offset, ndarray srcArray)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_SetField(arr.Array, dtype, offset, srcArray.Array);
            }
        }

        internal static void SetNumericOp(UFuncOperation op, NpyUFuncObject UFunc)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_SetNumericOp(op, UFunc);
            }
        }

        internal static ndarray ArrayAll(ndarray arr, int axis, ndarray ret = null, bool keepdims = false)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_All(arr.Array, axis, (ret == null ? null : ret.Array), keepdims));
            }
        }

        internal static ndarray ArrayAny(ndarray arr, int axis, ndarray ret = null, bool keepdims = false)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Any(arr.Array, axis, (ret == null ? null : ret.Array), keepdims));
            }
        }

        internal static ndarray NpyArray_UpscaleSourceArray(ndarray srcArray, ndarray operandArray)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_NumericOpUpscaleSourceArray(srcArray.Array, operandArray.Array));
            }
        }
        internal static ndarray NpyArray_UpscaleSourceArray(ndarray srcArray, shape newshape)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_NumericOpUpscaleSourceArray(srcArray.Array, newshape.iDims, newshape.iDims.Length));
            }
        }

        internal static ndarray ArrayArgMax(ndarray self, int axis, ndarray ret)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_ArgMax(self.Array, axis, (ret == null ? null : ret.Array)));
            }
        }

        internal static ndarray ArrayArgMin(ndarray self, int axis, ndarray ret)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_ArgMin(self.Array, axis, (ret == null ? null : ret.Array)));
            }
        }

        internal static ndarray ArgSort(ndarray arr, int axis, NPY_SORTKIND sortkind)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_ArgSort(arr.Array, axis, sortkind));
            }
        }

        internal static int ArrayBool(ndarray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_Bool(arr.Array);
            }
        }

        internal static ndarray NpyArray_Concatenate(IEnumerable<ndarray> arrays, int? axis, ndarray ret)
        {
            NpyArray[] coreArrays = arrays.Select(x => { Incref(x.Array); return x.Array; }).ToArray();
            NpyArray result;
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                result = numpyAPI.NpyArray_Concatenate(coreArrays, axis, ret != null ? ret.Array : null);
            }
            CheckError();
            return new ndarray(result);
        }


        internal static NPY_SCALARKIND ScalarKind(NPY_TYPES typenum, ref NpyArray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_ScalarKind(typenum, ref arr);
            }
        }

        internal static ndarray Choose(ndarray sel, ndarray[] arrays, ndarray ret = null, NPY_CLIPMODE clipMode = NPY_CLIPMODE.NPY_RAISE)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                var coreArrays = arrays.Select(x => x.Array).ToArray();
                return new ndarray(numpyAPI.NpyArray_Choose(sel.Array, coreArrays, coreArrays.Length, ret == null ? null : ret.Array, clipMode));
            }
        }

        internal static int Partition(ndarray op, ndarray ktharray, int axis, NPY_SELECTKIND which)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_Partition(op.Array, ktharray.Array, axis, which);
            }
        }


        internal static ndarray ArgPartition(ndarray op, ndarray ktharray, int axis, NPY_SELECTKIND which)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_ArgPartition(op.Array, ktharray.Array, axis, which));
            }
        }

        internal static ndarray Correlate(ndarray arr1, ndarray arr2, NPY_TYPES typenum, NPY_CONVOLE_MODE mode)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Correlate(arr1.Array, arr2.Array, typenum, mode));
            }
        }

        internal static ndarray CopyAndTranspose(ndarray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_CopyAndTranspose(arr.Array));
            }
        }

        internal static ndarray CumProd(ndarray arr, int axis, dtype rtype, ndarray ret = null)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_CumProd(arr.Array, axis,
                    (rtype == null ? arr.TypeNum : rtype.TypeNum),
                    (ret == null ? null : ret.Array)));
            }
        }

        internal static ndarray CumSum(ndarray arr, int axis, dtype rtype, ndarray ret = null)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_CumSum(arr.Array, axis,
                        (rtype == null ? arr.TypeNum : rtype.TypeNum),
                        (ret == null ? null : ret.Array)));
            }
        }

        internal static ndarray Floor(ndarray arr, ndarray ret = null)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Floor(arr.Array, (ret == null ? null : ret.Array)));
            }
        }

        internal static ndarray IsNaN(ndarray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_IsNaN(arr.Array));
            }
        }

        internal static void DestroySubarray(NpyArray_ArrayDescr subarrayPtr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
            }
        }

        internal static ndarray Flatten(ndarray arr, NPY_ORDER order)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Flatten(arr.Array, order));
            }
        }

        internal static ndarray InnerProduct(ndarray arr1, ndarray arr2, NPY_TYPES type)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_InnerProduct(arr1.Array, arr2.Array, type));
            }
        }

        internal static ndarray LexSort(ndarray[] arrays, int axis)
        {
            int n = arrays.Length;
            NpyArray[] coreArrays = arrays.Select(x => x.Array).ToArray();
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_LexSort(coreArrays, n, axis));
            }
        }


        internal static ndarray MatrixProduct(ndarray arr1, ndarray arr2, NPY_TYPES type)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_MatrixProduct(arr1.Array, arr2.Array, type));
            }
        }

        internal static ndarray ArrayMax(ndarray arr, int axis, ndarray ret = null, bool keepdims = false)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Max(arr.Array, axis, (ret == null ? null : ret.Array), keepdims));
            }
        }

        internal static ndarray ArrayMin(ndarray arr, int axis, ndarray ret = null, bool keepdims = false)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Min(arr.Array, axis, (ret == null ? null : ret.Array), keepdims));
            }
        }

        internal static ndarray[] NonZero(ndarray arr)
        {
            int nd = arr.ndim;
            NpyArray[] coreArrays = new NpyArray[nd];

#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                if (numpyAPI.NpyArray_NonZero(arr.Array, coreArrays, arr) < 0)
                {
                    NpyCoreApi.CheckError();
                }
            }

            return coreArrays.Select(x => new ndarray(x)).ToArray();
        }

        internal static ndarray Prod(ndarray arr, int axis, dtype rtype, ndarray ret = null, bool keepdims = false)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Prod(arr.Array, axis,
                        (rtype == null ? ScaleTypeUp(arr.TypeNum) : rtype.TypeNum),
                        (ret == null ? null : ret.Array), keepdims));
            }
        }

        internal static NPY_TYPES ScaleTypeUp(NPY_TYPES t)
        {
            switch (t)
            {
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                    return NPY_TYPES.NPY_DOUBLE;

                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_INT16:
                    return NPY_TYPES.NPY_INT32;

                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_UINT16:
                    return NPY_TYPES.NPY_UINT32;

                case NPY_TYPES.NPY_INT32:
                    return NPY_TYPES.NPY_INT64;

                case NPY_TYPES.NPY_UINT32:
                    return NPY_TYPES.NPY_UINT64;

                default:
                    return t;
            }
        }

        internal static int PutMask(ndarray arr, ndarray values, ndarray mask)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_PutMask(arr.Array, values.Array, mask.Array);
            }
        }

        internal static int PutTo(ndarray arr, ndarray values, ndarray indices, NPY_CLIPMODE clipmode)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_PutTo(arr.Array, values.Array, indices.Array, clipmode);
            }
        }


        internal static ndarray Ravel(ndarray arr, NPY_ORDER order)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Ravel(arr.Array, order));
            }
        }

        internal static ndarray Repeat(ndarray arr, ndarray repeats, int axis)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Repeat(arr.Array, repeats.Array, axis));
            }
        }

        internal static ndarray Searchsorted(ndarray arr, ndarray keys, NPY_SEARCHSIDE side)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_SearchSorted(arr.Array, keys.Array, side));
            }
        }

        internal static void Sort(ndarray arr, int axis, NPY_SORTKIND sortkind)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock) 
#endif
            {
                if (numpyAPI.NpyArray_Sort(arr.Array, axis, sortkind) < 0)
                {
                    NpyCoreApi.CheckError();
                }
            }
        }

        internal static ndarray Squeeze(ndarray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Squeeze(arr.Array));
            }
        }

        internal static ndarray SqueezeSelected(ndarray arr, int axis)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_SqueezeSelected(arr.Array, axis));
            }
        }

        internal static ndarray Sum(ndarray arr, int axis, dtype rtype, ndarray ret = null, bool keepdims = false)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_Sum(arr.Array, axis, (rtype == null ? NPY_TYPES.NPY_NOTYPE : rtype.TypeNum), (ret == null ? null : ret.Array), keepdims));
            }
        }

        internal static ndarray SwapAxis(ndarray arr, int a1, int a2)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_SwapAxes(arr.Array, a1, a2));
            }
        }

        internal static ndarray TakeFrom(ndarray arr, ndarray indices, int axis, ndarray ret, NPY_CLIPMODE clipMode)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return new ndarray(numpyAPI.NpyArray_TakeFrom(arr.Array, indices.Array, axis, (ret != null ? ret.Array : null), clipMode));
            }
        }

        internal static bool DescrIsNative(dtype type)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.npy_arraydescr_isnative(type.Descr);
            }
        }

        #endregion



        #region NpyAccessLib functions

        internal static List<string> DescrAllocNames(int n)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_DescrAllocNames(n);
            }
        }

        internal static NpyDict DescrAllocFields()
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_DescrAllocFields();
            }
        }
        internal static void DescrDestroyNames(List<string> p, int n)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArrayAccess_DescrDestroyNames(p, n);
            }
        }


        internal static void ArraySetDescr(ndarray arr, dtype newDescr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_SetDescr(arr.Array, newDescr.Descr);
            }
        }

        internal static long GetArrayDimension(ndarray arr, int dims)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return arr.Array.dimensions[dims];
            }
        }

        internal static long GetArrayStride(ndarray arr, int dims)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return arr.Array.strides[dims];
            }
        }

        internal static int BindIndex(ndarray arr, NpyIndexes indexes, NpyIndexes result)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_IndexBind(indexes.Indexes, indexes.NumIndexes, arr.Array.dimensions, arr.Array.nd, result.Indexes);
            }
        }

        internal static int GetFieldOffset(dtype descr, string fieldName, ref NpyArray_Descr descrPtr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArrayAccess_GetFieldOffset(descr.Descr, fieldName, ref descrPtr);
            }
        }

        internal static void Resize(ndarray arr, npy_intp[] newshape, bool refcheck, NPY_ORDER order)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                NpyArray_Dims newDims = new NpyArray_Dims()
                {
                    ptr = newshape,
                    len = newshape.Length,
                };

                if (numpyAPI.NpyArray_Resize(arr.Array, newDims, refcheck, order) < 0)
                {
                    NpyCoreApi.CheckError();
                }
            }
        }

        internal static ndarray Transpose(ndarray arr, npy_intp[] permute)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                if (permute == null)
                {
                    return new ndarray(numpyAPI.NpyArray_Transpose(arr.Array, null));
                }
                else
                {
                    NpyArray_Dims Dims = new NpyArray_Dims()
                    {
                        ptr = permute,
                        len = permute.Length,
                    };
                    return new ndarray(numpyAPI.NpyArray_Transpose(arr.Array, Dims));
                }
            }
        }

        internal static void ClearUPDATEIFCOPY(ndarray arr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArrayAccess_ClearUPDATEIFCOPY(arr.Array);
            }
        }


        internal static VoidPtr IterNext(NpyArrayIterObject corePtr)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_IterNext(corePtr);
            }
        }

        internal static void IterReset(NpyArrayIterObject iter)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_IterReset(iter);
            }
        }

        internal static NpyArrayMapIterObject IterGetNewMap(NpyIndex[] indexes, int n)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_MapIterNew(indexes, n);
            }
        }


        internal static int IterBindMap(NpyArrayMapIterObject mit, ndarray arr, NpyArray true_array)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArray_MapIterBind(mit, arr.Array, true_array);
            }
        }


        internal static void IterGetMap(NpyArrayMapIterObject iter)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_GetMap(iter);
            }
        }

        internal static void IterMapReset(NpyArrayMapIterObject iter)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_MapIterReset(iter);
            }
        }


        internal static void IterMapNext(NpyArrayMapIterObject iter)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_MapIterNext(iter);
            }
        }



        internal static VoidPtr IterGoto1D(flatiter iter, npy_intp index)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArrayAccess_IterGoto1D(iter.Iter, index);
            }
        }

        internal static npy_intp[] IterCoords(flatiter iter)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArrayAccess_IterCoords(iter.Iter);
            }
        }

        internal static void DescrReplaceSubarray(dtype descr, dtype baseDescr, npy_intp[] dims)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_DescrReplaceSubarray(descr.Descr, baseDescr.Descr, dims.Length, dims);
            }
        }

        internal static void DescrReplaceFields(dtype descr, List<string> namesPtr, NpyDict fieldsDict)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArrayAccess_DescrReplaceFields(descr.Descr, namesPtr, fieldsDict);
            }
        }

        internal static void ZeroFill(ndarray arr, npy_intp offset)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArrayAccess_ZeroFill(arr.Array, offset);
            }
        }

        internal static void SetNamesList(dtype descr, string[] nameslist)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArray_DescrReplaceNames(descr.Descr, nameslist.ToList());
            }
        }

    
        internal static NpyDict_Iter NpyDict_AllocIter()
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArrayAccess_DictAllocIter();
            }
        }

        internal static void NpyDict_FreeIter(NpyDict_Iter iter)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                numpyAPI.NpyArrayAccess_DictFreeIter(iter);
            }
        }

        /// <summary>
        /// Accesses the next dictionary item, returning the key and value.  Thread-safe when operating across
        /// separate iterators; caller must ensure that one iterator is not access simultaneously from two
        /// different threads.
        /// </summary>
        /// <param name="dict">Pointer to the dictionary object</param>
        /// <param name="iter">Iterator structure</param>
        /// <param name="key">Next key</param>
        /// <param name="value">Next value</param>
        /// <returns>True if an element was returned, false at the end of the sequence</returns>
        internal static bool NpyDict_Next(NpyDict dict, NpyDict_Iter iter, NpyDict_KVPair KVPair)
        {
#if ENABLELOCKING
            lock (GlobalIterpLock)
#endif
            {
                return numpyAPI.NpyArrayAccess_DictNext(dict, iter, KVPair);
            }
        }

        #endregion


        #region Callbacks and native access


        internal static byte oppositeByteOrder;



        #region Error handling

        /// <summary>
        /// Indicates the most recent error code or NpyExc_NoError if nothing pending
        /// </summary>
        [ThreadStatic]
        private static npyexc_type _ErrorCode = npyexc_type.NpyExc_NoError;
        private static npyexc_type ErrorCode
        {
            get { return _ErrorCode; }
            set
            {
                _ErrorCode = value;
            }
        }

            /// <summary>
            /// Stores the most recent error message per-thread
            /// </summary>
            [ThreadStatic]
        private static string ErrorMessage = null;

        internal static void CheckError() {
            if (ErrorCode != npyexc_type.NpyExc_NoError) {
                npyexc_type errTmp = ErrorCode;
                String msgTmp = ErrorMessage;

                ErrorCode = npyexc_type.NpyExc_NoError;
                ErrorMessage = null;

                switch (errTmp) {
                    case npyexc_type.NpyExc_MemoryError:
                        throw new InsufficientMemoryException(msgTmp);
                    case npyexc_type.NpyExc_IOError:
                        throw new System.IO.IOException(msgTmp);
                    case npyexc_type.NpyExc_ValueError:
                        throw new ArgumentException(msgTmp);
                    case npyexc_type.NpyExc_IndexError:
                        throw new IndexOutOfRangeException(msgTmp);
                    case npyexc_type.NpyExc_RuntimeError:
                        throw new RuntimeException(msgTmp);
                    case npyexc_type.NpyExc_AttributeError:
                        throw new MissingMemberException(msgTmp);
                    case npyexc_type.NpyExc_ComplexWarning:
                        throw new Exception("ComplexWarning");
                    case npyexc_type.NpyExc_TypeError:
                        throw new TypeErrorException(msgTmp);
                    case npyexc_type.NpyExc_NotImplementedError:
                        throw new NotImplementedException(msgTmp);
                    case npyexc_type.NpyExc_FloatingPointError:
                        throw new FloatingPointException(msgTmp);
                    default:
                        Console.WriteLine("Unhandled exception type {0} in CheckError.", errTmp);
                        throw new RuntimeException(msgTmp);
                }
            }
        }


        private static void SetError(npyexc_type exceptType, string msg) {
            if (exceptType == npyexc_type.NpyExc_ComplexWarning) {
                Console.WriteLine("Warning: {0}", msg);
            } else {
                ErrorCode = exceptType;
                ErrorMessage = msg;
            }
        }



#endregion

#region Thread handling
        // CPython uses a threading model that is single threaded unless the global interpreter lock
        // is explicitly released. While .NET supports true threading, the ndarray core has not been
        // completely checked to makes sure that it is re-entrant  much less modify each function to
        // perform fine-grained locking on individual objects.  Thus we artificially lock IronPython
        // down and force ndarray accesses to be single threaded for now.

        /// <summary>
        /// Equivalent to the CPython GIL.
        /// </summary>
        /// 
#if ENABLELOCKING
        private static readonly object GlobalIterpLock = new object();
#endif

#endregion


#endregion


    }
}
