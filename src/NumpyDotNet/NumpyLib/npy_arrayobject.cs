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
using npy_ucs4 = System.Int64;
using NpyArray_UCS4 = System.Int64;
#else
using npy_intp = System.Int32;
using npy_ucs4 = System.Int32;
using NpyArray_UCS4 = System.Int32;
#endif
using size_t = System.UInt64;

using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace NumpyLib
{
    #region definitions
    
    public class NpyArray : NpyObject_HEAD
    {
        ~NpyArray()
        {
            numpyinternal.NpyDimMem_FREE(dimensions);
            numpyinternal.NpyDimMem_FREE(strides);
            return;
        }

        public string Name
        {
            get; set;
        }
        
        public VoidPtr data;          /* pointer to raw data buffer */
        public int nd;                /* number of dimensions, also called ndim */

        public npy_intp[] dimensions; /* size in each dimension */
        public npy_intp[] strides;    /* bytes to jump to get to next element in each dimension */
    
        private NpyArray _base_arr;
        public NpyArray base_arr        /* Base when it's specifically an array object */
        {
            get { return _base_arr; }
        }
        public void SetBase(NpyArray arr)
        {
            _base_arr = arr;
            if (arr != null)
            {
                Name = arr.Name;
            }
        }


        internal object base_obj;       /* Base when it's an opaque interface object */

        public NpyArray_Descr descr;  /* Pointer to type structure */
        public NPYARRAYFLAGS flags;   /* Flags describing array -- see below */

        public bool IsScalar = false;

        public bool IsASlice
        {
            get
            {
                return base_arr != null;
            }
        }

        public int ItemSize
        {
            get
            {
                return descr.elsize;
            }
        }
        public NPY_TYPES ItemType
        {
            get
            {
                return descr.type_num;
            }
        }
 

        public long GetElementCount()
        {
            long totalElements = 1;
            foreach (var dim in this.dimensions)
            {
                totalElements *= dim;
            }
            return totalElements;
        }

        public IEnumerable<long> ViewOffsets
        {
            get
            {
                return GetViewOffsetEnumeration(0, 0);
            }
        }
  
        private IEnumerable<long> GetViewOffsetEnumeration(int dimIdx, long offset)
        {
            if (dimIdx == nd)
            {
                yield return offset / ItemSize;
            }
            else
            {
                for (int i = 0; i < dimensions[dimIdx]; i++)
                {
                    foreach (var offset2 in GetViewOffsetEnumeration(dimIdx + 1, offset + strides[dimIdx] * i))
                    {
                        yield return offset2;
                    }
                }
            }

        }

   

    }

    internal partial class numpyinternal
    {

        internal static long[] GetViewOffsets(NpyArray arr)
        {
            long[] offsets = new long[NpyArray_SIZE(arr)];
            long offset_index = 0;

            GetViewOffsets(arr, 0, 0, offsets, ref offset_index);
            return offsets;
        }

        private static void GetViewOffsets(NpyArray arr, int dimIdx, long offset, long[] offsets, ref long offset_index)
        {
            if (dimIdx == arr.nd)
            {
                offsets[offset_index++] = offset / arr.ItemSize;
            }
            else
            {
                for (int i = 0; i < arr.dimensions[dimIdx]; i++)
                {
                    GetViewOffsets(arr, dimIdx + 1, offset + arr.strides[dimIdx] * i, offsets, ref offset_index);
                }
            }

        }

        internal static npy_intp[] GetViewOffsets(NpyArrayIterObject iter, long count)
        {
            var itemSize = iter.ao.ItemSize;

            npy_intp[] offsets = new npy_intp[count];

            for (int i = 0; i < count; i++)
            {
                offsets[i] = iter.dataptr.data_offset / itemSize;
                NpyArray_ITER_NEXT(iter);
            }

            return offsets;
  
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Npy_DECREF(NpyObject_HEAD Head)
        {
            lock (Head)
            {
                Debug.Assert(Head.nob_refcnt > 0);

                if (0 == --Head.nob_refcnt)
                {
                    if (null != Head.nob_interface)
                    {
                        //_NpyInterface_Decref(arr.Head.nob_interface, ref arr.Head.nob_interface); 
                    }
                    else
                    {
                        Head.nob_magic_number = (UInt32)npy_defs.NPY_INVALID_MAGIC;
                    }
                }
            }
            return;
        }
 
        internal static bool NpyDataType_ISSTRING(NpyArray_Descr desc)
        {
            return (NpyTypeNum_ISSTRING(desc.type_num));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Npy_INCREF(NpyObject_HEAD Head)
        {
            lock (Head)
            {
                if ((1 == ++Head.nob_refcnt) && null != Head.nob_interface)
                {
                    //_NpyInterface_Incref(arr.Head.nob_interface, ref arr.Head.nob_interface);
                }
            }
            return;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyInterface_DECREF(object o)
        {

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void Npy_XINCREF(NpyArray a)
        {
            if (a != null)
            {
                Npy_INCREF(a);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Npy_XDECREF(NpyArray arr)
        {
            if (arr == null)
                return;
            Npy_DECREF(arr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Npy_XDECREF(NpyArray_Descr descr)
        {
            if (descr == null)
                return;
            Npy_DECREF(descr);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Npy_XDECREF(NpyArrayIterObject iterobject)
        {
            if (iterobject == null)
                return;
            Npy_DECREF(iterobject);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Npy_XDECREF(NpyArrayMultiIterObject multi)
        {
            if (multi == null)
                return;
            Npy_DECREF(multi);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void NpyArray_XDECREF_ERR(NpyArray obj)
        {
            if (obj != null)
            {
                if ((obj.flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) > 0)
                {
                    obj.base_arr.flags |= NPYARRAYFLAGS.NPY_WRITEABLE;
                    obj.flags &= ~NPYARRAYFLAGS.NPY_UPDATEIFCOPY;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static VoidPtr NpyInterface_INCREF(VoidPtr ptr)
        {
            return ptr;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static VoidPtr NpyInterface_DECREF(VoidPtr ptr)
        {
            return ptr;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void NpyInterface_CLEAR(VoidPtr castbuf)
        {
            return;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NpyArray_Descr NpyArray_DESCR_REPLACE(ref NpyArray_Descr descr)
        {
            NpyArray_Descr newDescr = NpyArray_DescrNew(descr);
            Npy_XDECREF(descr);
            descr = newDescr;
            return descr;
        }



        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_CHKFLAGS(NpyArray arr, NPYARRAYFLAGS FLAGS)
        {
            return ((arr.flags  & FLAGS) == FLAGS);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISCONTIGUOUS(NpyArray arr)
        {
            return NpyArray_CHKFLAGS(arr, NPYARRAYFLAGS.NPY_CONTIGUOUS);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISWRITEABLE(NpyArray arr)
        {
            return NpyArray_CHKFLAGS(arr, NPYARRAYFLAGS.NPY_WRITEABLE);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISALIGNED(NpyArray arr)
        {
            return true;
            //return NpyArray_CHKFLAGS(arr, NPYARRAYFLAGS.NPY_ALIGNED);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_IS_C_CONTIGUOUS(NpyArray arr)
        {
            return NpyArray_CHKFLAGS(arr, NPYARRAYFLAGS.NPY_C_CONTIGUOUS);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_IS_F_CONTIGUOUS(NpyArray arr)
        {
            return NpyArray_CHKFLAGS(arr, NPYARRAYFLAGS.NPY_F_CONTIGUOUS);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int NpyArray_NDIM(NpyArray arr)
        {
            return arr.nd;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void NpyArray_NDIM_Update(NpyArray arr, int newnd)
        {
            arr.nd = newnd;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISONESEGMENT(NpyArray arr)
        {
            bool b =( NpyArray_NDIM(arr) == 0) ||
                      NpyArray_CHKFLAGS(arr, NPYARRAYFLAGS.NPY_CONTIGUOUS) ||
                      NpyArray_CHKFLAGS(arr, NPYARRAYFLAGS.NPY_FORTRAN);

            return b;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISFORTRAN(NpyArray arr)
        {
            bool b = (NpyArray_NDIM(arr) > 1) &&
                      NpyArray_CHKFLAGS(arr, NPYARRAYFLAGS.NPY_FORTRAN);

            return b;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int NpyArray_FORTRAN_IF(NpyArray arr)
        {
            return NpyArray_CHKFLAGS(arr, NPYARRAYFLAGS.NPY_FORTRAN) ? (int)NPYARRAYFLAGS.NPY_FORTRAN : 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static VoidPtr NpyArray_DATA(NpyArray arr)
        {
            return arr.data;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static VoidPtr NpyArray_BYTES(NpyArray arr)
        {
            return arr.data;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ulong NpyArray_BYTES_Length(NpyArray arr)
        {
            return VoidPointer_BytesLength(arr.data);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int NpyArray_Array_Length(NpyArray arr)
        {
            return VoidPointer_Length(arr.data);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ulong VoidPointer_BytesLength(VoidPtr vp)
        {
            var ArrayHandler = DefaultArrayHandlers.GetArrayHandler(vp.type_num);

            var Length = ArrayHandler.GetLength(vp) * ArrayHandler.ItemSize;
            return (ulong)(Length - vp.data_offset);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int VoidPointer_Length(VoidPtr vp)
        {
            var ArrayHandler = DefaultArrayHandlers.GetArrayHandler(vp.type_num);

            var Length = ArrayHandler.GetLength(vp);
            return Length;
  
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static npy_intp[] NpyArray_DIMS(NpyArray arr)
        {
            return arr.dimensions;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void NpyArray_DIMS_Update(NpyArray arr, npy_intp[] newdimensions)
        {
            arr.dimensions = newdimensions;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static npy_intp NpyArray_DIM(NpyArray arr, npy_intp n)
        {
            return arr.dimensions[n];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void NpyArray_DIM_Update(NpyArray arr, int n, npy_intp newsize)
        {
            arr.dimensions[n] = newsize;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static npy_intp[] NpyArray_STRIDES(NpyArray arr)
        {
            return arr.strides;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void NpyArray_STRIDES_Update(NpyArray arr, npy_intp[] newStrides)
        {
            arr.strides = newStrides;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static npy_intp NpyArray_DIMS(NpyArray arr, int n)
        {
            return arr.dimensions[n];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static npy_intp NpyArray_STRIDE(NpyArray arr, int n)
        {
            return arr.strides[n];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void NpyArray_STRIDE_Update(NpyArray arr, int n, npy_intp newsize)
        {
            arr.strides[n] = newsize;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NpyArray_Descr NpyArray_DESCR(NpyArray arr)
        {
            return arr.descr;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void NpyArray_DESCR_Update(NpyArray arr, NpyArray_Descr newtype)
        {
            arr.descr = newtype;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NPYARRAYFLAGS NpyArray_FLAGS(NpyArray arr)
        {
            return arr.flags;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NPYARRAYFLAGS NpyArray_FLAGS_OR(NpyArray arr, NPYARRAYFLAGS flag)
        {
            return arr.flags |= flag;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int NpyArray_ITEMSIZE( NpyArray arr)
        {
            return arr.descr.elsize;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int NpyArray_ITEMSIZE(NpyArrayIterObject arr)
        {
            return arr.ao.descr.elsize;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NPY_TYPES NpyArray_TYPE(NpyArray arr)
        {
            return arr.descr.type_num;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NpyArray NpyArray_BASE_ARRAY(NpyArray arr)
        {
            return arr.base_arr;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NpyArray NpyArray_BASE_ARRAY_Update(NpyArray arr, NpyArray newArr)
        {
            arr.SetBase(arr);
            return arr.base_arr;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static object NpyArray_BASE(NpyArray arr)
        {
            return arr.base_obj;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static npy_intp NpyArray_SIZE(NpyArray arr)
        {
            return numpyinternal.NpyArray_MultiplyList(NpyArray_DIMS(arr), NpyArray_NDIM(arr));
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long NpyArray_NBYTES(NpyArray arr)
        {
            return (NpyArray_ITEMSIZE(arr) * NpyArray_SIZE(arr));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_SAMESHAPE(NpyArray a1, NpyArray a2)
        {
            return ((NpyArray_NDIM(a1) == NpyArray_NDIM(a2)) &&
                    numpyinternal.NpyArray_CompareLists(NpyArray_DIMS(a1), NpyArray_DIMS(a2), NpyArray_NDIM(a1)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISBOOL(NpyArray arr)
        {
            return NpyTypeNum_ISBOOL(NpyArray_TYPE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISUNSIGNED(NpyArray arr)
        {
            return NpyTypeNum_ISUNSIGNED(NpyArray_TYPE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISSIGNED(NpyArray arr)
        {
            return NpyTypeNum_ISSIGNED(NpyArray_TYPE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISINTEGER(NpyArray arr)
        {
            return NpyTypeNum_ISINTEGER(NpyArray_TYPE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISFLOAT(NpyArray arr)
        {
            return NpyTypeNum_ISFLOAT(NpyArray_TYPE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISNUMBER(NpyArray arr)
        {
            return NpyTypeNum_ISNUMBER(NpyArray_TYPE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISSTRING(NpyArray arr)
        {
            return NpyTypeNum_ISSTRING(NpyArray_TYPE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISCOMPLEX(NpyArray arr)
        {
            return NpyTypeNum_ISCOMPLEX(NpyArray_TYPE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISFLEXIBLE(NpyArray arr)
        {
            return NpyTypeNum_ISFLEXIBLE(NpyArray_TYPE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISUSERDEF(NpyArray arr)
        {
            return NpyTypeNum_ISUSERDEF(NpyArray_TYPE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISEXTENDED(NpyArray arr)
        {
            return NpyTypeNum_ISEXTENDED(NpyArray_TYPE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISOBJECT(NpyArray arr)
        {
            return NpyTypeNum_ISOBJECT(NpyArray_TYPE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_HASFIELDS(NpyArray arr)
        {
            return NpyArray_DESCR(arr).fields != null;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISVARIABLE(NpyArray arr)
        {
            return NpyTypeNum_ISFLEXIBLE(NpyArray_TYPE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_SAFEALIGNEDCOPY(NpyArray arr)
        {
            return (NpyArray_ISALIGNED(arr) && !NpyArray_ISVARIABLE(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISNOTSWAPPED(NpyArray arr)
        {
            return NpyArray_ISNBO(arr);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISBYTESWAPPED(NpyArray arr)
        {
            return !NpyArray_ISNOTSWAPPED(arr);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_FLAGSWAP(NpyArray arr, NPYARRAYFLAGS flags)
        {
            return (NpyArray_CHKFLAGS(arr, flags) && NpyArray_ISNOTSWAPPED(arr));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISCARRAY(NpyArray arr)
        {
            return (NpyArray_FLAGSWAP(arr, NPYARRAYFLAGS.NPY_CARRAY));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISCARRAY_RO(NpyArray arr)
        {
            return (NpyArray_FLAGSWAP(arr, NPYARRAYFLAGS.NPY_CARRAY_RO));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISFARRAY(NpyArray arr)
        {
            return (NpyArray_FLAGSWAP(arr, NPYARRAYFLAGS.NPY_FARRAY));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISFARRAY_RO(NpyArray arr)
        {
            return (NpyArray_FLAGSWAP(arr, NPYARRAYFLAGS.NPY_FARRAY_RO));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISBEHAVED(NpyArray arr)
        {
            return (NpyArray_FLAGSWAP(arr, NPYARRAYFLAGS.NPY_BEHAVED));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool NpyArray_ISBEHAVED_RO(NpyArray arr)
        {
            return (NpyArray_FLAGSWAP(arr, NPYARRAYFLAGS.NPY_ALIGNED));
        }
    }
    #endregion

    internal partial class numpyinternal
    {
        /*
        * Compute the size of an array (in number of items)
        */
        internal static npy_intp NpyArray_Size(NpyArray op)
        {

            Debug.Assert(Validate(op));
            return NpyArray_SIZE(op);
        }

 
        internal static int NpyArray_ElementStrides(NpyArray arr)
        {
            int itemsize = NpyArray_ITEMSIZE(arr);
            int N = NpyArray_NDIM(arr);
            npy_intp[] strides = NpyArray_STRIDES(arr);

            for (int i = 0; i < N; i++)
            {
                if ((strides[i] % itemsize) != 0)
                {
                    return 0;
                }
            }
            return 1;
        }

        /*
         * This routine checks to see if newstrides (of length nd) will not
         * ever be able to walk outside of the memory implied numbytes and offset.
         *
         * The available memory is assumed to start at -offset and proceed
         * to numbytes-offset.  The strides are checked to ensure
         * that accessing memory using striding will not try to reach beyond
         * this memory for any of the axes.
         *
         * If numbytes is 0 it will be calculated using the dimensions and
         * element-size.
         *
         * This function checks for walking beyond the beginning and right-end
         * of the buffer and therefore works for any integer stride (positive
         * or negative).
         */
        internal static bool NpyArray_CheckStrides(int elsize, int nd, npy_intp numbytes, npy_intp offset, npy_intp[] dims, npy_intp[] newstrides)
        {
            int i;
            npy_intp byte_begin;
            npy_intp begin;
            npy_intp end;

            if (numbytes == 0)
            {
                numbytes = (npy_intp)(NpyArray_MultiplyList(dims, nd) * elsize);
            }
            begin = (npy_intp)(-offset);
            end = (npy_intp)(numbytes - offset - elsize);
            for (i = 0; i < nd; i++)
            {
                byte_begin = newstrides[i] * (dims[i] - 1);
                if ((byte_begin < begin) || (byte_begin > end))
                {
                    return false;
                }
            }
            return true;
        }


        internal static void NpyArray_ForceUpdate(NpyArray self)
        {
            if (((self.flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) != 0) && (self.base_arr != null))
            {
                /*
                 * UPDATEIFCOPY means that base points to an
                 * array that should be updated with the contents
                 * of this array upon destruction.
                 * self.base.flags must have been WRITEABLE
                 * (checked previously) and it was locked here
                 * thus, unlock it.
                 */
                if ((self.flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) != 0)
                {
                    self.flags &= ~NPYARRAYFLAGS.NPY_UPDATEIFCOPY;
                    self.base_arr.flags |= NPYARRAYFLAGS.NPY_WRITEABLE;
                    Npy_INCREF(self); /* hold on to self in next call */
                    if (NpyArray_CopyAnyInto(self.base_arr, self) < 0)
                    {
                        /* NpyErr_Print(); */
                        NpyErr_Clear();
                    }
                    Npy_DECREF(self);
                    Npy_DECREF(self.base_arr);
                    self.SetBase(null);
                }
            }
        }

        internal static NpyArray NpyArray_CompareStringArrays(NpyArray a1, NpyArray a2, int cmp_op, int rstrip)
        {
            NpyArray result;
            int val;
            NpyArrayMultiIterObject mit;

            NPY_TYPES t1 = NpyArray_TYPE(a1);
            NPY_TYPES t2 = NpyArray_TYPE(a2);

            if (NpyArray_TYPE(a1) != NpyArray_TYPE(a2))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "Arrays must be of the same string type.");
                return null;
            }

            /* Broad-cast the arrays to a common shape */
            mit = NpyArray_MultiIterFromArrays(null, 0, 2, a1, a2);
            if (mit == null)
            {
                return null;
            }

            result = NpyArray_NewFromDescr(NpyArray_DescrFromType(NPY_TYPES.NPY_BOOL),
                                           mit.nd,
                                           mit.dimensions, null, 
                                           null, 0, true, null, null);
            if (result == null)
            {
                goto finish;
            }

            val = _compare_strings(result, mit, cmp_op, _mystrncmp, rstrip);
            if (val < 0)
            {
                Npy_DECREF(result);
                result = null;
            }

            finish:
            Npy_DECREF(mit);
            return result;
        }

        delegate int myArrayCompareFunc(NpyArray_UCS4[] s1, NpyArray_UCS4[] s2, int len1, int len2);
        delegate int MyStringCompareFunc(string s1, string s2, int len1, int len2);

        static int _compare_strings(NpyArray result, NpyArrayMultiIterObject multi, int cmp_op, MyStringCompareFunc func, int rstrip)
        {
            return 0;
        }

 
        /*NUMPY_API
         *
         * This function does nothing if obj is writeable, and raises an exception
         * (and returns -1) if obj is not writeable. It may also do other
         * house-keeping, such as issuing warnings on arrays which are transitioning
         * to become views. Always call this function at some point before writing to
         * an array.
         *
         * 'name' is a name for the array, used to give better error
         * messages. Something like "assignment destination", "output array", or even
         * just "array".
         */
        private static int NpyArray_FailUnlessWriteable(NpyArray obj, string name)
        {
            if (!NpyArray_ISWRITEABLE(obj))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, string.Format("{0} is read-only", name));

                return -1;
            }

            return 0;
        }

        /*
         * Compare s1 and s2 which are not necessarily null-terminated.
         * s1 is of length len1
         * s2 is of length len2
         * If they are null terminated, then stop comparison.
         */
        static int _mystrncmp(string s1, string s2, int len1, int len2)
        {
             return string.Compare(s1, s2);
        }

        /* Deallocs & destroy's the array object.
         *  Returns whether or not we did an artificial incref
         *  so we can keep track of the total refcount for debugging.
         */
        /* TODO: For now caller is expected to call _array_dealloc_buffer_info
                 and clear weak refs.  Need to revisit. */
        internal static int NpyArray_dealloc(NpyArray self)
        {
            int i;

            int result = 0;
            Debug.Assert(Validate(self));
            Debug.Assert(ValidateBaseArray(self));
   

            if (null != self.base_arr)
            {
                /*
                 * UPDATEIFCOPY means that base points to an
                 * array that should be updated with the contents
                 * of this array upon destruction.
                 * self.base.flags must have been WRITEABLE
                 * (checked previously) and it was locked here
                 * thus, unlock it.
                 */
                if ((self.flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) > 0)
                {
                    self.base_arr.flags |= NPYARRAYFLAGS.NPY_WRITEABLE;
                    Npy_INCREF(self); /* hold on to self in next call */
                    if (NpyArray_CopyAnyInto(self.base_arr, self) < 0)
                    {
                        /* NpyErr_Print(); */
                        NpyErr_Clear();
                    }
                    /*
                     * Don't need to DECREF -- because we are deleting
                     *self already...
                     */
                    result = 1;
                }
                /*
                 * In any case base is pointing to something that we need
                 * to DECREF -- either a view or a buffer object
                 */
                Npy_DECREF(self.base_arr);
                self.SetBase(null);
            }
            else if (null != self.base_obj)
            {
                NpyInterface_DECREF(self.base_obj);
                self.base_obj = null;
            }

            if ((self.flags & NPYARRAYFLAGS.NPY_OWNDATA) > 0 && self.data != null)
            {
                /* Free internal references if an Object array */
                if (NpyDataType_FLAGCHK(self.descr, NpyArray_Descr_Flags.NPY_ITEM_REFCOUNT))
                {
                    Npy_INCREF(self); /* hold on to self in next call */
                    NpyArray_XDECREF(self);
                    /*
                     * Don't need to DECREF -- because we are deleting
                     * self already...
                     */
                    if (self.nob_refcnt == 1)
                    {
                        result = 1;
                    }
                }

                NpyDataMem_FREE(self.data);
                self.data = null;
            }

            if (null != self.dimensions)
            {
                NpyDimMem_FREE(self.dimensions);
            }

            Npy_DECREF(self.descr);
            /* Flag that this object is now deallocated. */
            self.nob_magic_number = npy_defs.NPY_INVALID_MAGIC;

            NpyArray_free(self);

            return result;
        }


    }





}
