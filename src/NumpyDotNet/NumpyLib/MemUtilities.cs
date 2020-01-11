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
        internal static object GetIndex(VoidPtr obj, npy_intp index)
        {
            if (index < 0)
            {
                dynamic dyndatap = obj.datap;
                index = dyndatap.Length - Math.Abs(index);
            }

            object ret = DefaultArrayHandlers.GetArrayHandler(obj.type_num).GetIndex(obj, index);
            return ret;
        }
 
        internal static int SetIndex(VoidPtr data, npy_intp index, object invalue)
        {
            if (index < 0)
            {
                dynamic dyndatap = data.datap;
                index = dyndatap.Length - Math.Abs(index);
            }

            try
            {
                DefaultArrayHandlers.GetArrayHandler(data.type_num).SetIndex(data, index, invalue);
            }
            catch (System.OverflowException oe)
            {
                NpyErr_SetString(npyexc_type.NpyExc_OverflowError, oe.Message);
            }
            catch (Exception ex)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, ex.Message);
            }

            return 1;
        }

        private static long GetTypeSize(VoidPtr vp)
        {
            return GetTypeSize(vp.type_num);
        }

        private static long GetTypeSize(NPY_TYPES type_num)
        {
            return numpyAPI.GetArrayHandler(type_num).ItemSize;
        }


        #region memmove
        internal static void memmove(VoidPtr dest, npy_intp dest_offset, VoidPtr src, npy_intp src_offset, long len)
        {
            if (dest.type_num == NPY_TYPES.NPY_DECIMAL)
            {
                VoidPtr Temp = new VoidPtr(new decimal[len/sizeof(decimal)]);
                MemCopy.MemCpy(Temp, 0, src, src_offset, len);
                MemCopy.MemCpy(dest, dest_offset, Temp, 0, len);
            }
            else if (dest.type_num == NPY_TYPES.NPY_COMPLEX)
            {
                VoidPtr Temp = new VoidPtr(new System.Numerics.Complex[len / DefaultArrayHandlers.GetArrayHandler(NPY_TYPES.NPY_COMPLEX).ItemSize]);
                MemCopy.MemCpy(Temp, 0, src, src_offset, len);
                MemCopy.MemCpy(dest, dest_offset, Temp, 0, len);
            }
            else if (dest.type_num == NPY_TYPES.NPY_BIGINT)
            {
                VoidPtr Temp = new VoidPtr(new System.Numerics.BigInteger[len / DefaultArrayHandlers.GetArrayHandler(NPY_TYPES.NPY_BIGINT).ItemSize]);
                MemCopy.MemCpy(Temp, 0, src, src_offset, len);
                MemCopy.MemCpy(dest, dest_offset, Temp, 0, len);
            }
            else
            {
                VoidPtr Temp = new VoidPtr(new byte[len]);
                MemCopy.MemCpy(Temp, 0, src, src_offset, len);
                MemCopy.MemCpy(dest, dest_offset, Temp, 0, len);
            }

        }
        internal static void memmove(VoidPtr dest, VoidPtr src, long len)
        {
            memmove(dest, 0, src, 0, len);
        }

        #endregion

        #region memcpy

        internal static void memcpy(npy_intp[] dest, npy_intp[] src, long len)
        {
            Array.Copy(src, dest, len / sizeof(npy_intp));
            //MemCopy.MemCpy(new VoidPtr(dest), 0, new VoidPtr(src), 0, len);
        }

        internal static void memcpy(VoidPtr dest, VoidPtr src, long len)
        {
            MemCopy.MemCpy(dest, 0, src, 0, len);
        }

        #endregion

        #region memset
  
   
        internal static void memset(VoidPtr dest, byte setvalue, long len)
        {
            MemSet.memset(dest, 0, setvalue, len);
        }

        #endregion

        #region alloc and free

        public static VoidPtr NpyDataMem_RENEW(VoidPtr oldArray, ulong newSize)
        {
            VoidPtr newArray = NpyDataMem_NEW(oldArray.type_num, newSize, true);
     
            ulong preserveLength = System.Math.Min(VoidPointer_BytesLength(oldArray), newSize);
            if (preserveLength > 0)
            {
                MemCopy.MemCpy(newArray, 0, oldArray, 0, (long)preserveLength);
            }
            return newArray;
        }

        internal static VoidPtr NpyDataMem_NEW(NPY_TYPES type_num, ulong size, bool AdjustForBytes = true)
        {

            var ArrayHandler = DefaultArrayHandlers.GetArrayHandler(type_num);
            size = size / (AdjustForBytes ? (ulong)ArrayHandler.ItemSize : (ulong)1);

            VoidPtr vp = new VoidPtr();
            vp.datap = ArrayHandler.AllocateNewArray((int)size);
            vp.type_num = type_num;
            return vp;
        }


        internal static npy_intp[] NpyDimMem_NEW(npy_intp size)
        {
            return new npy_intp[size];
        }

        internal static void NpyDataMem_FREE(VoidPtr b)
        {
            return;
        }
        internal static void NpyDimMem_FREE(npy_intp[] dimensions)
        {
            return;
        }
        internal static void NpyArray_free(NpyArray array)
        {
            return;
        }
        internal static void NpyArray_free(object obj)
        {
            return;
        }
        internal static void NpyArray_free(NpyArray_ArrayDescr array)
        {
            return;
        }
        
        internal static void npy_free(object o)
        {
            return;
        }

        internal static void NpyArray_free(NpyArrayMultiIterObject iterobj)
        {
            return;
        }

        private static void NpyArray_free(NpyArray_DateTimeInfo dtinfo)
        {
            return;
        }

        private static void NpyArray_free(npy_intp[] shape_dims)
        {
            return;
        }
        #endregion
  
    }
}
