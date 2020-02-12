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

        private static int GetTypeSize(VoidPtr vp)
        {
            return GetTypeSize(vp.type_num);
        }

        private static int __ComplexSize = -1;
        private static int __BigIntSize = -1;
        private static int __ObjectSize = -1;
        private static int __StringSize = -1;

        private static int GetTypeSize(NPY_TYPES type_num)
        {
            int ItemSize = 0;
            switch (type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                {
                    ItemSize = sizeof(bool);
                    break;
                }
                case NPY_TYPES.NPY_BYTE:
                {
                    ItemSize = sizeof(sbyte);
                    break;
                }
                case NPY_TYPES.NPY_UBYTE:
                {
                    ItemSize = sizeof(byte);
                    break;
                }
                case NPY_TYPES.NPY_INT16:
                {
                    ItemSize = sizeof(Int16);
                    break;
                }
                case NPY_TYPES.NPY_UINT16:
                {
                    ItemSize = sizeof(UInt16);
                    break;
                }
                case NPY_TYPES.NPY_INT32:
                {
                    ItemSize = sizeof(Int32);
                    break;
                }
                case NPY_TYPES.NPY_UINT32:
                {
                    ItemSize = sizeof(UInt32);
                    break;
                }
                case NPY_TYPES.NPY_INT64:
                {
                    ItemSize = sizeof(Int64);
                    break;
                }
                case NPY_TYPES.NPY_UINT64:
                {
                    ItemSize = sizeof(UInt64);
                    break;
                }
                case NPY_TYPES.NPY_FLOAT:
                {
                    ItemSize = sizeof(float);
                    break;
                }
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_COMPLEXREAL:
                case NPY_TYPES.NPY_COMPLEXIMAG:
                {
                    ItemSize = sizeof(double);
                    break;
                }
                case NPY_TYPES.NPY_DECIMAL:
                {
                    ItemSize = sizeof(decimal);
                    break;
                }
                case NPY_TYPES.NPY_COMPLEX:
                {
                    if (__ComplexSize < 0)
                    {
                        __ComplexSize = DefaultArrayHandlers.GetArrayHandler(NPY_TYPES.NPY_COMPLEX).ItemSize;
                    }
                    ItemSize = __ComplexSize;
                    break;
                }
                case NPY_TYPES.NPY_BIGINT:
                {
                    if (__BigIntSize < 0)
                    {
                        __BigIntSize = DefaultArrayHandlers.GetArrayHandler(NPY_TYPES.NPY_BIGINT).ItemSize;
                    }
                    ItemSize = __BigIntSize;
                    break;
                }
                case NPY_TYPES.NPY_OBJECT:
                {
                    if (__ObjectSize < 0)
                    {
                        __ObjectSize = DefaultArrayHandlers.GetArrayHandler(NPY_TYPES.NPY_OBJECT).ItemSize;
                    }
                    ItemSize = __ObjectSize;
                    break;
                }
                case NPY_TYPES.NPY_STRING:
                {
                    if (__StringSize < 0)
                    {
                        __StringSize = DefaultArrayHandlers.GetArrayHandler(NPY_TYPES.NPY_STRING).ItemSize;
                    }
                    ItemSize = __StringSize;
                    break;
                }


            }

            return ItemSize;
        }


        #region memmove
 

        internal static void memmove(VoidPtr dest, VoidPtr src, long len)
        {
            memmove(dest, 0, src, 0, len);
        }

        internal static void memmove(VoidPtr dest, npy_intp dest_offset, VoidPtr src, npy_intp src_offset, long len)
        {
            #region special case data types
            if (dest.type_num == NPY_TYPES.NPY_DECIMAL)
            {
                long ElementCount = len / sizeof(decimal);
                long sOffset = (src.data_offset + src_offset) / sizeof(decimal);
                long dOffset = (dest.data_offset+ dest_offset) / sizeof(decimal);

                var temp = new decimal[ElementCount];
                Array.Copy(src.datap as decimal[], sOffset, temp, 0, ElementCount);
                Array.Copy(temp, 0, dest.datap as decimal[], dOffset, ElementCount);
            }
            else if (dest.type_num == NPY_TYPES.NPY_COMPLEX)
            {
                var _ComplexSize = GetTypeSize(dest.type_num);

                long ElementCount = len / _ComplexSize;
                long sOffset = (src.data_offset + src_offset) / _ComplexSize;
                long dOffset = (dest.data_offset + dest_offset) / _ComplexSize;

                var temp = new System.Numerics.Complex[ElementCount];
                Array.Copy(src.datap as System.Numerics.Complex[], sOffset, temp, 0, ElementCount);
                Array.Copy(temp, 0, dest.datap as System.Numerics.Complex[], dOffset, ElementCount);
            }
            else if (dest.type_num == NPY_TYPES.NPY_BIGINT)
            {
                var _BigIntSize = GetTypeSize(dest.type_num);
 
                long ElementCount = len / _BigIntSize;
                long sOffset = (src.data_offset + src_offset) / _BigIntSize;
                long dOffset = (dest.data_offset + dest_offset) / _BigIntSize;

                var temp = new System.Numerics.BigInteger[ElementCount];
                Array.Copy(src.datap as System.Numerics.BigInteger[], sOffset, temp, 0, ElementCount);
                Array.Copy(temp, 0, dest.datap as System.Numerics.BigInteger[], dOffset, ElementCount);
            }
            else if (dest.type_num == NPY_TYPES.NPY_OBJECT)
            {
                var _ObjectSize = GetTypeSize(dest.type_num);

                long ElementCount = len / _ObjectSize;
                long sOffset = (src.data_offset + src_offset) / _ObjectSize;
                long dOffset = (dest.data_offset + dest_offset) / _ObjectSize;

                var temp = new object[ElementCount];
                Array.Copy(src.datap as object[], sOffset, temp, 0, ElementCount);
                Array.Copy(temp, 0, dest.datap as object[], dOffset, ElementCount);
            }
            else if (dest.type_num == NPY_TYPES.NPY_STRING)
            {
                var _StringSize = GetTypeSize(dest.type_num);

                long ElementCount = len / _StringSize;
                long sOffset = (src.data_offset + src_offset) / _StringSize;
                long dOffset = (dest.data_offset + dest_offset) / _StringSize;

                var temp = new String[ElementCount];
                Array.Copy(src.datap as String[], sOffset, temp, 0, ElementCount);
                Array.Copy(temp, 0, dest.datap as String[], dOffset, ElementCount);
            }
            #endregion
            else
            {
                if ((src.type_num == dest.type_num) && IsElementAligned(src, src_offset) && IsElementAligned(dest, dest_offset))
                {
                    #region perfectly aligned arrays of same type can be processed much faster this way.
                    switch (src.type_num)
                    {
                        case NPY_TYPES.NPY_BOOL:
                        {
                            long ElementCount = len / sizeof(bool);
                            long sOffset = (src.data_offset + src_offset) / sizeof(bool);
                            long dOffset = (dest.data_offset + dest_offset) / sizeof(bool);

                            var temp = new bool[ElementCount];
                            Array.Copy(src.datap as bool[], sOffset, temp, 0, ElementCount);
                            Array.Copy(temp, 0, dest.datap as bool[], dOffset, ElementCount);
                            break;
                        }
                        case NPY_TYPES.NPY_BYTE:
                        {
                            long ElementCount = len / sizeof(sbyte);
                            long sOffset = (src.data_offset + src_offset) / sizeof(sbyte);
                            long dOffset = (dest.data_offset + dest_offset) / sizeof(sbyte);

                            var temp = new sbyte[ElementCount];
                            Array.Copy(src.datap as sbyte[], sOffset, temp, 0, ElementCount);
                            Array.Copy(temp, 0, dest.datap as sbyte[], dOffset, ElementCount);
                            break;
                        }
                        case NPY_TYPES.NPY_UBYTE:
                        {
                            long ElementCount = len / sizeof(byte);
                            long sOffset = (src.data_offset + src_offset) / sizeof(byte);
                            long dOffset = (dest.data_offset + dest_offset) / sizeof(byte);

                            var temp = new byte[ElementCount];
                            Array.Copy(src.datap as byte[], sOffset, temp, 0, ElementCount);
                            Array.Copy(temp, 0, dest.datap as byte[], dOffset, ElementCount);
                            break;
                        }
                        case NPY_TYPES.NPY_INT16:
                        {
                            long ElementCount = len / sizeof(Int16);
                            long sOffset = (src.data_offset + src_offset) / sizeof(Int16);
                            long dOffset = (dest.data_offset + dest_offset) / sizeof(Int16);

                            var temp = new Int16[ElementCount];
                            Array.Copy(src.datap as Int16[], sOffset, temp, 0, ElementCount);
                            Array.Copy(temp, 0, dest.datap as Int16[], dOffset, ElementCount);
                            break;
                        }
                        case NPY_TYPES.NPY_UINT16:
                        {
                            long ElementCount = len / sizeof(UInt16);
                            long sOffset = (src.data_offset + src_offset) / sizeof(UInt16);
                            long dOffset = (dest.data_offset + dest_offset) / sizeof(UInt16);

                            var temp = new UInt16[ElementCount];
                            Array.Copy(src.datap as UInt16[], sOffset, temp, 0, ElementCount);
                            Array.Copy(temp, 0, dest.datap as UInt16[], dOffset, ElementCount);
                            break;
                        }
                        case NPY_TYPES.NPY_INT32:
                        {
                            long ElementCount = len / sizeof(Int32);
                            long sOffset = (src.data_offset + src_offset) / sizeof(Int32);
                            long dOffset = (dest.data_offset + dest_offset) / sizeof(Int32);

                            var temp = new Int32[ElementCount];
                            Array.Copy(src.datap as Int32[], sOffset, temp, 0, ElementCount);
                            Array.Copy(temp, 0, dest.datap as Int32[], dOffset, ElementCount);
                            break;
                        }
                        case NPY_TYPES.NPY_UINT32:
                        {
                            long ElementCount = len / sizeof(UInt32);
                            long sOffset = (src.data_offset + src_offset) / sizeof(UInt32);
                            long dOffset = (dest.data_offset + dest_offset) / sizeof(UInt32);

                            var temp = new UInt32[ElementCount];
                            Array.Copy(src.datap as UInt32[], sOffset, temp, 0, ElementCount);
                            Array.Copy(temp, 0, dest.datap as UInt32[], dOffset, ElementCount);
                            break;
                        }
                        case NPY_TYPES.NPY_INT64:
                        {
                            long ElementCount = len / sizeof(Int64);
                            long sOffset = (src.data_offset + src_offset) / sizeof(Int64);
                            long dOffset = (dest.data_offset + dest_offset) / sizeof(Int64);

                            var temp = new Int64[ElementCount];
                            Array.Copy(src.datap as Int64[], sOffset, temp, 0, ElementCount);
                            Array.Copy(temp, 0, dest.datap as Int64[], dOffset, ElementCount);
                            break;
                        }
                        case NPY_TYPES.NPY_UINT64:
                        {
                            long ElementCount = len / sizeof(UInt64);
                            long sOffset = (src.data_offset + src_offset) / sizeof(UInt64);
                            long dOffset = (dest.data_offset + dest_offset) / sizeof(UInt64);

                            var temp = new UInt64[ElementCount];
                            Array.Copy(src.datap as UInt64[], sOffset, temp, 0, ElementCount);
                            Array.Copy(temp, 0, dest.datap as UInt64[], dOffset, ElementCount);
                            break;
                        }
                        case NPY_TYPES.NPY_FLOAT:
                        {
                            long ElementCount = len / sizeof(float);
                            long sOffset = (src.data_offset + src_offset) / sizeof(float);
                            long dOffset = (dest.data_offset + dest_offset) / sizeof(float);

                            var temp = new float[ElementCount];
                            Array.Copy(src.datap as float[], sOffset, temp, 0, ElementCount);
                            Array.Copy(temp, 0, dest.datap as float[], dOffset, ElementCount);
                            break;
                        }
                        case NPY_TYPES.NPY_DOUBLE:
                        {
                            long ElementCount = len / sizeof(double);
                            long sOffset = (src.data_offset + src_offset) / sizeof(double);
                            long dOffset = (dest.data_offset + dest_offset) / sizeof(double);

                            var temp = new double[ElementCount];
                            Array.Copy(src.datap as double[], sOffset, temp, 0, ElementCount);
                            Array.Copy(temp, 0, dest.datap as double[], dOffset, ElementCount);
                            break;
                        }

                    }
                    #endregion
                }
                else
                {
                    VoidPtr Temp = new VoidPtr(new byte[len]);
                    MemCopy.MemCpy(Temp, 0, src, src_offset, len);
                    MemCopy.MemCpy(dest, dest_offset, Temp, 0, len);
                }
    
                return;
            }

        }

        private static bool IsElementAligned(VoidPtr src, long src_offset)
        {
            long ItemSize = GetTypeSize(src.type_num);

            if ((src_offset % ItemSize == 0) && (src.data_offset % ItemSize == 0))
                return true;
            return false;
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

        private static void NpyArray_free(npy_intp[] shape_dims)
        {
            return;
        }
        #endregion
  
    }
}
