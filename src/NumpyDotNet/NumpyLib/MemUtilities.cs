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
                index = dyndatap.Length - -index;
            }

            object ret = DefaultArrayHandlers.GetArrayHandler(obj.type_num).GetIndex(obj, index);
            return ret;
        }
 
        internal static int SetIndex(VoidPtr data, npy_intp index, object invalue)
        {
            if (index < 0)
            {
                dynamic dyndatap = data.datap;
                index = dyndatap.Length - -index;
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


        public static int GetDivSize(int elsize)
        {
            switch (elsize)
            {
                case 1: return 0;
                case 2: return 1;
                case 4: return 2;
                case 8: return 3;
                case 16: return 4;
                case 32: return 5;
                case 64: return 6;
                //case 128: return 7;
                //case 256: return 8;
                //case 512: return 9;
                //case 1024: return 10;
                //case 2048: return 11;
                //case 4096: return 12;
            }

            throw new Exception("Unexpected elsize in GetDivSize");
        }

        internal static int IntpDivSize = GetDivSize(sizeof(npy_intp));

        #region memmove

        //internal static void memmove(VoidPtr dest, npy_intp dest_offset, VoidPtr src, npy_intp src_offset, long len)
        //{
        //    // we think it is impossible for memory to be not aligned in pure .NET so no need to do this big expensive check.
        //    //if ((src.type_num == dest.type_num) && IsElementAligned(src, src_offset) && IsElementAligned(dest, dest_offset))
        //    if (src.type_num == dest.type_num)
        //    {
        //        var helper = MemCopy.GetMemcopyHelper(dest);
        //        helper.memmove_real(dest, dest_offset, src, src_offset, len);
        //        return;
        //    }
        //    else
        //    {
        //        VoidPtr Temp = new VoidPtr(new byte[len]);
        //        MemCopy.MemCpy(Temp, 0, src, src_offset, len);
        //        MemCopy.MemCpy(dest, dest_offset, Temp, 0, len);
        //        return;
        //    }

        //}

        //private static bool IsElementAligned(VoidPtr src, long src_offset)
        //{
        //    long ItemSize = GetTypeSize(src.type_num);

        //    if ((src_offset % ItemSize == 0) && (src.data_offset % ItemSize == 0))
        //        return true;
        //    throw new Exception();
        //    return false;
        //}



        #endregion

        #region memcpy

        internal static void copydims(npy_intp[] dest, npy_intp[] src, long len)
        {
            Array.Copy(src, dest, len);
        }

        internal static void memcpy(VoidPtr dest, VoidPtr src, long len)
        {
            MemCopy.MemCpy(dest, 0, src, 0, len);
        }

        #endregion

        #region memset

        internal static void memclr(VoidPtr dest, long len)
        {
            var helper = MemCopy.GetMemcopyHelper(dest);
            helper.memclr(dest, dest.data_offset, len);
        }

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
