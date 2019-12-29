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

            switch (obj.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    var dbool = obj.datap as bool[];
                    return dbool[index];
                case NPY_TYPES.NPY_BYTE:
                    var dsbyte = obj.datap as sbyte[];
                    return dsbyte[index];
                case NPY_TYPES.NPY_UBYTE:
                    var dbyte = obj.datap as byte[];
                    return dbyte[index];
                case NPY_TYPES.NPY_UINT16:
                    var duint16 = obj.datap as UInt16[];
                    return duint16[index];
                case NPY_TYPES.NPY_INT16:
                    var dint16 = obj.datap as Int16[];
                    return dint16[index];
                case NPY_TYPES.NPY_UINT32:
                    var duint32 = obj.datap as UInt32[];
                    return duint32[index];
                case NPY_TYPES.NPY_INT32:
                    var dint32 = obj.datap as Int32[];
                    return dint32[index];
                case NPY_TYPES.NPY_INT64:
                    var dint64 = obj.datap as Int64[];
                    return dint64[index];
                case NPY_TYPES.NPY_UINT64:
                    var duint64 = obj.datap as UInt64[];
                    return duint64[index];
                case NPY_TYPES.NPY_FLOAT:
                    var float1 = obj.datap as float[];
                    return float1[index];
                case NPY_TYPES.NPY_DOUBLE:
                    var double1 = obj.datap as double[];
                    return double1[index];
                case NPY_TYPES.NPY_DECIMAL:
                    var decimal1 = obj.datap as decimal[];
                    return decimal1[index];
                case NPY_TYPES.NPY_COMPLEX:
                    var complex1 = obj.datap as System.Numerics.Complex[];
                    return complex1[index];
                default:
                    throw new Exception("Unsupported data type");
            }
        }

        private static object CoerceValue(object value, NPY_TYPES type_num)
        {

            if (value is double)
            {
                double iValue = (double)value;

                switch (type_num)
                {
                    case NPY_TYPES.NPY_BOOL:
                        return Convert.ToBoolean(iValue);
                    case NPY_TYPES.NPY_BYTE:
                        return Convert.ToSByte(0xFF & (Int64)iValue);
                    case NPY_TYPES.NPY_UBYTE:
                        return Convert.ToByte(0xFF & (UInt64)iValue);
                    case NPY_TYPES.NPY_UINT16:
                        return Convert.ToUInt16(0xFFFF & (UInt64)iValue);
                    case NPY_TYPES.NPY_INT16:
                        return Convert.ToInt16(0xFFFF & (Int64)iValue);
                    case NPY_TYPES.NPY_UINT32:
                        return Convert.ToUInt32(0xFFFFFFFF & (UInt64)iValue);
                    case NPY_TYPES.NPY_INT32:
                        return Convert.ToInt32(0xFFFFFFFF & (UInt64)iValue);
                    case NPY_TYPES.NPY_INT64:
                        return Convert.ToInt64(iValue);
                    case NPY_TYPES.NPY_UINT64:
                        return Convert.ToUInt64(iValue);
                    case NPY_TYPES.NPY_FLOAT:
                        return Convert.ToSingle(iValue);
                    case NPY_TYPES.NPY_DOUBLE:
                        return iValue;
                    case NPY_TYPES.NPY_DECIMAL:
                        return Convert.ToDecimal(iValue);
                    default:
                        throw new Exception("Unsupported data type");
                }
            }
            else if (value is Single)
            {
                Single iValue = (Single)value;

                switch (type_num)
                {
                    case NPY_TYPES.NPY_BOOL:
                        return Convert.ToBoolean(iValue);
                    case NPY_TYPES.NPY_BYTE:
                        return Convert.ToSByte(0xFF & (Int32)iValue);
                    case NPY_TYPES.NPY_UBYTE:
                        return Convert.ToByte(0xFF & (UInt32)iValue);
                    case NPY_TYPES.NPY_UINT16:
                        return Convert.ToUInt16(0xFFFF & (UInt32)iValue);
                    case NPY_TYPES.NPY_INT16:
                        return Convert.ToInt16(0xFFFF & (Int32)iValue);
                    case NPY_TYPES.NPY_UINT32:
                        return Convert.ToUInt32(0xFFFFFFFF & (UInt32)iValue);
                    case NPY_TYPES.NPY_INT32:
                        return Convert.ToInt32(0xFFFFFFFF & (Int32)iValue);
                    case NPY_TYPES.NPY_INT64:
                        return Convert.ToInt64(iValue);
                    case NPY_TYPES.NPY_UINT64:
                        return Convert.ToUInt64(iValue);
                    case NPY_TYPES.NPY_FLOAT:
                        return iValue;
                    case NPY_TYPES.NPY_DOUBLE:
                        return Convert.ToDouble(iValue);
                    case NPY_TYPES.NPY_DECIMAL:
                        return Convert.ToDecimal(iValue);
                    default:
                        throw new Exception("Unsupported data type");
                }
            }
            else if (value is Int64)
            {
                Int64 iValue = (Int64)value;
                switch (type_num)
                {
                    case NPY_TYPES.NPY_BOOL:
                        return Convert.ToBoolean(iValue);
                    case NPY_TYPES.NPY_BYTE:
                        return Convert.ToSByte(iValue);
                    case NPY_TYPES.NPY_UBYTE:
                        return Convert.ToByte(iValue);
                    case NPY_TYPES.NPY_UINT16:
                        return Convert.ToUInt16(iValue);
                    case NPY_TYPES.NPY_INT16:
                        return Convert.ToInt16(iValue);
                    case NPY_TYPES.NPY_UINT32:
                        return Convert.ToUInt32(iValue);
                    case NPY_TYPES.NPY_INT32:
                        return Convert.ToInt32(iValue);
                    case NPY_TYPES.NPY_INT64:
                        return iValue;
                    case NPY_TYPES.NPY_UINT64:
                        return Convert.ToUInt64(iValue);
                    case NPY_TYPES.NPY_FLOAT:
                        return Convert.ToSingle(iValue);
                    case NPY_TYPES.NPY_DOUBLE:
                        return Convert.ToDouble(iValue);
                    case NPY_TYPES.NPY_DECIMAL:
                        return Convert.ToDecimal(iValue);
                    default:
                        throw new Exception("Unsupported data type");
                }
            }
            else if (value is UInt64)
            {
                UInt64 iValue = (UInt64)value;

                switch (type_num)
                {
                    case NPY_TYPES.NPY_BOOL:
                        return Convert.ToBoolean(iValue);
                    case NPY_TYPES.NPY_BYTE:
                        return Convert.ToSByte(iValue);
                    case NPY_TYPES.NPY_UBYTE:
                        return Convert.ToByte(iValue);
                    case NPY_TYPES.NPY_UINT16:
                        return Convert.ToUInt16(iValue);
                    case NPY_TYPES.NPY_INT16:
                        return Convert.ToInt16(iValue);
                    case NPY_TYPES.NPY_UINT32:
                        return Convert.ToUInt32(iValue);
                    case NPY_TYPES.NPY_INT32:
                        return Convert.ToInt32(iValue);
                    case NPY_TYPES.NPY_INT64:
                        return Convert.ToInt64(iValue);
                    case NPY_TYPES.NPY_UINT64:
                        return iValue;
                    case NPY_TYPES.NPY_FLOAT:
                        return Convert.ToSingle(iValue);
                    case NPY_TYPES.NPY_DOUBLE:
                        return Convert.ToDouble(iValue);
                    case NPY_TYPES.NPY_DECIMAL:
                        return Convert.ToDecimal(iValue);
                    default:
                        throw new Exception("Unsupported data type");
                }
            }
            else if (value is Int32)
            {
                Int32 iValue = (Int32)value;

                switch (type_num)
                {
                    case NPY_TYPES.NPY_BOOL:
                        return Convert.ToBoolean(iValue);
                    case NPY_TYPES.NPY_BYTE:
                        return Convert.ToSByte(iValue);
                    case NPY_TYPES.NPY_UBYTE:
                        return Convert.ToByte(iValue);
                    case NPY_TYPES.NPY_UINT16:
                        return Convert.ToUInt16(iValue);
                    case NPY_TYPES.NPY_INT16:
                        return Convert.ToInt16(iValue);
                    case NPY_TYPES.NPY_UINT32:
                        return Convert.ToUInt32(iValue);
                    case NPY_TYPES.NPY_INT32:
                        return iValue;
                    case NPY_TYPES.NPY_INT64:
                        return Convert.ToInt64(iValue);
                    case NPY_TYPES.NPY_UINT64:
                        return Convert.ToUInt64(iValue);
                    case NPY_TYPES.NPY_FLOAT:
                        return Convert.ToSingle(iValue);
                    case NPY_TYPES.NPY_DOUBLE:
                        return Convert.ToDouble(iValue);
                    case NPY_TYPES.NPY_DECIMAL:
                        return Convert.ToDecimal(iValue);
                    default:
                        throw new Exception("Unsupported data type");
                }
            }
            else if (value is UInt32)
            {
                UInt32 iValue = (UInt32)value;

                switch (type_num)
                {
                    case NPY_TYPES.NPY_BOOL:
                        return Convert.ToBoolean(iValue);
                    case NPY_TYPES.NPY_BYTE:
                        return Convert.ToSByte(iValue);
                    case NPY_TYPES.NPY_UBYTE:
                        return Convert.ToByte(iValue);
                    case NPY_TYPES.NPY_UINT16:
                        return Convert.ToUInt16(iValue);
                    case NPY_TYPES.NPY_INT16:
                        return Convert.ToInt16(iValue);
                    case NPY_TYPES.NPY_UINT32:
                        return iValue;
                    case NPY_TYPES.NPY_INT32:
                        return Convert.ToInt32(iValue);
                    case NPY_TYPES.NPY_INT64:
                        return Convert.ToInt64(iValue);
                    case NPY_TYPES.NPY_UINT64:
                        return Convert.ToUInt64(iValue);
                    case NPY_TYPES.NPY_FLOAT:
                        return Convert.ToSingle(iValue);
                    case NPY_TYPES.NPY_DOUBLE:
                        return Convert.ToDouble(iValue);
                    case NPY_TYPES.NPY_DECIMAL:
                        return Convert.ToDecimal(iValue);
                    default:
                        throw new Exception("Unsupported data type");
                }
            }
            else if (value is Int16)
            {
                Int16 iValue = (Int16)value;

                switch (type_num)
                {
                    case NPY_TYPES.NPY_BOOL:
                        return Convert.ToBoolean(iValue);
                    case NPY_TYPES.NPY_BYTE:
                        return Convert.ToSByte(iValue);
                    case NPY_TYPES.NPY_UBYTE:
                        return Convert.ToByte(iValue);
                    case NPY_TYPES.NPY_UINT16:
                        return Convert.ToUInt16(iValue);
                    case NPY_TYPES.NPY_INT16:
                        return iValue;
                    case NPY_TYPES.NPY_UINT32:
                        return Convert.ToUInt32(iValue);
                    case NPY_TYPES.NPY_INT32:
                        return Convert.ToInt32(iValue);
                    case NPY_TYPES.NPY_INT64:
                        return Convert.ToInt64(iValue);
                    case NPY_TYPES.NPY_UINT64:
                        return Convert.ToUInt64(iValue);
                    case NPY_TYPES.NPY_FLOAT:
                        return Convert.ToSingle(iValue);
                    case NPY_TYPES.NPY_DOUBLE:
                        return Convert.ToDouble(iValue);
                    case NPY_TYPES.NPY_DECIMAL:
                        return Convert.ToDecimal(iValue);
                    default:
                        throw new Exception("Unsupported data type");
                }
            }
            else if (value is Int16)
            {
                UInt16 iValue = (UInt16)value;

                switch (type_num)
                {
                    case NPY_TYPES.NPY_BOOL:
                        return Convert.ToBoolean(iValue);
                    case NPY_TYPES.NPY_BYTE:
                        return Convert.ToSByte(iValue);
                    case NPY_TYPES.NPY_UBYTE:
                        return Convert.ToByte(iValue);
                    case NPY_TYPES.NPY_UINT16:
                        return iValue;
                    case NPY_TYPES.NPY_INT16:
                        return Convert.ToInt16(iValue);
                    case NPY_TYPES.NPY_UINT32:
                        return Convert.ToUInt32(iValue);
                    case NPY_TYPES.NPY_INT32:
                        return Convert.ToInt32(iValue);
                    case NPY_TYPES.NPY_INT64:
                        return Convert.ToInt64(iValue);
                    case NPY_TYPES.NPY_UINT64:
                        return Convert.ToUInt64(iValue);
                    case NPY_TYPES.NPY_FLOAT:
                        return Convert.ToSingle(iValue);
                    case NPY_TYPES.NPY_DOUBLE:
                        return Convert.ToDouble(iValue);
                    case NPY_TYPES.NPY_DECIMAL:
                        return Convert.ToDecimal(iValue);
                    default:
                        throw new Exception("Unsupported data type");
                }
            }
            else if (value is sbyte)
            {
                sbyte iValue = (sbyte)value;

                switch (type_num)
                {
                    case NPY_TYPES.NPY_BOOL:
                        return Convert.ToBoolean(iValue);
                    case NPY_TYPES.NPY_BYTE:
                        return Convert.ToSByte(iValue);
                    case NPY_TYPES.NPY_UBYTE:
                        return Convert.ToByte(iValue);
                    case NPY_TYPES.NPY_UINT16:
                        return Convert.ToUInt16(iValue);
                    case NPY_TYPES.NPY_INT16:
                        return Convert.ToInt16(iValue);
                    case NPY_TYPES.NPY_UINT32:
                        return Convert.ToUInt32(iValue);
                    case NPY_TYPES.NPY_INT32:
                        return Convert.ToInt32(iValue);
                    case NPY_TYPES.NPY_INT64:
                        return Convert.ToInt64(iValue);
                    case NPY_TYPES.NPY_UINT64:
                        return Convert.ToUInt64(iValue);
                    case NPY_TYPES.NPY_FLOAT:
                        return Convert.ToSingle(iValue);
                    case NPY_TYPES.NPY_DOUBLE:
                        return Convert.ToDouble(iValue);
                    case NPY_TYPES.NPY_DECIMAL:
                        return Convert.ToDecimal(iValue);
                    default:
                        throw new Exception("Unsupported data type");
                }
            }
            else if (value is byte)
            {
                byte iValue = (byte)value;

                switch (type_num)
                {
                    case NPY_TYPES.NPY_BOOL:
                        return Convert.ToBoolean(iValue);
                    case NPY_TYPES.NPY_BYTE:
                        return Convert.ToSByte(iValue);
                    case NPY_TYPES.NPY_UBYTE:
                        return Convert.ToByte(iValue);
                    case NPY_TYPES.NPY_UINT16:
                        return Convert.ToUInt16(iValue);
                    case NPY_TYPES.NPY_INT16:
                        return Convert.ToInt16(iValue);
                    case NPY_TYPES.NPY_UINT32:
                        return Convert.ToUInt32(iValue);
                    case NPY_TYPES.NPY_INT32:
                        return Convert.ToInt32(iValue);
                    case NPY_TYPES.NPY_INT64:
                        return Convert.ToInt64(iValue);
                    case NPY_TYPES.NPY_UINT64:
                        return Convert.ToUInt64(iValue);
                    case NPY_TYPES.NPY_FLOAT:
                        return Convert.ToSingle(iValue);
                    case NPY_TYPES.NPY_DOUBLE:
                        return Convert.ToDouble(iValue);
                    case NPY_TYPES.NPY_DECIMAL:
                        return Convert.ToDecimal(iValue);
                    default:
                        throw new Exception("Unsupported data type");
                }
            }
            else if (value is Boolean)
            {
                return value;
            }
            else if (value is string)
            {
                string iValue = (string)value;

                switch (type_num)
                {
                    case NPY_TYPES.NPY_BOOL:
                        throw new Exception("Can't convert string to boolean");
                    case NPY_TYPES.NPY_BYTE:
                        return Convert.ToSByte(iValue);
                    case NPY_TYPES.NPY_UBYTE:
                        return Convert.ToByte(iValue);
                    case NPY_TYPES.NPY_UINT16:
                        return Convert.ToUInt16(iValue);
                    case NPY_TYPES.NPY_INT16:
                        return Convert.ToInt16(iValue);
                    case NPY_TYPES.NPY_UINT32:
                        return Convert.ToUInt32(iValue);
                    case NPY_TYPES.NPY_INT32:
                        return Convert.ToInt32(iValue);
                    case NPY_TYPES.NPY_INT64:
                        return Convert.ToInt64(iValue);
                    case NPY_TYPES.NPY_UINT64:
                        return Convert.ToUInt64(iValue);
                    case NPY_TYPES.NPY_FLOAT:
                        return Convert.ToSingle(iValue);
                    case NPY_TYPES.NPY_DOUBLE:
                        return Convert.ToDouble(iValue);
                    case NPY_TYPES.NPY_DECIMAL:
                        return Convert.ToDecimal(iValue);
                    default:
                        throw new Exception("Unsupported data type");
                }
            }
            else
            {
                throw new Exception("Dumb ass");
            }

            return value;
        }

        internal static int SetIndex(VoidPtr obj, npy_intp index, object invalue)
        {
            if (index < 0)
            {
                dynamic dyndatap = obj.datap;
                index = dyndatap.Length - Math.Abs(index);
            }

            var value = invalue; // CoerceValue(invalue, obj.type_num);

            try
            {
                switch (obj.type_num)
                {
                    case NPY_TYPES.NPY_BOOL:
                        var dbool = obj.datap as bool[];
                        dbool[index] = Convert.ToBoolean(value);
                        break;
                    case NPY_TYPES.NPY_BYTE:
                        var dsbyte = obj.datap as sbyte[];
                        dsbyte[index] = Convert.ToSByte(value);
                        break;
                    case NPY_TYPES.NPY_UBYTE:
                        var dbyte = obj.datap as byte[];
                        dbyte[index] = Convert.ToByte(value);
                        break;
                    case NPY_TYPES.NPY_UINT16:
                        var duint16 = obj.datap as UInt16[];
                        duint16[index] = Convert.ToUInt16(value);
                        break;
                    case NPY_TYPES.NPY_INT16:
                        var dint16 = obj.datap as Int16[];
                        dint16[index] = Convert.ToInt16(value);
                        break;
                    case NPY_TYPES.NPY_UINT32:
                        var duint32 = obj.datap as UInt32[];
                        duint32[index] = Convert.ToUInt32(value);
                        break;
                    case NPY_TYPES.NPY_INT32:
                        var dint32 = obj.datap as Int32[];
                        dint32[index] = Convert.ToInt32(value);
                        break;
                    case NPY_TYPES.NPY_INT64:
                        var dint64 = obj.datap as Int64[];
                        dint64[index] = Convert.ToInt64(value);
                        break;
                    case NPY_TYPES.NPY_UINT64:
                        var duint64 = obj.datap as UInt64[];
                        duint64[index] = Convert.ToUInt64(value);
                        break;
                    case NPY_TYPES.NPY_FLOAT:
                        var float1 = obj.datap as float[];
                        float1[index] = Convert.ToSingle(value);
                        break;
                    case NPY_TYPES.NPY_DOUBLE:
                        var double1 = obj.datap as double[];
                        double1[index] = Convert.ToDouble(value);
                        break;
                    case NPY_TYPES.NPY_DECIMAL:
                        var decimal1 = obj.datap as decimal[];
                        decimal1[index] = Convert.ToDecimal(value);
                        break;
                    case NPY_TYPES.NPY_COMPLEX:
                        var complex1 = obj.datap as System.Numerics.Complex[];
                        if (value is System.Numerics.Complex)
                        {
                            complex1[index] = (System.Numerics.Complex)value;
                        }
                        else
                        {
                            complex1[index] = new System.Numerics.Complex(Convert.ToDouble(value), 0);
                        }
                        break;
                    default:
                        throw new Exception("Unsupported data type");
                }
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
                VoidPtr Temp = new VoidPtr(new System.Numerics.Complex[len / sizeof(decimal) * 2]);
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
