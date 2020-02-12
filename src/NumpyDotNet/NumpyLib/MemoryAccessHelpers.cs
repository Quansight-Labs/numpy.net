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

// turn this on to use the built int bit converter functions. 
// These take endian issues into account but are a little slower.

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
    class MemoryAccess
    {
        private static bool IsLittleEndian = BitConverter.IsLittleEndian;
        private static bool USE_BITCONVERTER
        {
            get
            {
                return !IsLittleEndian;
            }
        }

        public MemoryAccess()
        {
        }
        

        public static byte GetByteT<T>(T[] _Array, npy_intp byte_index)
        {
            if (_Array is byte[])
            {
                byte[] Array = _Array as byte[];
                return GetByte(Array, byte_index);
            }
            if (_Array is Int16[])
            {
                Int16[] Array = _Array as Int16[];
                return GetByte(Array, byte_index);
            }
            if (_Array is UInt16[])
            {
                UInt16[] Array = _Array as UInt16[];
                return GetByte(Array, byte_index);
            }
            if (_Array is Int32[])
            {
                Int32[] Array = _Array as Int32[];
                return GetByte(Array, byte_index);
            }
            if (_Array is UInt32[])
            {
                UInt32[] Array = _Array as UInt32[];
                return GetByte(Array, byte_index);
            }
            if (_Array is Int64[])
            {
                Int64[] Array = _Array as Int64[];
                return GetByte(Array, byte_index);
            }
            if (_Array is UInt64[])
            {
                UInt64[] Array = _Array as UInt64[];
                return GetByte(Array, byte_index);
            }
            if (_Array is float[])
            {
                float[] Array = _Array as float[];
                return GetByte(Array, byte_index);
            }
            if (_Array is double[])
            {
                double[] Array = _Array as double[];
                return GetByte(Array, byte_index);
            }

            return 0;
        }

        public static byte GetByte(byte[] Array, npy_intp byte_index)
        {
            return Array[byte_index];
        }
        public static sbyte GetByte(sbyte[] Array, npy_intp byte_index)
        {
            return Array[byte_index];
        }
        public static bool GetByte(bool[] Array, npy_intp byte_index)
        {
            return Array[byte_index];
        }

        public static byte GetByte(Int16[] Array, npy_intp byte_index)
        {
            npy_intp ArrayIndex = byte_index / 2;
            npy_intp ByteNumber = byte_index % 2;

            UInt16 ArrayValue = (UInt16)Array[ArrayIndex];

            if (USE_BITCONVERTER)
            {
                var byteArray = BitConverter.GetBytes(ArrayValue);
                return (byteArray[ByteNumber]);
            }
            else
            {
                switch (ByteNumber)
                {
                    case 1:
                        return (byte)((ArrayValue & 0xFF00) >> 8);
                    case 0:
                        return (byte)((ArrayValue & 0x00FF) >> 0);
                }
                return 0;
            }
        }

        public static byte GetByte(UInt16[] Array, npy_intp byte_index)
        {
            npy_intp ArrayIndex = byte_index / 2;
            npy_intp ByteNumber = byte_index % 2;

            UInt16 ArrayValue = (UInt16)Array[ArrayIndex];

            if (USE_BITCONVERTER)
            {
                var byteArray = BitConverter.GetBytes(ArrayValue);
                return (byteArray[ByteNumber]);
            }
            else
            {
                switch (ByteNumber)
                {
                    case 1:
                        return (byte)((ArrayValue & 0xFF00) >> 8);
                    case 0:
                        return (byte)((ArrayValue & 0x00FF) >> 0);
                }
                return 0;
            }
        }

        public static byte GetByte(Int32[] Array, npy_intp byte_index)
        {
            npy_intp ArrayIndex = byte_index / 4;
            npy_intp ByteNumber = byte_index % 4;

            UInt32 ArrayValue = (UInt32)Array[ArrayIndex];

            if (USE_BITCONVERTER)
            {
                var byteArray = BitConverter.GetBytes(ArrayValue);
                return (byteArray[ByteNumber]);
            }
            else
            {
                switch (ByteNumber)
                {
                    case 3:
                        return (byte)((ArrayValue & 0xFF000000) >> 24);
                    case 2:
                        return (byte)((ArrayValue & 0x00FF0000) >> 16);
                    case 1:
                        return (byte)((ArrayValue & 0x0000FF00) >> 8);
                    case 0:
                        return (byte)((ArrayValue & 0x000000FF));
                }
                return 0;
            }
        }
        public static byte GetByte(UInt32[] Array, npy_intp byte_index)
        {
            npy_intp ArrayIndex = byte_index / 4;
            npy_intp ByteNumber = byte_index % 4;

            UInt32 ArrayValue = (UInt32)Array[ArrayIndex];
            
            if (USE_BITCONVERTER)
            {
                var byteArray = BitConverter.GetBytes(ArrayValue);
                return (byteArray[ByteNumber]);
            }
            else
            {
                switch (ByteNumber)
                {
                    case 3:
                        return (byte)((ArrayValue & 0xFF000000) >> 24);
                    case 2:
                        return (byte)((ArrayValue & 0x00FF0000) >> 16);
                    case 1:
                        return (byte)((ArrayValue & 0x0000FF00) >> 8);
                    case 0:
                        return (byte)((ArrayValue & 0x000000FF));
                }
                return 0;
            }

        }

        public static byte GetByte(Int64[] Array, npy_intp byte_index)
        {
            npy_intp ArrayIndex = byte_index / 8;
            npy_intp ByteNumber = byte_index % 8;

            UInt64 ArrayValue = (UInt64)Array[ArrayIndex];

            if (USE_BITCONVERTER)
            {
                var byteArray = BitConverter.GetBytes(ArrayValue);
                return (byteArray[ByteNumber]);
            }
            else
            {
                switch (ByteNumber)
                {
                    case 7:
                        return (byte)((ArrayValue & 0xFF00000000000000) >> 56);
                    case 6:
                        return (byte)((ArrayValue & 0x00FF000000000000) >> 48);
                    case 5:
                        return (byte)((ArrayValue & 0x0000FF0000000000) >> 40);
                    case 4:
                        return (byte)((ArrayValue & 0x000000FF00000000) >> 32);
                    case 3:
                        return (byte)((ArrayValue & 0x00000000FF000000) >> 24);
                    case 2:
                        return (byte)((ArrayValue & 0x0000000000FF0000) >> 16);
                    case 1:
                        return (byte)((ArrayValue & 0x000000000000FF00) >> 8);
                    case 0:
                        return (byte)((ArrayValue & 0x00000000000000FF) >> 0);
                }
                return 0;
            }
        }

        public static byte GetByte(UInt64[] Array, npy_intp byte_index)
        {
            npy_intp ArrayIndex = byte_index / 8;
            npy_intp ByteNumber = byte_index % 8;

            UInt64 ArrayValue = (UInt64)Array[ArrayIndex];

            if (USE_BITCONVERTER)
            {
                var byteArray = BitConverter.GetBytes(ArrayValue);
                return (byteArray[ByteNumber]);
            }
            else
            {
                switch (ByteNumber)
                {
                    case 7:
                        return (byte)((ArrayValue & 0xFF00000000000000) >> 56);
                    case 6:
                        return (byte)((ArrayValue & 0x00FF000000000000) >> 48);
                    case 5:
                        return (byte)((ArrayValue & 0x0000FF0000000000) >> 40);
                    case 4:
                        return (byte)((ArrayValue & 0x000000FF00000000) >> 32);
                    case 3:
                        return (byte)((ArrayValue & 0x00000000FF000000) >> 24);
                    case 2:
                        return (byte)((ArrayValue & 0x0000000000FF0000) >> 16);
                    case 1:
                        return (byte)((ArrayValue & 0x000000000000FF00) >> 8);
                    case 0:
                        return (byte)((ArrayValue & 0x00000000000000FF) >> 0);
                }
                return 0;
            }
        }

        public static byte GetByte(double[] Array, npy_intp byte_index)
        {
            npy_intp ArrayIndex = byte_index / 8;
            npy_intp ByteNumber = byte_index % 8;

            var bytes = BitConverter.GetBytes(Array[ArrayIndex]);

            return bytes[ByteNumber];
        }

        public static byte GetByte(decimal[] Array, npy_intp byte_index)
        {
            //npy_intp ArrayIndex = byte_index / sizeof(decimal);
            //npy_intp ByteNumber = byte_index % sizeof(decimal);

            //var bytes = new byte[200]; 

            //return bytes[ByteNumber];

            return 0;
        }

        public static byte GetByte(float[] Array, npy_intp byte_index)
        {
            npy_intp ArrayIndex = byte_index / 4;
            npy_intp ByteNumber = byte_index % 4;

            var bytes = BitConverter.GetBytes(Array[ArrayIndex]);

            return bytes[ByteNumber];
        }

        public static void SetByteT<T>(T[] _Array, npy_intp byte_index, byte data)
        {
            if (_Array is byte[])
            {
                byte[] Array = _Array as byte[];
                SetByte(Array, byte_index, data);
            }
            else
            if (_Array is Int16[])
            {
                Int16[] Array = _Array as Int16[];
                SetByte(Array, byte_index, data);
            }
            else
            if (_Array is UInt16[])
            {
                UInt16[] Array = _Array as UInt16[];
                SetByte(Array, byte_index, data);
            }
            else
            if (_Array is Int32[])
            {
                Int32[] Array = _Array as Int32[];
                SetByte(Array, byte_index, data);
            }
            else
            if (_Array is UInt32[])
            {
                UInt32[] Array = _Array as UInt32[];
                SetByte(Array, byte_index, data);
            }
            else
            if (_Array is Int64[])
            {
                Int64[] Array = _Array as Int64[];
                SetByte(Array, byte_index, data);
            }
            else
            if (_Array is UInt64[])
            {
                UInt64[] Array = _Array as UInt64[];
                SetByte(Array, byte_index, data);
            }
            else
            if (_Array is float[])
            {
                float[] Array = _Array as float[];
                SetByte(Array, byte_index, data);
            }
            else
            if (_Array is double[])
            {
                double[] Array = _Array as double[];
                SetByte(Array, byte_index, data);
            }

            return;
        }

        public static void SetByte(bool[] Array, npy_intp byte_index, bool data)
        {
            Array[byte_index] = data;
        }

        public static void SetByte(byte[] Array, npy_intp byte_index, byte data)
        {
            Array[byte_index] = data;
        }

  

        public static void SetByte(Int16[] Array, npy_intp byte_index, byte data)
        {
            npy_intp ArrayIndex = byte_index / 2;
            npy_intp ByteNumber = byte_index % 2;

            UInt16 ArrayValue = (UInt16)Array[ArrayIndex];
            if (USE_BITCONVERTER)
            {
                var byteArray = BitConverter.GetBytes(ArrayValue);
                byteArray[ByteNumber] = data;
                Array[ArrayIndex] = BitConverter.ToInt16(byteArray, 0);
                return;
            }
            else
            {
                switch (ByteNumber)
                {
                    case 1:
                        ArrayValue = (UInt16)(ArrayValue & 0x00FF);
                        ArrayValue = (UInt16)(ArrayValue + (data << 8));
                        break;
                    case 0:
                        ArrayValue = (UInt16)(ArrayValue & 0xFF00);
                        ArrayValue = (UInt16)(ArrayValue + (data));
                        break;
                }
                Array[ArrayIndex] = (Int16)ArrayValue;
                return;
            }

        }

        public static void SetByte(UInt16[] Array, npy_intp byte_index, byte data)
        {
            npy_intp ArrayIndex = byte_index / 2;
            npy_intp ByteNumber = byte_index % 2;

            UInt16 ArrayValue = Array[ArrayIndex];

            if (USE_BITCONVERTER)
            {
                var byteArray = BitConverter.GetBytes(ArrayValue);
                byteArray[ByteNumber] = data;
                Array[ArrayIndex] = BitConverter.ToUInt16(byteArray, 0);
                return;
            }
            else
            {
                switch (ByteNumber)
                {
                    case 1:
                        ArrayValue = (UInt16)(ArrayValue & 0x00FF);
                        ArrayValue = (UInt16)(ArrayValue + (data << 8));
                        break;
                    case 0:
                        ArrayValue = (UInt16)(ArrayValue & 0xFF00);
                        ArrayValue = (UInt16)(ArrayValue + (data));
                        break;

                }

                Array[ArrayIndex] = (UInt16)ArrayValue;
                return;
            }
        }

        public static void SetByte(Int32[] Array, npy_intp byte_index, byte data)
        {
            npy_intp ArrayIndex = byte_index / 4;
            npy_intp ByteNumber = byte_index % 4;

            UInt32 ArrayValue = (UInt32)Array[ArrayIndex];

            if (USE_BITCONVERTER)
            {
                var byteArray = BitConverter.GetBytes(ArrayValue);
                byteArray[ByteNumber] = data;
                Array[ArrayIndex] = BitConverter.ToInt32(byteArray, 0);
                return;
            }
            else
            {
                switch (ByteNumber)
                {
                    case 3:
                        ArrayValue = (UInt32)(ArrayValue & 0x00FFFFFF);
                        ArrayValue = (UInt32)(ArrayValue + (data << 24));
                        break;
                    case 2:
                        ArrayValue = (UInt32)(ArrayValue & 0xFF00FFFF);
                        ArrayValue = (UInt32)(ArrayValue + (data << 16));
                        break;
                    case 1:
                        ArrayValue = (UInt32)(ArrayValue & 0xFFFF00FF);
                        ArrayValue = (UInt32)(ArrayValue + (data << 8));
                        break;
                    case 0:
                        ArrayValue = (UInt32)(ArrayValue & 0xFFFFFF00);
                        ArrayValue = (UInt32)(ArrayValue + (data));
                        break;
                }
                Array[ArrayIndex] = (Int32)ArrayValue;

                return;
            }

        }

        public static void SetByte(UInt32[] Array, npy_intp byte_index, byte data)
        {
            npy_intp ArrayIndex = byte_index / 4;
            npy_intp ByteNumber = byte_index % 4;

            UInt32 ArrayValue = (UInt32)Array[ArrayIndex];

            if (USE_BITCONVERTER)
            {
                var byteArray = BitConverter.GetBytes(ArrayValue);
                byteArray[ByteNumber] = data;
                Array[ArrayIndex] = BitConverter.ToUInt32(byteArray, 0);
                return;
            }
            else
            {
                switch (ByteNumber)
                {
                    case 3:
                        ArrayValue = (UInt32)(ArrayValue & 0x00FFFFFF);
                        ArrayValue = (UInt32)(ArrayValue + (data << 24));
                        break;
                    case 2:
                        ArrayValue = (UInt32)(ArrayValue & 0xFF00FFFF);
                        ArrayValue = (UInt32)(ArrayValue + (data << 16));
                        break;
                    case 1:
                        ArrayValue = (UInt32)(ArrayValue & 0xFFFF00FF);
                        ArrayValue = (UInt32)(ArrayValue + (data << 8));
                        break;
                    case 0:
                        ArrayValue = (UInt32)(ArrayValue & 0xFFFFFF00);
                        ArrayValue = (UInt32)(ArrayValue + (data));
                        break;
                }
                Array[ArrayIndex] = (UInt32)ArrayValue;

                return;
            }
        }

        public static void SetByte(Int64[] Array, npy_intp byte_index, byte data)
        {
            npy_intp ArrayIndex = byte_index / 8;
            npy_intp ByteNumber = byte_index % 8;

            UInt64 ArrayValue = (UInt64)Array[ArrayIndex];


            if (USE_BITCONVERTER)
            {
                var byteArray = BitConverter.GetBytes(ArrayValue);
                byteArray[ByteNumber] = data;
                Array[ArrayIndex] = BitConverter.ToInt64(byteArray, 0);
                return;
            }
            else
            {
                switch (ByteNumber)
                {
                    case 7:
                        ArrayValue = (UInt64)(ArrayValue & 0x00FFFFFFFFFFFFFF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 56));
                        break;
                    case 6:
                        ArrayValue = (UInt64)(ArrayValue & 0xFF00FFFFFFFFFFFF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 48));
                        break;
                    case 5:
                        ArrayValue = (UInt64)(ArrayValue & 0xFFFF00FFFFFFFFFF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 40));
                        break;
                    case 4:
                        ArrayValue = (UInt64)(ArrayValue & 0xFFFFFF00FFFFFFFF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 32));
                        break;
                    case 3:
                        ArrayValue = (UInt64)(ArrayValue & 0xFFFFFFFF00FFFFFF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 24));
                        break;
                    case 2:
                        ArrayValue = (UInt64)(ArrayValue & 0xFFFFFFFFFF00FFFF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 16));
                        break;
                    case 1:
                        ArrayValue = (UInt64)(ArrayValue & 0xFFFFFFFFFFFF00FF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 8));
                        break;
                    case 0:
                        ArrayValue = (UInt64)(ArrayValue & 0xFFFFFFFFFFFFFF00);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 0));
                        break;
                }
                Array[ArrayIndex] = (Int64)ArrayValue;

                return;
            }
        }

        public static void SetByte(UInt64[] Array, npy_intp byte_index, byte data)
        {
            npy_intp ArrayIndex = byte_index / 8;
            npy_intp ByteNumber = byte_index % 8;

            UInt64 ArrayValue = (UInt64)Array[ArrayIndex];

            if (USE_BITCONVERTER)
            {
                var byteArray = BitConverter.GetBytes(ArrayValue);
                byteArray[ByteNumber] = data;
                Array[ArrayIndex] = BitConverter.ToUInt64(byteArray, 0);
                return;
            }
            else
            {
                switch (ByteNumber)
                {
                    case 7:
                        ArrayValue = (UInt64)(ArrayValue & 0x00FFFFFFFFFFFFFF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 56));
                        break;
                    case 6:
                        ArrayValue = (UInt64)(ArrayValue & 0xFF00FFFFFFFFFFFF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 48));
                        break;
                    case 5:
                        ArrayValue = (UInt64)(ArrayValue & 0xFFFF00FFFFFFFFFF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 40));
                        break;
                    case 4:
                        ArrayValue = (UInt64)(ArrayValue & 0xFFFFFF00FFFFFFFF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 32));
                        break;
                    case 3:
                        ArrayValue = (UInt64)(ArrayValue & 0xFFFFFFFF00FFFFFF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 24));
                        break;
                    case 2:
                        ArrayValue = (UInt64)(ArrayValue & 0xFFFFFFFFFF00FFFF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 16));
                        break;
                    case 1:
                        ArrayValue = (UInt64)(ArrayValue & 0xFFFFFFFFFFFF00FF);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 8));
                        break;
                    case 0:
                        ArrayValue = (UInt64)(ArrayValue & 0xFFFFFFFFFFFFFF00);
                        ArrayValue = (UInt64)(ArrayValue + ((UInt64)data << 0));
                        break;
                }
                Array[ArrayIndex] = (UInt64)ArrayValue;

                return;
            }

        }

        public static void SetByte(float[] Array, npy_intp byte_index, byte data)
        {
            npy_intp ArrayIndex = byte_index / 4;
            npy_intp ByteNumber = byte_index % 4;

            var bytes = BitConverter.GetBytes(Array[ArrayIndex]);
            bytes[ByteNumber] = data;
            Array[ArrayIndex] = BitConverter.ToSingle(bytes, 0);

            return;
        }

        public static void SetByte(double[] Array, npy_intp byte_index, byte data)
        {
            npy_intp ArrayIndex = byte_index / 8;
            npy_intp ByteNumber = byte_index % 8;

            var bytes = BitConverter.GetBytes(Array[ArrayIndex]);
            bytes[ByteNumber] = data;
            Array[ArrayIndex] = BitConverter.ToDouble(bytes, 0);

            return;
        }

        public static void SetByte(decimal[] Array, npy_intp byte_index, byte data)
        {
            //npy_intp ArrayIndex = byte_index / 8;
            //npy_intp ByteNumber = byte_index % 8;

            // todo: figure out how to do this.
            //var bytes = BitConverter.GetBytes(Array[ArrayIndex]);
            //bytes[ByteNumber] = data;
            //Array[ArrayIndex] = BitConverter.ToDecimal(bytes, 0);

            return;
        }

    }

}
