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

namespace NumpyLib
{
    class ArrayConversions
    {
        #region First Level Routing
        public static VoidPtr ConvertToDesiredArrayType(VoidPtr Src, int SrcOffset, int Length, NPY_TYPES type_num)
        {
            switch (type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return ConvertToBools(Src, SrcOffset, Length); 
                case NPY_TYPES.NPY_BYTE:
                    return ConvertToBytes(Src, SrcOffset, Length);
                case NPY_TYPES.NPY_UBYTE:
                    return ConvertToUBytes(Src, SrcOffset, Length);
                case NPY_TYPES.NPY_INT16:
                    return ConvertToInt16s(Src, SrcOffset, Length);
                case NPY_TYPES.NPY_UINT16:
                    return ConvertToUInt16s(Src, SrcOffset, Length);
                case NPY_TYPES.NPY_INT32:
                    return ConvertToInt32s(Src, SrcOffset, Length);
                case NPY_TYPES.NPY_UINT32:
                    return ConvertToUInt32s(Src, SrcOffset, Length);
                case NPY_TYPES.NPY_INT64:
                    return ConvertToInt64s(Src, SrcOffset, Length);
                case NPY_TYPES.NPY_UINT64:
                    return ConvertToUInt64s(Src, SrcOffset, Length);
                case NPY_TYPES.NPY_FLOAT:
                    return ConvertToFloats(Src, SrcOffset, Length);
                case NPY_TYPES.NPY_DOUBLE:
                    return ConvertToDoubles(Src, SrcOffset, Length);
                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertToDecimals(Src, SrcOffset, Length);
            }
            return null;
        }
        #endregion

        #region Second level routing
        private static VoidPtr ConvertToBools(VoidPtr Src, int SrcOffset, int Length)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_INT64:
                case NPY_TYPES.NPY_UINT64:
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertAllToBools(Src, SrcOffset, Length);
            }
            return null;
        }

  
        private static VoidPtr ConvertToBytes(VoidPtr Src, int SrcOffset, int Length)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_INT64:
                case NPY_TYPES.NPY_UINT64:
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertAllToBytes(Src, SrcOffset, Length);
            }
            return null;
        }
        private static VoidPtr ConvertToUBytes(VoidPtr Src, int SrcOffset, int Length)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_INT64:
                case NPY_TYPES.NPY_UINT64:
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertAllToUBytes(Src, SrcOffset, Length);
            }
            return null;
        }

        private static VoidPtr ConvertToInt16s(VoidPtr Src, int SrcOffset, int Length)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_INT64:
                case NPY_TYPES.NPY_UINT64:
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertAllToInt16s(Src, SrcOffset, Length);
            }
            return null;
        }

        private static VoidPtr ConvertToUInt16s(VoidPtr Src, int SrcOffset, int Length)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_INT64:
                case NPY_TYPES.NPY_UINT64:
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertAllToUInt16s(Src, SrcOffset, Length);
            }
            return null;
        }

        private static VoidPtr ConvertToInt32s(VoidPtr Src, int SrcOffset, int Length)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_INT64:
                case NPY_TYPES.NPY_UINT64:
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertAllToInt32s(Src, SrcOffset, Length);
            }
            return null;
        }

        private static VoidPtr ConvertToUInt32s(VoidPtr Src, int SrcOffset, int Length)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_INT64:
                case NPY_TYPES.NPY_UINT64:
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertAllToUInt32s(Src, SrcOffset, Length);
            }
            return null;
        }

        private static VoidPtr ConvertToInt64s(VoidPtr Src, int SrcOffset, int Length)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_INT64:
                case NPY_TYPES.NPY_UINT64:
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertAllToInt64s(Src, SrcOffset, Length);
            }
            return null;
        }

        private static VoidPtr ConvertToUInt64s(VoidPtr Src, int SrcOffset, int Length)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_INT64:
                case NPY_TYPES.NPY_UINT64:
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertAllToUInt64s(Src, SrcOffset, Length);
            }
            return null;
        }

        private static VoidPtr ConvertToFloats(VoidPtr Src, int SrcOffset, int Length)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_INT64:
                case NPY_TYPES.NPY_UINT64:
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertAllToFloats(Src, SrcOffset, Length);
            }
            return null;
        }

        private static VoidPtr ConvertToDoubles(VoidPtr Src, int SrcOffset, int Length)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_INT64:
                case NPY_TYPES.NPY_UINT64:
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertAllToDoubles(Src, SrcOffset, Length);
            }
            return null;
        }


        private static VoidPtr ConvertToDecimals(VoidPtr Src, int SrcOffset, int Length)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_INT64:
                case NPY_TYPES.NPY_UINT64:
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertAllToDecimals(Src, SrcOffset, Length);
            }
            return null;
        }

        #endregion

        #region specific converters that create a new array and copy into it with MemCopy
        private static VoidPtr ConvertAllToBools(VoidPtr src, int srcOffset, int length)
        {
            int totalBytesToCopy = (length - srcOffset);

            bool[] destArray = new bool[(totalBytesToCopy+sizeof(bool)-1) / sizeof(bool)];

            VoidPtr Dest = new VoidPtr(destArray);
            MemCopy.MemCpy(Dest, 0, src, srcOffset, totalBytesToCopy);
            return Dest;
        }

        private static VoidPtr ConvertAllToBytes(VoidPtr src, int srcOffset, int length)
        {
            int totalBytesToCopy = (length - srcOffset);

            sbyte[] destArray = new sbyte[(totalBytesToCopy + sizeof(sbyte) - 1) / sizeof(sbyte)];

            VoidPtr Dest = new VoidPtr(destArray);
            MemCopy.MemCpy(Dest, 0, src, srcOffset, totalBytesToCopy);
            return Dest;
        }

        private static VoidPtr ConvertAllToUBytes(VoidPtr src, int srcOffset, int length)
        {
            int totalBytesToCopy = (length - srcOffset);

            byte[] destArray = new byte[(totalBytesToCopy + sizeof(byte) - 1) / sizeof(byte)];

            VoidPtr Dest = new VoidPtr(destArray);
            MemCopy.MemCpy(Dest, 0, src, srcOffset, totalBytesToCopy);
            return Dest;
        }

        private static VoidPtr ConvertAllToInt16s(VoidPtr src, int srcOffset, int length)
        {
            int totalBytesToCopy = (length - srcOffset);

            Int16[] destArray = new Int16[(totalBytesToCopy + sizeof(Int16) -1)/ sizeof(Int16)];

            VoidPtr Dest = new VoidPtr(destArray);
            MemCopy.MemCpy(Dest, 0, src, srcOffset, totalBytesToCopy);
            return Dest;
        }

        private static VoidPtr ConvertAllToUInt16s(VoidPtr src, int srcOffset, int length)
        {
            int totalBytesToCopy = (length - srcOffset);

            UInt16[] destArray = new UInt16[(totalBytesToCopy + sizeof(UInt16)-1) / sizeof(UInt16)];

            VoidPtr Dest = new VoidPtr(destArray);
            MemCopy.MemCpy(Dest, 0, src, srcOffset, totalBytesToCopy);
            return Dest;
        }

        private static VoidPtr ConvertAllToInt32s(VoidPtr src, int srcOffset, int length)
        {
            int totalBytesToCopy = (length - srcOffset);

            Int32[] destArray = new Int32[(totalBytesToCopy + sizeof(Int32) - 1) / sizeof(Int32)];

            VoidPtr Dest = new VoidPtr(destArray);
            MemCopy.MemCpy(Dest, 0, src, srcOffset, totalBytesToCopy);
            return Dest;
        }
   
        private static VoidPtr ConvertAllToUInt32s(VoidPtr src, int srcOffset, int length)
        {
            int totalBytesToCopy = (length - srcOffset);

            UInt32[] destArray = new UInt32[(totalBytesToCopy + sizeof(UInt32) - 1) / sizeof(UInt32)];

            VoidPtr Dest = new VoidPtr(destArray);
            MemCopy.MemCpy(Dest, 0, src, srcOffset, totalBytesToCopy);
            return Dest;
        }
  
        private static VoidPtr ConvertAllToInt64s(VoidPtr src, int srcOffset, int length)
        {
            int totalBytesToCopy = (length - srcOffset);

            Int64[] destArray = new Int64[(totalBytesToCopy + sizeof(Int64) - 1) / sizeof(Int64)];

            VoidPtr Dest = new VoidPtr(destArray);
            MemCopy.MemCpy(Dest, 0, src, srcOffset, totalBytesToCopy);
            return Dest;
        }

        private static VoidPtr ConvertAllToUInt64s(VoidPtr src, int srcOffset, int length)
        {
            int totalBytesToCopy = (length - srcOffset);

            UInt64[] destArray = new UInt64[(totalBytesToCopy + sizeof(UInt64) - 1) / sizeof(UInt64)];

            VoidPtr Dest = new VoidPtr(destArray);
            MemCopy.MemCpy(Dest, 0, src, srcOffset, totalBytesToCopy);
            return Dest;
        }

        private static VoidPtr ConvertAllToFloats(VoidPtr src, int srcOffset, int length)
        {
            int totalBytesToCopy = (length - srcOffset);

            float[] destArray = new float[(totalBytesToCopy + sizeof(float) - 1) / sizeof(float)];

            VoidPtr Dest = new VoidPtr(destArray);
            MemCopy.MemCpy(Dest, 0, src, srcOffset, totalBytesToCopy);
            return Dest;
        }
  
  
        private static VoidPtr ConvertAllToDoubles(VoidPtr src, int srcOffset, int length)
        {
            int totalBytesToCopy = (length - srcOffset);

            double[] destArray = new double[(totalBytesToCopy + sizeof(double) - 1) / sizeof(double)];

            VoidPtr Dest = new VoidPtr(destArray);
            MemCopy.MemCpy(Dest, 0, src, srcOffset, totalBytesToCopy);
            return Dest;
        }

        private static VoidPtr ConvertAllToDecimals(VoidPtr src, int srcOffset, int length)
        {
            int totalBytesToCopy = (length - srcOffset);

            decimal[] destArray = new decimal[(totalBytesToCopy + sizeof(decimal) - 1) / sizeof(decimal)];

            VoidPtr Dest = new VoidPtr(destArray);
            MemCopy.MemCpy(Dest, 0, src, srcOffset, totalBytesToCopy);
            return Dest;
        }
        #endregion

    }
}
