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
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyLib
{
    internal class MemCopy
    {

        static int ItemDivInt16 = numpyinternal.GetDivSize(sizeof(Int16));
        static int ItemDivUInt16 = numpyinternal.GetDivSize(sizeof(UInt16));
        static int ItemDivInt32 = numpyinternal.GetDivSize(sizeof(Int32));
        static int ItemDivUInt32 = numpyinternal.GetDivSize(sizeof(UInt32));
        static int ItemDivInt64 = numpyinternal.GetDivSize(sizeof(Int64));
        static int ItemDivUInt64 = numpyinternal.GetDivSize(sizeof(UInt64));
        static int ItemDivFloat = numpyinternal.GetDivSize(sizeof(float));
        static int ItemDivDouble = numpyinternal.GetDivSize(sizeof(double));
        static int ItemDivDecimal = numpyinternal.GetDivSize(sizeof(decimal));
        static int __ComplexSize = -1;
        static int ItemDivComplex = 0;
        static int __BigIntSize = -1;
        static int ItemDivBigInt = 0;
        static int __ObjectSize = -1;
        static int ItemDivObject = 0;
        static int __StringSize = -1;
        static int ItemDivString = 0;

        static int ItemMaskInt16 = numpyinternal.GetMaskSize(sizeof(Int16));
        static int ItemMaskUInt16 = numpyinternal.GetMaskSize(sizeof(UInt16));
        static int ItemMaskInt32 = numpyinternal.GetMaskSize(sizeof(Int32));
        static int ItemMaskUInt32 = numpyinternal.GetMaskSize(sizeof(UInt32));
        static int ItemMaskInt64 = numpyinternal.GetMaskSize(sizeof(Int64));
        static int ItemMaskUInt64 = numpyinternal.GetMaskSize(sizeof(UInt64));
        static int ItemMaskFloat = numpyinternal.GetMaskSize(sizeof(float));
        static int ItemMaskDouble = numpyinternal.GetMaskSize(sizeof(double));
        static int ItemMaskDecimal = numpyinternal.GetMaskSize(sizeof(decimal));
        static int ItemMaskComplex = 0;
        static int ItemMaskBigInt = 0;
        static int ItemMaskObject = 0;
        static int ItemMaskString = 0;

        // todo: take these out when performance improvements complete
        public static long MemCpy_TotalCount = 0;
        public static long MemCpy_ShortBufferCount = 0;
        public static long MemCpy_MediumBufferCount = 0;
        public static long MemCpy_LargeBufferCount = 0;
        public static long MemCpy_VeryLargeBufferCount = 0;
        public static long MemCpy_SameTypeCount = 0;
        public static long MemCpy_DifferentTypeCount = 0;
        public static long MemCpy_DifferentTypeCountLarge = 0;

        [System.Diagnostics.Conditional("DEBUG")]
        public static void MemCpyStats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            MemCpy_TotalCount++;

            if (Dest.type_num == Src.type_num)
            {
                MemCpy_SameTypeCount++;
            }
            else
            {
                MemCpy_DifferentTypeCount++;
                if (totalBytesToCopy > 32)
                    MemCpy_DifferentTypeCountLarge++;
            }
            if (totalBytesToCopy <= 4)
            {
                MemCpy_ShortBufferCount++;
            }
            else if (totalBytesToCopy <= 8)
            {
                MemCpy_MediumBufferCount++;
            }
            else if (totalBytesToCopy <= 128)
            {
                MemCpy_LargeBufferCount++;
            }
            else
            {
                MemCpy_VeryLargeBufferCount++;
            }
        }


        #region First Level Routing
        public static bool MemCpy(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            DestOffset += Dest.data_offset;
            SrcOffset += Src.data_offset;

            MemCpyStats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);


            switch (Dest.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyToBools(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyToBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyToUBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyToInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyToUInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyToInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyToUInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyToInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyToUInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyToFloats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyToDoubles(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyToDecimals(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_COMPLEX:
                    return MemCpyToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BIGINT:
                    return MemCpyToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_OBJECT:
                    return MemCpyToObject(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_STRING:
                    return MemCpyToString(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
            }
            return false;
        }
        #endregion

        #region Second level routing
        private static bool MemCpyToBools(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToBools(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToBools(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToBools(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToBools(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToBools(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToBools(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToBools(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToBools(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToBools(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToBools(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToBools(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToBools(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
   
            }
            return false;
        }


        private static bool MemCpyToBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
            }
            return false;
        }
        private static bool MemCpyToUBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToUBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToUBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToUBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToUBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToUBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToUBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToUBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToUBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToUBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToUBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToUBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToUBytes(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
            }
            return false;
        }

        private static bool MemCpyToInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
            }
            return false;
        }

        private static bool MemCpyToUInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToUInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToUInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToUInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToUInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToUInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToUInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToUInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToUInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToUInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToUInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToUInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToUInt16s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
            }
            return false;
        }

        private static bool MemCpyToInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
            }
            return false;
        }

        private static bool MemCpyToUInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToUInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToUInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToUInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToUInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToUInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToUInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToUInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToUInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToUInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToUInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToUInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToUInt32s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
            }
            return false;
        }

        private static bool MemCpyToInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
            }
            return false;
        }

        private static bool MemCpyToUInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToUInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToUInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToUInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToUInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToUInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToUInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToUInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToUInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToUInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToUInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToUInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToUInt64s(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
            }
            return false;
        }

        private static bool MemCpyToFloats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToFloats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToFloats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToFloats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToFloats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToFloats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToFloats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToFloats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToFloats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToFloats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToFloats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToFloats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToFloats(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
            }
            return false;
        }

        private static bool MemCpyToDoubles(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToDoubles(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToDoubles(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToDoubles(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToDoubles(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToDoubles(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToDoubles(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToDoubles(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToDoubles(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToDoubles(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToDoubles(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToDoubles(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToDoubles(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);

            }
            return false;
        }

        private static bool MemCpyToDecimals(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToDecimals(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToDecimals(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToDecimals(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToDecimals(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToDecimals(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToDecimals(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToDecimals(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToDecimals(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToDecimals(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToDecimals(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToDecimals(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToDecimals(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);

            }
            return false;
        }

        private static bool MemCpyToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_COMPLEX:
                    return MemCpyComplexToComplex(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                //default:
                //    throw new Exception("Attempt to copy non complex to complex number");


            }
            return false;
        }

        private static bool MemCpyToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return MemCpyBoolsToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BYTE:
                    return MemCpyBytesToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UBYTE:
                    return MemCpyUBytesToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT16:
                    return MemCpyInt16ToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT16:
                    return MemCpyUInt16ToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT32:
                    return MemCpyInt32ToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT32:
                    return MemCpyUInt32ToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_INT64:
                    return MemCpyInt64ToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_UINT64:
                    return MemCpyUInt64ToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_FLOAT:
                    return MemCpyFloatsToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DOUBLE:
                    return MemCpyDoublesToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_DECIMAL:
                    return MemCpyDecimalsToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_COMPLEX:
                    return MemCpyComplexToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                case NPY_TYPES.NPY_BIGINT:
                    return MemCpyBigIntToBigInt(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                //default:
                //    throw new Exception("Attempt to copy value to BigInt number");

            }
            return false;
        }

        private static bool MemCpyToObject(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_OBJECT:
                    return MemCpyObjectToObject(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                default:
                    throw new Exception("Attempt to copy non object value to object array");

            }
            return false;
        }

        private static bool MemCpyToString(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_STRING:
                    return MemCpyStringToString(Dest, DestOffset, Src, SrcOffset, totalBytesToCopy);
                default:
                    throw new Exception("Attempt to copy non object value to object array");

            }
            return false;
        }

        #endregion

        #region Common functions

        private static void CommonArrayCopy<T>(T[] destArray, npy_intp DestOffset, T[] sourceArray, npy_intp SrcOffset, long totalBytesToCopy, npy_intp DestOffsetAdjustment, npy_intp SrcOffsetAdjustment, int ItemDiv, int ItemMask)
        {
            for (npy_intp i = 0; i < DestOffsetAdjustment; i++)
            {
                byte data = MemoryAccess.GetByteT(sourceArray, i + SrcOffset);
                MemoryAccess.SetByteT(destArray, i + DestOffset, data);
            }

            totalBytesToCopy -= DestOffsetAdjustment;
            DestOffset += DestOffsetAdjustment;
            SrcOffset += SrcOffsetAdjustment;

            npy_intp LengthAdjustment = (npy_intp)(totalBytesToCopy & ItemMask);
            totalBytesToCopy -= LengthAdjustment;

            Array.Copy(sourceArray, SrcOffset >> ItemDiv, destArray, DestOffset >> ItemDiv, totalBytesToCopy >> ItemDiv);

            for (npy_intp i = (npy_intp)totalBytesToCopy; i < totalBytesToCopy + LengthAdjustment; i++)
            {
                byte data = MemoryAccess.GetByteT(sourceArray, i + SrcOffset);
                MemoryAccess.SetByteT(destArray, i + DestOffset, data);
            }

            return;
        }
        #endregion

        #region boolean specific
        private static bool MemCpyBoolsToBools(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {

            bool[] sourceArray = Src.datap as bool[];
            bool[] destArray = Dest.datap as bool[];

            Array.Copy(sourceArray, SrcOffset, destArray, DestOffset, totalBytesToCopy);

            return true;
        }
        private static bool MemCpyBytesToBools(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            sbyte[] sourceArray = Src.datap as sbyte[];
            bool[] destArray = Dest.datap as bool[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = sourceArray[i + SrcOffset];
                bool bdata = (bool)(data != 0 ? true : false);
                destArray[i + DestOffset] = bdata;
            }

            return true;
        }
        private static bool MemCpyUBytesToBools(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            byte[] sourceArray = Src.datap as byte[];
            bool[] destArray = Dest.datap as bool[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = sourceArray[i + SrcOffset];
                bool bdata = (bool)(data != 0 ? true : false);
                destArray[i + DestOffset] = bdata;
            }

            return true;
        }
        private static bool MemCpyInt16ToBools(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int16[] sourceArray = Src.datap as Int16[];
            bool[] destArray = Dest.datap as bool[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                bool bdata = (bool)(data != 0 ? true : false);
                destArray[i + DestOffset] =  bdata;
            }
            return true;
        }


        private static bool MemCpyUInt16ToBools(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {

            UInt16[] sourceArray = Src.datap as UInt16[];
            bool[] destArray = Dest.datap as bool[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                bool bdata = (bool)(data != 0 ? true : false);
                destArray[i + DestOffset] = bdata;
            }
            return true;
        }
        private static bool MemCpyInt32ToBools(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {

            Int32[] sourceArray = Src.datap as Int32[];
            bool[] destArray = Dest.datap as bool[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                bool bdata = (bool)(data != 0 ? true : false);
                destArray[i + DestOffset] = bdata;
            }
            return true;
        }
        private static bool MemCpyUInt32ToBools(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt32[] sourceArray = Src.datap as UInt32[];
            bool[] destArray = Dest.datap as bool[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                bool bdata = (bool)(data != 0 ? true : false);
                destArray[i + DestOffset] = bdata;
            }
            return true;
        }
        private static bool MemCpyInt64ToBools(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int64[] sourceArray = Src.datap as Int64[];
            bool[] destArray = Dest.datap as bool[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                bool bdata = (bool)(data != 0 ? true : false);
                destArray[i + DestOffset] = bdata;
            }
            return true;
        }
        private static bool MemCpyUInt64ToBools(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt64[] sourceArray = Src.datap as UInt64[];
            bool[] destArray = Dest.datap as bool[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                bool bdata = (bool)(data != 0 ? true : false);
                destArray[i + DestOffset] = bdata;
            }
            return true;
        }
        private static bool MemCpyFloatsToBools(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            float[] sourceArray = Src.datap as float[];
            bool[] destArray = Dest.datap as bool[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                bool bdata = (bool)(data != 0 ? true : false);
                destArray[i + DestOffset] = bdata;
            }
            return true;
        }
        private static bool MemCpyDoublesToBools(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            double[] sourceArray = Src.datap as double[];
            bool[] destArray = Dest.datap as bool[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                bool bdata = (bool)(data != 0 ? true : false);
                destArray[i + DestOffset] = bdata;
            }
            return true;
        }
        private static bool MemCpyDecimalsToBools(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            bool[] destArray = Dest.datap as bool[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                bool bdata = (bool)(data != 0 ? true : false);
                destArray[i + DestOffset] = bdata;
            }
            return true;
        }
  
        #endregion

        #region byte specific
        private static bool MemCpyBoolsToBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            bool[] sourceArray = Src.datap as bool[];
            sbyte[] destArray = Dest.datap as sbyte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                bool bdata = sourceArray[i + SrcOffset];
                sbyte data = (sbyte)(bdata == true ? 1 : 0);
                destArray[i + DestOffset] = data;
            }

            return true;
        }
        private static bool MemCpyBytesToBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            sbyte[] sourceArray = Src.datap as sbyte[];
            sbyte[] destArray = Dest.datap as sbyte[];

            Array.Copy(sourceArray, SrcOffset, destArray, DestOffset, totalBytesToCopy);
 
            return true;
        }
        private static bool MemCpyUBytesToBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            byte[] sourceArray = Src.datap as byte[];
            sbyte[] destArray = Dest.datap as sbyte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = (sbyte)MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }

            return true;
        }
        private static bool MemCpyInt16ToBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int16[] sourceArray = Src.datap as Int16[];
            sbyte[] destArray = Dest.datap as sbyte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = (sbyte)MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyUInt16ToBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt16[] sourceArray = Src.datap as UInt16[];
            sbyte[] destArray = Dest.datap as sbyte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = (sbyte)MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyInt32ToBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int32[] sourceArray = Src.datap as Int32[];
            sbyte[] destArray = Dest.datap as sbyte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = (sbyte)MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyUInt32ToBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt32[] sourceArray = Src.datap as UInt32[];
            sbyte[] destArray = Dest.datap as sbyte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = (sbyte)MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyInt64ToBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int64[] sourceArray = Src.datap as Int64[];
            sbyte[] destArray = Dest.datap as sbyte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = (sbyte)MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyUInt64ToBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt64[] sourceArray = Src.datap as UInt64[];
            sbyte[] destArray = Dest.datap as sbyte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = (sbyte)MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyFloatsToBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            float[] sourceArray = Src.datap as float[];
            sbyte[] destArray = Dest.datap as sbyte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = (sbyte)MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyDoublesToBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            double[] sourceArray = Src.datap as double[];
            sbyte[] destArray = Dest.datap as sbyte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = (sbyte)MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }

        private static bool MemCpyDecimalsToBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            sbyte[] destArray = Dest.datap as sbyte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = (sbyte)MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        #endregion

        #region ubyte specific
        private static bool MemCpyBoolsToUBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            bool[] sourceArray = Src.datap as bool[];
            byte[] destArray = Dest.datap as byte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                bool bdata = sourceArray[i + SrcOffset];
                byte data = (byte)(bdata == true ? 1 : 0);
                destArray[i + DestOffset] = data;
            }

            return true;
        }
        private static bool MemCpyBytesToUBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            sbyte[] sourceArray = Src.datap as sbyte[];
            byte[] destArray = Dest.datap as byte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = (byte)MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyUBytesToUBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            byte[] sourceArray = Src.datap as byte[];
            byte[] destArray = Dest.datap as byte[];

            Array.Copy(sourceArray, SrcOffset, destArray, DestOffset, totalBytesToCopy);

            return true;
        }
        private static bool MemCpyInt16ToUBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int16[] sourceArray = Src.datap as Int16[];
            byte[] destArray = Dest.datap as byte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyUInt16ToUBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt16[] sourceArray = Src.datap as UInt16[];
            byte[] destArray = Dest.datap as byte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyInt32ToUBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int32[] sourceArray = Src.datap as Int32[];
            byte[] destArray = Dest.datap as byte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyUInt32ToUBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt32[] sourceArray = Src.datap as UInt32[];
            byte[] destArray = Dest.datap as byte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyInt64ToUBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int64[] sourceArray = Src.datap as Int64[];
            byte[] destArray = Dest.datap as byte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyUInt64ToUBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt64[] sourceArray = Src.datap as UInt64[];
            byte[] destArray = Dest.datap as byte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyFloatsToUBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            float[] sourceArray = Src.datap as float[];
            byte[] destArray = Dest.datap as byte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyDoublesToUBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            double[] sourceArray = Src.datap as double[];
            byte[] destArray = Dest.datap as byte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        private static bool MemCpyDecimalsToUBytes(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            byte[] destArray = Dest.datap as byte[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = data;
            }
            return true;
        }
        #endregion

        #region Int16 specific
        private static bool MemCpyBoolsToInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            bool[] sourceArray = Src.datap as bool[];
            Int16[] destArray = Dest.datap as Int16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                bool bdata = sourceArray[i + SrcOffset];
                byte data = (byte)(bdata == true ? 1 : 0);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }

            return true;
        }

        private static bool MemCpyBytesToInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            sbyte[] sourceArray = Src.datap as sbyte[];
            Int16[] destArray = Dest.datap as Int16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, (byte)data);
            }

            return true;
        }
        private static bool MemCpyUBytesToInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            byte[] sourceArray = Src.datap as byte[];
            Int16[] destArray = Dest.datap as Int16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }

            return true;
        }
        private static bool MemCpyInt16ToInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int16[] sourceArray = Src.datap as Int16[];
            Int16[] destArray = Dest.datap as Int16[];

           //  try to use DestOffset & ItemDivInt16 for speed.  Validate in test app
            npy_intp DestOffsetAdjustment = DestOffset & ItemMaskInt16;
            npy_intp SrcOffsetAdjustment = SrcOffset & ItemMaskInt16;
 

            if (DestOffsetAdjustment == SrcOffsetAdjustment)
            {
                if ((DestOffsetAdjustment == 0) && ((totalBytesToCopy & ItemMaskInt16) == 0))
                {
                    Array.Copy(sourceArray, SrcOffset >> ItemDivInt16, destArray, DestOffset >> ItemDivInt16, totalBytesToCopy >> ItemDivInt16);
                }
                else
                {
                    CommonArrayCopy(destArray, DestOffset, sourceArray, SrcOffset, totalBytesToCopy, DestOffsetAdjustment, SrcOffsetAdjustment, ItemDivInt16, ItemMaskInt16);
                }
            }
            else
            {
                for (npy_intp i = 0; i < totalBytesToCopy; i++)
                {
                    byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                    MemoryAccess.SetByte(destArray, i + DestOffset, data);
                }
            }
 
            return true;
        }


        private static bool MemCpyUInt16ToInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt16[] sourceArray = Src.datap as UInt16[];
            Int16[] destArray = Dest.datap as Int16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt32ToInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int32[] sourceArray = Src.datap as Int32[];
            Int16[] destArray = Dest.datap as Int16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }


        private static bool MemCpyUInt32ToInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt32[] sourceArray = Src.datap as UInt32[];
            Int16[] destArray = Dest.datap as Int16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt64ToInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int64[] sourceArray = Src.datap as Int64[];
            Int16[] destArray = Dest.datap as Int16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt64ToInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt64[] sourceArray = Src.datap as UInt64[];
            Int16[] destArray = Dest.datap as Int16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyFloatsToInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            float[] sourceArray = Src.datap as float[];
            Int16[] destArray = Dest.datap as Int16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyDoublesToInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            double[] sourceArray = Src.datap as double[];
            Int16[] destArray = Dest.datap as Int16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }

        private static bool MemCpyDecimalsToInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            Int16[] destArray = Dest.datap as Int16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        #endregion

        #region UInt16 specific
        private static bool MemCpyBoolsToUInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            bool[] sourceArray = Src.datap as bool[];
            UInt16[] destArray = Dest.datap as UInt16[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                bool bdata = sourceArray[i + SrcOffset];
                byte data = (byte)(bdata == true ? 1 : 0);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyBytesToUInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            sbyte[] sourceArray = Src.datap as sbyte[];
            UInt16[] destArray = Dest.datap as UInt16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, (byte)data);
            }
            return true;
        }
        private static bool MemCpyUBytesToUInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            byte[] sourceArray = Src.datap as byte[];
            UInt16[] destArray = Dest.datap as UInt16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt16ToUInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int16[] sourceArray = Src.datap as Int16[];
            UInt16[] destArray = Dest.datap as UInt16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt16ToUInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt16[] sourceArray = Src.datap as UInt16[];
            UInt16[] destArray = Dest.datap as UInt16[];

            npy_intp DestOffsetAdjustment = DestOffset & ItemMaskUInt16;
            npy_intp SrcOffsetAdjustment = SrcOffset & ItemMaskUInt16;


            if (DestOffsetAdjustment == SrcOffsetAdjustment)
            {
                if ((DestOffsetAdjustment == 0) && ((totalBytesToCopy & ItemMaskUInt16) == 0))
                {
                    Array.Copy(sourceArray, SrcOffset >> ItemDivUInt16, destArray, DestOffset >> ItemDivUInt16, totalBytesToCopy >> ItemDivUInt16);
                }
                else
                {
                    CommonArrayCopy(destArray, DestOffset, sourceArray, SrcOffset, totalBytesToCopy, DestOffsetAdjustment, SrcOffsetAdjustment, ItemDivUInt16, ItemMaskUInt16);
                }
            }
            else
            {
                for (npy_intp i = 0; i < totalBytesToCopy; i++)
                {
                    byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                    MemoryAccess.SetByte(destArray, i + DestOffset, data);
                }
            }

            return true;
        }
        private static bool MemCpyInt32ToUInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int32[] sourceArray = Src.datap as Int32[];
            UInt16[] destArray = Dest.datap as UInt16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt32ToUInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt32[] sourceArray = Src.datap as UInt32[];
            UInt16[] destArray = Dest.datap as UInt16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt64ToUInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int64[] sourceArray = Src.datap as Int64[];
            UInt16[] destArray = Dest.datap as UInt16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt64ToUInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt64[] sourceArray = Src.datap as UInt64[];
            UInt16[] destArray = Dest.datap as UInt16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyFloatsToUInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            float[] sourceArray = Src.datap as float[];
            UInt16[] destArray = Dest.datap as UInt16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyDoublesToUInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            double[] sourceArray = Src.datap as double[];
            UInt16[] destArray = Dest.datap as UInt16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyDecimalsToUInt16s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            UInt16[] destArray = Dest.datap as UInt16[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        #endregion

        #region Int32 specific
        private static bool MemCpyBoolsToInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            bool[] sourceArray = Src.datap as bool[];
            Int32[] destArray = Dest.datap as Int32[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                bool bdata = sourceArray[i + SrcOffset];
                byte data = (byte)(bdata == true ? 1 : 0);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;

        }
        private static bool MemCpyBytesToInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            sbyte[] sourceArray = Src.datap as sbyte[];
            Int32[] destArray = Dest.datap as Int32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, (byte)data);
            }
            return true;
        }
        private static bool MemCpyUBytesToInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            byte[] sourceArray = Src.datap as byte[];
            Int32[] destArray = Dest.datap as Int32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt16ToInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int16[] sourceArray = Src.datap as Int16[];
            Int32[] destArray = Dest.datap as Int32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt16ToInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt16[] sourceArray = Src.datap as UInt16[];
            Int32[] destArray = Dest.datap as Int32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt32ToInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int32[] sourceArray = Src.datap as Int32[];
            Int32[] destArray = Dest.datap as Int32[];

            npy_intp DestOffsetAdjustment = DestOffset & ItemMaskInt32;
            npy_intp SrcOffsetAdjustment = SrcOffset & ItemMaskInt32;



            if (DestOffsetAdjustment == SrcOffsetAdjustment)
            {
                if ((DestOffsetAdjustment == 0) && ((totalBytesToCopy & ItemMaskInt32) == 0))
                {
                    Array.Copy(sourceArray, SrcOffset >> ItemDivInt32, destArray, DestOffset >> ItemDivInt32, totalBytesToCopy >> ItemDivInt32);
                }
                else
                {
                    CommonArrayCopy(destArray, DestOffset, sourceArray, SrcOffset, totalBytesToCopy, DestOffsetAdjustment, SrcOffsetAdjustment, ItemDivInt32, ItemMaskInt32);
                }
            }
            else
            {
                for (npy_intp i = 0; i < totalBytesToCopy; i++)
                {
                    byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                    MemoryAccess.SetByte(destArray, i + DestOffset, data);
                }
            }
            return true;
        }
        private static bool MemCpyUInt32ToInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt32[] sourceArray = Src.datap as UInt32[];
            Int32[] destArray = Dest.datap as Int32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt64ToInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int64[] sourceArray = Src.datap as Int64[];
            Int32[] destArray = Dest.datap as Int32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt64ToInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt64[] sourceArray = Src.datap as UInt64[];
            Int32[] destArray = Dest.datap as Int32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyFloatsToInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            float[] sourceArray = Src.datap as float[];
            Int32[] destArray = Dest.datap as Int32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyDoublesToInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            double[] sourceArray = Src.datap as double[];
            Int32[] destArray = Dest.datap as Int32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyDecimalsToInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            Int32[] destArray = Dest.datap as Int32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        #endregion

        #region UInt32 specific
        private static bool MemCpyBoolsToUInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            bool[] sourceArray = Src.datap as bool[];
            UInt32[] destArray = Dest.datap as UInt32[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                bool bdata = sourceArray[i + SrcOffset];
                byte data = (byte)(bdata == true ? 1 : 0);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyBytesToUInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            sbyte[] sourceArray = Src.datap as sbyte[];
            UInt32[] destArray = Dest.datap as UInt32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, (byte)data);
            }
            return true;
        }
        private static bool MemCpyUBytesToUInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            byte[] sourceArray = Src.datap as byte[];
            UInt32[] destArray = Dest.datap as UInt32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt16ToUInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int16[] sourceArray = Src.datap as Int16[];
            UInt32[] destArray = Dest.datap as UInt32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt16ToUInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt16[] sourceArray = Src.datap as UInt16[];
            UInt32[] destArray = Dest.datap as UInt32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt32ToUInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int32[] sourceArray = Src.datap as Int32[];
            UInt32[] destArray = Dest.datap as UInt32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt32ToUInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt32[] sourceArray = Src.datap as UInt32[];
            UInt32[] destArray = Dest.datap as UInt32[];

            npy_intp DestOffsetAdjustment = DestOffset & ItemMaskUInt32;
            npy_intp SrcOffsetAdjustment = SrcOffset & ItemMaskUInt32;


            if (DestOffsetAdjustment == SrcOffsetAdjustment)
            {
                if ((DestOffsetAdjustment == 0) && ((totalBytesToCopy & ItemMaskUInt32) == 0))
                {
                    Array.Copy(sourceArray, SrcOffset >> ItemDivUInt32, destArray, DestOffset >> ItemDivUInt32, totalBytesToCopy >> ItemDivUInt32);
                }
                else
                {
                    CommonArrayCopy(destArray, DestOffset, sourceArray, SrcOffset, totalBytesToCopy, DestOffsetAdjustment, SrcOffsetAdjustment, ItemDivUInt32, ItemMaskUInt32);
                }
            }
            else
            {
                for (npy_intp i = 0; i < totalBytesToCopy; i++)
                {
                    byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                    MemoryAccess.SetByte(destArray, i + DestOffset, data);
                }
            }
            return true;
        }
        private static bool MemCpyInt64ToUInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int64[] sourceArray = Src.datap as Int64[];
            UInt32[] destArray = Dest.datap as UInt32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt64ToUInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt64[] sourceArray = Src.datap as UInt64[];
            UInt32[] destArray = Dest.datap as UInt32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyFloatsToUInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            float[] sourceArray = Src.datap as float[];
            UInt32[] destArray = Dest.datap as UInt32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyDoublesToUInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            double[] sourceArray = Src.datap as double[];
            UInt32[] destArray = Dest.datap as UInt32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyDecimalsToUInt32s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            UInt32[] destArray = Dest.datap as UInt32[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        #endregion

        #region Int64 specific
        private static bool MemCpyBoolsToInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            bool[] sourceArray = Src.datap as bool[];
            Int64[] destArray = Dest.datap as Int64[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                bool bdata = sourceArray[i + SrcOffset];
                byte data = (byte)(bdata == true ? 1 : 0);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyBytesToInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            sbyte[] sourceArray = Src.datap as sbyte[];
            Int64[] destArray = Dest.datap as Int64[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, (byte)data);
            }
            return true;
        }
        private static bool MemCpyUBytesToInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            byte[] sourceArray = Src.datap as byte[];
            Int64[] destArray = Dest.datap as Int64[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt16ToInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int16[] sourceArray = Src.datap as Int16[];
            Int64[] destArray = Dest.datap as Int64[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt16ToInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt16[] sourceArray = Src.datap as UInt16[];
            Int64[] destArray = Dest.datap as Int64[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt32ToInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int32[] sourceArray = Src.datap as Int32[];
            Int64[] destArray = Dest.datap as Int64[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt32ToInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt32[] sourceArray = Src.datap as UInt32[];
            Int64[] destArray = Dest.datap as Int64[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt64ToInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int64[] sourceArray = Src.datap as Int64[];
            Int64[] destArray = Dest.datap as Int64[];

            npy_intp DestOffsetAdjustment = DestOffset & ItemMaskInt64;
            npy_intp SrcOffsetAdjustment = SrcOffset & ItemMaskInt64;


            if (DestOffsetAdjustment == SrcOffsetAdjustment)
            {
                if ((DestOffsetAdjustment == 0) && ((totalBytesToCopy & ItemMaskInt64) == 0))
                {
                    Array.Copy(sourceArray, SrcOffset >> ItemDivInt64, destArray, DestOffset >> ItemDivInt64, totalBytesToCopy >> ItemDivInt64);
                }
                else
                {
                    CommonArrayCopy(destArray, DestOffset, sourceArray, SrcOffset, totalBytesToCopy, DestOffsetAdjustment, SrcOffsetAdjustment, ItemDivInt64, ItemMaskInt64);
                }
            }
            else
            {
                for (npy_intp i = 0; i < totalBytesToCopy; i++)
                {
                    byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                    MemoryAccess.SetByte(destArray, i + DestOffset, data);
                }
            }
            return true;
        }
        private static bool MemCpyUInt64ToInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt64[] sourceArray = Src.datap as UInt64[];
            Int64[] destArray = Dest.datap as Int64[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyFloatsToInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            float[] sourceArray = Src.datap as float[];
            Int64[] destArray = Dest.datap as Int64[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyDoublesToInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            double[] sourceArray = Src.datap as double[];
            Int64[] destArray = Dest.datap as Int64[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyDecimalsToInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            Int64[] destArray = Dest.datap as Int64[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        #endregion

        #region UInt64 specific
        private static bool MemCpyBoolsToUInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            bool[] sourceArray = Src.datap as bool[];
            UInt64[] destArray = Dest.datap as UInt64[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                bool bdata = sourceArray[i + SrcOffset];
                byte data = (byte)(bdata == true ? 1 : 0);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyBytesToUInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            sbyte[] sourceArray = Src.datap as sbyte[];
            UInt64[] destArray = Dest.datap as UInt64[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, (byte)data);
            }
            return true;
        }
        private static bool MemCpyUBytesToUInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            byte[] sourceArray = Src.datap as byte[];
            UInt64[] destArray = Dest.datap as UInt64[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt16ToUInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int16[] sourceArray = Src.datap as Int16[];
            UInt64[] destArray = Dest.datap as UInt64[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt16ToUInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt16[] sourceArray = Src.datap as UInt16[];
            UInt64[] destArray = Dest.datap as UInt64[];

            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt32ToUInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int32[] sourceArray = Src.datap as Int32[];
            UInt64[] destArray = Dest.datap as UInt64[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt32ToUInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt32[] sourceArray = Src.datap as UInt32[];
            UInt64[] destArray = Dest.datap as UInt64[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt64ToUInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int64[] sourceArray = Src.datap as Int64[];
            UInt64[] destArray = Dest.datap as UInt64[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt64ToUInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt64[] sourceArray = Src.datap as UInt64[];
            UInt64[] destArray = Dest.datap as UInt64[];

            npy_intp DestOffsetAdjustment = DestOffset & ItemMaskUInt64;
            npy_intp SrcOffsetAdjustment = SrcOffset & ItemMaskUInt64;


            if (DestOffsetAdjustment == SrcOffsetAdjustment)
            {
                if ((DestOffsetAdjustment == 0) && ((totalBytesToCopy & ItemMaskUInt64) == 0))
                {
                    Array.Copy(sourceArray, SrcOffset >> ItemDivUInt64, destArray, DestOffset >> ItemDivUInt64, totalBytesToCopy >> ItemDivUInt64);
                }
                else
                {
                    CommonArrayCopy(destArray, DestOffset, sourceArray, SrcOffset, totalBytesToCopy, DestOffsetAdjustment, SrcOffsetAdjustment, ItemDivUInt64, ItemMaskUInt64);
                }
            }
            else
            {
                for (npy_intp i = 0; i < totalBytesToCopy; i++)
                {
                    byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                    MemoryAccess.SetByte(destArray, i + DestOffset, data);
                }
            }
    
            return true;
        }
        private static bool MemCpyFloatsToUInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            float[] sourceArray = Src.datap as float[];
            UInt64[] destArray = Dest.datap as UInt64[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyDoublesToUInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            double[] sourceArray = Src.datap as double[];
            UInt64[] destArray = Dest.datap as UInt64[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }

        private static bool MemCpyDecimalsToUInt64s(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            UInt64[] destArray = Dest.datap as UInt64[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        #endregion

        #region Float specific
        private static bool MemCpyBoolsToFloats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            bool[] sourceArray = Src.datap as bool[];
            float[] destArray = Dest.datap as float[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                bool bdata = sourceArray[i + SrcOffset];
                byte data = (byte)(bdata == true ? 1 : 0);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyBytesToFloats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            sbyte[] sourceArray = Src.datap as sbyte[];
            float[] destArray = Dest.datap as float[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, (byte)data);
            }
            return true;
        }
        private static bool MemCpyUBytesToFloats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            byte[] sourceArray = Src.datap as byte[];
            float[] destArray = Dest.datap as float[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt16ToFloats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int16[] sourceArray = Src.datap as Int16[];
            float[] destArray = Dest.datap as float[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt16ToFloats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt16[] sourceArray = Src.datap as UInt16[];
            float[] destArray = Dest.datap as float[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt32ToFloats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int32[] sourceArray = Src.datap as Int32[];
            float[] destArray = Dest.datap as float[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt32ToFloats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt32[] sourceArray = Src.datap as UInt32[];
            float[] destArray = Dest.datap as float[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt64ToFloats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int64[] sourceArray = Src.datap as Int64[];
            float[] destArray = Dest.datap as float[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt64ToFloats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt64[] sourceArray = Src.datap as UInt64[];
            float[] destArray = Dest.datap as float[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyFloatsToFloats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            float[] sourceArray = Src.datap as float[];
            float[] destArray = Dest.datap as float[];

            npy_intp DestOffsetAdjustment = DestOffset & ItemMaskFloat;
            npy_intp SrcOffsetAdjustment = SrcOffset & ItemMaskFloat;


            if (DestOffsetAdjustment == SrcOffsetAdjustment)
            {
                if ((DestOffsetAdjustment == 0) && ((totalBytesToCopy & ItemMaskFloat) == 0))
                {
                    Array.Copy(sourceArray, SrcOffset >> ItemDivFloat, destArray, DestOffset >> ItemDivFloat, totalBytesToCopy >> ItemDivFloat);
                }
                else
                {
                    CommonArrayCopy(destArray, DestOffset, sourceArray, SrcOffset, totalBytesToCopy, DestOffsetAdjustment, SrcOffsetAdjustment, ItemDivFloat, ItemMaskFloat);
                }
            }
            else
            {
                for (npy_intp i = 0; i < totalBytesToCopy; i++)
                {
                    byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                    MemoryAccess.SetByte(destArray, i + DestOffset, data);
                }
            }
            return true;
        }
        private static bool MemCpyDoublesToFloats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            double[] sourceArray = Src.datap as double[];
            float[] destArray = Dest.datap as float[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }

        private static bool MemCpyDecimalsToFloats(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            float[] destArray = Dest.datap as float[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        #endregion

        #region Double specific
        private static bool MemCpyBoolsToDoubles(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            bool[] sourceArray = Src.datap as bool[];
            double[] destArray = Dest.datap as double[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                bool bdata = sourceArray[i + SrcOffset];
                byte data = (byte)(bdata == true ? 1 : 0);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyBytesToDoubles(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            sbyte[] sourceArray = Src.datap as sbyte[];
            double[] destArray = Dest.datap as double[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, (byte)data);
            }
            return true;
        }
        private static bool MemCpyUBytesToDoubles(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            byte[] sourceArray = Src.datap as byte[];
            double[] destArray = Dest.datap as double[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt16ToDoubles(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int16[] sourceArray = Src.datap as Int16[];
            double[] destArray = Dest.datap as double[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt16ToDoubles(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt16[] sourceArray = Src.datap as UInt16[];
            double[] destArray = Dest.datap as double[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt32ToDoubles(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int32[] sourceArray = Src.datap as Int32[];
            double[] destArray = Dest.datap as double[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt32ToDoubles(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt32[] sourceArray = Src.datap as UInt32[];
            double[] destArray = Dest.datap as double[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt64ToDoubles(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int64[] sourceArray = Src.datap as Int64[];
            double[] destArray = Dest.datap as double[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt64ToDoubles(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt64[] sourceArray = Src.datap as UInt64[];
            double[] destArray = Dest.datap as double[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyFloatsToDoubles(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            float[] sourceArray = Src.datap as float[];
            double[] destArray = Dest.datap as double[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyDoublesToDoubles(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            double[] sourceArray = Src.datap as double[];
            double[] destArray = Dest.datap as double[];

            npy_intp DestOffsetAdjustment = DestOffset & ItemMaskDouble;
            npy_intp SrcOffsetAdjustment = SrcOffset & ItemMaskDouble;


            if (DestOffsetAdjustment == SrcOffsetAdjustment)
            {
                if ((DestOffsetAdjustment == 0) && ((totalBytesToCopy & ItemMaskDouble) == 0))
                {
                    Array.Copy(sourceArray, SrcOffset >> ItemDivDouble, destArray, DestOffset >> ItemDivDouble, totalBytesToCopy >> ItemDivDouble);
                }
                else
                {
                    CommonArrayCopy(destArray, DestOffset, sourceArray, SrcOffset, totalBytesToCopy, DestOffsetAdjustment, SrcOffsetAdjustment, ItemDivDouble, ItemMaskDouble);
                }
            }
            else
            {
                for (npy_intp i = 0; i < totalBytesToCopy; i++)
                {
                    byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                    MemoryAccess.SetByte(destArray, i + DestOffset, data);
                }
            }
            return true;
        }

        private static bool MemCpyDecimalsToDoubles(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            double[] destArray = Dest.datap as double[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        #endregion

        #region Decimal specific
        private static bool MemCpyBoolsToDecimals(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            bool[] sourceArray = Src.datap as bool[];
            decimal[] destArray = Dest.datap as decimal[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                bool bdata = sourceArray[i + SrcOffset];
                byte data = (byte)(bdata == true ? 1 : 0);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyBytesToDecimals(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            sbyte[] sourceArray = Src.datap as sbyte[];
            decimal[] destArray = Dest.datap as decimal[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, (byte)data);
            }
            return true;
        }
        private static bool MemCpyUBytesToDecimals(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            byte[] sourceArray = Src.datap as byte[];
            decimal[] destArray = Dest.datap as decimal[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = sourceArray[i + SrcOffset];
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt16ToDecimals(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int16[] sourceArray = Src.datap as Int16[];
            decimal[] destArray = Dest.datap as decimal[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt16ToDecimals(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt16[] sourceArray = Src.datap as UInt16[];
            decimal[] destArray = Dest.datap as decimal[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt32ToDecimals(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int32[] sourceArray = Src.datap as Int32[];
            decimal[] destArray = Dest.datap as decimal[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt32ToDecimals(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt32[] sourceArray = Src.datap as UInt32[];
            decimal[] destArray = Dest.datap as decimal[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyInt64ToDecimals(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int64[] sourceArray = Src.datap as Int64[];
            decimal[] destArray = Dest.datap as decimal[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyUInt64ToDecimals(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt64[] sourceArray = Src.datap as UInt64[];
            decimal[] destArray = Dest.datap as decimal[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyFloatsToDecimals(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            float[] sourceArray = Src.datap as float[];
            decimal[] destArray = Dest.datap as decimal[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyDoublesToDecimals(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            decimal[] destArray = Dest.datap as decimal[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                MemoryAccess.SetByte(destArray, i + DestOffset, data);
            }
            return true;
        }
        private static bool MemCpyDecimalsToDecimals(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            decimal[] destArray = Dest.datap as decimal[];

            npy_intp DestOffsetAdjustment = DestOffset & ItemMaskDecimal;
            npy_intp SrcOffsetAdjustment = SrcOffset & ItemMaskDecimal;


            if (DestOffsetAdjustment == SrcOffsetAdjustment)
            {
                if ((DestOffsetAdjustment == 0) && ((totalBytesToCopy & ItemMaskDecimal) == 0))
                {
                    Array.Copy(sourceArray, SrcOffset >> ItemDivDecimal, destArray, DestOffset >> ItemDivDecimal, totalBytesToCopy >> ItemDivDecimal);
                }
                else
                {
                    CommonArrayCopy(destArray, DestOffset, sourceArray, SrcOffset, totalBytesToCopy, DestOffsetAdjustment, SrcOffsetAdjustment, ItemDivDecimal, ItemMaskDecimal);
                }
            }
            else
            {
                for (npy_intp i = 0; i < totalBytesToCopy; i++)
                {
                    byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                    MemoryAccess.SetByte(destArray, i + DestOffset, data);
                }
            }
            return true;
        }
        #endregion

        #region Complex specific
        private static bool MemCpyBoolsToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            bool[] sourceArray = Src.datap as bool[];
            System.Numerics.Complex[] destArray = Dest.datap as System.Numerics.Complex[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                bool bdata = sourceArray[i + SrcOffset];
                byte data = (byte)(bdata == true ? 1 : 0);
                destArray[i + DestOffset] = new System.Numerics.Complex(data, 0);
            }
            return true;
        }
        private static bool MemCpyBytesToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            sbyte[] sourceArray = Src.datap as sbyte[];
            System.Numerics.Complex[] destArray = Dest.datap as System.Numerics.Complex[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                sbyte data = sourceArray[i + SrcOffset];
                destArray[i + DestOffset] = new System.Numerics.Complex(data, 0);
            }
            return true;
        }
        private static bool MemCpyUBytesToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            byte[] sourceArray = Src.datap as byte[];
            System.Numerics.Complex[] destArray = Dest.datap as System.Numerics.Complex[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = sourceArray[i + SrcOffset];
                destArray[i + DestOffset] = new System.Numerics.Complex(data, 0);
            }
            return true;
        }
        private static bool MemCpyInt16ToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int16[] sourceArray = Src.datap as Int16[];
            System.Numerics.Complex[] destArray = Dest.datap as System.Numerics.Complex[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = new System.Numerics.Complex(data, 0);
            }
            return true;
        }
        private static bool MemCpyUInt16ToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt16[] sourceArray = Src.datap as UInt16[];
            System.Numerics.Complex[] destArray = Dest.datap as System.Numerics.Complex[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = new System.Numerics.Complex(data, 0);
            }
            return true;
        }
        private static bool MemCpyInt32ToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int32[] sourceArray = Src.datap as Int32[];
            System.Numerics.Complex[] destArray = Dest.datap as System.Numerics.Complex[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = new System.Numerics.Complex(data, 0);
            }
            return true;
        }
        private static bool MemCpyUInt32ToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt32[] sourceArray = Src.datap as UInt32[];
            System.Numerics.Complex[] destArray = Dest.datap as System.Numerics.Complex[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = new System.Numerics.Complex(data, 0);
            }
            return true;
        }
        private static bool MemCpyInt64ToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            Int64[] sourceArray = Src.datap as Int64[];
            System.Numerics.Complex[] destArray = Dest.datap as System.Numerics.Complex[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = new System.Numerics.Complex(data, 0);
            }
            return true;
        }
        private static bool MemCpyUInt64ToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            UInt64[] sourceArray = Src.datap as UInt64[];
            System.Numerics.Complex[] destArray = Dest.datap as System.Numerics.Complex[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = new System.Numerics.Complex(data, 0);
            }
            return true;
        }
        private static bool MemCpyFloatsToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            float[] sourceArray = Src.datap as float[];
            System.Numerics.Complex[] destArray = Dest.datap as System.Numerics.Complex[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = new System.Numerics.Complex(data, 0);
            }
            return true;
        }
        private static bool MemCpyDoublesToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            double[] sourceArray = Src.datap as double[];
            System.Numerics.Complex[] destArray = Dest.datap as System.Numerics.Complex[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = new System.Numerics.Complex(data, 0);
            }
            return true;
        }

        private static bool MemCpyDecimalsToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            decimal[] sourceArray = Src.datap as decimal[];
            System.Numerics.Complex[] destArray = Dest.datap as System.Numerics.Complex[];


            for (npy_intp i = 0; i < totalBytesToCopy; i++)
            {
                byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                destArray[i + DestOffset] = new System.Numerics.Complex(data, 0);
            }
            return true;
        }
        private static bool MemCpyComplexToComplex(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            System.Numerics.Complex[] sourceArray = Src.datap as System.Numerics.Complex[];
            System.Numerics.Complex[] destArray = Dest.datap as System.Numerics.Complex[];

            if (__ComplexSize < 0)
            {
                __ComplexSize = DefaultArrayHandlers.GetArrayHandler(NPY_TYPES.NPY_COMPLEX).ItemSize;
                ItemDivComplex = numpyinternal.GetDivSize(__ComplexSize);
                ItemMaskComplex = numpyinternal.GetMaskSize(__ComplexSize);
            }


            npy_intp DestOffsetAdjustment = DestOffset & ItemMaskComplex;
            npy_intp SrcOffsetAdjustment = SrcOffset & ItemMaskComplex;


            if (DestOffsetAdjustment == SrcOffsetAdjustment)
            {
                if ((DestOffsetAdjustment == 0) && ((totalBytesToCopy & ItemMaskComplex) == 0))
                {
                    Array.Copy(sourceArray, SrcOffset >> ItemDivComplex, destArray, DestOffset >> ItemDivComplex, totalBytesToCopy >> ItemDivComplex);
                }
                else
                {
                    CommonArrayCopy(destArray, DestOffset, sourceArray, SrcOffset, totalBytesToCopy, DestOffsetAdjustment, SrcOffsetAdjustment, ItemDivComplex, ItemMaskComplex);
                }
            }
            else
            {
                //for (npy_intp i = 0; i < totalBytesToCopy; i++)
                //{
                //    byte data = MemoryAccess.GetByte(sourceArray, i + SrcOffset);
                //    MemoryAccess.SetByte(destArray, i + DestOffset, data);
                //}
            }
            return true;
        }

        #endregion

        #region BigInt specific
        private static bool MemCpyBoolsToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            return true;
        }
        private static bool MemCpyBytesToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            return true;
        }
        private static bool MemCpyUBytesToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            return true;
        }
        private static bool MemCpyInt16ToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            return true;
        }
        private static bool MemCpyUInt16ToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            return true;
        }
        private static bool MemCpyInt32ToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            return true;
        }
        private static bool MemCpyUInt32ToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            return true;
        }
        private static bool MemCpyInt64ToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            return true;
        }
        private static bool MemCpyUInt64ToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            return true;
        }
        private static bool MemCpyFloatsToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            return true;
        }
        private static bool MemCpyDoublesToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            return true;
        }

        private static bool MemCpyDecimalsToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            return true;
        }

        private static bool MemCpyComplexToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
             return true;
        }
        private static bool MemCpyBigIntToBigInt(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            System.Numerics.BigInteger[] sourceArray = Src.datap as System.Numerics.BigInteger[];
            System.Numerics.BigInteger[] destArray = Dest.datap as System.Numerics.BigInteger[];

            if (__BigIntSize < 0)
            {
                __BigIntSize = DefaultArrayHandlers.GetArrayHandler(NPY_TYPES.NPY_BIGINT).ItemSize;
                ItemDivBigInt = numpyinternal.GetDivSize(__BigIntSize);
                ItemMaskBigInt = numpyinternal.GetMaskSize(__BigIntSize);
            }

            npy_intp DestOffsetAdjustment = DestOffset & ItemMaskBigInt;
            npy_intp SrcOffsetAdjustment = SrcOffset & ItemMaskBigInt;

            if ((DestOffsetAdjustment == 0) && ((totalBytesToCopy & ItemMaskBigInt) == 0))
            {
                Array.Copy(sourceArray, SrcOffset >> ItemDivBigInt, destArray, DestOffset >> ItemDivBigInt, totalBytesToCopy >> ItemDivBigInt);
            }
            else
            {
                CommonArrayCopy(destArray, DestOffset, sourceArray, SrcOffset, totalBytesToCopy, DestOffsetAdjustment, SrcOffsetAdjustment, ItemDivBigInt, ItemMaskBigInt);
            }

            return true;
        }

        #endregion

        #region Object specific

        private static bool MemCpyObjectToObject(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            object[] sourceArray = Src.datap as object[];

            if (sourceArray == null)
            {
                throw new Exception(string.Format("Unable to copy array of this type to object array"));
            }

            object[] destArray = Dest.datap as object[];

            if (__ObjectSize < 0)
            {
                __ObjectSize = DefaultArrayHandlers.GetArrayHandler(NPY_TYPES.NPY_OBJECT).ItemSize;
                ItemDivObject = numpyinternal.GetDivSize(__ObjectSize);
                ItemMaskObject = numpyinternal.GetMaskSize(__ObjectSize);
            }

            npy_intp DestOffsetAdjustment = DestOffset & ItemMaskObject;
            npy_intp SrcOffsetAdjustment = SrcOffset & ItemMaskObject;

            if ((DestOffsetAdjustment == 0) && ((totalBytesToCopy & ItemMaskObject) == 0))
            {
                Array.Copy(sourceArray, SrcOffset >> ItemDivObject, destArray, DestOffset >> ItemDivObject, totalBytesToCopy >> ItemDivObject);
            }
            else
            {
                CommonArrayCopy(destArray, DestOffset, sourceArray, SrcOffset, totalBytesToCopy, DestOffsetAdjustment, SrcOffsetAdjustment, ItemDivObject, ItemMaskObject);
            }

            return true;
        }

        #endregion

        #region String specific

       private static bool MemCpyStringToString(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long totalBytesToCopy)
        {
            string[] sourceArray = Src.datap as string[];

            if (sourceArray == null)
            {
                throw new Exception(string.Format("Unable to copy array of this type to string array"));
            }

            string[] destArray = Dest.datap as string[];

            if (__StringSize < 0)
            {
                __StringSize = DefaultArrayHandlers.GetArrayHandler(NPY_TYPES.NPY_STRING).ItemSize;
                ItemDivString = numpyinternal.GetDivSize(__StringSize);
                ItemMaskString = numpyinternal.GetMaskSize(__StringSize);
            }
            int ItemSize = __StringSize;

            npy_intp DestOffsetAdjustment = DestOffset & ItemMaskString;
            npy_intp SrcOffsetAdjustment = SrcOffset & ItemMaskString;

            if ((DestOffsetAdjustment == 0) && ((totalBytesToCopy & ItemMaskString) == 0))
            {
                Array.Copy(sourceArray, SrcOffset >> ItemDivString, destArray, DestOffset >> ItemDivString, totalBytesToCopy >> ItemDivString);
            }
            else
            {
                CommonArrayCopy(destArray, DestOffset, sourceArray, SrcOffset, totalBytesToCopy, DestOffsetAdjustment, SrcOffsetAdjustment, ItemDivString, ItemMaskString);
            }

            return true;
        }

        #endregion

        internal static ICopyHelper GetMemcopyHelper(VoidPtr dst)
        {
            switch (dst.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                {
                    return new BoolCopyHelper();
                }
                case NPY_TYPES.NPY_BYTE:
                {
                    return new SByteCopyHelper();
                }
                case NPY_TYPES.NPY_UBYTE:
                {
                    return new UByteCopyHelper();
                }
                case NPY_TYPES.NPY_INT16:
                {
                    return new Int16CopyHelper();
                }
                case NPY_TYPES.NPY_UINT16:
                {
                    return new UInt16CopyHelper();
                }
                case NPY_TYPES.NPY_INT32:
                {
                    return new Int32CopyHelper();
                }
                case NPY_TYPES.NPY_UINT32:
                {
                    return new UInt32CopyHelper();
                }
                case NPY_TYPES.NPY_INT64:
                {
                    return new Int64CopyHelper();
                }
                case NPY_TYPES.NPY_UINT64:
                {
                    return new UInt64CopyHelper();
                }
                case NPY_TYPES.NPY_FLOAT:
                {
                    return new FloatCopyHelper();
                }
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_COMPLEXREAL:
                case NPY_TYPES.NPY_COMPLEXIMAG:
                {
                    return new DoubleCopyHelper();
                }
                case NPY_TYPES.NPY_DECIMAL:
                {
                    return new DecimalCopyHelper();
                }
                case NPY_TYPES.NPY_COMPLEX:
                {
                    return new ComplexCopyHelper();
                }
                case NPY_TYPES.NPY_BIGINT:
                {
                    return new BigIntCopyHelper();
                }
                case NPY_TYPES.NPY_OBJECT:
                {
                    return new ObjectCopyHelper();
                }
                case NPY_TYPES.NPY_STRING:
                {
                    return new StringCopyHelper();
                }
                default:
                    throw new NotImplementedException("The specified data type is not handled");
            }
        }

    }

    class MemSet
    {

        public static void memset(VoidPtr Dest, npy_intp DestOffset, byte setValue, long Length)
        {
            switch (Dest.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    MemSetToBools(Dest, DestOffset, setValue, Length);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    MemSetToBytes(Dest, DestOffset, setValue, Length);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    MemSetToUBytes(Dest, DestOffset, setValue, Length);
                    break;
                case NPY_TYPES.NPY_INT16:
                    MemSetToInt16s(Dest, DestOffset, setValue, Length);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    MemSetToUInt16s(Dest, DestOffset, setValue, Length);
                    break;
                case NPY_TYPES.NPY_INT32:
                    MemSetToInt32s(Dest, DestOffset, setValue, Length);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    MemSetToUInt32s(Dest, DestOffset, setValue, Length);
                    break;
                case NPY_TYPES.NPY_INT64:
                    MemSetToInt64s(Dest, DestOffset, setValue, Length);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    MemSetToUInt64s(Dest, DestOffset, setValue, Length);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    MemSetToFloats(Dest, DestOffset, setValue, Length);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    MemSetToDoubles(Dest, DestOffset, setValue, Length);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    MemSetToDecimals(Dest, DestOffset, setValue, Length);
                    break;

            }
            return;
        }
        private static void MemSetToBools(VoidPtr dest, npy_intp destOffset, byte setValue, long length)
        {
            bool[] destArray = dest.datap as bool[];

            for (npy_intp i = 0; i < length; i++)
            {
                destArray[i + destOffset] = setValue != 0 ? true : false;
            }
        }

        private static void MemSetToBytes(VoidPtr dest, npy_intp destOffset, byte setValue, long length)
        {
            sbyte[] destArray = dest.datap as sbyte[];

            for (npy_intp i = 0; i < length; i++)
            {
                destArray[i + destOffset] = (sbyte)setValue;
            }
        }
        private static void MemSetToUBytes(VoidPtr dest, npy_intp destOffset, byte setValue, long length)
        {
            byte[] destArray = dest.datap as byte[];

            for (npy_intp i = 0; i < length; i++)
            {
                destArray[i + destOffset] = setValue;
            }
        }
        private static void MemSetToInt16s(VoidPtr dest, npy_intp destOffset, byte setValue, long length)
        {
            Int16[] destArray = dest.datap as Int16[];

            for (npy_intp i = 0; i < length; i++)
            {
                MemoryAccess.SetByte(destArray, i + destOffset, setValue);
            }
        }
        private static void MemSetToUInt16s(VoidPtr dest, npy_intp destOffset, byte setValue, long length)
        {
            UInt16[] destArray = dest.datap as UInt16[];

            for (npy_intp i = 0; i < length; i++)
            {
                MemoryAccess.SetByte(destArray, i + destOffset, setValue);
            }
        }

        private static void MemSetToInt32s(VoidPtr dest, npy_intp destOffset, byte setValue, long length)
        {
            Int32[] destArray = dest.datap as Int32[];

            for (npy_intp i = 0; i < length; i++)
            {
                MemoryAccess.SetByte(destArray, i + destOffset, setValue);
            }
        }
        private static void MemSetToUInt32s(VoidPtr dest, npy_intp destOffset, byte setValue, long length)
        {
            UInt32[] destArray = dest.datap as UInt32[];

            for (npy_intp i = 0; i < length; i++)
            {
                MemoryAccess.SetByte(destArray, i + destOffset, setValue);
            }
        }
        private static void MemSetToInt64s(VoidPtr dest, npy_intp destOffset, byte setValue, long length)
        {
            Int64[] destArray = dest.datap as Int64[];

            for (npy_intp i = 0; i < length; i++)
            {
                MemoryAccess.SetByte(destArray, i + destOffset, setValue);
            }
        }
        private static void MemSetToUInt64s(VoidPtr dest, npy_intp destOffset, byte setValue, long length)
        {
            UInt64[] destArray = dest.datap as UInt64[];

            for (npy_intp i = 0; i < length; i++)
            {
                MemoryAccess.SetByte(destArray, i + destOffset, setValue);
            }
        }
        private static void MemSetToFloats(VoidPtr dest, npy_intp destOffset, byte setValue, long length)
        {
            float[] destArray = dest.datap as float[];

            for (npy_intp i = 0; i < length; i++)
            {
                MemoryAccess.SetByte(destArray, i + destOffset, setValue);
            }
        }
        private static void MemSetToDoubles(VoidPtr dest, npy_intp destOffset, byte setValue, long length)
        {
            double[] destArray = dest.datap as double[];

            for (npy_intp i = 0; i < length; i++)
            {
                MemoryAccess.SetByte(destArray, i + destOffset, setValue);
            }
        }

        private static void MemSetToDecimals(VoidPtr dest, npy_intp destOffset, byte setValue, long length)
        {
            decimal[] destArray = dest.datap as decimal[];

            for (npy_intp i = 0; i < length; i++)
            {
                MemoryAccess.SetByte(destArray, i + destOffset, setValue);
            }
        }

    }


    class BoolCopyHelper : CopyHelper<bool> , ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(bool);
        }

        protected override bool T_dot(bool tmp, bool[] ip1, bool[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            if ((ip1[ip1_index] == true) && (ip2[ip2_index] == true))
            {
                tmp = true;
                return tmp;
            }
            return tmp;
        }

    }

    class UByteCopyHelper : CopyHelper<byte>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(byte);
        }

        protected override byte T_dot(byte tmp, byte[] ip1, byte[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            tmp += (byte)(ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }

    }

    class SByteCopyHelper : CopyHelper<sbyte>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(sbyte);
        }


        protected override sbyte T_dot(sbyte tmp, sbyte[] ip1, sbyte[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            tmp += (sbyte)(ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }

    }

    class Int16CopyHelper : CopyHelper<Int16>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(Int16);
        }

        protected override Int16 T_dot(Int16 tmp, Int16[] ip1, Int16[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            tmp += (Int16)(ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }

    }


    class UInt16CopyHelper : CopyHelper<UInt16>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(UInt16);
        }


        protected override UInt16 T_dot(UInt16 tmp, UInt16[] ip1, UInt16[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            tmp += (UInt16)(ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }

    }

    class Int32CopyHelper : CopyHelper<Int32>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(Int32);
        }


        protected override Int32 T_dot(Int32 tmp, Int32[] ip1, Int32[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            tmp += (Int32)(ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }

    }


    class UInt32CopyHelper : CopyHelper<UInt32>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(UInt32);
        }


        protected override UInt32 T_dot(UInt32 tmp, UInt32[] ip1, UInt32[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            tmp += (UInt32)(ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }


    }

    class Int64CopyHelper : CopyHelper<Int64>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(Int64);
        }

        protected override Int64 T_dot(Int64 tmp, Int64[] ip1, Int64[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            tmp += (Int64)(ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }

    }


    class UInt64CopyHelper : CopyHelper<UInt64>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(UInt64);
        }

        protected override UInt64 T_dot(UInt64 tmp, UInt64[] ip1, UInt64[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            tmp += (UInt64)(ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }

    }



    class FloatCopyHelper : CopyHelper<float>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(float);
        }


        protected override float T_dot(float tmp, float[] ip1, float[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            tmp += (float)(ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }

    }


    class DoubleCopyHelper : CopyHelper<double>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(double);
        }


        protected override double T_dot(double tmp, double[] ip1, double[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            tmp += (double)(ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }

    }

    class DecimalCopyHelper : CopyHelper<decimal>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(decimal);
        }

        protected override decimal T_dot(decimal tmp, decimal[] ip1, decimal[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            tmp += (decimal)(ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }

    }

    class ComplexCopyHelper : CopyHelper<System.Numerics.Complex>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(double) * 2;
        }


        protected override System.Numerics.Complex T_dot(System.Numerics.Complex tmp, System.Numerics.Complex[] ip1, System.Numerics.Complex[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            tmp += (System.Numerics.Complex)(ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }

    }

    class BigIntCopyHelper : CopyHelper<System.Numerics.BigInteger>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return sizeof(double) * 4;
        }


        protected override System.Numerics.BigInteger T_dot(System.Numerics.BigInteger tmp, System.Numerics.BigInteger[] ip1, System.Numerics.BigInteger[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            tmp += (System.Numerics.BigInteger)(ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }


    }


    class ObjectCopyHelper : CopyHelper<System.Object>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return IntPtr.Size; 
        }


        protected override object T_dot(System.Object otmp, System.Object[] op1, System.Object[] op2, npy_intp ip1_index, npy_intp ip2_index)
        {
            dynamic tmp = (dynamic)otmp;
            if (tmp == null) tmp = 0;

            dynamic[] ip1 = op1 as dynamic[];
            dynamic[] ip2 = op2 as dynamic[];

            tmp += (ip1[ip1_index] * ip2[ip2_index]);
            return tmp;
        }

    }

    class StringCopyHelper : CopyHelper<System.String>, ICopyHelper
    {
        public override int GetTypeSize(VoidPtr vp)
        {
            return IntPtr.Size;
        }

        protected override System.String T_dot(System.String tmp, System.String[] ip1, System.String[] ip2, npy_intp ip1_index, npy_intp ip2_index)
        {
            if ((ip1[ip1_index] != null) && (ip2[ip2_index] != null))
            {
                tmp = (ip1[ip1_index] + ip2[ip2_index]);
                return tmp;
            }
            return tmp;
        }

    }


    internal interface ICopyHelper
    {
        bool IsSingleElementCopy();
        void copyswap(VoidPtr _dst, VoidPtr _src, bool swap);
        void default_copyswap(VoidPtr _dst, npy_intp dstride, VoidPtr _src, npy_intp sstride, npy_intp n, bool swap);
        void memclr(VoidPtr dest, npy_intp dest_offset, npy_intp len);
        void memmove_init(VoidPtr dest, VoidPtr src);
        void memmove(npy_intp dest_offset, npy_intp src_offset, npy_intp len);
        void memmoveitem(npy_intp dest_offset, npy_intp src_offset);
        void memcpy(npy_intp dest_offset, npy_intp src_offset, npy_intp len);
        void IterSubscriptSlice(npy_intp[] steps, NpyArrayIterObject srcIter, VoidPtr _dst, npy_intp start, npy_intp step_size, bool swap);
        void IterSubscriptBoolArray(NpyArrayIterObject srcIter, VoidPtr _dst, bool[] bool_array, npy_intp stride, npy_intp bool_array_size, bool swap);
        npy_intp? IterSubscriptIntpArray(NpyArrayIterObject srcIter, NpyArrayIterObject index_iter, VoidPtr _dst, bool swap);
        void IterSubscriptAssignSlice(NpyArrayIterObject destIter, NpyArrayIterObject srcIter, npy_intp steps, npy_intp start, npy_intp step_size, bool swap);
        void IterSubscriptAssignBoolArray(NpyArrayIterObject self, NpyArrayIterObject value_iter, npy_intp bool_size, bool[] dptr, npy_intp stride, bool swap);
        npy_intp? IterSubscriptAssignIntpArray(NpyArrayIterObject destIter, NpyArrayIterObject indexIter, NpyArrayIterObject srcIter, bool swap);
        void GetMap(NpyArrayIterObject it, NpyArrayMapIterObject mit, bool swap);
        void SetMap(NpyArrayMapIterObject destIter, NpyArrayIterObject srcIter, bool swap);
        void FillWithScalar(VoidPtr destPtr, VoidPtr srcPtr, npy_intp size, bool swap);
        void FillWithScalarIter(NpyArrayIterObject destIter, VoidPtr srcPtr, npy_intp size, bool swap);
        void MatrixProduct(NpyArrayIterObject it1, NpyArrayIterObject it2, VoidPtr op, npy_intp is1, npy_intp is2, npy_intp os, npy_intp l);
        void InnerProduct(NpyArrayIterObject it1, NpyArrayIterObject it2, VoidPtr op, npy_intp is1, npy_intp is2, npy_intp os, npy_intp l);
        void correlate(VoidPtr ip1, VoidPtr ip2, VoidPtr op, npy_intp is1, npy_intp is2, npy_intp os, npy_intp n, npy_intp n1, npy_intp n2, npy_intp n_left, npy_intp n_right);
        void strided_byte_copy_init(VoidPtr dst, npy_intp outstrides, VoidPtr src, npy_intp intstrides, int elsize, int eldiv);
        void strided_byte_copy(npy_intp dest_offset, npy_intp src_offset, npy_intp N);
        void strided_single_element_copy(npy_intp dest_offset, npy_intp src_offset, npy_intp N);
        //void flat_copyinto(VoidPtr dest, int outstride, NpyArrayIterObject srcIter, npy_intp instride, npy_intp N, npy_intp destOffset);
    }

    abstract class CopyHelper<T>
    {
        public abstract int GetTypeSize(VoidPtr vp);
        public int GetDivSize(VoidPtr vp)
        {
            return GetDivSize(GetTypeSize(vp));
        }
        public int GetDivSize(int elsize)
        {
            return numpyinternal.GetDivSize(elsize);
        }
        public bool IsSingleElementCopy()
        {
            return isSingleElementCopy;
        }


        VoidPtr dst;
        VoidPtr src;
        T[] da;
        T[] sa;
        T[] Temp = null;
        npy_intp outstrides;
        npy_intp instrides;
        int elsize;
        int eldiv;
        bool isSameType = false;
        bool isSingleElementCopy = false;
        bool useArrayCopy = false;

        public void memclr(VoidPtr dest, npy_intp dest_offset, npy_intp len)
        {
            var da = dest.datap as T[];
            Array.Clear(da, (int)(dest_offset >> GetDivSize(dst)), (int)len);
        }


        public void strided_byte_copy_init(VoidPtr dst, npy_intp outstrides, VoidPtr src, npy_intp instrides, int elsize, int eldiv)
        {
            if (dst.type_num == src.type_num)
            {
                da = dst.datap as T[];
                sa = src.datap as T[];
                this.eldiv = eldiv;

                this.outstrides = outstrides >> eldiv;
                this.instrides = instrides >> eldiv;
                if (this.instrides == 0)
                {
                    isSingleElementCopy = true;
                }

                isSameType = true;
                if (this.instrides == 1 && this.outstrides == 1)
                {
                    useArrayCopy = true;
                }
            }
            else
            {
                this.dst = dst;
                this.src = src;
                this.elsize = elsize;
                this.outstrides = outstrides;
                this.instrides = instrides;
            }
        }

        public void strided_byte_copy(npy_intp dest_offset, npy_intp src_offset, npy_intp N)
        {

            if (isSameType)
            {
                npy_intp tout_index = dest_offset >> eldiv;
                npy_intp tin_index = src_offset >> eldiv;

                if (useArrayCopy)
                {
                    Array.Copy(sa, tin_index, da, tout_index, N);
                }
                else
                {
                    if (instrides == 0)
                    {
                        T saValue = sa[tin_index];
                        for (int i = 0; i < N; i++)
                        {
                            da[tout_index] = saValue;
                            tout_index += outstrides;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < N; i++)
                        {
                            da[tout_index] = sa[tin_index];
                            tin_index += instrides;
                            tout_index += outstrides;
                        }
                    }
         
                }
            }
            else
            {
                npy_intp tin_index = 0;
                npy_intp tout_index = 0;

                for (int i = 0; i < N; i++)
                {
                    VoidPtr Temp = new VoidPtr(new byte[elsize]);
                    MemCopy.MemCpy(Temp, 0, src, tin_index, elsize);
                    MemCopy.MemCpy(dst, tout_index, Temp, 0, elsize);

                    tin_index += instrides;
                    tout_index += outstrides;
                }
            }
        }

        public void strided_single_element_copy(npy_intp dest_offset, npy_intp src_offset, npy_intp N)
        {
            npy_intp tout_index = dest_offset >> eldiv;
            npy_intp tin_index = src_offset >> eldiv;

            
            T saValue = sa[tin_index];
            for (int i = 0; i < N; i++)
            {
                da[tout_index] = saValue;
                tout_index += outstrides;
            }

        }

        public void IterSubscriptSlice(npy_intp[] steps, NpyArrayIterObject srcIter, VoidPtr _dst,
         npy_intp start, npy_intp step_size, bool swap)
        {

            int elsize = GetTypeSize(_dst);
            int divsize = GetDivSize(elsize);
            _dst.data_offset >>= divsize;

            srcIter = numpyinternal.NpyArray_ITER_ConvertToIndex(srcIter, divsize);

            T[] d = _dst.datap as T[];
            T[] s = srcIter.dataptr.datap as T[];

            npy_intp stepper = steps[0];

            if (swap)
            {
                while (stepper-- > 0)
                {
                    numpyinternal.NpyArray_ITER_GOTO1D_INDEX(srcIter, start);

                    d[_dst.data_offset] = s[srcIter.dataptr.data_offset];
                    numpyinternal.swapvalue(_dst, _dst.data_offset, 0);
                    _dst.data_offset += 1;

                    start += step_size;
                }
            }
            else
            {
                while (stepper-- > 0)
                {
                    numpyinternal.NpyArray_ITER_GOTO1D_INDEX(srcIter, start);

                    d[_dst.data_offset++] = s[srcIter.dataptr.data_offset];

                    start += step_size;
                }
            }

            steps[0] = stepper;
        }

        //public void IterSubscriptSlice_USECACHEMODEL(npy_intp[] steps, NpyArrayIterObject srcIter, VoidPtr _dst,
        //     npy_intp start, npy_intp step_size, bool swap)
        //{

        //    int elsize = GetTypeSize(_dst);
        //    int divsize = GetDivSize(elsize);


        //    T[] d = _dst.datap as T[];
        //    T[] s = srcIter.dataptr.datap as T[];

        //    npy_intp stepper = steps[0];

        //    if (swap)
        //    {
        //        while (stepper > 0)
        //        {
        //            numpyinternal.NpyArray_ITER_GOTO1D_CACHE(srcIter, start, step_size, stepper);

        //            for (int i = 0; i < srcIter.internalCacheLength; i++)
        //            {
        //                d[_dst.data_offset >> divsize] = s[srcIter.internalCache[i] >> divsize];
        //                numpyinternal.swapvalue(_dst, _dst.data_offset, divsize);
        //                _dst.data_offset += elsize;
        //            }
        //            stepper -= srcIter.internalCacheLength;
        //            start += step_size * srcIter.internalCacheLength;
        //        }
        //    }
        //    else
        //    {
        //        _dst.data_offset >>= divsize;

        //        while (stepper > 0)
        //        {
        //            numpyinternal.NpyArray_ITER_GOTO1D_CACHE(srcIter, start, step_size, stepper);
        //            for (int i = 0; i < srcIter.internalCacheLength; i++)
        //            {
        //                d[_dst.data_offset++] = s[srcIter.internalCache[i] >> divsize];
        //            }

        //            stepper -= srcIter.internalCacheLength;
        //            start += step_size * srcIter.internalCacheLength;
        //        }
        //    }



        //    steps[0] = stepper;
        //}

        public void IterSubscriptBoolArray(NpyArrayIterObject srcIter, VoidPtr _dst, bool[] bool_array, npy_intp stride, npy_intp bool_array_size, bool swap)
        {
            int elsize = GetTypeSize(_dst);
            int divsize = GetDivSize(elsize);
            _dst.data_offset >>= divsize;

            T[] d = _dst.datap as T[];
            T[] s = srcIter.dataptr.datap as T[];

            srcIter = numpyinternal.NpyArray_ITER_ConvertToIndex(srcIter, divsize);

            npy_intp dptr_index = 0;

            if (swap)
            {
                while (bool_array_size-- > 0)
                {
                    if (bool_array[dptr_index])
                    {
                        d[_dst.data_offset] = s[srcIter.dataptr.data_offset];
                        numpyinternal.swapvalue(_dst, _dst.data_offset, 0);
                        _dst.data_offset += 1;
                    }
                    dptr_index += stride;
                    numpyinternal.NpyArray_ITER_NEXT(srcIter);
                }
            }
            else
            {
                while (bool_array_size-- > 0)
                {
                    if (bool_array[dptr_index])
                    {
                        d[_dst.data_offset++] = s[srcIter.dataptr.data_offset];
                    }
                    dptr_index += stride;
                    numpyinternal.NpyArray_ITER_NEXT(srcIter);
                }
            }
     
        }

        public npy_intp? IterSubscriptIntpArray(NpyArrayIterObject srcIter, NpyArrayIterObject index_iter, VoidPtr _dst, bool swap)
        {
            npy_intp[] dataptr = index_iter.dataptr.datap as npy_intp[];
            var elsize = GetTypeSize(_dst);
            var divsize = GetDivSize(elsize);
            var divintp = GetDivSize(sizeof(npy_intp));
            var iterCount = index_iter.size;

            T[] d = _dst.datap as T[];
            T[] s = srcIter.dataptr.datap as T[];


            index_iter = numpyinternal.NpyArray_ITER_ConvertToIndex(index_iter, divintp);
            srcIter = numpyinternal.NpyArray_ITER_ConvertToIndex(srcIter, divsize);

            if (swap)
            {
                while (iterCount-- > 0)
                {
                    npy_intp num = dataptr[index_iter.dataptr.data_offset];
                    if (num < 0)
                    {
                        num += srcIter.size;
                    }
                    if (num < 0 || num >= srcIter.size)
                    {
                        return num;
                    }
                    numpyinternal.NpyArray_ITER_GOTO1D_INDEX(srcIter, num);

                    d[_dst.data_offset++] = s[srcIter.dataptr.data_offset];
                    _dst.data_offset += elsize;

                    numpyinternal.swapvalue(_dst, _dst.data_offset, 0);

                    numpyinternal.NpyArray_ITER_NEXT(index_iter);
                }
            }
            else
            {
                _dst.data_offset >>= divsize;

                while (iterCount-- > 0)
                {
                    npy_intp num = dataptr[index_iter.dataptr.data_offset];
                    if (num < 0)
                    {
                        num += srcIter.size;
                    }
                    if (num < 0 || num >= srcIter.size)
                    {
                        return num;
                    }
                    numpyinternal.NpyArray_ITER_GOTO1D_INDEX(srcIter, num);

                    d[_dst.data_offset++] = s[srcIter.dataptr.data_offset];

                    numpyinternal.NpyArray_ITER_NEXT(index_iter);
                }
            }

            return null;
        }

        public void IterSubscriptAssignSlice(NpyArrayIterObject destIter, NpyArrayIterObject srcIter, npy_intp steps, npy_intp start, npy_intp step_size, bool swap)
        {
            int elsize = GetTypeSize(destIter.dataptr);
            int divsize = GetDivSize(elsize);

            destIter = numpyinternal.NpyArray_ITER_ConvertToIndex(destIter, divsize);
            srcIter = numpyinternal.NpyArray_ITER_ConvertToIndex(srcIter, divsize);

            T[] d = destIter.dataptr.datap as T[];
            T[] s = srcIter.dataptr.datap as T[];


            if (srcIter.size == 1)
            {
                T sValue = s[srcIter.dataptr.data_offset];

                if (swap)
                {
                    while (steps-- > 0)
                    {
                        numpyinternal.NpyArray_ITER_GOTO1D_INDEX(destIter, start);
                        d[destIter.dataptr.data_offset] = sValue;
                        numpyinternal.swapvalue(destIter.dataptr, destIter.dataptr.data_offset, 0);
                        start += step_size;
                    }
                }
                else
                {
                    while (steps-- > 0)
                    {
                        numpyinternal.NpyArray_ITER_GOTO1D_INDEX(destIter, start);
                        d[destIter.dataptr.data_offset] = sValue;
                        start += step_size;
                    }
                }

      
            }
            else
            {
                if (swap)
                {
                    while (steps-- > 0)
                    {
                        numpyinternal.NpyArray_ITER_GOTO1D_INDEX(destIter, start);

                        d[destIter.dataptr.data_offset] = s[srcIter.dataptr.data_offset];
                        numpyinternal.swapvalue(destIter.dataptr, destIter.dataptr.data_offset, 0);

                        numpyinternal.NpyArray_ITER_NEXT(srcIter);
                        if (!numpyinternal.NpyArray_ITER_NOTDONE(srcIter))
                        {
                            numpyinternal.NpyArray_ITER_RESET(srcIter);
                        }
                        start += step_size;
                    }
                }
                else
                {
                    while (steps-- > 0)
                    {
                        numpyinternal.NpyArray_ITER_GOTO1D_INDEX(destIter, start);

                        d[destIter.dataptr.data_offset] = s[srcIter.dataptr.data_offset];

                        numpyinternal.NpyArray_ITER_NEXT(srcIter);
                        if (!numpyinternal.NpyArray_ITER_NOTDONE(srcIter))
                        {
                            numpyinternal.NpyArray_ITER_RESET(srcIter);
                        }
                        start += step_size;
                    }
                }
    
            }

        }

        //public void IterSubscriptAssignSlice_USECACHEMODEL(NpyArrayIterObject destIter, NpyArrayIterObject srcIter, npy_intp steps, npy_intp start, npy_intp step_size, bool swap)
        //{
        //    int elsize = GetTypeSize(destIter.dataptr);
        //    int divsize = GetDivSize(elsize);

        //    T[] d = destIter.dataptr.datap as T[];
        //    T[] s = srcIter.dataptr.datap as T[];


        //    if (srcIter.size == 1)
        //    {
        //        //srcIter.dataptr.data_offset >>= divsize;

        //        npy_intp src_offset = srcIter.dataptr.data_offset >> divsize;

        //        while (steps > 0)
        //        {
        //            numpyinternal.NpyArray_ITER_GOTO1D_CACHE(destIter, start, step_size, steps);

        //            for (int i = 0; i < destIter.internalCacheLength; i++)
        //            {
        //                d[destIter.internalCache[i] >> divsize] = s[src_offset];

        //                if (swap)
        //                {
        //                    numpyinternal.swapvalue(destIter.dataptr, destIter.internalCache[i], divsize);
        //                }
        //            }
        //            start += step_size * destIter.internalCacheLength;
        //            steps -= destIter.internalCacheLength;
        //        }
        //    }
        //    else
        //    {
        //        while (steps > 0)
        //        {
        //            numpyinternal.NpyArray_ITER_GOTO1D_CACHE(destIter, start, step_size, steps);

        //            for (int i = 0; i < destIter.internalCacheLength; i++)
        //            {
        //                d[destIter.internalCache[i] >> divsize] = s[srcIter.dataptr.data_offset >> divsize];

        //                if (swap)
        //                {
        //                    numpyinternal.swapvalue(destIter.dataptr, destIter.internalCache[i], divsize);
        //                }

        //                numpyinternal.NpyArray_ITER_NEXT(srcIter);
        //                if (!numpyinternal.NpyArray_ITER_NOTDONE(srcIter))
        //                {
        //                    numpyinternal.NpyArray_ITER_RESET(srcIter);
        //                }
        //            }

        //            start += step_size * destIter.internalCacheLength;
        //            steps -= destIter.internalCacheLength;
        //        }
        //    }

        //}

        public void IterSubscriptAssignBoolArray(NpyArrayIterObject destIter, NpyArrayIterObject srcIter, npy_intp bool_size, bool[] bool_mask, npy_intp stride, bool swap)
        {
            int elsize = GetTypeSize(destIter.dataptr);
            int divsize = GetDivSize(elsize);

            T[] d = destIter.dataptr.datap as T[];
            T[] s = srcIter.dataptr.datap as T[];

            destIter = numpyinternal.NpyArray_ITER_ConvertToIndex(destIter, divsize);

            npy_intp dptr_index = 0;

            if (srcIter.size == 1)
            {
                T sValue = s[srcIter.dataptr.data_offset >>= divsize];

                if (swap)
                {
                    while (bool_size-- > 0)
                    {
                        if (bool_mask[dptr_index])
                        {
                            d[destIter.dataptr.data_offset] = sValue;
                            numpyinternal.swapvalue(destIter.dataptr, destIter.dataptr.data_offset, 0);
                        }
                        dptr_index += stride;
                        numpyinternal.NpyArray_ITER_NEXT(destIter);
                    }
                }
                else
                {
                    while (bool_size-- > 0)
                    {
                        if (bool_mask[dptr_index])
                        {
                            d[destIter.dataptr.data_offset] = sValue;
                        }
                        dptr_index += stride;
                        numpyinternal.NpyArray_ITER_NEXT(destIter);
                    }
                }

    
            }
            else
            {
                srcIter = numpyinternal.NpyArray_ITER_ConvertToIndex(srcIter, divsize);

                if (swap)
                {
                    while (bool_size-- > 0)
                    {
                        if (bool_mask[dptr_index])
                        {
                            d[destIter.dataptr.data_offset] = s[srcIter.dataptr.data_offset];
                            numpyinternal.swapvalue(destIter.dataptr, destIter.dataptr.data_offset, 0);

                            numpyinternal.NpyArray_ITER_NEXT(srcIter);
                            if (!numpyinternal.NpyArray_ITER_NOTDONE(srcIter))
                            {
                                numpyinternal.NpyArray_ITER_RESET(srcIter, divsize);
                            }
                        }
                        dptr_index += stride;
                        numpyinternal.NpyArray_ITER_NEXT(destIter);
                    }
                }
                else
                {
                    while (bool_size-- > 0)
                    {
                        if (bool_mask[dptr_index])
                        {
                            d[destIter.dataptr.data_offset] = s[srcIter.dataptr.data_offset];

                            numpyinternal.NpyArray_ITER_NEXT(srcIter);
                            if (!numpyinternal.NpyArray_ITER_NOTDONE(srcIter))
                            {
                                numpyinternal.NpyArray_ITER_RESET(srcIter, divsize);
                            }
                        }
                        dptr_index += stride;
                        numpyinternal.NpyArray_ITER_NEXT(destIter);
                    }
                }

         
            }

      
        }

        public npy_intp? IterSubscriptAssignIntpArray(NpyArrayIterObject destIter, NpyArrayIterObject indexIter, NpyArrayIterObject srcIter, bool swap)
        {
            int elsize = GetTypeSize(destIter.dataptr);
            int divsize = GetDivSize(elsize);
            var divintp = GetDivSize(sizeof(npy_intp));

            T[] d = destIter.dataptr.datap as T[];
            T[] s = srcIter.dataptr.datap as T[];

            npy_intp[] dataptr = indexIter.dataptr.datap as npy_intp[];
            npy_intp i = indexIter.size;

            indexIter = numpyinternal.NpyArray_ITER_ConvertToIndex(indexIter, divintp);
            destIter = numpyinternal.NpyArray_ITER_ConvertToIndex(destIter, divsize);

            if (srcIter.size == 1)
            {
                T sValue = s[srcIter.dataptr.data_offset >> divsize];

                if (swap)
                {
                    while (i-- > 0)
                    {
                        npy_intp num = dataptr[indexIter.dataptr.data_offset];
                        if (num < 0)
                        {
                            num += destIter.size;
                        }
                        if (num < 0 || num >= destIter.size)
                        {
                            return num;
                        }
                        numpyinternal.NpyArray_ITER_GOTO1D_INDEX(destIter, num);

                        d[destIter.dataptr.data_offset] = sValue;
                        numpyinternal.swapvalue(destIter.dataptr, destIter.dataptr.data_offset, 0);

                        numpyinternal.NpyArray_ITER_NEXT(indexIter);
                    }
                }
                else
                {
                    while (i-- > 0)
                    {
                        npy_intp num = dataptr[indexIter.dataptr.data_offset];
                        if (num < 0)
                        {
                            num += destIter.size;
                        }
                        if (num < 0 || num >= destIter.size)
                        {
                            return num;
                        }
                        numpyinternal.NpyArray_ITER_GOTO1D_INDEX(destIter, num);

                        d[destIter.dataptr.data_offset] = sValue;
                        numpyinternal.NpyArray_ITER_NEXT(indexIter);
                    }
                }


   
            }
            else
            {

                srcIter = numpyinternal.NpyArray_ITER_ConvertToIndex(srcIter, divsize);

                if (swap)
                {
                    while (i-- > 0)
                    {
                        npy_intp num = dataptr[indexIter.dataptr.data_offset];
                        if (num < 0)
                        {
                            num += destIter.size;
                        }
                        if (num < 0 || num >= destIter.size)
                        {
                            return num;
                        }
                        numpyinternal.NpyArray_ITER_GOTO1D_INDEX(destIter, num);

                        d[destIter.dataptr.data_offset] = s[srcIter.dataptr.data_offset];
                        numpyinternal.swapvalue(destIter.dataptr, destIter.dataptr.data_offset, 0);

                        numpyinternal.NpyArray_ITER_NEXT(srcIter);
                        if (!numpyinternal.NpyArray_ITER_NOTDONE(srcIter))
                        {
                            numpyinternal.NpyArray_ITER_RESET(srcIter, divsize);
                        }
                        numpyinternal.NpyArray_ITER_NEXT(indexIter);
                    }
                }
                else
                {
                    while (i-- > 0)
                    {
                        npy_intp num = dataptr[indexIter.dataptr.data_offset];
                        if (num < 0)
                        {
                            num += destIter.size;
                        }
                        if (num < 0 || num >= destIter.size)
                        {
                            return num;
                        }
                        numpyinternal.NpyArray_ITER_GOTO1D_INDEX(destIter, num);

                        d[destIter.dataptr.data_offset] = s[srcIter.dataptr.data_offset];

                        numpyinternal.NpyArray_ITER_NEXT(srcIter);
                        if (!numpyinternal.NpyArray_ITER_NOTDONE(srcIter))
                        {
                            numpyinternal.NpyArray_ITER_RESET(srcIter, divsize);
                        }
                        numpyinternal.NpyArray_ITER_NEXT(indexIter);
                    }
                }

    
            }

   

            return null;
        }
        public void GetMap(NpyArrayIterObject destIter, NpyArrayMapIterObject srcIter, bool swap)
        {
            // see test_Taz145_2 for unit test that fails if this code is used.
            throw new Exception("Do not use. Defective for negative stride values");

            int elsize = GetTypeSize(destIter.dataptr);
            int divsize = GetDivSize(elsize);

            srcIter = numpyinternal.NpyArray_ITER_ConvertToIndex(srcIter);

            T[] d = destIter.dataptr.datap as T[];
            T[] s = srcIter.dataptr.datap as T[];

            var numIndexes = destIter.size;

            if (srcIter.subspace != null)
            {
                npy_intp[] offsets = new npy_intp[numpyinternal.maxIterOffsetCacheSize];
                npy_intp offsetsLength = Math.Min((npy_intp)offsets.Length, numIndexes);

                offsets[0] = srcIter.dataptr.data_offset >>= divsize;
                numpyinternal.NpyArray_MapIterNext_SubSpace(srcIter, offsets, offsetsLength, 1);
                int offsetsIndex = 0;


                if (destIter.contiguous)
                {
                    npy_intp destIndex = destIter.dataptr.data_offset >> divsize;

                    while (numIndexes > 0)
                    {
                        while (offsetsIndex < offsetsLength)
                        {
                            d[destIndex++] = s[offsets[offsetsIndex]];
                            offsetsIndex++;
                        }
                        numIndexes -= offsetsIndex;
                        if (numIndexes > 0)
                        {
                            offsetsLength = Math.Min((npy_intp)offsets.Length, numIndexes);
                            numpyinternal.NpyArray_MapIterNext_SubSpace(srcIter, offsets, offsetsLength, 0);
                            offsetsIndex = 0;
                        }

                    }
                }
                else
                {
                    destIter = numpyinternal.NpyArray_ITER_ConvertToIndex(destIter, divsize);

                    while (numIndexes > 0)
                    {
                        while (offsetsIndex < offsetsLength)
                        {
                            d[destIter.dataptr.data_offset] = s[offsets[offsetsIndex]];
                            offsetsIndex++;
                            numpyinternal.NpyArray_ITER_NEXT(destIter);
                        }
                        numIndexes -= offsetsIndex;
                        if (numIndexes > 0)
                        {
                            offsetsLength = Math.Min((npy_intp)offsets.Length, numIndexes);
                            numpyinternal.NpyArray_MapIterNext_SubSpace(srcIter, offsets, offsetsLength, 0);
                            offsetsIndex = 0;
                        }

                    }
                }
            }
            else
            {
                npy_intp[] offsets = new npy_intp[numpyinternal.maxIterOffsetCacheSize];
                npy_intp offsetsLength = Math.Min((npy_intp)offsets.Length, numIndexes);

                offsets[0] = srcIter.dataptr.data_offset >>= divsize;
                numpyinternal.NpyArray_MapIterNext(srcIter, offsets, offsetsLength, 1);
                int offsetsIndex = 0;

                if (destIter.contiguous)
                {
                    npy_intp destIndex = destIter.dataptr.data_offset >> divsize;

                    while (numIndexes > 0)
                    {
                        while (offsetsIndex < offsetsLength)
                        {
                            d[destIndex++] = s[offsets[offsetsIndex++]];
                        }

                        numIndexes -= offsetsIndex;
                        if (numIndexes > 0)
                        {
                            offsetsLength = Math.Min((npy_intp)offsets.Length, numIndexes);
                            numpyinternal.NpyArray_MapIterNext(srcIter, offsets, offsetsLength, 0);
                            offsetsIndex = 0;
                        }
                    }
                }
                else
                {
                    destIter = numpyinternal.NpyArray_ITER_ConvertToIndex(destIter, divsize);

                    while (numIndexes > 0)
                    {
                        while (offsetsIndex < offsetsLength)
                        {
                            d[destIter.dataptr.data_offset] = s[offsets[offsetsIndex++]];
                            numpyinternal.NpyArray_ITER_NEXT(destIter);
                        }

                        numIndexes -= offsetsIndex;
                        if (numIndexes > 0)
                        {
                            offsetsLength = Math.Min((npy_intp)offsets.Length, numIndexes);
                            numpyinternal.NpyArray_MapIterNext(srcIter, offsets, offsetsLength, 0);
                            offsetsIndex = 0;
                        }
                    }
                }
            }


        }

        public void SetMap(NpyArrayMapIterObject destIter, NpyArrayIterObject srcIter, bool swap)
        {
            // see test_Taz145_1 for unit test that fails if this code is used.
            throw new Exception("Do not use. Defective for negative stride values");

            int divsize = GetDivSize(destIter.dataptr);

            var numIndexes = destIter.size;


            if (destIter.dataptr.type_num != srcIter.dataptr.type_num)
            {
                throw new Exception(string.Format("Unable to set {0} values into {1} arrays", 
                    srcIter.dataptr.type_num.ToString(), destIter.dataptr.type_num.ToString()));
                //npy_intp[] offsets = new npy_intp[numpyinternal.maxIterOffsetCacheSize];
                //npy_intp offsetsLength = Math.Min((npy_intp)offsets.Length, numIndexes);

                //offsets[0] = destIter.dataptr.data_offset;

                //numpyinternal.NpyArray_MapIterNext(destIter, offsets, offsetsLength, 1);
                //int offsetsIndex = 0;

                //var helper = MemCopy.GetMemcopyHelper(destIter.dataptr);
                //helper.memmove_init(destIter.dataptr, srcIter.dataptr);

                //while (numIndexes > 0)
                //{
                //    if (swap)
                //    {
                //        while (offsetsIndex < offsetsLength)
                //        {
                //            helper.memmoveitem(offsets[offsetsIndex], srcIter.dataptr.data_offset);
 
                //            numpyinternal.swapvalue(destIter.dataptr, destIter.dataptr.data_offset, divsize);
                //            numpyinternal.NpyArray_ITER_NEXT(srcIter);
                //        }
                //    }
                //    else
                //    {
                //        while (offsetsIndex < offsetsLength)
                //        {
                //            helper.memmoveitem(offsets[offsetsIndex], srcIter.dataptr.data_offset);
 
                //            numpyinternal.NpyArray_ITER_NEXT(srcIter);
                //        }
                //    }
    

                //    numIndexes -= offsetsIndex;
                //    if (numIndexes > 0)
                //    {
                //        offsetsLength = Math.Min((npy_intp)offsets.Length, numIndexes);
                //        numpyinternal.NpyArray_MapIterNext(destIter, offsets, offsetsLength, 0);
                //        offsetsIndex = 0;
                //    }

                //}
      
                //return;
            }

            destIter = numpyinternal.NpyArray_ITER_ConvertToIndex(destIter);
            srcIter = numpyinternal.NpyArray_ITER_ConvertToIndex(srcIter, divsize);

            T[] d = destIter.dataptr.datap as T[];
            T[] s = srcIter.dataptr.datap as T[];


            if (destIter.subspace != null)
            {

                npy_intp[] offsets = new npy_intp[numpyinternal.maxIterOffsetCacheSize];
                npy_intp offsetsLength = Math.Min((npy_intp)offsets.Length, numIndexes);

                offsets[0] = destIter.dataptr.data_offset >> divsize;
                numpyinternal.NpyArray_MapIterNext_SubSpace(destIter, offsets, offsetsLength, 1);
                int offsetsIndex = 0;

                while (numIndexes > 0)
                {
                    if (swap)
                    {
                        while (offsetsIndex < offsetsLength)
                        {
                            d[offsets[offsetsIndex++]] = s[srcIter.dataptr.data_offset];
                            numpyinternal.swapvalue(destIter.dataptr, destIter.dataptr.data_offset, 0);

                            numpyinternal.NpyArray_ITER_NEXT(srcIter);
                        }
                    }
                    else
                    {
                        while (offsetsIndex < offsetsLength)
                        {
                            d[offsets[offsetsIndex++]] = s[srcIter.dataptr.data_offset];
 
                            numpyinternal.NpyArray_ITER_NEXT(srcIter);
                        }
                    }
 

                    numIndexes -= offsetsIndex;
                    if (numIndexes > 0)
                    {
                        offsetsLength = Math.Min((npy_intp)offsets.Length, numIndexes);
                        numpyinternal.NpyArray_MapIterNext_SubSpace(destIter, offsets, offsetsLength, 0);
                        offsetsIndex = 0;
                    }

                }
            }
            else
            {
                npy_intp[] offsets = new npy_intp[numpyinternal.maxIterOffsetCacheSize];
                npy_intp offsetsLength = Math.Min((npy_intp)offsets.Length, numIndexes);

                offsets[0] = destIter.dataptr.data_offset >> divsize;
                numpyinternal.NpyArray_MapIterNext(destIter, offsets, offsetsLength, 1);
                int offsetsIndex = 0;

                while (numIndexes > 0)
                {
                    if (swap)
                    {
                        while (offsetsIndex < offsetsLength)
                        {
                            d[offsets[offsetsIndex++]] = s[srcIter.dataptr.data_offset];
                            numpyinternal.swapvalue(destIter.dataptr, destIter.dataptr.data_offset, 0);

                            numpyinternal.NpyArray_ITER_NEXT(srcIter);
                        }
                    }
                    else
                    {
                        while (offsetsIndex < offsetsLength)
                        {
                            d[offsets[offsetsIndex++]] = s[srcIter.dataptr.data_offset];
                            numpyinternal.NpyArray_ITER_NEXT(srcIter);
                        }
                    }
    

                    numIndexes -= offsetsIndex;
                    if (numIndexes > 0)
                    {
                        offsetsLength = Math.Min((npy_intp)offsets.Length, numIndexes);
                        numpyinternal.NpyArray_MapIterNext(destIter, offsets, offsetsLength, 0);
                        offsetsIndex = 0;
                    }

                }
            }




        }

        public void FillWithScalar(VoidPtr destPtr, VoidPtr srcPtr, npy_intp size, bool swap)
        {
            int elsize = GetTypeSize(destPtr);
            int divsize = GetDivSize(elsize);

            T[] d = destPtr.datap as T[];
            T[] s = srcPtr.datap as T[];
            T fillValue = s[0];

            destPtr.data_offset >>= divsize;
            while (size-- > 0)
            {
                d[destPtr.data_offset++] = fillValue;

                if (swap)
                {
                    numpyinternal.swapvalue(destPtr, destPtr.data_offset, divsize);
                }
            }
        }

        public void FillWithScalarIter(NpyArrayIterObject destIter, VoidPtr srcPtr, npy_intp size, bool swap)
        {
            int elsize = GetTypeSize(destIter.dataptr);
            int divsize = GetDivSize(elsize);

            T[] d = destIter.dataptr.datap as T[];
            T[] s = srcPtr.datap as T[];
            T fillValue = s[0];

            while (size-- > 0)
            {
                d[destIter.dataptr.data_offset >> divsize] = fillValue;

                if (swap)
                {
                    numpyinternal.swapvalue(destIter.dataptr, destIter.dataptr.data_offset, divsize);
                }

                numpyinternal.NpyArray_ITER_NEXT(destIter);
            }
        }

        public void MatrixProduct(NpyArrayIterObject it1, NpyArrayIterObject it2, VoidPtr op, npy_intp is1, npy_intp is2, npy_intp os, npy_intp l)
        {
            var spread = it1.size - it1.index;
            var spread2 = it2.size - it2.index;
            if (spread < 4 || spread2 < 4)
            {
                MatrixProductExecute(it1, it2, op, is1, is2, os, l);
            }
            else
            {
                var it1a = it1.copy();
                var it1b = it1.copy();

                var it2a = it2.copy();
                var it2b = it2.copy();
                var it2c = it2.copy();
                var it2d = it2.copy();

                VoidPtr opa = new VoidPtr(op);
                VoidPtr opb = new VoidPtr(op);

                var taskSize = spread / 4;

                it1a.size = taskSize;

                while (it1b.index < it1a.size)
                {
                    numpyinternal.NpyArray_ITER_NEXT(it1b);
                    opb.data_offset += os * (it2.size - it2.index);
                }

                it1b.size = taskSize * 2;
                VoidPtr opc = new VoidPtr(opb);
                var it1c = it1b.copy();

                while (it1c.index < it1b.size)
                {
                    numpyinternal.NpyArray_ITER_NEXT(it1c);
                    opc.data_offset += os * (it2.size - it2.index);
                }

                it1c.size = taskSize * 3;
                VoidPtr opd = new VoidPtr(opc);
                var it1d = it1c.copy();

                while (it1d.index < it1c.size)
                {
                    numpyinternal.NpyArray_ITER_NEXT(it1d);
                    opd.data_offset += os * (it2.size - it2.index);
                }
                it1d.size = it1.size;

                var t1 = Task.Run(() =>
                {
                    MatrixProductExecute(it1a, it2a, opa, is1, is2, os, l);
                });

                var t2 = Task.Run(() =>
                {
                    MatrixProductExecute(it1b, it2b, opb, is1, is2, os, l);
                });

                var t3 = Task.Run(() =>
                {
                    MatrixProductExecute(it1c, it2c, opc, is1, is2, os, l);
                });

                // run the last task in the current thread.
                MatrixProductExecute(it1d, it2d, opd, is1, is2, os, l);

                Task.WaitAll(t1, t2, t3);
            }

            return;
        }

        private void MatrixProductExecute(NpyArrayIterObject it1d, NpyArrayIterObject it2d, VoidPtr opd, npy_intp is1, npy_intp is2, npy_intp os, npy_intp l)
        {
            while (true)
            {
                while (it2d.index < it2d.size)
                {
                    dot(it1d.dataptr, is1, it2d.dataptr, is2, opd, l);
                    opd.data_offset += os;
                    numpyinternal.NpyArray_ITER_NEXT(it2d);
                }
                numpyinternal.NpyArray_ITER_NEXT(it1d);
                if (it1d.index >= it1d.size)
                {
                    break;
                }
                numpyinternal.NpyArray_ITER_RESET(it2d);
            }
        }


        public void InnerProduct(NpyArrayIterObject it1, NpyArrayIterObject it2, VoidPtr op, npy_intp is1, npy_intp is2, npy_intp os, npy_intp l)
        {
            var spread = it1.size - it1.index;
            var spread2 = it2.size - it2.index;
            if (spread < 4 || spread2 < 40)
            {
                InnerProductExecute(it1, it2, op, is1, is2, os, l);
            }
            else
            {
                var it1a = it1.copy();
                var it1b = it1.copy();

                var it2a = it2.copy();
                var it2b = it2.copy();
                var it2c = it2.copy();
                var it2d = it2.copy();

                VoidPtr opa = new VoidPtr(op);
                VoidPtr opb = new VoidPtr(op);

                var taskSize = spread / 4;

                it1a.size = taskSize;

                while (it1b.index < it1a.size)
                {
                    numpyinternal.NpyArray_ITER_NEXT(it1b);
                    opb.data_offset += os * (it2.size - it2.index);
                }

                it1b.size = taskSize * 2;
                VoidPtr opc = new VoidPtr(opb);
                var it1c = it1b.copy();

                while (it1c.index < it1b.size)
                {
                    numpyinternal.NpyArray_ITER_NEXT(it1c);
                    opc.data_offset += os * (it2.size - it2.index);
                }

                it1c.size = taskSize * 3;
                VoidPtr opd = new VoidPtr(opc);
                var it1d = it1c.copy();

                while (it1d.index < it1c.size)
                {
                    numpyinternal.NpyArray_ITER_NEXT(it1d);
                    opd.data_offset += os * (it2.size - it2.index);
                }
                it1d.size = it1.size;

                var t1 = Task.Run(() =>
                {
                    MatrixProductExecute(it1a, it2a, opa, is1, is2, os, l);
                });

                var t2 = Task.Run(() =>
                {
                    MatrixProductExecute(it1b, it2b, opb, is1, is2, os, l);
                });

                var t3 = Task.Run(() =>
                {
                    MatrixProductExecute(it1c, it2c, opc, is1, is2, os, l);
                });

                // run the last task in the current thread.
                MatrixProductExecute(it1d, it2d, opd, is1, is2, os, l);

                Task.WaitAll(t1, t2, t3);
            }

            return;
        }

        private void InnerProductExecute(NpyArrayIterObject it1, NpyArrayIterObject it2, VoidPtr op, npy_intp is1, npy_intp is2, npy_intp os, npy_intp l)
        {
            while (true)
            {
                while (it2.index < it2.size)
                {
                    dot(it1.dataptr, is1, it2.dataptr, is2, op, l);
                    op.data_offset += os;
                    numpyinternal.NpyArray_ITER_NEXT(it2);
                }
                numpyinternal.NpyArray_ITER_NEXT(it1);
                if (it1.index >= it1.size)
                {
                    break;
                }
                numpyinternal.NpyArray_ITER_RESET(it2);
            }
        }


        public void correlate(VoidPtr ip1, VoidPtr ip2, VoidPtr op, npy_intp is1, npy_intp is2, npy_intp os, npy_intp n, npy_intp n1, npy_intp n2, npy_intp n_left, npy_intp n_right)
        {
            if (true)
            {
                Parallel.For(0, n_left, i =>
                //for (int i = 0; i < n_left; i++)
                {
                    npy_intp nn = n + i;

                    VoidPtr nip2 = new VoidPtr(ip2);
                    nip2.data_offset -= is2 * i;

                    VoidPtr nop = new VoidPtr(op);
                    nop.data_offset += os * i;

                    dot(ip1, is1, nip2, is2, nop, nn);

                });

                n += n_left;
                ip2.data_offset -= is2 * n_left;
                op.data_offset += os * n_left;


                npy_intp loop_cnt = n1 - n2 + 1;
                Parallel.For(0, loop_cnt, i =>
                //for (int i = 0; i < loop_cnt; i++)
                {
                    VoidPtr nip1 = new VoidPtr(ip1);
                    nip1.data_offset += is1 * i;

                    VoidPtr nop = new VoidPtr(op);
                    nop.data_offset += os * i;

                    dot(nip1, is1, ip2, is2, nop, n);
                });

                ip1.data_offset += is1 * loop_cnt;
                op.data_offset += os * loop_cnt;


                Parallel.For(0, n_right, i =>
                //for (int i = 0; i < n_right; i++)
                {
                    npy_intp nn = n;
                    nn -= i+1;

                    VoidPtr nip1 = new VoidPtr(ip1);
                    nip1.data_offset += is1 * i;

                    VoidPtr nop = new VoidPtr(op);
                    nop.data_offset += os * i;

                    dot(nip1, is1, ip2, is2, nop, nn);
                });

                n -= n_right;
                ip1.data_offset += is1 * n_right;
                op.data_offset += os * n_right;
            }
            else
            {
                for (int i = 0; i < n_left; i++)
                {
                    dot(ip1, is1, ip2, is2, op, n);
                    n++;
                    ip2.data_offset -= is2;
                    op.data_offset += os;
                }

                for (int i = 0; i < (n1 - n2 + 1); i++)
                {
                    dot(ip1, is1, ip2, is2, op, n);
                    ip1.data_offset += is1;
                    op.data_offset += os;
                }
                for (int i = 0; i < n_right; i++)
                {
                    n--;
                    dot(ip1, is1, ip2, is2, op, n);
                    ip1.data_offset += is1;
                    op.data_offset += os;
                }
            }

 
        }

        private void dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            // lets use specific functions.  It is horrible replication of code but much more performant.
            switch (_ip1.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    break;

                case NPY_TYPES.NPY_BYTE:
                    dot_BYTE(_ip1, is1, _ip2, is2, _op, n);
                    return;

                case NPY_TYPES.NPY_UBYTE:
                    dot_UBYTE(_ip1, is1, _ip2, is2, _op, n);
                    return;

                case NPY_TYPES.NPY_INT16:
                    dot_INT16(_ip1, is1, _ip2, is2, _op, n);
                    return;

                case NPY_TYPES.NPY_UINT16:
                    dot_UINT16(_ip1, is1, _ip2, is2, _op, n);
                    return;

                case NPY_TYPES.NPY_INT32:
                    dot_INT32(_ip1, is1, _ip2, is2, _op, n);
                    return;
                case NPY_TYPES.NPY_UINT32:
                    dot_UINT32(_ip1, is1, _ip2, is2, _op, n);
                    return;

                case NPY_TYPES.NPY_INT64:
                    dot_INT64(_ip1, is1, _ip2, is2, _op, n);
                    return;

                case NPY_TYPES.NPY_UINT64:
                    dot_UINT64(_ip1, is1, _ip2, is2, _op, n);
                    return;

                case NPY_TYPES.NPY_FLOAT:
                    dot_FLOAT(_ip1, is1, _ip2, is2, _op, n);
                    return;

                case NPY_TYPES.NPY_DOUBLE:
                    dot_DOUBLE(_ip1, is1, _ip2, is2, _op, n);
                    return;

                case NPY_TYPES.NPY_DECIMAL:
                    dot_DECIMAL(_ip1, is1, _ip2, is2, _op, n);
                    return;

                case NPY_TYPES.NPY_COMPLEX:
                    dot_COMPLEX(_ip1, is1, _ip2, is2, _op, n);
                    return;

                case NPY_TYPES.NPY_BIGINT:
                    dot_BIGINT(_ip1, is1, _ip2, is2, _op, n);
                    return;

                case NPY_TYPES.NPY_OBJECT:
                    dot_OBJECT(_ip1, is1, _ip2, is2, _op, n);
                    return;

                case NPY_TYPES.NPY_STRING:
                    break;

                default:
                    break;
            }
 

            int ip1Div = GetDivSize(_ip1);
            int ip2Div = GetDivSize(_ip2);
            int opDiv = GetDivSize(_op);


            T tmp = default(T);
            npy_intp i;

            T[] ip1 = _ip1.datap as T[];
            T[] ip2 = _ip2.datap as T[];
            T[] op = _op.datap as T[];

            npy_intp ip1_index = _ip1.data_offset >> ip1Div;
            npy_intp ip2_index = _ip2.data_offset >> ip2Div;
            is1 >>= ip1Div;
            is2 >>= ip2Div;


            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                tmp = T_dot(tmp, ip1, ip2, ip1_index, ip2_index);
            }


            op[_op.data_offset >> opDiv] = tmp;
        }
        #region type specific DOT methods
        private void dot_BYTE(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);

            sbyte tmp = 0;

            var ip1 = _ip1.datap as sbyte[];
            var ip2 = _ip2.datap as sbyte[];
            var op = _op.datap as sbyte[];

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += (sbyte)(ip1[ip1_index] * ip2[ip2_index]);
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }

        private void dot_UBYTE(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);

            byte tmp = 0;

            var ip1 = _ip1.datap as byte[];
            var ip2 = _ip2.datap as byte[];
            var op = _op.datap as byte[];

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += (byte)(ip1[ip1_index] * ip2[ip2_index]);
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }

        private void dot_INT16(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);

            Int16 tmp = 0;

            var ip1 = _ip1.datap as Int16[];
            var ip2 = _ip2.datap as Int16[];
            var op = _op.datap as Int16[];

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += (Int16)(ip1[ip1_index] * ip2[ip2_index]);
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }

        private void dot_UINT16(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);

            UInt16 tmp = 0;

            var ip1 = _ip1.datap as UInt16[];
            var ip2 = _ip2.datap as UInt16[];
            var op = _op.datap as UInt16[];

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += (UInt16)(ip1[ip1_index] * ip2[ip2_index]);
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }

        private void dot_INT32(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);
   
            Int32 tmp = 0;

            var ip1 = _ip1.datap as Int32[];
            var ip2 = _ip2.datap as Int32[];
            var op = _op.datap as Int32[];

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += ip1[ip1_index] * ip2[ip2_index];
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }

        private void dot_UINT32(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);

            UInt32 tmp = 0;

            var ip1 = _ip1.datap as UInt32[];
            var ip2 = _ip2.datap as UInt32[];
            var op = _op.datap as UInt32[];

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += ip1[ip1_index] * ip2[ip2_index];
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }

        private void dot_INT64(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);

            Int64 tmp = 0;

            var ip1 = _ip1.datap as Int64[];
            var ip2 = _ip2.datap as Int64[];
            var op = _op.datap as Int64[];

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += ip1[ip1_index] * ip2[ip2_index];
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }

        private void dot_UINT64(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);

            UInt64 tmp = 0;

            var ip1 = _ip1.datap as UInt64[];
            var ip2 = _ip2.datap as UInt64[];
            var op = _op.datap as UInt64[];

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += ip1[ip1_index] * ip2[ip2_index];
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }

        private void dot_FLOAT(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);

            float tmp = 0;

            var ip1 = _ip1.datap as float[];
            var ip2 = _ip2.datap as float[];
            var op = _op.datap as float[];

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += ip1[ip1_index] * ip2[ip2_index];
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }

        private void dot_DOUBLE(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);

            double tmp = 0;

            var ip1 = _ip1.datap as double[];
            var ip2 = _ip2.datap as double[];
            var op = _op.datap as double[];

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += ip1[ip1_index] * ip2[ip2_index];
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }

        private void dot_DECIMAL(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);

            decimal tmp = 0;

            var ip1 = _ip1.datap as decimal[];
            var ip2 = _ip2.datap as decimal[];
            var op = _op.datap as decimal[];

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += ip1[ip1_index] * ip2[ip2_index];
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }

        private void dot_BIGINT(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);

            System.Numerics.BigInteger tmp = 0;

            var ip1 = _ip1.datap as System.Numerics.BigInteger[];
            var ip2 = _ip2.datap as System.Numerics.BigInteger[];
            var op = _op.datap as System.Numerics.BigInteger[];

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += ip1[ip1_index] * ip2[ip2_index];
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }

        private void dot_COMPLEX(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);

            System.Numerics.Complex tmp = 0;

            var ip1 = _ip1.datap as System.Numerics.Complex[];
            var ip2 = _ip2.datap as System.Numerics.Complex[];
            var op = _op.datap as System.Numerics.Complex[];

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += ip1[ip1_index] * ip2[ip2_index];
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }

        private void dot_OBJECT(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            int ipDiv = GetDivSize(_ip1);

            dynamic tmp = 0;

            var ip1 = _ip1.datap as dynamic;
            var ip2 = _ip2.datap as dynamic;
            var op = _op.datap as dynamic;

            npy_intp ip1_index = _ip1.data_offset >> ipDiv;
            npy_intp ip2_index = _ip2.data_offset >> ipDiv;
            is1 >>= ipDiv;
            is2 >>= ipDiv;

            for (int i = 0; i < n; i++)
            {
                tmp += ip1[ip1_index] * ip2[ip2_index];
                ip1_index += is1; ip2_index += is2;
            }

            op[_op.data_offset >> ipDiv] = tmp;
        }
        #endregion

        protected abstract T T_dot(T otmp, T[] op1, T[] op2, npy_intp ip1_index, npy_intp ip2_index);
   
        public void copyswap(VoidPtr _dst, VoidPtr _src, bool swap)
        {

            int elsize = GetTypeSize(_dst);
            int divsize = GetDivSize(elsize);

            if (_src != null)
            {
                T[] d = _dst.datap as T[];
                T[] s = _src.datap as T[];

                d[_dst.data_offset >> divsize] = s[_src.data_offset >> divsize];
            }

            if (swap)
            {
                numpyinternal.swapvalue(_dst, _dst.data_offset, divsize);
            }
        }

        public void default_copyswap(VoidPtr _dst, npy_intp dstride, VoidPtr _src, npy_intp sstride, npy_intp n, bool swap)
        {
            int elsize = GetTypeSize(_dst);
            int divsize = GetDivSize(elsize);

            if (_src != null)
            {
                T[] d = _dst.datap as T[];
                T[] s = _src.datap as T[];

                if (swap)
                {
                    for (int i = 0; i < n; i++)
                    {
                        d[_dst.data_offset >> divsize] = s[_src.data_offset >> divsize];

                        numpyinternal.swapvalue(_dst, _dst.data_offset, divsize);

                        _dst.data_offset += dstride;
                        _src.data_offset += sstride;
                    }
                }
                else
                {
                    _dst.data_offset >>= divsize;
                    dstride >>= divsize;
                    sstride >>= divsize;
                    _src.data_offset >>= divsize;

                    for (int i = 0; i < n; i++)
                    {
                        d[_dst.data_offset] = s[_src.data_offset];
                        _dst.data_offset += dstride;
                        _src.data_offset += sstride;
                    }
                }
            }
            else
            {
                if (swap)
                {
                    for (int i = 0; i < n; i++)
                    {
                        numpyinternal.swapvalue(_dst, _dst.data_offset, divsize);
                        _dst.data_offset += dstride;
                    }
                }
                else
                {
                    _dst.data_offset += dstride * n;
                }

            }
        }

        //public void flat_copyinto(VoidPtr dest, int outstride, NpyArrayIterObject srcIter, npy_intp instride, npy_intp N, npy_intp destOffset)
        //{

        //    while (srcIter.index < srcIter.size)
        //    {
        //        strided_byte_copy(dest, outstride, srcIter.dataptr, instride, N, outstride);

        //        dest.data_offset += destOffset;
        //        numpyinternal.NpyArray_ITER_NEXT(srcIter);
        //    }
        //}

   

        public void memmove_init(VoidPtr dest, VoidPtr src)
        {
            this.dst = dest;
            this.src = src;
            this.da = dest.datap as T[];
            this.sa = src.datap as T[];

            this.eldiv = GetDivSize(dest);

            if (src.type_num == dest.type_num)
                this.isSameType = true;
            else
                this.isSameType = false;
        }

        public void memcpy(npy_intp dest_offset, npy_intp src_offset, npy_intp len)
        {
            if (isSameType)
            {
                long ElementCount = len >> eldiv;
                long sOffset = src_offset >> eldiv;
                long dOffset = dest_offset >> eldiv;

                if (ElementCount == 1)
                {
                    da[dOffset] = sa[sOffset];
                }
                else
                {
                    Array.Copy(sa, sOffset, da, dOffset, ElementCount);
                }
            }
            else
            {
                MemCopy.MemCpy(dst, dest_offset, src, src_offset, len);
                return;
            }
  

        }

        public void memmove(npy_intp dest_offset, npy_intp src_offset, npy_intp len)
        {
            if (isSameType)
            {
                long ElementCount = len >> eldiv;
                long sOffset = src_offset >> eldiv;
                long dOffset = dest_offset >> eldiv;

                if (ElementCount == 1)
                {
                    da[dOffset] = sa[sOffset];
                }
                else
                {
                    lock(this)
                    {
                        if (Temp == null || Temp.Length < ElementCount)
                        {
                            Temp = new T[ElementCount];
                        }

                        Array.Copy(sa, sOffset, Temp, 0, ElementCount);
                        Array.Copy(Temp, 0, da, dOffset, ElementCount);
                    }
    
                }
            }
            else
            {
                VoidPtr Temp = new VoidPtr(new byte[len]);
                MemCopy.MemCpy(Temp, 0, src, src_offset, len);
                MemCopy.MemCpy(dst, dest_offset, Temp, 0, len);
                return;
            }


        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void memmoveitem(npy_intp dest_offset, npy_intp src_offset)
        {
            da[dest_offset >> eldiv] = sa[src_offset >> eldiv];
        }


    }
}
