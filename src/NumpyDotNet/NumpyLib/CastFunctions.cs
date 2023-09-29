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

using NumpyLib;
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
 


    class CastFunctions
    {
        public static void DefaultCastFunction(VoidPtr Src, VoidPtr Dest, npy_intp srclen, NpyArray srcArray, NpyArray destArray)
        {
            CastFunctions.DefaultCastFunction(Src, 0, Dest, 0, srclen);
            return;
        }

        private static npy_intp AdjustedOffset(VoidPtr vp)
        {
            var Handler = numpyAPI.GetArrayHandler(vp.type_num);
            return vp.data_offset >> Handler.ItemDiv;
        }

        #region First Level Sorting
        public static void DefaultCastFunction(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            dest_offset += AdjustedOffset(Dest);
            src_offset += AdjustedOffset(Src);

            switch (Dest.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    DefaultCastsToBool(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    DefaultCastsToByte(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    DefaultCastsToUByte(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    DefaultCastsToInt16(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    DefaultCastsToUInt16(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    DefaultCastsToInt32(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    DefaultCastsToUInt32(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    DefaultCastsToInt64(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    DefaultCastsToUInt64(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    DefaultCastsToFloat(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    DefaultCastsToDouble(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    DefaultCastsToDecimal(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    DefaultCastsToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEXREAL:
                    DefaultCastsToComplexReal(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEXIMAG:
                    DefaultCastsToComplexImag(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    DefaultCastsToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    DefaultCastsToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    DefaultCastsToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }

            return;
        }

    
        #endregion

        #region Second Level Sorting
        static void DefaultCastsToBool(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    CastObjectToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringsToBools(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }
        }

        static void DefaultCastsToByte(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }
        }

        static void DefaultCastsToUByte(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToUBytes(Src, src_offset, Dest, dest_offset, srclen);
                    break;

            }
        }

        static void DefaultCastsToInt16(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }
        }

        static void DefaultCastsToUInt16(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToUInt16s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }
        }

        static void DefaultCastsToInt32(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }
        }

        static void DefaultCastsToUInt32(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToUInt32s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }
        }

        static void DefaultCastsToInt64(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }
        }

        static void DefaultCastsToUInt64(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToUInt64s(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }
        }

        static void DefaultCastsToFloat(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToFloats(Src, src_offset, Dest, dest_offset, srclen);
                    break;

            }
        }

        static void DefaultCastsToDouble(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToDoubles(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }


        }

        static void DefaultCastsToDecimal(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToDecimals(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }


        }

        static void DefaultCastsToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToComplex(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }


        }

        static void DefaultCastsToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToBigInt(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }


        }

        static void DefaultCastsToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    CastObjectToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToObject(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }


        }


        static void DefaultCastsToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastBoolsToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastBytesToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    CastUBytesToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastInt16sToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastUInt16sToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastInt32sToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastUInt32sToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastInt64sToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastUInt64sToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastFloatsToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastDoublesToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    CastDecimalsToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    CastBigIntToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    CastObjectToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_STRING:
                    CastStringToString(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }


        }


        static void DefaultCastsToComplexReal(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                 case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToComplexReal(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }

        }

        static void DefaultCastsToComplexImag(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_COMPLEX:
                    CastComplexToComplexImag(Src, src_offset, Dest, dest_offset, srclen);
                    break;
            }


        }

        #endregion

        #region Boolean specific casts
        static void CastBoolsToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as bool[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (bool)(s[index + src_offset]);
            //    index++;
            //}
        }
        static void CastBytesToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != 0 ? true : false);
                index++;
            }
        }
        static void CastUBytesToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != 0 ? true : false);
                index++;
            }
        }
        static void CastInt16sToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != 0 ? true : false);
                index++;
            }
        }
        static void CastUInt16sToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != 0 ? true : false);
                index++;
            }
        }
        static void CastInt32sToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != 0 ? true : false);
                index++;
            }
        }
        static void CastUInt32sToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != 0 ? true : false);
                index++;
            }
        }
        static void CastInt64sToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != 0 ? true : false);
                index++;
            }
        }
        static void CastUInt64sToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != 0 ? true : false);
                index++;
            }
        }
        static void CastFloatsToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != 0 ? true : false);
                index++;
            }
        }
        static void CastDoublesToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != 0 ? true : false);
                index++;
            }
        }

        static void CastDecimalsToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != 0 ? true : false);
                index++;
            }
        }
        static void CastComplexToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset].Real != 0 ? true : false);
                index++;
            }
        }
        static void CastBigIntToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != 0 ? true : false);
                index++;
            }
        }
        static void CastObjectToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Object[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != null ? true : false);
                index++;
            }
        }
        static void CastStringsToBools(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as bool[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (bool)(s[index + src_offset] != null ? true : false);
                index++;
            }
        }
        #endregion

        #region Bytes specific casts
        static void CastBoolsToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (sbyte)(s[index + src_offset] == true ? 1 : 0);
                index++;
            }
        }
        static void CastBytesToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as sbyte[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (sbyte)(s[index + src_offset]);
            //    index++;
            //}
        }
        static void CastUBytesToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (sbyte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt16sToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (sbyte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt16sToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (sbyte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt32sToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (sbyte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt32sToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (sbyte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt64sToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (sbyte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt64sToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (sbyte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastFloatsToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (sbyte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDoublesToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (sbyte)(s[index + src_offset]);
                index++;
            }
        }

        static void CastDecimalsToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToSByte(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = SByte.MinValue;
                }

                index++;
            }
        }
        static void CastComplexToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToSByte(s[index + src_offset].Real);
                }
                catch
                {
                    d[index + dest_offset] = SByte.MinValue;
                }

                index++;
            }
        }
        static void CastBigIntToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = (sbyte)s[index + src_offset];
                }
                catch
                {
                    d[index + dest_offset] = SByte.MinValue;
                }

                index++;
            }
        }
        static void CastStringToBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as sbyte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = sbyte.Parse(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = 0;
                }

                index++;
            }
        }
        #endregion

        #region UBytes specific casts
        static void CastBoolsToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (byte)(s[index + src_offset] == true ? 1 : 0);
                index++;
            }
        }
        static void CastBytesToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (byte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUBytesToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as byte[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (byte)(s[index + src_offset]);
            //    index++;
            //}
        }
        static void CastInt16sToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (byte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt16sToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (byte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt32sToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (byte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt32sToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (byte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt64sToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (byte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt64sToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (byte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastFloatsToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (byte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDoublesToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (byte)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDecimalsToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToByte(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = Byte.MinValue;
                }

                index++;
            }
        }
        static void CastComplexToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToByte(s[index + src_offset].Real);
                }
                catch
                {
                    d[index + dest_offset] = Byte.MinValue;
                }

                index++;
            }
        }
        static void CastBigIntToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = (Byte)s[index + src_offset];
                }
                catch
                {
                    d[index + dest_offset] = Byte.MinValue;
                }

                index++;
            }
        }
        static void CastStringToUBytes(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as byte[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Byte.Parse(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = 0;
                }

                index++;
            }
        }
        #endregion

        #region Int16 specific casts
        static void CastBoolsToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int16)(s[index + src_offset] == true ? 1 : 0);
                index++;
            }
        }
        static void CastBytesToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUBytesToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt16sToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as Int16[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (Int16)(s[index + src_offset]);
            //    index++;
            //}
        }
        static void CastUInt16sToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt32sToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt32sToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt64sToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt64sToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastFloatsToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDoublesToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDecimalsToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToInt16(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = Int16.MinValue;
                }
                index++;
            }
        }
        static void CastComplexToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToInt16(s[index + src_offset].Real);
                }
                catch
                {
                    d[index + dest_offset] = Int16.MinValue;
                }
                index++;
            }
        }
        static void CastBigIntToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] =  (Int16)s[index + src_offset];
                }
                catch
                {
                    d[index + dest_offset] = Int16.MinValue;
                }
                index++;
            }
        }
        static void CastStringToInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as Int16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Int16.Parse(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = 0;
                }
                index++;
            }
        }
        #endregion

        #region UInt16 specific casts
        static void CastBoolsToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt16)(s[index + src_offset] == true ? 1 : 0);
                index++;
            }
        }
        static void CastBytesToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUBytesToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt16sToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt16sToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as UInt16[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (UInt16)(s[index + src_offset]);
            //    index++;
            //}
        }
        static void CastInt32sToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt32sToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt64sToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt64sToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastFloatsToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDoublesToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt16)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDecimalsToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToUInt16(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = UInt16.MinValue;
                }
                index++;
            }
        }
        static void CastComplexToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToUInt16(s[index + src_offset].Real);
                }
                catch
                {
                    d[index + dest_offset] = UInt16.MinValue;
                }
                index++;
            }
        }
        static void CastBigIntToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = (UInt16)s[index + src_offset];
                }
                catch
                {
                    d[index + dest_offset] = UInt16.MinValue;
                }
                index++;
            }
        }
        static void CastStringToUInt16s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as UInt16[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = UInt16.Parse(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = 0;
                }
                index++;
            }
        }
        #endregion

        #region Int32 specific casts
        static void CastBoolsToInt32s(VoidPtr Src, npy_intp src_offset,  VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int32)(s[index + src_offset] == true ? 1 : 0);
                index++;
            }
        }
        static void CastBytesToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int32)s[index + src_offset];
                index++;
            }
        }
        static void CastUBytesToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int32)s[index + src_offset];
                index++;
            }
        }

        static void CastInt16sToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int32)s[index + src_offset];
                index++;
            }
        }
        static void CastUInt16sToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int32)s[index + src_offset];
                index++;
            }
        }
        static void CastInt32sToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as Int32[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (Int32)s[index + src_offset];
            //    index++;
            //}
        }
        static void CastUInt32sToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int32)s[index + src_offset];
                index++;
            }
        }
        static void CastInt64sToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int32)s[index + src_offset];
                index++;
            }
        }
        static void CastUInt64sToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int32)s[index + src_offset];
                index++;
            }
        }
        static void CastFloatsToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int32)s[index + src_offset];
                index++;
            }
        }
        static void CastDoublesToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int32)s[index + src_offset];
                index++;
            }
        }
        static void CastDecimalsToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToInt32(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = Int32.MinValue;
                }
                index++;
            }
        }
        static void CastComplexToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToInt32(s[index + src_offset].Real);
                }
                catch
                {
                    d[index + dest_offset] = Int32.MinValue;
                }
                index++;
            }
        }
        static void CastBigIntToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = (Int32)s[index + src_offset];
                }
                catch
                {
                    d[index + dest_offset] = Int32.MinValue;
                }
                index++;
            }
        }
        static void CastStringToInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as Int32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Int32.Parse(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = 0;
                }
                index++;
            }
        }
        #endregion

        #region UInt32 specific casts
        static void CastBoolsToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as UInt32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt32)(s[index + src_offset] == true ? 1 : 0);
                index++;
            }
        }
        static void CastBytesToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as UInt32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt32)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUBytesToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as UInt32[];


            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt32)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt16sToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as UInt32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt32)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt16sToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as UInt32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt32)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt32sToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as UInt32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt32)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt32sToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as UInt32[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (UInt32)(s[index + src_offset]);
            //    index++;
            //}
        }
        static void CastInt64sToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as UInt32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt32)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt64sToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as UInt32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt32)(s[index + src_offset]);
                index++;
            }
        }
        static void CastFloatsToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as UInt32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt32)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDoublesToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as UInt32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt32)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDecimalsToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as UInt32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToUInt32(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = UInt32.MinValue;
                }
                index++;
            }
        }
        static void CastComplexToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as UInt32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToUInt32(s[index + src_offset].Real);
                }
                catch
                {
                    d[index + dest_offset] = UInt32.MinValue;
                }
                index++;
            }
        }
        static void CastBigIntToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as UInt32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = (UInt32)s[index + src_offset];
                }
                catch
                {
                    d[index + dest_offset] = UInt32.MinValue;
                }
                index++;
            }
        }
        static void CastStringToUInt32s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as UInt32[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = UInt32.Parse(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = 0;
                }
                index++;
            }
        }
        #endregion

        #region Int64 specific casts
        static void CastBoolsToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int64)(s[index + src_offset] == true ? 1 : 0);
                index++;
            }
        }
        static void CastBytesToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUBytesToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt16sToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt16sToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt32sToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt32sToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt64sToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as Int64[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (Int64)(s[index + src_offset]);
            //    index++;
            //}
        }
        static void CastUInt64sToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int64)(s[index + src_offset]);
                index++;
            }

        }
        static void CastFloatsToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDoublesToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (Int64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDecimalsToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToInt64(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = Int64.MinValue;
                }
                index++;
            }
        }
        static void CastComplexToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToInt64(s[index + src_offset].Real);
                }
                catch
                {
                    d[index + dest_offset] = Int64.MinValue;
                }
                index++;
            }
        }
        static void CastBigIntToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = (Int64)s[index + src_offset];
                }
                catch
                {
                    d[index + dest_offset] = Int64.MinValue;
                }
                index++;
            }
        }
        static void CastStringToInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as Int64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Int64.Parse(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = 0;
                }
                index++;
            }
        }
        #endregion

        #region UInt64 specific casts
        static void CastBoolsToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt64)(s[index + src_offset] == true ? 1 : 0);
                index++;
            }
        }
        static void CastBytesToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUBytesToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt16sToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt16sToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt32sToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt32sToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt64sToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt64sToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as UInt64[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (UInt64)(s[index + src_offset]);
            //    index++;
            //}
        }
        static void CastFloatsToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDoublesToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (UInt64)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDecimalsToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToUInt64(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = UInt64.MinValue;
                }

                index++;
            }
        }
        static void CastComplexToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToUInt64(s[index + src_offset].Real);
                }
                catch
                {
                    d[index + dest_offset] = UInt64.MinValue;
                }

                index++;
            }
        }
        static void CastBigIntToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = (UInt64)s[index + src_offset];
                }
                catch
                {
                    d[index + dest_offset] = UInt64.MinValue;
                }

                index++;
            }
        }
        static void CastStringToUInt64s(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as UInt64[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = UInt64.Parse(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = 0;
                }

                index++;
            }
        }
        #endregion

        #region Float specific casts
        static void CastBoolsToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (float)(s[index + src_offset] == true ? 1 : 0);
                index++;
            }
        }
        static void CastBytesToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (float)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUBytesToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (float)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt16sToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (float)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt16sToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (float)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt32sToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (float)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt32sToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (float)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt64sToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (float)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt64sToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (float)(s[index + src_offset]);
                index++;
            }
        }
        static void CastFloatsToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as float[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (float)(s[index + src_offset]);
            //    index++;
            //}
        }
        static void CastDoublesToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (float)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDecimalsToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToSingle(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = Single.MinValue;
                }

                index++;
            }
        }
        static void CastComplexToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToSingle(s[index + src_offset].Real);
                }
                catch
                {
                    d[index + dest_offset] = Single.MinValue;
                }

                index++;
            }
        }
        static void CastBigIntToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = (float)s[index + src_offset];
                }
                catch
                {
                    d[index + dest_offset] = Single.MinValue;
                }

                index++;
            }
        }
        static void CastStringToFloats(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as float[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = float.Parse(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = 0f;
                }

                index++;
            }
        }
        #endregion

        #region Double specific casts
        static void CastBoolsToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (double)(s[index + src_offset] == true ? 1 : 0);
                index++;
            }
        }
        static void CastBytesToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (double)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUBytesToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (double)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt16sToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (double)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt16sToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (double)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt32sToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (double)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt32sToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (double)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt64sToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (double)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt64sToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (double)(s[index + src_offset]);
                index++;
            }
        }
        static void CastFloatsToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (double)(s[index + src_offset]);
                index++;
            }
        }
        static void CastDoublesToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as double[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (double)(s[index + src_offset]);
            //    index++;
            //}
        }
        static void CastDecimalsToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToDouble(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = Double.MinValue;
                }

                index++;
            }
        }
        static void CastComplexToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToDouble(s[index + src_offset].Real);
                }
                catch
                {
                    d[index + dest_offset] = Double.MinValue;
                }

                index++;
            }
        }
        static void CastBigIntToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = (double)s[index + src_offset];
                }
                catch
                {
                    d[index + dest_offset] = Double.MinValue;
                }

                index++;
            }
        }
        static void CastStringToDoubles(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = double.Parse(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = 0.0;
                }

                index++;
            }
        }
        #endregion

        #region Decimal specific casts
        static void CastBoolsToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (decimal)(s[index + src_offset] == true ? 1 : 0);
                index++;
            }
        }
        static void CastBytesToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (decimal)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUBytesToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (decimal)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt16sToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (decimal)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt16sToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (decimal)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt32sToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (decimal)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt32sToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (decimal)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt64sToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (decimal)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt64sToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (decimal)(s[index + src_offset]);
                index++;
            }
        }
        static void CastFloatsToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToDecimal(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = decimal.MinValue;
                }
                index++;
            }
        }
        static void CastDoublesToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToDecimal(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = decimal.MinValue;
                }
                index++;
            }
        }
        static void CastDecimalsToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as decimal[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (decimal)(s[index + src_offset]);
            //    index++;
            //}
        }
        static void CastComplexToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (decimal)(s[index + src_offset].Real);
                index++;
            }
        }
        static void CastBigIntToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (decimal)s[index + src_offset];
                index++;
            }
        }
        static void CastStringToDecimals(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as decimal[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = decimal.Parse(s[index + src_offset]);
                index++;
            }
        }
        #endregion

        #region Complex specific casts
        static void CastBoolsToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (System.Numerics.Complex)(s[index + src_offset] == true ? 1 : 0);
                index++;
            }
        }
        static void CastBytesToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (System.Numerics.Complex)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUBytesToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (System.Numerics.Complex)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt16sToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (System.Numerics.Complex)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt16sToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (System.Numerics.Complex)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt32sToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (System.Numerics.Complex)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt32sToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (System.Numerics.Complex)(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt64sToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (System.Numerics.Complex)(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt64sToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (System.Numerics.Complex)(s[index + src_offset]);
                index++;
            }
        }
        static void CastFloatsToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToDouble(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = double.MinValue;
                }
                index++;
            }
        }
        static void CastDoublesToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = Convert.ToDouble(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = double.MinValue;
                }
                index++;
            }
        }
        static void CastDecimalsToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = Convert.ToDouble(s[index + src_offset]);
                index++;
            }
        }
        static void CastComplexToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as System.Numerics.Complex[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (System.Numerics.Complex)(s[index + src_offset]);
            //    index++;
            //}
        }
        static void CastBigIntToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (double)s[index + src_offset];
                index++;
            }
        }
        static void CastStringToComplex(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as System.Numerics.Complex[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = double.Parse(s[index + src_offset]);
                index++;
            }
        }

        static void CastComplexToComplexReal(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (double)(s[index + src_offset].Real);
                index++;
            }
        }
        static void CastComplexToComplexImag(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as double[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = (double)(s[index + src_offset].Imaginary);
                index++;
            }
        }

        #endregion

        #region BigInt specific casts
        static void CastBoolsToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = new System.Numerics.BigInteger(s[index + src_offset] == true ? 1 : 0);
                index++;
            }
        }
        static void CastBytesToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = new System.Numerics.BigInteger(s[index + src_offset]);
                index++;
            }
        }
        static void CastUBytesToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = new System.Numerics.BigInteger(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt16sToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = new System.Numerics.BigInteger(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt16sToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = new System.Numerics.BigInteger(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt32sToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = new System.Numerics.BigInteger(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt32sToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = new System.Numerics.BigInteger(s[index + src_offset]);
                index++;
            }
        }
        static void CastInt64sToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = new System.Numerics.BigInteger(s[index + src_offset]);
                index++;
            }
        }
        static void CastUInt64sToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = new System.Numerics.BigInteger(s[index + src_offset]);
                index++;
            }
        }
        static void CastFloatsToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = new System.Numerics.BigInteger(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = new System.Numerics.BigInteger(float.MinValue);
                }
                index++;
            }
        }
        static void CastDoublesToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = new System.Numerics.BigInteger(s[index + src_offset]);
                }
                catch
                {
                    d[index + dest_offset] = new System.Numerics.BigInteger(double.MinValue);
                }
                index++;
            }
        }
        static void CastDecimalsToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = new System.Numerics.BigInteger(s[index + src_offset]);
                index++;
            }
        }
        static void CastComplexToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = new System.Numerics.BigInteger(s[index + src_offset].Real);
                index++;
            }
        }
        static void CastBigIntToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            Array.Copy(s, src_offset, d, dest_offset, srclen);

            //npy_intp index = 0;
            //while (srclen-- > 0)
            //{
            //    d[index + dest_offset] = (System.Numerics.BigInteger)(s[index + src_offset]);
            //    index++;
            //}
        }
        static void CastStringToBigInt(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as System.Numerics.BigInteger[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] =  System.Numerics.BigInteger.Parse(s[index + src_offset]);
                index++;
            }
        }

        #endregion

        #region Object specific casts
        static void CastBoolsToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset] == true ? 1 : 0;
                index++;
            }
        }
        static void CastBytesToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset];
                index++;
            }
        }
        static void CastUBytesToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset];
                index++;
            }
        }
        static void CastInt16sToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset];
                index++;
            }
        }
        static void CastUInt16sToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset];
                index++;
            }
        }
        static void CastInt32sToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset];
                index++;
            }
        }
        static void CastUInt32sToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset];
                index++;
            }
        }
        static void CastInt64sToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset];
                index++;
            }
        }
        static void CastUInt64sToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset];
                index++;
            }
        }
        static void CastFloatsToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = s[index + src_offset];
                }
                catch
                {
                    d[index + dest_offset] = float.MinValue;
                }
                index++;
            }
        }
        static void CastDoublesToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = s[index + src_offset];
                }
                catch
                {
                    d[index + dest_offset] = double.MinValue;
                }
                index++;
            }
        }
        static void CastDecimalsToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset];
                index++;
            }
        }
        static void CastComplexToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset];
                index++;
            }
        }
        static void CastBigIntToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset];
                index++;
            }
        }
        static void CastObjectToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as object[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset];
                index++;
            }
        }

        static void CastStringToObject(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.String[];
            var d = Dest.datap as object[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset];
                index++;
            }
        }

        #endregion

        #region String specific casts
        static void CastBoolsToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as bool[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }
        static void CastBytesToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as sbyte[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }
        static void CastUBytesToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as byte[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }
        static void CastInt16sToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int16[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }
        static void CastUInt16sToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt16[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }
        static void CastInt32sToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int32[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }
        static void CastUInt32sToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt32[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }
        static void CastInt64sToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as Int64[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }
        static void CastUInt64sToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as UInt64[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }
        static void CastFloatsToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as float[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = s[index + src_offset].ToString();
                }
                catch
                {
                    d[index + dest_offset] = float.MinValue.ToString();
                }
                index++;
            }
        }
        static void CastDoublesToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as double[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                try
                {
                    d[index + dest_offset] = s[index + src_offset].ToString();
                }
                catch
                {
                    d[index + dest_offset] = double.MinValue.ToString();
                }
                index++;
            }
        }
        static void CastDecimalsToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as decimal[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }
        static void CastComplexToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.Complex[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }
        static void CastBigIntToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as System.Numerics.BigInteger[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }
        static void CastObjectToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as object[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }
        static void CastStringToString(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            var s = Src.datap as string[];
            var d = Dest.datap as string[];

            npy_intp index = 0;
            while (srclen-- > 0)
            {
                d[index + dest_offset] = s[index + src_offset].ToString();
                index++;
            }
        }

        #endregion

        #region Experimental Generic code

        // This code was an experiment in using a smaller set of generic functions to do the work.
        // The code is functional and it seems to work but the performance was dramatically worse.
        // In testing, it took 5 times longer to execute the same amount of data processing.
        // The reason for this is the need to call the Convert.To functions to satisfy the generics.
        // Since this is the most highly used portion of the library, performance is critical.

        static void DefaultCastsToInt32Generics(VoidPtr Src, npy_intp src_offset, VoidPtr Dest, npy_intp dest_offset, npy_intp srclen)
        {
            switch (Src.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    CastToInt32s(Src.datap as bool[], src_offset, Dest.datap as Int32[], dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    CastToInt32s(Src.datap as byte[], src_offset, Dest.datap as Int32[], dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT16:
                    CastToInt32s(Src.datap as Int16[], src_offset, Dest.datap as Int32[], dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    CastToInt32s(Src.datap as UInt16[], src_offset, Dest.datap as Int32[], dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT32:
                    CastToInt32s(Src.datap as Int32[], src_offset, Dest.datap as Int32[], dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    CastToInt32s(Src.datap as UInt32[], src_offset, Dest.datap as Int32[], dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_INT64:
                    CastToInt32s(Src.datap as Int64[], src_offset, Dest.datap as Int32[], dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    CastToInt32s(Src.datap as UInt64[], src_offset, Dest.datap as Int32[], dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    CastToInt32s(Src.datap as float[], src_offset, Dest.datap as Int32[], dest_offset, srclen);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    CastToInt32s(Src.datap as Double[], src_offset, Dest.datap as Int32[], dest_offset, srclen);
                    break;
            }
        }


        static void CastToInt32s<T1>(T1[] Src, npy_intp src_offset, Int32[] Dest, npy_intp dest_offset, npy_intp srclen)
        {
            npy_intp index = 0;
            while (srclen-- > 0)
            {
                Dest[index + dest_offset] = Convert.ToInt32(Src[index + src_offset]);
                index++;
            }
        }
        static void CastToInt32s<T1, T2>(T1[] Src, npy_intp src_offset, T2[] Dest, npy_intp dest_offset, npy_intp srclen)
        {
            npy_intp index = 0;
            while (srclen-- > 0)
            {
                Dest[index + dest_offset] = (T2)Convert.ChangeType(Src[index + src_offset], typeof(T2));
                index++;
            }
        }

        #endregion

    }
}
