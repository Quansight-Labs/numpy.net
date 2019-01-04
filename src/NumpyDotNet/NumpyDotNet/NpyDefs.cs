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
using NumpyLib;
#if NPY_INTP_64
using npy_intp = System.Int64;
using npy_ucs4 = System.Int64;
using NpyArray_UCS4 = System.UInt64;
#else
using npy_intp = System.Int32;
using npy_ucs4 = System.Int32;
using NpyArray_UCS4 = System.UInt32;
#endif

namespace NumpyDotNet {
    public class NpyDefs {
        #region ConstantDefs

        public const int NPY_VALID_MAGIC = 1234567;

 
        internal const NPY_TYPES DefaultType = NPY_TYPES.NPY_DOUBLE;
        #if NPY_INTP_64
        internal static readonly NPY_TYPES NPY_INTP = NpyCoreApi.TypeOf_Int64;
        internal static readonly NPY_TYPES NPY_UINTP = NpyCoreApi.TypeOf_UInt64;
        #else
        internal static readonly NPY_TYPES NPY_INTP = NpyCoreApi.TypeOf_Int32;
        internal static readonly NPY_TYPES NPY_UINTP = NpyCoreApi.TypeOf_UInt32;
        #endif

        internal const int NPY_NTYPES = (int)NPY_TYPES.NPY_NTYPES;

        public enum NPY_COMPARE_OP {
            NPY_LT = 0,
            NPY_LE = 1,
            NPY_EQ = 2,
            NPY_NE = 3,
            NPY_GT = 4,
            NPY_GE = 5,
        };

        internal const int NPY_MAXDIMS = 32;
        internal const int NPY_MAXARGS = 32;

        #endregion


   
        #region umath errors

        public const int NPY_BUFSIZE = 10000;
        public const int NPY_MIN_BUFSIZE = 2 * sizeof(double);
        public const int NPY_MAX_BUFSIZE = NPY_MIN_BUFSIZE * 1000000;

        public enum NPY_UFUNC_FPE
        {
            DIVIDEBYZERO = 1,
            OVERFLOW = 2,
            UNDERFLOW = 4,
            INVALID = 8
        }

        public enum NPY_UFUNC_ERR
        {
            IGNORE = 0,
            WARN = 1,
            RAISE = 2,
            CALL = 3,
            PRINT = 4,
            LOG = 5
        }

        public enum NPY_UFUNC_MASK
        {
            DIVIDEBYZERO = 0x07,
            OVERFLOW = 0x3f,
            UNDERFLOW = 0x1ff,
            INVALID = 0xfff
        }

        public enum NPY_UFUNC_SHIFT
        {
            DIVIDEBYZERO = 0,
            OVERFLOW = 3,
            UNDERFLOW = 6,
            INVALID = 9
        }

        public enum NPY_DATETIMEUNIT : int
        {
            NPY_FR_Y=0,
            NPY_FR_M,
            NPY_FR_W,
            NPY_FR_B,
            NPY_FR_D,
            NPY_FR_h,
            NPY_FR_m,
            NPY_FR_s,
            NPY_FR_ms,
            NPY_FR_us,
            NPY_FR_ns,
            NPY_FR_ps,
            NPY_FR_fs,
            NPY_FR_as
        }

        public const int NPY_UFUNC_ERR_DEFAULT = 0;
        public const int NPY_UFUNC_ERR_DEFAULT2 =
            ((int)NPY_UFUNC_ERR.PRINT << (int)NPY_UFUNC_SHIFT.DIVIDEBYZERO) +
            ((int)NPY_UFUNC_ERR.PRINT << (int)NPY_UFUNC_SHIFT.OVERFLOW) +
            ((int)NPY_UFUNC_ERR.PRINT << (int)NPY_UFUNC_SHIFT.INVALID);

        #endregion


        #region Type functions

        public static bool IsBool(NPY_TYPES type) {
            return type == NPY_TYPES.NPY_BOOL;
        }

        public static bool IsUnsigned(NPY_TYPES type) {
            return type == NPY_TYPES.NPY_UBYTE || type == NPY_TYPES.NPY_USHORT ||
                type == NPY_TYPES.NPY_UINT || type == NPY_TYPES.NPY_ULONG ||
                type == NPY_TYPES.NPY_ULONGLONG;
        }

        public static bool IsSigned(NPY_TYPES type) {
            return type == NPY_TYPES.NPY_BYTE || type == NPY_TYPES.NPY_SHORT ||
                type == NPY_TYPES.NPY_INT || type == NPY_TYPES.NPY_LONG ||
                type == NPY_TYPES.NPY_LONGLONG;
        }

        public static bool IsInteger(NPY_TYPES type) {
            return NPY_TYPES.NPY_BYTE <= type && type <= NPY_TYPES.NPY_ULONGLONG;
        }

        public static bool IsFloat(NPY_TYPES type) {
            return NPY_TYPES.NPY_FLOAT <= type && type <= NPY_TYPES.NPY_LONGDOUBLE;
        }

        public static bool IsNumber(NPY_TYPES type) {
            return type <= NPY_TYPES.NPY_CLONGDOUBLE;
        }

        public static bool IsString(NPY_TYPES type) {
            return type == NPY_TYPES.NPY_STRING || type == NPY_TYPES.NPY_UNICODE;
        }

        public static bool IsComplex(NPY_TYPES type) {
            return NPY_TYPES.NPY_CFLOAT <= type && type <= NPY_TYPES.NPY_CLONGDOUBLE;
        }

        public static bool IsPython(NPY_TYPES type) {
            return type == NPY_TYPES.NPY_LONG || type == NPY_TYPES.NPY_DOUBLE ||
                type == NPY_TYPES.NPY_CDOUBLE || type == NPY_TYPES.NPY_BOOL ||
                type == NPY_TYPES.NPY_OBJECT;
        }

        public static bool IsFlexible(NPY_TYPES type) {
            return NPY_TYPES.NPY_STRING <= type && type <= NPY_TYPES.NPY_VOID;
        }

        public static bool IsDatetime(NPY_TYPES type) {
            return NPY_TYPES.NPY_DATETIME <= type && type <= NPY_TYPES.NPY_TIMEDELTA;
        }

        public static bool IsUserDefined(NPY_TYPES type) {
            return NPY_TYPES.NPY_USERDEF <= type &&
                (int)type <= (int)NPY_TYPES.NPY_USERDEF + 0; // TODO: Need GetNumUserTypes
        }

        public static bool IsExtended(NPY_TYPES type) {
            return IsFlexible(type) || IsUserDefined(type);
        }

        public static bool IsNativeByteOrder(char endian)
        {
            return BitConverter.IsLittleEndian;
        }

        #endregion

    }
}
