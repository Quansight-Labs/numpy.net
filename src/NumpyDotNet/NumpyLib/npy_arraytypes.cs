﻿/*
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
    internal partial class numpyinternal
    {
        const int _MAX_LETTER = 128;
        static byte []_npy_letter_to_num = new byte[_MAX_LETTER];

        static int NotSupportedSizeYet = 8;

        static NpyArray_Descr npy_BOOL_Descr = new NpyArray_Descr(NPY_TYPES.NPY_BOOL) {  elsize = 1, kind = 'b'};
        static NpyArray_Descr npy_BYTE_Descr = new NpyArray_Descr(NPY_TYPES.NPY_BYTE) { elsize = 1, kind = 'i' };
        static NpyArray_Descr npy_UBYTE_Descr = new NpyArray_Descr(NPY_TYPES.NPY_UBYTE) {  elsize = 1, kind = 'u' };
        static NpyArray_Descr npy_SHORT_Descr = new NpyArray_Descr(NPY_TYPES.NPY_SHORT) {  elsize = 2, kind = 'i' };
        static NpyArray_Descr npy_USHORT_Descr = new NpyArray_Descr(NPY_TYPES.NPY_USHORT) {  elsize = 2, kind = 'u' };
        static NpyArray_Descr npy_INT_Descr = new NpyArray_Descr(NPY_TYPES.NPY_INT) {  elsize = 4, kind = 'i' };
        static NpyArray_Descr npy_UINT_Descr = new NpyArray_Descr(NPY_TYPES.NPY_UINT) { elsize = 4, kind = 'u' };
        static NpyArray_Descr npy_LONG_Descr = new NpyArray_Descr(NPY_TYPES.NPY_LONG) {  elsize = 8, kind = 'i' };
        static NpyArray_Descr npy_ULONG_Descr = new NpyArray_Descr(NPY_TYPES.NPY_ULONG) {  elsize = 8, kind = 'u' };
        static NpyArray_Descr npy_LONGLONG_Descr = new NpyArray_Descr(NPY_TYPES.NPY_LONGLONG) {  elsize = NotSupportedSizeYet, kind = 'u' };
        static NpyArray_Descr npy_ULONGLONG_Descr = new NpyArray_Descr(NPY_TYPES.NPY_ULONGLONG) {  elsize = NotSupportedSizeYet, kind = 'u' };
        static NpyArray_Descr npy_FLOAT_Descr = new NpyArray_Descr(NPY_TYPES.NPY_FLOAT) {  elsize = 4, kind = 'f' };
        static NpyArray_Descr npy_DOUBLE_Descr = new NpyArray_Descr(NPY_TYPES.NPY_DOUBLE) {  elsize = 8, kind = 'f' };
        static NpyArray_Descr npy_DECIMAL_Descr = new NpyArray_Descr(NPY_TYPES.NPY_DECIMAL) {  elsize = sizeof(decimal), kind = 'c' };
        static NpyArray_Descr npy_LONGDOUBLE_Descr = new NpyArray_Descr(NPY_TYPES.NPY_DECIMAL) {  elsize = NotSupportedSizeYet, kind = 'u' };
        static NpyArray_Descr npy_CFLOAT_Descr = new NpyArray_Descr(NPY_TYPES.NPY_CFLOAT) {  elsize = NotSupportedSizeYet, kind = 'c' };
        static NpyArray_Descr npy_CDOUBLE_Descr = new NpyArray_Descr(NPY_TYPES.NPY_CDOUBLE) {  elsize = NotSupportedSizeYet, kind = 'c' };
        static NpyArray_Descr npy_CLONGDOUBLE_Descr = new NpyArray_Descr(NPY_TYPES.NPY_CLONGDOUBLE) {  elsize = NotSupportedSizeYet, kind = 'u' };
        static NpyArray_Descr npy_DATETIME_Descr = new NpyArray_Descr(NPY_TYPES.NPY_DATETIME) {  elsize = 8, kind = 'M' };
        static NpyArray_Descr npy_TIMEDELTA_Descr = new NpyArray_Descr(NPY_TYPES.NPY_TIMEDELTA) {  elsize = 8, kind = 'm' };
        static NpyArray_Descr npy_OBJECT_Descr = new NpyArray_Descr(NPY_TYPES.NPY_OBJECT) {  elsize = NotSupportedSizeYet, kind = 'O' };
        static NpyArray_Descr npy_STRING_Descr = new NpyArray_Descr(NPY_TYPES.NPY_STRING) {  elsize = NotSupportedSizeYet, kind = 'S' };
        static NpyArray_Descr npy_UNICODE_Descr = new NpyArray_Descr(NPY_TYPES.NPY_UNICODE) {  elsize = NotSupportedSizeYet, kind = 'U' };
        static NpyArray_Descr npy_VOID_Descr = new NpyArray_Descr(NPY_TYPES.NPY_VOID) {  elsize = NotSupportedSizeYet, kind = 'V' };

        static NpyArray_Descr[] _builtin_descrs = new NpyArray_Descr[]
        {
            npy_BOOL_Descr,
            npy_BYTE_Descr,
            npy_UBYTE_Descr,
            npy_SHORT_Descr,
            npy_USHORT_Descr,
            npy_INT_Descr,
            npy_UINT_Descr,
            npy_LONG_Descr,
            npy_ULONG_Descr,
            npy_LONGLONG_Descr,
            npy_ULONGLONG_Descr,
            npy_FLOAT_Descr,
            npy_DOUBLE_Descr,
            npy_DECIMAL_Descr,
            npy_LONGDOUBLE_Descr,
            npy_CFLOAT_Descr,
            npy_CDOUBLE_Descr,
            npy_CLONGDOUBLE_Descr,
            npy_DATETIME_Descr,
            npy_TIMEDELTA_Descr,
            npy_OBJECT_Descr,
            npy_STRING_Descr,
            npy_UNICODE_Descr,
            npy_VOID_Descr,
        };
  
        internal static NpyArray_Descr _get_builtin_descrs(int type)
        {
            var ret = _builtin_descrs[type];
            return ret;
        }

        internal static NpyArray_Descr NpyArray_DescrFromType(NPY_TYPES type)
        {
            NpyArray_Descr ret = null;

            if (type < NPY_TYPES.NPY_NTYPES)
            {
                ret = NpyArray_DescrNew(_get_builtin_descrs((int)type));
            }
            else if (type == NPY_TYPES.NPY_NOTYPE)
            {
                /*
                 * This needs to not raise an error so
                 * that NpyArray_DescrFromType(NpyArray_NOTYPE)
                 * works for backwards-compatible C-API
                 */
                return null;
            }
            else if ((type == NPY_TYPES.NPY_CHAR) || ((int)type == (int)NPY_TYPECHAR.NPY_CHARLTR))
            {
                ret = NpyArray_DescrNew(_get_builtin_descrs((int)NPY_TYPES.NPY_STRING));
                if (ret == null)
                {
                    return null;
                }
                ret.elsize = 1;
                ret.type = (byte)NPY_TYPECHAR.NPY_CHARLTR;
                return ret;
            }
            else if (NpyTypeNum_ISUSERDEF((NPY_TYPES)type))
            {
                ret = NpyArray_UserDescrFromTypeNum(type);
            }
            else
            {
                NPY_TYPES num = NPY_TYPES.NPY_NTYPES;
                if ((int)type < _MAX_LETTER)
                {
                    num = (NPY_TYPES)_npy_letter_to_num[(int)type];
                }
                if (num >= NPY_TYPES.NPY_NTYPES)
                {
                    ret = null;
                }
                else
                {
                    ret = _get_builtin_descrs((int)num);
                }
            }
            if (ret == null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "Invalid data-type for array");
            }
            else
            {
                Npy_INCREF(ret);
            }

            /* Make sure dtype metadata is initialized for DATETIME */
            if (NpyTypeNum_ISDATETIME((NPY_TYPES)type))
            {
                if (ret.dtinfo == null)
                {
                    _init_datetime_descr(ret);
                }
            }

            return ret;
        }

        static void _init_datetime_descr(NpyArray_Descr descr)
        {
            NpyArray_DateTimeInfo dt_data;

            dt_data = new NpyArray_DateTimeInfo();
            dt_data._base = NPY_DATETIMEUNIT.NPY_FR_us;
            dt_data.num = 1;
            dt_data.den = 1;
            dt_data.events = 1;

            /* FIXME
             * There is no error check here and no way to indicate an error
             * until the metadata turns up NULL.
             */
            descr.dtinfo = dt_data;
        }

    }
}
