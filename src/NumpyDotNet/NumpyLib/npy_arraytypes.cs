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
using size_t = System.UInt64;


namespace NumpyLib
{
    internal partial class numpyinternal
    {
        const int _MAX_LETTER = 128;
        static byte[] _npy_letter_to_num = new byte[_MAX_LETTER];

        static int NotSupportedSizeYet = 8;
 

        internal static void _intialize_builtin_descrs()
        {
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_BOOL) { kind = 'b' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_BYTE) { kind = 'i' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_UBYTE) { kind = 'u' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_SHORT) { kind = 'i' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_USHORT) { kind = 'u' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_INT) { kind = 'i' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_UINT) { kind = 'u' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_LONG) { kind = 'i' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_ULONG) { kind = 'u' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_FLOAT) { kind = 'f' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_DOUBLE) { kind = 'f' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_DECIMAL) { kind = 'd' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_COMPLEX) { kind = 'c' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_BIGINT) { kind = 'I' });
            //_register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_DATETIME) { kind = 'M' });
            // _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_TIMEDELTA) { kind = 'm' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_OBJECT) { kind = 'O' });
            _register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_STRING) { kind = 'S' });
            //_register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_UNICODE) { kind = 'U' });
            //_register_builtin_descrs(new NpyArray_Descr(NPY_TYPES.NPY_VOID) { kind = 'V' });
        }

        private static void _register_builtin_descrs(NpyArray_Descr descr)
        {
            NPY_TYPES index = descr.type_num;

            _builtin_descrs[(int)index] = descr;
        }

        static NpyArray_Descr[] _builtin_descrs = new NpyArray_Descr[255];
   
        internal static NpyArray_Descr _get_builtin_descrs(NPY_TYPES type)
        {

            var ret = _builtin_descrs[(int)type];
            if (ret == null)
            {
                throw new Exception(string.Format("the type '{0}' is not registered as a built in descriptor", type.ToString()));
            }
            return ret;
        }

        internal static NpyArray_Descr NpyArray_DescrFromType(NPY_TYPES type)
        {
            NpyArray_Descr ret = null;

            if (type < NPY_TYPES.NPY_NTYPES)
            {
                ret = NpyArray_DescrNew(_get_builtin_descrs(type));
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
                    ret = _get_builtin_descrs(num);
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

        /************************************************************
        * A struct used by PyArray_CreateSortedStridePerm, new in 1.7.
        ************************************************************/

        class npy_stride_sort_item  : IComparable
        {
            public npy_intp perm;
            public npy_intp stride;

            /*
            * Sorts items so stride is descending, because C-order
            * is the default in the face of ambiguity.
            */
            public int CompareTo(object obj)
            {
                npy_stride_sort_item a = this;
                npy_stride_sort_item b = obj as npy_stride_sort_item;

                npy_intp astride = a.stride;
                npy_intp bstride = b.stride;

                /* Sort the absolute value of the strides */
                if (astride < 0)
                {
                    astride = -astride;
                }
                if (bstride < 0)
                {
                    bstride = -bstride;
                }

                if (astride == bstride)
                {
                    /*
                     * Make the qsort stable by next comparing the perm order.
                     * (Note that two perm entries will never be equal)
                     */
                    npy_intp aperm = a.perm;
                    npy_intp bperm = b.perm;
                    return (aperm < bperm) ? -1 : 1;
                }
                if (astride > bstride)
                {
                    return -1;
                }
                return 1;
            }


        }

        /*
         * When creating an auxiliary data struct, this should always appear
         * as the first member, like this:
         *
         * typedef struct {
         *     NpyAuxData base;
         *     double constant;
         * } constant_multiplier_aux_data;
         */

        /* Function pointers for freeing or cloning auxiliary data */
        delegate void NpyAuxData_FreeFunc(NpyAuxData x);
        delegate NpyAuxData NpyAuxData_CloneFunc(NpyAuxData x);

        class NpyAuxData
        {
            public NpyAuxData_FreeFunc free;
            public NpyAuxData_CloneFunc clone;

        };

        private static void NPY_AUXDATA_FREE(NpyAuxData auxdata)
        {
            if (auxdata != null)
            {
                auxdata.free(auxdata);
            }
        }


    }
}
