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
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
using npy_ucs4 = System.Int64;
#else
using npy_intp = System.Int32;
using npy_ucs4 = System.Int32;
#endif


namespace NumpyLib
{
    internal partial class numpyinternal
    {
        static numpyinternal()
        {
            DefaultArrayHandlers.Initialize();
        }

        internal static NpyArray_ArrFuncs GetArrFuncs(NPY_TYPES type_num)
        {
            NpyArray_ArrFuncs arrFuncs = new NpyArray_ArrFuncs()
            {
                ///
                // Functions to get and set items with standard Python types
                // -- not array scalars
                //

                getitem = null,
                setitem = null,
                //
                // Copy and/or swap data.  Memory areas may not overlap
                // Use memmove first if they might
                //
                copyswapn = NpyArray_CopySwapNFunc,
                copyswap = null,

                // 
                // Function to compare items
                // Can be null
                //
                compare = NpyArray_CompareFunc,

                ///
                //  Function to select largest
                //  Can be null
                // 
                argmax = NpyArray_ArgMaxFunc,

                ///
                //  Function to select largest
                //  Can be null
                // 
                argmin = NpyArray_ArgMinFunc,

                ///
                //  Function to compute dot product
                //  Can be null
                // 
                dotfunc = NpyArray_DotFunc,

                //
                // Function to scan an ASCII file and
                // place a single value plus possible separator
                // Can be null
                //
                scanfunc = NpyArray_ScanFunc,

                ///
                //  Function to read a single value from a string
                //  and adjust the pointer; Can be null
                // 
                fromstr = NpyArray_FromStrFunc,

                ///
                //  Function to determine if data is zero or not
                //  If null a default version is
                //  used at Registration time.
                // 
                nonzero = null,


                //
                // Used for arange.
                // Can be null.
                //
                fill = NpyArray_FillFunc,

                ///
                // Function to fill arrays with scalar values
                // Can be null
                // 
                fillwithscalar = NpyArray_FillWithScalarFunc,

                //
                // Array of PyArray_CastFuncsItem given cast functions to
                // user defined types. The array it terminated with PyArray_NOTYPE.
                // Can be null.
                //
                castfuncs = new List<NpyArray_CastFuncsItem>(),

                // 
                // Functions useful for generalizing
                // the casting rules.
                // Can be null;
                //
                scalarkind = NpyArray_ScalarKindFunc,
                cancastscalarkindto = null,
                cancastto = null,


                fastclip = NpyArray_FastClipFunc,
                fastputmask = NpyArray_FastPutmaskFunc,
                fasttake = null, // NpyArray_FastTakeFunc,
            };

            
            switch (type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    arrFuncs.copyswap = Common_copyswap<bool>;
                    arrFuncs.getitem = BOOL_GetItemFunc;
                    arrFuncs.setitem = BOOL_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_BYTE:
                    arrFuncs.copyswap = Common_copyswap<sbyte>;
                    arrFuncs.getitem = BYTE_GetItemFunc;
                    arrFuncs.setitem = BYTE_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    arrFuncs.copyswap = Common_copyswap<byte>;
                    arrFuncs.getitem = UBYTE_GetItemFunc;
                    arrFuncs.setitem = UBYTE_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_INT16:
                    arrFuncs.copyswap = Common_copyswap<Int16>;
                    arrFuncs.getitem = INT16_GetItemFunc;
                    arrFuncs.setitem = INT16_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_UINT16:
                    arrFuncs.copyswap = Common_copyswap<UInt16>;
                    arrFuncs.getitem = UINT16_GetItemFunc;
                    arrFuncs.setitem = UINT16_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_INT32:
                    arrFuncs.copyswap = Common_copyswap<Int32>;
                    arrFuncs.getitem = INT32_GetItemFunc;
                    arrFuncs.setitem = INT32_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_UINT32:
                    arrFuncs.copyswap = Common_copyswap<UInt32>;
                    arrFuncs.getitem = UINT32_GetItemFunc;
                    arrFuncs.setitem = UINT32_SetItemFunc;
                    arrFuncs.setitem = UINT32_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_INT64:
                    arrFuncs.copyswap = Common_copyswap<Int64>;
                    arrFuncs.getitem = INT64_GetItemFunc;
                    arrFuncs.setitem = INT64_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_UINT64:
                    arrFuncs.copyswap = Common_copyswap<UInt64>;
                    arrFuncs.getitem = UINT64_GetItemFunc;
                    arrFuncs.setitem = UINT64_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    arrFuncs.copyswap = Common_copyswap<float>;
                    arrFuncs.getitem = FLOAT_GetItemFunc;
                    arrFuncs.setitem = FLOAT_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_COMPLEXREAL:
                case NPY_TYPES.NPY_COMPLEXIMAG:
                    arrFuncs.copyswap = Common_copyswap<double>;
                    arrFuncs.getitem = DOUBLE_GetItemFunc;
                    arrFuncs.setitem = DOUBLE_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    arrFuncs.copyswap = Common_copyswap<decimal>;
                    arrFuncs.getitem = DECIMAL_GetItemFunc;
                    arrFuncs.setitem = DECIMAL_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    arrFuncs.copyswap = Common_copyswap<System.Numerics.Complex>;
                    arrFuncs.getitem = COMPLEX_GetItemFunc;
                    arrFuncs.setitem = COMPLEX_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    arrFuncs.copyswap = Common_copyswap<System.Numerics.BigInteger>;
                    arrFuncs.getitem = BIGINT_GetItemFunc;
                    arrFuncs.setitem = BIGINT_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    arrFuncs.copyswap = Common_copyswap<System.Object>;
                    arrFuncs.getitem = OBJECT_GetItemFunc;
                    arrFuncs.setitem = OBJECT_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_STRING:
                    arrFuncs.copyswap = Common_copyswap<System.String>;
                    arrFuncs.getitem = STRING_GetItemFunc;
                    arrFuncs.setitem = STRING_SetItemFunc;
                    break;

                default:
                    arrFuncs.copyswap = NpyArray_CopySwapFunc;
                    arrFuncs.getitem = NpyArray_GetItemFunc;
                    arrFuncs.setitem = NpyArray_SetItemFunc;
                    break;
            }

            for (int i = 0; i < arrFuncs.cast.Length; i++)
            {
                arrFuncs.cast[i] = CastFunctions.DefaultCastFunction;
            }

            for (int i = 0; i < arrFuncs.sort.Length; i++)
            {
                arrFuncs.sort[i] = NpyArray_SortFunc;
            }

            for (int i = 0; i < arrFuncs.argsort.Length; i++)
            {
                arrFuncs.argsort[i] = NpyArray_ArgSortFunc;
            }

            return arrFuncs;

        }

        internal static void NpyErr_SetString(npyexc_type et, string error, [CallerMemberName] string FunctionName = null)
        {
            if (NpyErr_SetString_callback != null)
            {
                NpyErr_SetString_callback(FunctionName, et, error);
            }
        }

        internal static bool NpyErr_Occurred([CallerMemberName] string FunctionName = null)
        {
            if (NpyErr_Occurred_callback != null)
            {
                return NpyErr_Occurred_callback(FunctionName);
            }

            return false;
        }

        internal static void NpyErr_Clear([CallerMemberName] string FunctionName = null)
        {
            if (NpyErr_Clear_callback != null)
            {
                NpyErr_Clear_callback(FunctionName);
            }

        }

        #region GetItemFunc
        internal static object DifferentSizes_GetItemFunc(npy_intp index, NpyArray npa)
        {
            // handles case of views mapped to different size arrays
            var DestArray = NpyDataMem_NEW(npa.ItemType, 1, false);
            MemCopy.MemCpy(DestArray, 0, npa.data, index, npa.ItemSize);
           
            return GetIndex(DestArray, 0);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long AdjustedIndex_GetItemFunction(npy_intp index, NpyArray npa, int dpLength)
        {
            long AdjustedIndex = (npa.data.data_offset + index) / npa.ItemSize;

            if (AdjustedIndex < 0)
            {
                AdjustedIndex = dpLength - Math.Abs(AdjustedIndex);
            }

            return AdjustedIndex;

        }

        internal static object BOOL_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                bool[] dp = npa.data.datap as bool[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }
            else
            {
                return DifferentSizes_GetItemFunc(index, npa);
            }
        }
        internal static object BYTE_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                sbyte[] dp = npa.data.datap as sbyte[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }
            else
            {
                return DifferentSizes_GetItemFunc(index, npa);
            }
        }
        internal static object UBYTE_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                byte[] dp = npa.data.datap as byte[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }
            else
            {
                return DifferentSizes_GetItemFunc(index, npa);
            }
        }
        internal static object INT16_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                Int16[] dp = npa.data.datap as Int16[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }
            else
            {
                return DifferentSizes_GetItemFunc(index, npa);
            }
        }
        internal static object UINT16_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                UInt16[] dp = npa.data.datap as UInt16[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }
            else
            {
                return DifferentSizes_GetItemFunc(index, npa);
            }
        }
        internal static object INT32_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                Int32[] dp = npa.data.datap as Int32[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }
            else
            {
                return DifferentSizes_GetItemFunc(index, npa);
            }
        }
        internal static object UINT32_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                UInt32[] dp = npa.data.datap as UInt32[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }
            else
            {
                return DifferentSizes_GetItemFunc(index, npa);
            }
        }
        internal static object INT64_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                Int64[] dp = npa.data.datap as Int64[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }
            else
            {
                return DifferentSizes_GetItemFunc(index, npa);
            }
        }
        internal static object UINT64_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                UInt64[] dp = npa.data.datap as UInt64[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }
            else
            {
                return DifferentSizes_GetItemFunc(index, npa);
            }
        }
        internal static object FLOAT_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                float[] dp = npa.data.datap as float[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }
            else
            {
                return DifferentSizes_GetItemFunc(index, npa);
            }
        }
        internal static object DOUBLE_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                double[] dp = npa.data.datap as double[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }
            else
            {
                return DifferentSizes_GetItemFunc(index, npa);
            }
        }
        internal static object DECIMAL_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                decimal[] dp = npa.data.datap as decimal[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }
            else
            {
                return DifferentSizes_GetItemFunc(index, npa);
            }
        }
        internal static object COMPLEX_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                System.Numerics.Complex[] dp = npa.data.datap as System.Numerics.Complex[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }

            return null;
        }
        internal static object BIGINT_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                System.Numerics.BigInteger[] dp = npa.data.datap as System.Numerics.BigInteger[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }

            return null;
        }
        internal static object OBJECT_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                Object[] dp = npa.data.datap as Object[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }

            return null;
        }
        internal static object STRING_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                String[] dp = npa.data.datap as String[];
                long AdjustedIndex = AdjustedIndex_GetItemFunction(index, npa, dp.Length);
                return dp[AdjustedIndex];
            }

            return null;
        }

        internal static object NpyArray_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                return DefaultArrayHandlers.GetArrayHandler(npa.ItemType).GetItem(npa.data, index);
            }
            else
            {
                return DefaultArrayHandlers.GetArrayHandler(npa.ItemType).GetItemDifferentType(npa.data, index, npa.ItemType, npa.ItemSize);
            }
        }
           
        #endregion

        #region SetItemFunc
        internal static int DifferentSizes_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            // handles case of views mapped to different size arrays
            var SrcArray = NpyDataMem_NEW(npa.ItemType, 1, false);
            SetIndex(SrcArray, 0, value);
            MemCopy.MemCpy(npa.data, index, SrcArray, 0, npa.ItemSize);
            return 1;
        }

        internal static long AdjustedIndex_SetItemFunction(npy_intp index, NpyArray npa, int dpLength)
        {
            long AdjustedIndex = (npa.data.data_offset + index) / npa.ItemSize;

            if (AdjustedIndex < 0)
            {
                AdjustedIndex = dpLength - Math.Abs(AdjustedIndex);
            }

            return AdjustedIndex;

        }

        internal static int BOOL_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                bool[] dp = npa.data.datap as bool[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (bool)value;
                return 1;
            }
            else
            {
                return DifferentSizes_SetItemFunc(index, value, npa);
            }
        }
        internal static int BYTE_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                sbyte[] dp = npa.data.datap as sbyte[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (sbyte)value;
                return 1;
            }
            else
            {
                return DifferentSizes_SetItemFunc(index, value, npa);
            }
        }
        internal static int UBYTE_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                byte[] dp = npa.data.datap as byte[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (byte)value;
                return 1;
            }
            else
            {
                return DifferentSizes_SetItemFunc(index, value, npa);
            }
        }
        internal static int INT16_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                Int16[] dp = npa.data.datap as Int16[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (Int16)value;
                return 1;
            }
            else
            {
                return DifferentSizes_SetItemFunc(index, value, npa);
            }
        }
        internal static int UINT16_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                UInt16[] dp = npa.data.datap as UInt16[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (UInt16)value;
                return 1;
            }
            else
            {
                return DifferentSizes_SetItemFunc(index, value, npa);
            }
        }
        internal static int INT32_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                Int32[] dp = npa.data.datap as Int32[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (Int32)value;
                return 1;
            }
            else
            {
                return DifferentSizes_SetItemFunc(index, value, npa);
            }
        }
        internal static int UINT32_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                UInt32[] dp = npa.data.datap as UInt32[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (UInt32)value;
                return 1;
            }
            else
            {
                return DifferentSizes_SetItemFunc(index, value, npa);
            }
        }
        internal static int INT64_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                Int64[] dp = npa.data.datap as Int64[];
                dp[AdjustedIndex_SetItemFunction(index, npa, dp.Length)] = (Int64)value;
                return 1;
            }
            else
            {
                return DifferentSizes_SetItemFunc(index, value, npa);
            }
        }
        internal static int UINT64_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                UInt64[] dp = npa.data.datap as UInt64[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (UInt64)value;
                return 1;
            }
            else
            {
                return DifferentSizes_SetItemFunc(index, value, npa);
            }
        }
        internal static int FLOAT_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                float[] dp = npa.data.datap as float[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (float)value;
                return 1;
            }
            else
            {
                return DifferentSizes_SetItemFunc(index, value, npa);
            }
        }
        internal static int DOUBLE_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                double[] dp = npa.data.datap as double[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (double)value;
                return 1;
            }
            else
            {
                return DifferentSizes_SetItemFunc(index, value, npa);
            }
        }
        internal static int DECIMAL_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                decimal[] dp = npa.data.datap as decimal[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (decimal)value;
                return 1;
            }
            else
            {
                return DifferentSizes_SetItemFunc(index, value, npa);
            }
        }

        internal static int COMPLEX_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                System.Numerics.Complex[] dp = npa.data.datap as System.Numerics.Complex[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (System.Numerics.Complex)value;
                return 1;
            }
            return 0;
        }

        internal static int BIGINT_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                System.Numerics.BigInteger[] dp = npa.data.datap as System.Numerics.BigInteger[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (System.Numerics.BigInteger)value;
                return 1;
            }
            return 0;
        }


        internal static int OBJECT_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                Object[] dp = npa.data.datap as Object[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (Object)value;
                return 1;
            }
            return 0;
        }


        internal static int STRING_SetItemFunc(npy_intp index, dynamic value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                String[] dp = npa.data.datap as String[];
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = value != null ? value.ToString() : null;
                return 1;
            }
            return 0;
        }




        internal static int NpyArray_SetItemFunc(npy_intp index, object value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                return DefaultArrayHandlers.GetArrayHandler(npa.ItemType).SetItem(npa.data, index, value);
            }
            else
            {
                return DefaultArrayHandlers.GetArrayHandler(npa.ItemType).SetItemDifferentType(npa.data, index, value);
            }
        }
        #endregion

        internal static void NpyArray_CopySwapNFunc(VoidPtr Dest, npy_intp dstride, VoidPtr Src, npy_intp sstride, npy_intp N, bool swap, NpyArray npa)
        {
            _default_copyswap(Dest, dstride, Src, sstride, N, swap, npa);
        }

        #region copyswap

        // kevin: todo: call this instead Common_copyswap and we might get a slight gain.
        internal static NpyArray_CopySwapFunc GetTestCopySwap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            switch (dest.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                {
                    var cs = new testcopyswap<bool>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_BYTE:
                {
                    var cs = new testcopyswap<sbyte>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_UBYTE:
                {
                    var cs = new testcopyswap<byte>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_INT16:
                {
                    var cs = new testcopyswap<Int16>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_UINT16:
                {
                    var cs = new testcopyswap<UInt16>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_INT32:
                {
                    var cs = new testcopyswap<Int32>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_UINT32:
                {
                    var cs = new testcopyswap<UInt32>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_INT64:
                {
                    var cs = new testcopyswap<Int64>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_UINT64:
                {
                    var cs = new testcopyswap<UInt64>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_FLOAT:
                {
                    var cs = new testcopyswap<float>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_DOUBLE:
                {
                    var cs = new testcopyswap<double>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_DECIMAL:
                {
                    var cs = new testcopyswap<decimal>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_COMPLEX:
                {
                    var cs = new testcopyswap<System.Numerics.Complex>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_BIGINT:
                {
                    var cs = new testcopyswap<System.Numerics.BigInteger>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_OBJECT:
                {
                    var cs = new testcopyswap<System.Object>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                case NPY_TYPES.NPY_STRING:
                {
                    var cs = new testcopyswap<System.String>(dest, Source, swap, arr);
                    return cs.copyswap;
                }
                default:
                    throw new Exception("Not handled data type");
            }
        }

   
        internal class testcopyswap<T>
        {
            bool swap;
            bool matching_type_num;
            int destSize;
            int srcSize;
            NpyArray result;
            T[] d;
            T[] s;

            public testcopyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
            {
                this.swap = swap;
                this.matching_type_num = dest.type_num == Source.type_num;
                this.destSize = GetTypeSize(dest);
                this.srcSize = GetTypeSize(Source);
                this.result = arr;
                d = dest.datap as T[];
                s = Source.datap as T[];
            }

            public void copyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
            {
                if (Source != null)
                {
                    if (!matching_type_num)
                    {
                        NpyArray_CopySwapFunc(dest, Source, swap, result);
                        return;
                    }

                    d[dest.data_offset / destSize] = s[Source.data_offset / srcSize];
                }

                if (swap)
                {
                    swapvalue(dest, result.ItemSize);
                }
     
            }
        }
  
        internal static void Common_copyswap<T>(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            if (Source != null)
            {
                if (dest.type_num != Source.type_num)
                {
                    NpyArray_CopySwapFunc(dest, Source, swap, arr);
                    return;
                }

                T[] d = dest.datap as T[];
                T[] s = Source.datap as T[];

                d[dest.data_offset / GetTypeSize(dest)] = s[Source.data_offset / GetTypeSize(Source)];
            }

            if (swap)
            {
                swapvalue(dest, arr.ItemSize);
            }

        }
        internal static void NpyArray_CopySwapFunc(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {

            if (arr == null)
            {
                return;
            }

            if (NpyArray_HASFIELDS(arr))
            {
                NpyDict_KVPair KVPair = new NpyDict_KVPair();
                NpyArray_Descr descr;
                NpyDict_Iter pos = new NpyDict_Iter();

                descr = arr.descr;
                NpyDict_IterInit(pos);


                while (NpyDict_IterNext(descr.fields, pos, KVPair))
                {
                    NpyArray_DescrField value = KVPair.value as NpyArray_DescrField;
                    string key = KVPair.key as string;

                    if (null != value.title && key.CompareTo(value.title) != 0)
                    {
                        continue;
                    }
                    arr.descr = value.descr;
                    value.descr.f.copyswap(dest + value.offset, Source + value.offset, swap, arr);
                }
                arr.descr = descr;
                return;
            }
            if (swap && arr.descr.subarray != null)
            {
                NpyArray_Descr descr = new NpyArray_Descr(NPY_TYPES.NPY_OBJECT);
                NpyArray_Descr newDescr = null;
                npy_intp num;
                int itemsize;

                descr = arr.descr;
                newDescr = descr.subarray._base;
                arr.descr = newDescr;
                itemsize = newDescr.elsize;
                num = descr.elsize / itemsize;

                newDescr.f.copyswapn(dest, itemsize, Source, itemsize, num, swap, arr);
                arr.descr = descr;
                return;
            }
            if (Source != null)
            {
                memcpy(dest, Source, NpyArray_ITEMSIZE(arr));
            }

            if (swap)
            {
                swapvalue(dest, NpyArray_ITEMSIZE(arr));
            }
            return;
        }
        #endregion

        internal static void swapvalue(VoidPtr dest, int ItemSize)
        {
            npy_intp item_offset = dest.data_offset / ItemSize;

            switch (dest.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    {
                        return;
                    }
                case NPY_TYPES.NPY_BYTE:
                    {
                        return;
                    }
                case NPY_TYPES.NPY_UBYTE:
                    {
                        return;
                    }
                case NPY_TYPES.NPY_INT16:
                    {
                        Int16[] bdata = dest.datap as Int16[];
                        Int16 value = bdata[item_offset];

                        var uvalue = (UInt16)value;
                        UInt16 swapped = (UInt16)
                            ((0x00FF) & (uvalue >> 8) 
                            |(0xFF00) & (uvalue << 8));
                        bdata[item_offset] = (Int16)swapped;

                        return;
                    }

                case NPY_TYPES.NPY_UINT16:
                    {
                        UInt16[] bdata = dest.datap as UInt16[];
                        UInt16 value = bdata[item_offset];

                        var uvalue = (UInt16)value;
                        UInt16 swapped = (UInt16)
                            ((0x00FF) & (uvalue >> 8) 
                            |(0xFF00) & (uvalue << 8));
                        bdata[item_offset] = (UInt16)swapped;

                        return;
                    }

                case NPY_TYPES.NPY_INT32:
                    {
                        Int32[] bdata = dest.datap as Int32[];
                        Int32 value = bdata[item_offset];

                        var uvalue = (UInt32)value;
                        UInt32 swapped =
                             ( (0x000000FF) & (uvalue >> 24)
                             | (0x0000FF00) & (uvalue >> 8)
                             | (0x00FF0000) & (uvalue << 8)
                             | (0xFF000000) & (uvalue << 24));
                        bdata[item_offset] = (Int32)swapped;
                        break;

                    }

                case NPY_TYPES.NPY_UINT32:
                    {
                        UInt32[] bdata = dest.datap as UInt32[];
                        UInt32 value = bdata[item_offset];

                        var uvalue = (UInt32)value;
                        UInt32 swapped =
                             ((0x000000FF) & (uvalue >> 24)
                             | (0x0000FF00) & (uvalue >> 8)
                             | (0x00FF0000) & (uvalue << 8)
                             | (0xFF000000) & (uvalue << 24));
                        bdata[item_offset] = (UInt32)swapped;
                        break;

                    }

                case NPY_TYPES.NPY_INT64:
                    {
                        Int64[] bdata = dest.datap as Int64[];
                        Int64 value = bdata[item_offset];

                        var uvalue = (UInt64)value;
                        UInt64 swapped =
                             ((0x00000000000000FF) & (uvalue >> 56)
                             | (0x000000000000FF00) & (uvalue >> 40)
                             | (0x0000000000FF0000) & (uvalue >> 24)
                             | (0x00000000FF000000) & (uvalue >> 8)
                             | (0x000000FF00000000) & (uvalue << 8)
                             | (0x0000FF0000000000) & (uvalue << 24)
                             | (0x00FF000000000000) & (uvalue << 40)
                             | (0xFF00000000000000) & (uvalue << 56));
                        bdata[item_offset] =(Int64)swapped;
                        break;

                    }

                case NPY_TYPES.NPY_UINT64:
                    {
                        UInt64[] bdata = dest.datap as UInt64[];
                        UInt64 value = bdata[item_offset];

                        var uvalue = (UInt64)value;
                        UInt64 swapped =
                             ((0x00000000000000FF) & (uvalue >> 56)
                             | (0x000000000000FF00) & (uvalue >> 40)
                             | (0x0000000000FF0000) & (uvalue >> 24)
                             | (0x00000000FF000000) & (uvalue >> 8)
                             | (0x000000FF00000000) & (uvalue << 8)
                             | (0x0000FF0000000000) & (uvalue << 24)
                             | (0x00FF000000000000) & (uvalue << 40)
                             | (0xFF00000000000000) & (uvalue << 56));
                        bdata[item_offset] = (UInt64)swapped;
                        break;

                    }

                // these data types can't be swapped
                case NPY_TYPES.NPY_DECIMAL:
                case NPY_TYPES.NPY_COMPLEX:
                case NPY_TYPES.NPY_BIGINT:
                case NPY_TYPES.NPY_OBJECT:
                     return;



            }
        }

        internal static int NpyArray_CompareFunc(VoidPtr o1, VoidPtr o2, int elSize, NpyArray npa)
        {
            dynamic d1 = o1.datap;
            dynamic d2 = o2.datap;

            return DefaultArrayHandlers.GetArrayHandler(o1.type_num).CompareTo(d1[o1.data_offset / elSize], d2[o2.data_offset / elSize]);
        }


        #region ArgMax
        internal static npy_intp NpyArray_ArgMaxFunc(VoidPtr ip, npy_intp startIndex, npy_intp endIndex)
        {
            return DefaultArrayHandlers.GetArrayHandler(ip.type_num).ArgMax(ip.datap, startIndex, endIndex);
        }
        #endregion

        #region ArgMin
        internal static npy_intp NpyArray_ArgMinFunc(VoidPtr ip, npy_intp startIndex, npy_intp endIndex)
        {
            return DefaultArrayHandlers.GetArrayHandler(ip.type_num).ArgMin(ip.datap, startIndex, endIndex);
        }
        #endregion

        #region DotFunc
        internal static void NpyArray_DotFunc(VoidPtr ip1, npy_intp is1, VoidPtr ip2, npy_intp is2, VoidPtr op, npy_intp n, NpyArray NOT_USED)
        {
            DefaultArrayHandlers.GetArrayHandler(ip1.type_num).dot(ip1, is1, ip2, is2, op, n);
            return;
        }
        #endregion


        internal static int NpyArray_ScanFunc(FileInfo fp, object dptr, string ignore, NpyArray_Descr a)
        {
            return 2;
        }
        internal static int NpyArray_FromStrFunc(string s, object dptr, object[] endptr, NpyArray_Descr a)
        {
            return 2;
        }

        #region FillWithScalarFunc
        internal static int NpyArray_FillFunc(VoidPtr dest, npy_intp length, NpyArray arr)
        {
            return NpyArray_FillWithScalarFunc(dest, length, arr.data, arr);
        }

        internal static int NpyArray_FillWithScalarFunc(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            npy_intp dest_offset = dest.data_offset / arr.ItemSize;
            npy_intp scalar_offset = scalar.data_offset / arr.ItemSize;

            return DefaultArrayHandlers.GetArrayHandler(dest.type_num).ArrayFill(dest, scalar, (int)length, (int)dest_offset, (int)scalar_offset);
        }

        #endregion

        internal static int NpyArray_SortFunc(object o1, npy_intp length, NpyArray NOTUSED, NPY_SORTKIND kind)
        {
            VoidPtr arr = o1 as VoidPtr;
            return NpyArray_SortFuncTypeNum(arr, (int)(arr.data_offset / GetTypeSize(arr.type_num)), (int)length);
        }

        private static int NpyArray_SortFuncTypeNum(VoidPtr data, int offset, int length)
        {
            DefaultArrayHandlers.GetArrayHandler(data.type_num).SortArray(data, offset, length);
            return 0;
        }


        internal static int NpyArray_ArgSortFunc(object o1, VoidPtr indices, npy_intp m, NpyArray a, NPY_SORTKIND kind)
        {
            VoidPtr sortData = o1 as VoidPtr;
            ArgSortIndexes(indices, m, new VoidPtr(sortData), 0);
            return 0;
        }

        internal static NPY_SCALARKIND NpyArray_ScalarKindFunc(NpyArray a)
        {
            return NPY_SCALARKIND.NPY_BOOL_SCALAR;
        }

        internal static void NpyArray_FastClipFunc(VoidPtr _in, npy_intp n_in, VoidPtr min, VoidPtr max, VoidPtr _out)
        {

        }
        internal static void NpyArray_FastPutmaskFunc(VoidPtr _in, VoidPtr mask, npy_intp ni, VoidPtr vals, npy_intp nv)
        {
            npy_intp i, j;

            if (nv == 1)
            {
                double s_val =  (double)GetIndex(vals,0);
                for (i = 0; i < ni; i++)
                {
                    if ((bool)GetIndex(mask,i))
                    {
                        SetIndex(_in, i, s_val);
                    }
                }
            }
            else
            {
                for (i = 0, j = 0; i<ni; i++, j++)
                {
                    if (j >= nv)
                    {
                        j = 0;
                    }
                    if ((bool)GetIndex(mask, i))
                    {
                        SetIndex(_in, i, GetIndex(vals, j));
                    }
                }
            }
            return;
        }
        internal static void NpyArray_FastPutmaskFunc<T>(VoidPtr _in, VoidPtr mask, npy_intp ni, VoidPtr vals, npy_intp nv)
        {
            npy_intp i, j;

            if (nv == 1)
            {
                T s_val = (T)GetIndex(vals, 0);
                for (i = 0; i < ni; i++)
                {
                    if ((bool)GetIndex(mask, i))
                    {
                        SetIndex(_in, i, s_val);
                    }
                }
            }
            else
            {
                for (i = 0, j = 0; i < ni; i++, j++)
                {
                    if (j >= nv)
                    {
                        j = 0;
                    }
                    if ((bool)GetIndex(mask, i))
                    {
                        SetIndex(_in, i, (T)GetIndex(vals, j));
                    }
                }
            }
            return;
        }

        internal static int NpyArray_FastTakeFunc(VoidPtr dest, VoidPtr src, npy_intp[] indarray, npy_intp nindarray, npy_intp n_outer, npy_intp m_middle, npy_intp nelem, NPY_CLIPMODE clipmode)
        {
            return 1;
        }



    }
}
