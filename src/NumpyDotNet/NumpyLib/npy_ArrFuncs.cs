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
                    arrFuncs.copyswap = BOOL_copyswap;
                    arrFuncs.nonzero = BOOL_NonZeroFunc;
                    arrFuncs.getitem = BOOL_GetItemFunc;
                    arrFuncs.setitem = BOOL_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_BYTE:
                    arrFuncs.copyswap = BYTE_copyswap;
                    arrFuncs.nonzero = BYTE_NonZeroFunc;
                    arrFuncs.getitem = BYTE_GetItemFunc;
                    arrFuncs.setitem = BYTE_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    arrFuncs.copyswap = UBYTE_copyswap;
                    arrFuncs.nonzero = UBYTE_NonZeroFunc;
                    arrFuncs.getitem = UBYTE_GetItemFunc;
                    arrFuncs.setitem = UBYTE_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_INT16:
                    arrFuncs.copyswap = INT16_copyswap;
                    arrFuncs.nonzero = INT16_NonZeroFunc;
                    arrFuncs.getitem = INT16_GetItemFunc;
                    arrFuncs.setitem = INT16_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_UINT16:
                    arrFuncs.copyswap = UINT16_copyswap;
                    arrFuncs.nonzero = UINT16_NonZeroFunc;
                    arrFuncs.getitem = UINT16_GetItemFunc;
                    arrFuncs.setitem = UINT16_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_INT32:
                    arrFuncs.copyswap = INT32_copyswap;
                    arrFuncs.nonzero = INT32_NonZeroFunc;
                    arrFuncs.getitem = INT32_GetItemFunc;
                    arrFuncs.setitem = INT32_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_UINT32:
                    arrFuncs.copyswap = UINT32_copyswap;
                    arrFuncs.nonzero = UINT32_NonZeroFunc;
                    arrFuncs.getitem = UINT32_GetItemFunc;
                    arrFuncs.setitem = UINT32_SetItemFunc;
                    arrFuncs.setitem = UINT32_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_INT64:
                    arrFuncs.copyswap = INT64_copyswap;
                    arrFuncs.nonzero = INT64_NonZeroFunc;
                    arrFuncs.getitem = INT64_GetItemFunc;
                    arrFuncs.setitem = INT64_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_UINT64:
                    arrFuncs.copyswap = UINT64_copyswap;
                    arrFuncs.nonzero = UINT64_NonZeroFunc;
                    arrFuncs.getitem = UINT64_GetItemFunc;
                    arrFuncs.setitem = UINT64_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    arrFuncs.copyswap = FLOAT_copyswap;
                    arrFuncs.nonzero = FLOAT_NonZeroFunc;
                    arrFuncs.getitem = FLOAT_GetItemFunc;
                    arrFuncs.setitem = FLOAT_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    arrFuncs.copyswap = DOUBLE_copyswap;
                    arrFuncs.nonzero = DOUBLE_NonZeroFunc;
                    arrFuncs.getitem = DOUBLE_GetItemFunc;
                    arrFuncs.setitem = DOUBLE_SetItemFunc;
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    arrFuncs.copyswap = DECIMAL_copyswap;
                    arrFuncs.nonzero = DECIMAL_NonZeroFunc;
                    arrFuncs.getitem = DECIMAL_GetItemFunc;
                    arrFuncs.setitem = DECIMAL_SetItemFunc;
                    break;

                default:
                    arrFuncs.copyswap = NpyArray_CopySwapFunc;
                    arrFuncs.nonzero = NpyArray_NonzeroFunc;
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

        internal static long AdjustedIndex_GetItemFunction(npy_intp index, NpyArray npa, int dpLength)
        {
            long AdjustedIndex = 0;

            AdjustedIndex = (npa.data.data_offset + index) / npa.ItemSize;

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



        internal static object NpyArray_GetItemFunc(npy_intp index, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                long AdjustedIndex = index < 0 ? index / npa.ItemSize : (npa.data.data_offset + index) / npa.ItemSize;
                return GetIndex(npa.data, AdjustedIndex);
            }
            else
            {
                return DifferentSizes_GetItemFunc(index, npa);
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
            long AdjustedIndex = 0;

            AdjustedIndex = (npa.data.data_offset + index) / npa.ItemSize;

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
                long AdjustedIndex = AdjustedIndex_SetItemFunction(index, npa, dp.Length);
                dp[AdjustedIndex] = (Int64)value;
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

        internal static int NpyArray_SetItemFunc(npy_intp index, object value, NpyArray npa)
        {
            if (npa.ItemType == npa.data.type_num)
            {
                long AdjustedIndex = index < 0 ? index / npa.ItemSize : (npa.data.data_offset + index) / npa.ItemSize;
                return SetIndex(npa.data, AdjustedIndex, value);
            }
            else
            {
                return DifferentSizes_SetItemFunc(index, value, npa);
            }
        }
        #endregion

        internal static void NpyArray_CopySwapNFunc(VoidPtr Dest, npy_intp dstride, VoidPtr Src, npy_intp sstride, npy_intp N, bool swap, NpyArray npa)
        {
            _default_copyswap(Dest, dstride, Src, sstride, N, swap, npa);
        }

        #region copyswap
        internal static void BOOL_copyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            Common_copyswap<bool>(dest, Source, swap, arr);
        }
        internal static void BYTE_copyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            Common_copyswap<sbyte>(dest, Source, swap, arr);
        }
        internal static void UBYTE_copyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            Common_copyswap<byte>(dest, Source, swap, arr);
        }
        internal static void INT16_copyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            Common_copyswap<Int16>(dest, Source, swap, arr);
        }
        internal static void UINT16_copyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            Common_copyswap<UInt16>(dest, Source, swap, arr);
        }
        internal static void INT32_copyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            Common_copyswap<Int32>(dest, Source, swap, arr);
        }
        internal static void UINT32_copyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            Common_copyswap<UInt32>(dest, Source, swap, arr);
        }
        internal static void INT64_copyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            Common_copyswap<Int64>(dest, Source, swap, arr);
        }
        internal static void UINT64_copyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            Common_copyswap<UInt64>(dest, Source, swap, arr);
        }
        internal static void FLOAT_copyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            Common_copyswap<float>(dest, Source, swap, arr);
        }
        internal static void DOUBLE_copyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            Common_copyswap<double>(dest, Source, swap, arr);
        }
        internal static void DECIMAL_copyswap(VoidPtr dest, VoidPtr Source, bool swap, NpyArray arr)
        {
            Common_copyswap<decimal>(dest, Source, swap, arr);
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
                NpyArray_Descr descr = new NpyArray_Descr(NPY_TYPES.NPY_VOID);
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
            }
        }

        internal static int NpyArray_CompareFunc(VoidPtr o1, VoidPtr o2, int elSize, NpyArray npa)
        {
            dynamic d1 = o1.datap;
            dynamic d2 = o2.datap;

            return d1[o1.data_offset/elSize].CompareTo(d2[o2.data_offset/elSize]);
        }


        #region ArgMax
        internal static int NpyArray_ArgMaxFunc(VoidPtr ip, npy_intp startIndex, npy_intp endIndex, ref npy_intp max_ind, NpyArray NOT_USED)
        {
            switch (ip.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return NpyArray_ArgMaxFunc(ip.datap as bool[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_BYTE:
                    return NpyArray_ArgMaxFunc(ip.datap as sbyte[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_UBYTE:
                    return NpyArray_ArgMaxFunc(ip.datap as byte[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_INT16:
                    return NpyArray_ArgMaxFunc(ip.datap as Int16[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_UINT16:
                    return NpyArray_ArgMaxFunc(ip.datap as UInt16[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_INT32:
                    return NpyArray_ArgMaxFunc(ip.datap as Int32[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_UINT32:
                    return NpyArray_ArgMaxFunc(ip.datap as UInt32[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_INT64:
                    return NpyArray_ArgMaxFunc(ip.datap as Int64[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_UINT64:
                    return NpyArray_ArgMaxFunc(ip.datap as UInt64[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_FLOAT:
                    return NpyArray_ArgMaxFunc(ip.datap as float[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_DOUBLE:
                    return NpyArray_ArgMaxFunc(ip.datap as double[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_DECIMAL:
                    return NpyArray_ArgMaxFunc(ip.datap as decimal[], startIndex, endIndex, ref max_ind);
            }

            return 0;

        }

        internal static int NpyArray_ArgMaxFunc<T>(T[] ip, npy_intp startIndex, npy_intp endIndex, ref npy_intp max_ind) where T : IComparable
        {
            T mp = ip[0 + startIndex];

            max_ind = 0;
            for (npy_intp i = 1+startIndex; i < endIndex+startIndex; i++)
            {
                if (ip[i].CompareTo(mp) > 0)
                {
                    mp = ip[i];
                    max_ind = i-startIndex;
                }
            }
            return 0;
        }
        #endregion

        #region ArgMin
        internal static int NpyArray_ArgMinFunc(VoidPtr ip, npy_intp startIndex, npy_intp endIndex, ref npy_intp max_ind, NpyArray NOT_USED)
        {
            switch (ip.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return NpyArray_ArgMinFunc(ip.datap as bool[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_BYTE:
                    return NpyArray_ArgMinFunc(ip.datap as sbyte[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_UBYTE:
                    return NpyArray_ArgMinFunc(ip.datap as byte[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_INT16:
                    return NpyArray_ArgMinFunc(ip.datap as Int16[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_UINT16:
                    return NpyArray_ArgMinFunc(ip.datap as UInt16[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_INT32:
                    return NpyArray_ArgMinFunc(ip.datap as Int32[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_UINT32:
                    return NpyArray_ArgMinFunc(ip.datap as UInt32[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_INT64:
                    return NpyArray_ArgMinFunc(ip.datap as Int64[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_UINT64:
                    return NpyArray_ArgMinFunc(ip.datap as UInt64[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_FLOAT:
                    return NpyArray_ArgMinFunc(ip.datap as float[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_DOUBLE:
                    return NpyArray_ArgMinFunc(ip.datap as double[], startIndex, endIndex, ref max_ind);
                case NPY_TYPES.NPY_DECIMAL:
                    return NpyArray_ArgMinFunc(ip.datap as decimal[], startIndex, endIndex, ref max_ind);
            }

            return 0;

        }

        internal static int NpyArray_ArgMinFunc<T>(T[] ip, npy_intp startIndex, npy_intp endIndex, ref npy_intp max_ind) where T : IComparable
        {
            T mp = ip[0+ startIndex];

            max_ind = 0;
            for (npy_intp i = 1+ startIndex; i < endIndex+ startIndex; i++)
            {
                if (ip[i].CompareTo(mp) < 0)
                {
                    mp = ip[i];
                    max_ind = i-startIndex;
                }
            }
            return 0;
        }
        #endregion

        #region DotFunc
        internal static void NpyArray_DotFunc(VoidPtr ip1, npy_intp is1, VoidPtr ip2, npy_intp is2, VoidPtr op, npy_intp n, NpyArray NOT_USED)
        {
            switch (ip1.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    BOOL_dot(ip1, is1, ip2, is2, op, n);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    BYTE_dot(ip1, is1, ip2, is2, op, n);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    UBYTE_dot(ip1, is1, ip2, is2, op, n);
                    break;
                case NPY_TYPES.NPY_INT16:
                    INT16_dot(ip1, is1, ip2, is2, op, n);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    UINT16_dot(ip1, is1, ip2, is2, op, n);
                    break;
                case NPY_TYPES.NPY_INT32:
                    INT32_dot(ip1, is1, ip2, is2, op, n);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    UINT32_dot(ip1, is1, ip2, is2, op, n);
                    break;
                case NPY_TYPES.NPY_INT64:
                    INT64_dot(ip1, is1, ip2, is2, op, n);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    UINT64_dot(ip1, is1, ip2, is2, op, n);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    FLOAT_dot(ip1, is1, ip2, is2, op, n);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    DOUBLE_dot(ip1, is1, ip2, is2, op, n);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    DECIMAL_dot(ip1, is1, ip2, is2, op, n);
                    break;
            }

            return;
        }

        internal static void BOOL_dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            bool tmp = false;
            npy_intp i;

            bool[] ip1 = _ip1.datap as bool[];
            bool[] ip2 = _ip2.datap as bool[];
            bool[] op = _op.datap as bool[];
            npy_intp ip1_index = 0;
            npy_intp ip2_index = 0;

            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                if ((ip1[ip1_index] == true) && (ip2[ip2_index] == true))
                {
                    tmp = true;
                    break;
                }
            }
            op[0] = tmp;
        }

        internal static void BYTE_dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            long tmp = 0;
            npy_intp i;

            sbyte[] ip1 = _ip1.datap as sbyte[];
            sbyte[] ip2 = _ip2.datap as sbyte[];
            long[] op = _op.datap as long[];
            npy_intp ip1_index = 0;
            npy_intp ip2_index = 0;

            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                tmp += (long)ip1[ip1_index] * (long)ip2[ip2_index];

            }
            op[0] = tmp;
        }

        internal static void UBYTE_dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            long tmp = 0;
            npy_intp i;

            byte[] ip1 = _ip1.datap as byte[];
            byte[] ip2 = _ip2.datap as byte[];
            long[] op = _op.datap as long[];
            npy_intp ip1_index = 0;
            npy_intp ip2_index = 0;

            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                tmp += (long)ip1[ip1_index] * (long)ip2[ip2_index];

            }
            op[0] = tmp;
        }

        internal static void INT16_dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            long tmp = 0;
            npy_intp i;

            Int16[] ip1 = _ip1.datap as Int16[];
            Int16[] ip2 = _ip2.datap as Int16[];
            long[] op = _op.datap as long[];
            npy_intp ip1_index = 0;
            npy_intp ip2_index = 0;

            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                tmp += (long)ip1[ip1_index] * (long)ip2[ip2_index];

            }
            op[0] = tmp;
        }

        internal static void UINT16_dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            ulong tmp = 0;
            npy_intp i;

            Int16[] ip1 = _ip1.datap as Int16[];
            Int16[] ip2 = _ip2.datap as Int16[];
            ulong[] op = _op.datap as ulong[];
            npy_intp ip1_index = 0;
            npy_intp ip2_index = 0;

            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                tmp += (ulong)ip1[ip1_index] * (ulong)ip2[ip2_index];

            }
            op[0] = tmp;
        }

        internal static void INT32_dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            long tmp = 0;
            npy_intp i;

            Int32[] ip1 = _ip1.datap as Int32[];
            Int32[] ip2 = _ip2.datap as Int32[];
            long[] op = _op.datap as long[];
            npy_intp ip1_index = 0;
            npy_intp ip2_index = 0;

            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                tmp += (long)ip1[ip1_index] * (long)ip2[ip2_index];

            }
            op[0] = tmp;
        }

        internal static void UINT32_dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            ulong tmp = 0;
            npy_intp i;

            UInt32[] ip1 = _ip1.datap as UInt32[];
            UInt32[] ip2 = _ip2.datap as UInt32[];
            ulong[] op = _op.datap as ulong[];
            npy_intp ip1_index = 0;
            npy_intp ip2_index = 0;

            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                tmp += (ulong)ip1[ip1_index] * (ulong)ip2[ip2_index];

            }
            op[0] = tmp;
        }

        internal static void INT64_dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            long tmp = 0;
            npy_intp i;

            Int64[] ip1 = _ip1.datap as Int64[];
            Int64[] ip2 = _ip2.datap as Int64[];
            long[] op = _op.datap as long[];
            npy_intp ip1_index = 0;
            npy_intp ip2_index = 0;

            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                tmp += (long)ip1[ip1_index] * (long)ip2[ip2_index];

            }
            op[0] = tmp;
        }

        internal static void UINT64_dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            ulong tmp = 0;
            npy_intp i;

            UInt64[] ip1 = _ip1.datap as UInt64[];
            UInt64[] ip2 = _ip2.datap as UInt64[];
            ulong[] op = _op.datap as ulong[];
            npy_intp ip1_index = 0;
            npy_intp ip2_index = 0;

            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                tmp += (ulong)ip1[ip1_index] * (ulong)ip2[ip2_index];

            }
            op[0] = tmp;
        }

        internal static void FLOAT_dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            long tmp = 0;
            npy_intp i;

            float[] ip1 = _ip1.datap as float[];
            float[] ip2 = _ip2.datap as float[];
            long[] op = _op.datap as long[];
            npy_intp ip1_index = 0;
            npy_intp ip2_index = 0;

            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                tmp += (long)ip1[ip1_index] * (long)ip2[ip2_index];

            }
            op[0] = tmp;
        }

        internal static void DOUBLE_dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            ulong tmp = 0;
            npy_intp i;

            double[] ip1 = _ip1.datap as double[];
            double[] ip2 = _ip2.datap as double[];
            ulong[] op = _op.datap as ulong[];
            npy_intp ip1_index = 0;
            npy_intp ip2_index = 0;

            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                tmp += (ulong)ip1[ip1_index] * (ulong)ip2[ip2_index];

            }
            op[0] = tmp;
        }

        internal static void DECIMAL_dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            ulong tmp = 0;
            npy_intp i;

            decimal[] ip1 = _ip1.datap as decimal[];
            decimal[] ip2 = _ip2.datap as decimal[];
            ulong[] op = _op.datap as ulong[];
            npy_intp ip1_index = 0;
            npy_intp ip2_index = 0;

            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                tmp += (ulong)ip1[ip1_index] * (ulong)ip2[ip2_index];

            }
            op[0] = tmp;
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

        #region NonZeroFunc
        internal static bool BOOL_NonZeroFunc(VoidPtr vp, NpyArray npa)
        {
            bool[] bp = vp.datap as bool[];
            return (bp[vp.data_offset / npa.ItemSize]);
        }
        internal static bool BYTE_NonZeroFunc(VoidPtr vp, NpyArray npa)
        {
            sbyte[] bp = vp.datap as sbyte[];
            return (bp[vp.data_offset / npa.ItemSize] != 0);
        }
        internal static bool UBYTE_NonZeroFunc(VoidPtr vp, NpyArray npa)
        {
            byte[] bp = vp.datap as byte[];
            return (bp[vp.data_offset / npa.ItemSize] != 0);
        }
        internal static bool INT16_NonZeroFunc(VoidPtr vp, NpyArray npa)
        {
            Int16[] bp = vp.datap as Int16[];
            return (bp[vp.data_offset / npa.ItemSize] != 0);
        }
        internal static bool UINT16_NonZeroFunc(VoidPtr vp, NpyArray npa)
        {
            UInt16[] bp = vp.datap as UInt16[];
            return (bp[vp.data_offset / npa.ItemSize] != 0);
        }
        internal static bool INT32_NonZeroFunc(VoidPtr vp, NpyArray npa)
        {
            Int32[] bp = vp.datap as Int32[];
            return (bp[vp.data_offset / npa.ItemSize] != 0);
        }
        internal static bool UINT32_NonZeroFunc(VoidPtr vp, NpyArray npa)
        {
            UInt32[] bp = vp.datap as UInt32[];
            return (bp[vp.data_offset / npa.ItemSize] != 0);
        }
        internal static bool INT64_NonZeroFunc(VoidPtr vp, NpyArray npa)
        {
            Int64[] bp = vp.datap as Int64[];
            return (bp[vp.data_offset / npa.ItemSize] != 0);
        }
        internal static bool UINT64_NonZeroFunc(VoidPtr vp, NpyArray npa)
        {
            UInt64[] bp = vp.datap as UInt64[];
            return (bp[vp.data_offset / npa.ItemSize] != 0);
        }
        internal static bool FLOAT_NonZeroFunc(VoidPtr vp, NpyArray npa)
        {
            float[] bp = vp.datap as float[];
            return (bp[vp.data_offset / npa.ItemSize] != 0);
        }
        internal static bool DOUBLE_NonZeroFunc(VoidPtr vp, NpyArray npa)
        {
            double[] bp = vp.datap as double[];
            return (bp[vp.data_offset / npa.ItemSize] != 0);
        }
        internal static bool DECIMAL_NonZeroFunc(VoidPtr vp, NpyArray npa)
        {
            decimal[] bp = vp.datap as decimal[];
            return (bp[vp.data_offset / npa.ItemSize] != 0);
        }
        internal static bool NpyArray_NonzeroFunc(VoidPtr vp, NpyArray npa)
        {
            dynamic Item = npa.descr.f.getitem(vp.data_offset, npa);
            if (npa.ItemType == NPY_TYPES.NPY_BOOL)
                return Item;

            return Item != 0;
        }
        #endregion

        internal static int NpyArray_FillFunc(VoidPtr dest, npy_intp length, NpyArray arr)
        {
            return NpyArray_FillWithScalarFunc(dest, length, arr.data, arr);
        }

        #region FillWithScalarFunc
        internal static int NpyArray_FillWithScalarFunc(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            switch (dest.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return BOOL_fillwithscalar(dest, length, scalar, arr);
                case NPY_TYPES.NPY_BYTE:
                    return BYTE_fillwithscalar(dest, length, scalar, arr);
                case NPY_TYPES.NPY_UBYTE:
                    return UBYTE_fillwithscalar(dest, length, scalar, arr);
                case NPY_TYPES.NPY_INT16:
                    return INT16_fillwithscalar(dest, length, scalar, arr);
                case NPY_TYPES.NPY_UINT16:
                    return UINT16_fillwithscalar(dest, length, scalar, arr);
                case NPY_TYPES.NPY_INT32:
                    return INT32_fillwithscalar(dest, length, scalar, arr);
                case NPY_TYPES.NPY_UINT32:
                    return UINT32_fillwithscalar(dest, length, scalar, arr);
                case NPY_TYPES.NPY_INT64:
                    return INT64_fillwithscalar(dest, length, scalar, arr);
                case NPY_TYPES.NPY_UINT64:
                    return UINT64_fillwithscalar(dest, length, scalar, arr);
                case NPY_TYPES.NPY_FLOAT:
                    return FLOAT_fillwithscalar(dest, length, scalar, arr);
                case NPY_TYPES.NPY_DOUBLE:
                    return DOUBLE_fillwithscalar(dest, length, scalar, arr);
                case NPY_TYPES.NPY_DECIMAL:
                    return DECIMAL_fillwithscalar(dest, length, scalar, arr);
            }

            return -1;
        }

        private static int BOOL_fillwithscalar(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            return Common_Fillwithscalar<bool>(dest, length, scalar, arr);
        }
        private static int BYTE_fillwithscalar(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            return Common_Fillwithscalar<sbyte>(dest, length, scalar, arr);
        }
        private static int UBYTE_fillwithscalar(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            return Common_Fillwithscalar<byte>(dest, length, scalar, arr);
        }
        private static int INT16_fillwithscalar(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            return Common_Fillwithscalar<Int16>(dest, length, scalar, arr);
        }
        private static int UINT16_fillwithscalar(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            return Common_Fillwithscalar<UInt16>(dest, length, scalar, arr);
        }
        private static int INT32_fillwithscalar(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            return Common_Fillwithscalar<Int32>(dest, length, scalar, arr);
        }
        private static int UINT32_fillwithscalar(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            return Common_Fillwithscalar<UInt32>(dest, length, scalar, arr);
        }
        private static int INT64_fillwithscalar(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            return Common_Fillwithscalar<Int64>(dest, length, scalar, arr);
        }
        private static int UINT64_fillwithscalar(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            return Common_Fillwithscalar<UInt64>(dest, length, scalar, arr);
        }
        private static int FLOAT_fillwithscalar(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            return Common_Fillwithscalar<float>(dest, length, scalar, arr);
        }
        private static int DOUBLE_fillwithscalar(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            return Common_Fillwithscalar<double>(dest, length, scalar, arr);
        }
        private static int DECIMAL_fillwithscalar(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            return Common_Fillwithscalar<decimal>(dest, length, scalar, arr);
        }
        private static int Common_Fillwithscalar<T>(VoidPtr dest, npy_intp length, VoidPtr scalar, NpyArray arr)
        {
            T[] destp = dest.datap as T[];
            T[] scalarp = scalar.datap as T[];
            if (destp == null || scalarp == null)
                return -1;

            npy_intp dest_offset = dest.data_offset / arr.ItemSize;
            npy_intp scalar_offset = scalar.data_offset / arr.ItemSize;

            for (int i = 0; i < length; i++)
            {
                destp[i + dest_offset] = scalarp[0 + scalar_offset];
            }

            return 0;
        }
        #endregion

        internal static int NpyArray_SortFunc(object o1, npy_intp length, NpyArray NOTUSED)
        {
            VoidPtr arr = o1 as VoidPtr;
            return NpyArray_SortFuncTypeNum(arr, (int)(arr.data_offset / GetTypeSize(arr.type_num)), (int)length);
        }

        private static int NpyArray_SortFuncTypeNum(VoidPtr data, int offset, int length)
        {
            switch (data.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    var dbool = data.datap as bool[];
                    Array.Sort(dbool, offset, length);
                    return 0;
                case NPY_TYPES.NPY_BYTE:
                    var dsbyte = data.datap as sbyte[];
                    Array.Sort(dsbyte, offset, length);
                    return 0;
                case NPY_TYPES.NPY_UBYTE:
                    var dbyte = data.datap as byte[];
                    Array.Sort(dbyte, offset, length);
                    return 0;
                case NPY_TYPES.NPY_UINT16:
                    var duint16 = data.datap as UInt16[];
                    Array.Sort(duint16, offset, length);
                    return 0;
                case NPY_TYPES.NPY_INT16:
                    var dint16 = data.datap as Int16[];
                    Array.Sort(dint16, offset, length);
                    return 0;
                case NPY_TYPES.NPY_UINT32:
                    var duint32 = data.datap as UInt32[];
                    Array.Sort(duint32, offset, length);
                    return 0;
                case NPY_TYPES.NPY_INT32:
                    var dint32 = data.datap as Int32[];
                    Array.Sort(dint32, offset, length);
                    return 0;
                case NPY_TYPES.NPY_INT64:
                    var dint64 = data.datap as Int64[];
                    Array.Sort(dint64, offset, length);
                    return 0;
                case NPY_TYPES.NPY_UINT64:
                    var duint64 = data.datap as UInt64[];
                    Array.Sort(duint64, offset, length);
                    return 0;
                case NPY_TYPES.NPY_FLOAT:
                    var float1 = data.datap as float[];
                    Array.Sort(float1, offset, length);
                    return 0;
                case NPY_TYPES.NPY_DOUBLE:
                    var double1 = data.datap as double[];
                    Array.Sort(double1, offset, length);
                    return 0;
                case NPY_TYPES.NPY_DECIMAL:
                    var decimal1 = data.datap as decimal[];
                    Array.Sort(decimal1, offset, length);
                    return 0;

                default:
                    throw new Exception("Unsupported data type");
            }


        }


        internal static int NpyArray_ArgSortFunc(object o1, VoidPtr indices, npy_intp m, NpyArray a)
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
        internal static void NpyArray_FastPutmaskFunc(VoidPtr _in, VoidPtr mask, npy_intp n_in, VoidPtr values, npy_intp nv)
        {

        }
        internal static int NpyArray_FastTakeFunc(VoidPtr dest, VoidPtr src, npy_intp[] indarray, npy_intp nindarray, npy_intp n_outer, npy_intp m_middle, npy_intp nelem, NPY_CLIPMODE clipmode)
        {
            return 1;
        }



    }
}
