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
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NumpyLib.numpyinternal;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyLib
{

    #region UFUNC INT32

    internal class UFUNC_Int32 : UFUNC_Operations
    {
        public void PerformOuterOpArrayIter(NpyArray a, NpyArray b, NpyArray destArray, NumericOperations operations, UFuncOperation op)
        {
            var destSize = NpyArray_Size(destArray);
            var aSize = NpyArray_Size(a);
            var bSize = NpyArray_Size(b);

            if (bSize == 0 || aSize == 0)
            {
                NpyArray_Resize(destArray, new NpyArray_Dims() { len = 0, ptr = new npy_intp[] { } }, false, NPY_ORDER.NPY_ANYORDER);
                return;
            }

            var aIter = NpyArray_IterNew(a);
            var bIter = NpyArray_IterNew(b);
            var DestIter = NpyArray_IterNew(destArray);

            Int32[] aValues = new Int32[aSize];
            for (long i = 0; i < aSize; i++)
            {
                aValues[i] = Convert.ToInt32(operations.srcGetItem(aIter.dataptr.data_offset - a.data.data_offset, a));
                NpyArray_ITER_NEXT(aIter);
            }

            Int32[] bValues = new Int32[bSize];
            for (long i = 0; i < bSize; i++)
            {
                bValues[i] = Convert.ToInt32(operations.operandGetItem(bIter.dataptr.data_offset - b.data.data_offset, b));
                NpyArray_ITER_NEXT(bIter);
            }


            Int32[] dp = destArray.data.datap as Int32[];


            if (DestIter.contiguous && destSize > UFUNC_PARALLEL_DEST_MINSIZE && aSize > UFUNC_PARALLEL_DEST_ASIZE)
            {

                Parallel.For(0, aSize, i =>
                {
                    var aValue = aValues[i];

                    long destIndex = (destArray.data.data_offset / destArray.ItemSize) + i * bSize;

                    for (long j = 0; j < bSize; j++)
                    {
                        var bValue = bValues[j];

                        Int32 destValue = PerformUFuncOperation(op, aValue, bValue);

                        try
                        {
                            dp[destIndex] = destValue;
                        }
                        catch
                        {
                            operations.destSetItem(destIndex, 0, destArray);
                        }
                        destIndex++;
                    }

                });
            }
            else
            {
                for (long i = 0; i < aSize; i++)
                {
                    var aValue = aValues[i];

                    for (long j = 0; j < bSize; j++)
                    {
                        var bValue = bValues[j];

                        Int32 destValue = PerformUFuncOperation(op, aValue, bValue);

                        try
                        {
                            long AdjustedIndex = AdjustedIndex_SetItemFunction(DestIter.dataptr.data_offset - destArray.data.data_offset, destArray, dp.Length);
                            dp[AdjustedIndex] = destValue;
                        }
                        catch
                        {
                            long AdjustedIndex = AdjustedIndex_SetItemFunction(DestIter.dataptr.data_offset - destArray.data.data_offset, destArray, dp.Length);
                            operations.destSetItem(AdjustedIndex, 0, destArray);
                        }
                        NpyArray_ITER_NEXT(DestIter);
                    }

                }
            }


        }

        private Int32 PerformUFuncOperation(UFuncOperation op, Int32 aValue, Int32 bValue)
        {
            Int32 destValue;
            switch (op)
            {
                case UFuncOperation.npy_op_add:
                    destValue = UFuncAdd(aValue, bValue);
                    break;
                case UFuncOperation.npy_op_subtract:
                    destValue = UFuncSubtract(aValue, bValue);
                    break;
                case UFuncOperation.npy_op_multiply:
                    destValue = UFuncMultiply(aValue, bValue);
                    break;
                case UFuncOperation.npy_op_divide:
                    destValue = UFuncDivide(aValue, bValue);
                    break;
                case UFuncOperation.npy_op_remainder:
                    destValue = UFuncRemainder(aValue, bValue);
                    break;
                case UFuncOperation.npy_op_fmod:
                    destValue = UFuncFMod(aValue, bValue);
                    break;
                case UFuncOperation.npy_op_power:
                    destValue = UFuncPower(aValue, bValue);
                    break;

                default:
                    destValue = 0;
                    break;

            }

            return destValue;
        }

        private Int32 UFuncAdd(Int32 aValue, Int32 bValue)
        {
            return aValue + bValue;
        }
        private Int32 UFuncSubtract(Int32 aValue, Int32 bValue)
        {
            return aValue - bValue;
        }
        private Int32 UFuncMultiply(Int32 aValue, Int32 bValue)
        {
            return aValue * bValue;
        }
        private Int32 UFuncDivide(Int32 aValue, Int32 bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        private Int32 UFuncRemainder(Int32 aValue, Int32 bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        private Int32 UFuncFMod(Int32 aValue, Int32 bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        private Int32 UFuncPower(Int32 aValue, Int32 bValue)
        {
            return Convert.ToInt32(Math.Pow((double)aValue, (double)bValue));
        }
    }

    #endregion
}
