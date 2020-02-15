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
    #region UFUNC DOUBLE
    internal class UFUNC_Double : UFUNC_Operations
    {
        public void PerformOuterOpArrayIter(NpyArray a, NpyArray b, NpyArray destArray, NumericOperations operations, NpyArray_Ops op)
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

            double[] aValues = new double[aSize];
            for (long i = 0; i < aSize; i++)
            {
                aValues[i] = Convert.ToDouble(operations.srcGetItem(aIter.dataptr.data_offset - a.data.data_offset, a));
                NpyArray_ITER_NEXT(aIter);
            }

            double[] bValues = new double[bSize];
            for (long i = 0; i < bSize; i++)
            {
                bValues[i] = Convert.ToDouble(operations.operandGetItem(bIter.dataptr.data_offset - b.data.data_offset, b));
                NpyArray_ITER_NEXT(bIter);
            }


            double[] dp = destArray.data.datap as double[];


            if (DestIter.contiguous && destSize > UFUNC_PARALLEL_DEST_MINSIZE && aSize > UFUNC_PARALLEL_DEST_ASIZE)
            {

                Parallel.For(0, aSize, i =>
                {
                    var aValue = aValues[i];

                    long destIndex = (destArray.data.data_offset / destArray.ItemSize) + i * bSize;

                    for (long j = 0; j < bSize; j++)
                    {
                        var bValue = bValues[j];

                        double destValue = PerformUFuncOperation(op, aValue, bValue);

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

                        double destValue = PerformUFuncOperation(op, aValue, bValue);

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

        private double PerformUFuncOperation(NpyArray_Ops op, double aValue, double bValue)
        {
            double destValue;
            switch (op)
            {
                case NpyArray_Ops.npy_op_add:
                    destValue = UFuncAdd(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_subtract:
                    destValue = UFuncSubtract(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_multiply:
                    destValue = UFuncMultiply(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_divide:
                    destValue = UFuncDivide(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_remainder:
                    destValue = UFuncRemainder(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_fmod:
                    destValue = UFuncFMod(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_power:
                    destValue = UFuncPower(aValue, bValue);
                    break;

                default:
                    destValue = 0;
                    break;

            }

            return destValue;
        }


        private double UFuncAdd(double aValue, double bValue)
        {
            return aValue + bValue;
        }

        private double UFuncSubtract(double aValue, double bValue)
        {
            return aValue - bValue;
        }
        private double UFuncMultiply(double aValue, double bValue)
        {
            return aValue * bValue;
        }

        private double UFuncDivide(double aValue, double bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        private double UFuncRemainder(double aValue, double bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        private double UFuncFMod(double aValue, double bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        private double UFuncPower(double aValue, double bValue)
        {
            return Math.Pow(aValue, bValue);
        }
    }


    #endregion
}
