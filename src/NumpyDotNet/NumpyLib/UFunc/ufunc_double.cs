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
        #region UFUNC Outer
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
        #endregion

        private double PerformUFuncOperation(NpyArray_Ops op, double aValue, double bValue)
        {
            double destValue = 0;
            bool boolValue = false;

            switch (op)
            {
                case NpyArray_Ops.npy_op_add:
                    destValue = Add(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_subtract:
                    destValue = Subtract(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_multiply:
                    destValue = Multiply(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_divide:
                    destValue = Divide(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_remainder:
                    destValue = Remainder(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_fmod:
                    destValue = FMod(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_power:
                    destValue = Power(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_square:
                    destValue = Square(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_reciprocal:
                    destValue = Reciprocal(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_ones_like:
                    destValue = OnesLike(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_sqrt:
                    destValue = Sqrt(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_negative:
                    destValue = Negative(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_absolute:
                    destValue = Absolute(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_invert:
                    destValue = Invert(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_left_shift:
                    destValue = LeftShift(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_right_shift:
                    destValue = RightShift(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_bitwise_and:
                    destValue = BitWiseAnd(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_bitwise_xor:
                    destValue = BitWiseXor(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_bitwise_or:
                    destValue = BitWiseOr(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_less:
                    boolValue = Less(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_less_equal:
                    boolValue = LessEqual(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_equal:
                    boolValue = Equal(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_not_equal:
                    boolValue = NotEqual(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_greater:
                    boolValue = Greater(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_greater_equal:
                    boolValue = GreaterEqual(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_floor_divide:
                    destValue = FloorDivide(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_true_divide:
                    destValue = TrueDivide(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_logical_or:
                    boolValue = LogicalOr(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_logical_and:
                    boolValue = LogicalAnd(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_floor:
                    destValue = Floor(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_ceil:
                    destValue = Ceiling(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_maximum:
                    destValue = Maximum(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_minimum:
                    destValue = Minimum(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_rint:
                    destValue = Rint(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_conjugate:
                    destValue = Conjugate(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_isnan:
                    boolValue = IsNAN(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_fmax:
                    destValue = FMax(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_fmin:
                    destValue = FMin(aValue, bValue);
                    break;
                case NpyArray_Ops.npy_op_heaviside:
                    destValue = Heaviside(aValue, bValue);
                    break;
                default:
                    destValue = 0;
                    break;

            }

            return destValue;
        }

        #region double specific operation handlers
        private double Add(double aValue, double bValue)
        {
            return aValue + bValue;
        }

        private double Subtract(double aValue, double bValue)
        {
            return aValue - bValue;
        }
        private double Multiply(double aValue, double bValue)
        {
            return aValue * bValue;
        }

        private double Divide(double aValue, double bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        private double Remainder(double aValue, double bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        private double FMod(double aValue, double bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        private double Power(double aValue, double bValue)
        {
            return Math.Pow(aValue, bValue);
        }
        private double Square(double bValue, double operand)
        {
            return bValue * bValue;
        }
        private double Reciprocal(double bValue, double operand)
        {
            return 1 / bValue;
        }
        private double OnesLike(double bValue, double operand)
        {
            return 1;
        }
        private double Sqrt(double bValue, double operand)
        {
            return Math.Sqrt(bValue);
        }
        private double Negative(double bValue, double operand)
        {
            return -bValue;
        }
        private double Absolute(double bValue, double operand)
        {
            return Math.Abs(bValue);
        }
        private double Invert(double bValue, double operand)
        {
            return bValue;
        }
        private double LeftShift(double bValue, double operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        private double RightShift(double bValue, double operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        private double BitWiseAnd(double bValue, double operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue & Convert.ToUInt64(operand);
        }
        private double BitWiseXor(double bValue, double operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue ^ Convert.ToUInt64(operand);
        }
        private double BitWiseOr(double bValue, double operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue | Convert.ToUInt64(operand);
        }
        private bool Less(double bValue, double operand)
        {
            return bValue < operand;
        }
        private bool LessEqual(double bValue, double operand)
        {
            return bValue <= operand;
        }
        private bool Equal(double bValue, double operand)
        {
            return bValue == operand;
        }
        private bool NotEqual(double bValue, double operand)
        {
            return bValue != operand;
        }
        private bool Greater(double bValue, double operand)
        {
            return bValue > operand;
        }
        private bool GreaterEqual(double bValue, double operand)
        {
            return bValue >= (dynamic)operand;
        }
        private double FloorDivide(double bValue, double operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Math.Floor(bValue / operand);
        }
        private double TrueDivide(double bValue, double operand)
        {
            return bValue / operand;
        }
        private bool LogicalOr(double bValue, double operand)
        {
            return bValue != 0 || operand != 0;
        }
        private bool LogicalAnd(double bValue, double operand)
        {
            return bValue != 0 && operand != 0;
        }
        private double Floor(double bValue, double operand)
        {
            return Math.Floor(bValue);
        }
        private double Ceiling(double bValue, double operand)
        {
            return Math.Ceiling(bValue);
        }
        private double Maximum(double bValue, double operand)
        {
            return Math.Max(bValue, operand);
        }
        private double Minimum(double bValue, double operand)
        {
            return Math.Min(bValue, operand);
        }
        private double Rint(double bValue, double operand)
        {
            return Math.Round(bValue);
        }
        private double Conjugate(double bValue, double operand)
        {
            return bValue;
        }
        private bool IsNAN(double bValue, double operand)
        {
            return double.IsNaN(bValue);
        }
        private double FMax(double bValue, double operand)
        {
            return Math.Max(bValue, operand);
        }
        private double FMin(double bValue, double operand)
        {
            return Math.Min(bValue, operand);
        }
        private double Heaviside(double bValue, double operand)
        {
            if (double.IsNaN(bValue))
                return double.NaN;

            if (bValue == 0.0)
                return operand;

            if (bValue < 0.0)
                return 0.0;

            return 1.0;

        }
        #endregion

    }


    #endregion
}
