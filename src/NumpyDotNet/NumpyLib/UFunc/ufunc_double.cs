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
    internal class UFUNC_Double : iUFUNC_Operations
    {
        #region UFUNC Outer
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

        #region UFUNC Reduce

        public void PerformReduceOpArrayIter(VoidPtr[] bufPtr, npy_intp[] steps, UFuncOperation ops, npy_intp N)
        {
            VoidPtr Operand1 = bufPtr[0];
            VoidPtr Operand2 = bufPtr[1];
            VoidPtr Result = bufPtr[2];

            npy_intp O1_Step = steps[0];
            npy_intp O2_Step = steps[1];
            npy_intp R_Step = steps[2];
              
            npy_intp O1_Offset = Operand1.data_offset;
            npy_intp O2_Offset = Operand2.data_offset;
            npy_intp R_Offset = Result.data_offset;


            double[] retArray = Result.datap as double[];
            double[] Op1Array = Operand1.datap as double[];
            double[] Op2Array = Operand2.datap as double[];

            npy_intp R_Index = AdjustNegativeIndex(retArray, R_Offset / sizeof(double));
            npy_intp O1_Index = AdjustNegativeIndex(Op1Array, O1_Offset / sizeof(double));

            npy_intp O2_CalculatedStep = (O2_Step / sizeof(double));
            npy_intp O2_CalculatedOffset = (O2_Offset / sizeof(double));


            double retValue = retArray[R_Index];

            // note: these can't be parrallized.
            for (int i = 0; i < N; i++)
            {
                npy_intp O2_Index = ((i * O2_CalculatedStep) + O2_CalculatedOffset);

                var Op1Value = retValue;
                var Op2Value = Op2Array[O2_Index];

                // for the common operations, do inline for speed.
                switch (ops)
                {
                    case UFuncOperation.add:
                        retValue = Add(Op1Value, Op2Value);
                        break;
                    case UFuncOperation.subtract:
                        retValue = Subtract(Op1Value, Op2Value);
                        break;
                    case UFuncOperation.multiply:
                        retValue = Multiply(Op1Value, Op2Value);
                        break;
                    case UFuncOperation.divide:
                        retValue = Divide(Op1Value, Op2Value);
                        break;
                    case UFuncOperation.power:
                        retValue = Power(Op1Value, Op2Value);
                        break;

                    default:
                        retValue = PerformUFuncOperation(ops, Op1Value, Op2Value);
                        break;

                }
            }

            retArray[R_Index] = retValue;
            return;
        }

        private npy_intp AdjustNegativeIndex(double[] data, npy_intp index)
        {
            if (index < 0)
            {
                index = data.Length - Math.Abs(index);
            }
            return index;
        }

        #endregion

        #region UFUNC Accumulate

        public void PerformAccumulateOpArrayIter(VoidPtr[] bufPtr, npy_intp[] steps, UFuncOperation ops, npy_intp N)
        {
            VoidPtr Operand1 = bufPtr[0];
            VoidPtr Operand2 = bufPtr[1];
            VoidPtr Result = bufPtr[2];

            npy_intp O1_Step = steps[0];
            npy_intp O2_Step = steps[1];
            npy_intp R_Step = steps[2];

            if (Operand2 == null)
            {
                Operand2 = Operand1;
                O2_Step = O1_Step;
            }
            if (Result == null)
            {
                Result = Operand1;
                R_Step = O1_Step;
            }

            npy_intp O1_Offset = Operand1.data_offset;
            npy_intp O2_Offset = Operand2.data_offset;
            npy_intp R_Offset = Result.data_offset;


            double[] retArray = Result.datap as double[];
            double[] Op1Array = Operand1.datap as double[];
            double[] Op2Array = Operand2.datap as double[];

            npy_intp O1_CalculatedStep = (O1_Step / sizeof(double));
            npy_intp O1_CalculatedOffset = (O1_Offset / sizeof(double));

            npy_intp O2_CalculatedStep = (O2_Step / sizeof(double));
            npy_intp O2_CalculatedOffset = (O2_Offset / sizeof(double));

            npy_intp R_CalculatedStep = (R_Step / sizeof(double));
            npy_intp R_CalculatedOffset = (R_Offset / sizeof(double));

            for (int i = 0; i < N; i++)
            {
                npy_intp O1_Index = ((i * O1_CalculatedStep) + O1_CalculatedOffset);
                npy_intp O2_Index = ((i * O2_CalculatedStep) + O2_CalculatedOffset);
                npy_intp R_Index = ((i * R_CalculatedStep) + R_CalculatedOffset);

                var O1Value = Op1Array[O1_Index];                                            // get operand 1
                var O2Value = Op2Array[O2_Index];                                            // get operand 2
                double retValue;

                // for the common operations, do inline for speed.
                switch (ops)
                {
                    case UFuncOperation.add:
                        retValue = Add(O1Value, O2Value);
                        break;
                    case UFuncOperation.subtract:
                        retValue = Subtract(O1Value, O2Value);
                        break;
                    case UFuncOperation.multiply:
                        retValue = Multiply(O1Value, O2Value);
                        break;
                    case UFuncOperation.divide:
                        retValue = Divide(O1Value, O2Value);
                        break;
                    case UFuncOperation.power:
                        retValue = Power(O1Value, O2Value);
                        break;

                    default:
                        retValue = PerformUFuncOperation(ops, O1Value, O2Value);
                        break;

                }
                retArray[R_Index] = retValue;

            }


        }

        #endregion

        private double PerformUFuncOperation(UFuncOperation op, double aValue, double bValue)
        {
            double destValue = 0;
            bool boolValue = false;

            switch (op)
            {
                case UFuncOperation.add:
                    destValue = Add(aValue, bValue);
                    break;
                case UFuncOperation.subtract:
                    destValue = Subtract(aValue, bValue);
                    break;
                case UFuncOperation.multiply:
                    destValue = Multiply(aValue, bValue);
                    break;
                case UFuncOperation.divide:
                    destValue = Divide(aValue, bValue);
                    break;
                case UFuncOperation.remainder:
                    destValue = Remainder(aValue, bValue);
                    break;
                case UFuncOperation.fmod:
                    destValue = FMod(aValue, bValue);
                    break;
                case UFuncOperation.power:
                    destValue = Power(aValue, bValue);
                    break;
                case UFuncOperation.square:
                    destValue = Square(aValue, bValue);
                    break;
                case UFuncOperation.reciprocal:
                    destValue = Reciprocal(aValue, bValue);
                    break;
                case UFuncOperation.ones_like:
                    destValue = OnesLike(aValue, bValue);
                    break;
                case UFuncOperation.sqrt:
                    destValue = Sqrt(aValue, bValue);
                    break;
                case UFuncOperation.negative:
                    destValue = Negative(aValue, bValue);
                    break;
                case UFuncOperation.absolute:
                    destValue = Absolute(aValue, bValue);
                    break;
                case UFuncOperation.invert:
                    destValue = Invert(aValue, bValue);
                    break;
                case UFuncOperation.left_shift:
                    destValue = LeftShift(aValue, bValue);
                    break;
                case UFuncOperation.right_shift:
                    destValue = RightShift(aValue, bValue);
                    break;
                case UFuncOperation.bitwise_and:
                    destValue = BitWiseAnd(aValue, bValue);
                    break;
                case UFuncOperation.bitwise_xor:
                    destValue = BitWiseXor(aValue, bValue);
                    break;
                case UFuncOperation.bitwise_or:
                    destValue = BitWiseOr(aValue, bValue);
                    break;
                case UFuncOperation.less:
                    boolValue = Less(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.less_equal:
                    boolValue = LessEqual(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.equal:
                    boolValue = Equal(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.not_equal:
                    boolValue = NotEqual(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.greater:
                    boolValue = Greater(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.greater_equal:
                    boolValue = GreaterEqual(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.floor_divide:
                    destValue = FloorDivide(aValue, bValue);
                    break;
                case UFuncOperation.true_divide:
                    destValue = TrueDivide(aValue, bValue);
                    break;
                case UFuncOperation.logical_or:
                    boolValue = LogicalOr(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.logical_and:
                    boolValue = LogicalAnd(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.floor:
                    destValue = Floor(aValue, bValue);
                    break;
                case UFuncOperation.ceil:
                    destValue = Ceiling(aValue, bValue);
                    break;
                case UFuncOperation.maximum:
                    destValue = Maximum(aValue, bValue);
                    break;
                case UFuncOperation.minimum:
                    destValue = Minimum(aValue, bValue);
                    break;
                case UFuncOperation.rint:
                    destValue = Rint(aValue, bValue);
                    break;
                case UFuncOperation.conjugate:
                    destValue = Conjugate(aValue, bValue);
                    break;
                case UFuncOperation.isnan:
                    boolValue = IsNAN(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.fmax:
                    destValue = FMax(aValue, bValue);
                    break;
                case UFuncOperation.fmin:
                    destValue = FMin(aValue, bValue);
                    break;
                case UFuncOperation.heaviside:
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
