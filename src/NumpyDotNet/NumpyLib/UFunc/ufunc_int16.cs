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

    #region UFUNC INT16

    internal class UFUNC_Int16 : UFUNC_BASE<Int16>, IUFUNC_Operations
    {
        public UFUNC_Int16() : base(sizeof(Int16))
        {

        }

        protected override Int16 ConvertToTemplate(object value)
        {
            return Convert.ToInt16(value);
        }

        protected override Int16 PerformUFuncOperation(UFuncOperation op, Int16 aValue, Int16 bValue)
        {
            Int16 destValue = 0;

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
                    destValue = Less(aValue, bValue);
                    break;
                case UFuncOperation.less_equal:
                    destValue = LessEqual(aValue, bValue);
                    break;
                case UFuncOperation.equal:
                    destValue = Equal(aValue, bValue);
                    break;
                case UFuncOperation.not_equal:
                    destValue = NotEqual(aValue, bValue);
                    break;
                case UFuncOperation.greater:
                    destValue = Greater(aValue, bValue);
                    break;
                case UFuncOperation.greater_equal:
                    destValue = GreaterEqual(aValue, bValue);
                    break;
                case UFuncOperation.floor_divide:
                    destValue = FloorDivide(aValue, bValue);
                    break;
                case UFuncOperation.true_divide:
                    destValue = TrueDivide(aValue, bValue);
                    break;
                case UFuncOperation.logical_or:
                    destValue = LogicalOr(aValue, bValue);
                    break;
                case UFuncOperation.logical_and:
                    destValue = LogicalAnd(aValue, bValue);
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
                    destValue = IsNAN(aValue, bValue);
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

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                    return AddAccumulate;
                case UFuncOperation.multiply:
                    return MultiplyAccumulate;
            }

            return null;
        }

        #region Int16 specific operation handlers
        protected override Int16 Add(Int16 aValue, Int16 bValue)
        {
            return (Int16)(aValue + bValue);
        }
        protected override Int16 AddReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (Int16)(result + OperandArray[OperIndex]);
                OperIndex += OperStep;
            }
            return result;
        }
        protected void AddAccumulate(
                Int16[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                Int16[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                Int16[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = (Int16)(Op1Array[O1_Index] + Op2Array[O2_Index]);

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }


        protected override Int16 Subtract(Int16 aValue, Int16 bValue)
        {
            return (Int16)(aValue - bValue);
        }
        protected override Int16 SubtractReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (Int16)(result - OperandArray[OperIndex]);
                OperIndex += OperStep;
            }
            return result;
        }

        protected override Int16 Multiply(Int16 aValue, Int16 bValue)
        {
            return (Int16)(aValue * bValue);
        }
        protected override Int16 MultiplyReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (Int16)(result * OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected void MultiplyAccumulate(
                Int16[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                Int16[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                Int16[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = (Int16)(Op1Array[O1_Index] * Op2Array[O2_Index]);

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }

        protected override Int16 Divide(Int16 aValue, Int16 bValue)
        {
            if (bValue == 0)
                return 0;
            return (Int16)(aValue / bValue);
        }
        protected override Int16 DivideReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = 0;
                else
                    result = (Int16)(result / bValue);

                OperIndex += OperStep;
            }

            return result;
        }
        protected override Int16 Remainder(Int16 aValue, Int16 bValue)
        {
            if (bValue == 0)
            {
                return 0;
            }
            var rem = aValue % bValue;
            if ((aValue > 0) == (bValue > 0) || rem == 0)
            {
                return (Int16)(rem);
            }
            else
            {
                return (Int16)(rem + bValue);
            }
        }
        protected override Int16 FMod(Int16 aValue, Int16 bValue)
        {
            if (bValue == 0)
                return 0;
            return (Int16)(aValue % bValue);
        }
        protected override Int16 Power(Int16 aValue, Int16 bValue)
        {
            return Convert.ToInt16(Math.Pow(aValue, bValue));
        }
        protected override Int16 Square(Int16 bValue, Int16 operand)
        {
            return (Int16)(bValue * bValue);
        }
        protected override Int16 Reciprocal(Int16 bValue, Int16 operand)
        {
            if (bValue == 0)
                return 0;

            return (Int16)(1 / bValue);
        }
        protected override Int16 OnesLike(Int16 bValue, Int16 operand)
        {
            return 1;
        }
        protected override Int16 Sqrt(Int16 bValue, Int16 operand)
        {
            return Convert.ToInt16(Math.Sqrt(bValue));
        }
        protected override Int16 Negative(Int16 bValue, Int16 operand)
        {
            return (Int16)(-bValue);
        }
        protected override Int16 Absolute(Int16 bValue, Int16 operand)
        {
            return Math.Abs(bValue);
        }
        protected override Int16 Invert(Int16 bValue, Int16 operand)
        {
            return (Int16)(~bValue);
        }
        protected override Int16 LeftShift(Int16 bValue, Int16 operand)
        {
            return (Int16)(bValue << Convert.ToInt32(operand));
        }
        protected override Int16 RightShift(Int16 bValue, Int16 operand)
        {
            return (Int16)(bValue >> Convert.ToInt32(operand));
        }
        protected override Int16 BitWiseAnd(Int16 bValue, Int16 operand)
        {
            return (Int16)(bValue & operand);
        }
        protected override Int16 BitWiseXor(Int16 bValue, Int16 operand)
        {
            return (Int16)(bValue ^ operand);
        }
        protected override Int16 BitWiseOr(Int16 bValue, Int16 operand)
        {
            return (Int16)(bValue | operand);
        }
        protected override Int16 Less(Int16 bValue, Int16 operand)
        {
            bool boolValue = bValue < operand;
            return (Int16)(boolValue ? 1 : 0);
        }
        protected override Int16 LessEqual(Int16 bValue, Int16 operand)
        {
            bool boolValue = bValue <= operand;
            return (Int16)(boolValue ? 1 : 0);
        }
        protected override Int16 Equal(Int16 bValue, Int16 operand)
        {
            bool boolValue = bValue == operand;
            return (Int16)(boolValue ? 1 : 0);
        }
        protected override Int16 NotEqual(Int16 bValue, Int16 operand)
        {
            bool boolValue = bValue != operand;
            return (Int16)(boolValue ? 1 : 0);
        }
        protected override Int16 Greater(Int16 bValue, Int16 operand)
        {
            bool boolValue = bValue > operand;
            return (Int16)(boolValue ? 1 : 0);
        }
        protected override Int16 GreaterEqual(Int16 bValue, Int16 operand)
        {
            bool boolValue = bValue >= operand;
            return (Int16)(boolValue ? 1 : 0);
        }
        protected override Int16 FloorDivide(Int16 bValue, Int16 operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Convert.ToInt16(Math.Floor(Convert.ToDouble(bValue) / Convert.ToDouble(operand)));
        }
        protected override Int16 TrueDivide(Int16 bValue, Int16 operand)
        {
            if (operand == 0)
                return 0;

            return (Int16)(bValue / operand);
        }
        protected override Int16 LogicalOr(Int16 bValue, Int16 operand)
        {
            bool boolValue = bValue != 0 || operand != 0;
            return (Int16)(boolValue ? 1 : 0);
        }
        protected override Int16 LogicalOrReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = (Int16)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override Int16 LogicalAnd(Int16 bValue, Int16 operand)
        {
            bool boolValue = bValue != 0 && operand != 0;
            return (Int16)(boolValue ? 1 : 0);
        }
        protected override Int16 LogicalAndReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = (Int16)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override Int16 Floor(Int16 bValue, Int16 operand)
        {
            return Convert.ToInt16(Math.Floor(Convert.ToDouble(bValue)));
        }
        protected override Int16 Ceiling(Int16 bValue, Int16 operand)
        {
            return Convert.ToInt16(Math.Ceiling(Convert.ToDouble(bValue)));
        }
        protected override Int16 Maximum(Int16 bValue, Int16 operand)
        {
            return Math.Max(bValue, operand);
        }
        protected override Int16 MaximumReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override Int16 Minimum(Int16 bValue, Int16 operand)
        {
            return Math.Min(bValue, operand);
        }
        protected override Int16 MinimumReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Min(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override Int16 Rint(Int16 bValue, Int16 operand)
        {
            return Convert.ToInt16(Math.Round(Convert.ToDouble(bValue)));
        }
        protected override Int16 Conjugate(Int16 bValue, Int16 operand)
        {
            return bValue;
        }
        protected override Int16 IsNAN(Int16 bValue, Int16 operand)
        {
            return 0;
        }
        protected override Int16 FMax(Int16 bValue, Int16 operand)
        {
            return Math.Max(bValue, operand);
        }
        protected override Int16 FMin(Int16 bValue, Int16 operand)
        {
            return Math.Min(bValue, operand);
        }
        protected override Int16 Heaviside(Int16 bValue, Int16 operand)
        {
            if (bValue == 0)
                return operand;

            if (bValue < 0)
                return 0;

            return 1;

        }

        #endregion

    }

    #endregion
}
