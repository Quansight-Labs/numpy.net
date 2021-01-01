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
using System.Collections.Concurrent;
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
    #region UFUNC UINT32
    internal class UFUNC_UInt32 : UFUNC_BASE<UInt32>, IUFUNC_Operations
    {
        public UFUNC_UInt32() : base(sizeof(UInt32))
        {

        }

        protected override UInt32 ConvertToTemplate(object value)
        {
            return Convert.ToUInt32(Convert.ToDouble(value));
        }

        protected override UInt32 PerformUFuncOperation(UFuncOperation op, UInt32 aValue, UInt32 bValue)
        {
            UInt32 destValue = 0;

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

        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                    return AddReduce;

                case UFuncOperation.subtract:
                    return SubtractReduce;

                case UFuncOperation.multiply:
                    return MultiplyReduce;

                case UFuncOperation.divide:
                    return DivideReduce;

                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                    return MaximumReduce;

                case UFuncOperation.minimum:
                    return MinimumReduce;

            }

            return null;
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

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        #region UInt32 specific operation handlers
        protected override UInt32 Add(UInt32 aValue, UInt32 bValue)
        {
            return aValue + bValue;
        }
        protected UInt32 AddReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected void AddAccumulate(
                UInt32[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                UInt32[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                UInt32[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] + Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }

        protected override UInt32 Subtract(UInt32 aValue, UInt32 bValue)
        {
            return aValue - bValue;
        }
        protected UInt32 SubtractReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result - OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected override UInt32 Multiply(UInt32 aValue, UInt32 bValue)
        {
            return aValue * bValue;
        }
        protected UInt32 MultiplyReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result * OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected void MultiplyAccumulate(
                UInt32[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                UInt32[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                UInt32[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] * Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }

        protected override UInt32 Divide(UInt32 aValue, UInt32 bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        protected UInt32 DivideReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = 0;
                else
                    result = result / bValue;

                OperIndex += OperStep;
            }

            return result;
        }

        protected override UInt32 Remainder(UInt32 aValue, UInt32 bValue)
        {
            if (bValue == 0)
            {
                return 0;
            }
            var rem = aValue % bValue;
            if ((aValue > 0) == (bValue > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + bValue;
            }
        }
        protected override UInt32 FMod(UInt32 aValue, UInt32 bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        protected override UInt32 Power(UInt32 aValue, UInt32 bValue)
        {
            return Convert.ToUInt32(Math.Pow(aValue, bValue));
        }
        protected override UInt32 Square(UInt32 bValue, UInt32 operand)
        {
            return bValue * bValue;
        }
        protected override UInt32 Reciprocal(UInt32 bValue, UInt32 operand)
        {
            if (bValue == 0)
                return 0;

            return 1 / bValue;
        }
        protected override UInt32 OnesLike(UInt32 bValue, UInt32 operand)
        {
            return 1;
        }
        protected override UInt32 Sqrt(UInt32 bValue, UInt32 operand)
        {
            return Convert.ToUInt32(Math.Sqrt(bValue));
        }
        protected override UInt32 Negative(UInt32 bValue, UInt32 operand)
        {
            return bValue;
        }
        protected override UInt32 Absolute(UInt32 bValue, UInt32 operand)
        {
            return bValue;
        }
        protected override UInt32 Invert(UInt32 bValue, UInt32 operand)
        {
            return ~bValue;
        }
        protected override UInt32 LeftShift(UInt32 bValue, UInt32 operand)
        {
            return bValue << Convert.ToInt32(operand);
        }
        protected override UInt32 RightShift(UInt32 bValue, UInt32 operand)
        {
            return bValue >> Convert.ToInt32(operand);
        }
        protected override UInt32 BitWiseAnd(UInt32 bValue, UInt32 operand)
        {
            return bValue & operand;
        }
        protected override UInt32 BitWiseXor(UInt32 bValue, UInt32 operand)
        {
            return bValue ^ operand;
        }
        protected override UInt32 BitWiseOr(UInt32 bValue, UInt32 operand)
        {
            return bValue | operand;
        }
        protected override UInt32 Less(UInt32 bValue, UInt32 operand)
        {
            bool boolValue = bValue < operand;
            return (UInt32)(boolValue ? 1 : 0);
        }
        protected override UInt32 LessEqual(UInt32 bValue, UInt32 operand)
        {
            bool boolValue = bValue <= operand;
            return (UInt32)(boolValue ? 1 : 0);
        }
        protected override UInt32 Equal(UInt32 bValue, UInt32 operand)
        {
            bool boolValue = bValue == operand;
            return (UInt32)(boolValue ? 1 : 0);
        }
        protected override UInt32 NotEqual(UInt32 bValue, UInt32 operand)
        {
            bool boolValue = bValue != operand;
            return (UInt32)(boolValue ? 1 : 0);
        }
        protected override UInt32 Greater(UInt32 bValue, UInt32 operand)
        {
            bool boolValue = bValue > operand;
            return (UInt32)(boolValue ? 1 : 0);
        }
        protected override UInt32 GreaterEqual(UInt32 bValue, UInt32 operand)
        {
            bool boolValue = bValue >= operand;
            return (UInt32)(boolValue ? 1 : 0);
        }
        protected override UInt32 FloorDivide(UInt32 bValue, UInt32 operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Convert.ToUInt32(Math.Floor(Convert.ToDouble(bValue) / Convert.ToDouble(operand)));
        }
        protected override UInt32 TrueDivide(UInt32 bValue, UInt32 operand)
        {
            if (operand == 0)
                return 0;

            return bValue / operand;
        }
        protected override UInt32 LogicalOr(UInt32 bValue, UInt32 operand)
        {
            bool boolValue = bValue != 0 || operand != 0;
            return (UInt32)(boolValue ? 1 : 0);
        }
        protected UInt32 LogicalOrReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = (UInt32)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override UInt32 LogicalAnd(UInt32 bValue, UInt32 operand)
        {
            bool boolValue = bValue != 0 && operand != 0;
            return (UInt32)(boolValue ? 1 : 0);
        }
        protected UInt32 LogicalAndReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = (UInt32)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }

        protected override UInt32 Floor(UInt32 bValue, UInt32 operand)
        {
            return Convert.ToUInt32(Math.Floor(Convert.ToDouble(bValue)));
        }
        protected override UInt32 Ceiling(UInt32 bValue, UInt32 operand)
        {
            return Convert.ToUInt32(Math.Ceiling(Convert.ToDouble(bValue)));
        }
        protected override UInt32 Maximum(UInt32 bValue, UInt32 operand)
        {
            return Math.Max(bValue, operand);
        }
        protected UInt32 MaximumReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override UInt32 Minimum(UInt32 bValue, UInt32 operand)
        {
            return Math.Min(bValue, operand);
        }
        protected UInt32 MinimumReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Min(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override UInt32 Rint(UInt32 bValue, UInt32 operand)
        {
            return Convert.ToUInt32(Math.Round(Convert.ToDouble(bValue)));
        }
        protected override UInt32 Conjugate(UInt32 bValue, UInt32 operand)
        {
            return bValue;
        }
        protected override UInt32 IsNAN(UInt32 bValue, UInt32 operand)
        {
            return 0;
        }
        protected override UInt32 FMax(UInt32 bValue, UInt32 operand)
        {
            return Math.Max(bValue, operand);
        }
        protected override UInt32 FMin(UInt32 bValue, UInt32 operand)
        {
            return Math.Min(bValue, operand);
        }
        protected override UInt32 Heaviside(UInt32 bValue, UInt32 operand)
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
