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
    #region UFUNC BIGINT
    internal class UFUNC_BigInt : UFUNC_BASE<System.Numerics.BigInteger>, IUFUNC_Operations
    {
        public UFUNC_BigInt() : base(sizeof(double) * 4)
        {

        }

        protected override System.Numerics.BigInteger ConvertToTemplate(object value)
        {

            if (value is System.Numerics.BigInteger)
            {
                System.Numerics.BigInteger c = (System.Numerics.BigInteger)value;
                return c;
            }
            else
            {
                return new System.Numerics.BigInteger(Convert.ToDouble(value));
            }

        }

        protected override System.Numerics.BigInteger PerformUFuncOperation(UFuncOperation op, System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            System.Numerics.BigInteger destValue = 0;

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


        #region System.Numerics.BigInteger specific operation handlers
        protected override System.Numerics.BigInteger Add(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            return aValue + bValue;
        }
        protected System.Numerics.BigInteger AddReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }

        protected void AddAccumulate(
                 System.Numerics.BigInteger[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                 System.Numerics.BigInteger[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                 System.Numerics.BigInteger[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] + Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }

        protected override System.Numerics.BigInteger Subtract(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            return aValue - bValue;
        }
        protected System.Numerics.BigInteger SubtractReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result - OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected override System.Numerics.BigInteger Multiply(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            return aValue * bValue;
        }
        protected System.Numerics.BigInteger MultiplyReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result * OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected void MultiplyAccumulate(
                System.Numerics.BigInteger[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                System.Numerics.BigInteger[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                System.Numerics.BigInteger[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] * Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }

        protected override System.Numerics.BigInteger Divide(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        protected System.Numerics.BigInteger DivideReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
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

        protected override System.Numerics.BigInteger Remainder(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            if (bValue == 0)
            {
                aValue = 0;
                return aValue;
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
        protected override System.Numerics.BigInteger FMod(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            if (bValue == 0)
            {
                aValue = 0;
                return aValue;
            }
            return aValue % bValue;
        }
        protected override System.Numerics.BigInteger Power(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            return System.Numerics.BigInteger.Pow(aValue, (int)bValue);
        }
        protected override System.Numerics.BigInteger Square(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue * bValue;
        }
        protected override System.Numerics.BigInteger Reciprocal(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            if (bValue == 0)
                return 0;

            return 1 / bValue;
        }
        protected override System.Numerics.BigInteger OnesLike(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return 1;
        }
        protected override System.Numerics.BigInteger Sqrt(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            var dd = Math.Round(Math.Pow(Math.E, System.Numerics.BigInteger.Log((System.Numerics.BigInteger)bValue) / 2));
            return new System.Numerics.BigInteger(dd);
        }
        protected override System.Numerics.BigInteger Negative(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return -bValue;
        }
        protected override System.Numerics.BigInteger Absolute(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return System.Numerics.BigInteger.Abs(bValue);
        }
        protected override System.Numerics.BigInteger Invert(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue;
        }
        protected override System.Numerics.BigInteger LeftShift(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            UInt64 rValue = (UInt64)bValue;
            rValue = rValue << Convert.ToInt32((Int64)operand);
            return new System.Numerics.BigInteger(rValue);
        }
        protected override System.Numerics.BigInteger RightShift(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            UInt64 rValue = (UInt64)bValue;
            rValue = rValue >> Convert.ToInt32((Int64)operand);
            return new System.Numerics.BigInteger(rValue);

        }
        protected override System.Numerics.BigInteger BitWiseAnd(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            UInt64 rValue = (UInt64)bValue;
            rValue = rValue & Convert.ToUInt64((UInt64)operand);

            return new System.Numerics.BigInteger(rValue);
        }
        protected override System.Numerics.BigInteger BitWiseXor(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            UInt64 rValue = (UInt64)bValue;
            rValue = rValue ^ Convert.ToUInt64((UInt64)operand);

            return new System.Numerics.BigInteger(rValue);
        }
        protected override System.Numerics.BigInteger BitWiseOr(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            UInt64 rValue = (UInt64)bValue;
            rValue = rValue | Convert.ToUInt64((UInt64)operand);

            return new System.Numerics.BigInteger(rValue);
        }
        protected override System.Numerics.BigInteger Less(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            bool boolValue = bValue < operand;
            return boolValue ? 1 : 0;
        }
        protected override System.Numerics.BigInteger LessEqual(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            bool boolValue = bValue <= operand;
            return boolValue ? 1 : 0;
        }
        protected override System.Numerics.BigInteger Equal(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            bool boolValue = bValue == operand;
            return boolValue ? 1 : 0;
        }
        protected override System.Numerics.BigInteger NotEqual(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            bool boolValue = bValue != operand;
            return boolValue ? 1 : 0;
        }
        protected override System.Numerics.BigInteger Greater(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            bool boolValue = bValue > operand;
            return boolValue ? 1 : 0;
        }
        protected override System.Numerics.BigInteger GreaterEqual(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            bool boolValue = bValue >= operand;
            return boolValue ? 1 : 0;
        }
        protected override System.Numerics.BigInteger FloorDivide(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return bValue / operand;
        }
        protected override System.Numerics.BigInteger TrueDivide(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            if (operand == 0)
                return 0;

            return bValue / operand;
        }
        protected override System.Numerics.BigInteger LogicalOr(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            bool boolValue = bValue != 0 || operand != 0;
            return boolValue ? 1 : 0;
        }
        protected System.Numerics.BigInteger LogicalOrReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected override System.Numerics.BigInteger LogicalAnd(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            bool boolValue = bValue != 0 && operand != 0;
            return boolValue ? 1 : 0;
        }
        protected System.Numerics.BigInteger LogicalAndReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected override System.Numerics.BigInteger Floor(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue;
        }
        protected override System.Numerics.BigInteger Ceiling(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue;
        }
        protected override System.Numerics.BigInteger Maximum(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return System.Numerics.BigInteger.Max(bValue, operand);
        }
        protected System.Numerics.BigInteger MaximumReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = System.Numerics.BigInteger.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override System.Numerics.BigInteger Minimum(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return System.Numerics.BigInteger.Min(bValue, operand);
        }
        protected System.Numerics.BigInteger MinimumReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = System.Numerics.BigInteger.Min(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override System.Numerics.BigInteger Rint(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue;
        }
        protected override System.Numerics.BigInteger Conjugate(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue;
        }
        protected override System.Numerics.BigInteger IsNAN(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return 0;
        }
        protected override System.Numerics.BigInteger FMax(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return System.Numerics.BigInteger.Max(bValue, operand);
        }
        protected override System.Numerics.BigInteger FMin(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return System.Numerics.BigInteger.Min(bValue, operand);
        }
        protected override System.Numerics.BigInteger Heaviside(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
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
