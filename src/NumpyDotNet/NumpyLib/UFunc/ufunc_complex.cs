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
    #region UFUNC COMPLEX
    internal class UFUNC_Complex : UFUNC_BASE<System.Numerics.Complex>, IUFUNC_Operations
    {
        public UFUNC_Complex() : base(sizeof(double) * 2)
        {

        }

        protected override System.Numerics.Complex ConvertToTemplate(object value)
        {

            if (value is System.Numerics.Complex)
            {
                System.Numerics.Complex c = (System.Numerics.Complex)value;
                return c;
            }
            else
            {
                return new System.Numerics.Complex(Convert.ToDouble(value), 0);
            }

        }

        protected override System.Numerics.Complex PerformUFuncOperation(UFuncOperation op, System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            System.Numerics.Complex destValue = 0;

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

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        #region System.Numerics.Complex specific operation handlers
        protected override System.Numerics.Complex Add(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            return aValue + bValue;
        }
        protected System.Numerics.Complex AddReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected void AddAccumulate(
            System.Numerics.Complex[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
            System.Numerics.Complex[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
            System.Numerics.Complex[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] + Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }

        protected override System.Numerics.Complex Subtract(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            return aValue - bValue;
        }
        protected System.Numerics.Complex SubtractReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result - OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected override System.Numerics.Complex Multiply(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            return aValue * bValue;
        }
        protected System.Numerics.Complex MultiplyReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result * OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected void MultiplyAccumulate(
                System.Numerics.Complex[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                System.Numerics.Complex[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                System.Numerics.Complex[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] * Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }


        protected override System.Numerics.Complex Divide(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        protected System.Numerics.Complex DivideReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
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

        protected override System.Numerics.Complex Remainder(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            if (bValue == 0)
            {
                aValue = 0;
                return aValue;
            }
            var rem = aValue.Real % bValue.Real;
            if ((aValue.Real > 0) == (bValue.Real > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + bValue;
            }
        }
        protected override System.Numerics.Complex FMod(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            if (bValue == 0)
            {
                aValue = 0;
                return aValue;
            }
            return aValue.Real % bValue.Real;
        }
        protected override System.Numerics.Complex Power(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            return System.Numerics.Complex.Pow(aValue, bValue);
        }
        protected override System.Numerics.Complex Square(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return bValue * bValue;
        }
        protected override System.Numerics.Complex Reciprocal(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            if (bValue == 0)
                return 0;

            return 1 / bValue;
        }
        protected override System.Numerics.Complex OnesLike(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return 1;
        }
        protected override System.Numerics.Complex Sqrt(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return System.Numerics.Complex.Sqrt(bValue);
        }
        protected override System.Numerics.Complex Negative(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return -bValue;
        }
        protected override System.Numerics.Complex Absolute(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return System.Numerics.Complex.Abs(bValue);
        }
        protected override System.Numerics.Complex Invert(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return bValue;
        }
        protected override System.Numerics.Complex LeftShift(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            UInt64 rValue = (UInt64)bValue.Real;
            rValue = rValue << Convert.ToInt32(operand.Real);

            UInt64 iValue = (UInt64)bValue.Imaginary;
            iValue = iValue << Convert.ToInt32(operand.Imaginary);

            return new System.Numerics.Complex((double)rValue, (double)iValue);
        }
        protected override System.Numerics.Complex RightShift(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            UInt64 rValue = (UInt64)bValue.Real;
            rValue = rValue >> Convert.ToInt32(operand.Real);

            UInt64 iValue = (UInt64)bValue.Imaginary;
            iValue = iValue >> Convert.ToInt32(operand.Imaginary);

            return new System.Numerics.Complex((double)rValue, (double)iValue);
        }
        protected override System.Numerics.Complex BitWiseAnd(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            UInt64 rValue = (UInt64)bValue.Real;
            rValue = rValue & Convert.ToUInt64(operand.Real);

            UInt64 iValue = (UInt64)bValue.Imaginary;
            iValue = iValue & Convert.ToUInt64(operand.Imaginary);

            return new System.Numerics.Complex((double)rValue, (double)iValue);
        }
        protected override System.Numerics.Complex BitWiseXor(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            UInt64 rValue = (UInt64)bValue.Real;
            rValue = rValue ^ Convert.ToUInt64(operand.Real);

            UInt64 iValue = (UInt64)bValue.Imaginary;
            iValue = iValue ^ Convert.ToUInt64(operand.Imaginary);

            return new System.Numerics.Complex((double)rValue, (double)iValue);
        }
        protected override System.Numerics.Complex BitWiseOr(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            UInt64 rValue = (UInt64)bValue.Real;
            rValue = rValue | Convert.ToUInt64(operand.Real);

            UInt64 iValue = (UInt64)bValue.Imaginary;
            iValue = iValue | Convert.ToUInt64(operand.Imaginary);

            return new System.Numerics.Complex((double)rValue, (double)iValue);
        }
        protected override System.Numerics.Complex Less(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            bool boolValue = false;
            if (operand.Imaginary == 0)
            {
                boolValue =  bValue.Real < operand.Real;
            }

            return boolValue ? 1 : 0;
        }
        protected override System.Numerics.Complex LessEqual(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            bool boolValue = false;
            if (operand.Imaginary == 0)
            {
                boolValue = bValue.Real <= operand.Real;
            }

            return boolValue ? 1 : 0;
        }
        protected override System.Numerics.Complex Equal(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            bool boolValue = bValue == operand;
            return boolValue ? 1 : 0;
        }
        protected override System.Numerics.Complex NotEqual(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            bool boolValue = bValue != operand;
            return boolValue ? 1 : 0;
        }
        protected override System.Numerics.Complex Greater(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            bool boolValue = false;
            if (operand.Imaginary == 0)
            {
                boolValue = bValue.Real > operand.Real;
            }
            return boolValue ? 1 : 0;
        }
        protected override System.Numerics.Complex GreaterEqual(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            bool boolValue = false;
            if (operand.Imaginary == 0)
            {
                boolValue = bValue.Real >= operand.Real;
            }
            return boolValue ? 1 : 0;
        }
        protected override System.Numerics.Complex FloorDivide(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }

            double Real = 0;
            if (operand.Real != 0)
                Real = Math.Floor(bValue.Real / operand.Real);

            double Imaginary = 0;
            if (operand.Imaginary != 0)
                Imaginary = Math.Floor(bValue.Imaginary / operand.Imaginary);

            return new System.Numerics.Complex(Real, Imaginary);
        }
        protected override System.Numerics.Complex TrueDivide(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            if (operand == 0)
                return 0;

            return bValue / operand;
        }
        protected override System.Numerics.Complex LogicalOr(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            bool boolValue = bValue != 0 || operand != 0;
            return boolValue ? 1 : 0;
        }
        protected System.Numerics.Complex LogicalOrReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected override System.Numerics.Complex LogicalAnd(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            bool boolValue = bValue != 0 && operand != 0;
            return boolValue ? 1 : 0;
        }
        protected System.Numerics.Complex LogicalAndReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected override System.Numerics.Complex Floor(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return new System.Numerics.Complex(Math.Floor(bValue.Real), Math.Floor(bValue.Imaginary));
        }
        protected override System.Numerics.Complex Ceiling(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return new System.Numerics.Complex(Math.Ceiling(bValue.Real), Math.Ceiling(bValue.Imaginary));
        }
        protected override System.Numerics.Complex Maximum(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return Math.Max(bValue.Real, operand.Real);
        }
        protected System.Numerics.Complex MaximumReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result.Real, OperandArray[OperIndex].Real);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override System.Numerics.Complex Minimum(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return Math.Min(bValue.Real, operand.Real);
        }
        protected System.Numerics.Complex MinimumReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Min(result.Real, OperandArray[OperIndex].Real);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override System.Numerics.Complex Rint(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return new System.Numerics.Complex(Math.Round(bValue.Real), Math.Round(bValue.Imaginary));
        }
        protected override System.Numerics.Complex Conjugate(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            var cc = new System.Numerics.Complex(bValue.Real, -bValue.Imaginary);
            return cc;
        }
        protected override System.Numerics.Complex IsNAN(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return 0;
        }
        protected override System.Numerics.Complex FMax(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return Math.Max(bValue.Real, operand.Real);
        }
        protected override System.Numerics.Complex FMin(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return Math.Min(bValue.Real, operand.Real);
        }
        protected override System.Numerics.Complex Heaviside(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            if (bValue == 0.0)
                return operand;

            if (bValue.Real < 0.0)
                return 0.0;

            return 1.0;

        }

        #endregion

    }


    #endregion
}
