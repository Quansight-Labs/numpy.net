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

    #region UFUNC BOOL

    internal class UFUNC_Bool : UFUNC_BASE<bool>, IUFUNC_Operations
    {
        public UFUNC_Bool() : base(sizeof(bool))
        {

        }

        protected override bool ConvertToTemplate(object value)
        {
            return Convert.ToBoolean(value);
        }

        protected override bool PerformUFuncOperation(UFuncOperation op, bool aValue, bool bValue)
        {
            bool destValue = false;

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
                    destValue = false;
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
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                case UFuncOperation.divide:
                    break;
                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    break;
            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            return null;
        }
        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            return null;
        }

        #region bool specific operation handlers
        protected override bool Add(bool aValue, bool bValue)
        {
            return aValue | bValue;
        }

        protected override bool Subtract(bool aValue, bool bValue)
        {
            return aValue | bValue;
        }
 
        protected override bool Multiply(bool aValue, bool bValue)
        {
            return aValue ^ bValue;
        }

        protected override bool Divide(bool aValue, bool bValue)
        {
            return aValue ^ bValue;
        }

        protected override bool Remainder(bool aValue, bool bValue)
        {
            return aValue ^ bValue;
        }
        protected override bool FMod(bool aValue, bool bValue)
        {
            return aValue ^ bValue;
        }
        protected override bool Power(bool aValue, bool bValue)
        {
            return aValue ^ bValue;
        }
        protected override bool Square(bool bValue, bool operand)
        {
            return bValue ^ operand;
        }
        protected override bool Reciprocal(bool bValue, bool operand)
        {
            return bValue ^ operand;
        }
        protected override bool OnesLike(bool bValue, bool operand)
        {
            return true;
        }
        protected override bool Sqrt(bool bValue, bool operand)
        {
            return bValue ^ operand;
        }
        protected override bool Negative(bool bValue, bool operand)
        {
            return bValue ^ operand;
        }
        protected override bool Absolute(bool bValue, bool operand)
        {
            return false;
        }
        protected override bool Invert(bool bValue, bool operand)
        {
            return !bValue;
        }
        protected override bool LeftShift(bool bValue, bool operand)
        {
            return bValue;
        }
        protected override bool RightShift(bool bValue, bool operand)
        {
            return bValue;
        }
        protected override bool BitWiseAnd(bool bValue, bool operand)
        {
            return bValue & operand;
        }
        protected override bool BitWiseXor(bool bValue, bool operand)
        {
            return bValue ^ operand;
        }
        protected override bool BitWiseOr(bool bValue, bool operand)
        {
            return bValue | operand;
        }
        protected override bool Less(bool bValue, bool operand)
        {
            return false;
        }
        protected override bool LessEqual(bool bValue, bool operand)
        {
            return false;
        }
        protected override bool Equal(bool bValue, bool operand)
        {
            return bValue == operand;
        }
        protected override bool NotEqual(bool bValue, bool operand)
        {
            return bValue != operand;
        }
        protected override bool Greater(bool bValue, bool operand)
        {
            return true;
        }
        protected override bool GreaterEqual(bool bValue, bool operand)
        {
            return true;
        }
        protected override bool FloorDivide(bool bValue, bool operand)
        {
            return bValue ^ operand;
        }
        protected override bool TrueDivide(bool bValue, bool operand)
        {
            return bValue ^ operand;
        }
        protected override bool LogicalOr(bool bValue, bool operand)
        {
            return bValue || operand;
        }
        protected bool LogicalOrReduce(bool result, bool[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result || OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected override bool LogicalAnd(bool bValue, bool operand)
        {
            return bValue && operand;
        }
        protected bool LogicalAndReduce(bool result, bool[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result && OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected override bool Floor(bool bValue, bool operand)
        {
            return bValue ^ operand;
        }
        protected override bool Ceiling(bool bValue, bool operand)
        {
            return bValue ^ operand;
        }
        protected override bool Maximum(bool bValue, bool operand)
        {
            if (bValue)
                return true;
            return operand;
        }

        protected override bool Minimum(bool bValue, bool operand)
        {
            if (!bValue)
                return false;
            return operand;
        }

        protected override bool Rint(bool bValue, bool operand)
        {
            return false;
        }
        protected override bool Conjugate(bool bValue, bool operand)
        {
            return bValue;
        }
        protected override bool IsNAN(bool bValue, bool operand)
        {
            return false;
        }
        protected override bool FMax(bool bValue, bool operand)
        {
            if (bValue)
                return true;
            return operand;
        }
        protected override bool FMin(bool bValue, bool operand)
        {
            if (!bValue)
                return false;
            return operand;
        }
        protected override bool Heaviside(bool bValue, bool operand)
        {
            if (bValue == false)
                return operand;
            return bValue;
        }

        #endregion

    }

    #endregion
}
