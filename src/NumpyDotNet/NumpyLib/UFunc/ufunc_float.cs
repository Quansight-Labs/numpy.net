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
    #region UFUNC FLOAT
    internal class UFUNC_Float : UFUNC_BASE<float>, IUFUNC_Operations
    {
        public UFUNC_Float() : base(sizeof(float))
        {

        }

        protected override float ConvertToTemplate(object value)
        {
            return Convert.ToSingle(Convert.ToDouble(value));
        }

        protected override float PerformUFuncOperation(UFuncOperation op, float aValue, float bValue)
        {
            float destValue = 0;

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

        #region float specific operation handlers
        protected override float Add(float aValue, float bValue)
        {
            return aValue + bValue;
        }

        protected override float Subtract(float aValue, float bValue)
        {
            return aValue - bValue;
        }
        protected override float Multiply(float aValue, float bValue)
        {
            return aValue * bValue;
        }

        protected override float Divide(float aValue, float bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        protected override float Remainder(float aValue, float bValue)
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
        protected override float FMod(float aValue, float bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        protected override float Power(float aValue, float bValue)
        {
            return Convert.ToSingle(Math.Pow(aValue, bValue));
        }
        protected override float Square(float bValue, float operand)
        {
            return bValue * bValue;
        }
        protected override float Reciprocal(float bValue, float operand)
        {
            if (bValue == 0)
                return 0;

            return 1 / bValue;
        }
        protected override float OnesLike(float bValue, float operand)
        {
            return 1;
        }
        protected override float Sqrt(float bValue, float operand)
        {
            return Convert.ToSingle(Math.Sqrt(bValue));
        }
        protected override float Negative(float bValue, float operand)
        {
            return -bValue;
        }
        protected override float Absolute(float bValue, float operand)
        {
            return Math.Abs(bValue);
        }
        protected override float Invert(float bValue, float operand)
        {
            return bValue;
        }
        protected override float LeftShift(float bValue, float operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected override float RightShift(float bValue, float operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected override float BitWiseAnd(float bValue, float operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue & Convert.ToUInt64(operand);
        }
        protected override float BitWiseXor(float bValue, float operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue ^ Convert.ToUInt64(operand);
        }
        protected override float BitWiseOr(float bValue, float operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue | Convert.ToUInt64(operand);
        }
        protected override float Less(float bValue, float operand)
        {
            bool boolValue = bValue < operand;
            return boolValue ? 1 : 0;
        }
        protected override float LessEqual(float bValue, float operand)
        {
            bool boolValue = bValue <= operand;
            return boolValue ? 1 : 0;
        }
        protected override float Equal(float bValue, float operand)
        {
            bool boolValue = bValue == operand;
            return boolValue ? 1 : 0;
        }
        protected override float NotEqual(float bValue, float operand)
        {
            bool boolValue = bValue != operand;
            return boolValue ? 1 : 0;
        }
        protected override float Greater(float bValue, float operand)
        {
            bool boolValue = bValue > operand;
            return boolValue ? 1 : 0;
        }
        protected override float GreaterEqual(float bValue, float operand)
        {
            bool boolValue = bValue >= operand;
            return boolValue ? 1 : 0;
        }
        protected override float FloorDivide(float bValue, float operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Convert.ToSingle(Math.Floor(bValue / operand));
        }
        protected override float TrueDivide(float bValue, float operand)
        {
            if (operand == 0)
                return 0;

            return bValue / operand;
        }
        protected override float LogicalOr(float bValue, float operand)
        {
            bool boolValue = bValue != 0 || operand != 0;
            return boolValue ? 1 : 0;
        }
        protected override float LogicalAnd(float bValue, float operand)
        {
            bool boolValue = bValue != 0 && operand != 0;
            return boolValue ? 1 : 0;
        }
        protected override float Floor(float bValue, float operand)
        {
            return Convert.ToSingle(Math.Floor(bValue));
        }
        protected override float Ceiling(float bValue, float operand)
        {
            return Convert.ToSingle(Math.Ceiling(bValue));
        }
        protected override float Maximum(float bValue, float operand)
        {
            return Math.Max(bValue, operand);
        }
        protected override float Minimum(float bValue, float operand)
        {
            return Math.Min(bValue, operand);
        }
        protected override float Rint(float bValue, float operand)
        {
            return Convert.ToSingle(Math.Round(bValue));
        }
        protected override float Conjugate(float bValue, float operand)
        {
            return bValue;
        }
        protected override float IsNAN(float bValue, float operand)
        {
            bool boolValue = float.IsNaN(bValue);
            return boolValue ? 1 : 0;
        }
        protected override float FMax(float bValue, float operand)
        {
            if (float.IsNaN(operand))
                return bValue;
            if (float.IsNaN(bValue))
                return operand;

            return Math.Max(bValue, operand);
        }
        protected override float FMin(float bValue, float operand)
        {
            if (float.IsNaN(operand))
                return bValue;
            if (float.IsNaN(bValue))
                return operand;

            return Math.Min(bValue, operand);
        }
        protected override float Heaviside(float bValue, float operand)
        {
            if (float.IsNaN(bValue))
                return float.NaN;

            if (bValue == 0.0)
                return operand;

            if (bValue < 0.0)
                return 0.0f;

            return 1.0f;

        }

        #endregion

    }


    #endregion
}
