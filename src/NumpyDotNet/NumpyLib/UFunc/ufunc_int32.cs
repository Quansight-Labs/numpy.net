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

    internal class UFUNC_Int32 : UFUNC_BASE<Int32>, IUFUNC_Operations
    {
        public UFUNC_Int32() : base(sizeof(Int32))
        {

        }

        protected override Int32 ConvertToTemplate(object value)
        {
            return Convert.ToInt32(Convert.ToDouble(value));
        }

        protected override Int32 PerformUFuncOperation(UFuncOperation op, Int32 aValue, Int32 bValue)
        {
            Int32 destValue = 0;

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

        #region Int32 specific operation handlers
        protected override Int32 Add(Int32 aValue, Int32 bValue)
        {
            return aValue + bValue;
        }

        protected override Int32 Subtract(Int32 aValue, Int32 bValue)
        {
            return aValue - bValue;
        }
        protected override Int32 Multiply(Int32 aValue, Int32 bValue)
        {
            return aValue * bValue;
        }

        protected override Int32 Divide(Int32 aValue, Int32 bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        protected override Int32 Remainder(Int32 aValue, Int32 bValue)
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
        protected override Int32 FMod(Int32 aValue, Int32 bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        protected override Int32 Power(Int32 aValue, Int32 bValue)
        {
            return Convert.ToInt32(Math.Pow(aValue, bValue));
        }
        protected override Int32 Square(Int32 bValue, Int32 operand)
        {
            return bValue * bValue;
        }
        protected override Int32 Reciprocal(Int32 bValue, Int32 operand)
        {
            if (bValue == 0)
                return 0;

            return 1 / bValue;
        }
        protected override Int32 OnesLike(Int32 bValue, Int32 operand)
        {
            return 1;
        }
        protected override Int32 Sqrt(Int32 bValue, Int32 operand)
        {
            return Convert.ToInt32(Math.Sqrt(bValue));
        }
        protected override Int32 Negative(Int32 bValue, Int32 operand)
        {
            return -bValue;
        }
        protected override Int32 Absolute(Int32 bValue, Int32 operand)
        {
            return Math.Abs(bValue);
        }
        protected override Int32 Invert(Int32 bValue, Int32 operand)
        {
            return ~bValue;
        }
        protected override Int32 LeftShift(Int32 bValue, Int32 operand)
        {
            return bValue << Convert.ToInt32(operand);
        }
        protected override Int32 RightShift(Int32 bValue, Int32 operand)
        {
            return bValue >> Convert.ToInt32(operand);
        }
        protected override Int32 BitWiseAnd(Int32 bValue, Int32 operand)
        {
            return bValue & operand;
        }
        protected override Int32 BitWiseXor(Int32 bValue, Int32 operand)
        {
            return bValue ^ operand;
        }
        protected override Int32 BitWiseOr(Int32 bValue, Int32 operand)
        {
            return bValue | operand;
        }
        protected override Int32 Less(Int32 bValue, Int32 operand)
        {
            bool boolValue = bValue < operand;
            return boolValue ? 1 : 0;
        }
        protected override Int32 LessEqual(Int32 bValue, Int32 operand)
        {
            bool boolValue = bValue <= operand;
            return boolValue ? 1 : 0;
        }
        protected override Int32 Equal(Int32 bValue, Int32 operand)
        {
            bool boolValue = bValue == operand;
            return boolValue ? 1 : 0;
        }
        protected override Int32 NotEqual(Int32 bValue, Int32 operand)
        {
            bool boolValue = bValue != operand;
            return boolValue ? 1 : 0;
        }
        protected override Int32 Greater(Int32 bValue, Int32 operand)
        {
            bool boolValue = bValue > operand;
            return boolValue ? 1 : 0;
        }
        protected override Int32 GreaterEqual(Int32 bValue, Int32 operand)
        {
            bool boolValue = bValue >= operand;
            return boolValue ? 1 : 0;
        }
        protected override Int32 FloorDivide(Int32 bValue, Int32 operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Convert.ToInt32(Math.Floor(Convert.ToDouble(bValue) / Convert.ToDouble(operand)));
        }
        protected override Int32 TrueDivide(Int32 bValue, Int32 operand)
        {
            if (operand == 0)
                return 0;

            return bValue / operand;
        }
        protected override Int32 LogicalOr(Int32 bValue, Int32 operand)
        {
            bool boolValue = bValue != 0 || operand != 0;
            return boolValue ? 1 : 0;
        }
        protected override Int32 LogicalAnd(Int32 bValue, Int32 operand)
        {
            bool boolValue = bValue != 0 && operand != 0;
            return boolValue ? 1 : 0;
        }
        protected override Int32 Floor(Int32 bValue, Int32 operand)
        {
            return Convert.ToInt32(Math.Floor(Convert.ToDouble(bValue)));
        }
        protected override Int32 Ceiling(Int32 bValue, Int32 operand)
        {
            return Convert.ToInt32(Math.Ceiling(Convert.ToDouble(bValue)));
        }
        protected override Int32 Maximum(Int32 bValue, Int32 operand)
        {
            return Math.Max(bValue, operand);
        }
        protected override Int32 Minimum(Int32 bValue, Int32 operand)
        {
            return Math.Min(bValue, operand);
        }
        protected override Int32 Rint(Int32 bValue, Int32 operand)
        {
            return Convert.ToInt32(Math.Round(Convert.ToDouble(bValue)));
        }
        protected override Int32 Conjugate(Int32 bValue, Int32 operand)
        {
            return bValue;
        }
        protected override Int32 IsNAN(Int32 bValue, Int32 operand)
        {
            return 0;
        }
        protected override Int32 FMax(Int32 bValue, Int32 operand)
        {
            return Math.Max(bValue, operand);
        }
        protected override Int32 FMin(Int32 bValue, Int32 operand)
        {
            return Math.Min(bValue, operand);
        }
        protected override Int32 Heaviside(Int32 bValue, Int32 operand)
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
