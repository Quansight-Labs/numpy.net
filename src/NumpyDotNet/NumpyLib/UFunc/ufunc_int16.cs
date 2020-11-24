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
                    destValue = (Int16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.less_equal:
                    boolValue = LessEqual(aValue, bValue);
                    destValue = (Int16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.equal:
                    boolValue = Equal(aValue, bValue);
                    destValue = (Int16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.not_equal:
                    boolValue = NotEqual(aValue, bValue);
                    destValue = (Int16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.greater:
                    boolValue = Greater(aValue, bValue);
                    destValue = (Int16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.greater_equal:
                    boolValue = GreaterEqual(aValue, bValue);
                    destValue = (Int16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.floor_divide:
                    destValue = FloorDivide(aValue, bValue);
                    break;
                case UFuncOperation.true_divide:
                    destValue = TrueDivide(aValue, bValue);
                    break;
                case UFuncOperation.logical_or:
                    boolValue = LogicalOr(aValue, bValue);
                    destValue = (Int16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.logical_and:
                    boolValue = LogicalAnd(aValue, bValue);
                    destValue = (Int16)(boolValue ? 1 : 0);
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
                    destValue = (Int16)(boolValue ? 1 : 0);
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

        #region Int16 specific operation handlers
        protected override Int16 Add(Int16 aValue, Int16 bValue)
        {
            return (Int16)(aValue + bValue);
        }

        protected override Int16 Subtract(Int16 aValue, Int16 bValue)
        {
            return (Int16)(aValue - bValue);
        }
        protected override Int16 Multiply(Int16 aValue, Int16 bValue)
        {
            return (Int16)(aValue * bValue);
        }

        protected override Int16 Divide(Int16 aValue, Int16 bValue)
        {
            if (bValue == 0)
                return 0;
            return (Int16)(aValue / bValue);
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
        private Int16 OnesLike(Int16 bValue, Int16 operand)
        {
            return 1;
        }
        private Int16 Sqrt(Int16 bValue, Int16 operand)
        {
            return Convert.ToInt16(Math.Sqrt(bValue));
        }
        private Int16 Negative(Int16 bValue, Int16 operand)
        {
            return (Int16)(-bValue);
        }
        private Int16 Absolute(Int16 bValue, Int16 operand)
        {
            return Math.Abs(bValue);
        }
        private Int16 Invert(Int16 bValue, Int16 operand)
        {
            return (Int16)(~bValue);
        }
        private Int16 LeftShift(Int16 bValue, Int16 operand)
        {
            return (Int16)(bValue << Convert.ToInt32(operand));
        }
        private Int16 RightShift(Int16 bValue, Int16 operand)
        {
            return (Int16)(bValue >> Convert.ToInt32(operand));
        }
        private Int16 BitWiseAnd(Int16 bValue, Int16 operand)
        {
            return (Int16)(bValue & operand);
        }
        private Int16 BitWiseXor(Int16 bValue, Int16 operand)
        {
            return (Int16)(bValue ^ operand);
        }
        private Int16 BitWiseOr(Int16 bValue, Int16 operand)
        {
            return (Int16)(bValue | operand);
        }
        private bool Less(Int16 bValue, Int16 operand)
        {
            return bValue < operand;
        }
        private bool LessEqual(Int16 bValue, Int16 operand)
        {
            return bValue <= operand;
        }
        private bool Equal(Int16 bValue, Int16 operand)
        {
            return bValue == operand;
        }
        private bool NotEqual(Int16 bValue, Int16 operand)
        {
            return bValue != operand;
        }
        private bool Greater(Int16 bValue, Int16 operand)
        {
            return bValue > operand;
        }
        private bool GreaterEqual(Int16 bValue, Int16 operand)
        {
            return bValue >= (dynamic)operand;
        }
        private Int16 FloorDivide(Int16 bValue, Int16 operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Convert.ToInt16(Math.Floor(Convert.ToDouble(bValue) / Convert.ToDouble(operand)));
        }
        private Int16 TrueDivide(Int16 bValue, Int16 operand)
        {
            if (operand == 0)
                return 0;

            return (Int16)(bValue / operand);
        }
        private bool LogicalOr(Int16 bValue, Int16 operand)
        {
            return bValue != 0 || operand != 0;
        }
        private bool LogicalAnd(Int16 bValue, Int16 operand)
        {
            return bValue != 0 && operand != 0;
        }
        private Int16 Floor(Int16 bValue, Int16 operand)
        {
            return Convert.ToInt16(Math.Floor(Convert.ToDouble(bValue)));
        }
        private Int16 Ceiling(Int16 bValue, Int16 operand)
        {
            return Convert.ToInt16(Math.Ceiling(Convert.ToDouble(bValue)));
        }
        private Int16 Maximum(Int16 bValue, Int16 operand)
        {
            return Math.Max(bValue, operand);
        }
        private Int16 Minimum(Int16 bValue, Int16 operand)
        {
            return Math.Min(bValue, operand);
        }
        private Int16 Rint(Int16 bValue, Int16 operand)
        {
            return Convert.ToInt16(Math.Round(Convert.ToDouble(bValue)));
        }
        private Int16 Conjugate(Int16 bValue, Int16 operand)
        {
            return bValue;
        }
        private bool IsNAN(Int16 bValue, Int16 operand)
        {
            return false;
        }
        private Int16 FMax(Int16 bValue, Int16 operand)
        {
            return Math.Max(bValue, operand);
        }
        private Int16 FMin(Int16 bValue, Int16 operand)
        {
            return Math.Min(bValue, operand);
        }
        private Int16 Heaviside(Int16 bValue, Int16 operand)
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
