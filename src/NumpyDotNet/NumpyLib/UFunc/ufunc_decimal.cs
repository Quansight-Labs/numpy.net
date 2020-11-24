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
    #region UFUNC DECIMAL
    internal class UFUNC_Decimal : UFUNC_BASE<decimal>, IUFUNC_Operations
    {
        public UFUNC_Decimal() : base(sizeof(decimal))
        {

        }

        protected override decimal ConvertToTemplate(object value)
        {
            return Convert.ToDecimal(value);
        }

        protected override decimal PerformUFuncOperation(UFuncOperation op, decimal aValue, decimal bValue)
        {
            decimal destValue = 0;
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

        #region decimal specific operation handlers
        protected override decimal Add(decimal aValue, decimal bValue)
        {
            return aValue + bValue;
        }

        protected override decimal Subtract(decimal aValue, decimal bValue)
        {
            return aValue - bValue;
        }
        protected override decimal Multiply(decimal aValue, decimal bValue)
        {
            return aValue * bValue;
        }

        protected override decimal Divide(decimal aValue, decimal bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        protected override decimal Remainder(decimal aValue, decimal bValue)
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
        protected override decimal FMod(decimal aValue, decimal bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        protected override decimal Power(decimal aValue, decimal bValue)
        {
            return Convert.ToDecimal(Math.Pow(Convert.ToDouble(aValue), Convert.ToDouble(bValue)));
        }
        protected override decimal Square(decimal bValue, decimal operand)
        {
            return bValue * bValue;
        }
        protected override decimal Reciprocal(decimal bValue, decimal operand)
        {
            if (bValue == 0)
                return 0;

            return 1 / bValue;
        }
        protected override decimal OnesLike(decimal bValue, decimal operand)
        {
            return 1;
        }
        protected override decimal Sqrt(decimal bValue, decimal operand)
        {
            decimal dValue = (decimal)bValue;
            decimal epsilon = 0.0M;

            if (dValue < 0)
                throw new OverflowException("Cannot calculate square root from a negative number");

            decimal current = (decimal)Math.Sqrt((double)dValue), previous;
            do
            {
                previous = current;
                if (previous == 0.0M)
                    return 0;

                current = (previous + dValue / previous) / 2;
            }
            while (Math.Abs(previous - current) > epsilon);
            return current;
        }
        protected override decimal Negative(decimal bValue, decimal operand)
        {
            return -bValue;
        }
        protected override decimal Absolute(decimal bValue, decimal operand)
        {
            return Math.Abs(bValue);
        }
        protected override decimal Invert(decimal bValue, decimal operand)
        {
            return bValue;
        }
        protected override decimal LeftShift(decimal bValue, decimal operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected override decimal RightShift(decimal bValue, decimal operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected override decimal BitWiseAnd(decimal bValue, decimal operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue & Convert.ToUInt64(operand);
        }
        protected override decimal BitWiseXor(decimal bValue, decimal operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue ^ Convert.ToUInt64(operand);
        }
        protected override decimal BitWiseOr(decimal bValue, decimal operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue | Convert.ToUInt64(operand);
        }
        protected override decimal Less(decimal bValue, decimal operand)
        {
            bool boolValue = bValue < operand;
            return boolValue ? 1 : 0;
        }
        protected override decimal LessEqual(decimal bValue, decimal operand)
        {
            bool boolValue = bValue <= operand;
            return boolValue ? 1 : 0;
        }
        protected override decimal Equal(decimal bValue, decimal operand)
        {
            bool boolValue = bValue == operand;
            return boolValue ? 1 : 0;
        }
        protected override decimal NotEqual(decimal bValue, decimal operand)
        {
            bool boolValue = bValue != operand;
            return boolValue ? 1 : 0;
        }
        protected override decimal Greater(decimal bValue, decimal operand)
        {
            bool boolValue = bValue > operand;
            return boolValue ? 1 : 0;
        }
        protected override decimal GreaterEqual(decimal bValue, decimal operand)
        {
            bool boolValue = bValue >= operand;
            return boolValue ? 1 : 0;
        }
        protected override decimal FloorDivide(decimal bValue, decimal operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Math.Floor(bValue / operand);
        }
        protected override decimal TrueDivide(decimal bValue, decimal operand)
        {
            if (operand == 0)
                return 0;

            return bValue / operand;
        }
        private bool LogicalOr(decimal bValue, decimal operand)
        {
            return bValue != 0 || operand != 0;
        }
        private bool LogicalAnd(decimal bValue, decimal operand)
        {
            return bValue != 0 && operand != 0;
        }
        private decimal Floor(decimal bValue, decimal operand)
        {
            return Math.Floor(bValue);
        }
        private decimal Ceiling(decimal bValue, decimal operand)
        {
            return Math.Ceiling(bValue);
        }
        private decimal Maximum(decimal bValue, decimal operand)
        {
            return Math.Max(bValue, operand);
        }
        private decimal Minimum(decimal bValue, decimal operand)
        {
            return Math.Min(bValue, operand);
        }
        private decimal Rint(decimal bValue, decimal operand)
        {
            return Math.Round(bValue);
        }
        private decimal Conjugate(decimal bValue, decimal operand)
        {
            return bValue;
        }
        private bool IsNAN(decimal bValue, decimal operand)
        {
            return false;
        }
        private decimal FMax(decimal bValue, decimal operand)
        {
            return Math.Max(bValue, operand);
        }
        private decimal FMin(decimal bValue, decimal operand)
        {
            return Math.Min(bValue, operand);
        }
        private decimal Heaviside(decimal bValue, decimal operand)
        {
            if (bValue == 0.0m)
                return operand;

            if (bValue < 0.0m)
                return 0.0m;

            return 1.0m;

        }

        #endregion

    }


    #endregion
}
