/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018-2021
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
    #region UFUNC UINT64
    internal partial class UFUNC_UInt64 : UFUNC_BASE<UInt64>, IUFUNC_Operations
    {
        public UFUNC_UInt64() : base(sizeof(UInt64))
        {

        }

        protected override UInt64 ConvertToTemplate(object value)
        {
            return Convert.ToUInt64(Convert.ToDouble(value));
        }

        protected override UInt64 PerformUFuncOperation(UFuncOperation op, UInt64 aValue, UInt64 bValue)
        {
            UInt64 destValue = 0;

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

        #region UInt64 specific operation handlers
        protected override UInt64 Add(UInt64 aValue, UInt64 bValue)
        {
            return aValue + bValue;
        }
        protected override UInt64 Subtract(UInt64 aValue, UInt64 bValue)
        {
            return aValue - bValue;
        }
        protected override UInt64 Multiply(UInt64 aValue, UInt64 bValue)
        {
            return aValue * bValue;
        }
        protected override UInt64 Divide(UInt64 aValue, UInt64 bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        protected override UInt64 Remainder(UInt64 aValue, UInt64 bValue)
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
        protected override UInt64 FMod(UInt64 aValue, UInt64 bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        protected override UInt64 Power(UInt64 aValue, UInt64 bValue)
        {
            return Convert.ToUInt64(Math.Pow(aValue, bValue));
        }
        protected override UInt64 Square(UInt64 bValue, UInt64 operand)
        {
            return bValue * bValue;
        }
        protected override UInt64 Reciprocal(UInt64 bValue, UInt64 operand)
        {
            if (bValue == 0)
                return 0;

            return 1 / bValue;
        }
        protected override UInt64 OnesLike(UInt64 bValue, UInt64 operand)
        {
            return 1;
        }
        protected override UInt64 Sqrt(UInt64 bValue, UInt64 operand)
        {
            return Convert.ToUInt64(Math.Sqrt(bValue));
        }
        protected override UInt64 Negative(UInt64 bValue, UInt64 operand)
        {
            return bValue;
        }
        protected override UInt64 Absolute(UInt64 bValue, UInt64 operand)
        {
            return bValue;
        }
        protected override UInt64 Invert(UInt64 bValue, UInt64 operand)
        {
            return ~bValue;
        }
        protected override UInt64 LeftShift(UInt64 bValue, UInt64 operand)
        {
            return bValue << Convert.ToInt32(operand);
        }
        protected override UInt64 RightShift(UInt64 bValue, UInt64 operand)
        {
            return bValue >> Convert.ToInt32(operand);
        }
        protected override UInt64 BitWiseAnd(UInt64 bValue, UInt64 operand)
        {
            return bValue & operand;
        }
        protected override UInt64 BitWiseXor(UInt64 bValue, UInt64 operand)
        {
            return bValue ^ operand;
        }
        protected override UInt64 BitWiseOr(UInt64 bValue, UInt64 operand)
        {
            return bValue | operand;
        }
        protected override UInt64 Less(UInt64 bValue, UInt64 operand)
        {
            bool boolValue = bValue < operand;
            return (UInt64)(boolValue ? 1 : 0);
        }
        protected override UInt64 LessEqual(UInt64 bValue, UInt64 operand)
        {
            bool boolValue = bValue <= operand;
            return (UInt64)(boolValue ? 1 : 0);
        }
        protected override UInt64 Equal(UInt64 bValue, UInt64 operand)
        {
            bool boolValue = bValue == operand;
            return (UInt64)(boolValue ? 1 : 0);
        }
        protected override UInt64 NotEqual(UInt64 bValue, UInt64 operand)
        {
            bool boolValue = bValue != operand;
            return (UInt64)(boolValue ? 1 : 0);
        }
        protected override UInt64 Greater(UInt64 bValue, UInt64 operand)
        {
            bool boolValue = bValue > operand;
            return (UInt64)(boolValue ? 1 : 0);
        }
        protected override UInt64 GreaterEqual(UInt64 bValue, UInt64 operand)
        {
            bool boolValue = bValue >= operand;
            return (UInt64)(boolValue ? 1 : 0);
        }
        protected override UInt64 FloorDivide(UInt64 bValue, UInt64 operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Convert.ToUInt64(Math.Floor(Convert.ToDouble(bValue) / Convert.ToDouble(operand)));
        }
        protected override UInt64 TrueDivide(UInt64 bValue, UInt64 operand)
        {
            if (operand == 0)
                return 0;

            return bValue / operand;
        }
        protected override UInt64 LogicalOr(UInt64 bValue, UInt64 operand)
        {
            bool boolValue = bValue != 0 || operand != 0;
            return (UInt64)(boolValue ? 1 : 0);
        }
        protected override UInt64 LogicalAnd(UInt64 bValue, UInt64 operand)
        {
            bool boolValue = bValue != 0 && operand != 0;
            return (UInt64)(boolValue ? 1 : 0);
        }
        protected override UInt64 Floor(UInt64 bValue, UInt64 operand)
        {
            return Convert.ToUInt64(Math.Floor(Convert.ToDouble(bValue)));
        }
        protected override UInt64 Ceiling(UInt64 bValue, UInt64 operand)
        {
            return Convert.ToUInt64(Math.Ceiling(Convert.ToDouble(bValue)));
        }
        protected override UInt64 Maximum(UInt64 bValue, UInt64 operand)
        {
            return Math.Max(bValue, operand);
        }
        protected override UInt64 Minimum(UInt64 bValue, UInt64 operand)
        {
            return Math.Min(bValue, operand);
        }
        protected override UInt64 Rint(UInt64 bValue, UInt64 operand)
        {
            return Convert.ToUInt64(Math.Round(Convert.ToDouble(bValue)));
        }
        protected override UInt64 Conjugate(UInt64 bValue, UInt64 operand)
        {
            return bValue;
        }
        protected override UInt64 IsNAN(UInt64 bValue, UInt64 operand)
        {
            return 0;
        }
        protected override UInt64 FMax(UInt64 bValue, UInt64 operand)
        {
            return Math.Max(bValue, operand);
        }
        protected override UInt64 FMin(UInt64 bValue, UInt64 operand)
        {
            return Math.Min(bValue, operand);
        }
        protected override UInt64 Heaviside(UInt64 bValue, UInt64 operand)
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
