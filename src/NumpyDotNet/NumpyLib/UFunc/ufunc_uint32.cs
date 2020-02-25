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
                    destValue = (UInt32)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.less_equal:
                    boolValue = LessEqual(aValue, bValue);
                    destValue = (UInt32)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.equal:
                    boolValue = Equal(aValue, bValue);
                    destValue = (UInt32)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.not_equal:
                    boolValue = NotEqual(aValue, bValue);
                    destValue = (UInt32)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.greater:
                    boolValue = Greater(aValue, bValue);
                    destValue = (UInt32)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.greater_equal:
                    boolValue = GreaterEqual(aValue, bValue);
                    destValue = (UInt32)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.floor_divide:
                    destValue = FloorDivide(aValue, bValue);
                    break;
                case UFuncOperation.true_divide:
                    destValue = TrueDivide(aValue, bValue);
                    break;
                case UFuncOperation.logical_or:
                    boolValue = LogicalOr(aValue, bValue);
                    destValue = (UInt32)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.logical_and:
                    boolValue = LogicalAnd(aValue, bValue);
                    destValue = (UInt32)(boolValue ? 1 : 0);
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
                    destValue = (UInt32)(boolValue ? 1 : 0);
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

        #region UInt32 specific operation handlers
        protected override UInt32 Add(UInt32 aValue, UInt32 bValue)
        {
            return aValue + bValue;
        }

        protected override UInt32 Subtract(UInt32 aValue, UInt32 bValue)
        {
            return aValue - bValue;
        }
        protected override UInt32 Multiply(UInt32 aValue, UInt32 bValue)
        {
            return aValue * bValue;
        }

        protected override UInt32 Divide(UInt32 aValue, UInt32 bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        private UInt32 Remainder(UInt32 aValue, UInt32 bValue)
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
        private UInt32 FMod(UInt32 aValue, UInt32 bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        protected override UInt32 Power(UInt32 aValue, UInt32 bValue)
        {
            return Convert.ToUInt32(Math.Pow(aValue, bValue));
        }
        private UInt32 Square(UInt32 bValue, UInt32 operand)
        {
            return bValue * bValue;
        }
        private UInt32 Reciprocal(UInt32 bValue, UInt32 operand)
        {
            if (bValue == 0)
                return 0;

            return 1 / bValue;
        }
        private UInt32 OnesLike(UInt32 bValue, UInt32 operand)
        {
            return 1;
        }
        private UInt32 Sqrt(UInt32 bValue, UInt32 operand)
        {
            return Convert.ToUInt32(Math.Sqrt(bValue));
        }
        private UInt32 Negative(UInt32 bValue, UInt32 operand)
        {
            return bValue;
        }
        private UInt32 Absolute(UInt32 bValue, UInt32 operand)
        {
            return bValue;
        }
        private UInt32 Invert(UInt32 bValue, UInt32 operand)
        {
            return ~bValue;
        }
        private UInt32 LeftShift(UInt32 bValue, UInt32 operand)
        {
            return bValue << Convert.ToInt32(operand);
        }
        private UInt32 RightShift(UInt32 bValue, UInt32 operand)
        {
            return bValue >> Convert.ToInt32(operand);
        }
        private UInt32 BitWiseAnd(UInt32 bValue, UInt32 operand)
        {
            return bValue & operand;
        }
        private UInt32 BitWiseXor(UInt32 bValue, UInt32 operand)
        {
            return bValue ^ operand;
        }
        private UInt32 BitWiseOr(UInt32 bValue, UInt32 operand)
        {
            return bValue | operand;
        }
        private bool Less(UInt32 bValue, UInt32 operand)
        {
            return bValue < operand;
        }
        private bool LessEqual(UInt32 bValue, UInt32 operand)
        {
            return bValue <= operand;
        }
        private bool Equal(UInt32 bValue, UInt32 operand)
        {
            return bValue == operand;
        }
        private bool NotEqual(UInt32 bValue, UInt32 operand)
        {
            return bValue != operand;
        }
        private bool Greater(UInt32 bValue, UInt32 operand)
        {
            return bValue > operand;
        }
        private bool GreaterEqual(UInt32 bValue, UInt32 operand)
        {
            return bValue >= (dynamic)operand;
        }
        private UInt32 FloorDivide(UInt32 bValue, UInt32 operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Convert.ToUInt32(Math.Floor(Convert.ToDouble(bValue) / Convert.ToDouble(operand)));
        }
        private UInt32 TrueDivide(UInt32 bValue, UInt32 operand)
        {
            if (operand == 0)
                return 0;

            return bValue / operand;
        }
        private bool LogicalOr(UInt32 bValue, UInt32 operand)
        {
            return bValue != 0 || operand != 0;
        }
        private bool LogicalAnd(UInt32 bValue, UInt32 operand)
        {
            return bValue != 0 && operand != 0;
        }
        private UInt32 Floor(UInt32 bValue, UInt32 operand)
        {
            return Convert.ToUInt32(Math.Floor(Convert.ToDouble(bValue)));
        }
        private UInt32 Ceiling(UInt32 bValue, UInt32 operand)
        {
            return Convert.ToUInt32(Math.Ceiling(Convert.ToDouble(bValue)));
        }
        private UInt32 Maximum(UInt32 bValue, UInt32 operand)
        {
            return Math.Max(bValue, operand);
        }
        private UInt32 Minimum(UInt32 bValue, UInt32 operand)
        {
            return Math.Min(bValue, operand);
        }
        private UInt32 Rint(UInt32 bValue, UInt32 operand)
        {
            return Convert.ToUInt32(Math.Round(Convert.ToDouble(bValue)));
        }
        private UInt32 Conjugate(UInt32 bValue, UInt32 operand)
        {
            return bValue;
        }
        private bool IsNAN(UInt32 bValue, UInt32 operand)
        {
            return false;
        }
        private UInt32 FMax(UInt32 bValue, UInt32 operand)
        {
            return Math.Max(bValue, operand);
        }
        private UInt32 FMin(UInt32 bValue, UInt32 operand)
        {
            return Math.Min(bValue, operand);
        }
        private UInt32 Heaviside(UInt32 bValue, UInt32 operand)
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
