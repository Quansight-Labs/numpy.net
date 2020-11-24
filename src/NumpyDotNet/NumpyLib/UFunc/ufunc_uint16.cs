﻿/*
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
    #region UFUNC UINT16
    internal class UFUNC_UInt16 : UFUNC_BASE<UInt16>, IUFUNC_Operations
    {
        public UFUNC_UInt16() : base(sizeof(UInt16))
        {

        }

        protected override UInt16 ConvertToTemplate(object value)
        {
            return Convert.ToUInt16(Convert.ToDouble(value));
        }

        protected override UInt16 PerformUFuncOperation(UFuncOperation op, UInt16 aValue, UInt16 bValue)
        {
            UInt16 destValue = 0;
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
                    destValue = (UInt16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.less_equal:
                    boolValue = LessEqual(aValue, bValue);
                    destValue = (UInt16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.equal:
                    boolValue = Equal(aValue, bValue);
                    destValue = (UInt16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.not_equal:
                    boolValue = NotEqual(aValue, bValue);
                    destValue = (UInt16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.greater:
                    boolValue = Greater(aValue, bValue);
                    destValue = (UInt16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.greater_equal:
                    boolValue = GreaterEqual(aValue, bValue);
                    destValue = (UInt16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.floor_divide:
                    destValue = FloorDivide(aValue, bValue);
                    break;
                case UFuncOperation.true_divide:
                    destValue = TrueDivide(aValue, bValue);
                    break;
                case UFuncOperation.logical_or:
                    boolValue = LogicalOr(aValue, bValue);
                    destValue = (UInt16)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.logical_and:
                    boolValue = LogicalAnd(aValue, bValue);
                    destValue = (UInt16)(boolValue ? 1 : 0);
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
                    destValue = (UInt16)(boolValue ? 1 : 0);
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

        #region UInt16 specific operation handlers
        protected override UInt16 Add(UInt16 aValue, UInt16 bValue)
        {
            return (UInt16)(aValue + bValue);
        }

        protected override UInt16 Subtract(UInt16 aValue, UInt16 bValue)
        {
            return (UInt16)(aValue - bValue);
        }
        protected override UInt16 Multiply(UInt16 aValue, UInt16 bValue)
        {
            return (UInt16)(aValue * bValue);
        }

        protected override UInt16 Divide(UInt16 aValue, UInt16 bValue)
        {
            if (bValue == 0)
                return 0;
            return (UInt16)(aValue / bValue);
        }
        protected override UInt16 Remainder(UInt16 aValue, UInt16 bValue)
        {
            if (bValue == 0)
            {
                return 0;
            }
            var rem = aValue % bValue;
            if ((aValue > 0) == (bValue > 0) || rem == 0)
            {
                return (UInt16)(rem);
            }
            else
            {
                return (UInt16)(rem + bValue);
            }
        }
        protected override UInt16 FMod(UInt16 aValue, UInt16 bValue)
        {
            if (bValue == 0)
                return 0;
            return (UInt16)(aValue % bValue);
        }
        protected override UInt16 Power(UInt16 aValue, UInt16 bValue)
        {
            return Convert.ToUInt16(Math.Pow(aValue, bValue));
        }
        protected override UInt16 Square(UInt16 bValue, UInt16 operand)
        {
            return (UInt16)(bValue * bValue);
        }
        protected override UInt16 Reciprocal(UInt16 bValue, UInt16 operand)
        {
            if (bValue == 0)
                return 0;

            return (UInt16)(1 / bValue);
        }
        protected override UInt16 OnesLike(UInt16 bValue, UInt16 operand)
        {
            return 1;
        }
        protected override UInt16 Sqrt(UInt16 bValue, UInt16 operand)
        {
            return Convert.ToUInt16(Math.Sqrt(bValue));
        }
        protected override UInt16 Negative(UInt16 bValue, UInt16 operand)
        {
            return bValue;
        }
        protected override UInt16 Absolute(UInt16 bValue, UInt16 operand)
        {
            return bValue;
        }
        protected override UInt16 Invert(UInt16 bValue, UInt16 operand)
        {
            return (UInt16)(~bValue);
        }
        protected override UInt16 LeftShift(UInt16 bValue, UInt16 operand)
        {
            return (UInt16)(bValue << Convert.ToInt32(operand));
        }
        protected override UInt16 RightShift(UInt16 bValue, UInt16 operand)
        {
            return (UInt16)(bValue >> Convert.ToInt32(operand));
        }
        protected override UInt16 BitWiseAnd(UInt16 bValue, UInt16 operand)
        {
            return (UInt16)(bValue & operand);
        }
        protected override UInt16 BitWiseXor(UInt16 bValue, UInt16 operand)
        {
            return (UInt16)(bValue ^ operand);
        }
        protected override UInt16 BitWiseOr(UInt16 bValue, UInt16 operand)
        {
            return (UInt16)(bValue | operand);
        }
        private bool Less(UInt16 bValue, UInt16 operand)
        {
            return bValue < operand;
        }
        private bool LessEqual(UInt16 bValue, UInt16 operand)
        {
            return bValue <= operand;
        }
        private bool Equal(UInt16 bValue, UInt16 operand)
        {
            return bValue == operand;
        }
        private bool NotEqual(UInt16 bValue, UInt16 operand)
        {
            return bValue != operand;
        }
        private bool Greater(UInt16 bValue, UInt16 operand)
        {
            return bValue > operand;
        }
        private bool GreaterEqual(UInt16 bValue, UInt16 operand)
        {
            return bValue >= (dynamic)operand;
        }
        private UInt16 FloorDivide(UInt16 bValue, UInt16 operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Convert.ToUInt16(Math.Floor(Convert.ToDouble(bValue) / Convert.ToDouble(operand)));
        }
        private UInt16 TrueDivide(UInt16 bValue, UInt16 operand)
        {
            if (operand == 0)
                return 0;

            return (UInt16)(bValue / operand);
        }
        private bool LogicalOr(UInt16 bValue, UInt16 operand)
        {
            return bValue != 0 || operand != 0;
        }
        private bool LogicalAnd(UInt16 bValue, UInt16 operand)
        {
            return bValue != 0 && operand != 0;
        }
        private UInt16 Floor(UInt16 bValue, UInt16 operand)
        {
            return Convert.ToUInt16(Math.Floor(Convert.ToDouble(bValue)));
        }
        private UInt16 Ceiling(UInt16 bValue, UInt16 operand)
        {
            return Convert.ToUInt16(Math.Ceiling(Convert.ToDouble(bValue)));
        }
        private UInt16 Maximum(UInt16 bValue, UInt16 operand)
        {
            return Math.Max(bValue, operand);
        }
        private UInt16 Minimum(UInt16 bValue, UInt16 operand)
        {
            return Math.Min(bValue, operand);
        }
        private UInt16 Rint(UInt16 bValue, UInt16 operand)
        {
            return Convert.ToUInt16(Math.Round(Convert.ToDouble(bValue)));
        }
        private UInt16 Conjugate(UInt16 bValue, UInt16 operand)
        {
            return bValue;
        }
        private bool IsNAN(UInt16 bValue, UInt16 operand)
        {
            return false;
        }
        private UInt16 FMax(UInt16 bValue, UInt16 operand)
        {
            return Math.Max(bValue, operand);
        }
        private UInt16 FMin(UInt16 bValue, UInt16 operand)
        {
            return Math.Min(bValue, operand);
        }
        private UInt16 Heaviside(UInt16 bValue, UInt16 operand)
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
