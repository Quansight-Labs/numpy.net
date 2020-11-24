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

    #region UFUNC BYTE

    internal class UFUNC_UByte : UFUNC_BASE<byte>, IUFUNC_Operations
    {
        public UFUNC_UByte() : base(sizeof(byte))
        {

        }

        protected override Byte ConvertToTemplate(object value)
        {
            return Convert.ToByte(value);
        }

        protected override Byte PerformUFuncOperation(UFuncOperation op, Byte aValue, Byte bValue)
        {
            Byte destValue = 0;
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
                    destValue = (Byte)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.less_equal:
                    boolValue = LessEqual(aValue, bValue);
                    destValue = (Byte)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.equal:
                    boolValue = Equal(aValue, bValue);
                    destValue = (Byte)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.not_equal:
                    boolValue = NotEqual(aValue, bValue);
                    destValue = (Byte)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.greater:
                    boolValue = Greater(aValue, bValue);
                    destValue = (Byte)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.greater_equal:
                    boolValue = GreaterEqual(aValue, bValue);
                    destValue = (Byte)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.floor_divide:
                    destValue = FloorDivide(aValue, bValue);
                    break;
                case UFuncOperation.true_divide:
                    destValue = TrueDivide(aValue, bValue);
                    break;
                case UFuncOperation.logical_or:
                    boolValue = LogicalOr(aValue, bValue);
                    destValue = (Byte)(boolValue ? 1 : 0);
                    break;
                case UFuncOperation.logical_and:
                    boolValue = LogicalAnd(aValue, bValue);
                    destValue = (Byte)(boolValue ? 1 : 0);
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
                    destValue = (Byte)(boolValue ? 1 : 0);
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

        #region Byte specific operation handlers
        protected override Byte Add(Byte aValue, Byte bValue)
        {
            return (Byte)(aValue + bValue);
        }

        protected override Byte Subtract(Byte aValue, Byte bValue)
        {
            return (Byte)(aValue - bValue);
        }
        protected override Byte Multiply(Byte aValue, Byte bValue)
        {
            return (Byte)(aValue * bValue);
        }

        protected override Byte Divide(Byte aValue, Byte bValue)
        {
            if (bValue == 0)
                return 0;
            return (Byte)(aValue / bValue);
        }
        protected override Byte Remainder(Byte aValue, Byte bValue)
        {
            if (bValue == 0)
            {
                return 0;
            }
            var rem = aValue % bValue;
            if ((aValue > 0) == (bValue > 0) || rem == 0)
            {
                return (Byte)(rem);
            }
            else
            {
                return (Byte)(rem + bValue);
            }
        }
        protected override Byte FMod(Byte aValue, Byte bValue)
        {
            if (bValue == 0)
                return 0;
            return (Byte)(aValue % bValue);
        }
        protected override Byte Power(Byte aValue, Byte bValue)
        {
            return Convert.ToByte(Math.Pow(aValue, bValue));
        }
        private Byte Square(Byte bValue, Byte operand)
        {
            return (Byte)(bValue * bValue);
        }
        private Byte Reciprocal(Byte bValue, Byte operand)
        {
            if (bValue == 0)
                return 0;

            return (Byte)(1 / bValue);
        }
        private Byte OnesLike(Byte bValue, Byte operand)
        {
            return 1;
        }
        private Byte Sqrt(Byte bValue, Byte operand)
        {
            return Convert.ToByte(Math.Sqrt(bValue));
        }
        private Byte Negative(Byte bValue, Byte operand)
        {
            return (Byte)(-bValue);
        }
        private Byte Absolute(Byte bValue, Byte operand)
        {
            return Convert.ToByte(Math.Abs(bValue));
        }
        private Byte Invert(Byte bValue, Byte operand)
        {
            return (Byte)(~bValue);
        }
        private Byte LeftShift(Byte bValue, Byte operand)
        {
            return (Byte)(bValue << Convert.ToInt32(operand));
        }
        private Byte RightShift(Byte bValue, Byte operand)
        {
            return (Byte)(bValue >> Convert.ToInt32(operand));
        }
        private Byte BitWiseAnd(Byte bValue, Byte operand)
        {
            return (Byte)(bValue & operand);
        }
        private Byte BitWiseXor(Byte bValue, Byte operand)
        {
            return (Byte)(bValue ^ operand);
        }
        private Byte BitWiseOr(Byte bValue, Byte operand)
        {
            return (Byte)(bValue | operand);
        }
        private bool Less(Byte bValue, Byte operand)
        {
            return bValue < operand;
        }
        private bool LessEqual(Byte bValue, Byte operand)
        {
            return bValue <= operand;
        }
        private bool Equal(Byte bValue, Byte operand)
        {
            return bValue == operand;
        }
        private bool NotEqual(Byte bValue, Byte operand)
        {
            return bValue != operand;
        }
        private bool Greater(Byte bValue, Byte operand)
        {
            return bValue > operand;
        }
        private bool GreaterEqual(Byte bValue, Byte operand)
        {
            return bValue >= (dynamic)operand;
        }
        private Byte FloorDivide(Byte bValue, Byte operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Convert.ToByte(Math.Floor(Convert.ToDouble(bValue) / Convert.ToDouble(operand)));
        }
        private Byte TrueDivide(Byte bValue, Byte operand)
        {
            if (operand == 0)
                return 0;

            return (Byte)(bValue / operand);
        }
        private bool LogicalOr(Byte bValue, Byte operand)
        {
            return bValue != 0 || operand != 0;
        }
        private bool LogicalAnd(Byte bValue, Byte operand)
        {
            return bValue != 0 && operand != 0;
        }
        private Byte Floor(Byte bValue, Byte operand)
        {
            return Convert.ToByte(Math.Floor(Convert.ToDouble(bValue)));
        }
        private Byte Ceiling(Byte bValue, Byte operand)
        {
            return Convert.ToByte(Math.Ceiling(Convert.ToDouble(bValue)));
        }
        private Byte Maximum(Byte bValue, Byte operand)
        {
            return Math.Max(bValue, operand);
        }
        private Byte Minimum(Byte bValue, Byte operand)
        {
            return Math.Min(bValue, operand);
        }
        private Byte Rint(Byte bValue, Byte operand)
        {
            return Convert.ToByte(Math.Round(Convert.ToDouble(bValue)));
        }
        private Byte Conjugate(Byte bValue, Byte operand)
        {
            return bValue;
        }
        private bool IsNAN(Byte bValue, Byte operand)
        {
            return false;
        }
        private Byte FMax(Byte bValue, Byte operand)
        {
            return Math.Max(bValue, operand);
        }
        private Byte FMin(Byte bValue, Byte operand)
        {
            return Math.Min(bValue, operand);
        }
        private Byte Heaviside(Byte bValue, Byte operand)
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
