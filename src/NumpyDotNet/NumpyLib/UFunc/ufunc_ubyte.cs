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
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                case UFuncOperation.divide:
                case UFuncOperation.logical_or:
                case UFuncOperation.logical_and:
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
        protected override Byte Square(Byte bValue, Byte operand)
        {
            return (Byte)(bValue * bValue);
        }
        protected override Byte Reciprocal(Byte bValue, Byte operand)
        {
            if (bValue == 0)
                return 0;

            return (Byte)(1 / bValue);
        }
        protected override Byte OnesLike(Byte bValue, Byte operand)
        {
            return 1;
        }
        protected override Byte Sqrt(Byte bValue, Byte operand)
        {
            return Convert.ToByte(Math.Sqrt(bValue));
        }
        protected override Byte Negative(Byte bValue, Byte operand)
        {
            return (Byte)(-bValue);
        }
        protected override Byte Absolute(Byte bValue, Byte operand)
        {
            return Convert.ToByte(Math.Abs(bValue));
        }
        protected override Byte Invert(Byte bValue, Byte operand)
        {
            return (Byte)(~bValue);
        }
        protected override Byte LeftShift(Byte bValue, Byte operand)
        {
            return (Byte)(bValue << Convert.ToInt32(operand));
        }
        protected override Byte RightShift(Byte bValue, Byte operand)
        {
            return (Byte)(bValue >> Convert.ToInt32(operand));
        }
        protected override Byte BitWiseAnd(Byte bValue, Byte operand)
        {
            return (Byte)(bValue & operand);
        }
        protected override Byte BitWiseXor(Byte bValue, Byte operand)
        {
            return (Byte)(bValue ^ operand);
        }
        protected override Byte BitWiseOr(Byte bValue, Byte operand)
        {
            return (Byte)(bValue | operand);
        }
        protected override Byte Less(Byte bValue, Byte operand)
        {
            bool boolValue = bValue < operand;
            return (Byte)(boolValue ? 1 : 0);
        }
        protected override Byte LessEqual(Byte bValue, Byte operand)
        {
            bool boolValue = bValue <= operand;
            return (Byte)(boolValue ? 1 : 0);
        }
        protected override Byte Equal(Byte bValue, Byte operand)
        {
            bool boolValue = bValue == operand;
            return (Byte)(boolValue ? 1 : 0);
        }
        protected override Byte NotEqual(Byte bValue, Byte operand)
        {
            bool boolValue = bValue != operand;
            return (Byte)(boolValue ? 1 : 0);
        }
        protected override Byte Greater(Byte bValue, Byte operand)
        {
            bool boolValue = bValue > operand;
            return (Byte)(boolValue ? 1 : 0);
        }
        protected override Byte GreaterEqual(Byte bValue, Byte operand)
        {
            bool boolValue = bValue >= operand;
            return (Byte)(boolValue ? 1 : 0);
        }
        protected override Byte FloorDivide(Byte bValue, Byte operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Convert.ToByte(Math.Floor(Convert.ToDouble(bValue) / Convert.ToDouble(operand)));
        }
        protected override Byte TrueDivide(Byte bValue, Byte operand)
        {
            if (operand == 0)
                return 0;

            return (Byte)(bValue / operand);
        }
        protected override Byte LogicalOr(Byte bValue, Byte operand)
        {
            bool boolValue = bValue != 0 || operand != 0;
            return (Byte)(boolValue ? 1 : 0);
        }
        protected override Byte LogicalAnd(Byte bValue, Byte operand)
        {
            bool boolValue = bValue != 0 && operand != 0;
            return (Byte)(boolValue ? 1 : 0);
        }

        protected override Byte Floor(Byte bValue, Byte operand)
        {
            return Convert.ToByte(Math.Floor(Convert.ToDouble(bValue)));
        }
        protected override Byte Ceiling(Byte bValue, Byte operand)
        {
            return Convert.ToByte(Math.Ceiling(Convert.ToDouble(bValue)));
        }
        protected override Byte Maximum(Byte bValue, Byte operand)
        {
            return Math.Max(bValue, operand);
        }
 
        protected override Byte Minimum(Byte bValue, Byte operand)
        {
            return Math.Min(bValue, operand);
        }
        protected override Byte Rint(Byte bValue, Byte operand)
        {
            return Convert.ToByte(Math.Round(Convert.ToDouble(bValue)));
        }
        protected override Byte Conjugate(Byte bValue, Byte operand)
        {
            return bValue;
        }
        protected override Byte IsNAN(Byte bValue, Byte operand)
        {
            return 0;
        }
        protected override Byte FMax(Byte bValue, Byte operand)
        {
            return Math.Max(bValue, operand);
        }
        protected override Byte FMin(Byte bValue, Byte operand)
        {
            return Math.Min(bValue, operand);
        }
        protected override Byte Heaviside(Byte bValue, Byte operand)
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
