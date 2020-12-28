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

    #region UFUNC SBYTE

    internal class UFUNC_SByte : UFUNC_BASE<sbyte>, IUFUNC_Operations
    {
        public UFUNC_SByte() : base(sizeof(sbyte))
        {

        }

        protected override sbyte ConvertToTemplate(object value)
        {
            return Convert.ToSByte(value);
        }

        protected override sbyte PerformUFuncOperation(UFuncOperation op, sbyte aValue, sbyte bValue)
        {
            sbyte destValue = 0;

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

        #region sbyte specific operation handlers
        protected override sbyte Add(sbyte aValue, sbyte bValue)
        {
            return (sbyte)(aValue + bValue);
        }
        protected override sbyte AddReduce(sbyte result, sbyte[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (sbyte)(result + OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }

        protected override sbyte Subtract(sbyte aValue, sbyte bValue)
        {
            return (sbyte)(aValue - bValue);
        }
        protected override sbyte SubtractReduce(sbyte result, sbyte[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (sbyte)(result - OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }

        protected override sbyte Multiply(sbyte aValue, sbyte bValue)
        {
            return (sbyte)(aValue * bValue);
        }
        protected override sbyte MultiplyReduce(sbyte result, sbyte[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (sbyte)(result * OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }

        protected override sbyte Divide(sbyte aValue, sbyte bValue)
        {
            if (bValue == 0)
                return 0;
            return (sbyte)(aValue / bValue);
        }
        protected override sbyte DivideReduce(sbyte result, sbyte[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = 0;
                else
                    result = (sbyte)(result / bValue);

                OperIndex += OperStep;
            }

            return result;
        }

        protected override sbyte Remainder(sbyte aValue, sbyte bValue)
        {
            if (bValue == 0)
            {
                return 0;
            }
            var rem = aValue % bValue;
            if ((aValue > 0) == (bValue > 0) || rem == 0)
            {
                return (sbyte)(rem);
            }
            else
            {
                return (sbyte)(rem + bValue);
            }
        }
        protected override sbyte FMod(sbyte aValue, sbyte bValue)
        {
            if (bValue == 0)
                return 0;
            return (sbyte)(aValue % bValue);
        }
        protected override sbyte Power(sbyte aValue, sbyte bValue)
        {
            return Convert.ToSByte(Math.Pow(aValue, bValue));
        }
        protected override sbyte Square(sbyte bValue, sbyte operand)
        {
            return (sbyte)(bValue * bValue);
        }
        protected override sbyte Reciprocal(sbyte bValue, sbyte operand)
        {
            if (bValue == 0)
                return 0;

            return (sbyte)(1 / bValue);
        }
        protected override sbyte OnesLike(sbyte bValue, sbyte operand)
        {
            return 1;
        }
        protected override sbyte Sqrt(sbyte bValue, sbyte operand)
        {
            return Convert.ToSByte(Math.Sqrt(bValue));
        }
        protected override sbyte Negative(sbyte bValue, sbyte operand)
        {
            return (sbyte)(-bValue);
        }
        protected override sbyte Absolute(sbyte bValue, sbyte operand)
        {
            return Convert.ToSByte(Math.Abs(bValue));
        }
        protected override sbyte Invert(sbyte bValue, sbyte operand)
        {
            return (sbyte)(~bValue);
        }
        protected override sbyte LeftShift(sbyte bValue, sbyte operand)
        {
            return (sbyte)(bValue << Convert.ToInt32(operand));
        }
        protected override sbyte RightShift(sbyte bValue, sbyte operand)
        {
            return (sbyte)(bValue >> Convert.ToInt32(operand));
        }
        protected override sbyte BitWiseAnd(sbyte bValue, sbyte operand)
        {
            return (sbyte)(bValue & operand);
        }
        protected override sbyte BitWiseXor(sbyte bValue, sbyte operand)
        {
            return (sbyte)(bValue ^ operand);
        }
        protected override sbyte BitWiseOr(sbyte bValue, sbyte operand)
        {
            return (sbyte)(bValue | operand);
        }
        protected override sbyte Less(sbyte bValue, sbyte operand)
        {
            bool boolValue = bValue < operand;
            return (sbyte)(boolValue ? 1 : 0);
        }
        protected override sbyte LessEqual(sbyte bValue, sbyte operand)
        {
            bool boolValue = bValue <= operand;
            return (sbyte)(boolValue ? 1 : 0);
        }
        protected override sbyte Equal(sbyte bValue, sbyte operand)
        {
            bool boolValue = bValue == operand;
            return (sbyte)(boolValue ? 1 : 0);
        }
        protected override sbyte NotEqual(sbyte bValue, sbyte operand)
        {
            bool boolValue = bValue != operand;
            return (sbyte)(boolValue ? 1 : 0);
        }
        protected override sbyte Greater(sbyte bValue, sbyte operand)
        {
            bool boolValue = bValue > operand;
            return (sbyte)(boolValue ? 1 : 0);
        }
        protected override sbyte GreaterEqual(sbyte bValue, sbyte operand)
        {
            bool boolValue = bValue >= operand;
            return (sbyte)(boolValue ? 1 : 0);
        }
        protected override sbyte FloorDivide(sbyte bValue, sbyte operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Convert.ToSByte(Math.Floor(Convert.ToDouble(bValue) / Convert.ToDouble(operand)));
        }
        protected override sbyte TrueDivide(sbyte bValue, sbyte operand)
        {
            if (operand == 0)
                return 0;

            return (sbyte)(bValue / operand);
        }
        protected override sbyte LogicalOr(sbyte bValue, sbyte operand)
        {
            bool boolValue = bValue != 0 || operand != 0;
            return (sbyte)(boolValue ? 1 : 0);
        }
        protected override sbyte LogicalOrReduce(sbyte result, sbyte[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = (sbyte)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override sbyte LogicalAnd(sbyte bValue, sbyte operand)
        {
            bool boolValue = bValue != 0 && operand != 0;
            return (sbyte)(boolValue ? 1 : 0);
        }
        protected override sbyte LogicalAndReduce(sbyte result, sbyte[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = (sbyte)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override sbyte Floor(sbyte bValue, sbyte operand)
        {
            return Convert.ToSByte(Math.Floor(Convert.ToDouble(bValue)));
        }
        protected override sbyte Ceiling(sbyte bValue, sbyte operand)
        {
            return Convert.ToSByte(Math.Ceiling(Convert.ToDouble(bValue)));
        }
        protected override sbyte Maximum(sbyte bValue, sbyte operand)
        {
            return Math.Max(bValue, operand);
        }
        protected override sbyte MaximumReduce(sbyte result, sbyte[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected override sbyte Minimum(sbyte bValue, sbyte operand)
        {
            return Math.Min(bValue, operand);
        }
        protected override sbyte Rint(sbyte bValue, sbyte operand)
        {
            return Convert.ToSByte(Math.Round(Convert.ToDouble(bValue)));
        }
        protected override sbyte Conjugate(sbyte bValue, sbyte operand)
        {
            return bValue;
        }
        protected override sbyte IsNAN(sbyte bValue, sbyte operand)
        {
            return 0;
        }
        protected override sbyte FMax(sbyte bValue, sbyte operand)
        {
            return Math.Max(bValue, operand);
        }
        protected override sbyte FMin(sbyte bValue, sbyte operand)
        {
            return Math.Min(bValue, operand);
        }
        protected override sbyte Heaviside(sbyte bValue, sbyte operand)
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
