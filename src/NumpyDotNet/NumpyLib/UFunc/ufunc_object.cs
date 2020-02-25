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
    #region UFUNC OBJECT
    internal class UFUNC_Object : UFUNC_BASE<System.Object>, IUFUNC_Operations
    {
        public UFUNC_Object() : base(IntPtr.Size)
        {

        }

        protected override System.Object ConvertToTemplate(object value)
        {
            return value;
        }

        protected override System.Object PerformUFuncOperation(UFuncOperation op, System.Object aValue, System.Object bValue)
        {
            System.Object destValue = 0;
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
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.less_equal:
                    boolValue = LessEqual(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.equal:
                    boolValue = Equal(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.not_equal:
                    boolValue = NotEqual(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.greater:
                    boolValue = Greater(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
                    break;
                case UFuncOperation.greater_equal:
                    boolValue = GreaterEqual(aValue, bValue);
                    destValue = boolValue ? 1 : 0;
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

        #region System.Object specific operation handlers
        protected override System.Object Add(dynamic aValue, dynamic bValue)
        {
            return aValue + bValue;
        }

        protected override System.Object Subtract(dynamic aValue, dynamic bValue)
        {
            return aValue - bValue;
        }
        protected override System.Object Multiply(dynamic aValue, dynamic bValue)
        {
            return aValue * bValue;
        }

        protected override System.Object Divide(dynamic aValue, dynamic bValue)
        {
            return aValue / bValue;
        }
        private System.Object Remainder(dynamic aValue, dynamic bValue)
        {
            if (bValue == 0)
            {
                return 0;
            }
            return aValue % bValue;

        }
        private System.Object FMod(dynamic aValue, dynamic bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        protected override System.Object Power(dynamic aValue, dynamic bValue)
        {
            return Math.Pow(aValue, bValue);
        }
        private System.Object Square(dynamic bValue, dynamic operand)
        {
            return bValue * bValue;
        }
        private System.Object Reciprocal(dynamic bValue, dynamic operand)
        {
            if (bValue == 0)
                return 0;

            return 1 / bValue;
        }
        private System.Object OnesLike(dynamic bValue, dynamic operand)
        {
            return 1;
        }
        private System.Object Sqrt(dynamic bValue, dynamic operand)
        {
            return Math.Sqrt(bValue);
        }
        private System.Object Negative(dynamic bValue, dynamic operand)
        {
            return -bValue;
        }
        private System.Object Absolute(dynamic bValue, dynamic operand)
        {
            return Math.Abs(bValue);
        }
        private System.Object Invert(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return ~dValue;
        }
        private System.Object LeftShift(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue << Convert.ToInt32(operand);
        }
        private System.Object RightShift(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        private System.Object BitWiseAnd(dynamic bValue, dynamic operand)
        {
            return bValue & operand;
        }
        private System.Object BitWiseXor(dynamic bValue, dynamic operand)
        {
            return bValue ^ operand;

        }
        private System.Object BitWiseOr(dynamic bValue, dynamic operand)
        {
            return bValue | operand;

        }
        private bool Less(dynamic bValue, dynamic operand)
        {
            return bValue < operand;
        }
        private bool LessEqual(dynamic bValue, dynamic operand)
        {
            return bValue <= operand;
        }
        private bool Equal(dynamic bValue, dynamic operand)
        {
            return bValue == operand;
        }
        private bool NotEqual(dynamic bValue, dynamic operand)
        {
            return bValue != operand;
        }
        private bool Greater(dynamic bValue, dynamic operand)
        {
            return bValue > operand;
        }
        private bool GreaterEqual(dynamic bValue, dynamic operand)
        {
            return bValue >= (dynamic)operand;
        }
        private System.Object FloorDivide(dynamic bValue, dynamic operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return Math.Floor(bValue / operand);
        }
        private System.Object TrueDivide(dynamic bValue, dynamic operand)
        {
            if (operand == 0)
                return 0;

            return bValue / operand;
        }
        private bool LogicalOr(dynamic bValue, dynamic operand)
        {
            return bValue != 0 || operand != 0;
        }
        private bool LogicalAnd(dynamic bValue, dynamic operand)
        {
            return bValue != 0 && operand != 0;
        }
        private System.Object Floor(dynamic bValue, dynamic operand)
        {
            if (bValue is decimal)
            {
                return Math.Floor(Convert.ToDecimal(bValue));
            }
            return Math.Floor(Convert.ToDouble(bValue));
        }
        private System.Object Ceiling(dynamic bValue, dynamic operand)
        {
            if (bValue is decimal)
            {
                return Math.Ceiling(Convert.ToDecimal(bValue));
            }
            return Math.Ceiling(Convert.ToDouble(bValue));
        }
        private System.Object Maximum(dynamic bValue, dynamic operand)
        {
            if (bValue >= operand)
                return bValue;
            return operand;
        }
        private System.Object Minimum(dynamic bValue, dynamic operand)
        {
            if (bValue <= operand)
                return bValue;
            return operand;
        }
        private System.Object Rint(dynamic bValue, dynamic operand)
        {
            if (bValue is decimal)
            {
                return Math.Round(Convert.ToDecimal(bValue));
            }
            return Math.Round(Convert.ToDouble(bValue));
        }
        private System.Object Conjugate(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
        private bool IsNAN(dynamic bValue, dynamic operand)
        {
            return false;
        }
        private System.Object FMax(dynamic bValue, dynamic operand)
        {
            if (bValue >= operand)
                return bValue;
            return operand;
        }
        private System.Object FMin(dynamic bValue, dynamic operand)
        {
            if (bValue <= operand)
                return bValue;
            return operand;
        }
        private System.Object Heaviside(dynamic bValue, dynamic operand)
        {
            double x = Convert.ToDouble(bValue);

            if (double.IsNaN(x))
                return double.NaN;

            if (x == 0.0)
                return Convert.ToDouble(operand);

            if (x < 0.0)
                return 0.0;

            return 1.0;

        }

        #endregion

    }


    #endregion
}
