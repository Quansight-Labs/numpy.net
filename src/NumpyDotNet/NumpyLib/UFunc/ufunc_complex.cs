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
    #region UFUNC COMPLEX
    internal class UFUNC_Complex : UFUNC_BASE<System.Numerics.Complex>, IUFUNC_Operations
    {
        public UFUNC_Complex() : base(sizeof(double) * 2)
        {

        }

        protected override System.Numerics.Complex ConvertToTemplate(object value)
        {

            if (value is System.Numerics.Complex)
            {
                System.Numerics.Complex c = (System.Numerics.Complex)value;
                return c;
            }
            else
            {
                return new System.Numerics.Complex(Convert.ToDouble(value), 0);
            }

        }

        protected override System.Numerics.Complex PerformUFuncOperation(UFuncOperation op, System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            System.Numerics.Complex destValue = 0;
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

        #region System.Numerics.Complex specific operation handlers
        protected override System.Numerics.Complex Add(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            return aValue + bValue;
        }

        protected override System.Numerics.Complex Subtract(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            return aValue - bValue;
        }
        protected override System.Numerics.Complex Multiply(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            return aValue * bValue;
        }

        protected override System.Numerics.Complex Divide(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        protected override System.Numerics.Complex Remainder(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            if (bValue == 0)
            {
                aValue = 0;
                return aValue;
            }
            var rem = aValue.Real % bValue.Real;
            if ((aValue.Real > 0) == (bValue.Real > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + bValue;
            }
        }
        private System.Numerics.Complex FMod(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            if (bValue == 0)
            {
                aValue = 0;
                return aValue;
            }
            return aValue.Real % bValue.Real;
        }
        protected override System.Numerics.Complex Power(System.Numerics.Complex aValue, System.Numerics.Complex bValue)
        {
            return System.Numerics.Complex.Pow(aValue, bValue);
        }
        private System.Numerics.Complex Square(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return bValue * bValue;
        }
        private System.Numerics.Complex Reciprocal(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            if (bValue == 0)
                return 0;

            return 1 / bValue;
        }
        private System.Numerics.Complex OnesLike(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return 1;
        }
        private System.Numerics.Complex Sqrt(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return System.Numerics.Complex.Sqrt(bValue);
        }
        private System.Numerics.Complex Negative(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return -bValue;
        }
        private System.Numerics.Complex Absolute(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return System.Numerics.Complex.Abs(bValue);
        }
        private System.Numerics.Complex Invert(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return bValue;
        }
        private System.Numerics.Complex LeftShift(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            UInt64 rValue = (UInt64)bValue.Real;
            rValue = rValue << Convert.ToInt32(operand.Real);

            UInt64 iValue = (UInt64)bValue.Imaginary;
            iValue = iValue << Convert.ToInt32(operand.Imaginary);

            return new System.Numerics.Complex((double)rValue, (double)iValue);
        }
        private System.Numerics.Complex RightShift(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            UInt64 rValue = (UInt64)bValue.Real;
            rValue = rValue >> Convert.ToInt32(operand.Real);

            UInt64 iValue = (UInt64)bValue.Imaginary;
            iValue = iValue >> Convert.ToInt32(operand.Imaginary);

            return new System.Numerics.Complex((double)rValue, (double)iValue);
        }
        private System.Numerics.Complex BitWiseAnd(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            UInt64 rValue = (UInt64)bValue.Real;
            rValue = rValue & Convert.ToUInt64(operand.Real);

            UInt64 iValue = (UInt64)bValue.Imaginary;
            iValue = iValue & Convert.ToUInt64(operand.Imaginary);

            return new System.Numerics.Complex((double)rValue, (double)iValue);
        }
        private System.Numerics.Complex BitWiseXor(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            UInt64 rValue = (UInt64)bValue.Real;
            rValue = rValue ^ Convert.ToUInt64(operand.Real);

            UInt64 iValue = (UInt64)bValue.Imaginary;
            iValue = iValue ^ Convert.ToUInt64(operand.Imaginary);

            return new System.Numerics.Complex((double)rValue, (double)iValue);
        }
        private System.Numerics.Complex BitWiseOr(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            UInt64 rValue = (UInt64)bValue.Real;
            rValue = rValue | Convert.ToUInt64(operand.Real);

            UInt64 iValue = (UInt64)bValue.Imaginary;
            iValue = iValue | Convert.ToUInt64(operand.Imaginary);

            return new System.Numerics.Complex((double)rValue, (double)iValue);
        }
        private bool Less(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            if (operand.Imaginary == 0)
            {
                return bValue.Real < operand.Real;
            }
            return false;
        }
        private bool LessEqual(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            if (operand.Imaginary == 0)
            {
                return bValue.Real <= operand.Real;
            }
            return false;
        }
        private bool Equal(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return bValue == operand;
        }
        private bool NotEqual(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return bValue != operand;
        }
        private bool Greater(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            if (operand.Imaginary == 0)
            {
                return bValue.Real > operand.Real;
            }
            return false;
        }
        private bool GreaterEqual(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            if (operand.Imaginary == 0)
            {
                return bValue.Real >= operand.Real;
            }
            return false;
        }
        private System.Numerics.Complex FloorDivide(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }

            double Real = 0;
            if (operand.Real != 0)
                Real = Math.Floor(bValue.Real / operand.Real);

            double Imaginary = 0;
            if (operand.Imaginary != 0)
                Imaginary = Math.Floor(bValue.Imaginary / operand.Imaginary);

            return new System.Numerics.Complex(Real, Imaginary);
        }
        private System.Numerics.Complex TrueDivide(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            if (operand == 0)
                return 0;

            return bValue / operand;
        }
        private bool LogicalOr(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return bValue != 0 || operand != 0;
        }
        private bool LogicalAnd(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return bValue != 0 && operand != 0;
        }
        private System.Numerics.Complex Floor(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return new System.Numerics.Complex(Math.Floor(bValue.Real), Math.Floor(bValue.Imaginary));
        }
        private System.Numerics.Complex Ceiling(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return new System.Numerics.Complex(Math.Ceiling(bValue.Real), Math.Ceiling(bValue.Imaginary));
        }
        private System.Numerics.Complex Maximum(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return Math.Max(bValue.Real, operand.Real);
        }
        private System.Numerics.Complex Minimum(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return Math.Min(bValue.Real, operand.Real);
        }
        private System.Numerics.Complex Rint(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return new System.Numerics.Complex(Math.Round(bValue.Real), Math.Round(bValue.Imaginary));
        }
        private System.Numerics.Complex Conjugate(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            var cc = new System.Numerics.Complex(bValue.Real, -bValue.Imaginary);
            return cc;
        }
        private bool IsNAN(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return false;
        }
        private System.Numerics.Complex FMax(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return Math.Max(bValue.Real, operand.Real);
        }
        private System.Numerics.Complex FMin(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            return Math.Min(bValue.Real, operand.Real);
        }
        private System.Numerics.Complex Heaviside(System.Numerics.Complex bValue, System.Numerics.Complex operand)
        {
            if (bValue == 0.0)
                return operand;

            if (bValue.Real < 0.0)
                return 0.0;

            return 1.0;

        }

        #endregion

    }


    #endregion
}
