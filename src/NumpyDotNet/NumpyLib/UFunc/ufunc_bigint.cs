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
    #region UFUNC BIGINT
    internal class UFUNC_BigInt : UFUNC_BASE<System.Numerics.BigInteger>, IUFUNC_Operations
    {
        public UFUNC_BigInt() : base(sizeof(double) * 4)
        {

        }

        protected override System.Numerics.BigInteger ConvertToTemplate(object value)
        {

            if (value is System.Numerics.BigInteger)
            {
                System.Numerics.BigInteger c = (System.Numerics.BigInteger)value;
                return c;
            }
            else
            {
                return new System.Numerics.BigInteger(Convert.ToDouble(value));
            }

        }

        protected override System.Numerics.BigInteger PerformUFuncOperation(UFuncOperation op, System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            System.Numerics.BigInteger destValue = 0;
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

        #region System.Numerics.BigInteger specific operation handlers
        protected override System.Numerics.BigInteger Add(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            return aValue + bValue;
        }

        protected override System.Numerics.BigInteger Subtract(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            return aValue - bValue;
        }
        protected override System.Numerics.BigInteger Multiply(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            return aValue * bValue;
        }

        protected override System.Numerics.BigInteger Divide(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        protected override System.Numerics.BigInteger Remainder(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            if (bValue == 0)
            {
                aValue = 0;
                return aValue;
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
        protected override System.Numerics.BigInteger FMod(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            if (bValue == 0)
            {
                aValue = 0;
                return aValue;
            }
            return aValue % bValue;
        }
        protected override System.Numerics.BigInteger Power(System.Numerics.BigInteger aValue, System.Numerics.BigInteger bValue)
        {
            return System.Numerics.BigInteger.Pow(aValue, (int)bValue);
        }
        protected override System.Numerics.BigInteger Square(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue * bValue;
        }
        protected override System.Numerics.BigInteger Reciprocal(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            if (bValue == 0)
                return 0;

            return 1 / bValue;
        }
        protected override System.Numerics.BigInteger OnesLike(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return 1;
        }
        protected override System.Numerics.BigInteger Sqrt(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            var dd = Math.Round(Math.Pow(Math.E, System.Numerics.BigInteger.Log((System.Numerics.BigInteger)bValue) / 2));
            return new System.Numerics.BigInteger(dd);
        }
        protected override System.Numerics.BigInteger Negative(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return -bValue;
        }
        protected override System.Numerics.BigInteger Absolute(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return System.Numerics.BigInteger.Abs(bValue);
        }
        protected override System.Numerics.BigInteger Invert(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue;
        }
        protected override System.Numerics.BigInteger LeftShift(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            UInt64 rValue = (UInt64)bValue;
            rValue = rValue << Convert.ToInt32((Int64)operand);
            return new System.Numerics.BigInteger(rValue);
        }
        protected override System.Numerics.BigInteger RightShift(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            UInt64 rValue = (UInt64)bValue;
            rValue = rValue >> Convert.ToInt32((Int64)operand);
            return new System.Numerics.BigInteger(rValue);

        }
        protected override System.Numerics.BigInteger BitWiseAnd(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            UInt64 rValue = (UInt64)bValue;
            rValue = rValue & Convert.ToUInt64((UInt64)operand);

            return new System.Numerics.BigInteger(rValue);
        }
        protected override System.Numerics.BigInteger BitWiseXor(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            UInt64 rValue = (UInt64)bValue;
            rValue = rValue ^ Convert.ToUInt64((UInt64)operand);

            return new System.Numerics.BigInteger(rValue);
        }
        protected override System.Numerics.BigInteger BitWiseOr(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            UInt64 rValue = (UInt64)bValue;
            rValue = rValue | Convert.ToUInt64((UInt64)operand);

            return new System.Numerics.BigInteger(rValue);
        }
        private bool Less(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue < operand;
        }
        private bool LessEqual(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue <= operand;
        }
        private bool Equal(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue == operand;
        }
        private bool NotEqual(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue != operand;
        }
        private bool Greater(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue > operand;
        }
        private bool GreaterEqual(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue >= operand;
        }
        private System.Numerics.BigInteger FloorDivide(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            if (operand == 0)
            {
                bValue = 0;
                return bValue;
            }
            return bValue / operand;
        }
        private System.Numerics.BigInteger TrueDivide(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            if (operand == 0)
                return 0;

            return bValue / operand;
        }
        private bool LogicalOr(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue != 0 || operand != 0;
        }
        private bool LogicalAnd(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue != 0 && operand != 0;
        }
        private System.Numerics.BigInteger Floor(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue;
        }
        private System.Numerics.BigInteger Ceiling(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue;
        }
        private System.Numerics.BigInteger Maximum(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return System.Numerics.BigInteger.Max(bValue, operand);
        }
        private System.Numerics.BigInteger Minimum(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return System.Numerics.BigInteger.Min(bValue, operand);
        }
        private System.Numerics.BigInteger Rint(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue;
        }
        private System.Numerics.BigInteger Conjugate(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return bValue;
        }
        private bool IsNAN(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return false;
        }
        private System.Numerics.BigInteger FMax(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return System.Numerics.BigInteger.Max(bValue, operand);
        }
        private System.Numerics.BigInteger FMin(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
        {
            return System.Numerics.BigInteger.Min(bValue, operand);
        }
        private System.Numerics.BigInteger Heaviside(System.Numerics.BigInteger bValue, System.Numerics.BigInteger operand)
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
