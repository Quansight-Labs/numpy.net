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
    internal class UFUNC_String : UFUNC_BASE<System.String>, IUFUNC_Operations
    {
        public UFUNC_String() : base(IntPtr.Size)
        {

        }

        protected override System.String ConvertToTemplate(object value)
        {
            return value.ToString();
        }

        protected override System.String PerformUFuncOperation(UFuncOperation op, System.String aValue, System.String bValue)
        {
            System.String destValue = null;

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
                    destValue = null;
                    break;

            }

            return destValue;
        }

        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            return null;
        }

        #region System.String specific operation handlers
        protected override System.String Add(string aValue, string bValue)
        {
            if (bValue == null)
                return aValue;

            return aValue + bValue;
        }
 
        protected override System.String Subtract(string aValue, string bValue)
        {
            string sValue = (string)aValue;
            if (sValue == null)
                return sValue;

            return sValue.Replace(bValue.ToString(), "");
        }
   
        protected override System.String Multiply(string aValue, string bValue)
        {
            return aValue;
        }
  
        protected override System.String Divide(string aValue, string bValue)
        {
            return aValue;
        }
        protected override System.String Remainder(string aValue, string bValue)
        {
            return aValue;
        }
        protected override System.String FMod(string aValue, string bValue)
        {
            return aValue;
        }
        protected override System.String Power(string aValue, string bValue)
        {
            return aValue;
        }
        protected override System.String Square(string bValue, string operand)
        {
            return bValue;
        }
        protected override System.String Reciprocal(string bValue, string operand)
        {
            return bValue;
        }
        protected override System.String OnesLike(string bValue, string operand)
        {
            return "1";
        }
        protected override System.String Sqrt(string bValue, string operand)
        {
            return bValue;
        }
        protected override System.String Negative(string bValue, string operand)
        {
            if (bValue == null)
                return bValue;

            char[] arr = bValue.ToCharArray();
            Array.Reverse(arr);
            return new string(arr);
        }
        protected override System.String Absolute(string bValue, string operand)
        {
            return bValue;
        }
        protected override System.String Invert(string bValue, string operand)
        {
            string sValue = bValue;

            if (sValue == null)
                return sValue;

            string lowercase = sValue.ToLower();
            if (lowercase != sValue)
                return lowercase;
            else
                return sValue.ToUpper();
        }
        protected override System.String LeftShift(string bValue, string operand)
        {
            if (bValue == null || operand == null)
                return bValue;

            string sValue = bValue;
            int shiftCount = Convert.ToInt32(operand);

            if (string.IsNullOrEmpty(sValue))
                return sValue;

            for (int i = 0; i < shiftCount; i++)
            {
                string first = sValue.Substring(0, 1);
                sValue = sValue.Substring(1) + first;
            }
            return sValue;
        }
        protected override System.String RightShift(string bValue, string operand)
        {
            if (bValue == null || operand == null)
                return bValue;

            string sValue = bValue;
            int shiftCount = Convert.ToInt32(operand);

            if (string.IsNullOrEmpty(sValue))
                return sValue;

            for (int i = 0; i < shiftCount; i++)
            {
                string last = sValue.Substring(sValue.Length - 1, 1);
                sValue = last + sValue.Substring(0, sValue.Length - 1);
            }
            return sValue;
        }
        protected override System.String BitWiseAnd(string bValue, string operand)
        {
            return bValue;
        }
        protected override System.String BitWiseXor(string bValue, string operand)
        {
            return bValue;
        }
        protected override System.String BitWiseOr(string bValue, string operand)
        {
            return bValue;
        }

        private int CompareTo(string invalue, string comparevalue)
        {
            if (invalue == null)
            {
                if (comparevalue == null)
                {
                    return 0;
                }
                return 1;
            }

            return string.Compare(invalue.ToString(), comparevalue.ToString());
        }

        protected override System.String Less(string bValue, string operand)
        {
            bool result = CompareTo(bValue, operand) < 0;
            return result.ToString();
        }
        protected override System.String LessEqual(string bValue, string operand)
        {
            bool result = CompareTo(bValue, operand) <= 0;
            return result.ToString();
        }
        protected override System.String Equal(string bValue, string operand)
        {
            bool result = CompareTo(bValue, operand) == 0;
            return result.ToString();
        }
        protected override System.String NotEqual(string bValue, string operand)
        {
            bool result = CompareTo(bValue, operand) != 0;
            return result.ToString();
        }
        protected override System.String Greater(string bValue, string operand)
        {
            bool result = CompareTo(bValue, operand) > 0;
            return result.ToString();
        }
        protected override System.String GreaterEqual(string bValue, string operand)
        {
            bool result = CompareTo(bValue, operand) >= 0;
            return result.ToString();
        }
        protected override System.String FloorDivide(string bValue, string operand)
        {
            return bValue;
        }
        protected override System.String TrueDivide(string bValue, string operand)
        {
            return bValue;
        }
        protected override System.String LogicalOr(string bValue, string operand)
        {
            if (bValue == null || operand == null)
                return bValue;

            string sValue = (string)bValue;
            bool result = !sValue.Contains(operand.ToString());
            return result.ToString();
        }
        protected override System.String LogicalAnd(string bValue, string operand)
        {
            if (bValue == null || operand == null)
                return null;

            string sValue = (string)bValue;
            bool result = sValue.Contains(operand.ToString());
            return result.ToString();
        }
        protected override System.String Floor(string bValue, string operand)
        {
            return bValue;
        }
        protected override System.String Ceiling(string bValue, string operand)
        {
            return bValue;
        }
        protected override System.String Maximum(string bValue, string operand)
        {
            if (CompareTo(bValue, operand) >= 0)
                return bValue.ToString();
            return operand.ToString();
        }
        protected override System.String Minimum(string bValue, string operand)
        {
            if (CompareTo(bValue, operand) <= 0)
                return bValue.ToString();
            return operand.ToString();
        }
        protected override System.String Rint(string bValue, string operand)
        {
            return bValue;
        }
        protected override System.String Conjugate(string bValue, string operand)
        {
            return bValue;
        }
        protected override System.String IsNAN(string bValue, string operand)
        {
            return bValue;
        }
        protected override System.String FMax(string bValue, string operand)
        {
            if (CompareTo(bValue, operand) >= 0)
                return bValue.ToString();
            return operand.ToString();
        }
        protected override System.String FMin(string bValue, string operand)
        {
            if (CompareTo(bValue, operand) <= 0)
                return bValue.ToString();
            return operand.ToString();
        }
        protected override System.String Heaviside(string bValue, string operand)
        {
            return bValue;
        }

        #endregion

    }


    #endregion
}
