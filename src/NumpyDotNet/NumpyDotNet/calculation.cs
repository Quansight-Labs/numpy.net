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
using System.Linq;
using System.Text;
using NumpyLib;

#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{
    public partial class ndarray
    {

    

        private static readonly double[] p10 = new double[] { 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9 };

        private static double PowerOfTen(int n)
        {
            double ret;
            if (n < p10.Length)
            {
                ret = p10[n];
            }
            else
            {
                int start = p10.Length - 1;
                ret = p10[start];
                while (n-- > start)
                {
                    ret *= 10;
                }
            }
            return ret;
        }

        internal object Round(int decimals, ndarray ret = null)
        {
            // For complex just round both parts.
            if (IsComplex)
            {
                if (ret == null)
                {
                    ret = Copy();
                }
                Real.Round(decimals, ret.Real);
                Imag.Round(decimals, ret.Imag);
                return ret;
            }

            if (decimals >= 0 && IsInteger)
            {
                // There is nothing to do for integers.
                if (ret != null)
                {
                    NpyCoreApi.CopyAnyInto(ret, this);
                    return ret;
                }
                else
                {
                    return this;
                }
            }


            if (decimals == 0)
            {
                // This is just a ufunc
                return UnaryOpInPlace(this, NpyArray_Ops.npy_op_rint, ret);
            }

            // Set up to do a multiply, round, divide, or the other way around.
            ufunc pre;
            ufunc post;
            if (decimals >= 0)
            {
                pre = NpyCoreApi.GetNumericOp(NpyArray_Ops.npy_op_multiply);
                post = NpyCoreApi.GetNumericOp(NpyArray_Ops.npy_op_divide);
            }
            else
            {
                pre = NpyCoreApi.GetNumericOp(NpyArray_Ops.npy_op_divide);
                post = NpyCoreApi.GetNumericOp(NpyArray_Ops.npy_op_multiply);
                decimals = -decimals;
            }
            var factor = PowerOfTen(decimals);

            // Make a temporary array, if we need it.
            NPY_TYPES tmpType = NPY_TYPES.NPY_DOUBLE;
            if (!IsInteger)
            {
                tmpType = Dtype.TypeNum;
            }
            ndarray tmp;
            if (ret != null && ret.Dtype.TypeNum == tmpType)
            {
                tmp = ret;
            }
            else
            {
                tmp = NpyCoreApi.NewFromDescr(NpyCoreApi.DescrFromType(tmpType), Dims, null, 0, null);
            }

            // Do the work
            BinaryOpInPlace(this, factor, pre, tmp);
            UnaryOpInPlace(tmp, NpyArray_Ops.npy_op_rint, tmp);
            BinaryOpInPlace(tmp, factor, post, tmp);

            if (ret != null && tmp != ret)
            {
                NpyCoreApi.CopyAnyInto(ret, tmp);
                return ret;
            }
            return tmp;
        }

        internal ndarray Clip(object min, object max, ndarray ret = null)
        {
            if (ret != null)
            {
                if (min == null && max == null)
                {
                    throw new ArgumentException("must set either max or min");
                }
                if (min == null)
                {
                    return BinaryOpInPlace(this, max, NpyArray_Ops.npy_op_minimum, ret) as ndarray;
                }
                else if (max == null)
                {
                    return BinaryOpInPlace(this, min, NpyArray_Ops.npy_op_maximum, ret) as ndarray;
                }
                else
                {
                    ndarray tmp = BinaryOpInPlace(this, max, NpyArray_Ops.npy_op_minimum, ret) as ndarray;
                    return BinaryOpInPlace(tmp, min, NpyArray_Ops.npy_op_maximum, ret) as ndarray;
                }
            }
            else
            {
                if (min == null && max == null)
                {
                    throw new ArgumentException("must set either max or min");
                }
                if (min == null)
                {
                    return BinaryOp(this, max, NpyArray_Ops.npy_op_minimum) as ndarray;
                }
                else if (max == null)
                {
                    return BinaryOp(this, min, NpyArray_Ops.npy_op_maximum) as ndarray;
                }
                else
                {
                    ndarray tmp = BinaryOp(this, max, NpyArray_Ops.npy_op_minimum) as ndarray;
                    return BinaryOp(tmp, min, NpyArray_Ops.npy_op_maximum) as ndarray;
                }
            }

        }

        internal ndarray Conjugate(ndarray ret = null)
        {
            return NpyCoreApi.Conjugate(this, ret);
        }
    }
}
