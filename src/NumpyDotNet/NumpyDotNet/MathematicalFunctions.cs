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

using NumpyLib;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
using npy_ucs4 = System.Int64;
#else
using npy_intp = System.Int32;
using npy_ucs4 = System.Int32;
#endif


namespace NumpyDotNet
{
    public static partial class np
    {
        #region MathFunctionHelper

        class MathFunctionHelper<T>
        {
            private ndarray a;
            private ndarray b;

            public shape expectedShape = null;
            public long expectedLength = 0;

            private long[] offsets = null;
            private long[] offsets2 = null;

            private T[] x1 = null;
            private T[] x2 = null;

            public T[] results = null;

            private NPY_TYPES target_nptype;
            private dtype target_dtype;

            public void SetupTypes()
            {
                T[] n = new T[1];

                switch (Type.GetTypeCode(n[0].GetType()))
                {
                    case TypeCode.Boolean:
                        target_nptype = NPY_TYPES.NPY_BOOL;
                        target_dtype = np.Bool;
                        break;

                    case TypeCode.Byte:
                        target_nptype = NPY_TYPES.NPY_UBYTE;
                        target_dtype = np.UInt8;
                        break;

                    case TypeCode.SByte:
                        target_nptype = NPY_TYPES.NPY_BYTE;
                        target_dtype = np.Int8;
                        break;

                    case TypeCode.UInt16:
                        target_nptype = NPY_TYPES.NPY_UINT16;
                        target_dtype = np.UInt16;
                        break;

                    case TypeCode.UInt32:
                        target_nptype = NPY_TYPES.NPY_UINT32;
                        target_dtype = np.UInt32;
                        break;

                    case TypeCode.UInt64:
                        target_nptype = NPY_TYPES.NPY_UINT64;
                        target_dtype = np.UInt64;
                        break;

                    case TypeCode.Int16:
                        target_nptype = NPY_TYPES.NPY_INT16;
                        target_dtype = np.Int16;
                        break;

                    case TypeCode.Int32:
                        target_nptype = NPY_TYPES.NPY_INT32;
                        target_dtype = np.Int32;
                        break;

                    case TypeCode.Int64:
                        target_nptype = NPY_TYPES.NPY_INT64;
                        target_dtype = np.Int64;
                        break;

                    case TypeCode.Decimal:
                        target_nptype = NPY_TYPES.NPY_DECIMAL;
                        target_dtype = np.Decimal;
                        break;

                    case TypeCode.Double:
                        target_nptype = NPY_TYPES.NPY_DOUBLE;
                        target_dtype = np.Float64;
                        break;

                    case TypeCode.Single:
                        target_nptype = NPY_TYPES.NPY_FLOAT;
                        target_dtype = np.Float32;
                        break;

                    default:
                        throw new Exception("Data type not supported");
                }
            }


            public MathFunctionHelper(object x)
            {
                SetupTypes();

                a = asanyarray(x);
                if (a.Dtype.TypeNum != target_nptype)
                {
                    a = a.astype(dtype: target_dtype);
                }

                offsets = NpyCoreApi.GetViewOffsets(a);
                x1 = a.Array.data.datap as T[];
                results = new T[offsets.Length];
                expectedShape = a.shape;
                expectedLength = offsets.Length;
            }

            public MathFunctionHelper(object x1, object x2)
            {
                SetupTypes();

                a = asanyarray(x1);
                b = asanyarray(x2);

                if (!broadcastable(a,b))
                {
                    throw new Exception(string.Format("operands could not be broadcast together with shapes ({0}),({1})", a.shape.ToString(), b.shape.ToString()));
                }

                long asize = NpyCoreApi.ArraySize(a);
                long bsize = NpyCoreApi.ArraySize(b);

                if (asize < bsize)
                {
                    var bcastIter = NpyCoreApi.BroadcastToShape(a, b.shape.iDims, b.shape.iDims.Length);
                    offsets = NpyCoreApi.GetViewOffsets(bcastIter, bsize);
                    offsets2 = NpyCoreApi.GetViewOffsets(b);
                    expectedShape = b.shape;
                }
                else if (bsize < asize)
                {
                    var bcastIter = NpyCoreApi.BroadcastToShape(b, a.shape.iDims, a.shape.iDims.Length);
                    offsets2 = NpyCoreApi.GetViewOffsets(bcastIter, bsize);
                    offsets = NpyCoreApi.GetViewOffsets(a);
                    expectedShape = a.shape;
                }
                else
                {
                    offsets = NpyCoreApi.GetViewOffsets(a);
                    offsets2 = NpyCoreApi.GetViewOffsets(b);
                    expectedShape = a.shape;
                }

                if (a.Dtype.TypeNum != target_nptype)
                {
                    a = a.astype(dtype: target_dtype);
                }
                if (b.Dtype.TypeNum != target_nptype)
                {
                    b = b.astype(dtype: target_dtype);
                }

                this.x1 = a.Array.data.datap as T[];
                this.x2 = b.Array.data.datap as T[];

                results = new T[offsets.Length];
                expectedLength = offsets.Length;
            }


            private long GetOffsetX1(long index)
            {
                long j = index;
                if (index >= this.offsets.Length)
                {
                    j = index % this.offsets.Length;
                }

                return offsets[j];
            }

            private long GetOffsetX2(long index)
            {
                long j = index;
                if (index >= this.offsets2.Length)
                {
                    j = index % this.offsets2.Length;
                }

                return offsets2[j];
            }

            public T X1(int i)
            {
                return this.x1[GetOffsetX1(i)];
            }

            public T X2(int i)
            {
                return this.x2[GetOffsetX2(i)];
            }
        }
        #endregion

        #region Trigonometric Functions

        public static ndarray sin(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);
            
            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Sin(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray cos(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Cos(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray tan(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Tan(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray arcsin(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Asin(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray arccos(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Acos(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray arctan(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Atan(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray hypot(object x1, object x2, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x1, x2);

            var hypot = np.sqrt(np.power(x1, 2) + np.power(x2, 2));

            return hypot;
        }

        public static ndarray arctan2(object x1, object x2, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x1, x2);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Atan2(ch.X1(i), ch.X2(i));
            }

            
            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray rad2deg(object x, object where = null)
        {
            return degrees(x, where);
        }

        public static ndarray degrees(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = ch.X1(i) * (180 / Math.PI);
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray deg2rad(object x, object where = null)
        {
            return radians(x, where);
        }

        public static ndarray radians(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.PI * ch.X1(i) / 180;
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        #endregion

        #region Hyperbolic functions

        public static ndarray sinh(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Sinh(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray cosh(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Cosh(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray tanh(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Tanh(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray arcsinh(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {

                ch.results[i] = MathHelper.HArcsin(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray arccosh(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = MathHelper.HArccos(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }
 


        public static ndarray arctanh(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = MathHelper.HArctan(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }


        #endregion

        #region Rounding Functions

        public static ndarray rint(object x, object where = null)
        {
            var a = asanyarray(x);

            var ret = NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_rint, 0);
            ret = ret.reshape(new shape(a.dims));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray fix(object x)
        {
            var a = asanyarray(x);
            var y1 = np.floor(a);
            var y2 = np.ceil(a);

            y1["..."] = np.where(a >= 0, y1, y2);
            return y1;
        }

        public static ndarray ceil(object x, object where = null)
        {
            var a = asanyarray(x);

            var ret = NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_ceil, 0);
            ret = ret.reshape(new shape(a.dims));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray trunc(object x)
        {
            var a = asanyarray(x);
            var y1 = np.floor(a);
            var y2 = np.ceil(a);

            y1["..."] = np.where(a >= 0, y1, y2);
            return y1;
        }

        #endregion

        #region Exponents and logarithms

        public static ndarray exp(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Exp(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray expm1(object x, object where = null)
        {

            /* from numpy C code
            
            nc_expm1@c@(@ctype@ *x, @ctype@ *r)
            {
                @ftype@ a = npy_exp@c@(x->real);
                r->real = a*npy_cos@c@(x->imag) - 1.0@c@;
                r->imag = a*npy_sin@c@(x->imag);
                return;
            }
            */

            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Exp(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray exp2(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Pow(2, ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray log(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Log(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray log10(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Log10(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray log2(object x, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Log(ch.X1(i), 2);
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray logn(object x, int n, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Log(ch.X1(i), n);
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray log1p(object x, object where = null)
        {
            /* from numpy C code
            static void
            nc_log1p@c@(@ctype@ *x, @ctype@ *r)
            {
                @ftype@ l = npy_hypot@c@(x->real + 1, x->imag);
                r->imag = npy_atan2@c@(x->imag, x->real + 1);
                r->real = npy_log@c@(l);
                return;
            }
            */


            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Log(ch.X1(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray logaddexp(object x1, object x2, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x1, x2);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Log(Math.Exp(ch.X1(i)) + Math.Exp(ch.X2(i)));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray logaddexp2(object x1, object x2, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x1, x2);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Log(Math.Pow(2, ch.X1(i)) + Math.Pow(2, ch.X2(i)), 2);
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray logaddexpn(object x1, object x2, int n, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x1, x2);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Log(Math.Pow(n, ch.X1(i)) + Math.Pow(n, ch.X2(i)), 2);
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }


        #endregion

        #region Other special functions

        #endregion

        #region Floating point routines

        public static ndarray signbit(object x, object where = null)
        {
            var xa = asanyarray(x);
            if (xa.IsFloatingPoint)
            {
                if (xa.itemsize <= sizeof(float))
                {
                    MathFunctionHelper<float> ch = new MathFunctionHelper<float>(x);
                    ch.results = null;

                    bool[] bret = new bool[ch.expectedLength];

                    for (int i = 0; i < ch.expectedLength; i++)
                    {
                        float f = ch.X1(i);
                        if (float.IsNegativeInfinity(f))
                        {
                            bret[i] = true;
                        }
                        else
                        if (float.IsPositiveInfinity(f))
                        {
                            bret[i] = false;
                        }
                        if (float.IsNaN(f))
                        {
                            bret[i] = false;
                        }
                        else
                        {
                            bret[i] = Math.Sign(f) < 0 ? true : false;
                        }

                    }

                    var ret = np.array(bret).reshape(new shape(ch.expectedShape));
                    if (where != null)
                    {
                        ret[np.invert(where)] = np.NaN;
                    }

                    return ret;
                }
                else
                {
                    MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);
                    ch.results = null;

                    bool[] bret = new bool[ch.expectedLength];

                    for (int i = 0; i < ch.expectedLength; i++)
                    {
                        double f = ch.X1(i);
                        if (double.IsNegativeInfinity(f))
                        {
                            bret[i] = true;
                        }
                        else
                        if (double.IsPositiveInfinity(f))
                        {
                            bret[i] = false;
                        }
                        if (double.IsNaN(f))
                        {
                            bret[i] = false;
                        }
                        else
                        {
                            bret[i] = Math.Sign(f) < 0 ? true : false;
                        }

                    }

                    var ret = np.array(bret).reshape(new shape(ch.expectedShape));
                    if (where != null)
                    {
                        ret[np.invert(where)] = np.NaN;
                    }

                    return ret;
                }
            }
            else
            {
                MathFunctionHelper<long> ch = new MathFunctionHelper<long>(x);
                ch.results = null;

                bool[] bret = new bool[ch.expectedLength];

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    long f = ch.X1(i);
                    bret[i] = Math.Sign(f) < 0 ? true : false;
                }

                var ret = np.array(bret).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    ret[np.invert(where)] = np.NaN;
                }

                return ret;
            }

        }


        public static ndarray copysign(object x1, object x2, object where = null)
        {
            var xa = asanyarray(x1);
            if (xa.IsFloatingPoint)
            {
                if (xa.itemsize <= sizeof(float))
                {
                    MathFunctionHelper<float> ch = new MathFunctionHelper<float>(x1, x2);

                    for (int i = 0; i < ch.expectedLength; i++)
                    {
                        var a = ch.X1(i);
                        var b = ch.X2(i);

                        if (Math.Sign(a) != Math.Sign(b))
                        {
                            a = -a;
                        }

                        ch.results[i] = a;
                    }

                    var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                    if (where != null)
                    {
                        ret[np.invert(where)] = np.NaN;
                    }

                    return ret;
                }
                else
                {
                    MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x1, x2);

                    for (int i = 0; i < ch.expectedLength; i++)
                    {
                        var a = ch.X1(i);
                        var b = ch.X2(i);

                        if (Math.Sign(a) != Math.Sign(b))
                        {
                            a = -a;
                        }

                        ch.results[i] = a;
                    }

                    var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                    if (where != null)
                    {
                        ret[np.invert(where)] = np.NaN;
                    }

                    return ret;
                }
            }
            else
            {
                MathFunctionHelper<long> ch = new MathFunctionHelper<long>(x1, x2);

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    var a = ch.X1(i);
                    var b = ch.X2(i);

                    if (Math.Sign(a) != Math.Sign(b))
                    {
                        a = -a;
                    }

                    ch.results[i] = a;
                }

                var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    ret[np.invert(where)] = np.NaN;
                }

                return ret.astype(xa.Dtype);
            }


    
        }

        public static ndarray[] frexp(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (xa.Dtype.TypeNum == NPY_TYPES.NPY_FLOAT)
            {
                MathFunctionHelper<float> ch = new MathFunctionHelper<float>(x);

                int[] _exponents = new int[ch.expectedLength];

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    float f = ch.X1(i);

                    int exponent = 0;
                    ch.results[i] = MachineCognitus.math.frexp(f, ref exponent);
                    _exponents[i] = exponent;
                }

                var mantissas = np.array(ch.results).reshape(new shape(ch.expectedShape));
                var expononents = np.array(_exponents).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    mantissas[np.invert(where)] = np.NaN;
                    expononents[np.invert(where)] = 0;
                }

                return new ndarray[] { mantissas, expononents };
            }
            else
            {
                MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

                int[] _exponents = new int[ch.expectedLength];

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    double f = ch.X1(i);

                    int exponent = 0;
                    ch.results[i] = MachineCognitus.math.frexp(f, ref exponent);
                    _exponents[i] = exponent;
                }

                var mantissas = np.array(ch.results).reshape(new shape(ch.expectedShape));
                var expononents = np.array(_exponents).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    mantissas[np.invert(where)] = np.NaN;
                    expononents[np.invert(where)] = 0;
                }

                return new ndarray[] { mantissas, expononents };
            }


        }

        public static ndarray ldexp(object x1, object x2, object where = null)
        {
            var a1 = asanyarray(x1);
            var a2 = asanyarray(x2);

            if (a1.itemsize <= sizeof(float) && a2.itemsize <= sizeof(float))
            {
                MathFunctionHelper<float> ch = new MathFunctionHelper<float>(x1, x2);

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    ch.results[i] = MachineCognitus.math.ldexp(ch.X1(i), Convert.ToInt32(ch.X2(i)));
                }

                var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    ret[np.invert(where)] = np.NaN;
                }

                return ret;
            }
            else
            {
                MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x1, x2);

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    ch.results[i] = MachineCognitus.math.ldexp(ch.X1(i), Convert.ToInt32(ch.X2(i)));
                }

                var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    ret[np.invert(where)] = np.NaN;
                }

                return ret;
            }

 
        }

        public static ndarray nextafter(object x1, object x2, object where = null)
        {
            var a1 = asanyarray(x1);
            var a2 = asanyarray(x2);

            bool a1b = !a1.IsAScalar && a1.IsFloatingPoint && a1.itemsize <= sizeof(float);
            bool a2b = !a2.IsAScalar && a2.IsFloatingPoint && a2.itemsize <= sizeof(float);

            if (a1b && a2b)
            {
                MathFunctionHelper<float> ch = new MathFunctionHelper<float>(x1, x2);

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    ch.results[i] = MachineCognitus.math.nextafter(ch.X1(i), Convert.ToSingle(ch.X2(i)));
                }

                var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    ret[np.invert(where)] = np.NaN;
                }

                return ret;
            }
            else
            {
                MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x1, x2);

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    ch.results[i] = MachineCognitus.math.nextafter(ch.X1(i), Convert.ToInt32(ch.X2(i)));
                }

                var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    ret[np.invert(where)] = np.NaN;
                }

                return ret;
            }


        }


        #endregion

        #region Rational routines


        private static long _gcdl(long a, long b)
        {
            while (b != 0)
            {
                var tempb = b;
                b = a % b;
                a = tempb;
            }
            return a;
        }
        private static int _gcdi(int a, int b)
        {
            while (b != 0)
            {
                var tempb = b;
                b = a % b;
                a = tempb;
            }
            return a;
        }


        private static long _lcml(long a, long b)
        {
            return a / _gcdl(a, b) * b;
        }
        private static int _lcmi(int a, int b)
        {
            return a / _gcdi(a, b) * b;
        }

        public static ndarray lcm(object x1, object x2, object where = null)
        {
            ndarray ret;

            var xa = asanyarray(x1);
            if (xa.IsInteger)
            {
                if (xa.itemsize <= sizeof(Int32))
                {
                    MathFunctionHelper<Int32> ch = new MathFunctionHelper<Int32>(x1, x2);

                    for (int i = 0; i < ch.expectedLength; i++)
                    {
                        ch.results[i] = _lcmi(ch.X1(i), ch.X2(i));
                    }
                    ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                    if (where != null)
                    {
                        ret[np.invert(where)] = np.NaN;
                    }
                }
                else
                {
                    MathFunctionHelper<Int64> ch = new MathFunctionHelper<Int64>(x1, x2);

                    for (int i = 0; i < ch.expectedLength; i++)
                    {
                        ch.results[i] = _lcml(ch.X1(i), ch.X2(i));
                    }
                    ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                    if (where != null)
                    {
                        ret[np.invert(where)] = np.NaN;
                    }
                }
            }
            else
            {
                throw new ValueError("This function only operates on integer type values");
            }

            return ret;
        }

        public static ndarray gcd(object x1, object x2, object where = null)
        {
            ndarray ret;

            var xa = asanyarray(x1);
            if (xa.IsInteger)
            {
                if (xa.itemsize <= sizeof(Int32))
                {
                    MathFunctionHelper<Int32> ch = new MathFunctionHelper<Int32>(x1, x2);

                    for (int i = 0; i < ch.expectedLength; i++)
                    {
                        ch.results[i] = _gcdi(ch.X1(i), ch.X2(i));
                    }
                    ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                    if (where != null)
                    {
                        ret[np.invert(where)] = np.NaN;
                    }
                }
                else
                {
                    MathFunctionHelper<Int64> ch = new MathFunctionHelper<Int64>(x1, x2);

                    for (int i = 0; i < ch.expectedLength; i++)
                    {
                        ch.results[i] = _gcdl(ch.X1(i), ch.X2(i));
                    }
                    ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                    if (where != null)
                    {
                        ret[np.invert(where)] = np.NaN;
                    }
                }
            }
            else
            {
                throw new ValueError("This function only operates on integer type values");
            }

            return ret;
        }

        #endregion

        #region Arithmetic operations


        public static ndarray add(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(NpyArray_Ops.npy_op_add, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }

        public static ndarray reciprocal(object a)
        {
            return NpyCoreApi.PerformNumericOp(asanyarray(a), NpyArray_Ops.npy_op_reciprocal, 0);
        }

        public static ndarray positive(object o)
        {
            var a = asanyarray(o);
            return a.Copy();
        }

        public static ndarray negative(object a, ndarray @out = null)
        {
            var result = NpyCoreApi.PerformNumericOp(asanyarray(a), NpyArray_Ops.npy_op_negative, 0, UseSrcAsDest: false );
            if (@out != null)
            {
                np.copyto(@out, result);
            }
            return result;
        }

        public static ndarray multiply(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(NpyArray_Ops.npy_op_multiply, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }

        public static ndarray divide(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(NpyArray_Ops.npy_op_divide, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }

        public static ndarray power(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(NpyArray_Ops.npy_op_power, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }

        public static ndarray subtract(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(NpyArray_Ops.npy_op_subtract, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }

 
        public static ndarray true_divide(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(NpyArray_Ops.npy_op_true_divide, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
          
        public static ndarray floor_divide(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(NpyArray_Ops.npy_op_floor_divide, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
 
        public static ndarray float_power(object x1, object x2, object where = null)
        {
            MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x1, x2);

            for (int i = 0; i < ch.expectedLength; i++)
            {
                ch.results[i] = Math.Pow(ch.X1(i), ch.X2(i));
            }

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;

        }

        #region mod/remainder

        public static ndarray mod(object x1, object x2)
        {
            return remainder(x1, x2);
        }
        public static ndarray mod(object x1, int x2)
        {
            return remainder(x1, x2);
        }
        public static ndarray remainder(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(NpyArray_Ops.npy_op_remainder, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
 
        public static ndarray fmod(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(NpyArray_Ops.npy_op_fmod, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
 

        #endregion


        #region divmod

        public static ndarray[] divmod(object x1, object x2, ndarray @out1 = null, ndarray @out2 = null, object where = null)
        {
            ndarray[] results = new ndarray[2];

            results[0] = NpyCoreApi.PerformUFUNC(NpyArray_Ops.npy_op_floor_divide, asanyarray(x1), asanyarray(x2), @out1, asanyarray(where));
            results[1] = NpyCoreApi.PerformUFUNC(NpyArray_Ops.npy_op_remainder, asanyarray(x1), asanyarray(x2), @out2, asanyarray(where));

            return results;
        }


        #endregion

        #region modf

        public static ndarray[] modf(object x1, ndarray @out1 = null, ndarray @out2 = null, object where = null)
        {
            ndarray[] results = new ndarray[2];

            dtype ret_type = np.Float64;
            var x1a = asanyarray(x1);
            if (x1a.Array.ItemType == NPY_TYPES.NPY_DECIMAL)
            {
                ret_type = np.Decimal;
            }

            results[1] = NpyCoreApi.PerformUFUNC(NpyArray_Ops.npy_op_floor_divide, asanyarray(x1).astype(ret_type), asanyarray(1), @out1, asanyarray(where));
            results[0] = NpyCoreApi.PerformUFUNC(NpyArray_Ops.npy_op_remainder, asanyarray(x1).astype(ret_type), asanyarray(1), @out2, asanyarray(where));

            return results;
        }


        #endregion

        #endregion





        #region Misc

        public static ndarray sign(object x, object where = null)
        {
            var xa = asanyarray(x);
            if (xa.IsFloatingPoint)
            {
                if (xa.itemsize <= sizeof(float))
                {
                    MathFunctionHelper<float> ch = new MathFunctionHelper<float>(x);

                    for (int i = 0; i < ch.expectedLength; i++)
                    {
                        float r = 0.0f;

                        float f = ch.X1(i);
                        if (float.IsNegativeInfinity(f))
                        {
                            r = -1.0f;
                        }
                        else
                        if (float.IsPositiveInfinity(f))
                        {
                            r = 1.0f;
                        }
                        if (float.IsNaN(f))
                        {
                            r = float.NaN;
                        }
                        else
                        {
                            if (f == 0.0f)
                            {
                                r = 0.0f;
                            }
                            else
                            {
                                r = Math.Sign(f) < 0 ? -1.0f : 1.0f;
                            }
                        }
                        ch.results[i] = r;
                    }
 

                    var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                    if (where != null)
                    {
                        ret[np.invert(where)] = np.NaN;
                    }

                    return ret;
                }
                else
                {
                    MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x);

                    for (int i = 0; i < ch.expectedLength; i++)
                    {
                        double r = 0.0;

                        double d = ch.X1(i);
                        if (double.IsNegativeInfinity(d))
                        {
                            r = -1.0;
                        }
                        else
                        if (double.IsPositiveInfinity(d))
                        {
                            r = 1.0;
                        }
                        if (double.IsNaN(d))
                        {
                            r = double.NaN;
                        }
                        else
                        {
                            if (d == 0)
                            {
                                r = 0;
                            }
                            else
                            {
                                r = Math.Sign(d) < 0 ? -1.0 : 1.0;
                            }
                        }
                        ch.results[i] = r;
                    }


                    var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                    if (where != null)
                    {
                        ret[np.invert(where)] = np.NaN;
                    }

                    return ret;
                }
            }
            else
            {
                MathFunctionHelper<long> ch = new MathFunctionHelper<long>(x);


                for (int i = 0; i < ch.expectedLength; i++)
                {
                    long f = ch.X1(i);

                    if (f == 0)
                    {
                        ch.results[i] = 0;
                    }
                    else
                    {
                        ch.results[i] = Math.Sign(f) < 0 ? -1 : 1;
                    }
                }

                var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    ret[np.invert(where)] = np.NaN;
                }

                return ret;
            }

        }


        public static ndarray convolve(object a, object v, NPY_CONVOLE_MODE mode = NPY_CONVOLE_MODE.NPY_CONVOLVE_FULL)
        {
            //  Returns the discrete, linear convolution of two one - dimensional sequences.

            //  The convolution operator is often seen in signal processing, where it
            //  models the effect of a linear time-invariant system on a signal[1]_.In
            // probability theory, the sum of two independent random variables is
            // distributed according to the convolution of their individual
            //  distributions.

            //  If `v` is longer than `a`, the arrays are swapped before computation.

            //  Parameters
            //  ----------
            //  a: (N,) array_like
            //     First one - dimensional input array.
            //  v: (M,) array_like
            //     Second one - dimensional input array.
            //  mode: { 'full', 'valid', 'same'}, optional
            //      'full':
            //        By default, mode is 'full'.This returns the convolution
            //      at each point of overlap, with an output shape of(N+M - 1,). At
            //       the end - points of the convolution, the signals do not overlap

            //         completely, and boundary effects may be seen.

            //      'same':
            //        Mode 'same' returns output of length ``max(M, N)``.  Boundary
            //        effects are still visible.

            //      'valid':
            //        Mode 'valid' returns output of length
            //        ``max(M, N) - min(M, N) + 1``.  The convolution product is only given
            //        for points where the signals overlap completely.Values outside
            //        the signal boundary have no effect.

            //  Returns
            //  ------ -
            //  out : ndarray
            //      Discrete, linear convolution of `a` and `v`.

            //  See Also
            //  --------
            //  scipy.signal.fftconvolve : Convolve two arrays using the Fast Fourier
            //                             Transform.
            //  scipy.linalg.toeplitz : Used to construct the convolution operator.
            //  polymul: Polynomial multiplication. Same output as convolve, but also
            //            accepts poly1d objects as input.

            //  Notes
            //  -----
            //  The discrete convolution operation is defined as

            //  .. math:: (a * v)[n] = \\sum_{ m = -\\infty}^{\\infty}
            //          a[m] v[n - m]

            //  It can be shown that a convolution: math:`x(t) * y(t)` in time / space
            //  is equivalent to the multiplication :math:`X(f) Y(f)` in the Fourier
            //  domain, after appropriate padding(padding is necessary to prevent
            //  circular convolution).Since multiplication is more efficient (faster)
            //  than convolution, the function `scipy.signal.fftconvolve` exploits the
            //  FFT to calculate the convolution of large data-sets.

            //  References
            //  ----------
            //  .. [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.

            //          Examples
            //          --------
            //  Note how the convolution operator flips the second array
            //  before "sliding" the two across one another:

            //  >>> np.convolve([1, 2, 3], [0, 1, 0.5])
            //  array([ 0. ,  1. ,  2.5,  4. ,  1.5])

            //  Only return the middle values of the convolution.
            //  Contains boundary effects, where zeros are taken
            //  into account:

            //  >>> np.convolve([1, 2, 3],[0, 1, 0.5], 'same')
            //  array([ 1. ,  2.5,  4. ])

            //  The two arrays are of the same length, so there
            //  is only one position where they completely overlap:

            //  >>> np.convolve([1, 2, 3],[0, 1, 0.5], 'valid')
            //  array([ 2.5])


            var a_arr = array(a, copy: false, ndmin: 1);
            var v_arr = array(v, copy: false, ndmin: 1);

            if (len(v_arr) > len(a_arr))
            {
                var temp = a_arr;
                v_arr = a_arr;
                a_arr = v_arr;
            }

            if (len(v_arr) == 0)
                throw new ValueError("a cannot be empty");
            if (len(v_arr) == 0)
                throw new ValueError("v cannot be empty");

            return correlate(a_arr, v_arr["::-1"], mode);
        }


        #endregion

    }

    #region MathHelper
    // special thanks to David Relihan
    // derived from here: https://stackoverflow.com/posts/5790661/edit
    public static class MathHelper
    {
        // Secant 
        public static double Sec(double x)
        {
            return 1 / Math.Cos(x);
        }

        // Cosecant
        public static double Cosec(double x)
        {
            return 1 / Math.Sin(x);
        }

        // Cotangent 
        public static double Cotan(double x)
        {
            return 1 / Math.Tan(x);
        }

        // Inverse Sine 
        public static double Arcsin(double x)
        {
            return Math.Atan(x / Math.Sqrt(-x * x + 1));
        }

        // Inverse Cosine 
        public static double Arccos(double x)
        {
            return Math.Atan(-x / Math.Sqrt(-x * x + 1)) + 2 * Math.Atan(1);
        }


        // Inverse Secant 
        public static double Arcsec(double x)
        {
            return 2 * Math.Atan(1) - Math.Atan(Math.Sign(x) / Math.Sqrt(x * x - 1));
        }

        // Inverse Cosecant 
        public static double Arccosec(double x)
        {
            return Math.Atan(Math.Sign(x) / Math.Sqrt(x * x - 1));
        }

        // Inverse Cotangent 
        public static double Arccotan(double x)
        {
            return 2 * Math.Atan(1) - Math.Atan(x);
        }

        // Hyperbolic Sine 
        public static double HSin(double x)
        {
            return (Math.Exp(x) - Math.Exp(-x)) / 2;
        }

        // Hyperbolic Cosine 
        public static double HCos(double x)
        {
            return (Math.Exp(x) + Math.Exp(-x)) / 2;
        }

        // Hyperbolic Tangent 
        public static double HTan(double x)
        {
            return (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
        }

        // Hyperbolic Secant 
        public static double HSec(double x)
        {
            return 2 / (Math.Exp(x) + Math.Exp(-x));
        }

        // Hyperbolic Cosecant 
        public static double HCosec(double x)
        {
            return 2 / (Math.Exp(x) - Math.Exp(-x));
        }

        // Hyperbolic Cotangent 
        public static double HCotan(double x)
        {
            return (Math.Exp(x) + Math.Exp(-x)) / (Math.Exp(x) - Math.Exp(-x));
        }

        // Inverse Hyperbolic Sine 
        public static double HArcsin(double x)
        {
            return Math.Log(x + Math.Sqrt(x * x + 1));
        }

        // Inverse Hyperbolic Cosine 
        public static double HArccos(double x)
        {
            return Math.Log(x + Math.Sqrt(x * x - 1));
        }

        // Inverse Hyperbolic Tangent 
        public static double HArctan(double x)
        {
            return Math.Log((1 + x) / (1 - x)) / 2;
        }

        // Inverse Hyperbolic Secant 
        public static double HArcsec(double x)
        {
            return Math.Log((Math.Sqrt(-x * x + 1) + 1) / x);
        }

        // Inverse Hyperbolic Cosecant 
        public static double HArccosec(double x)
        {
            return Math.Log((Math.Sign(x) * Math.Sqrt(x * x + 1) + 1) / x);
        }

        // Inverse Hyperbolic Cotangent 
        public static double HArccotan(double x)
        {
            return Math.Log((x + 1) / (x - 1)) / 2;
        }

        // Logarithm to base N 
        public static double LogN(double x, double n)
        {
            return Math.Log(x) / Math.Log(n);
        }
    }
    #endregion
}
