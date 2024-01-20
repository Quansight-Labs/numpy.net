/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018-2021
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
        private static double DNAN = double.NaN;
        private static float FNAN = float.NaN;
        private static System.Numerics.Complex CNAN = new System.Numerics.Complex(double.NaN, double.NaN);
        private static System.Numerics.BigInteger BNAN = new System.Numerics.BigInteger(0);

        #region MathFunctionHelper

        class MathFunctionHelper<T>
        {
            private ndarray a;
            private ndarray b;

            public shape expectedShape = null;
            public int expectedLength = 0;

            private npy_intp[] offsets = null;
            private npy_intp[] offsets2 = null;

            private T[] x1 = null;
            private T[] x2 = null;

            public T[] results = null;

            private NPY_TYPES target_nptype;
            private dtype target_dtype;

            public void SetupTypes()
            {
                T[] n = new T[1];

                target_nptype = DefaultArrayHandlers.GetArrayType(n[0]);
                target_dtype = NpyCoreApi.DescrFromType(target_nptype);
            }


            public MathFunctionHelper(object x)
            {
                SetupTypes();

                a = asanyarray(x);
                if (a.TypeNum != target_nptype)
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

                npy_intp asize = NpyCoreApi.ArraySize(a);
                npy_intp bsize = NpyCoreApi.ArraySize(b);

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

                if (a.TypeNum != target_nptype)
                {
                    a = a.astype(dtype: target_dtype);
                }
                if (b.TypeNum != target_nptype)
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

        #region Templated common functions
 
        private static ndarray MathFunction<T>(object x, object where, T NAN, Func<T, T> mathfunc)
        {
            var ch = new MathFunctionHelper<T>(x);

            Parallel.For(0, ch.expectedLength, i =>
            {
                ch.results[i] = mathfunc(ch.X1((int)i));
            });

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = NAN;
            }

            return ret;
        }
        private static ndarray MathFunction<T>(object x1, object x2, object where, T NAN, Func<T, T, T> mathfunc)
        {
            var ch = new MathFunctionHelper<T>(x1, x2);

            Parallel.For(0, ch.expectedLength, i =>
            {
                ch.results[i] = mathfunc(ch.X1(i), ch.X2(i));
            });

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = NAN;
            }

            return ret;
        }

        private static ndarray MathFunction<T, I>(object x1, I n, object where, T NAN, Func<T, I, T> mathfunc)
        {
            var ch = new MathFunctionHelper<T>(x1);

            Parallel.For(0, ch.expectedLength, i =>
            {
                ch.results[i] = mathfunc(ch.X1(i), n);
            });

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = NAN;
            }

            return ret;
        }

        private static ndarray MathFunction<T,I>(object x1, object x2, I n, object where, T NAN, Func<T, T, I, T> mathfunc)
        {
            var ch = new MathFunctionHelper<T>(x1, x2);

            Parallel.For(0, ch.expectedLength, i =>
            {
                ch.results[i] = mathfunc(ch.X1(i), ch.X2(i), n);
            });

            var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
            if (where != null)
            {
                ret[np.invert(where)] = NAN;
            }

            return ret;
        }


        #endregion

        #region Trigonometric Functions

        /// <summary>
        /// Trigonometric sine, element-wise.
        /// </summary>
        /// <param name="x">Angle, in radians (2pi rad equals 360 degrees).</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray sin(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Sin(value); };
                return MathFunction<System.Numerics.Complex>(x, where, CNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return Math.Sin(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Sin(value);};
                return MathFunction(x, where, DNAN, mathfunc);
            }
 
        }

        /// <summary>
        /// Cosine element-wise.
        /// </summary>
        /// <param name="x">Input array in radians.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray cos(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Cos(value); };
                return MathFunction<System.Numerics.Complex>(x, where, CNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return Math.Cos(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Cos(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }

        }

        /// <summary>
        /// Compute tangent element-wise.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray tan(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Tan(value); };
                return MathFunction<System.Numerics.Complex>(x, where, CNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return Math.Tan(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Tan(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
 
        }

        /// <summary>
        /// Inverse sine, element-wise.
        /// </summary>
        /// <param name="x">y-coordinate on the unit circle.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray arcsin(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Asin(value); };
                return MathFunction<System.Numerics.Complex>(x, where, CNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return Math.Asin(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Asin(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
 
        }
        /// <summary>
        /// Trigonometric inverse cosine, element-wise.
        /// </summary>
        /// <param name="x">x-coordinate on the unit circle.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray arccos(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Acos(value); };
                return MathFunction<System.Numerics.Complex>(x, where, CNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return Math.Acos(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Acos(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
        }
        /// <summary>
        /// Trigonometric inverse tangent, element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray arctan(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Atan(value); };
                return MathFunction<System.Numerics.Complex>(x, where, CNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return Math.Atan(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Atan(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Given the “legs” of a right triangle, return its hypotenuse.
        /// </summary>
        /// <param name="x1">Leg of the triangle(s).</param>
        /// <param name="x2">Leg of the triangle(s).</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray hypot(object x1, object x2, object where = null)
        {
            var xa = asanyarray(x1);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                MathFunctionHelper<System.Numerics.Complex> ch = new MathFunctionHelper<System.Numerics.Complex>(x1, x2);

                var hypot = np.sqrt(np.power(x1, 2) + np.power(x2, 2));

                return hypot;
            }
            else
            if (xa.IsBigInt)
            {
                MathFunctionHelper<System.Numerics.BigInteger> ch = new MathFunctionHelper<System.Numerics.BigInteger>(x1, x2);

                var hypot = np.sqrt(np.power(x1, 2) + np.power(x2, 2));

                return hypot;
            }
            else
            {
                MathFunctionHelper<double> ch = new MathFunctionHelper<double>(x1, x2);

                var hypot = np.sqrt(np.power(x1, 2) + np.power(x2, 2));

                return hypot;
            }

        }
        /// <summary>
        /// Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
        /// </summary>
        /// <param name="x1">y-coordinates.</param>
        /// <param name="x2">x-coordinates.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray arctan2(object x1, object x2, object where = null)
        {
            var xa = asanyarray(x1);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value1, value2) => { return Math.Atan2(value1.Real, value2.Real); };
                return MathFunction<System.Numerics.Complex>(x1, x2, where, CNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double, double> mathfunc = (value1, value2) => { return Math.Atan2(value1, value2); };
                return MathFunction<double>(x1, x2, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double, double> mathfunc = (value1, value2) => { return Math.Atan2(value1, value2); };
                return MathFunction<double>(x1, x2, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Convert angles from radians to degrees.
        /// </summary>
        /// <param name="x">Angle in radians.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray rad2deg(object x, object where = null)
        {
            return degrees(x, where);
        }
        /// <summary>
        /// Convert angles from radians to degrees.
        /// </summary>
        /// <param name="x">Angle in radians.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray degrees(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return value * (180 / Math.PI); };
                return MathFunction<System.Numerics.Complex>(x, where, CNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return value * (180 / Math.PI); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return value * (180 / Math.PI); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
 
        }
        /// <summary>
        /// Convert angles from degrees to radians.
        /// </summary>
        /// <param name="x">Input array in degrees.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray deg2rad(object x, object where = null)
        {
            return radians(x, where);
        }
        /// <summary>
        /// Convert angles from degrees to radians.
        /// </summary>
        /// <param name="x">Input array in degrees.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray radians(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return Math.PI * value / 180; };
                return MathFunction<System.Numerics.Complex>(x, where, CNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return Math.PI * value / 180; };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.PI * value / 180; };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            
        }

        #endregion

        #region Hyperbolic functions
        /// <summary>
        /// Hyperbolic sine, element-wise.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray sinh(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Sinh(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return Math.Sinh(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Sinh(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Hyperbolic cosine, element-wise.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray cosh(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Cosh(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return Math.Cosh(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Cosh(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Compute hyperbolic tangent element-wise.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray tanh(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Tanh(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return Math.Tanh(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Tanh(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Inverse hyperbolic sine element-wise.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray arcsinh(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return MathHelper.HArcsin(value.Real); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return MathHelper.HArcsin(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return MathHelper.HArcsin(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Inverse hyperbolic cosine, element-wise.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray arccosh(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return MathHelper.HArccos(value.Real); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return MathHelper.HArccos(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return MathHelper.HArccos(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Inverse hyperbolic tangent element-wise.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray arctanh(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return MathHelper.HArctan(value.Real); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return MathHelper.HArctan(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return MathHelper.HArctan(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }

        }
        #endregion

        #region Rounding Functions
        /// <summary>
        /// Round elements of the array to the nearest integer.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray rint(object x, object where = null)
        {
            var a = asanyarray(x);

            if (!a.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(a);
            }

            var ret = NpyCoreApi.PerformNumericOp(a, UFuncOperation.rint, 0);
            ret = ret.reshape(new shape(a.dims));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }
        /// <summary>
        /// Round to nearest integer towards zero.
        /// </summary>
        /// <param name="x">An array of floats to be rounded</param>
        /// <returns></returns>
        public static ndarray fix(object x)
        {
            var a = asanyarray(x);

            if (!a.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(a);
            }

            var y1 = np.floor(a);
            var y2 = np.ceil(a);

            y1["..."] = np.where(a >= 0, y1, y2);
            return y1;
        }
        /// <summary>
        /// Return the ceiling of the input, element-wise.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray ceil(object x, object where = null)
        {
            var a = asanyarray(x);

 
            var ret = NpyCoreApi.PerformNumericOp(a, UFuncOperation.ceil, 0);
            ret = ret.reshape(new shape(a.dims));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }
        /// <summary>
        /// Return the truncated value of the input, element-wise.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <returns></returns>
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
        /// <summary>
        /// Calculate the exponential of all elements in the input array.
        /// </summary>
        /// <param name="x">Input values.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result. </param>
        /// <returns></returns>
        public static ndarray exp(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Exp(value); };
                return MathFunction(x, where, CNAN, mathfunc);
            }
            if (xa.IsBigInt)
            {
                Func<double, double> mathfunc = (value) => { return Math.Exp(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Exp(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
  
        }
        /// <summary>
        /// Calculate exp(x) - 1 for all elements in the array.
        /// </summary>
        /// <param name="x">Input values.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray expm1(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => 
                {
                    if (System.Numerics.Complex.Abs(value) < 1e-05)
                    {
                        return value + 0.5 * value * value;
                    }
                    else
                    {
                        return System.Numerics.Complex.Exp(value) - 1.0;
                    }
                };
                return MathFunction(x, where, CNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => 
                {
                    if (Math.Abs(value) < 1e-05)
                    {
                        return  value + 0.5 * value * value;
                    }
                    else
                    {
                        return Math.Exp(value) - 1.0;
                    }
                };
                return MathFunction(x, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Calculate 2**p for all p in the input array.
        /// </summary>
        /// <param name="x">Input values.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray exp2(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Pow(2, value); };
                return MathFunction(x, where, CNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Pow(2, value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }
  
        }
        /// <summary>
        /// Natural logarithm, element-wise.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray log(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Log(value); };
                return MathFunction(x, where, CNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Log(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Return the base 10 logarithm of the input array, element-wise.
        /// </summary>
        /// <param name="x">Input values.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray log10(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Log10(value); };
                return MathFunction(x, where, CNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Log10(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Base-2 logarithm of x.
        /// </summary>
        /// <param name="x">Input values.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result. </param>
        /// <returns></returns>
        public static ndarray log2(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Log(value, 2); };
                return MathFunction(x, where, CNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Log(value, 2); };
                return MathFunction(x, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Take log base n of x.
        /// </summary>
        /// <param name="x">The value(s) whose log base n is (are) required.</param>
        /// <param name="n">The integer base(s) in which the log is taken.</param>
        /// <param name="where"></param>
        /// <returns></returns>
        public static ndarray logn(object x, int n, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, int, System.Numerics.Complex> mathfunc = (value, n1) => { return System.Numerics.Complex.Log(value, n1); };
                return MathFunction(x, n, where, CNAN, mathfunc);
            }
            else
            {
                Func<double,int,double> mathfunc = (value, n1) => { return Math.Log(value, n1); };
                return MathFunction(x, n, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Return the natural logarithm of one plus the input array, element-wise.
        /// </summary>
        /// <param name="x">Input values.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray log1p(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value) => { return System.Numerics.Complex.Log(value); };
                return MathFunction(x, where, CNAN, mathfunc);
            }
            else
            {
                Func<double, double> mathfunc = (value) => { return Math.Log(value); };
                return MathFunction(x, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Logarithm of the sum of exponentiations of the inputs.
        /// </summary>
        /// <param name="x1">Input values.</param>
        /// <param name="x2">Input values.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray logaddexp(object x1, object x2, object where = null)
        {
            var xa = asanyarray(x1);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value1, value2) => 
                {
                    return System.Numerics.Complex.Exp(value1) + System.Numerics.Complex.Exp(value2);
                };
                return MathFunction(x1, x2, where, CNAN, mathfunc);
            }
            else
            {
                Func<double, double,double> mathfunc = (value1, value2) => 
                {
                    return Math.Log(Math.Exp(value1) + Math.Exp(value2));
                };
                return MathFunction(x1, x2, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Logarithm of the sum of exponentiations of the inputs in base-2.
        /// </summary>
        /// <param name="x1">Input values.</param>
        /// <param name="x2">Input values.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray logaddexp2(object x1, object x2, object where = null)
        {
            var xa = asanyarray(x1);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex, System.Numerics.Complex> mathfunc = (value1, value2) =>
                {
                    return System.Numerics.Complex.Log(System.Numerics.Complex.Pow(2, value1) + System.Numerics.Complex.Pow(2, value2), 2);
                };
                return MathFunction(x1, x2, where, CNAN, mathfunc);
            }
            else
            {
                Func<double, double, double> mathfunc = (value1, value2) =>
                {
                    return Math.Log(Math.Pow(2, value1) + Math.Pow(2, value2), 2);
                };
                return MathFunction(x1, x2, where, DNAN, mathfunc);
            }

        }
        /// <summary>
        /// Logarithm of the sum of exponentiations of the inputs in base-n.
        /// </summary>
        /// <param name="x1">Input values.</param>
        /// <param name="x2">Input values.</param>
        /// <param name="n">The integer base(s) in which the log is taken.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray logaddexpn(object x1, object x2, int n, object where = null)
        {
            var xa = asanyarray(x1);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                Func<System.Numerics.Complex, System.Numerics.Complex, int, System.Numerics.Complex> mathfunc = (value1, value2, n1) =>
                {
                    return System.Numerics.Complex.Log(System.Numerics.Complex.Pow(n1, value1) + System.Numerics.Complex.Pow(n1, value2), 2);
                };
                return MathFunction(x1, x2, n, where, CNAN, mathfunc);
            }
            else
            {
                Func<double, double, int, double> mathfunc = (value1, value2, n1) =>
                {
                    return Math.Log(Math.Pow(n1, value1) + Math.Pow(n1, value2), 2);
                };
                return MathFunction(x1, x2, n, where, DNAN, mathfunc);
            }

        }


        #endregion

        #region Other special functions

        #endregion

        #region Floating point routines
        /// <summary>
        /// Returns element-wise True where signbit is set (less than zero).
        /// </summary>
        /// <param name="x">The input value(s).</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray signbit(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                MathFunctionHelper<System.Numerics.Complex> ch = new MathFunctionHelper<System.Numerics.Complex>(x);
                ch.results = null;

                bool[] bret = new bool[ch.expectedLength];

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    System.Numerics.Complex f = ch.X1(i);
                    bret[i] = Math.Sign(f.Real) < 0 ? true : false;
                }

                var ret = np.array(bret).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    ret[np.invert(where)] = np.NaN;
                }

                return ret;
            }

            if (xa.IsFloatingPoint)
            {
                if (xa.ItemSize <= sizeof(float))
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

        /// <summary>
        /// Change the sign of x1 to that of x2, element-wise.
        /// </summary>
        /// <param name="x1">Values to change the sign of.</param>
        /// <param name="x2">The sign of x2 is copied to x1</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray copysign(object x1, object x2, object where = null)
        {
            var xa = asanyarray(x1);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                MathFunctionHelper<System.Numerics.Complex> ch = new MathFunctionHelper<System.Numerics.Complex>(x1, x2);

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    var a = ch.X1(i);
                    var b = ch.X2(i);

                    double Real = a.Real;
                    double Imaginary = a.Imaginary;
                    if (Math.Sign(a.Real) != Math.Sign(b.Real))
                    {
                        Real = -a.Real;
                    }
                    if (Math.Sign(a.Imaginary) != Math.Sign(b.Imaginary))
                    {
                        Imaginary = -a.Imaginary;
                    }
                    
                    ch.results[i] = new System.Numerics.Complex(Real, Imaginary);
                }

                var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    ret[np.invert(where)] = np.NaN;
                }

                return ret;
            }

            if (xa.IsFloatingPoint)
            {
                if (xa.ItemSize <= sizeof(float))
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
        /// <summary>
        /// Decompose the elements of x into mantissa and twos exponent.
        /// </summary>
        /// <param name="x">Array of numbers to be decomposed.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray[] frexp(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                MathFunctionHelper<System.Numerics.Complex> ch = new MathFunctionHelper<System.Numerics.Complex>(x);

                int[] _exponents = new int[ch.expectedLength];

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    System.Numerics.Complex f = ch.X1(i);

                    int exponent = 0;
                    ch.results[i] = MachineCognitus.math.frexp(f.Real, ref exponent);
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


            if (xa.TypeNum == NPY_TYPES.NPY_FLOAT)
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
        /// <summary>
        /// Returns x1 * 2**x2, element-wise.
        /// </summary>
        /// <param name="x1">Array of multipliers.</param>
        /// <param name="x2">Array of twos exponents.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result. </param>
        /// <returns></returns>
        public static ndarray ldexp(object x1, object x2, object where = null)
        {
            var a1 = asanyarray(x1);
            var a2 = asanyarray(x2);

            if (!a1.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(a1);
            }
            if (!a2.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(a2);
            }

            if (a1.IsComplex)
            {
                MathFunctionHelper<System.Numerics.Complex> ch = new MathFunctionHelper<System.Numerics.Complex>(x1, x2);

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    ch.results[i] = MachineCognitus.math.ldexp(ch.X1(i).Real, Convert.ToInt32(ch.X2(i).Real));
                }

                var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    ret[np.invert(where)] = np.NaN;
                }

                return ret;
            }

            if (a1.ItemSize <= sizeof(float) && a2.ItemSize <= sizeof(float))
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
        /// <summary>
        /// Return the next floating-point value after x1 towards x2, element-wise.
        /// </summary>
        /// <param name="x1">Values to find the next representable value of.</param>
        /// <param name="x2">The direction where to look for the next representable value of x1.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray nextafter(object x1, object x2, object where = null)
        {
            var a1 = asanyarray(x1);
            var a2 = asanyarray(x2);

            if (!a1.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(a1);
            }
            if (!a2.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(a2);
            }

            bool a1b = !a1.IsAScalar && a1.IsFloatingPoint && a1.ItemSize <= sizeof(float);
            bool a2b = !a2.IsAScalar && a2.IsFloatingPoint && a2.ItemSize <= sizeof(float);

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
        private static System.Numerics.BigInteger _gcdb(System.Numerics.BigInteger a, System.Numerics.BigInteger b)
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
        private static System.Numerics.BigInteger _lcmb(System.Numerics.BigInteger a, System.Numerics.BigInteger b)
        {
            return a / _gcdb(a, b) * b;
        }
        /// <summary>
        /// Returns the lowest common multiple of |x1| and |x2|
        /// </summary>
        /// <param name="x1">Arrays of values</param>
        /// <param name="x2">Arrays of values</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray lcm(object x1, object x2, object where = null)
        {
            var xa = asanyarray(x1);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsInteger)
            {
                if (xa.IsBigInt)
                {
                    Func<System.Numerics.BigInteger, System.Numerics.BigInteger, System.Numerics.BigInteger> mathfunc = (value1, value2) => { return _lcmb(value1, value2);};
                    return MathFunction<System.Numerics.BigInteger>(x1, x2, where, BNAN, mathfunc);
                }
                else
                if (xa.ItemSize <= sizeof(Int32))
                {
                    Func<Int32, Int32, Int32> mathfunc = (value1, value2) => { return _lcmi(value1, value2);};
                    return MathFunction<Int32>(x1, x2, where, 0, mathfunc);
                }
                else
                {
                    Func<Int64, Int64, Int64> mathfunc = (value1, value2) => { return _lcml(value1, value2);};
                    return MathFunction<Int64>(x1, x2, where, 0, mathfunc);
                }
            }
            else
            {
                throw new ValueError("This function only operates on integer type values");
            }

        }
        /// <summary>
        /// Returns the greatest common divisor of |x1| and |x2|
        /// </summary>
        /// <param name="x1">Arrays of values.</param>
        /// <param name="x2">Arrays of values.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray gcd(object x1, object x2, object where = null)
        {
            var xa = asanyarray(x1);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsInteger)
            {
                if (xa.IsBigInt)
                {
                    Func<System.Numerics.BigInteger, System.Numerics.BigInteger, System.Numerics.BigInteger> mathfunc = (value1, value2) => { return _gcdb(value1, value2); };
                    return MathFunction<System.Numerics.BigInteger>(x1, x2, where, BNAN, mathfunc);
                }
                else
                if (xa.ItemSize <= sizeof(Int32))
                {
                    Func<Int32, Int32, Int32> mathfunc = (value1, value2) => { return _gcdi(value1, value2); };
                    return MathFunction<Int32>(x1, x2, where, 0, mathfunc);
                }
                else
                {
                    Func<Int64, Int64, Int64> mathfunc = (value1, value2) => { return _gcdl(value1, value2); };
                    return MathFunction<Int64>(x1, x2, where, 0, mathfunc);
                }
            }
            else
            {
                throw new ValueError("This function only operates on integer type values");
            }

        }

        #endregion

        #region Arithmetic operations

        /// <summary>
        /// Add arguments element-wise.
        /// </summary>
        /// <param name="x1">The arrays to be added.</param>
        /// <param name="x2">The arrays to be added.</param>
        /// <param name="out">A location into which the result is stored.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray add(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.add, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        /// <summary>
        /// Return the reciprocal of the argument, element-wise.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns></returns>
        public static ndarray reciprocal(object a)
        {
            return NpyCoreApi.PerformNumericOp(asanyarray(a), UFuncOperation.reciprocal, 0);
        }
        /// <summary>
        /// Numerical positive, element-wise.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <returns></returns>
        public static ndarray positive(object x)
        {
            var a = asanyarray(x);
            return a.Copy();
        }
        /// <summary>
        /// Numerical negative, element-wise.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="out">A location into which the result is stored.</param>
        /// <returns></returns>
        public static ndarray negative(object x, ndarray @out = null)
        {
            var result = NpyCoreApi.PerformNumericOp(asanyarray(x), UFuncOperation.negative, 0, UseSrcAsDest: false );
            if (@out != null)
            {
                np.copyto(@out, result);
            }
            return result;
        }
        /// <summary>
        /// Multiply arguments element-wise.
        /// </summary>
        /// <param name="x1">Input arrays to be multiplied.</param>
        /// <param name="x2">Input arrays to be multiplied.</param>
        /// <param name="out">A location into which the result is stored.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray multiply(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.multiply, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        /// <summary>
        /// Returns a true division of the inputs, element-wise.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Divisor array.</param>
        /// <param name="out">A location into which the result is stored. </param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray divide(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.divide, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        /// <summary>
        /// First array elements raised to powers from second array, element-wise.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents.</param>
        /// <param name="out">A location into which the result is stored.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray power(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.power, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        /// <summary>
        /// Subtract arguments, element-wise.
        /// </summary>
        /// <param name="x1">The arrays to be subtracted from each other.</param>
        /// <param name="x2">The arrays to be subtracted from each other.</param>
        /// <param name="out">A location into which the result is stored. </param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result. </param>
        /// <returns></returns>
        public static ndarray subtract(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.subtract, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }

        /// <summary>
        /// Returns a true division of the inputs, element-wise.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Divisor array.</param>
        /// <param name="out">A location into which the result is stored.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray true_divide(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.true_divide, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        /// <summary>
        /// Return the largest integer smaller or equal to the division of the inputs. 
        /// </summary>
        /// <param name="x1">Numerator.</param>
        /// <param name="x2">Denominator. </param>
        /// <param name="out">A location into which the result is stored.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray floor_divide(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.floor_divide, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        /// <summary>
        /// First array elements raised to powers from second array, element-wise.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray float_power(object x1, object x2, object where = null)
        {
            var x1a = asanyarray(x1);

            if (!x1a.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(x1a);
            }
            if (!asanyarray(x2).IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(asanyarray(x2));
            }

            if (x1a.IsComplex)
            {
                MathFunctionHelper<System.Numerics.Complex> ch = new MathFunctionHelper<System.Numerics.Complex> (x1, x2);

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    ch.results[i] = System.Numerics.Complex.Pow(ch.X1(i), ch.X2(i));
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
                    ch.results[i] = Math.Pow(ch.X1(i), ch.X2(i));
                }

                var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    ret[np.invert(where)] = np.NaN;
                }

                return ret;
            }

 

        }

        #region mod/remainder
        /// <summary>
        /// Return element-wise remainder of division.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Divisor array</param>
        /// <returns></returns>
        public static ndarray mod(object x1, object x2)
        {
            return remainder(x1, x2);
        }
        /// <summary>
        /// Return element-wise remainder of division.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Divisor</param>
        /// <returns></returns>
        public static ndarray mod(object x1, int x2)
        {
            return remainder(x1, x2);
        }
        /// <summary>
        /// Return element-wise remainder of division.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Divisor array. </param>
        /// <param name="out">A location into which the result is stored. </param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray remainder(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.remainder, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        /// <summary>
        /// Return the element-wise remainder of division.
        /// </summary>
        /// <param name="x1">Dividend.</param>
        /// <param name="x2">Divisor</param>
        /// <returns></returns>
        public static ndarray fmod(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.fmod, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }


        #endregion


        #region divmod
        /// <summary>
        /// Return element-wise quotient and remainder simultaneously.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Divisor array.</param>
        /// <param name="out1">A location into which the result is stored.</param>
        /// <param name="out2">A location into which the result is stored.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray[] divmod(object x1, object x2, ndarray @out1 = null, ndarray @out2 = null, object where = null)
        {
            ndarray[] results = new ndarray[2];

            results[0] = NpyCoreApi.PerformUFUNC(UFuncOperation.floor_divide, asanyarray(x1), asanyarray(x2), @out1, asanyarray(where));
            results[1] = NpyCoreApi.PerformUFUNC(UFuncOperation.remainder, asanyarray(x1), asanyarray(x2), @out2, asanyarray(where));

            return results;
        }


        #endregion

        #region modf
        /// <summary>
        /// Return the fractional and integral parts of an array, element-wise.
        /// </summary>
        /// <param name="x1">Input array.</param>
        /// <param name="out1">A location into which the result is stored. </param>
        /// <param name="out2">A location into which the result is stored. </param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray[] modf(object x1, ndarray @out1 = null, ndarray @out2 = null, object where = null)
        {
            ndarray[] results = new ndarray[2];

            var x1a = asanyarray(x1);

            dtype ret_type = result_type(x1a.TypeNum);

            results[1] = NpyCoreApi.PerformUFUNC(UFuncOperation.floor_divide, asanyarray(x1).astype(ret_type), asanyarray(1), @out1, asanyarray(where));
            results[0] = NpyCoreApi.PerformUFUNC(UFuncOperation.remainder, asanyarray(x1).astype(ret_type), asanyarray(1), @out2, asanyarray(where));

            return results;
        }


        #endregion

        #endregion

        #region Misc
        /// <summary>
        /// Returns an element-wise indication of the sign of a number.
        /// </summary>
        /// <param name="x">Input values.</param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray sign(object x, object where = null)
        {
            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            if (xa.IsComplex)
            {
                MathFunctionHelper<System.Numerics.Complex> ch = new MathFunctionHelper<System.Numerics.Complex>(x);

                for (int i = 0; i < ch.expectedLength; i++)
                {
                    double Real = 0.0;
                    double Imaginary = 0.0;

                    System.Numerics.Complex f = ch.X1(i);


                    if (f.Real == 0.0)
                    {
                        Real = 0.0;
                    }
                    else
                    {
                        Real = Math.Sign(f.Real) < 0 ? -1.0 : 1.0;
                    }

                    if (f.Imaginary == 0.0)
                    {
                        Imaginary = 0.0;
                    }
                    else
                    {
                        Imaginary = Math.Sign(f.Real) < 0 ? -1.0 : 1.0;
                    }
                    ch.results[i] = new System.Numerics.Complex(Real, Imaginary);
                }


                var ret = np.array(ch.results).reshape(new shape(ch.expectedShape));
                if (where != null)
                {
                    ret[np.invert(where)] = np.NaN;
                }

                return ret;
            }

            if (xa.IsFloatingPoint)
            {
                if (xa.ItemSize <= sizeof(float))
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

        /// <summary>
        /// Returns the discrete, linear convolution of two one-dimensional sequences.
        /// </summary>
        /// <param name="a">First one-dimensional input array.</param>
        /// <param name="v">Second one-dimensional input array.</param>
        /// <param name="mode">{‘full’, ‘valid’, ‘same’}, optional</param>
        /// <returns></returns>
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
                a_arr = v_arr;
                v_arr = temp;
            }

            if (len(v_arr) == 0)
                throw new ValueError("a cannot be empty");
            if (len(v_arr) == 0)
                throw new ValueError("v cannot be empty");

            return correlate(a_arr, v_arr["::-1"], mode);
        }
        /// <summary>
        /// Return the complex conjugate, element-wise.
        /// </summary>
        /// <param name="a">Input value.</param>
        /// <param name="out">A location into which the result is stored. </param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result. </param>
        /// <returns></returns>
        public static ndarray conj(object a, ndarray @out = null, object where = null)
        {
            var result = NpyCoreApi.PerformNumericOp(asanyarray(a), UFuncOperation.conjugate, 0, UseSrcAsDest: false);
            if (@out != null)
            {
                np.copyto(@out, result);
            }
            return result;
        }
        /// <summary>
        /// Return the complex conjugate, element-wise.
        /// </summary>
        /// <param name="x1">Input value.</param>
        /// <param name="out">A location into which the result is stored. </param>
        /// <param name="where">This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray conjugate(object x1, ndarray @out = null, object where = null)
        {
            return conj(x1, @out, where);
        }

        #endregion

        #region Error handling

        private static void ArrayTypeNotSupported(ndarray a, [System.Runtime.CompilerServices.CallerMemberName] string memberName = "")
        {
            string arrayType = a.TypeNum.ToString().Substring(4);
            throw new Exception(string.Format("Arrays of type {0} are not supported by {1}", arrayType, "np." + memberName));
        }

        #endregion
    }

    #region MathHelper
    // special thanks to David Relihan
    // derived from here: https://stackoverflow.com/posts/5790661/edit
    internal static class MathHelper
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
