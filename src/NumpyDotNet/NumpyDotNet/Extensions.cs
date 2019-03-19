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

using NumpyDotNet;
using NumpyLib;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{
    public static class NumpyExtensions
    {
        public static ndarray reshape(this ndarray a, shape newshape, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            return np.reshape(a, newshape, order);
        }
        public static ndarray reshape(this ndarray a, object oshape, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            shape newshape = ConvertTupleToShape(oshape);
            if (newshape == null)
            {
                throw new Exception("Unable to convert shape object");
            }
            
            return np.reshape(a, newshape, order);
        }

        public static shape ConvertTupleToShape(object oshape)
        {
            var T = oshape.GetType();

            if (oshape is Int32)
            {
                return new shape((Int32)oshape);
            }
            if (oshape is Int64)
            {
                return new shape((Int64)oshape);
            }

            if (oshape is ValueTuple<int>)
            {
                ValueTuple<int> T1 = (ValueTuple<int>)oshape;
                return new shape(T1.Item1);
            }
            if (oshape is ValueTuple<long>)
            {
                ValueTuple<long> T1 = (ValueTuple<long>)oshape;
                return new shape(T1.Item1);
            }

            if (oshape is ValueTuple<int, int>)
            {
                ValueTuple<int, int> T2 = (ValueTuple<int, int>)oshape;
                return new shape(T2.Item1, T2.Item2);
            }
            if (oshape is ValueTuple<long, long>)
            {
                ValueTuple<long, long> T2 = (ValueTuple<long, long>)oshape;
                return new shape(T2.Item1, T2.Item2);
            }

            if (oshape is ValueTuple<int, int, int>)
            {
                ValueTuple<int, int, int> T3 = (ValueTuple<int, int, int>)oshape;
                return new shape(T3.Item1, T3.Item2, T3.Item3);
            }
            if (oshape is ValueTuple<long, long, long>)
            {
                ValueTuple<long, long, long> T3 = (ValueTuple<long, long, long>)oshape;
                return new shape(T3.Item1, T3.Item2, T3.Item3);
            }

            if (oshape is ValueTuple<int, int, int, int>)
            {
                ValueTuple<int, int, int, int> T4 = (ValueTuple<int, int, int, int>)oshape;
                return new shape(T4.Item1, T4.Item2, T4.Item3, T4.Item4);
            }
            if (oshape is ValueTuple<long, long, long, long>)
            {
                ValueTuple<long, long, long, long> T4 = (ValueTuple<long, long, long, long>)oshape;
                return new shape(T4.Item1, T4.Item2, T4.Item3, T4.Item4);
            }

            if (oshape is ValueTuple<int, int, int, int, int>)
            {
                ValueTuple<int, int, int, int, int> T5 = (ValueTuple<int, int, int, int, int>)oshape;
                return new shape(new npy_intp[] { T5.Item1, T5.Item2, T5.Item3, T5.Item4, T5.Item5 });
            }
            if (oshape is ValueTuple<long, long, long, long, long>)
            {
                ValueTuple<long, long, long, long, long> T5 = (ValueTuple<long, long, long, long, long>)oshape;
                return new shape(new npy_intp[] { T5.Item1, T5.Item2, T5.Item3, T5.Item4, T5.Item5 });
            }

            if (oshape is ValueTuple<int, int, int, int, int, int>)
            {
                ValueTuple<int, int, int, int, int, int> T6 = (ValueTuple<int, int, int, int, int, int>)oshape;
                return new shape(new npy_intp[] { T6.Item1, T6.Item2, T6.Item3, T6.Item4, T6.Item5, T6.Item6 });
            }
            if (oshape is ValueTuple<long, long, long, long, long, long>)
            {
                ValueTuple<long, long, long, long, long, long> T6 = (ValueTuple<long, long, long, long, long, long>)oshape;
                return new shape(new npy_intp[] { T6.Item1, T6.Item2, T6.Item3, T6.Item4, T6.Item5, T6.Item6 });
            }

            if (oshape is ValueTuple<int, int, int, int, int, int, int>)
            {
                ValueTuple<int, int, int, int, int, int, int> T7 = (ValueTuple<int, int, int, int, int, int, int>)oshape;
                return new shape(new npy_intp[] { T7.Item1, T7.Item2, T7.Item3, T7.Item4, T7.Item5, T7.Item6, T7.Item7 });
            }
            if (oshape is ValueTuple<long, long, long, long, long, long, long>)
            {
                ValueTuple<long, long, long, long, long, long, long> T7 = (ValueTuple<long, long, long, long, long, long, long>)oshape;
                return new shape(new npy_intp[] { T7.Item1, T7.Item2, T7.Item3, T7.Item4, T7.Item5, T7.Item6, T7.Item7 });
            }

            if (oshape is ValueTuple<int, int, int, int, int, int, int, int>)
            {
                ValueTuple<int, int, int, int, int, int, int, int> T8 = (ValueTuple<int, int, int, int, int, int, int, int>)oshape;
                return new shape(new npy_intp[] { T8.Item1, T8.Item2, T8.Item3, T8.Item4, T8.Item5, T8.Item6, T8.Item7, T8.Rest });
            }
            if (oshape is ValueTuple<long, long, long, long, long, long, long, long>)
            {
                ValueTuple<long, long, long, long, long, long, long, long> T8 = (ValueTuple<long, long, long, long, long, long, long, long>)oshape;
                return new shape(new npy_intp[] { T8.Item1, T8.Item2, T8.Item3, T8.Item4, T8.Item5, T8.Item6, T8.Item7, T8.Rest });
            }
            return null;
        }


        public static void tofile(this ndarray a, string fileName, string sep = null, string format = null)
        {
            np.tofile(a, fileName, sep, format);
        }

        public static void tofile(this ndarray a, Stream stream, string sep = null, string format = null)
        {
            np.tofile(a, stream, sep, format);
        }

        public static ndarray view(this ndarray a, dtype dtype = null, object type = null)
        {
            return np.view(a, dtype, type);
        }

        public static ndarray Flatten(this ndarray a, NPY_ORDER order)
        {
            return NpyCoreApi.Flatten(a, order);
        }

        public static ndarray Ravel(this ndarray a, NPY_ORDER order)
        {
            return np.ravel(a, order);
        }

        public static void Resize(this ndarray a, npy_intp[] newdims, bool refcheck, NPY_ORDER order)
        {
            np.resize(a, newdims, refcheck, order);
        }

        public static ndarray Squeeze(this ndarray a)
        {
            return np.squeeze(a);
        }

        public static ndarray SwapAxes(this ndarray a, int a1, int a2)
        {
            return np.swapaxes(a, a1, a2);
        }
 
        public static ndarray Transpose(this ndarray a, npy_intp[] permute = null)
        {
            return np.transpose(a, permute);
        }

        public static ndarray Choose(this ndarray a, IEnumerable<ndarray> choices, ndarray ret = null, NPY_CLIPMODE clipMode = NPY_CLIPMODE.NPY_RAISE)
        {
            return np.choose(a, choices, ret, clipMode);
        }


        public static ndarray Repeat(this ndarray a, object repeats, int? axis)
        {
            return np.repeat(a, repeats, axis);
        }

        public static void PutTo(this ndarray a, ndarray values, ndarray indices, NPY_CLIPMODE mode)
        {
            int ret = np.put(a, indices, values, mode);
        }

        public static ndarray Sort(this ndarray a, int? axis = -1, NPY_SORTKIND sortkind = NPY_SORTKIND.NPY_QUICKSORT)
        {
            return np.sort(a, axis, sortkind, null);
        }


        public static ndarray ArgSort(this ndarray a, int? axis =-1, NPY_SORTKIND kind = NPY_SORTKIND.NPY_QUICKSORT)
        {
            return np.argsort(a, axis, kind);
        }

        public static ndarray ArgMax(this ndarray a, int? axis = -1, ndarray ret = null)
        {
            return np.argmax(a, axis, ret);
        }

        public static ndarray ArgMin(this ndarray a, int? axis = -1, ndarray ret = null)
        {
            return np.argmin(a, axis, ret);
        }

        public static ndarray SearchSorted(this ndarray a, ndarray keys, NPY_SEARCHSIDE side = NPY_SEARCHSIDE.NPY_SEARCHLEFT)
        {
            return np.searchsorted(a, keys, side);
        }

        public static ndarray diagonal(this ndarray a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            return np.diagonal(a, offset, axis1, axis2);
        }

        public static ndarray trace(this ndarray a, int offset = 0, int axis1 = 0, int axis2 = 1, dtype dtype = null, ndarray @out = null)
        {
            return np.trace(a, offset, axis1, axis2, dtype, @out);
        }


        public static ndarray trace(this ndarray a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            return np.diagonal(a, offset, axis1, axis2);
        }


        public static ndarray[] NonZero(this ndarray a)
        {
            return np.nonzero(a);
        }

        public static ndarray compress(this ndarray a, object condition, object axis = null, ndarray @out = null)
        {
            return np.compress(condition, a, axis, @out);
        }

        public static ndarray compress(this ndarray a, ndarray condition, int? axis = null, ndarray @out = null)
        {
            return np.compress(condition, a, axis, @out);
        }

        public static ndarray clip(this ndarray a, object a_min, object a_max, ndarray ret = null)
        {
            return np.clip(a, a_min, a_max, ret);
        }

        public static ndarray Sum(this ndarray a, int? axis = null, dtype dtype = null, ndarray ret = null)
        {
            return np.sum(a, axis, dtype, ret);
        }

        public static ndarray Any(this ndarray a, object axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.any(a, axis, @out, keepdims);
        }

        public static ndarray All(this ndarray a, object axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.all(a, axis, @out, keepdims);
        }

        public static ndarray cumsum(this ndarray a, int? axis = null, dtype dtype = null, ndarray ret = null)
        {
            return np.cumsum(a, axis, dtype, ret);
        }

        public static ndarray ptp(this ndarray a, object axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.ptp(a, axis, @out, keepdims);
        }

        public static ndarray AMax(this ndarray a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.amax(a, axis, @out, keepdims);
        }

        public static ndarray AMin(this ndarray a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.amin(a, axis, @out, keepdims);
        }


        public static ndarray Prod(this ndarray a, int? axis = null, dtype dtype = null, ndarray ret = null)
        {
            return np.prod(a, axis, dtype, ret);
        }

        public static ndarray CumProd(this ndarray a, int axis, dtype dtype, ndarray ret = null)
        {
            return np.cumprod(a, axis, dtype, ret);
        }
        
        public static ndarray Mean(this ndarray a, int? axis = null, dtype dtype = null)
        {
            return np.mean(a, axis, dtype);
        }

        public static ndarray Std(this ndarray a, int? axis = null, dtype dtype = null)
        {
            return np.std(a, axis, dtype);
        }


        public static List<T> ToList<T>(this ndarray a)
        {
            if (a.IsASlice)
            {
                List<T> Data = new List<T>();
                foreach (T d in a)
                {
                    Data.Add(d);
                }
                return Data;
            }
            else
            {
                T[] data = (T[])a.rawdata(0).datap;
                return data.ToList();
            }

        }

        public static T[] ToArray<T>(this ndarray a)
        {
            if (a.IsASlice)
            {
                List<T> Data = a.ToList<T>();
                return Data.ToArray();
            }
            else
            {
                T[] data = (T[])a.rawdata(0).datap;
                return data;
            }
 
        }

        public static double Mean<T>(this ndarray a)
        {
            return a.ToList<T>().Mean();
        }
        public static double Mean<T>(this IList<T> values)
        {
            return values.Count == 0 ? 0 : values.Mean(0, values.Count);
        }
        public static double Mean<T>(this IList<T> values, int start, int end)
        {
            double s = 0;

            for (int i = start; i < end; i++)
            {
                s += Convert.ToDouble(values[i]);
            }

            return s / (end - start);
        }

        public static double Variance<T>(this ndarray a)
        {
            return a.ToList<T>().Variance();
        }
        public static double Variance<T>(this IList<T> values)
        {
            return values.Variance(values.Mean(), 0, values.Count);
        }
        public static double Variance<T>(this IList<T> values, double mean)
        {
            return values.Variance(mean, 0, values.Count);
        }
        public static double Variance<T>(this IList<T> values, double mean, int start, int end)
        {
            double variance = 0;

            for (int i = start; i < end; i++)
            {
                variance += Math.Pow((Convert.ToDouble(values[i]) - mean), 2);
            }

            int n = end - start;
            if (start > 0) n -= 1;

            return variance / (n);
        }

        public static double StandardDeviation<T>(this ndarray a)
        {
            return a.ToList<T>().StandardDeviation();
        }
        public static double StandardDeviation<T>(this IList<T> values)
        {
            return values.Count == 0 ? 0 : values.StandardDeviation(0, values.Count);
        }
        public static double StandardDeviation<T>(this IList<T> values, int start, int end)
        {
            double mean = values.Mean(start, end);
            double variance = values.Variance(mean, start, end);

            return Math.Sqrt(variance);
        }

        public static sbyte[] AsSByteArray(this ndarray a)
        {
            return np.AsSByteArray(a);
        }
        public static byte[] AsByteArray(this ndarray a)
        {
            return np.AsByteArray(a);
        }
        public static Int16[] AsInt16Array(this ndarray a)
        {
            return np.AsInt16Array(a);
        }
        public static UInt16[] AsUInt16Array(this ndarray a)
        {
            return np.AsUInt16Array(a);
        }
        public static Int32[] AsInt32Array(this ndarray a)
        {
            return np.AsInt32Array(a);
        }
        public static UInt32[] AsUInt32Array(this ndarray a)
        {
            return np.AsUInt32Array(a);
        }
        public static Int64[] AsInt64Array(this ndarray a)
        {
            return np.AsInt64Array(a);
        }
        public static UInt64[] AsUInt64Array(this ndarray a)
        {
            return np.AsUInt64Array(a);
        }
        public static float[] AsFloatArray(this ndarray a)
        {
            return np.AsFloatArray(a);
        }
        public static double[] AsDoubleArray(this ndarray a)
        {
            return np.AsDoubleArray(a);
        }
        public static decimal[] AsDecimalArray(this ndarray a)
        {
            return np.AsDecimalArray(a);
        }
    }

    public static partial class np
    {
        #region as(.NET System.Array)

        public static sbyte[] AsSByteArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.Dtype.TypeNum != NPY_TYPES.NPY_BYTE)
            {
                a = a.astype(np.Int8);
            }

            return a.rawdata(0).datap as sbyte[];
        }
        public static byte[] AsByteArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.Dtype.TypeNum != NPY_TYPES.NPY_UBYTE)
            {
                a = a.astype(np.UInt8);
            }

            return a.rawdata(0).datap as byte[];
        }
        public static Int16[] AsInt16Array(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.Dtype.TypeNum != NPY_TYPES.NPY_INT16)
            {
                a = a.astype(np.Int16);
            }

            return a.rawdata(0).datap as Int16[];
        }
        public static UInt16[] AsUInt16Array(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.Dtype.TypeNum != NPY_TYPES.NPY_UINT16)
            {
                a = a.astype(np.UInt16);
            }

            return a.rawdata(0).datap as UInt16[];
        }
        public static Int32[] AsInt32Array(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.Dtype.TypeNum != NPY_TYPES.NPY_INT32)
            {
                a = a.astype(np.Int32);
            }

            return a.rawdata(0).datap as Int32[];
        }
        public static UInt32[] AsUInt32Array(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.Dtype.TypeNum != NPY_TYPES.NPY_UINT32)
            {
                a = a.astype(np.UInt32);
            }

            return a.rawdata(0).datap as UInt32[];
        }
        public static Int64[] AsInt64Array(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.Dtype.TypeNum != NPY_TYPES.NPY_INT64)
            {
                a = a.astype(np.Int64);
            }

            return a.rawdata(0).datap as Int64[];
        }
        public static UInt64[] AsUInt64Array(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.Dtype.TypeNum != NPY_TYPES.NPY_UINT64)
            {
                a = a.astype(np.UInt32);
            }

            return a.rawdata(0).datap as UInt64[];
        }
        public static float[] AsFloatArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.Dtype.TypeNum != NPY_TYPES.NPY_FLOAT)
            {
                a = a.astype(np.Float32);
            }

            return a.rawdata(0).datap as float[];
        }
        public static double[] AsDoubleArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.Dtype.TypeNum != NPY_TYPES.NPY_DOUBLE)
            {
                a = a.astype(np.Float64);
            }

            return a.rawdata(0).datap as double[];
        }
        public static decimal[] AsDecimalArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.Dtype.TypeNum != NPY_TYPES.NPY_DECIMAL)
            {
                a = a.astype(np.Float64);
            }

            return a.rawdata(0).datap as decimal[];
        }

        private static ndarray ConvertToFlattenedArray(object input)
        {
            ndarray arr = null;

            try
            {
                arr = asanyarray(input);
            }
            catch (Exception ex)
            {
                throw new ValueError("Unable to convert input into an ndarray.");
            }


            if (arr.IsASlice)
            {
                arr = arr.flatten();
            }

            return arr;
        }
        #endregion
    }

}
