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
        /// <summary>
        /// Returns an array containing the same data with a new shape.
        /// </summary>
        /// <param name="a">array to reshape</param>
        /// <param name="newshape">New shape for the array. The new shape should be compatible with the original shape</param>
        /// <returns></returns>
        public static ndarray reshape(this ndarray a, params int[] newshape)
        {
            npy_intp[] newdims = new npy_intp[newshape.Length];
            for (int i = 0; i < newshape.Length; i++)
                newdims[i] = newshape[i];
            return a.reshape(newdims, NPY_ORDER.NPY_ANYORDER);
        }
        /// <summary>
        /// Returns an array containing the same data with a new shape.
        /// </summary>
        /// <param name="a">array to reshape</param>
        /// <param name="newshape">New shape for the array. The new shape should be compatible with the original shape</param>
        /// <returns></returns>
        public static ndarray reshape(this ndarray a, params long[] newshape)
        {
            npy_intp[] newdims = new npy_intp[newshape.Length];
            for (int i = 0; i < newshape.Length; i++)
                newdims[i] = (npy_intp)newshape[i];
            return a.reshape(newdims, NPY_ORDER.NPY_ANYORDER);
        }
        /// <summary>
        /// Returns an array containing the same data with a new shape.
        /// </summary>
        /// <param name="a">array to reshape</param>
        /// <param name="newshape">New shape for the array. The new shape should be compatible with the original shape</param>
        /// <param name="order">{‘C’, ‘F’, ‘A’}, optional</param>
        /// <returns></returns>
        public static ndarray reshape(this ndarray a, shape newshape, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            return np.reshape(a, newshape, order);
        }
        /// <summary>
        /// Returns an array containing the same data with a new shape.
        /// </summary>
        /// <param name="a">array to reshape</param>
        /// <param name="newshape">New shape for the array. The new shape should be compatible with the original shape</param>
        /// <param name="order">{‘C’, ‘F’, ‘A’}, optional</param>
        /// <returns></returns>
        public static ndarray reshape(this ndarray a, object oshape, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            shape newshape = ConvertTupleToShape(oshape);
            if (newshape == null)
            {
                throw new Exception("Unable to convert shape object");
            }
            
            return np.reshape(a, newshape, order);
        }

        internal static shape ConvertTupleToShape(object oshape)
        {
            var T = oshape.GetType();

            if (oshape is shape)
            {
                return oshape as shape;
            }

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
                return new shape(new long[] { T5.Item1, T5.Item2, T5.Item3, T5.Item4, T5.Item5 });
            }

            if (oshape is ValueTuple<int, int, int, int, int, int>)
            {
                ValueTuple<int, int, int, int, int, int> T6 = (ValueTuple<int, int, int, int, int, int>)oshape;
                return new shape(new npy_intp[] { T6.Item1, T6.Item2, T6.Item3, T6.Item4, T6.Item5, T6.Item6 });
            }
            if (oshape is ValueTuple<long, long, long, long, long, long>)
            {
                ValueTuple<long, long, long, long, long, long> T6 = (ValueTuple<long, long, long, long, long, long>)oshape;
                return new shape(new long[] { T6.Item1, T6.Item2, T6.Item3, T6.Item4, T6.Item5, T6.Item6 });
            }

            if (oshape is ValueTuple<int, int, int, int, int, int, int>)
            {
                ValueTuple<int, int, int, int, int, int, int> T7 = (ValueTuple<int, int, int, int, int, int, int>)oshape;
                return new shape(new npy_intp[] { T7.Item1, T7.Item2, T7.Item3, T7.Item4, T7.Item5, T7.Item6, T7.Item7 });
            }
            if (oshape is ValueTuple<long, long, long, long, long, long, long>)
            {
                ValueTuple<long, long, long, long, long, long, long> T7 = (ValueTuple<long, long, long, long, long, long, long>)oshape;
                return new shape(new long[] { T7.Item1, T7.Item2, T7.Item3, T7.Item4, T7.Item5, T7.Item6, T7.Item7 });
            }

            if (oshape is ValueTuple<int, int, int, int, int, int, int, int>)
            {
                ValueTuple<int, int, int, int, int, int, int, int> T8 = (ValueTuple<int, int, int, int, int, int, int, int>)oshape;
                return new shape(new npy_intp[] { T8.Item1, T8.Item2, T8.Item3, T8.Item4, T8.Item5, T8.Item6, T8.Item7, T8.Rest });
            }
            if (oshape is ValueTuple<long, long, long, long, long, long, long, long>)
            {
                ValueTuple<long, long, long, long, long, long, long, long> T8 = (ValueTuple<long, long, long, long, long, long, long, long>)oshape;
                return new shape(new long[] { T8.Item1, T8.Item2, T8.Item3, T8.Item4, T8.Item5, T8.Item6, T8.Item7, T8.Rest });
            }
            return null;
        }

        /// <summary>
        /// Write array to a file as text or binary (default).
        /// </summary>
        /// <param name="a">array to write to file</param>
        /// <param name="fileName">string containing a filename</param>
        /// <param name="sep">Separator between array items for text output.</param>
        /// <param name="format">Format string for text file output.</param>
        public static void tofile(this ndarray a, string fileName, string sep = null, string format = null)
        {
            np.tofile(a, fileName, sep, format);
        }
        /// <summary>
        /// Write array to a stream as text or binary (default).
        /// </summary>
        /// <param name="a">array to write to stream</param>
        /// <param name="stream">stream to write to</param>
        /// <param name="sep">Separator between array items for text output.</param>
        /// <param name="format">Format string for text file output.</param>
        public static void tofile(this ndarray a, Stream stream, string sep = null, string format = null)
        {
            np.tofile(a, stream, sep, format);
        }

        /// <summary>
        /// New view of array with the same data.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="dtype">Data-type descriptor of the returned view</param>
        /// <param name="type">Type of the returned view, e.g., ndarray or matrix (not used)</param>
        /// <returns></returns>
        public static ndarray view(this ndarray a, dtype dtype = null, object type = null)
        {
            return np.view(a, dtype, type);
        }
        /// <summary>
        /// Return a copy of the array collapsed into one dimension.
        /// </summary>
        /// <param name="a">array to flatten</param>
        /// <param name="order">{‘C’, ‘F’, ‘A’, ‘K’}</param>
        /// <returns></returns>
        public static ndarray Flatten(this ndarray a, NPY_ORDER order)
        {
            return NpyCoreApi.Flatten(a, order);
        }
        /// <summary>
        /// Return a contiguous flattened array.
        /// </summary>
        /// <param name="a">array to flatten</param>
        /// <param name="order">{‘C’, ‘F’, ‘A’, ‘K’}</param>
        /// <returns></returns>
        public static ndarray Ravel(this ndarray a, NPY_ORDER order)
        {
            return np.ravel(a, order);
        }
        /// <summary>
        /// Change shape and size of array in-place.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="newdims">Shape of resized array.</param>
        /// <param name="order"></param>
        public static void Resize(this ndarray a, npy_intp[] newdims)
        {
            np.resize(a, newdims);
        }
        /// <summary>
        /// Remove axes of length one from a.
        /// </summary>
        /// <param name="a">array to squeeze</param>
        /// <returns></returns>
        public static ndarray Squeeze(this ndarray a)
        {
            return np.squeeze(a);
        }
        /// <summary>
        /// Interchange two axes of an array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis1">First axis.</param>
        /// <param name="axis2">Second axis.</param>
        /// <returns></returns>
        public static ndarray SwapAxes(this ndarray a, int axis1, int axis2)
        {
            return np.swapaxes(a, axis1, axis2);
        }
        /// <summary>
        /// Returns a view of the array with axes transposed.
        /// </summary>
        /// <param name="a">array to transpose</param>
        /// <param name="axes">array of npy_intp: i in the j-th place in the array means a’s i-th axis becomes a.transpose()’s j-th axis</param>
        /// <returns></returns>
        public static ndarray Transpose(this ndarray a, npy_intp[] axes = null)
        {
            return np.transpose(a, axes);
        }
        /// <summary>
        /// Construct an array from an index array and a set of arrays to choose from.
        /// </summary>
        /// <param name="a">array to perform choose operation on.</param>
        /// <param name="choices">Choice arrays. a and all of the choices must be broadcastable to the same shape</param>
        /// <param name="out">f provided, the result will be inserted into this array</param>
        /// <param name="clipMode">{‘raise’ (default), ‘wrap’, ‘clip’}</param>
        /// <returns></returns>
        public static ndarray Choose(this ndarray a, IEnumerable<ndarray> choices, ndarray @out = null, NPY_CLIPMODE clipMode = NPY_CLIPMODE.NPY_RAISE)
        {
            return np.choose(a, choices, @out, clipMode);
        }

        /// <summary>
        /// Repeat elements of an array.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="repeats">The number of repetitions for each element</param>
        /// <param name="axis">The axis along which to repeat values</param>
        /// <returns></returns>
        public static ndarray Repeat(this ndarray a, object repeats, int? axis)
        {
            return np.repeat(a, repeats, axis);
        }
        /// <summary>
        /// Replaces specified elements of an array with given values.
        /// </summary>
        /// <param name="a">target ndarray</param>
        /// <param name="values">Values to place in a at target indices</param>
        /// <param name="ind">Target indices, interpreted as integers.</param>
        /// <param name="mode">{‘raise’, ‘wrap’, ‘clip’}</param>
        public static void PutTo(this ndarray a, ndarray values, ndarray ind, NPY_CLIPMODE mode)
        {
            int ret = np.put(a, ind, values, mode);
        }
        /// <summary>
        /// Sort an array in-place.
        /// </summary>
        /// <param name="a">array to sort</param>
        /// <param name="axis">Axis along which to sort. Default is -1, which means sort along the last axis.</param>
        /// <param name="sortkind">Sorting algorithm. The default is ‘quicksort’.</param>
        /// <returns></returns>
        public static ndarray Sort(this ndarray a, int? axis = -1, NPY_SORTKIND sortkind = NPY_SORTKIND.NPY_QUICKSORT)
        {
            return np.sort(a, axis, sortkind, null);
        }

        /// <summary>
        /// Returns the indices that would sort an array.
        /// </summary>
        /// <param name="a">array to sort</param>
        /// <param name="axis">Axis along which to sort.</param>
        /// <param name="kind">Sorting algorithm. The default is ‘quicksort’.</param>
        /// <returns></returns>
        public static ndarray ArgSort(this ndarray a, int? axis =-1, NPY_SORTKIND kind = NPY_SORTKIND.NPY_QUICKSORT)
        {
            return np.argsort(a, axis, kind);
        }

        /// <summary>
        /// Returns the indices of the maximum values along an axis.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <param name="axis">By default, the index is into the flattened array, otherwise along the specified axis.</param>
        /// <param name="ret">If provided, the result will be inserted into this array</param>
        /// <returns></returns>
        public static ndarray ArgMax(this ndarray a, int? axis = -1, ndarray ret = null)
        {
            return np.argmax(a, axis, ret);
        }

        /// <summary>
        /// Returns the indices of the minimum values along an axis.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <param name="axis">By default, the index is into the flattened array, otherwise along the specified axis.</param>
        /// <param name="ret">If provided, the result will be inserted into this array</param>
        /// <returns></returns>
        public static ndarray ArgMin(this ndarray a, int? axis = -1, ndarray ret = null)
        {
            return np.argmin(a, axis, ret);
        }

        /// <summary>
        /// Find indices where elements of v should be inserted in a to maintain order.
        /// </summary>
        /// <param name="a">Input array 1-D</param>
        /// <param name="v">Values to insert into a.</param>
        /// <param name="side">{‘left’, ‘right’}</param>
        /// <returns></returns>
        public static ndarray SearchSorted(this ndarray a, ndarray v, NPY_SEARCHSIDE side = NPY_SEARCHSIDE.NPY_SEARCHLEFT)
        {
            return np.searchsorted(a, v, side);
        }
        /// <summary>
        /// Return specified diagonals.
        /// </summary>
        /// <param name="a">Array from which the diagonals are taken.</param>
        /// <param name="offset">Offset of the diagonal from the main diagonal</param>
        /// <param name="axis1">Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken</param>
        /// <param name="axis2">Axis to be used as the second axis of the 2-D sub-arrays from which the diagonals should be taken.</param>
        /// <returns></returns>
        public static ndarray diagonal(this ndarray a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            return np.diagonal(a, offset, axis1, axis2);
        }

        /// <summary>
        /// Return the sum along diagonals of the array.
        /// </summary>
        /// <param name="a">Input array, from which the diagonals are taken.</param>
        /// <param name="offset">Offset of the diagonal from the main diagonal</param>
        /// <param name="axis1">Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken</param>
        /// <param name="axis2">Axis to be used as the second axis of the 2-D sub-arrays from which the diagonals should be taken.</param>
        /// <param name="dtype">Determines the data-type of the returned array and of the accumulator where the elements are summed.</param>
        /// <param name="ret">Array into which the output is placed</param>
        /// <returns></returns>
        public static ndarray trace(this ndarray a, int offset = 0, int axis1 = 0, int axis2 = 1, dtype dtype = null, ndarray ret = null)
        {
            return np.trace(a, offset, axis1, axis2, dtype, ret);
        }

        /// <summary>
        /// Return the indices of the elements that are non-zero.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns></returns>
        public static ndarray[] NonZero(this ndarray a)
        {
            return np.nonzero(a);
        }

        /// <summary>
        /// Return selected slices of an array along given axis.
        /// </summary>
        /// <param name="a">Array from which to extract a part.</param>
        /// <param name="condition">Array that selects which entries to return. If len(condition) is less than the size of a along the given axis, then output is truncated to the length of the condition array.</param>
        /// <param name="axis">Axis along which to take slices. If None (default), work on the flattened array.</param>
        /// <param name="out">Output array. Its type is preserved and it must be of the right shape to hold the output.</param>
        /// <returns></returns>
        public static ndarray compress(this ndarray a, object condition, object axis = null, ndarray @out = null)
        {
            return np.compress(condition, a, axis, @out);
        }

        /// <summary>
        /// Return selected slices of an array along given axis.
        /// </summary>
        /// <param name="a">Array from which to extract a part.</param>
        /// <param name="condition">Array that selects which entries to return. If len(condition) is less than the size of a along the given axis, then output is truncated to the length of the condition array.</param>
        /// <param name="axis">Axis along which to take slices. If None (default), work on the flattened array.</param>
        /// <param name="out">Output array. Its type is preserved and it must be of the right shape to hold the output.</param>
        /// <returns></returns>
        public static ndarray compress(this ndarray a, ndarray condition, int? axis = null, ndarray @out = null)
        {
            return np.compress(condition, a, axis, @out);
        }
        /// <summary>
        /// Clip (limit) the values in an array.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_min">Minimum value</param>
        /// <param name="a_max">maximum value</param>
        /// <param name="ret">The results will be placed in this array</param>
        /// <returns></returns>
        public static ndarray clip(this ndarray a, object a_min, object a_max, ndarray ret = null)
        {
            return np.clip(a, a_min, a_max, ret);
        }

        /// <summary>
        /// Sum of array elements over a given axis.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis or axes along which a sum is performed.</param>
        /// <param name="dtype">The type of the returned array and of the accumulator in which the elements are summed. </param>
        /// <param name="ret">Alternative output array in which to place the result.</param>
        /// <returns></returns>
        public static ndarray Sum(this ndarray a, int? axis = null, dtype dtype = null, ndarray ret = null)
        {
            return np.sum(a, axis, dtype, ret);
        }

        /// <summary>
        /// Test whether any array element along a given axis evaluates to True.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <param name="axis">Axis or axes along which a logical OR reduction is performed</param>
        /// <param name="out">Alternate output array in which to place the result</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one</param>
        /// <returns></returns>
        public static ndarray Any(this ndarray a, object axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.any(a, axis, @out, keepdims);
        }

        /// <summary>
        /// return bool result from np.any
        /// </summary>
        /// <param name="a">Input array</param>
        /// <param name="axis">Axis or axes along which a logical OR reduction is performed</param>
        /// <param name="out">Alternate output array in which to place the result</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one</param>
        /// <returns></returns>
        public static bool Anyb(this ndarray a, object axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.anyb(a, axis, @out, keepdims);
        }

        /// <summary>
        /// Test whether all array elements along a given axis evaluate to True.
        /// </summary>
        /// <param name="a">Input array </param>
        /// <param name="axis">Axis or axes along which a logical AND reduction is performed</param>
        /// <param name="out">Alternate output array in which to place the result</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one</param>
        /// <returns></returns>
        public static ndarray All(this ndarray a, object axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.all(a, axis, @out, keepdims);
        }

        /// <summary>
        /// Return the cumulative sum of the elements along a given axis.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">Axis along which the cumulative sum is computed. </param>
        /// <param name="dtype">Type of the returned array and of the accumulator in which the elements are summed.</param>
        /// <param name="ret">Alternative output array in which to place the result.</param>
        /// <returns></returns>
        public static ndarray cumsum(this ndarray a, int? axis = null, dtype dtype = null, ndarray ret = null)
        {
            return np.cumsum(a, axis, dtype, ret);
        }
        /// <summary>
        /// Range of values (maximum - minimum) along an axis.
        /// </summary>
        /// <param name="a">Input values.</param>
        /// <param name="axis">Axis along which to find the peaks.</param>
        /// <param name="out">Alternative output array in which to place the result.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray ptp(this ndarray a, object axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.ptp(a, axis, @out, keepdims);
        }
        /// <summary>
        /// Return the maximum of an array or maximum along an axis.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis or axes along which to operate.</param>
        /// <param name="out">Alternative output array in which to place the result.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray AMax(this ndarray a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.amax(a, axis, @out, keepdims);
        }
        /// <summary>
        /// Return the minimum of an array or minimum along an axis.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis or axes along which to operate.</param>
        /// <param name="out">Alternative output array in which to place the result.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray AMin(this ndarray a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.amin(a, axis, @out, keepdims);
        }

        /// <summary>
        /// Return the product of array elements over a given axis.
        /// </summary>
        /// <param name="a">Input data</param>
        /// <param name="axis">Axis or axes along which a product is performed. </param>
        /// <param name="dtype">The type of the returned array, as well as of the accumulator in which the elements are multiplied.</param>
        /// <param name="ret">Alternative output array in which to place the result.</param>
        /// <returns></returns>
        public static ndarray Prod(this ndarray a, int? axis = null, dtype dtype = null, ndarray ret = null)
        {
            return np.prod(a, axis, dtype, ret);
        }
        /// <summary>
        /// Return the cumulative product of elements along a given axis.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <param name="axis">Axis along which the cumulative product is computed.</param>
        /// <param name="dtype">Type of the returned array, as well as of the accumulator in which the elements are multiplied.</param>
        /// <param name="ret">Alternative output array in which to place the result</param>
        /// <returns></returns>
        public static ndarray CumProd(this ndarray a, int axis, dtype dtype, ndarray ret = null)
        {
            return np.cumprod(a, axis, dtype, ret);
        }
        /// <summary>
        /// Compute the arithmetic mean along the specified axis.
        /// </summary>
        /// <param name="a">Array containing numbers whose mean is desired.</param>
        /// <param name="axis">Axis or axes along which the means are computed.</param>
        /// <param name="dtype">Type to use in computing the mean. </param>
        /// <returns></returns>
        public static ndarray Mean(this ndarray a, int? axis = null, dtype dtype = null)
        {
            return np.mean(a, axis, dtype);
        }
        /// <summary>
        /// Compute the standard deviation along the specified axis.
        /// </summary>
        /// <param name="a">Calculate the standard deviation of these values.</param>
        /// <param name="axis">Axis or axes along which the standard deviation is computed.</param>
        /// <param name="dtype">Type to use in computing the standard deviation.</param>
        /// <returns></returns>
        public static ndarray Std(this ndarray a, int? axis = null, dtype dtype = null)
        {
            return np.std(a, axis, dtype);
        }
        /// <summary>
        /// Return a partitioned copy of an array.
        /// </summary>
        /// <param name="a">Array to be sorted.</param>
        /// <param name="kth">Element index to partition by</param>
        /// <param name="axis">Axis along which to sort.</param>
        /// <param name="kind">Selection algorithm</param>
        /// <param name="order">When a is an array with fields defined, this argument specifies which fields to compare first, second, etc.</param>
        /// <returns></returns>
        public static ndarray partition(this ndarray a, npy_intp[] kth, int? axis = null, string kind = "introselect", IEnumerable<string> order = null)
        {
            return np.partition(a, kth, axis, kind, order);
        }

        /// <summary>
        /// Converts ndarray data items into a .NET List<>/>
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
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


        /// <summary>
        /// Converts ndarray data items into a raw data array
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
        public static System.Array ToArray(this ndarray a)
        {
            ndarray b = a.ravel();
            return b.ToSystemArray();
        }
        /// <summary>
        /// Converts ndarray data items into a raw data array
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static System.Array ToSystemArray(this ndarray a)
        {
            switch (a.TypeNum)
            {
                case NPY_TYPES.NPY_BOOL:
                    return ConvertToMultiDimArray<bool>(a);

                case NPY_TYPES.NPY_BYTE:
                    return ConvertToMultiDimArray<sbyte>(a);

                case NPY_TYPES.NPY_UBYTE:
                    return ConvertToMultiDimArray<byte>(a);

                case NPY_TYPES.NPY_INT16:
                    return ConvertToMultiDimArray<Int16>(a);

                case NPY_TYPES.NPY_UINT16:
                    return ConvertToMultiDimArray<UInt16>(a);

                case NPY_TYPES.NPY_INT32:
                    return ConvertToMultiDimArray<Int32>(a);

                case NPY_TYPES.NPY_UINT32:
                    return ConvertToMultiDimArray<UInt32>(a);

                case NPY_TYPES.NPY_INT64:
                    return ConvertToMultiDimArray<Int64>(a);

                case NPY_TYPES.NPY_UINT64:
                    return ConvertToMultiDimArray<UInt64>(a);

                case NPY_TYPES.NPY_FLOAT:
                    return ConvertToMultiDimArray<float>(a);

                case NPY_TYPES.NPY_DOUBLE:
                    return ConvertToMultiDimArray<double>(a);

                case NPY_TYPES.NPY_DECIMAL:
                    return ConvertToMultiDimArray<decimal>(a);

                case NPY_TYPES.NPY_COMPLEX:
                    return ConvertToMultiDimArray<System.Numerics.Complex>(a);

                case NPY_TYPES.NPY_BIGINT:
                    return ConvertToMultiDimArray<System.Numerics.BigInteger>(a);

                case NPY_TYPES.NPY_OBJECT:
                    return ConvertToMultiDimArray<System.Object>(a);

                case NPY_TYPES.NPY_STRING:
                    return ConvertToMultiDimArray<System.String>(a);

                default:
                    throw new Exception("unable to convert ndarray of this type to MultiDim .NET array");
            }

        }

        private static Array ConvertToMultiDimArray<T>(ndarray a)
        {
            Array array = Array.CreateInstance(typeof(T), a.dims);
            npy_intp[] indexes = new npy_intp[array.Rank];

            int count = 0;
            while (true)
            {
                array.SetValue((T)a.item_byindex(indexes), indexes);
                count++;

                for (int i = array.Rank - 1; i >= 0; i--)
                {
                    if (indexes[i] < array.GetLength(i) - 1)
                    {
                        indexes[i]++;
                        break;
                    }
                    else
                    {
                        indexes[i] = 0;
                        if (i == 0)
                        {
                            return array;
                        }
                    }
                }
            }
        }

#if Not_used // replaced by ConvertToMultiDimArray above
        private static Array ConvertToMultiDimArrayx<T>(ndarray a)
        {
            if (a.ndim == 1)
            {
                return ConvertTo1dArray<T>(a);
            }
            if (a.ndim == 2)
            {
                return ConvertTo2dArray<T>(a);
            }
            if (a.ndim == 3)
            {
                return ConvertTo3dArray<T>(a);
            }
            if (a.ndim == 4)
            {
                return ConvertTo4dArray<T>(a);
            }
            if (a.ndim == 5)
            {
                return ConvertTo5dArray<T>(a);
            }
            if (a.ndim == 6)
            {
                return ConvertTo6dArray<T>(a);
            }
            if (a.ndim == 7)
            {
                return ConvertTo7dArray<T>(a);
            }
            if (a.ndim == 8)
            {
                return ConvertTo8dArray<T>(a);
            }
            if (a.ndim == 9)
            {
                return ConvertTo9dArray<T>(a);
            }
            if (a.ndim == 10)
            {
                return ConvertTo10dArray<T>(a);
            }
            if (a.ndim == 11)
            {
                return ConvertTo11dArray<T>(a);
            }
            if (a.ndim == 12)
            {
                return ConvertTo12dArray<T>(a);
            }
            if (a.ndim == 13)
            {
                return ConvertTo13dArray<T>(a);
            }
            if (a.ndim == 14)
            {
                return ConvertTo14dArray<T>(a);
            }
            if (a.ndim == 15)
            {
                return ConvertTo15dArray<T>(a);
            }
            if (a.ndim == 16)
            {
                return ConvertTo16dArray<T>(a);
            }
            if (a.ndim == 17)
            {
                return ConvertTo17dArray<T>(a);
            }
            if (a.ndim == 18)
            {
                return ConvertTo18dArray<T>(a);
            }

            throw new Exception(string.Format("Can't convert {0}D array.  Max dims supported is 18", a.ndim));
        }
        private static System.Array ConvertTo1dArray<T>(ndarray nd)
        {
            if (!nd.IsASlice)
            {
                System.Array data = (System.Array)nd.rawdata(0).datap;
                return (System.Array)data;
            }

            T[] output = new T[nd.dims[0]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                output[i] = (T)nd[i];
            }

            return output;
        }
        private static System.Array ConvertTo2dArray<T>(ndarray nd)
        {
            T[,] output = new T[nd.dims[0], nd.dims[1]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    output[i, j] = (T)nd[i, j];
                }
            }

            return output;
        }
        private static System.Array ConvertTo3dArray<T>(ndarray nd)
        {
            T[,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        output[i, j, k] = (T)nd[i, j, k];
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo4dArray<T>(ndarray nd)
        {
            T[,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            output[i, j, k, l] = (T)nd[i, j, k, l];
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo5dArray<T>(ndarray nd)
        {
            T[,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                output[i, j, k, l, m] = (T)nd[i, j, k, l, m];
                            }
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo6dArray<T>(ndarray nd)
        {
            T[,,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4], nd.dims[5]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                for (int n = 0; n < nd.dims[5]; n++)
                                {
                                    output[i, j, k, l, m, n] = (T)nd[i, j, k, l, m, n];
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo7dArray<T>(ndarray nd)
        {
            T[,,,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4], nd.dims[5], nd.dims[6]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                for (int n = 0; n < nd.dims[5]; n++)
                                {
                                    for (int o = 0; o < nd.dims[6]; o++)
                                    {
                                        output[i, j, k, l, m, n,o] = (T)nd[i, j, k, l, m, n, o];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo8dArray<T>(ndarray nd)
        {
            T[,,,,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4], nd.dims[5], nd.dims[6], nd.dims[7]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                for (int n = 0; n < nd.dims[5]; n++)
                                {
                                    for (int o = 0; o < nd.dims[6]; o++)
                                    {
                                        for (int p = 0; p < nd.dims[7]; p++)
                                        {
                                            output[i, j, k, l, m, n, o, p] = (T)nd[i, j, k, l, m, n, o, p];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo9dArray<T>(ndarray nd)
        {
            T[,,,,,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4], nd.dims[5], nd.dims[6], nd.dims[7], nd.dims[8]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                for (int n = 0; n < nd.dims[5]; n++)
                                {
                                    for (int o = 0; o < nd.dims[6]; o++)
                                    {
                                        for (int p = 0; p < nd.dims[7]; p++)
                                        {
                                            for (int q = 0; q < nd.dims[8]; q++)
                                            {
                                                output[i, j, k, l, m, n, o, p, q] = (T)nd[i, j, k, l, m, n, o, p, q];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo10dArray<T>(ndarray nd)
        {
            T[,,,,,,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4], nd.dims[5], nd.dims[6], nd.dims[7], nd.dims[8], nd.dims[9]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                for (int n = 0; n < nd.dims[5]; n++)
                                {
                                    for (int o = 0; o < nd.dims[6]; o++)
                                    {
                                        for (int p = 0; p < nd.dims[7]; p++)
                                        {
                                            for (int q = 0; q < nd.dims[8]; q++)
                                            {
                                                for (int r = 0; r < nd.dims[9]; r++)
                                                {
                                                    output[i, j, k, l, m, n, o, p, q,r] = (T)nd[i, j, k, l, m, n, o, p, q, r];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo11dArray<T>(ndarray nd)
        {
            T[,,,,,,,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4], nd.dims[5], nd.dims[6], nd.dims[7], nd.dims[8], nd.dims[9], nd.dims[10]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                for (int n = 0; n < nd.dims[5]; n++)
                                {
                                    for (int o = 0; o < nd.dims[6]; o++)
                                    {
                                        for (int p = 0; p < nd.dims[7]; p++)
                                        {
                                            for (int q = 0; q < nd.dims[8]; q++)
                                            {
                                                for (int r = 0; r < nd.dims[9]; r++)
                                                {
                                                    for (int s = 0; s < nd.dims[10]; s++)
                                                    {
                                                        output[i, j, k, l, m, n, o, p, q, r, s] = (T)nd[i, j, k, l, m, n, o, p, q, r, s];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo12dArray<T>(ndarray nd)
        {
            T[,,,,,,,,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4], nd.dims[5], nd.dims[6], nd.dims[7], nd.dims[8], nd.dims[9], nd.dims[10], nd.dims[11]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                for (int n = 0; n < nd.dims[5]; n++)
                                {
                                    for (int o = 0; o < nd.dims[6]; o++)
                                    {
                                        for (int p = 0; p < nd.dims[7]; p++)
                                        {
                                            for (int q = 0; q < nd.dims[8]; q++)
                                            {
                                                for (int r = 0; r < nd.dims[9]; r++)
                                                {
                                                    for (int s = 0; s < nd.dims[10]; s++)
                                                    {
                                                        for (int t = 0; t < nd.dims[11]; t++)
                                                        {
                                                            output[i, j, k, l, m, n, o, p, q, r, s, t] = (T)nd[i, j, k, l, m, n, o, p, q, r, s, t];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo13dArray<T>(ndarray nd)
        {
            T[,,,,,,,,,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4], nd.dims[5], nd.dims[6], nd.dims[7], nd.dims[8], nd.dims[9], nd.dims[10], nd.dims[11], nd.dims[12]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                for (int n = 0; n < nd.dims[5]; n++)
                                {
                                    for (int o = 0; o < nd.dims[6]; o++)
                                    {
                                        for (int p = 0; p < nd.dims[7]; p++)
                                        {
                                            for (int q = 0; q < nd.dims[8]; q++)
                                            {
                                                for (int r = 0; r < nd.dims[9]; r++)
                                                {
                                                    for (int s = 0; s < nd.dims[10]; s++)
                                                    {
                                                        for (int t = 0; t < nd.dims[11]; t++)
                                                        {
                                                            for (int u = 0; u < nd.dims[12]; u++)
                                                            {
                                                                output[i, j, k, l, m, n, o, p, q, r, s, t, u] = (T)nd[i, j, k, l, m, n, o, p, q, r, s, t, u];
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo14dArray<T>(ndarray nd)
        {
            T[,,,,,,,,,,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4], nd.dims[5], nd.dims[6], nd.dims[7], nd.dims[8], nd.dims[9], nd.dims[10], nd.dims[11], nd.dims[12], nd.dims[13]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                for (int n = 0; n < nd.dims[5]; n++)
                                {
                                    for (int o = 0; o < nd.dims[6]; o++)
                                    {
                                        for (int p = 0; p < nd.dims[7]; p++)
                                        {
                                            for (int q = 0; q < nd.dims[8]; q++)
                                            {
                                                for (int r = 0; r < nd.dims[9]; r++)
                                                {
                                                    for (int s = 0; s < nd.dims[10]; s++)
                                                    {
                                                        for (int t = 0; t < nd.dims[11]; t++)
                                                        {
                                                            for (int u = 0; u < nd.dims[12]; u++)
                                                            {
                                                                for (int v = 0; v < nd.dims[13]; v++)
                                                                {
                                                                   output[i, j, k, l, m, n, o, p, q, r, s, t, u, v] = (T)nd[i, j, k, l, m, n, o, p, q, r, s, t, u, v];
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo15dArray<T>(ndarray nd)
        {
            T[,,,,,,,,,,,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4], nd.dims[5], nd.dims[6], nd.dims[7], nd.dims[8], nd.dims[9], nd.dims[10], nd.dims[11], nd.dims[12], nd.dims[13], nd.dims[14]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                for (int n = 0; n < nd.dims[5]; n++)
                                {
                                    for (int o = 0; o < nd.dims[6]; o++)
                                    {
                                        for (int p = 0; p < nd.dims[7]; p++)
                                        {
                                            for (int q = 0; q < nd.dims[8]; q++)
                                            {
                                                for (int r = 0; r < nd.dims[9]; r++)
                                                {
                                                    for (int s = 0; s < nd.dims[10]; s++)
                                                    {
                                                        for (int t = 0; t < nd.dims[11]; t++)
                                                        {
                                                            for (int u = 0; u < nd.dims[12]; u++)
                                                            {
                                                                for (int v = 0; v < nd.dims[13]; v++)
                                                                {
                                                                    for (int w = 0; w < nd.dims[14]; w++)
                                                                    {
                                                                        output[i, j, k, l, m, n, o, p, q, r, s, t, u, v, w] = (T)nd[i, j, k, l, m, n, o, p, q, r, s, t, u, v, w];
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo16dArray<T>(ndarray nd)
        {
            T[,,,,,,,,,,,,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4], nd.dims[5], nd.dims[6], nd.dims[7], nd.dims[8], nd.dims[9], nd.dims[10], nd.dims[11], nd.dims[12], nd.dims[13], nd.dims[14], nd.dims[15]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                for (int n = 0; n < nd.dims[5]; n++)
                                {
                                    for (int o = 0; o < nd.dims[6]; o++)
                                    {
                                        for (int p = 0; p < nd.dims[7]; p++)
                                        {
                                            for (int q = 0; q < nd.dims[8]; q++)
                                            {
                                                for (int r = 0; r < nd.dims[9]; r++)
                                                {
                                                    for (int s = 0; s < nd.dims[10]; s++)
                                                    {
                                                        for (int t = 0; t < nd.dims[11]; t++)
                                                        {
                                                            for (int u = 0; u < nd.dims[12]; u++)
                                                            {
                                                                for (int v = 0; v < nd.dims[13]; v++)
                                                                {
                                                                    for (int w = 0; w < nd.dims[14]; w++)
                                                                    {
                                                                        for (int x = 0; x < nd.dims[15]; x++)
                                                                        {
                                                                            output[i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x] = (T)nd[i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x];
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo17dArray<T>(ndarray nd)
        {
            T[,,,,,,,,,,,,,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4], nd.dims[5], nd.dims[6], nd.dims[7], nd.dims[8], nd.dims[9], nd.dims[10], nd.dims[11], nd.dims[12], nd.dims[13], nd.dims[14], nd.dims[15], nd.dims[16]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                for (int n = 0; n < nd.dims[5]; n++)
                                {
                                    for (int o = 0; o < nd.dims[6]; o++)
                                    {
                                        for (int p = 0; p < nd.dims[7]; p++)
                                        {
                                            for (int q = 0; q < nd.dims[8]; q++)
                                            {
                                                for (int r = 0; r < nd.dims[9]; r++)
                                                {
                                                    for (int s = 0; s < nd.dims[10]; s++)
                                                    {
                                                        for (int t = 0; t < nd.dims[11]; t++)
                                                        {
                                                            for (int u = 0; u < nd.dims[12]; u++)
                                                            {
                                                                for (int v = 0; v < nd.dims[13]; v++)
                                                                {
                                                                    for (int w = 0; w < nd.dims[14]; w++)
                                                                    {
                                                                        for (int x = 0; x < nd.dims[15]; x++)
                                                                        {
                                                                            for (int y = 0; y < nd.dims[16]; y++)
                                                                            {
                                                                                output[i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y] = (T)nd[i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y];
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
        private static System.Array ConvertTo18dArray<T>(ndarray nd)
        {
            T[,,,,,,,,,,,,,,,,,] output = new T[nd.dims[0], nd.dims[1], nd.dims[2], nd.dims[3], nd.dims[4], nd.dims[5], nd.dims[6], nd.dims[7], nd.dims[8], nd.dims[9], nd.dims[10], nd.dims[11], nd.dims[12], nd.dims[13], nd.dims[14], nd.dims[15], nd.dims[16], nd.dims[17]];

            for (int i = 0; i < nd.dims[0]; i++)
            {
                for (int j = 0; j < nd.dims[1]; j++)
                {
                    for (int k = 0; k < nd.dims[2]; k++)
                    {
                        for (int l = 0; l < nd.dims[3]; l++)
                        {
                            for (int m = 0; m < nd.dims[4]; m++)
                            {
                                for (int n = 0; n < nd.dims[5]; n++)
                                {
                                    for (int o = 0; o < nd.dims[6]; o++)
                                    {
                                        for (int p = 0; p < nd.dims[7]; p++)
                                        {
                                            for (int q = 0; q < nd.dims[8]; q++)
                                            {
                                                for (int r = 0; r < nd.dims[9]; r++)
                                                {
                                                    for (int s = 0; s < nd.dims[10]; s++)
                                                    {
                                                        for (int t = 0; t < nd.dims[11]; t++)
                                                        {
                                                            for (int u = 0; u < nd.dims[12]; u++)
                                                            {
                                                                for (int v = 0; v < nd.dims[13]; v++)
                                                                {
                                                                    for (int w = 0; w < nd.dims[14]; w++)
                                                                    {
                                                                        for (int x = 0; x < nd.dims[15]; x++)
                                                                        {
                                                                            for (int y = 0; y < nd.dims[16]; y++)
                                                                            {
                                                                                for (int z = 0; z < nd.dims[17]; z++)
                                                                                {
                                                                                    output[i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z] = (T)nd[i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z];
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
#endif

        /// <summary>
        /// Compute the arithmetic mean
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
        public static double Mean<T>(this ndarray a)
        {
            return a.ToList<T>().Mean();
        }
        /// <summary>
        /// Compute the arithmetic mean
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
        public static double Mean<T>(this IList<T> values)
        {
            return values.Count == 0 ? 0 : values.Mean(0, values.Count);
        }
        /// <summary>
        /// Compute the arithmetic mean
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
        public static double Mean<T>(this IList<T> values, int start, int end)
        {
            double s = 0;

            for (int i = start; i < end; i++)
            {
                s += Convert.ToDouble(values[i]);
            }

            return s / (end - start);
        }
        /// <summary>
        /// Compute the variance
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
        public static double Variance<T>(this ndarray a)
        {
            return a.ToList<T>().Variance();
        }
        /// <summary>
        /// Compute the variance
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
        public static double Variance<T>(this IList<T> values)
        {
            return values.Variance(values.Mean(), 0, values.Count);
        }
        /// <summary>
        /// Compute the variance
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
        public static double Variance<T>(this IList<T> values, double mean)
        {
            return values.Variance(mean, 0, values.Count);
        }
        /// <summary>
        /// Compute the variance
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
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
        /// <summary>
        /// Compute the standard deviation
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
        public static double StandardDeviation<T>(this ndarray a)
        {
            return a.ToList<T>().StandardDeviation();
        }
        /// <summary>
        /// Compute the standard deviation
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
        public static double StandardDeviation<T>(this IList<T> values)
        {
            return values.Count == 0 ? 0 : values.StandardDeviation(0, values.Count);
        }
        /// <summary>
        /// Compute the standard deviation
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <returns></returns>
        public static double StandardDeviation<T>(this IList<T> values, int start, int end)
        {
            double mean = values.Mean(start, end);
            double variance = values.Variance(mean, start, end);

            return Math.Sqrt(variance);
        }

        /// <summary>
        /// Returns bool array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static bool[] AsBoolArray(this ndarray a)
        {
            return np.AsBoolArray(a);
        }
        /// <summary>
        /// Returns sbyte array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static sbyte[] AsSByteArray(this ndarray a)
        {
            return np.AsSByteArray(a);
        }
        /// <summary>
        /// Returns byte array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static byte[] AsByteArray(this ndarray a)
        {
            return np.AsByteArray(a);
        }
        /// <summary>
        /// Returns Int16 array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static Int16[] AsInt16Array(this ndarray a)
        {
            return np.AsInt16Array(a);
        }
        /// <summary>
        /// Returns UInt16 array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static UInt16[] AsUInt16Array(this ndarray a)
        {
            return np.AsUInt16Array(a);
        }
        /// <summary>
        /// Returns Int32 array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static Int32[] AsInt32Array(this ndarray a)
        {
            return np.AsInt32Array(a);
        }
        /// <summary>
        /// Returns UInt32 array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static UInt32[] AsUInt32Array(this ndarray a)
        {
            return np.AsUInt32Array(a);
        }
        /// <summary>
        /// Returns Int64 array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static Int64[] AsInt64Array(this ndarray a)
        {
            return np.AsInt64Array(a);
        }
        /// <summary>
        /// Returns UInt64 array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static UInt64[] AsUInt64Array(this ndarray a)
        {
            return np.AsUInt64Array(a);
        }
        /// <summary>
        /// Returns float array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static float[] AsFloatArray(this ndarray a)
        {
            return np.AsFloatArray(a);
        }
        /// <summary>
        /// Returns double array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static double[] AsDoubleArray(this ndarray a)
        {
            return np.AsDoubleArray(a);
        }
        /// <summary>
        /// Returns decimal array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static decimal[] AsDecimalArray(this ndarray a)
        {
            return np.AsDecimalArray(a);
        }
        /// <summary>
        /// Returns Complex array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static System.Numerics.Complex[] AsComplexArray(this ndarray a)
        {
            return np.AsComplexArray(a);
        }
        /// <summary>
        /// Returns BigInt array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static System.Numerics.BigInteger[] AsBigIntArray(this ndarray a)
        {
            return np.AsBigIntArray(a);
        }
        /// <summary>
        /// Returns Object array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static object[] AsObjectArray(this ndarray a)
        {
            return np.AsObjectArray(a);
        }
        /// <summary>
        /// Returns String array.  Converts if necessary
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static string[] AsStringArray(this ndarray a)
        {
            return np.AsStringArray(a);
        }
    }

    public partial class np
    {
        internal class ufuncbase
        {

            internal static ndarray reduce(UFuncOperation ops, object a, int axis = 0, dtype dtype = null, ndarray @out = null, bool keepdims = false)
            {
                ndarray arr = asanyarray(a);
                if (arr == null)
                {
                    throw new ValueError("unable to convert a to ndarray");
                }

                NPY_TYPES rtype = dtype != null ? dtype.TypeNum : arr.TypeNum;
                return NpyCoreApi.PerformReduceOp(arr, axis, ops, rtype, @out, keepdims);
            }

            internal static ndarray reduceat(UFuncOperation ops, object a, object indices, int axis = 0, dtype dtype = null, ndarray @out = null)
            {
                ndarray arr = asanyarray(a);
                if (arr == null)
                {
                    throw new ValueError("unable to convert a to ndarray");
                }

                ndarray indicesarr = asanyarray(indices);
                if (indicesarr == null)
                {
                    throw new ValueError("unable to convert indices to ndarray");
                }

                NPY_TYPES rtype = dtype != null ? dtype.TypeNum : arr.TypeNum;
                return NpyCoreApi.PerformReduceAtOp(arr, indicesarr, axis, ops, rtype, @out);
            }

            internal static ndarray accumulate(UFuncOperation ops, object a, int axis = 0, dtype dtype = null, ndarray @out = null)
            {
                ndarray arr = asanyarray(a);
                if (arr == null)
                {
                    throw new ValueError("unable to convert a to ndarray");
                }


                NPY_TYPES rtype = dtype != null ? dtype.TypeNum : arr.TypeNum;
                return NpyCoreApi.PerformAccumulateOp(arr, axis, ops, rtype, @out);
            }

            internal static ndarray outer(UFuncOperation ops, dtype dtype, object a, object b, ndarray @out = null)
            {

                var a1 = np.asanyarray(a);
                var b1 = np.asanyarray(b);


                List<npy_intp> destdims = new List<npy_intp>();
                foreach (var dim in a1.shape.iDims)
                    destdims.Add(dim);
                foreach (var dim in b1.shape.iDims)
                    destdims.Add(dim);



                ndarray dest = @out;
                if (dest == null)
                    dest = np.empty(new shape(destdims), dtype: dtype != null ? dtype : a1.Dtype);
                
                return NpyCoreApi.PerformOuterOp(a1, b1, dest, ops);
            }

        }

        public class ufunc 
        {
            /// <summary>
            /// Accumulate the result of applying the operator to all elements.
            /// </summary>
            /// <param name="operation">ufunc operation to perform</param>
            /// <param name="a">The array to act on.</param>
            /// <param name="axis">The axis along which to apply the accumulation; default is zero.</param>
            /// <param name="out">A location into which the result is stored.</param>
            /// <returns></returns>
            public static ndarray accumulate(UFuncOperation operation, object a, int axis = 0, ndarray @out = null)
            {
                return ufuncbase.accumulate(operation, a, axis, null, @out);
            }
            /// <summary>
            /// Reduces array’s dimension by one, by applying ufunc along one axis.
            /// </summary>
            /// <param name="operation">ufunc operation to peform</param>
            /// <param name="a">The array to act on.</param>
            /// <param name="axis">Axis or axes along which a reduction is performed. </param>
            /// <param name="out">A location into which the result is stored.</param>
            /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
            /// <returns></returns>
            public static ndarray reduce(UFuncOperation operation, object a, int axis = 0, ndarray @out = null, bool keepdims = false)
            {
                return ufuncbase.reduce(operation, a, axis, null, @out, keepdims);
            }
            /// <summary>
            /// Performs a (local) reduce with specified slices over a single axis.
            /// </summary>
            /// <param name="operation">ufunc operation to peform</param>
            /// <param name="a">The array to act on.</param>
            /// <param name="indices">Paired indices, comma separated (not colon), specifying slices to reduce</param>
            /// <param name="axis">A location into which the result is stored..</param>
            /// <param name="out">A location into which the result is stored. </param>
            /// <returns></returns>
            public static ndarray reduceat(UFuncOperation operation, object a, object indices, int axis = 0, ndarray @out = null)
            {
                return ufuncbase.reduceat(operation, a, indices, axis, null, @out);
            }
            /// <summary>
            /// Apply the ufunc op to all pairs (a, b) with a in A and b in B.
            /// </summary>
            /// <param name="operation">ufunc operation to perform</param>
            /// <param name="dtype"></param>
            /// <param name="a">First array</param>
            /// <param name="b">Second array</param>
            /// <param name="out">A location into which the result is stored.</param>
            /// <returns></returns>
            public static ndarray outer(UFuncOperation operation, dtype dtype, object a, object b, ndarray @out = null)
            {
                return ufuncbase.outer(operation, dtype, a, b, @out);
            }



        }

    }


    public static partial class np
    {
#region as(.NET System.Array)

        /// <summary>
        /// Returns bool array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static bool[] AsBoolArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_BOOL)
            {
                a = a.astype(np.Bool);
            }

            return a.rawdata(0).datap as bool[];
        }
        /// <summary>
        /// Returns sbyte array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static sbyte[] AsSByteArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_BYTE)
            {
                a = a.astype(np.Int8);
            }

            return a.rawdata(0).datap as sbyte[];
        }
        /// <summary>
        /// Returns byte array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static byte[] AsByteArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_UBYTE)
            {
                a = a.astype(np.UInt8);
            }

            return a.rawdata(0).datap as byte[];
        }
        /// <summary>
        /// Returns Int16 array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static Int16[] AsInt16Array(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_INT16)
            {
                a = a.astype(np.Int16);
            }

            return a.rawdata(0).datap as Int16[];
        }
        /// <summary>
        /// Returns UInt16 array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static UInt16[] AsUInt16Array(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_UINT16)
            {
                a = a.astype(np.UInt16);
            }

            return a.rawdata(0).datap as UInt16[];
        }
        /// <summary>
        /// Returns Int32 array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static Int32[] AsInt32Array(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_INT32)
            {
                a = a.astype(np.Int32);
            }

            return a.rawdata(0).datap as Int32[];
        }
        /// <summary>
        /// Returns UInt32 array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static UInt32[] AsUInt32Array(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_UINT32)
            {
                a = a.astype(np.UInt32);
            }

            return a.rawdata(0).datap as UInt32[];
        }
        /// <summary>
        /// Returns Int64 array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static Int64[] AsInt64Array(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_INT64)
            {
                a = a.astype(np.Int64);
            }

            return a.rawdata(0).datap as Int64[];
        }
        /// <summary>
        /// Returns UInt64 array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static UInt64[] AsUInt64Array(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_UINT64)
            {
                a = a.astype(np.UInt64);
            }

            return a.rawdata(0).datap as UInt64[];
        }
        /// <summary>
        /// Returns float array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static float[] AsFloatArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_FLOAT)
            {
                a = a.astype(np.Float32);
            }

            return a.rawdata(0).datap as float[];
        }
        /// <summary>
        /// Returns double array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static double[] AsDoubleArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_DOUBLE)
            {
                a = a.astype(np.Float64);
            }

            return a.rawdata(0).datap as double[];
        }
        /// <summary>
        /// Returns decimal array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static decimal[] AsDecimalArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_DECIMAL)
            {
                a = a.astype(np.Decimal);
            }

            return a.rawdata(0).datap as decimal[];
        }
        /// <summary>
        /// Returns Complex array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static System.Numerics.Complex[] AsComplexArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_COMPLEX)
            {
                a = a.astype(np.Complex);
            }

            return a.rawdata(0).datap as System.Numerics.Complex[];
        }
        /// <summary>
        /// Returns BigInt array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static System.Numerics.BigInteger[] AsBigIntArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_BIGINT)
            {
                a = a.astype(np.BigInt);
            }

            return a.rawdata(0).datap as System.Numerics.BigInteger[];
        }
        /// <summary>
        /// Returns Object array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static object[] AsObjectArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_OBJECT)
            {
                a = a.astype(np.Object);
            }

            return a.rawdata(0).datap as Object[];
        }
        /// <summary>
        /// Returns String array.  Converts if necessary
        /// </summary>
        /// <param name="oa"></param>
        /// <returns></returns>
        public static string[] AsStringArray(object oa)
        {
            var a = ConvertToFlattenedArray(oa);

            if (a.TypeNum != NPY_TYPES.NPY_STRING)
            {
                a = a.astype(np.Strings);
            }

            return a.rawdata(0).datap as string[];
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
