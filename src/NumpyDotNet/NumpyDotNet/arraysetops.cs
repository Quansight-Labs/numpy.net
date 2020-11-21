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
using System.Linq;
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
        #region ediff1d

        public static ndarray ediff1d(ndarray ary, ndarray to_end= null, ndarray to_begin= null)
        {
            /*
            The differences between consecutive elements of an array.

            Parameters
            ----------
            ary : array_like
                If necessary, will be flattened before the differences are taken.
            to_end : array_like, optional
                Number(s) to append at the end of the returned differences.
            to_begin : array_like, optional
                Number(s) to prepend at the beginning of the returned differences.

            Returns
            -------
            ediff1d : ndarray
                The differences. Loosely, this is ``ary.flat[1:] - ary.flat[:-1]``.

            See Also
            --------
            diff, gradient

            Notes
            -----
            When applied to masked arrays, this function drops the mask information
            if the `to_begin` and/or `to_end` parameters are used.

            Examples
            --------
            >>> x = np.array([1, 2, 4, 7, 0])
            >>> np.ediff1d(x)
            array([ 1,  2,  3, -7])

            >>> np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99]))
            array([-99,   1,   2,   3,  -7,  88,  99])

            The returned array is always 1D.

            >>> y = [[1, 2, 4], [1, 6, 24]]
            >>> np.ediff1d(y)
            array([ 1,  2, -3,  5, 18])
            */

            // force a 1d array
            ary = np.asanyarray(ary).ravel();

            // fast track default case
            if (to_begin is null && to_end is null)
            {
                return ary.A("1:") - ary.A(":-1");
            }

            int l_begin = 0;
            int l_end = 0;

            if (to_begin is null)
            {
                l_begin = 0;
            }
            else
            {
                to_begin = np.asanyarray(to_begin).ravel();
                l_begin = len(to_begin);
            }

            if (to_end == null)
            {
                l_end = 0;
            }
            else
            {
                to_end = np.asanyarray(to_end).ravel();
                l_end = len(to_end);
            }

            // do the calculation in place and copy to_begin and to_end
            int l_diff = Math.Max(len(ary) - 1, 0);
            ndarray result = np.empty(new shape(l_diff + l_begin + l_end), dtype: ary.Dtype);
            result = ary.__array_wrap__(result);
            if (l_begin > 0)
            {
                result[":" + l_begin.ToString()] = to_begin;
            }

            if (l_end > 0)
            {
                result[(l_begin + l_diff).ToString() + ":"] = to_end;
            }

            ndarray _out = result.A(l_begin.ToString() + ":" + (l_begin + l_diff).ToString());
            _out = np.subtract(ary.A("1:"), ary.A(":-1"));
            result[l_begin.ToString() + ":" + (l_begin + l_diff).ToString()] = _out;
            return result;

        }

        #endregion

        #region unique

        public class uniqueData
        {
            public ndarray data;
            public ndarray indices;
            public ndarray inverse;
            public ndarray counts;
        }

        public static uniqueData unique(ndarray ar, bool return_index = false, bool return_inverse = false, bool return_counts = false, int? axis = null)
        {
            /*
            Find the unique elements of an array.

            Returns the sorted unique elements of an array. There are three optional
            outputs in addition to the unique elements:

            * the indices of the input array that give the unique values
            * the indices of the unique array that reconstruct the input array
            * the number of times each unique value comes up in the input array

            Parameters
            ----------
            ar : array_like
                Input array. Unless `axis` is specified, this will be flattened if it
                is not already 1-D.
            return_index : bool, optional
                If True, also return the indices of `ar` (along the specified axis,
                if provided, or in the flattened array) that result in the unique array.
            return_inverse : bool, optional
                If True, also return the indices of the unique array (for the specified
                axis, if provided) that can be used to reconstruct `ar`.
            return_counts : bool, optional
                If True, also return the number of times each unique item appears
                in `ar`.

                .. versionadded:: 1.9.0

            axis : int or None, optional
                The axis to operate on. If None, `ar` will be flattened. If an integer,
                the subarrays indexed by the given axis will be flattened and treated
                as the elements of a 1-D array with the dimension of the given axis,
                see the notes for more details.  Object arrays or structured arrays
                that contain objects are not supported if the `axis` kwarg is used. The
                default is None.

                .. versionadded:: 1.13.0

            Returns
            -------
            unique : ndarray
                The sorted unique values.
            unique_indices : ndarray, optional
                The indices of the first occurrences of the unique values in the
                original array. Only provided if `return_index` is True.
            unique_inverse : ndarray, optional
                The indices to reconstruct the original array from the
                unique array. Only provided if `return_inverse` is True.
            unique_counts : ndarray, optional
                The number of times each of the unique values comes up in the
                original array. Only provided if `return_counts` is True.             
            */

            ar = np.asanyarray(ar);
            if (axis == null)
            {
                var ret = _unique1d(ar, return_index, return_inverse, return_counts);
                return ret;
            }

            // axis was specified and not None
            try
            {
                ar = swapaxes(ar, axis.Value, 0);
            }
            catch (Exception ex)
            {
                string Error = ex.Message;
                throw new Exception(Error);
            }

            npy_intp[] orig_shape = ar.dims;
            dtype orig_dtype = ar.Dtype;

            ar = ar.reshape(new shape((int)orig_shape[0], -1));
            ar = np.ascontiguousarray(ar);

            ndarray consolidated = null;
            try
            {
                // todo: this nasty code needs to be implemented in order for this to work correctly.
                // dtype = [('f{i}'.format(i = i), ar.dtype) for i in range(ar.shape[1])]
                // consolidated = ar.view(dtype);
                consolidated = ar.view(ar.Dtype);
            }
            catch (Exception ex)
            {
                throw new Exception(ex.Message);
            }

            var output = _unique1d(consolidated, return_index, return_inverse, return_counts);
            output.data = reshape_uniq(output.data, axis.Value, orig_dtype, orig_shape);

            return output;
        }

        private static ndarray reshape_uniq(ndarray uniq, int axis, dtype orig_dtype, npy_intp[] orig_shape)
        {
            uniq = uniq.view(orig_dtype);

            npy_intp[] orig_shape_adjusted = new npy_intp[orig_shape.Length];
            Array.Copy(orig_shape, 0, orig_shape_adjusted, 0, orig_shape.Length);
            orig_shape_adjusted[0] = -1;

            uniq = uniq.reshape(new shape(orig_shape_adjusted));
            uniq = np.swapaxes(uniq, 0, axis);
            return uniq;
        }
    

        private static uniqueData _unique1d(ndarray ar1, bool return_index= false, 
            bool return_inverse= false, bool return_counts= false)
        {
            //Find the unique elements of an array, ignoring shape.

            var ar = np.asanyarray(ar1).flatten();

            bool optional_indices = return_index || return_inverse;


            ndarray perm = null;
            ndarray aux = null;

            if (optional_indices)
            {
                if (return_index)
                {
                    perm = ar.ArgSort(kind: NPY_SORTKIND.NPY_QUICKSORT);
                }
                else
                {
                    perm = ar.ArgSort(kind: NPY_SORTKIND.NPY_MERGESORT);
                }
                aux = ar.A(perm);
            }
            else
            {
                ar = ar.Sort();
                aux = ar;
            }

            ndarray mask = np.empty(aux.shape, dtype : np.Bool);
            mask[":1"] = true;

            ndarray T1 = aux.A("1:");
            ndarray T2 = aux.A(":-1");
            mask["1:"] = T1.NotEquals(T2);

            var ret = new uniqueData();

            ret.data = aux.A(mask);
            if (return_index)
            {
                ret.indices = perm.A(mask);
            }

            if (return_inverse)
            {
                ndarray imask = np.cumsum(mask) - 1;
                ndarray inv_idx = np.empty(mask.shape, dtype : np.intp);
                inv_idx[perm] = imask;
                ret.inverse = inv_idx;
            }
            if (return_counts)
            {
                List<ndarray> parts = new List<ndarray>();
                parts.AddRange(np.nonzero(mask));
                parts.Add(np.array(new npy_intp[] { mask.size }));
                ndarray idx = np.concatenate(parts);
                ret.counts = np.diff(idx);
            }

            return ret;
        }

        #endregion

        #region intersect1d

        public static ndarray intersect1d(ndarray ar1, ndarray ar2, bool assume_unique = false)
        {
            /*
            Find the intersection of two arrays.

            Return the sorted, unique values that are in both of the input arrays.

            Parameters
            ----------
            ar1, ar2 : array_like
                Input arrays.
            assume_unique : bool
                If True, the input arrays are both assumed to be unique, which
                can speed up the calculation.  Default is False.

            Returns
            -------
            intersect1d : ndarray
                Sorted 1D array of common and unique elements.

            See Also
            --------
            numpy.lib.arraysetops : Module with a number of other functions for
                                    performing set operations on arrays.

            Examples
            --------
            >>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
            array([1, 3])

            To intersect more than two arrays, use functools.reduce:

            >>> from functools import reduce
            >>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
            array([3])             
            */

            if (!assume_unique)
            {
                //Might be faster than unique( intersect1d( ar1, ar2 ) )?
                ar1 = unique(ar1).data;
                ar2 = unique(ar2).data;
            }

            ndarray aux = np.concatenate(new ndarray[] { ar1, ar2 });
            aux = aux.Sort();

            ndarray mask = (aux.A("1:")).Equals(aux.A(":-1"));
            return (aux.A(":-1", mask));

        }

        #endregion

        #region setxor1d

        public static ndarray setxor1d(ndarray ar1, ndarray ar2, bool assume_unique = false)
        {
            /*
            Find the set exclusive-or of two arrays.

            Return the sorted, unique values that are in only one (not both) of the
            input arrays.

            Parameters
            ----------
            ar1, ar2 : array_like
                Input arrays.
            assume_unique : bool
                If True, the input arrays are both assumed to be unique, which
                can speed up the calculation.  Default is False.

            Returns
            -------
            setxor1d : ndarray
                Sorted 1D array of unique values that are in only one of the input
                arrays.

            Examples
            --------
            >>> a = np.array([1, 2, 3, 2, 4])
            >>> b = np.array([2, 3, 5, 7, 5])
            >>> np.setxor1d(a,b)
            array([1, 4, 5, 7])             
            */

            if (!assume_unique)
            {
                ar1 = unique(ar1).data;
                ar2 = unique(ar2).data;
            }

            ndarray aux = np.concatenate(new ndarray[] { ar1, ar2 });
            if (aux.size == 0)
                return aux;

            aux = aux.Sort();

            ndarray True1 = np.array(new bool[] { true });
            ndarray True2 = np.array(new bool[] { true });

            ndarray flag = np.concatenate(new ndarray[] { True1, (aux.A("1:")).NotEquals(aux.A(":-1")), True2 });

            ndarray mask = flag.A("1:") & flag.A(":-1");
            return aux.A(mask);

        }

        #endregion

        #region in1d
        public static ndarray in1d(ndarray ar1, ndarray ar2, bool assume_unique = false, bool invert = false)
        {
            /*
            Test whether each element of a 1-D array is also present in a second array.

            Returns a boolean array the same length as `ar1` that is True
            where an element of `ar1` is in `ar2` and False otherwise.

            We recommend using :func:`isin` instead of `in1d` for new code.

            Parameters
            ----------
            ar1 : (M,) array_like
                Input array.
            ar2 : array_like
                The values against which to test each value of `ar1`.
            assume_unique : bool, optional
                If True, the input arrays are both assumed to be unique, which
                can speed up the calculation.  Default is False.
            invert : bool, optional
                If True, the values in the returned array are inverted (that is,
                False where an element of `ar1` is in `ar2` and True otherwise).
                Default is False. ``np.in1d(a, b, invert=True)`` is equivalent
                to (but is faster than) ``np.invert(in1d(a, b))``.

                .. versionadded:: 1.8.0

            Returns
            -------
            in1d : (M,) ndarray, bool
                The values `ar1[in1d]` are in `ar2`.

            See Also
            --------
            isin                  : Version of this function that preserves the
                                    shape of ar1.
            numpy.lib.arraysetops : Module with a number of other functions for
                                    performing set operations on arrays.

            Notes
            -----
            `in1d` can be considered as an element-wise function version of the
            python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is roughly
            equivalent to ``np.array([item in b for item in a])``.
            However, this idea fails if `ar2` is a set, or similar (non-sequence)
            container:  As ``ar2`` is converted to an array, in those cases
            ``asarray(ar2)`` is an object array rather than the expected array of
            contained values.

            .. versionadded:: 1.4.0

            Examples
            --------
            >>> test = np.array([0, 1, 2, 5, 0])
            >>> states = [0, 2]
            >>> mask = np.in1d(test, states)
            >>> mask
            array([ True, False,  True, False,  True])
            >>> test[mask]
            array([0, 2, 0])
            >>> mask = np.in1d(test, states, invert=True)
            >>> mask
            array([False,  True, False,  True, False])
            >>> test[mask]
            array([1, 5])             
            */

            // Ravel both arrays, behavior for the first array could be different
            ar1 = np.asarray(ar1).ravel();
            ar2 = np.asarray(ar2).ravel();


            // This code is run when
            // a) the first condition is true, making the code significantly faster
            // b) the second condition is true (i.e. `ar1` or `ar2` may contain
            // arbitrary objects), since then sorting is not guaranteed to work

            //if (len(ar2) < 10 * Math.Pow(len(ar1), 0.145))
            //{
            //    ndarray mask;
            //    if (invert)
            //    {
            //        mask = np.ones(new shape(len(ar1)), dtype: np.Bool);
            //        foreach (dynamic a in ar2)
            //        {
            //            if (ar1.TypeNum == NPY_TYPES.NPY_STRING)
            //            {
            //                string aa = a.ToString();
            //                ndarray temp = ar1.NotEquals(aa);
            //                mask &= temp;
            //            }
            //            else
            //            {
            //                ndarray temp = ar1 != a;
            //                mask &= temp;
            //            }
            //        }
            //    }
            //    else
            //    {
            //        mask = np.zeros(new shape(len(ar1)), dtype: np.Bool);
            //        foreach (dynamic a in ar2)
            //        {
            //            if (ar1.TypeNum == NPY_TYPES.NPY_STRING)
            //            {
            //                string aa = a.ToString();
            //                ndarray temp = ar1.Equals(aa);
            //                mask |= temp;
            //            }
            //            else
            //            {
            //                ndarray temp = ar1 == a;
            //                mask |= temp;
            //            }

            //        }

            //    }
            //    return mask;
            //}

            // Otherwise use sorting

            ndarray rev_idx = null;
            if (!assume_unique)
            {
                var temp = np.unique(ar1, return_inverse: true);
                ar1 = temp.data;
                rev_idx = temp.inverse;

                ar2 = np.unique(ar2).data;
            }

            ndarray ar = np.concatenate(ar1, ar2);

            // We need this to be a stable sort, so always use 'mergesort'
            // here. The values from the first array should always come before
            // the values from the second array.
            ndarray order = ar.ArgSort(kind: NPY_SORTKIND.NPY_MERGESORT);
            ndarray bool_ar;

            ndarray sar = ar.A(order);
            if (invert)
            {
                bool_ar = (sar.A("1:")).NotEquals(sar.A(":-1"));
            }
            else
            {
                bool_ar = (sar.A("1:")).Equals(sar.A(":-1"));
            }
            ndarray flag = np.concatenate(new ndarray[] { bool_ar, np.array(new bool[] { invert }) });
            ndarray ret = np.empty(ar.shape, dtype: np.Bool);
            ret[order] = flag;

            if (assume_unique)
            {
                return ret.A(":" + len(ar1).ToString());
            }
            else
            {
                return ret.A(rev_idx);
            }

        }
        #endregion

        #region isin

        public static ndarray isin(ndarray element, ndarray test_elements, bool assume_unique = false, bool invert = false)
        {
            /*
            Calculates `element in test_elements`, broadcasting over `element` only.
            Returns a boolean array of the same shape as `element` that is True
            where an element of `element` is in `test_elements` and False otherwise.

            Parameters
            ----------
            element : array_like
                Input array.
            test_elements : array_like
                The values against which to test each value of `element`.
                This argument is flattened if it is an array or array_like.
                See notes for behavior with non-array-like parameters.
            assume_unique : bool, optional
                If True, the input arrays are both assumed to be unique, which
                can speed up the calculation.  Default is False.
            invert : bool, optional
                If True, the values in the returned array are inverted, as if
                calculating `element not in test_elements`. Default is False.
                ``np.isin(a, b, invert=True)`` is equivalent to (but faster
                than) ``np.invert(np.isin(a, b))``.

            Returns
            -------
            isin : ndarray, bool
                Has the same shape as `element`. The values `element[isin]`
                are in `test_elements`.

            See Also
            --------
            in1d                  : Flattened version of this function.
            numpy.lib.arraysetops : Module with a number of other functions for
                                    performing set operations on arrays.

            Notes
            -----

            `isin` is an element-wise function version of the python keyword `in`.
            ``isin(a, b)`` is roughly equivalent to
            ``np.array([item in b for item in a])`` if `a` and `b` are 1-D sequences.

            `element` and `test_elements` are converted to arrays if they are not
            already. If `test_elements` is a set (or other non-sequence collection)
            it will be converted to an object array with one element, rather than an
            array of the values contained in `test_elements`. This is a consequence
            of the `array` constructor's way of handling non-sequence collections.
            Converting the set to a list usually gives the desired behavior.

            .. versionadded:: 1.13.0

            Examples
            --------
            >>> element = 2*np.arange(4).reshape((2, 2))
            >>> element
            array([[0, 2],
                   [4, 6]])
            >>> test_elements = [1, 2, 4, 8]
            >>> mask = np.isin(element, test_elements)
            >>> mask
            array([[ False,  True],
                   [ True,  False]])
            >>> element[mask]
            array([2, 4])
            >>> mask = np.isin(element, test_elements, invert=True)
            >>> mask
            array([[ True, False],
                   [ False, True]])
            >>> element[mask]
            array([0, 6])

            Because of how `array` handles sets, the following does not
            work as expected:

            >>> test_set = {1, 2, 4, 8}
            >>> np.isin(element, test_set)
            array([[ False, False],
                   [ False, False]])

            Casting the set to a list gives the expected result:

            >>> np.isin(element, list(test_set))
            array([[ False,  True],
                   [ True,  False]])             
            */

            element = np.asarray(element);
            return in1d(element, test_elements, assume_unique: assume_unique,
                        invert: invert).reshape(element.shape);

        }
        #endregion

        #region union1d

        public static ndarray union1d(ndarray ar1, ndarray ar2)
        {
            /*
            Find the union of two arrays.

            Return the unique, sorted array of values that are in either of the two
            input arrays.

            Parameters
            ----------
            ar1, ar2 : array_like
                Input arrays. They are flattened if they are not already 1D.

            Returns
            -------
            union1d : ndarray
                Unique, sorted union of the input arrays.

            See Also
            --------
            numpy.lib.arraysetops : Module with a number of other functions for
                                    performing set operations on arrays.

            Examples
            --------
            >>> np.union1d([-1, 0, 1], [-2, 0, 2])
            array([-2, -1,  0,  1,  2])

            To find the union of more than two arrays, use functools.reduce:

            >>> from functools import reduce
            >>> reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
            array([1, 2, 3, 4, 6])             
            */

            return unique(np.concatenate(new ndarray[] { ar1, ar2 }, axis : null)).data;
        }
        #endregion

        #region setdiff1d
        public static ndarray setdiff1d(ndarray ar1, ndarray ar2, bool assume_unique = false)
        {
            /*
            Find the set difference of two arrays.

            Return the sorted, unique values in `ar1` that are not in `ar2`.

            Parameters
            ----------
            ar1 : array_like
                Input array.
            ar2 : array_like
                Input comparison array.
            assume_unique : bool
                If True, the input arrays are both assumed to be unique, which
                can speed up the calculation.  Default is False.

            Returns
            -------
            setdiff1d : ndarray
                Sorted 1D array of values in `ar1` that are not in `ar2`.

            See Also
            --------
            numpy.lib.arraysetops : Module with a number of other functions for
                                    performing set operations on arrays.

            Examples
            --------
            >>> a = np.array([1, 2, 3, 2, 4, 1])
            >>> b = np.array([3, 4, 5, 6])
            >>> np.setdiff1d(a, b)
            array([1, 2])             
            */

            if (assume_unique)
            {
                ar1 = ar1.ravel();
            }
            else
            {
                ar1 = unique(ar1).data;
                ar2 = unique(ar2).data;
            }
            return ar1.A(in1d(ar1, ar2, assume_unique: true, invert: true));
        }
        #endregion
    }
}
