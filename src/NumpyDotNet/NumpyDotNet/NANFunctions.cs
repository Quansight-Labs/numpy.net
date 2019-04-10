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
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif



namespace NumpyDotNet
{
    public static partial class np
    {
        private static object _get_infinity_value(ndarray arr, bool positiveInfinity)
        {
            switch (arr.Dtype.TypeNum)
            {
                case NPY_TYPES.NPY_FLOAT:
                    if (positiveInfinity)
                        return float.PositiveInfinity;
                    else
                        return float.NegativeInfinity;

                case NPY_TYPES.NPY_DOUBLE:
                    if (positiveInfinity)
                        return double.PositiveInfinity;
                    else
                        return double.NegativeInfinity;

                default:
                    break;
            }

            return 0;
        }

        private static object _get_NAN_value(ndarray arr)
        {
            switch (arr.Dtype.TypeNum)
            {
                case NPY_TYPES.NPY_FLOAT:
                    return float.NaN;

                case NPY_TYPES.NPY_DOUBLE:
                    return double.NaN;

                default:
                    break;
            }

            return 0;
        }

        private static (ndarray a, ndarray mask) _replace_nan(ndarray a, object val)
        {
            /*
             If `a` is of inexact type, make a copy of `a`, replace NaNs with
             the `val` value, and return the copy together with a boolean mask
             marking the locations where NaNs were present. If `a` is not of
             inexact type, do nothing and return `a` together with a mask of None.

             Note that scalars will end up as array scalars, which is important
             for using the result as the value of the out argument in some
             operations.

             Parameters
             ----------
             a : array-like
                 Input array.
             val : float
                 NaN values are set to val before doing the operation.

             Returns
             -------
             y : ndarray
                 If `a` is of inexact type, return a copy of `a` with the NaNs
                 replaced by the fill value, otherwise return `a`.
             mask: {bool, None}
                 If `a` is of inexact type, return a boolean mask marking locations of
                 NaNs, otherwise return None.
             */

            a = np.array(a, subok: true, copy: true);

            ndarray mask;

            if (a.Dtype.TypeNum == NPY_TYPES.NPY_OBJECT)
            {
                // object arrays do not support `isnan` (gh-9009), so make a guess
                mask = a.NotEquals(a);
            }
            else if (a.IsInexact)
            {
                mask = np.isnan(a);
            }
            else
            {
                mask = null;
            }

            if (mask != null)
            {
                a[mask] = val;
                np.copyto(a, val, where: mask);
            }

            return (a: a, mask: mask);
        }

        private static ndarray _copyto(ndarray a, object val, ndarray mask)
        {
            /*
            Replace values in `a` with NaN where `mask` is True.  This differs from
            copyto in that it will deal with the case where `a` is a numpy scalar.

            Parameters
            ----------
            a : ndarray or numpy scalar
                Array or numpy scalar some of whose values are to be replaced
                by val.
            val : numpy scalar
                Value used a replacement.
            mask : ndarray, scalar
                Boolean array. Where True the corresponding element of `a` is
                replaced by `val`. Broadcasts.

            Returns
            -------
            res : ndarray, scalar
                Array with elements replaced or scalar `val`.
            */

            np.copyto(a, val, where: mask, casting: NPY_CASTING.NPY_SAME_KIND_CASTING);
            return a;
        }

        private static (ndarray a, bool overwrite_input) _remove_nan_1d(ndarray arr1d, bool overwrite_input = false)
        {
            /*
            Equivalent to arr1d[~arr1d.isnan()], but in a different order

            Presumably faster as it incurs fewer copies

            Parameters
            ----------
            arr1d : ndarray
                Array to remove nans from
            overwrite_input : bool
                True if `arr1d` can be modified in place

            Returns
            -------
            res : ndarray
                Array with nan elements removed
            overwrite_input : bool
                True if `res` can be modified in place, given the constraint on the
                input
           */

            var c = np.isnan(arr1d);
            var s = np.nonzero(c)[0];
            if (s.size == arr1d.size)
            {
                Console.WriteLine("All-NaN slice encountered");
                //warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel = 4)
                return (a: arr1d[":0"] as ndarray, overwrite_input: true);
            }
            else if (s.size == 0)
            {
                return (a: arr1d, overwrite_input: overwrite_input);
            }
            else
            {
                if (!overwrite_input)
                {
                    arr1d = arr1d.Copy();
                }
                // select non-nans at end of array
                ndarray enonan = arr1d.A(string.Format("-{0}:", s.size)).A(~c.A(string.Format("-{0}:", s.size)));
                // fill nans in beginning of array with non-nans of end
                arr1d[s.A(string.Format(":{0}", enonan.size))] = enonan;

                return (a: arr1d.A(string.Format(":-{0}", s.size)), overwrite_input: true);
            }
        }

        private static ndarray _divide_by_count(ndarray a, ndarray b, ndarray _out = null)
        {
            /*
                Compute a/b ignoring invalid results. If `a` is an array the division
                is done in place. If `a` is a scalar, then its type is preserved in the
                output. If out is None, then then a is used instead so that the
                division is in place. Note that this is only called with `a` an inexact
                type.

                Parameters
                ----------
                a : {ndarray, numpy scalar}
                    Numerator. Expected to be of inexact type but not checked.
                b : {ndarray, numpy scalar}
                    Denominator.
                out : ndarray, optional
                    Alternate output array in which to place the result.  The default
                    is ``None``; if provided, it must have the same shape as the
                    expected output, but the type will be cast if necessary.

                Returns
                -------
                ret : {ndarray, numpy scalar}
                    The return value is a/b. If `a` was an ndarray the division is done
                    in place. If `a` is a numpy scalar, the division preserves its type.
            */

            return np.divide(a, b);
        }

        public static ndarray nanmin(object a, int? axis = null)
        {
            /*
            Return minimum of an array or minimum along an axis, ignoring any NaNs.
            When all-NaN slices are encountered a ``RuntimeWarning`` is raised and
            Nan is returned for that slice.

            Parameters
            ----------
            a : array_like
                Array containing numbers whose minimum is desired.If `a` is not an
                array, a conversion is attempted.
            axis : { int, tuple of int, None}, optional
                Axis or axes along which the minimum is computed.The default is to compute
                the minimum of the flattened array.
            out : ndarray, optional
                Alternate output array in which to place the result.The default
              is ``None``; if provided, it must have the same shape as the
                expected output, but the type will be cast if necessary.See
                `doc.ufuncs` for details.

                ..versionadded:: 1.8.0
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one.With this option,
                the result will broadcast correctly against the original `a`.

                If the value is anything but the default, then
                `keepdims` will be passed through to the `min` method
                of sub - classes of `ndarray`.  If the sub - classes methods
                does not implement `keepdims` any exceptions will be raised.

                ..versionadded:: 1.8.0

            Returns
            ------ -
            nanmin : ndarray
                An array with the same shape as `a`, with the specified axis
                removed.If `a` is a 0 - d array, or if axis is None, an ndarray
                scalar is returned.The same dtype as `a` is returned.

          See Also
            --------
            nanmax :
                The maximum value of an array along a given axis, ignoring any NaNs.
            amin:
                    The minimum value of an array along a given axis, propagating any NaNs.
            fmin:
                    Element - wise minimum of two arrays, ignoring any NaNs.
            minimum:
                    Element - wise minimum of two arrays, propagating any NaNs.
            isnan:
                    Shows which elements are Not a Number(NaN).
                isfinite:
                Shows which elements are neither NaN nor infinity.

            amax, fmax, maximum

            Notes
            -----
            NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
            (IEEE 754).This means that Not a Number is not equivalent to infinity.
            Positive infinity is treated as a very large number and negative
            infinity is treated as a very small(i.e.negative) number.

            If the input has a integer type the function is equivalent to np.min.

            Examples
            --------
            >>> a = np.array([[1, 2], [3, np.nan]])
            >>> np.nanmin(a)
            1.0
            >>> np.nanmin(a, axis = 0)
            array([1., 2.])
            >>> np.nanmin(a, axis = 1)
            array([1., 3.])

            When positive infinity and negative infinity are present:

            >>> np.nanmin([1, 2, np.nan, np.inf])
            1.0
            >>> np.nanmin([1, 2, np.nan, np.NINF])
            -inf
            */

            ndarray res = null;

            var arr = asanyarray(a);

            if (false) //  type(arr) is np.ndarray and a.dtype != np.object_:
            {
                // Fast, but not safe for subclasses of ndarray, or object arrays,
                // which do not implement isnan (gh-9009), or fmin correctly (gh-8975)
                res = null; // np.fmin.reduce(a, axis = axis, out=out, **kwargs)
                if (np.anyb(np.isnan(res)))
                {
                    Console.WriteLine("All-NaN slice encountered");
                }
            }

            else
            {
                // Slow, but safe for subclasses of ndarray
                var replaced = _replace_nan(arr, _get_infinity_value(arr, true));
                res = np.amin(replaced.a, axis: axis);
                if (replaced.mask == null)
                    return res;

                // Check for all-NaN axis
                replaced.mask = np.all(replaced.mask, axis: axis);
                if (np.anyb(replaced.mask))
                {
                    res = _copyto(res, _get_NAN_value(res), replaced.mask);
                    Console.WriteLine("All-NaN axis encountered");
                }
            }

            return res;
        }

        public static ndarray nanmax(object a, int? axis = null)
        {
            /*
            Return the maximum of an array or maximum along an axis, ignoring any
                NaNs.When all-NaN slices are encountered a ``RuntimeWarning`` is
               raised and NaN is returned for that slice.


            Parameters
            ----------

            a : array_like

                Array containing numbers whose maximum is desired.If `a` is not an

                array, a conversion is attempted.
            axis : { int, tuple of int, None}, optional
                 Axis or axes along which the maximum is computed.The default is to compute
                  the maximum of the flattened array.
              out : ndarray, optional
                  Alternate output array in which to place the result.The default
                is ``None``; if provided, it must have the same shape as the
                  expected output, but the type will be cast if necessary.See
                  `doc.ufuncs` for details.

                  ..versionadded:: 1.8.0
              keepdims : bool, optional
                  If this is set to True, the axes which are reduced are left
                  in the result as dimensions with size one.With this option,
                  the result will broadcast correctly against the original `a`.

                  If the value is anything but the default, then
                  `keepdims` will be passed through to the `max` method
                  of sub - classes of `ndarray`.  If the sub - classes methods
                  does not implement `keepdims` any exceptions will be raised.

                  ..versionadded:: 1.8.0

              Returns
              ------ -
              nanmax : ndarray
                  An array with the same shape as `a`, with the specified axis removed.
                  If `a` is a 0 - d array, or if axis is None, an ndarray scalar is
                    returned.The same dtype as `a` is returned.

              See Also
              --------
              nanmin :
                  The minimum value of an array along a given axis, ignoring any NaNs.
              amax:
                      The maximum value of an array along a given axis, propagating any NaNs.
              fmax:
                      Element - wise maximum of two arrays, ignoring any NaNs.
              maximum:
                      Element - wise maximum of two arrays, propagating any NaNs.
              isnan:
                      Shows which elements are Not a Number(NaN).
                  isfinite:
                  Shows which elements are neither NaN nor infinity.

              amin, fmin, minimum

              Notes
              -----
              NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
              (IEEE 754).This means that Not a Number is not equivalent to infinity.
              Positive infinity is treated as a very large number and negative
              infinity is treated as a very small(i.e.negative) number.

              If the input has a integer type the function is equivalent to np.max.

              Examples
              --------
              >>> a = np.array([[1, 2], [3, np.nan]])
              >>> np.nanmax(a)
              3.0
              >>> np.nanmax(a, axis = 0)
              array([3., 2.])
              >>> np.nanmax(a, axis = 1)
              array([2., 3.])

              When positive infinity and negative infinity are present:

              >>> np.nanmax([1, 2, np.nan, np.NINF])
              2.0
              >>> np.nanmax([1, 2, np.nan, np.inf])
              inf
              */

            ndarray res = null;

            var arr = asanyarray(a);

            if (false) //  type(arr) is np.ndarray and a.dtype != np.object_:
            {
                // Fast, but not safe for subclasses of ndarray, or object arrays,
                // which do not implement isnan (gh-9009), or fmin correctly (gh-8975)
                res = null; // np.fmax.reduce(a, axis = axis, out=out, **kwargs)
                if (np.anyb(np.isnan(res)))
                {
                    Console.WriteLine("All-NaN slice encountered");
                }
            }

            else
            {
                // Slow, but safe for subclasses of ndarray
                var replaced = _replace_nan(arr, _get_infinity_value(arr, false));
                res = np.amax(replaced.a, axis: axis);
                if (replaced.mask == null)
                    return res;

                // Check for all-NaN axis
                replaced.mask = np.all(replaced.mask, axis: axis);
                if (np.anyb(replaced.mask))
                {
                    res = _copyto(res, _get_NAN_value(res), replaced.mask);
                    Console.WriteLine("All-NaN axis encountered");
                }
            }

            return res;
        }

        public static ndarray nanargmin(object a, int? axis = null)
        {
            /*
            Return the indices of the minimum values in the specified axis ignoring
            NaNs.For all-NaN slices ``ValueError`` is raised.Warning: the results
            cannot be trusted if a slice contains only NaNs and Infs.

            Parameters
            ----------
            a: array_like
               Input data.
            axis : int, optional
                Axis along which to operate.By default flattened input is used.

            Returns
            -------
            index_array : ndarray
                An array of indices or a single index value.

            See Also
            --------
            argmin, nanargmax

            Examples
            --------
            >>> a = np.array([[np.nan, 4], [2, 3]])
            >>> np.argmin(a)
            0
            >>> np.nanargmin(a)
            2
            >>> np.nanargmin(a, axis=0)
            array([1, 1])
            >>> np.nanargmin(a, axis=1)
            array([1, 0])
            */

            var arr = asanyarray(a);

            var replaced = _replace_nan(arr, _get_infinity_value(arr, true));
            ndarray res = np.argmin(replaced.a, axis: axis);
            if (replaced.mask != null)
            {
                replaced.mask = np.all(replaced.mask, axis: axis);
                if (np.anyb(replaced.mask))
                {
                    throw new ValueError("All-NaN slice encountered");
                }

            }
            return res;
        }

        public static ndarray nanargmax(object a, int? axis = null)
        {
            /*
            Return the indices of the maximum values in the specified axis ignoring
            NaNs.For all-NaN slices ``ValueError`` is raised.Warning: the
            results cannot be trusted if a slice contains only NaNs and - Infs.


            Parameters
            ----------
            a: array_like
               Input data.
           axis : int, optional
                Axis along which to operate.By default flattened input is used.

            Returns
            -------
            index_array : ndarray
                An array of indices or a single index value.

            See Also
            --------
            argmax, nanargmin

            Examples
            --------
            >>> a = np.array([[np.nan, 4], [2, 3]])
            >>> np.argmax(a)
            0
            >>> np.nanargmax(a)
            1
            >>> np.nanargmax(a, axis=0)
            array([1, 0])
            >>> np.nanargmax(a, axis=1)
            array([1, 1])
            */

            var arr = asanyarray(a);

            var replaced = _replace_nan(arr, _get_infinity_value(arr, false));
            ndarray res = np.argmax(replaced.a, axis: axis);
            if (replaced.mask != null)
            {
                replaced.mask = np.all(replaced.mask, axis: axis);
                if (np.anyb(replaced.mask))
                {
                    throw new ValueError("All-NaN slice encountered");
                }

            }
            return res;
        }

        public static ndarray nansum(object a, int? axis = null, dtype dtype = null, ndarray @out = null)
        {
            /*
            Return the sum of array elements over a given axis treating Not a
            Numbers (NaNs) as zero.

            In NumPy versions <= 1.9.0 Nan is returned for slices that are all-NaN or
            empty. In later versions zero is returned.

            Parameters
            ----------
            a : array_like
                Array containing numbers whose sum is desired. If `a` is not an
                array, a conversion is attempted.
            axis : {int, tuple of int, None}, optional
                Axis or axes along which the sum is computed. The default is to compute the
                sum of the flattened array.
            dtype : data-type, optional
                The type of the returned array and of the accumulator in which the
                elements are summed.  By default, the dtype of `a` is used.  An
                exception is when `a` has an integer type with less precision than
                the platform (u)intp. In that case, the default will be either
                (u)int32 or (u)int64 depending on whether the platform is 32 or 64
                bits. For inexact inputs, dtype must be inexact.

                .. versionadded:: 1.8.0
            out : ndarray, optional
                Alternate output array in which to place the result.  The default
                is ``None``. If provided, it must have the same shape as the
                expected output, but the type will be cast if necessary.  See
                `doc.ufuncs` for details. The casting of NaN to integer can yield
                unexpected results.

                .. versionadded:: 1.8.0
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the original `a`.


                If the value is anything but the default, then
                `keepdims` will be passed through to the `mean` or `sum` methods
                of sub-classes of `ndarray`.  If the sub-classes methods
                does not implement `keepdims` any exceptions will be raised.

                .. versionadded:: 1.8.0

            Returns
            -------
            nansum : ndarray.
                A new array holding the result is returned unless `out` is
                specified, in which it is returned. The result has the same
                size as `a`, and the same shape as `a` if `axis` is not None
                or `a` is a 1-d array.

            See Also
            --------
            numpy.sum : Sum across array propagating NaNs.
            isnan : Show which elements are NaN.
            isfinite: Show which elements are not NaN or +/-inf.

            Notes
            -----
            If both positive and negative infinity are present, the sum will be Not
            A Number (NaN).

            Examples
            --------
            >>> np.nansum(1)
            1
            >>> np.nansum([1])
            1
            >>> np.nansum([1, np.nan])
            1.0
            >>> a = np.array([[1, 1], [1, np.nan]])
            >>> np.nansum(a)
            3.0
            >>> np.nansum(a, axis=0)
            array([ 2.,  1.])
            >>> np.nansum([1, np.nan, np.inf])
            inf
            >>> np.nansum([1, np.nan, np.NINF])
            -inf
            >>> np.nansum([1, np.nan, np.inf, -np.inf]) # both +/- infinity present
            nan
            */


            var replaced = _replace_nan(asanyarray(a), 0);
            return np.sum(replaced.a, axis: axis, dtype: dtype, ret: @out);
        }

        public static ndarray nanprod(object a, int? axis = null, dtype dtype = null, ndarray @out = null)
        {
            /*
            Return the product of array elements over a given axis treating Not a
            Numbers (NaNs) as ones.

            One is returned for slices that are all-NaN or empty.

            .. versionadded:: 1.10.0

            Parameters
            ----------
            a : array_like
                Array containing numbers whose product is desired. If `a` is not an
                array, a conversion is attempted.
            axis : {int, tuple of int, None}, optional
                Axis or axes along which the product is computed. The default is to compute
                the product of the flattened array.
            dtype : data-type, optional
                The type of the returned array and of the accumulator in which the
                elements are summed.  By default, the dtype of `a` is used.  An
                exception is when `a` has an integer type with less precision than
                the platform (u)intp. In that case, the default will be either
                (u)int32 or (u)int64 depending on whether the platform is 32 or 64
                bits. For inexact inputs, dtype must be inexact.
            out : ndarray, optional
                Alternate output array in which to place the result.  The default
                is ``None``. If provided, it must have the same shape as the
                expected output, but the type will be cast if necessary.  See
                `doc.ufuncs` for details. The casting of NaN to integer can yield
                unexpected results.
            keepdims : bool, optional
                If True, the axes which are reduced are left in the result as
                dimensions with size one. With this option, the result will
                broadcast correctly against the original `arr`.

            Returns
            -------
            nanprod : ndarray
                A new array holding the result is returned unless `out` is
                specified, in which case it is returned.

            See Also
            --------
            numpy.prod : Product across array propagating NaNs.
            isnan : Show which elements are NaN.

            Examples
            --------
            >>> np.nanprod(1)
            1
            >>> np.nanprod([1])
            1
            >>> np.nanprod([1, np.nan])
            1.0
            >>> a = np.array([[1, 2], [3, np.nan]])
            >>> np.nanprod(a)
            6.0
            >>> np.nanprod(a, axis=0)
            array([ 3.,  2.])
            */

            var replaced = _replace_nan(asanyarray(a), 1);
            return np.prod(replaced.a, axis: axis, dtype: dtype, @out: @out);
        }

        public static ndarray nancumsum(object a, int? axis = null, dtype dtype = null, ndarray @out = null)
        {
            /*
            Return the cumulative sum of array elements over a given axis treating Not a
            Numbers (NaNs) as zero.  The cumulative sum does not change when NaNs are
            encountered and leading NaNs are replaced by zeros.

            Zeros are returned for slices that are all-NaN or empty.

            .. versionadded:: 1.12.0

            Parameters
            ----------
            a : array_like
                Input array.
            axis : int, optional
                Axis along which the cumulative sum is computed. The default
                (None) is to compute the cumsum over the flattened array.
            dtype : dtype, optional
                Type of the returned array and of the accumulator in which the
                elements are summed.  If `dtype` is not specified, it defaults
                to the dtype of `a`, unless `a` has an integer dtype with a
                precision less than that of the default platform integer.  In
                that case, the default platform integer is used.
            out : ndarray, optional
                Alternative output array in which to place the result. It must
                have the same shape and buffer length as the expected output
                but the type will be cast if necessary. See `doc.ufuncs`
                (Section "Output arguments") for more details.

            Returns
            -------
            nancumsum : ndarray.
                A new array holding the result is returned unless `out` is
                specified, in which it is returned. The result has the same
                size as `a`, and the same shape as `a` if `axis` is not None
                or `a` is a 1-d array.

            See Also
            --------
            numpy.cumsum : Cumulative sum across array propagating NaNs.
            isnan : Show which elements are NaN.

            Examples
            --------
            >>> np.nancumsum(1)
            array([1])
            >>> np.nancumsum([1])
            array([1])
            >>> np.nancumsum([1, np.nan])
            array([ 1.,  1.])
            >>> a = np.array([[1, 2], [3, np.nan]])
            >>> np.nancumsum(a)
            array([ 1.,  3.,  6.,  6.])
            >>> np.nancumsum(a, axis=0)
            array([[ 1.,  2.],
                   [ 4.,  2.]])
            >>> np.nancumsum(a, axis=1)
            array([[ 1.,  3.],
                   [ 3.,  3.]])

            */

            var replaced = _replace_nan(asanyarray(a), 0);
            return np.cumsum(replaced.a, axis: axis, dtype: dtype, ret: @out);
        }

        public static ndarray nancumprod(object a, int? axis = null, dtype dtype = null, ndarray @out = null)
        {
            /*
            Return the cumulative product of array elements over a given axis treating Not a
            Numbers (NaNs) as one.  The cumulative product does not change when NaNs are
            encountered and leading NaNs are replaced by ones.

            Ones are returned for slices that are all-NaN or empty.

            .. versionadded:: 1.12.0

            Parameters
            ----------
            a : array_like
                Input array.
            axis : int, optional
                Axis along which the cumulative product is computed.  By default
                the input is flattened.
            dtype : dtype, optional
                Type of the returned array, as well as of the accumulator in which
                the elements are multiplied.  If *dtype* is not specified, it
                defaults to the dtype of `a`, unless `a` has an integer dtype with
                a precision less than that of the default platform integer.  In
                that case, the default platform integer is used instead.
            out : ndarray, optional
                Alternative output array in which to place the result. It must
                have the same shape and buffer length as the expected output
                but the type of the resulting values will be cast if necessary.

            Returns
            -------
            nancumprod : ndarray
                A new array holding the result is returned unless `out` is
                specified, in which case it is returned.

            See Also
            --------
            numpy.cumprod : Cumulative product across array propagating NaNs.
            isnan : Show which elements are NaN.

            Examples
            --------
            >>> np.nancumprod(1)
            array([1])
            >>> np.nancumprod([1])
            array([1])
            >>> np.nancumprod([1, np.nan])
            array([ 1.,  1.])
            >>> a = np.array([[1, 2], [3, np.nan]])
            >>> np.nancumprod(a)
            array([ 1.,  2.,  6.,  6.])
            >>> np.nancumprod(a, axis=0)
            array([[ 1.,  2.],
                   [ 3.,  2.]])
            >>> np.nancumprod(a, axis=1)
            array([[ 1.,  2.],
                   [ 3.,  3.]])
            */

            var replaced = _replace_nan(asanyarray(a), 1);
            return np.cumprod(replaced.a, axis: axis, dtype: dtype, @out: @out);
        }


        public static ndarray nanmean(object a, int? axis = null, dtype dtype = null)
        {
            /*
            Compute the arithmetic mean along the specified axis, ignoring NaNs.

            Returns the average of the array elements.  The average is taken over
            the flattened array by default, otherwise over the specified axis.
            `float64` intermediate and return values are used for integer inputs.

            For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.

            .. versionadded:: 1.8.0

            Parameters
            ----------
            a : array_like
                Array containing numbers whose mean is desired. If `a` is not an
                array, a conversion is attempted.
            axis : {int, tuple of int, None}, optional
                Axis or axes along which the means are computed. The default is to compute
                the mean of the flattened array.
            dtype : data-type, optional
                Type to use in computing the mean.  For integer inputs, the default
                is `float64`; for inexact inputs, it is the same as the input
                dtype.
            out : ndarray, optional
                Alternate output array in which to place the result.  The default
                is ``None``; if provided, it must have the same shape as the
                expected output, but the type will be cast if necessary.  See
                `doc.ufuncs` for details.
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the original `a`.

                If the value is anything but the default, then
                `keepdims` will be passed through to the `mean` or `sum` methods
                of sub-classes of `ndarray`.  If the sub-classes methods
                does not implement `keepdims` any exceptions will be raised.

            Returns
            -------
            m : ndarray, see dtype parameter above
                If `out=None`, returns a new array containing the mean values,
                otherwise a reference to the output array is returned. Nan is
                returned for slices that contain only NaNs.

            See Also
            --------
            average : Weighted average
            mean : Arithmetic mean taken while not ignoring NaNs
            var, nanvar

            Notes
            -----
            The arithmetic mean is the sum of the non-NaN elements along the axis
            divided by the number of non-NaN elements.

            Note that for floating-point input, the mean is computed using the same
            precision the input has.  Depending on the input data, this can cause
            the results to be inaccurate, especially for `float32`.  Specifying a
            higher-precision accumulator using the `dtype` keyword can alleviate
            this issue.

            Examples
            --------
            >>> a = np.array([[1, np.nan], [3, 4]])
            >>> np.nanmean(a)
            2.6666666666666665
            >>> np.nanmean(a, axis=0)
            array([ 2.,  4.])
            >>> np.nanmean(a, axis=1)
            array([ 1.,  3.5])
            */

            var replaced = _replace_nan(asanyarray(a), 0);
            if (replaced.mask == null)
            {
                return np.mean(replaced.a, axis: axis, dtype: dtype);
            }

            if (dtype != null && replaced.a.IsInexact)
            {
                if (!dtype.IsInexact)
                {
                    throw new TypeError("If a is inexact, then dtype must be inexact");
                }

            }

            var cnt = np.sum(~replaced.mask, axis: axis, dtype: np.intp);
            var tot = np.sum(replaced.a, axis: axis, dtype: dtype);
            var avg = _divide_by_count(tot, cnt, null);

            var isbad = (cnt == 0);
            if (np.anyb(isbad))
            {
                Console.WriteLine("Mean of empty slice");
                //warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel = 2)
                // NaN is the only possible bad value, so no further
                // action is needed to handle bad results.
            }

            return avg;
        }

        private static ndarray _nanmedian1d(ndarray arr1d, params object[] args)
        {
            bool ow_input = false;
            if (args != null && args[0] is bool)
            {
                ow_input = (bool)args[0];
            }


            // Private function for rank 1 arrays.Compute the median ignoring NaNs.
            // See nanmedian for parameter usage
            var removed = _remove_nan_1d(arr1d, overwrite_input: ow_input);
            if (removed.a.size == 0)
                return asanyarray(_get_NAN_value(removed.a));

            return asanyarray(np.median(removed.a));
        }

        private static ndarray _nanmedian(ndarray a, ndarray q, bool IsQarray, int? axis = null, ndarray @out = null,
            bool overwrite_input = false, string interpolation = "linear", bool keepdims = false)
        {
            //Private function that doesn't support extended axis or keepdims.
            //These methods are extended to this function using _ureduce
            //See nanmedian for parameter usage

            if (axis == null || a.ndim == 1)
            {
                ndarray part = a.ravel();
                if (@out == null)
                {
                    return _nanmedian1d(part, overwrite_input);
                }
                else
                {
                    @out["..."] = _nanmedian1d(part, overwrite_input);
                    return @out;
                }
            }
            else
            {
                // for small medians use sort + indexing which is still faster than
                // apply_along_axis
                // benchmarked with shuffled (50, 50, x) containing a few NaN
                //if (a.shape.iDims[axis.Value] < 600)
                //{
                //    return _nanmedian_small(a, axis, @out, overwrite_input);
                //}

                var result = np.apply_along_axis(_nanmedian1d, axis.Value, a, (object)overwrite_input);
                if (@out != null)
                {
                    @out["..."] = result;
                }
                return result;
            }

        }

        private static ndarray _nanmedian_small(ndarray a, int? axis = null, ndarray @out = null, bool overwrite_input = false)
        {
            //sort + indexing median, faster for small medians along multiple
            //dimensions due to the high overhead of apply_along_axis

            //see nanmedian for parameter usage

            //a = np.ma.masked_array(a, np.isnan(a))
            //m = np.ma.median(a, axis = axis, overwrite_input = overwrite_input)
            //for i in range(np.count_nonzero(m.mask.ravel())):
            //    warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel = 3)
            //if out is not None:
            //    out[...] = m.filled(np.nan)
            //    return out
            //return m.filled(np.nan)

            throw new NotImplementedException();
        }

        public static ndarray nanmedian(object a, int? axis = null, ndarray @out = null, bool? keepdims = null)
        {
            /*
            Compute the median along the specified axis, while ignoring NaNs.

            Returns the median of the array elements.

            .. versionadded:: 1.9.0

            Parameters
            ----------
            a : array_like
                Input array or object that can be converted to an array.
            axis : {int, sequence of int, None}, optional
                Axis or axes along which the medians are computed. The default
                is to compute the median along a flattened version of the array.
                A sequence of axes is supported since version 1.9.0.
            out : ndarray, optional
                Alternative output array in which to place the result. It must
                have the same shape and buffer length as the expected output,
                but the type (of the output) will be cast if necessary.
            overwrite_input : bool, optional
               If True, then allow use of memory of input array `a` for
               calculations. The input array will be modified by the call to
               `median`. This will save memory when you do not need to preserve
               the contents of the input array. Treat the input as undefined,
               but it will probably be fully or partially sorted. Default is
               False. If `overwrite_input` is ``True`` and `a` is not already an
               `ndarray`, an error will be raised.
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the original `a`.

                If this is anything but the default value it will be passed
                through (in the special case of an empty array) to the
                `mean` function of the underlying array.  If the array is
                a sub-class and `mean` does not have the kwarg `keepdims` this
                will raise a RuntimeError.

            Returns
            -------
            median : ndarray
                A new array holding the result. If the input contains integers
                or floats smaller than ``float64``, then the output data-type is
                ``np.float64``.  Otherwise, the data-type of the output is the
                same as that of the input. If `out` is specified, that array is
                returned instead.

            See Also
            --------
            mean, median, percentile

            Notes
            -----
            Given a vector ``V`` of length ``N``, the median of ``V`` is the
            middle value of a sorted copy of ``V``, ``V_sorted`` - i.e.,
            ``V_sorted[(N-1)/2]``, when ``N`` is odd and the average of the two
            middle values of ``V_sorted`` when ``N`` is even.

            Examples
            --------
            >>> a = np.array([[10.0, 7, 4], [3, 2, 1]])
            >>> a[0, 1] = np.nan
            >>> a
            array([[ 10.,  nan,   4.],
               [  3.,   2.,   1.]])
            >>> np.median(a)
            nan
            >>> np.nanmedian(a)
            3.0
            >>> np.nanmedian(a, axis=0)
            array([ 6.5,  2.,  2.5])
            >>> np.median(a, axis=1)
            array([ 7.,  2.])
            >>> b = a.copy()
            >>> np.nanmedian(b, axis=1, overwrite_input=True)
            array([ 7.,  2.])
            >>> assert not np.all(a==b)
            >>> b = a.copy()
            >>> np.nanmedian(b, axis=None, overwrite_input=True)
            3.0
            >>> assert not np.all(a==b)             
            */

            bool overwrite_input = false;

            ndarray arr = asanyarray(a);
            //apply_along_axis in _nanmedian doesn't handle empty arrays well,
            // so deal them upfront
            if (arr.size == 0)
            {
                return nanmean(a, axis);
            }

            int[] axisarray = null;
            if (axis != null)
            {
                axisarray = new[] { axis.Value };
            }


            var _ureduce_ret = _ureduce(arr, func: _nanmedian, q: null, IsQarray: false, axisarray: axisarray, @out: @out, overwrite_input: overwrite_input);
            if (keepdims != null && keepdims.Value == true)
            {
                return _ureduce_ret.r.reshape(_ureduce_ret.keepdims);
            }
            else
            {
                return _ureduce_ret.r;
            }

        }

        public static ndarray nanpercentile(object a, object q, int? axis = null, bool overwrite_input = false,
                  string interpolation = "linear", bool keepdims = false)
        {
            /*
            Compute the qth percentile of the data along the specified axis,
            while ignoring nan values.

            Returns the qth percentile(s) of the array elements.

            .. versionadded:: 1.9.0

            Parameters
            ----------
            a : array_like
                Input array or object that can be converted to an array, containing
                nan values to be ignored.
            q : array_like of float
                Percentile or sequence of percentiles to compute, which must be between
                0 and 100 inclusive.
            axis : {int, tuple of int, None}, optional
                Axis or axes along which the percentiles are computed. The
                default is to compute the percentile(s) along a flattened
                version of the array.
            out : ndarray, optional
                Alternative output array in which to place the result. It must
                have the same shape and buffer length as the expected output,
                but the type (of the output) will be cast if necessary.
            overwrite_input : bool, optional
                If True, then allow the input array `a` to be modified by intermediate
                calculations, to save memory. In this case, the contents of the input
                `a` after this function completes is undefined.
            interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
                This optional parameter specifies the interpolation method to
                use when the desired quantile lies between two data points
                ``i < j``:
                    * linear: ``i + (j - i) * fraction``, where ``fraction``
                      is the fractional part of the index surrounded by ``i``
                      and ``j``.
                    * lower: ``i``.
                    * higher: ``j``.
                    * nearest: ``i`` or ``j``, whichever is nearest.
                    * midpoint: ``(i + j) / 2``.
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left in
                the result as dimensions with size one. With this option, the
                result will broadcast correctly against the original array `a`.

                If this is anything but the default value it will be passed
                through (in the special case of an empty array) to the
                `mean` function of the underlying array.  If the array is
                a sub-class and `mean` does not have the kwarg `keepdims` this
                will raise a RuntimeError.

            Returns
            -------
            percentile : scalar or ndarray
                If `q` is a single percentile and `axis=None`, then the result
                is a scalar. If multiple percentiles are given, first axis of
                the result corresponds to the percentiles. The other axes are
                the axes that remain after the reduction of `a`. If the input
                contains integers or floats smaller than ``float64``, the output
                data-type is ``float64``. Otherwise, the output data-type is the
                same as that of the input. If `out` is specified, that array is
                returned instead.

            See Also
            --------
            nanmean
            nanmedian : equivalent to ``nanpercentile(..., 50)``
            percentile, median, mean

            Notes
            -----
            Given a vector ``V`` of length ``N``, the ``q``-th percentile of
            ``V`` is the value ``q/100`` of the way from the minimum to the
            maximum in a sorted copy of ``V``. The values and distances of
            the two nearest neighbors as well as the `interpolation` parameter
            will determine the percentile if the normalized ranking does not
            match the location of ``q`` exactly. This function is the same as
            the median if ``q=50``, the same as the minimum if ``q=0`` and the
            same as the maximum if ``q=100``.

            Examples
            --------
            >>> a = np.array([[10., 7., 4.], [3., 2., 1.]])
            >>> a[0][1] = np.nan
            >>> a
            array([[ 10.,  nan,   4.],
                  [  3.,   2.,   1.]])
            >>> np.percentile(a, 50)
            nan
            >>> np.nanpercentile(a, 50)
            3.5
            >>> np.nanpercentile(a, 50, axis=0)
            array([ 6.5,  2.,   2.5])
            >>> np.nanpercentile(a, 50, axis=1, keepdims=True)
            array([[ 7.],
                   [ 2.]])
            >>> m = np.nanpercentile(a, 50, axis=0)
            >>> out = np.zeros_like(m)
            >>> np.nanpercentile(a, 50, axis=0, out=out)
            array([ 6.5,  2.,   2.5])
            >>> m
            array([ 6.5,  2. ,  2.5])

            >>> b = a.copy()
            >>> np.nanpercentile(b, 50, axis=1, overwrite_input=True)
            array([  7.,  2.])
            >>> assert not np.all(a==b)
            */

            bool IsQArray = false;
            if (q.GetType().IsArray)
            {
                IsQArray = true;
            }

            var arr = np.asanyarray(a);
            var qarr = np.true_divide(q, 100.0);  // handles the asarray for us too
            if (!_quantile_is_valid(qarr))
            {
                throw new ValueError("Percentiles must be in the range [0, 100]");
            }

            return _nanquantile_unchecked(arr, qarr,IsQArray, axis, overwrite_input, interpolation, keepdims);
        }



        public static ndarray nanquantile(object a, object q, int? axis = null, bool overwrite_input = false,
                  string interpolation = "linear", bool keepdims = false)
        {

            /*
            Compute the qth quantile of the data along the specified axis,
            while ignoring nan values.
            Returns the qth quantile(s) of the array elements.
            .. versionadded:: 1.15.0

            Parameters
            ----------
            a : array_like
                Input array or object that can be converted to an array, containing
                nan values to be ignored
            q : array_like of float
                Quantile or sequence of quantiles to compute, which must be between
                0 and 1 inclusive.
            axis : {int, tuple of int, None}, optional
                Axis or axes along which the quantiles are computed. The
                default is to compute the quantile(s) along a flattened
                version of the array.
            out : ndarray, optional
                Alternative output array in which to place the result. It must
                have the same shape and buffer length as the expected output,
                but the type (of the output) will be cast if necessary.
            overwrite_input : bool, optional
                If True, then allow the input array `a` to be modified by intermediate
                calculations, to save memory. In this case, the contents of the input
                `a` after this function completes is undefined.
            interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
                This optional parameter specifies the interpolation method to
                use when the desired quantile lies between two data points
                ``i < j``:
                    * linear: ``i + (j - i) * fraction``, where ``fraction``
                      is the fractional part of the index surrounded by ``i``
                      and ``j``.
                    * lower: ``i``.
                    * higher: ``j``.
                    * nearest: ``i`` or ``j``, whichever is nearest.
                    * midpoint: ``(i + j) / 2``.
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left in
                the result as dimensions with size one. With this option, the
                result will broadcast correctly against the original array `a`.

                If this is anything but the default value it will be passed
                through (in the special case of an empty array) to the
                `mean` function of the underlying array.  If the array is
                a sub-class and `mean` does not have the kwarg `keepdims` this
                will raise a RuntimeError.

            Returns
            -------
            quantile : scalar or ndarray
                If `q` is a single percentile and `axis=None`, then the result
                is a scalar. If multiple quantiles are given, first axis of
                the result corresponds to the quantiles. The other axes are
                the axes that remain after the reduction of `a`. If the input
                contains integers or floats smaller than ``float64``, the output
                data-type is ``float64``. Otherwise, the output data-type is the
                same as that of the input. If `out` is specified, that array is
                returned instead.

            See Also
            --------
            quantile
            nanmean, nanmedian
            nanmedian : equivalent to ``nanquantile(..., 0.5)``
            nanpercentile : same as nanquantile, but with q in the range [0, 100].

            Examples
            --------
            >>> a = np.array([[10., 7., 4.], [3., 2., 1.]])
            >>> a[0][1] = np.nan
            >>> a
            array([[ 10.,  nan,   4.],
                  [  3.,   2.,   1.]])
            >>> np.quantile(a, 0.5)
            nan
            >>> np.nanquantile(a, 0.5)
            3.5
            >>> np.nanquantile(a, 0.5, axis=0)
            array([ 6.5,  2.,   2.5])
            >>> np.nanquantile(a, 0.5, axis=1, keepdims=True)
            array([[ 7.],
                   [ 2.]])
            >>> m = np.nanquantile(a, 0.5, axis=0)
            >>> out = np.zeros_like(m)
            >>> np.nanquantile(a, 0.5, axis=0, out=out)
            array([ 6.5,  2.,   2.5])
            >>> m
            array([ 6.5,  2. ,  2.5])
            >>> b = a.copy()
            >>> np.nanquantile(b, 0.5, axis=1, overwrite_input=True)
            array([  7.,  2.])
            >>> assert not np.all(a==b)
            */

            bool IsQArray = false;
            if (q.GetType().IsArray)
            {
                IsQArray = true;
            }

            var arr = np.asanyarray(a);
            var qarr = np.asanyarray(q);
            if (!_quantile_is_valid(qarr))
            {
                throw new ValueError("Percentiles must be in the range [0, 1]");
            }

            return _nanquantile_unchecked(arr, qarr, IsQArray, axis, overwrite_input, interpolation, keepdims);
        }


        private static ndarray _nanquantile_unchecked(ndarray a, ndarray q, bool IsQArray, int? axis = null, bool overwrite_input = false,
                                   string interpolation = "linear", bool keepdims = false)
        {
            // Assumes that q is in [0, 1], and is an ndarray

            // apply_along_axis in _nanpercentile doesn't handle empty arrays well,
            // so deal them upfront
            if (a.size == 0)
            {
                return np.nanmean(a, axis);
            }

            int[] axisarray = null;
            if (axis != null)
            {
                axisarray = new[] { axis.Value };
            }

            var _ureduce_ret = _ureduce(a, func: _nanquantile_ureduce_func, q: q, IsQarray: IsQArray, axisarray: axisarray, overwrite_input: overwrite_input,
                interpolation: interpolation);

            if (keepdims)
            {
                List<npy_intp> newshape = new List<npy_intp>();
                if (IsQArray)
                {
                    newshape.AddRange(q.shape.iDims);
                }
                newshape.AddRange(_ureduce_ret.keepdims);
                return _ureduce_ret.r.reshape(new shape(newshape.ToArray()));
            }
            else
            {
                return _ureduce_ret.r;
            }
        }

        private static ndarray _nanquantile_ureduce_func(ndarray a, ndarray q, bool IsQarray, int? axis = null, ndarray @out = null,
                                        bool overwrite_input = false, string interpolation = "linear", bool keepdims = false)
        {
            // Private function that doesn't support extended axis or keepdims.
            // These methods are extended to this function using _ureduce
            // See nanpercentile for parameter usage

            ndarray result;

            if (axis == null || a.ndim == 1)
            {
                var part = a.ravel();
                result = _nanquantile_1d(part, q, IsQarray, overwrite_input, interpolation);
            }

            else
            {
                result = np.apply_along_axis(_nanquantile_1d, axis.Value, a, q, IsQarray, overwrite_input, interpolation);
                // apply_along_axis fills in collapsed axis with results.
                // Move that axis to the beginning to match percentile's
                // convention.
                if (IsQarray && q.ndim != 0)
                {
                    result = np.moveaxis(result, axis, 0);
                }
            }


            if (@out != null)
                @out["..."] = result;
            return result;
        }

        private static ndarray _nanquantile_1d(ndarray arr1d, params object[] args)
        {
            ndarray q = null;
            bool IsQArray = false;
            bool ow_input = false;
            string interpolation = "linear";


            if (args != null && args[0] is ndarray)
            {
                q = (ndarray)args[0];
            }
            if (args != null && args[1] is bool)
            {
                IsQArray = (bool)args[1];
            }
            if (args != null && args[2] is bool)
            {
                ow_input = (bool)args[2];
            }
            if (args != null && args[3] is string)
            {
                interpolation = (string)args[3];
            }

            // Private function for rank 1 arrays.Compute the median ignoring NaNs.
            // See nanmedian for parameter usage
            var removed = _remove_nan_1d(arr1d, overwrite_input: ow_input);
            if (removed.a.size == 0)
                return np.full(q.shape, _get_NAN_value(q));  // convert to scalar

            return _quantile_unchecked(removed.a, q, IsQArray: IsQArray, overwrite_input: removed.overwrite_input, interpolation: interpolation);

        }

   
        public static ndarray nanvar(object a, int? axis = null, dtype dtype = null, int ddof = 0, bool keepdims = false)
        {
            /*
            Compute the variance along the specified axis, while ignoring NaNs.

            Returns the variance of the array elements, a measure of the spread of
            a distribution.  The variance is computed for the flattened array by
            default, otherwise over the specified axis.

            For all-NaN slices or slices with zero degrees of freedom, NaN is
            returned and a `RuntimeWarning` is raised.

            .. versionadded:: 1.8.0

            Parameters
            ----------
            a : array_like
                Array containing numbers whose variance is desired.  If `a` is not an
                array, a conversion is attempted.
            axis : {int, tuple of int, None}, optional
                Axis or axes along which the variance is computed.  The default is to compute
                the variance of the flattened array.
            dtype : data-type, optional
                Type to use in computing the variance.  For arrays of integer type
                the default is `float32`; for arrays of float types it is the same as
                the array type.
            out : ndarray, optional
                Alternate output array in which to place the result.  It must have
                the same shape as the expected output, but the type is cast if
                necessary.
            ddof : int, optional
                "Delta Degrees of Freedom": the divisor used in the calculation is
                ``N - ddof``, where ``N`` represents the number of non-NaN
                elements. By default `ddof` is zero.
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the original `a`.


            Returns
            -------
            variance : ndarray, see dtype parameter above
                If `out` is None, return a new array containing the variance,
                otherwise return a reference to the output array. If ddof is >= the
                number of non-NaN elements in a slice or the slice contains only
                NaNs, then the result for that slice is NaN.

            See Also
            --------
            std : Standard deviation
            mean : Average
            var : Variance while not ignoring NaNs
            nanstd, nanmean
            numpy.doc.ufuncs : Section "Output arguments"

            Notes
            -----
            The variance is the average of the squared deviations from the mean,
            i.e.,  ``var = mean(abs(x - x.mean())**2)``.

            The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
            If, however, `ddof` is specified, the divisor ``N - ddof`` is used
            instead.  In standard statistical practice, ``ddof=1`` provides an
            unbiased estimator of the variance of a hypothetical infinite
            population.  ``ddof=0`` provides a maximum likelihood estimate of the
            variance for normally distributed variables.

            Note that for complex numbers, the absolute value is taken before
            squaring, so that the result is always real and nonnegative.

            For floating-point input, the variance is computed using the same
            precision the input has.  Depending on the input data, this can cause
            the results to be inaccurate, especially for `float32` (see example
            below).  Specifying a higher-accuracy accumulator using the ``dtype``
            keyword can alleviate this issue.

            For this function to work on sub-classes of ndarray, they must define
            `sum` with the kwarg `keepdims`

            Examples
            --------
            >>> a = np.array([[1, np.nan], [3, 4]])
            >>> np.var(a)
            1.5555555555555554
            >>> np.nanvar(a, axis=0)
            array([ 1.,  0.])
            >>> np.nanvar(a, axis=1)
            array([ 0.,  0.25])
            */

            bool _keepdims = false;
            ndarray sqr = null;

            var replaced = _replace_nan(asanyarray(a), 0);
            var arr = replaced.a;
            var mask = replaced.mask;

            if (mask == null)
            {
                return np.var(arr, axis: axis, dtype: dtype, ddof: ddof, keep_dims: keepdims);
            }

            if (dtype != null && arr.IsInexact)
            {
                if (!dtype.IsInexact)
                {
                    throw new TypeError("If a is inexact, then dtype must be inexact");
                }

            }

            // Compute mean
            if (arr.IsMatrix)
                _keepdims = false;
            else
                _keepdims = true;


            // we need to special case matrix for reverse compatibility
            // in order for this to work, these sums need to be called with
            // keepdims=True, however matrix now raises an error in this case, but
            // the reason that it drops the keepdims kwarg is to force keepdims=True
            // so this used to work by serendipity.

            var cnt = np.sum(~mask, axis: axis, dtype: np.intp, keepdims: _keepdims);
            var avg = np.sum(arr, axis: axis, dtype: dtype, keepdims: _keepdims);
  

            avg = _divide_by_count(avg, cnt);

            // Compute squared deviation from mean.
            arr = np.subtract(arr, avg);
            arr = _copyto(arr, 0, mask);
            if (arr.IsComplex)
                sqr = np.multiply(arr, arr.conj()).real as ndarray;
            else
                sqr = np.multiply(arr, arr);

            // Compute variance.
            var var = np.sum(sqr, axis: axis, dtype: dtype);
            if (var.ndim < cnt.ndim)
            {
                // Subclasses of ndarray may ignore keepdims, so check here.
                cnt = np.squeeze(cnt, axis);
            }
            var dof = cnt - ddof;
            var = _divide_by_count(var, dof);


            var isbad = (dof <= 0);
            if (np.anyb(isbad))
            {
                Console.WriteLine("Degrees of freedom <= 0 for slice.");
                // NaN, inf, or negative numbers are all possible bad
                // values, so explicitly replace them with NaN.
                var = _copyto(var, _get_NAN_value(var), isbad);
            }
            return var;
        }

        public static ndarray nanstd(object a, int? axis = null, dtype dtype = null, int ddof = 0, bool keepdims = false)
        {
            /*
            Compute the standard deviation along the specified axis, while
            ignoring NaNs.

            Returns the standard deviation, a measure of the spread of a
            distribution, of the non-NaN array elements. The standard deviation is
            computed for the flattened array by default, otherwise over the
            specified axis.

            For all-NaN slices or slices with zero degrees of freedom, NaN is
            returned and a `RuntimeWarning` is raised.

            .. versionadded:: 1.8.0

            Parameters
            ----------
            a : array_like
                Calculate the standard deviation of the non-NaN values.
            axis : {int, tuple of int, None}, optional
                Axis or axes along which the standard deviation is computed. The default is
                to compute the standard deviation of the flattened array.
            dtype : dtype, optional
                Type to use in computing the standard deviation. For arrays of
                integer type the default is float64, for arrays of float types it
                is the same as the array type.
            out : ndarray, optional
                Alternative output array in which to place the result. It must have
                the same shape as the expected output but the type (of the
                calculated values) will be cast if necessary.
            ddof : int, optional
                Means Delta Degrees of Freedom.  The divisor used in calculations
                is ``N - ddof``, where ``N`` represents the number of non-NaN
                elements.  By default `ddof` is zero.

            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the original `a`.

                If this value is anything but the default it is passed through
                as-is to the relevant functions of the sub-classes.  If these
                functions do not have a `keepdims` kwarg, a RuntimeError will
                be raised.

            Returns
            -------
            standard_deviation : ndarray, see dtype parameter above.
                If `out` is None, return a new array containing the standard
                deviation, otherwise return a reference to the output array. If
                ddof is >= the number of non-NaN elements in a slice or the slice
                contains only NaNs, then the result for that slice is NaN.

            See Also
            --------
            var, mean, std
            nanvar, nanmean
            numpy.doc.ufuncs : Section "Output arguments"

            Notes
            -----
            The standard deviation is the square root of the average of the squared
            deviations from the mean: ``std = sqrt(mean(abs(x - x.mean())**2))``.

            The average squared deviation is normally calculated as
            ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is
            specified, the divisor ``N - ddof`` is used instead. In standard
            statistical practice, ``ddof=1`` provides an unbiased estimator of the
            variance of the infinite population. ``ddof=0`` provides a maximum
            likelihood estimate of the variance for normally distributed variables.
            The standard deviation computed in this function is the square root of
            the estimated variance, so even with ``ddof=1``, it will not be an
            unbiased estimate of the standard deviation per se.

            Note that, for complex numbers, `std` takes the absolute value before
            squaring, so that the result is always real and nonnegative.

            For floating-point input, the *std* is computed using the same
            precision the input has. Depending on the input data, this can cause
            the results to be inaccurate, especially for float32 (see example
            below).  Specifying a higher-accuracy accumulator using the `dtype`
            keyword can alleviate this issue.

            Examples
            --------
            >>> a = np.array([[1, np.nan], [3, 4]])
            >>> np.nanstd(a)
            1.247219128924647
            >>> np.nanstd(a, axis=0)
            array([ 1.,  0.])
            >>> np.nanstd(a, axis=1)
            array([ 0.,  0.5])             
            */

            var var = nanvar(a, axis: axis, dtype: dtype, ddof: ddof, keepdims: keepdims);
            var std = np.sqrt(var);
            return std;

        }

    }


}
