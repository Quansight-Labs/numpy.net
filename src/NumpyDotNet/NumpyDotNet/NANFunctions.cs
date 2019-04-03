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
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Text;
using NumpyLib;
using npy_intp = System.Int32;


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

        private static (ndarray a, bool overwrite_input) _remove_nan_1d(ndarray arr1d,bool overwrite_input = false)
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
                //warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel = 4)
                return  (a: arr1d[":0"] as ndarray, overwrite_input: true);
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
                ndarray enonan = arr1d.A(string.Format("{0}:",-s.size)).A(~c.A(string.Format("{0}:", -s.size)));
                // fill nans in beginning of array with non-nans of end
                arr1d[s.A(string.Format(":{0}", enonan.size))] = enonan;

                return (a: arr1d.A(string.Format(":{0}",-s.size)), overwrite_input: true);
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
            return np.sum(replaced.a, axis: axis, dtype: dtype, ret : @out);
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

        private static object _nanmedian1d(ndarray arr1d, bool ow_input= false)
        {
            // Private function for rank 1 arrays.Compute the median ignoring NaNs.
            // See nanmedian for parameter usage
            var removed = _remove_nan_1d(arr1d, overwrite_input: ow_input);
            if (removed.a.size == 0)
                return _get_NAN_value(removed.a);

            return np.median(arr1d)[0];
        }


    }
}
