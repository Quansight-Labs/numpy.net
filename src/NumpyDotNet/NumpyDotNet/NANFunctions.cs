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

        private static (ndarray a, ndarray mask) _replace_nan(ndarray a, float val)
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
                    //warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel = 2)
                }
            }

            else
            {
                // Slow, but safe for subclasses of ndarray
                var replaced = _replace_nan(arr, float.PositiveInfinity);
                res = np.amin(replaced.a, axis: axis);
                if (replaced.mask == null)
                    return res;

                // Check for all-NaN axis
                replaced.mask = np.all(replaced.mask, axis: axis);
                if (np.anyb(replaced.mask))
                {
                    res = _copyto(res, float.NaN, replaced.mask);
                    //warnings.warn("All-NaN axis encountered", RuntimeWarning, stacklevel = 2)
                }
            }

            return res;
        }

        public static ndarray nanmax(object a, int? axis = null)
        {
            ndarray res = null;

            var arr = asanyarray(a);

            if (false) //  type(arr) is np.ndarray and a.dtype != np.object_:
            {
                // Fast, but not safe for subclasses of ndarray, or object arrays,
                // which do not implement isnan (gh-9009), or fmin correctly (gh-8975)
                res = null; // np.fmax.reduce(a, axis = axis, out=out, **kwargs)
                if (np.anyb(np.isnan(res)))
                {
                    //warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel = 2)
                }
            }

            else
            {
                // Slow, but safe for subclasses of ndarray
                var replaced = _replace_nan(arr, float.NegativeInfinity);
                res = np.amax(replaced.a, axis: axis);
                if (replaced.mask == null)
                    return res;

                // Check for all-NaN axis
                replaced.mask = np.all(replaced.mask, axis: axis);
                if (np.anyb(replaced.mask))
                {
                    res = _copyto(res, float.NaN, replaced.mask);
                    //warnings.warn("All-NaN axis encountered", RuntimeWarning, stacklevel = 2)
                }
            }

            return res;
        }

        public static ndarray nanargmin(object a, int? axis = null)
        {
            var arr = asanyarray(a);

            var replaced = _replace_nan(arr, float.PositiveInfinity);
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
            var arr = asanyarray(a);

            var replaced = _replace_nan(arr, float.PositiveInfinity);
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
            var replaced = _replace_nan(asanyarray(a), 0);
            return np.sum(replaced.a, axis: axis, dtype: dtype, ret : @out);
        }

        public static ndarray nanprod(object a, int? axis = null, dtype dtype = null, ndarray @out = null)
        {

            var replaced = _replace_nan(asanyarray(a), 1);
            return np.prod(replaced.a, axis: axis, dtype: dtype, @out: @out);
        }

        public static ndarray nancumsum(object a, int? axis = null, dtype dtype = null, ndarray @out = null)
        {
            var replaced = _replace_nan(asanyarray(a), 0);
            return np.cumsum(replaced.a, axis: axis, dtype: dtype, ret: @out);
        }

        public static ndarray nancumprod(object a, int? axis = null, dtype dtype = null, ndarray @out = null)
        {

            var replaced = _replace_nan(asanyarray(a), 1);
            return np.cumprod(replaced.a, axis: axis, dtype: dtype, @out: @out);
        }


        public static ndarray nanmean(object a, int? axis = null, dtype dtype = null)
        {

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
                //warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel = 2)
                // NaN is the only possible bad value, so no further
                // action is needed to handle bad results.
            }

            return avg;
        }

        private static float _nanmedian1d(ndarray arr1d, bool ow_input= false)
        {
            // Private function for rank 1 arrays.Compute the median ignoring NaNs.
            // See nanmedian for parameter usage
            var removed = _remove_nan_1d(arr1d, overwrite_input: ow_input);
            if (removed.a.size == 0)
                return float.NaN;

            return Convert.ToSingle(np.median(arr1d)[0]);
        }


    }
}
