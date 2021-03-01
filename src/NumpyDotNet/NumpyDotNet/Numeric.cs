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
using System.Diagnostics;
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
        #region zeros
        /// <summary>
        /// Return a new array of given shape and type, filled with zeros
        /// </summary>
        /// <param name="shape">int or sequence of ints, Shape of the new array</param>
        /// <param name="dtype">(optional) Desired output data-type</param>
        /// <param name="order">(optional) {‘C’, ‘F’}, Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.</param>
        /// <returns>Array of zeros with the given shape, dtype, and order.</returns>
        public static ndarray zeros(object shape, dtype dtype = null, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            if (shape == null)
            {
                throw new Exception("shape can't be null");
            }

            double FillValue = 0;

            return CommonFill(dtype, shape, FillValue, CheckOnlyCorF(order), false, 0);
        }
        #endregion

        #region zeros_like
        /// <summary>
        /// Return an array of zeros with the same shape and type as a given array.
        /// </summary>
        /// <param name="src">The shape and data-type of a define these same attributes of the returned array.</param>
        /// <param name="dtype">(optional) Overrides the data type of the result</param>
        /// <param name="order">(optional) {‘C’, ‘F’, ‘A’, or ‘K’}, Overrides the memory layout of the result. ‘C’ means C-order, ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous, ‘C’ otherwise. ‘K’ means match the layout of a as closely as possible.</param>
        /// <param name="subok">(optional) If True, then the newly created array will use the sub-class type of ‘a’, otherwise it will be a base-class array. Defaults to True.</param>
        /// <returns>Array of zeros with the same shape and type as a.</returns>
        public static ndarray zeros_like(object osrc, dtype dtype = null, NPY_ORDER order = NPY_ORDER.NPY_CORDER, bool subok = true)
        {
            if (osrc == null)
            {
                throw new Exception("array can't be null");
            }

            var src = asanyarray(osrc);
            shape shape = new shape(src.Array.dimensions, src.Array.nd);
            double FillValue = 0;

            if (dtype == null)
            {
                dtype = src.Dtype;
            }

            return CommonFill(dtype, shape, FillValue, ConvertOrder(src, order), subok, 0);
        }
        #endregion

        #region ones
        /// <summary>
        /// Return a new array of given shape and type, filled with ones
        /// </summary>
        /// <param name="shape">int or sequence of ints, Shape of the new array</param>
        /// <param name="dtype">(optional) Desired output data-type</param>
        /// <param name="order">(optional) {‘C’, ‘F’}, Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.</param>
        /// <returns>Array of ones with the given shape, dtype, and order.</returns>
        public static ndarray ones(object shape, dtype dtype = null, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            if (shape == null)
            {
                throw new Exception("shape can't be null");
            }

            double FillValue = 1;

            return CommonFill(dtype, shape, FillValue, CheckOnlyCorF(order), false, 0);
        }
        #endregion

        #region ones_like
        /// <summary>
        /// Return an array of ones with the same shape and type as a given array.
        /// </summary>
        /// <param name="src">The shape and data-type of a define these same attributes of the returned array.</param>
        /// <param name="dtype">(optional) Overrides the data type of the result</param>
        /// <param name="order">(optional) {‘C’, ‘F’, ‘A’, or ‘K’}, Overrides the memory layout of the result. ‘C’ means C-order, ‘F’ means F-order, ‘A’ means ‘F’ if src is Fortran contiguous, ‘C’ otherwise. ‘K’ means match the layout of a as closely as possible.</param>
        /// <param name="subok">(optional) If True, then the newly created array will use the sub-class type of ‘a’, otherwise it will be a base-class array. Defaults to True.</param>
        /// <returns>Array of ones with the same shape and type as a.</returns>
        public static ndarray ones_like(object osrc, dtype dtype = null, NPY_ORDER order = NPY_ORDER.NPY_KORDER, bool subok = true)
        {
            if (osrc == null)
            {
                throw new Exception("array can't be null");
            }

            var src = asanyarray(osrc);

            shape shape = new shape(src.Array.dimensions, src.Array.nd);
            double FillValue = 1;


            if (dtype == null)
            {
                dtype = src.Dtype;
            }

            return CommonFill(dtype, shape, FillValue, ConvertOrder(src, order), subok, 0);
        }
        #endregion

        #region empty
        /// <summary>
        /// Return a new array of given shape and type, without initializing entries
        /// </summary>
        /// <param name="shape">int or tuple of int, Shape of the empty array</param>
        /// <param name="dtype">(optional) Desired output data-type</param>
        /// <param name="order">(optional) {‘C’, ‘F’}, Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.</param>
        /// <returns>Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to None.</returns>
        public static ndarray empty(object shape, dtype dtype = null, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            if (dtype != null)
            {
                if (dtype.TypeNum == NPY_TYPES.NPY_OBJECT || dtype.TypeNum == NPY_TYPES.NPY_STRING)
                {
                    return full(shape, null, dtype, order);
                }
            }
            return zeros(shape, dtype, order);
        }
        #endregion

        #region empty_like
        /// <summary>
        /// Return a new array with the same shape and type as a given array.
        /// </summary>
        /// <param name="src">The shape and data-type of a define these same attributes of the returned array.</param>
        /// <param name="dtype">(optional) Overrides the data type of the result</param>
        /// <param name="order">(optional) {‘C’, ‘F’, ‘A’, or ‘K’}, Overrides the memory layout of the result. ‘C’ means C-order, ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous, ‘C’ otherwise. ‘K’ means match the layout of a as closely as possible.</param>
        /// <param name="subok">(optional) If True, then the newly created array will use the sub-class type of ‘a’, otherwise it will be a base-class array. Defaults to True.</param>
        /// <returns>Array of uninitialized (arbitrary) data with the same shape and type as a.</returns>
        public static ndarray empty_like(object src, dtype dtype = null, NPY_ORDER order = NPY_ORDER.NPY_KORDER, bool subok = true)
        {
            if (dtype != null)
            {
                if (dtype.TypeNum == NPY_TYPES.NPY_OBJECT || dtype.TypeNum == NPY_TYPES.NPY_STRING)
                {
                    return full_like(src, null, dtype, order);
                }
            }
            return zeros_like(src, dtype, order, subok);
        }
        #endregion

        #region full
        /// <summary>
        /// Return a new array of given shape and type, filled with fill_value
        /// </summary>
        /// <param name="shape">int or sequence of ints, Shape of the new array</param>
        /// <param name="fill_value">Fill value.  Must be scalar type</param>
        /// <param name="dtype">(optional) Desired output data-type</param>
        /// <param name="order">(optional) {‘C’, ‘F’}, Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        public static ndarray full(object shape, object fill_value, dtype dtype = null, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            if (shape == null)
            {
                throw new Exception("shape can't be null");
            }

            return CommonFill(dtype, shape, fill_value, CheckOnlyCorF(order), false, 0);
        }
        #endregion

        #region full_like
        /// <summary>
        /// Return an array of zeros with the same shape and type as a given array.
        /// </summary>
        /// <param name="src">The shape and data-type of a define these same attributes of the returned array.</param>
        /// <param name="fill_value">Fill value.  Must be scalar type</param>
        /// <param name="dtype">(optional) Overrides the data type of the result</param>
        /// <param name="order">(optional) {‘C’, ‘F’, ‘A’, or ‘K’}, Overrides the memory layout of the result. ‘C’ means C-order, ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous, ‘C’ otherwise. ‘K’ means match the layout of a as closely as possible.</param>
        /// <param name="subok">(optional) If True, then the newly created array will use the sub-class type of ‘a’, otherwise it will be a base-class array. Defaults to True.</param>
        /// <returns>Array of fill_value with the same shape and type as a.</returns>
        public static ndarray full_like(object osrc, object fill_value, dtype dtype = null, NPY_ORDER order = NPY_ORDER.NPY_KORDER, bool subok = true)
        {
            if (osrc == null)
            {
                throw new Exception("array can't be null");
            }

            var src = asanyarray(osrc);

            shape shape = new shape(src.Array.dimensions, src.Array.nd);

            if (dtype == null)
            {
                dtype = src.Dtype;
            }

            return CommonFill(dtype, shape, fill_value, ConvertOrder(src, order), subok, 0);
        }
        #endregion

        #region CommonFill

        private static ndarray CommonFill(dtype dtype, object oshape, object FillValue, NPY_ORDER order, bool subok, int ndmin)
        {
            if (dtype == null)
            {
                dtype = np.Float64;
            }

            shape shape = null;
            if (oshape is shape)
            {
                shape = oshape as shape;
            }
            else if ((shape = NumpyExtensions.ConvertTupleToShape(oshape)) == null)
            {
                throw new Exception("Unable to convert shape object");
            }

            long ArrayLen = CalculateNewShapeSize(shape);


            // allocate a new array based on the calculated type and length
            var a = numpyAPI.Alloc_NewArray(dtype.TypeNum, (UInt64)ArrayLen);

            DefaultArrayHandlers.GetArrayHandler(dtype.TypeNum).ArrayFill(a, FillValue);

            // load this into a ndarray and return it to the caller
            var ndArray = array(a, dtype, false, order, subok, ndmin).reshape(shape);
            return ndArray;
        }
          

        #endregion

        #region count_nonzero
    
        /// <summary>
        /// Counts the number of non-zero values in the array a.
        /// </summary>
        /// <param name="a">The array for which to count non-zeros.</param>
        /// <param name="axis">Axis along which to count non-zeros. </param>
        /// <returns></returns>
        public static ndarray count_nonzero(object a, int? axis = null)
        {
            //  Counts the number of non - zero values in the array ``a``.

            //  The word "non-zero" is in reference to the Python 2.x
            //  built -in method ``__nonzero__()`` (renamed ``__bool__()``
            //  in Python 3.x) of Python objects that tests an object's
            //  "truthfulness".For example, any number is considered
            // truthful if it is nonzero, whereas any string is considered
            // truthful if it is not the empty string.Thus, this function
            //(recursively) counts how many elements in ``a`` (and in
            //  sub - arrays thereof) have their ``__nonzero__()`` or ``__bool__()``
            //  method evaluated to ``True``.

            //  Parameters
            //  ----------
            //  a: array_like
            //     The array for which to count non - zeros.
            // axis : int or tuple, optional

            //     Axis or tuple of axes along which to count non - zeros.
            //     Default is None, meaning that non - zeros will be counted

            //     along a flattened version of ``a``.

            //      .. versionadded:: 1.12.0


            // Returns
            // ------ -
            // count : int or array of int

            //     Number of non - zero values in the array along a given axis.
            //     Otherwise, the total number of non - zero values in the array
            //     is returned.

            // See Also

            // --------
            // nonzero : Return the coordinates of all the non - zero values.

            // Examples
            // --------
            // >>> np.count_nonzero(np.eye(4))

            // 4
            // >>> np.count_nonzero([[0, 1, 7, 0, 0],[3, 0, 0, 2, 19]])
            //              5
            //              >>> np.count_nonzero([[0, 1, 7, 0, 0],[3,0,0,2,19]], axis=0)
            //  array([1, 1, 1, 1, 1])
            //  >>> np.count_nonzero([[0, 1, 7, 0, 0],[3, 0, 0, 2, 19]], axis=1)
            //  array([2, 3])

 
            a = asanyarray(a);

            var a_bool = asanyarray(a).astype(np.Bool, copy: false);

            return a_bool.Sum(axis: axis, dtype: np.intp);
        }
        #endregion

        #region asarray

        /// <summary>
        /// Convert the input to an array.
        /// </summary>
        /// <param name="a">Input data, in any form that can be converted to an array. </param>
        /// <param name="dtype">data-type, optional. By default, the data-type is inferred from the input data.</param>
        /// <param name="order">{‘C’, ‘F’, ‘A’, ‘K’}, optional</param>
        /// <returns></returns>
        public static ndarray asarray(object a, dtype dtype = null, NPY_ORDER order = NPY_ORDER.NPY_ANYORDER)
        {
            //    Convert the input to an ndarray, but pass ndarray subclasses through.

            //    Parameters
            //    ----------
            //    a: array_like
            //       Input data, in any form that can be converted to an array.This
            //       includes scalars, lists, lists of tuples, tuples, tuples of tuples,
            //        tuples of lists, and ndarrays.
            //    dtype: data - type, optional
            //         By default, the data-type is inferred from the input data.
            //     order : { 'C', 'F'}, optional
            //          Whether to use row - major(C - style) or column-major
            //          (Fortran - style) memory representation.  Defaults to 'C'.

            //      Returns
            //      ------ -
            //      out : ndarray or an ndarray subclass
            //        Array interpretation of `a`.  If `a` is an ndarray or a subclass
            //        of ndarray, it is returned as-is and no copy is performed.

            //    See Also
            //    --------
            //    asarray : Similar function which always returns ndarrays.
            //    ascontiguousarray: Convert input to a contiguous array.
            //    asfarray: Convert input to a floating point ndarray.
            //   asfortranarray : Convert input to an ndarray with column - major
            //                     memory order.
            //    asarray_chkfinite: Similar function which checks input for NaNs and

            //                       Infs.
            //   fromiter : Create an array from an iterator.
            //   fromfunction : Construct an array by executing a function on grid

            //                  positions.

            //   Examples
            //   --------

            //   Convert a list into an array:


            //   >>> a = [1, 2]
            //   >>> np.asanyarray(a)

            //   array([1, 2])

            //   Instances of `ndarray` subclasses are passed through as-is:

            //    >>> a = np.matrix([1, 2])
            //    >>> np.asanyarray(a) is a
            //    True

            return array(a, dtype, copy: false, order: order, subok: true);
        }
        #endregion

        #region asanyarray
        /// <summary>
        /// Convert the input to an ndarray, but pass ndarray subclasses through.
        /// </summary>
        /// <param name="a">Input data, in any form that can be converted to an array.</param>
        /// <param name="dtype">data-type, optional. By default, the data-type is inferred from the input data.</param>
        /// <param name="order">{‘C’, ‘F’, ‘A’, ‘K’}, optional</param>
        /// <returns></returns>
        public static ndarray asanyarray(object a, dtype dtype = null, NPY_ORDER order = NPY_ORDER.NPY_ANYORDER)
        {
            //  Convert the input to a masked array, conserving subclasses.

            //  If `a` is a subclass of `MaskedArray`, its class is conserved.
            //  No copy is performed if the input is already an `ndarray`.

            //  Parameters
            //  ----------
            //  a : array_like
            //      Input data, in any form that can be converted to an array.
            //  dtype : dtype, optional
            //      By default, the data-type is inferred from the input data.
            //  order : {'C', 'F'}, optional
            //      Whether to use row-major('C') or column-major('FORTRAN') memory
            //    representation.Default is 'C'.
            //
            //  Returns
            //  -------
            //
            // out : MaskedArray
            //    MaskedArray interpretation of `a`.
            //
            //
            //See Also
            //  --------
            //
            //asarray : Similar to `asanyarray`, but does not conserve subclass.
            //
            //Examples
            //  --------
            //  >>> x = np.arange(10.).reshape(2, 5)
            //  >>> x
            //
            //array([[0., 1., 2., 3., 4.],
            //
            //       [5., 6., 7., 8., 9.]])
            //  >>> np.ma.asanyarray(x)
            //  masked_array(data =
            //   [[0.  1.  2.  3.  4.]
            //   [5.  6.  7.  8.  9.]],
            //               mask =
            //   False,
            //         fill_value = 1e+20)
            //  >>> type(np.ma.asanyarray(x))
            //  <class 'numpy.ma.core.MaskedArray'>

            //if (a is MaskedArray && (dtype == null || dtype == a.Dtype))
            //{
            //    return a;
            //}
            //return masked_array(a, dtype: dtype, copy: false, keep_mask: true, sub_ok: true);

            if (a == null)
            {
                return null;
            }

            if (dtype != null)
            {
                return np.array(a, dtype, copy: false, order: order, subok: true);
            }

            if (a is ndarray)
            {
                return a as ndarray;
            }
            if (a is ndarray[])
            {
                if (sametype(a as ndarray[]))
                {
                    if (sameshape(a as ndarray[]))
                    {
                        return np.vstack(a as ndarray[]);
                    }
                }

                var b = np.Concatenate(a as ndarray[]);
                return b;
            }
   
 
            if (a.GetType().IsArray)
            {
                System.Array ssrc = a as System.Array;
                NPY_TYPES type_num;

                try
                {
                    object nonnull = FindFirstNonNullValue(ssrc);
                    type_num = Get_NPYType(nonnull);
                }
                catch (Exception ex)
                {
                    throw;
                }
    

                return ndArrayFromMD(ssrc, type_num, ssrc.Rank);
            }

            if (IsNumericType(a))
            {
                ndarray ret = np.array(GetSingleElementArray(a), null);
                ret.Array.IsScalar = true;
                return ret;
            }

            if (a is string)
            {
                ndarray ret = np.array(GetSingleElementArray(string.Empty), null);
                ret.SetItem(a, 0);
                ret.Array.IsScalar = false;
                return ret;
            }

            if (a is object)
            {
                ndarray ret = np.array(GetSingleElementArray(new object()), null);
                ret.SetItem(a, 0);
                ret.Array.IsScalar = false;
                return ret;
            }

            throw new Exception("Unable to convert object to ndarray");
        }

        private static bool sametype(ndarray[] ndarray)
        {
            if (ndarray.Length <= 1)
                return false;

            foreach (var arr in ndarray)
            {
                if (arr.TypeNum != ndarray[0].TypeNum)
                    return false;
            }

            return true;

        }

        private static bool sameshape(ndarray[] ndarray)
        {
            if (ndarray.Length <= 1)
                return false;

            foreach (var arr in ndarray)
            {
                if (!arr.shape.Equals(ndarray[0].shape))
                    return false;
            }

            return true;

        }


        private static object FindFirstNonNullValue(Array ssrc)
        {
            switch (ssrc.Rank)
            {
                case 1:
                    for (int i = 0; i < ssrc.GetLength(0); i++)
                    {
                        object oValue = ssrc.GetValue(i);
                        if (oValue != null)
                        {
                            return oValue;
                        }
                    }
                    break;
                case 2:
                    for (int i = 0; i < ssrc.GetLength(0); i++)
                    {
                        for (int j = 0; j < ssrc.GetLength(1); j++)
                        {
                            object oValue = ssrc.GetValue(i, j);
                            if (oValue != null)
                            {
                                return oValue;
                            }
                        }
                    }
                    break;
                case 3:
                    for (int i = 0; i < ssrc.GetLength(0); i++)
                    {
                        for (int j = 0; j < ssrc.GetLength(1); j++)
                        {
                            for (int k = 0; k < ssrc.GetLength(2); k++)
                            {
                                object oValue = ssrc.GetValue(i, j, k);
                                if (oValue != null)
                                {
                                    return oValue;
                                }
                            }
                        }
                    }
                    break;
                case 4:
                    for (int i = 0; i < ssrc.GetLength(0); i++)
                    {
                        for (int j = 0; j < ssrc.GetLength(1); j++)
                        {
                            for (int k = 0; k < ssrc.GetLength(2); k++)
                            {
                                for (int l = 0; l < ssrc.GetLength(3); l++)
                                {
                                    object oValue = ssrc.GetValue(i, j, k, l);
                                    if (oValue != null)
                                    {
                                        return oValue;
                                    }
                                }
                            }
                        }
                    }
                    break;

                case 5:
                    for (int i = 0; i < ssrc.GetLength(0); i++)
                    {
                        for (int j = 0; j < ssrc.GetLength(1); j++)
                        {
                            for (int k = 0; k < ssrc.GetLength(2); k++)
                            {
                                for (int l = 0; l < ssrc.GetLength(3); l++)
                                {
                                    for (int m = 0; m < ssrc.GetLength(4); m++)
                                    {
                                        object oValue = ssrc.GetValue(i, j, k, l, m);
                                        if (oValue != null)
                                        {
                                            return oValue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    break;
                case 6:
                    for (int i = 0; i < ssrc.GetLength(0); i++)
                    {
                        for (int j = 0; j < ssrc.GetLength(1); j++)
                        {
                            for (int k = 0; k < ssrc.GetLength(2); k++)
                            {
                                for (int l = 0; l < ssrc.GetLength(3); l++)
                                {
                                    for (int m = 0; m < ssrc.GetLength(4); m++)
                                    {
                                        for (int n = 0; n < ssrc.GetLength(5); n++)
                                        {
                                            object oValue = ssrc.GetValue(i, j, k, l, m, n);
                                            if (oValue != null)
                                            {
                                                return oValue;
                                            }
                                        }
  
                                    }
                                }
                            }
                        }
                    }
                    break;
                case 7:
                    for (int i = 0; i < ssrc.GetLength(0); i++)
                    {
                        for (int j = 0; j < ssrc.GetLength(1); j++)
                        {
                            for (int k = 0; k < ssrc.GetLength(2); k++)
                            {
                                for (int l = 0; l < ssrc.GetLength(3); l++)
                                {
                                    for (int m = 0; m < ssrc.GetLength(4); m++)
                                    {
                                        for (int n = 0; n < ssrc.GetLength(5); n++)
                                        {
                                            for (int o = 0; o < ssrc.GetLength(6); o++)
                                            {
                                                object oValue = ssrc.GetValue(i, j, k, l, m, n, o);
                                                if (oValue != null)
                                                {
                                                    return oValue;
                                                }
                                            }
   
                                        }

                                    }
                                }
                            }
                        }
                    }
                    break;
                case 8:
                    for (int i = 0; i < ssrc.GetLength(0); i++)
                    {
                        for (int j = 0; j < ssrc.GetLength(1); j++)
                        {
                            for (int k = 0; k < ssrc.GetLength(2); k++)
                            {
                                for (int l = 0; l < ssrc.GetLength(3); l++)
                                {
                                    for (int m = 0; m < ssrc.GetLength(4); m++)
                                    {
                                        for (int n = 0; n < ssrc.GetLength(5); n++)
                                        {
                                            for (int o = 0; o < ssrc.GetLength(6); o++)
                                            {
                                                for (int p = 0; p < ssrc.GetLength(7); p++)
                                                {
                                                    object oValue = ssrc.GetValue(i, j, k, l, m, n, o, p);
                                                    if (oValue != null)
                                                    {
                                                        return oValue;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    break;
                default:
                    throw new Exception("Number of dimensions is not supported");
            }

            throw new Exception("Unable to determine array type. Could not find any non-null entries. Please specify dtype");
        }
        #endregion

        #region ascontiguousarray
        /// <summary>
        /// Return a contiguous array (ndim >= 1) in memory (C order).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="dtype">Data-type of returned array. By default, the data-type is inferred from the input data.</param>
        /// <returns></returns>
        public static ndarray ascontiguousarray(object a, dtype dtype = null)
        {
            // Return a contiguous array in memory(C order).

            // Parameters
            // ----------
            // a: array_like
            //    Input array.
            //dtype : str or dtype object, optional
            //     Data - type of returned array.

            //   Returns
            //   ------ -
            // out : ndarray
            //     Contiguous array of same shape and content as `a`, with type `dtype`
            //     if specified.

            // See Also
            // --------
            // asfortranarray : Convert input to an ndarray with column - major
            //                  memory order.
            // require: Return an ndarray that satisfies requirements.
            // ndarray.flags : Information about the memory layout of the array.

            // Examples
            // --------
            // >>> x = np.arange(6).reshape(2, 3)
            // >>> np.ascontiguousarray(x, dtype = np.float32)
            // array([[0., 1., 2.],
            //        [ 3.,  4.,  5.]], dtype=float32)
            // >>> x.flags['C_CONTIGUOUS']
            // True

            return array(a, dtype, copy: false, order: NPY_ORDER.NPY_CORDER, ndmin: 1);

        }

        #endregion

        #region asfortranarray
        /// <summary>
        /// Return an array (ndim >= 1) laid out in Fortran order in memory.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="dtype">Data-type of returned array. By default, the data-type is inferred from the input data.</param>
        /// <returns></returns>
        public static ndarray asfortranarray(ndarray a, dtype dtype = null)
        {
            // Return an array laid out in Fortran order in memory.

            // Parameters
            // ----------
            // a: array_like
            //    Input array.
            //dtype : str or dtype object, optional
            //     By default, the data-type is inferred from the input data.

            // Returns
            // ------ -
            // out : ndarray
            //     The input `a` in Fortran, or column-major, order.

            // See Also
            // --------
            // ascontiguousarray : Convert input to a contiguous(C order) array.
            //asanyarray : Convert input to an ndarray with either row or
            //     column - major memory order.
            // require: Return an ndarray that satisfies requirements.
            // ndarray.flags : Information about the memory layout of the array.

            // Examples
            // --------
            // >>> x = np.arange(6).reshape(2, 3)
            // >>> y = np.asfortranarray(x)
            // >>> x.flags['F_CONTIGUOUS']
            // False
            // >>> y.flags['F_CONTIGUOUS']
            // True

            return array(a, dtype, copy: false, order: NPY_ORDER.NPY_FORTRANORDER, ndmin: 1);
        }
        #endregion

        #region asfarray
        /// <summary>
        /// Return an array converted to a float type.
        /// </summary>
        /// <param name="a">The input array.</param>
        /// <param name="dtype">Float type code to coerce input array a. If dtype is one of the ‘int’ dtypes, it is replaced with float64.</param>
        /// <returns></returns>
        public static ndarray asfarray(object a, dtype dtype = null)
        {
            AssertConvertableToFloating(a, dtype);

            var a1 = asanyarray(a);

            if (dtype == null)
            {
                dtype = np.Float64;
            }

            var arr = a1.Array;
            if (NpyCoreApi.ScalarKind(dtype.TypeNum, ref arr) != NPY_SCALARKIND.NPY_FLOAT_SCALAR)
            {
                dtype = np.Float64;
            }

            return asarray(a1, dtype: dtype);
        }


        #endregion


        #region require
        /// <summary>
        /// Return an ndarray of the provided type that satisfies requirements.
        /// </summary>
        /// <param name="a">The object to be converted to a type - and - requirement - satisfying array.</param>
        /// <param name="dtype">The required data - type.If None preserve the current dtype.</param>
        /// <param name="requirements">str or list of str</param>
        /// <returns></returns>
        public static ndarray require(ndarray a, dtype dtype = null, char[] requirements = null)
        {
            // Return an ndarray of the provided type that satisfies requirements.

            // This function is useful to be sure that an array with the correct flags
            // is returned for passing to compiled code(perhaps through ctypes).

            // Parameters
            // ----------
            // a : array_like
            //    The object to be converted to a type - and - requirement - satisfying array.
            // dtype : data - type
            //    The required data - type.If None preserve the current dtype.If your
            //    application requires the data to be in native byteorder, include
            //    a byteorder specification as a part of the dtype specification.
            // requirements : str or list of str
            //    The requirements list can be any of the following

            //    * 'F_CONTIGUOUS'('F') - ensure a Fortran - contiguous array
            //    * 'C_CONTIGUOUS'('C') - ensure a C - contiguous array
            //    * 'ALIGNED'('A') - ensure a data - type aligned array
            //    * 'WRITEABLE'('W') - ensure a writable array
            //    * 'OWNDATA'('O') - ensure an array that owns its own data
            //    * 'ENSUREARRAY', ('E') - ensure a base array, instead of a subclass

            // See Also
            // --------
            // asarray : Convert input to an ndarray.
            // asanyarray : Convert to an ndarray, but pass through ndarray subclasses.
            // ascontiguousarray : Convert input to a contiguous array.
            // asfortranarray : Convert input to an ndarray with column - major
            //                  memory order.
            // ndarray.flags : Information about the memory layout of the array.

            // Notes
            // ---- -
            // The returned array will be guaranteed to have the listed requirements
            // by making a copy if needed.

            // Examples
            // --------
            // >>> x = np.arange(6).reshape(2, 3)
            // >>> x.flags
            //   C_CONTIGUOUS: True
            //  F_CONTIGUOUS : False
            //  OWNDATA : False
            //  WRITEABLE : True
            //  ALIGNED : True
            //  WRITEBACKIFCOPY : False
            //  UPDATEIFCOPY : False

            //>>> y = np.require(x, dtype = np.float32, requirements =['A', 'O', 'W', 'F'])
            //>>> y.flags
            //   C_CONTIGUOUS: False
            //  F_CONTIGUOUS : True
            //  OWNDATA : True
            //  WRITEABLE : True
            //  ALIGNED : True
            //  WRITEBACKIFCOPY : False
            //  UPDATEIFCOPY : False

            return asanyarray(a, dtype);

        }

        #endregion

        #region isfortran
        /// <summary>
        /// Returns True if the array is Fortran contiguous but* not*C contiguous.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns></returns>
        public static bool isfortran(ndarray a)
        {
           // Returns True if the array is Fortran contiguous but* not*C contiguous.

           // This function is obsolete and, because of changes due to relaxed stride
           // checking, its return value for the same array may differ for versions
           // of NumPy >= 1.10.0 and previous versions.If you only want to check if an
           // array is Fortran contiguous use ``a.flags.f_contiguous`` instead.

           // Parameters
           // ----------
           // a: ndarray
           //    Input array.


           //Examples
           //--------

           // np.array allows to specify whether the array is written in C - contiguous
           // order(last index varies the fastest), or FORTRAN-contiguous order in
           // memory(first index varies the fastest).

           // >>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
           // >>> a
           // array([[1, 2, 3],
           //        [4, 5, 6]])
           // >>> np.isfortran(a)
           // False

           // >>> b = np.array([[1, 2, 3], [4, 5, 6]], order='FORTRAN')
           // >>> b
           // array([[1, 2, 3],
           //        [4, 5, 6]])
           // >>> np.isfortran(b)
           // True


           // The transpose of a C-ordered array is a FORTRAN-ordered array.

           // >>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
           // >>> a
           // array([[1, 2, 3],
           //        [4, 5, 6]])
           // >>> np.isfortran(a)
           // False
           // >>> b = a.T
           // >>> b
           // array([[1, 4],
           //        [2, 5],
           //        [3, 6]])
           // >>> np.isfortran(b)
           // True

           // C-ordered arrays evaluate as False even if they are also FORTRAN-ordered.

           // >>> np.isfortran(np.array([1, 2], order='FORTRAN'))
           // False

            return a.IsFortran;
        }

        #endregion

        #region argwhere
        /// <summary>
        /// Find the indices of array elements that are non - zero, grouped by element.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <returns></returns>
        public static ndarray argwhere(ndarray a)
        {
            //    Find the indices of array elements that are non - zero, grouped by element.

            //      Parameters
            //      ----------
            //    a: array_like
            //       Input data.

            //   Returns
            //   ------ -
            //   index_array : ndarray
            //       Indices of elements that are non - zero.Indices are grouped by element.

            //    See Also
            //    --------
            //    where, nonzero

            //    Notes
            //    -----
            //    ``np.argwhere(a)`` is the same as ``np.transpose(np.nonzero(a))``.

            //    The output of ``argwhere`` is not suitable for indexing arrays.
            //    For this purpose use ``nonzero(a)`` instead.

            //    Examples
            //    --------
            //    >>> x = np.arange(6).reshape(2, 3)
            //    >>> x
            //    array([[0, 1, 2],
            //           [3, 4, 5]])
            //    >>> np.argwhere(x > 1)
            //    array([[0, 2],
            //           [1, 0],
            //           [1, 1],
            //           [1, 2]])

            return transpose(nonzero(a));
        }

        #endregion

        #region flatnonzero
        /// <summary>
        /// Return indices that are non - zero in the flattened version of a.
        /// </summary>
        /// <param name="a">Input data</param>
        /// <returns></returns>
        public static ndarray flatnonzero(object a)
        {
            // Return indices that are non - zero in the flattened version of a.

            //   This is equivalent to np.nonzero(np.ravel(a))[0].

            //   Parameters
            //   ----------
            // a: array_like
            //    Input data.

            //Returns
            //------ -
            //res : ndarray
            //    Output array, containing the indices of the elements of `a.ravel()`
            //     that are non - zero.

            // See Also
            // --------
            // nonzero : Return the indices of the non-zero elements of the input array.
            // ravel : Return a 1 - D array containing the elements of the input array.

            // Examples
            // --------
            // >>> x = np.arange(-2, 3)
            // >>> x
            // array([-2, -1, 0, 1, 2])
            // >>> np.flatnonzero(x)
            // array([0, 1, 3, 4])

            // Use the indices of the non-zero elements as an index array to extract
            // these elements:

            // >>> x.ravel()[np.flatnonzero(x)]
            // array([-2, -1, 1, 2])

            return np.nonzero(np.ravel(asanyarray(a)))[0];

        }

        #endregion


        #region outer
        /// <summary>
        /// Compute the outer product of two vectors.
        /// </summary>
        /// <param name="a">First input vector.</param>
        /// <param name="b">Second input vector.</param>
        /// <returns></returns>
        public static ndarray outer(object a, object b)
        {
            // Compute the outer product of two vectors.

            // Given two vectors, ``a = [a0, a1, ..., aM]`` and
            // ``b = [b0, b1, ..., bN]``,
            // the outer product[1]_ is::

            //  [[a0 * b0  a0 * b1...a0 * bN]
            //   [a1 * b0.
            //   [ ...          .
            //   [aM * b0            aM * bN]]

            // Parameters
            // ----------
            // a: (M,) array_like
            //    First input vector.  Input is flattened if
            //    not already 1 - dimensional.
            // b: (N,) array_like
            //    Second input vector.  Input is flattened if
            //    not already 1 - dimensional.
            // out : (M, N) ndarray, optional
            //     A location where the result is stored

            //     ..versionadded:: 1.9.0

            // Returns
            // ------ -
            // out : (M, N) ndarray
            //     ``out[i, j] = a[i] * b[j]``

            // See also
            // --------
            // inner
            // einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` is the equivalent.
            // ufunc.outer : A generalization to N dimensions and other operations.
            //               ``np.multiply.outer(a.ravel(), b.ravel())`` is the equivalent.


            var a1 = asarray(a);
            var b1 = asarray(b);

            return ufunc.outer(UFuncOperation.multiply, null, a1.ravel(), b1.ravel());
        }


        #endregion

        #region tensordot
        /// <summary>
        /// Compute tensor dot product along specified axes.
        /// </summary>
        /// <param name="a">Tensor to “dot”.</param>
        /// <param name="b">Tensor to “dot”.</param>
        /// <param name="axis">sum over the last N axes of a and the first N axes of b in order.</param>
        /// <returns></returns>
        public static ndarray tensordot(object a, object b, int axis = 2)
        {
            return tensordot(a, b, (PythonFunction.range(-axis, 0), PythonFunction.range(0, axis)));
        }
        /// <summary>
        /// Compute tensor dot product along specified axes.
        /// </summary>
        /// <param name="a">Tensor to “dot”.</param>
        /// <param name="b">Tensor to “dot”.</param>
        /// <param name="axes">a list of axes to be summed over</param>
        /// <returns></returns>
        public static ndarray tensordot(object a, object b, (npy_intp[], npy_intp[]) axes)
        {

            npy_intp[] axes_a = axes.Item1;
            npy_intp[] axes_b = axes.Item2;

            int na = axes_a.Length;
            int nb = axes_b.Length;

            var aa = asanyarray(a);
            var bb = asanyarray(b);

            var as_ = aa.shape;
            var nda = aa.ndim;
            var bs = bb.shape;
            var ndb = bb.ndim;
            bool equal = true;

            if (na != nb)
            {
                equal = false;
            }
            else
            {
                for (int k = 0; k < na; k++)
                {
                    long asindex = axes_a[k];
                    asindex = asindex >= 0 ? asindex : asindex += as_.iDims.Length;

                    long bsindex = axes_b[k];
                    bsindex = bsindex >= 0 ? bsindex : bsindex += bs.iDims.Length;

                    if (as_.iDims[asindex] != bs.iDims[bsindex])
                    {
                        equal = false;
                        break;
                    }
                    if (axes_a[k] < 0)
                    {
                        axes_a[k] += nda;
                    }
                    if (axes_b[k] < 0)
                    {
                        axes_b[k] += ndb;
                    }
                }
            }

            if (!equal)
            {
                throw new ValueError("shape-mismatch for sum");
            }

            /**********************************/

            List<int> notin = new List<int>();
            for (int k = 0; k < nda; k++)
            {
                if (!axes_a.Contains(k))
                {
                    notin.Add(k);
                }
            }

            List<npy_intp> newaxes_a = new List<npy_intp>();
            foreach (var k in notin)
                newaxes_a.Add(k);
            foreach (var k in axes_a)
                newaxes_a.Add(k);

            npy_intp N2 = 1;
            foreach (var axis in axes_a)
            {
                N2 *= as_.iDims[axis];
            }

            List<npy_intp> asax = new List<npy_intp>();
            foreach (var ax in notin)
            {
                asax.Add(as_.iDims[ax]);
            }

            var multreduce = ufunc.reduce(UFuncOperation.multiply, asanyarray(asax.ToArray()));
            var newshape_a = new shape((npy_intp)multreduce.GetItem(0), N2);

            List<npy_intp> olda = new List<npy_intp>();
            foreach (var axis in notin)
            {
                olda.Add(as_.iDims[axis]);
            }

            /**********************************/

            notin = new List<int>();
            for (int k = 0; k < ndb; k++)
            {
                if (!axes_b.Contains(k))
                {
                    notin.Add(k);
                }
            }
            List<npy_intp> newaxes_b = new List<npy_intp>();
            foreach (var k in axes_b)
                newaxes_b.Add(k);
            foreach (var k in notin)
                newaxes_b.Add(k);

            N2 = 1;
            foreach (var axis in axes_b)
            {
                N2 *= bs.iDims[axis];
            }

            List<npy_intp> bsax = new List<npy_intp>();
            foreach (var ax in notin)
            {
                bsax.Add(bs.iDims[ax]);
            }

            multreduce = ufunc.reduce(UFuncOperation.multiply, asanyarray(bsax.ToArray()));
            var newshape_b = new shape(N2, (npy_intp)multreduce.GetItem(0));

            List<npy_intp> oldb = new List<npy_intp>();
            foreach (var axis in notin)
            {
                oldb.Add(bs.iDims[axis]);
            }

            var at = aa.Transpose(newaxes_a.ToArray()).reshape(newshape_a);
            var bt = bb.Transpose(newaxes_b.ToArray()).reshape(newshape_b);
            var res = np.dot(at, bt);

            olda.AddRange(oldb);
            return res.reshape(olda);
        }

        #endregion

        #region roll
        /// <summary>
        /// Roll array elements along a given axis.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="shift">The number of places by which elements are shifted.</param>
        /// <param name="axis">Axis along which elements are shifted</param>
        /// <returns></returns>
        public static ndarray roll(ndarray a, int shift, int? axis = null)
        {
            // Roll array elements along a given axis.

            // Elements that roll beyond the last position are re - introduced at
            //   the first.

            //   Parameters
            //   ----------
            // a: array_like
            //    Input array.
            //shift : int or tuple of ints
            //    The number of places by which elements are shifted.  If a tuple,
            //     then `axis` must be a tuple of the same size, and each of the
            //     given axes is shifted by the corresponding number.If an int
            //     while `axis` is a tuple of ints, then the same value is used for
            //     all given axes.
            // axis : int or tuple of ints, optional
            //     Axis or axes along which elements are shifted.By default, the
            //     array is flattened before shifting, after which the original
            //     shape is restored.

            // Returns
            // -------
            // res : ndarray
            //     Output array, with the same shape as `a`.

            // See Also
            // --------
            // rollaxis : Roll the specified axis backwards, until it lies in a
            //            given position.

            // Notes
            // ---- -
            // ..versionadded:: 1.12.0

            // Supports rolling over multiple dimensions simultaneously.

            ndarray AdjustedArray = a;

            if (axis.HasValue)
            {
                if (axis.Value == 0)
                {
                    AdjustedArray = a.A(":");
                }
                else
                {
                    throw new Exception("axis != 0 not implemented yet");
                }

                throw new Exception("axis != 0 not implemented yet");

            }
            else
            {
                var copy = a.Copy();
                var rawdatavp = copy.rawdata(0);
                dynamic RawData = (dynamic)rawdatavp.datap;
                dynamic LastElement = 0;

                if (shift < 0)
                {
                    for (long shiftCnt = 0; shiftCnt < Math.Abs(shift); shiftCnt++)
                    {
                        LastElement = RawData[0];

                        for (long index = 1; index < RawData.Length; index++)
                        {
                            RawData[index - 1] = RawData[index];
                        }
                        RawData[RawData.Length - 1] = LastElement;
                    }
                }
                else
                {
                    for (long shiftCnt = 0; shiftCnt < Math.Abs(shift); shiftCnt++)
                    {
                        LastElement = RawData[RawData.Length - 1];

                        for (long index = RawData.Length - 2; index >= 0; index--)
                        {
                            RawData[index + 1] = RawData[index];
                        }
                        RawData[0] = LastElement;
                    }
                }


                return array(RawData, dtype: a.Dtype);
            }
        }

        private static ndarray roll_needs_work(object a, int shift, object axis = null)
        {
            // Roll array elements along a given axis.

            // Elements that roll beyond the last position are re - introduced at
            //   the first.

            //   Parameters
            //   ----------
            // a: array_like
            //    Input array.
            //shift : int or tuple of ints
            //    The number of places by which elements are shifted.  If a tuple,
            //     then `axis` must be a tuple of the same size, and each of the
            //     given axes is shifted by the corresponding number.If an int
            //     while `axis` is a tuple of ints, then the same value is used for
            //     all given axes.
            // axis : int or tuple of ints, optional
            //     Axis or axes along which elements are shifted.By default, the
            //     array is flattened before shifting, after which the original
            //     shape is restored.

            // Returns
            // -------
            // res : ndarray
            //     Output array, with the same shape as `a`.

            // See Also
            // --------
            // rollaxis : Roll the specified axis backwards, until it lies in a
            //            given position.

            // Notes
            // ---- -
            // ..versionadded:: 1.12.0

            // Supports rolling over multiple dimensions simultaneously.


            var arr = asanyarray(a);
            if (axis == null)
            {
                return roll(arr.ravel(), shift, 0).reshape(arr.shape);
            }
            else
            {
                int[] axisarray = normalize_axis_tuple(axis, arr.ndim, allow_duplicates: true);
                var broadcasted = broadcast(shift, axisarray);

                if (broadcasted.ndim > 1)
                {
                    throw new ValueError("'shift' and 'axis' should be scalars or 1D sequences");
                }

                Dictionary<int, int> shifts = new Dictionary<int, int>();
                for (int i = 0; i < arr.ndim; i++)
                {
                    shifts.Add(i, 0);
                }
                foreach (var _b in broadcasted)
                {
                    ndarray[] ss = _b as ndarray[];
                    shifts[(int)ss[1].GetItem(0)] = (int)ss[0].GetItem(0);
                }

                //object[] rolls = new object[arr.ndim];
                //for (int i = 0; i < arr.ndim; i++)
                //{
                //    rolls[i] = new Slice()
                //}

                //var rolls = BuildSliceArray()
                return null;
            }

            return null;
        }


        #endregion

        #region rollaxis
        /// <summary>
        /// Roll the specified axis backwards, until it lies in a given position.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">The axis to roll backwards.The positions of the other axes do not change relative to one another.</param>
        /// <param name="start">The axis is rolled until it lies before this position.</param>
        /// <returns></returns>
        public static ndarray rollaxis(ndarray a, int axis, int start = 0)
        {
            //  Roll the specified axis backwards, until it lies in a given position.

            //  This function continues to be supported for backward compatibility, but you
            //  should prefer `moveaxis`. The `moveaxis` function was added in NumPy
            //  1.11.

            //  Parameters
            //  ----------
            //  a : ndarray
            //      Input array.
            //  axis : int
            //      The axis to roll backwards.The positions of the other axes do not
            //    change relative to one another.
            //start : int, optional
            //      The axis is rolled until it lies before this position.The default,
            //      0, results in a "complete" roll.

            //  Returns
            //  ------ -
            //  res : ndarray
            //      For NumPy >= 1.10.0 a view of `a` is always returned. For earlier
            //      NumPy versions a view of `a` is returned only if the order of the
            //      axes is changed, otherwise the input array is returned.

            //  See Also
            //  --------
            //  moveaxis : Move array axes to new positions.
            //  roll : Roll the elements of an array by a number of positions along a
            //      given axis.

            //  Examples
            //  --------
            //  >>> a = np.ones((3, 4, 5, 6))
            //  >>> np.rollaxis(a, 3, 1).shape
            //  (3, 6, 4, 5)
            //  >>> np.rollaxis(a, 2).shape
            //  (5, 3, 4, 6)
            //  >>> np.rollaxis(a, 1, 4).shape
            //  (3, 5, 6, 4)

            var n = a.ndim;
            axis = normalize_axis_index(axis, n);
            if (start < 0)
                start += n;

            string msg = "{0} arg requires {1} <= {2} < {3}, but {4} was passed in";
            if (((0 <= start) || (start < n + 1)) == false)
            {
                throw new AxisError(string.Format(msg, "start", -n, "start", n + 1, start));
            }

            if (axis < start)
            {
                // it's been removed
                start -= 1;
            }
            if (axis == start)
            {
                return a["..."] as ndarray;
            }

            List<npy_intp> axes = new List<npy_intp>();
            for (int i = 0; i < n; i++)
                axes.Add(i);
            axes.Remove(axis);
            axes.Insert(start, axis);
            return a.Transpose(axes.ToArray());
        }
        #endregion

        #region normalize_axis_tuple
        private static int[] normalize_axis_tuple(object axis, int ndim, string argname = null, bool allow_duplicates = false)
        {
            //Normalizes an axis argument into a tuple of non - negative integer axes.

            //This handles shorthands such as ``1`` and converts them to ``(1,)``,
            //as well as performing the handling of negative indices covered by
            //`normalize_axis_index`.

            //By default, this forbids axes from being specified multiple times.

            //Used internally by multi-axis - checking logic.


            //  ..versionadded:: 1.13.0

            //Parameters
            //----------
            //axis: int, iterable of int
            //   The un - normalized index or indices of the axis.
            //ndim: int
            //   The number of dimensions of the array that `axis` should be normalized
            //    against.
            //argname : str, optional
            //    A prefix to put before the error message, typically the name of the
            //    argument.
            //allow_duplicate : bool, optional
            //    If False, the default, disallow an axis from being specified twice.

            //Returns
            //------ -
            //normalized_axes : tuple of int
            //    The normalized axis index, such that `0 <= normalized_axis < ndim`

            //Raises
            //------
            //AxisError
            //    If any axis provided is out of range
            //ValueError
            //    If an axis is repeated

            //See also
            //--------
            //normalize_axis_index: normalizing a single scalar axis

            int[] _axis = null;

            try
            {
                if (axis.GetType().IsArray)
                {
                    System.Array tempAxis = axis as System.Array;
                    _axis = new int[tempAxis.Length];
                    int index = 0;
                    foreach (var t in tempAxis)
                    {
                        _axis[index++] = Convert.ToInt32(t);
                    }
                }
                else
                {
                    _axis = new int[1];
                    _axis[0] = Convert.ToInt32(axis);
                }
            }
            catch (Exception ex)
            {
                throw new ValueError("index must be integer or array of integers");
            }


            List<int> axes = new List<int>();

            foreach (var a in _axis)
            {
                axes.Add(normalize_axis_index(a, ndim));
            }
            return axes.ToArray();
        }
        #endregion

        #region moveaxis
        /// <summary>
        /// Move axes of an array to new positions.
        /// </summary>
        /// <param name="a">The array whose axes should be reordered.</param>
        /// <param name="source">Original positions of the axes to move. These must be unique.</param>
        /// <param name="destination">Destination positions for each of the original axes.These must also be unique.</param>
        /// <returns></returns>
        public static ndarray moveaxis(ndarray a, object source, object destination)
        {
            // Move axes of an array to new positions.

            // Other axes remain in their original order.

            // ..versionadded:: 1.11.0

            // Parameters
            // ----------
            // a: np.ndarray
            //    The array whose axes should be reordered.
            // source: int or sequence of int
            //    Original positions of the axes to move. These must be unique.
            // destination: int or sequence of int
            //    Destination positions for each of the original axes.These must also be
            //    unique.

            //Returns
            //------ -
            //result : np.ndarray

            //    Array with moved axes.This array is a view of the input array.


            //See Also
            //--------

            //transpose: Permute the dimensions of an array.
            //swapaxes: Interchange two axes of an array.

            //Examples
            //--------

            //>>> x = np.zeros((3, 4, 5))
            //>>> np.moveaxis(x, 0, -1).shape
            //(4, 5, 3)
            //>>> np.moveaxis(x, -1, 0).shape
            //(5, 3, 4)


            //These all achieve the same result:


            //>>> np.transpose(x).shape
            //(5, 4, 3)
            //>>> np.swapaxes(x, 0, -1).shape
            //(5, 4, 3)
            //>>> np.moveaxis(x, [0, 1], [-1, -2]).shape
            //(5, 4, 3)
            //>>> np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
            //(5, 4, 3)

            try
            {
                // allow duck-array types if they define transpose
                var transpose = a.Transpose();
            }
            catch (Exception ex)
            {
                throw new Exception("moveaxis:Failure on transpose");
            }

            var source_axes = normalize_axis_tuple(source, a.ndim, "source");
            var destination_axes = normalize_axis_tuple(destination, a.ndim, "destination");
            if (source_axes.Length != destination_axes.Length)
            {
                throw new Exception("`source` and `destination` arguments must have the same number of elements");
            }


            List<int> order = new List<int>();
            for (int n = 0; n < a.ndim; n++)
            {
                if (!source_axes.Contains(n))
                {
                    order.Add(n);
                }
            }

            var zip = new zip2IntArrays(destination_axes, source_axes);
            zip.Sort();
            foreach (var d in zip.data)
            {
                order.Insert(d.Item1, d.Item2);
            }

            var result = np.transpose(a, order.ToArray());
            return result;
        }

        #endregion

        #region cross
        /// <summary>
        /// Return the cross product of two (arrays of) vectors.
        /// </summary>
        /// <param name="a">Components of the first vector(s)</param>
        /// <param name="b">Components of the second vector(s)</param>
        /// <param name="axisa">Axis of `a` that defines the vector(s)</param>
        /// <param name="axisb">Axis of `b` that defines the vector(s)</param>
        /// <param name="axisc">Axis of `c` containing the cross product vector(s)</param>
        /// <param name="axis">If defined, the axis of `a`, `b` and `c` that defines the vector(s) and cross product(s).</param>
        /// <returns></returns>
        public static ndarray cross(object a, object b, int axisa = -1, int axisb = -1, int axisc = -1, int ?axis = null)
        {
            /*
            Return the cross product of two (arrays of) vectors.

            The cross product of `a` and `b` in :math:`R^3` is a vector perpendicular
            to both `a` and `b`.  If `a` and `b` are arrays of vectors, the vectors
            are defined by the last axis of `a` and `b` by default, and these axes
            can have dimensions 2 or 3.  Where the dimension of either `a` or `b` is
            2, the third component of the input vector is assumed to be zero and the
            cross product calculated accordingly.  In cases where both input vectors
            have dimension 2, the z-component of the cross product is returned.

            Parameters
            ----------
            a : array_like
                Components of the first vector(s).
            b : array_like
                Components of the second vector(s).
            axisa : int, optional
                Axis of `a` that defines the vector(s).  By default, the last axis.
            axisb : int, optional
                Axis of `b` that defines the vector(s).  By default, the last axis.
            axisc : int, optional
                Axis of `c` containing the cross product vector(s).  Ignored if
                both input vectors have dimension 2, as the return is scalar.
                By default, the last axis.
            axis : int, optional
                If defined, the axis of `a`, `b` and `c` that defines the vector(s)
                and cross product(s).  Overrides `axisa`, `axisb` and `axisc`.

            Returns
            -------
            c : ndarray
                Vector cross product(s).

            Raises
            ------
            ValueError
                When the dimension of the vector(s) in `a` and/or `b` does not
                equal 2 or 3.

            See Also
            --------
            inner : Inner product
            outer : Outer product.
            ix_ : Construct index arrays.

            Notes
            -----
            .. versionadded:: 1.9.0

            Supports full broadcasting of the inputs.
            */

            if (axis != null)
            {
                axisa = axis.Value;
                axisb = axis.Value;
                axisc = axis.Value;
            }
            var aa = asarray(a);
            var bb = asarray(b);

            // Check axisa and axisb are within bounds
            axisa = normalize_axis_index(axisa, aa.ndim);
            axisb = normalize_axis_index(axisb, bb.ndim);

            // Move working axis to the end of the shape
            aa = moveaxis(aa, axisa, -1);
            bb = moveaxis(bb, axisb, -1);

            npy_intp[] acceptableDims = new npy_intp[] { 2, 3 };
            if (!acceptableDims.Contains(aa.shape.lastDim) || 
                !acceptableDims.Contains(bb.shape.lastDim))
            {
                throw new ValueError("incompatible dimensions for cross product (dimension must be 2 or 3)");
            }

            // Create the output array
            var shape = np.broadcast(aa["...", 0], bb["...", 0]).shape;
            if (aa.shape.lastDim == 3 || bb.shape.lastDim == 3)
            {
                List<npy_intp> newshape = new List<npy_intp>();
                newshape.AddRange(shape.iDims);
                newshape.Add(3);
                shape = new shape(newshape);

                // Check axisc is within bounds
                axisc = normalize_axis_index(axisc, shape.iDims.Length);
            }

            var dtype = promote_types(aa.Dtype, bb.Dtype);
            ndarray cp = empty(shape, dtype);


            // create local aliases for readability
            ndarray a0 = aa["...", 0] as ndarray;
            ndarray a1 = aa["...", 1] as ndarray;
            ndarray a2 = null;
            if (aa.shape.lastDim == 3)
            {
                a2 = aa["...", 2] as ndarray;
            }

            ndarray b0 = bb["...", 0] as ndarray;
            ndarray b1 = bb["...", 1] as ndarray;
            ndarray b2 = null;
            if (bb.shape.lastDim == 3)
            {
                b2 = bb["...", 2] as ndarray;
            }

            ndarray cp0 = null;
            ndarray cp1 = null;
            ndarray cp2 = null;
            if (cp.ndim != 0 && cp.shape.lastDim == 3)
            {
                cp0 = cp["...", 0] as ndarray;
                cp1 = cp["...", 1] as ndarray;
                cp2 = cp["...", 2] as ndarray;
            }


            if (aa.shape.lastDim == 2)
            {
                if (bb.shape.lastDim == 2)
                {
                    // a0 * b1 - a1 * b0
                    np.multiply(a0, b1, @out: cp);
                    cp.InPlaceSubtract(a1 * b0);
                    return cp;
                }
                else
                {
                    Debug.Assert(bb.shape.lastDim == 3);
                    // cp0 = a1 * b2 - 0  (a2 = 0)
                    // cp1 = 0 - a0 * b2  (a2 = 0)
                    // cp2 = a0 * b1 - a1 * b0
                    np.multiply(a1, b2, @out: cp0);
                    np.multiply(a0, b2, @out: cp1);
                    np.negative(cp1, @out: cp1);
                    np.multiply(a0, b1, @out: cp2);
                    cp2.InPlaceSubtract(a1 * b0);
                }

            }
            else
            {
                Debug.Assert(aa.shape.lastDim == 3);
                if (bb.shape.lastDim == 3)
                {
                    // cp0 = a1 * b2 - a2 * b1
                    // cp1 = a2 * b0 - a0 * b2
                    // cp2 = a0 * b1 - a1 * b0
                    np.multiply(a1, b2, @out: cp0);
                    var tmp = array(a2 * b1);
                    cp0.InPlaceSubtract(tmp);
                    np.multiply(a2, b0, @out: cp1);
                    np.multiply(a0, b2, @out: tmp);
                    cp1.InPlaceSubtract(tmp);
                    np.multiply(a0, b1, @out: cp2);
                    np.multiply(a1, b0, @out: tmp);
                    cp2.InPlaceSubtract(tmp);
                }
                else
                {
                    Debug.Assert(bb.shape.lastDim == 2);
                    // cp0 = 0 - a2 * b1  (b2 = 0)
                    // cp1 = a2 * b0 - 0  (b2 = 0)
                    // cp2 = a0 * b1 - a1 * b0
                    np.multiply(a2, b1, @out: cp0);
                    np.negative(cp0, @out: cp0);
                    np.multiply(a2, b0, @out: cp1);
                    np.multiply(a0, b1, @out: cp2);
                    cp2.InPlaceSubtract(a1 * b0);
                }
            }


            return moveaxis(cp, -1, axisc);
        }

        #endregion

        #region indices
        /// <summary>
        /// Return an array representing the indices of a grid.
        /// </summary>
        /// <param name="dimensions">The shape of the grid.</param>
        /// <param name="dtype">Data type of the result.</param>
        /// <returns></returns>
        public static ndarray indices(object dimensions, dtype dtype = null)
        {
            /*
            Return an array representing the indices of a grid.

            Compute an array where the subarrays contain index values 0,1,...
            varying only along the corresponding axis.

            Parameters
            ----------
            dimensions : sequence of ints
                The shape of the grid.
            dtype : dtype, optional
                Data type of the result.

            Returns
            -------
            grid : ndarray
                The array of grid indices,
                ``grid.shape = (len(dimensions),) + tuple(dimensions)``.

            See Also
            --------
            mgrid, meshgrid

            Notes
            -----
            The output shape is obtained by prepending the number of dimensions
            in front of the tuple of dimensions, i.e. if `dimensions` is a tuple
            ``(r0, ..., rN-1)`` of length ``N``, the output shape is
            ``(N,r0,...,rN-1)``.

            The subarrays ``grid[k]`` contains the N-D array of indices along the
            ``k-th`` axis. Explicitly::

                grid[k,i0,i1,...,iN-1] = ik

            Examples
            --------
            >>> grid = np.indices((2, 3))
            >>> grid.shape
            (2, 2, 3)
            >>> grid[0]        # row indices
            array([[0, 0, 0],
                   [1, 1, 1]])
            >>> grid[1]        # column indices
            array([[0, 1, 2],
                   [0, 1, 2]])

            The indices can be used as an index into an array.

            >>> x = np.arange(20).reshape(5, 4)
            >>> row, col = np.indices((2, 3))
            >>> x[row, col]
            array([[0, 1, 2],
                   [4, 5, 6]])

            Note that it would be more straightforward in the above example to
            extract the required elements directly with ``x[:2, :3]``.
            */

            shape _dimensions = NumpyExtensions.ConvertTupleToShape(dimensions);
            if (_dimensions == null)
            {
                throw new Exception("Unable to convert shape object");
            }

            if (dtype == null)
                dtype = np.Int32;

            int N = _dimensions.iDims.Length;

            npy_intp[] shape = new npy_intp[N];
            for (int i = 0; i < N; i++)
                shape[i] = 1;


            List<npy_intp> res_shape = new List<npy_intp>();
            res_shape.Add(N);
            foreach (var dim in _dimensions.iDims)
                res_shape.Add(dim);
            var res = np.empty(new shape(res_shape), dtype: dtype);

            for (int i = 0; i < _dimensions.iDims.Length; i++)
            {
                var dim = _dimensions.iDims[i];

                List<npy_intp> adjustedshape = new List<npy_intp>();
                int j;
                for (j = 0; j < i; j++)
                    adjustedshape.Add(shape[j]);
                adjustedshape.Add(dim);
                for (j = i + 1; j < shape.Length; j++)
                    adjustedshape.Add(shape[j]);


                var indices = np.arange(dim, dtype: dtype).reshape(adjustedshape);
                res[i] = indices;
            }
            return res;

        }

        #endregion

        #region fromfunction

        public static ndarray fromfunction(ndarray a, object source, object destination)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region isscalar
        /// <summary>
        /// Returns True if the type of element is a scalar type.
        /// </summary>
        /// <param name="element">Input argument, can be of any type and shape.</param>
        /// <returns></returns>
        public static bool isscalar(object element)
        {
            if (element.GetType().IsPrimitive)
                return true;

            if (element is Decimal)
                return true;

            if (element is System.Numerics.Complex)
                return true;

            if (element is System.Numerics.BigInteger)
                return true;

            return false;
        }

        #endregion

        #region binary_repr

        public static ndarray binary_repr()
        {
            throw new NotImplementedException();
        }

        #endregion

        #region base_repr

        public static ndarray base_repr()
        {
            throw new NotImplementedException();
        }

        #endregion

        #region identity
        /// <summary>
        /// Return the identity array.
        /// </summary>
        /// <param name="n">Number of rows (and columns) in n x n output</param>
        /// <param name="dtype">(optional) Data-type of the output. Defaults to float</param>
        /// <returns> x n array with its main diagonal set to one, and all other elements 0.</returns>
        public static ndarray identity(int n, dtype dtype = null)
        {
            /*
               Return the identity array.

                The identity array is a square array with ones on
                the main diagonal.

                Parameters
                ----------
                n : int
                    Number of rows (and columns) in `n` x `n` output.
                dtype : data-type, optional
                    Data-type of the output.  Defaults to ``float``.

                Returns
                -------
                out : ndarray
                    `n` x `n` array with its main diagonal set to one,
                    and all other elements 0.

                Examples
                --------
                >>> np.identity(3)
                array([[ 1.,  0.,  0.],
                       [ 0.,  1.,  0.],
                       [ 0.,  0.,  1.]])
            */

            return eye(n, dtype: dtype);
        }
        #endregion

        #region allclose
        /// <summary>
        /// Returns True if two arrays are element - wise equal within a tolerance.
        /// </summary>
        /// <param name="a">Input array to compare.</param>
        /// <param name="b">Input array to compare.</param>
        /// <param name="rtol">The relative tolerance parameter.</param>
        /// <param name="atol">The absolute tolerance parameter.</param>
        /// <param name="equal_nan">Whether to compare NaN's as equal.</param>
        /// <returns></returns>
        public static bool allclose(object a, object b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan=false)
        {
            //  Returns True if two arrays are element - wise equal within a tolerance.
            //
            //  The tolerance values are positive, typically very small numbers.  The
            //  relative difference(`rtol` *abs(`b`)) and the absolute difference
            //  `atol` are added together to compare against the absolute difference
            //  between `a` and `b`.

            //  If either array contains one or more NaNs, False is returned.
            //  Infs are treated as equal if they are in the same place and of the same
            //  sign in both arrays.
            //
            //  Parameters
            //  ----------
            //  a, b: array_like
            //     Input arrays to compare.
            // rtol : float
            //     The relative tolerance parameter(see Notes).
            // atol : float
            //     The absolute tolerance parameter(see Notes).
            // equal_nan : bool
            //     Whether to compare NaN's as equal.  If True, NaN's in `a` will be
            //      considered equal to NaN's in `b` in the output array.

            //      ..versionadded:: 1.10.0

            //  Returns
            //  ------ -
            //  allclose : bool
            //      Returns True if the two arrays are equal within the given
            //      tolerance; False otherwise.

            //  See Also
            //  --------
            //  isclose, all, any, equal

            //  Notes
            //  -----
            //  If the following equation is element - wise True, then allclose returns
            //  True.

            //   absolute(`a` - `b`) <= (`atol` + `rtol` *absolute(`b`))

            //  The above equation is not symmetric in `a` and `b`, so that
            //  ``allclose(a, b)`` might be different from ``allclose(b, a)`` in
            //  some rare cases.

            //  The comparison of `a` and `b` uses standard broadcasting, which
            //  means that `a` and `b` need not have the same shape in order for
            //  ``allclose(a, b)`` to evaluate to True.The same is true for
            //  `equal` but not `array_equal`.


            //Examples
            //--------
            //>>> np.allclose([1e10, 1e-7], [1.00001e10, 1e-8])
            //                  False
            //                  >>> np.allclose([1e10, 1e-8], [1.00001e10,1e-9])
            //  True
            //  >>> np.allclose([1e10, 1e-8], [1.0001e10, 1e-9])
            //  False
            //  >>> np.allclose([1.0, np.nan], [1.0, np.nan])
            //  False
            //  >>> np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)

            ndarray res = all(isclose(a, b, rtol : rtol, atol : atol, equal_nan : equal_nan));
            return Convert.ToBoolean(res.GetItem(0));
        }

        #endregion

        #region isclose
        /// <summary>
        /// Returns a boolean array where two arrays are element - wise equal within a tolerance.
        /// </summary>
        /// <param name="a">Input array to compare</param>
        /// <param name="b">Input array to compare</param>
        /// <param name="rtol">The relative tolerance parameter</param>
        /// <param name="atol">The absolute tolerance parameter</param>
        /// <param name="equal_nan">Whether to compare NaN's as equal.</param>
        /// <returns></returns>
        public static ndarray isclose(object a, object b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false)
        {
            // Returns a boolean array where two arrays are element - wise equal within a
            // tolerance.

            // The tolerance values are positive, typically very small numbers.  The
            // relative difference(`rtol` *abs(`b`)) and the absolute difference
            // `atol` are added together to compare against the absolute difference
            // between `a` and `b`.
            // .. warning::The default `atol` is not appropriate for comparing numbers
            //             that are much smaller than one(see Notes).
            //Parameters
            //----------
            //a, b : array_like
            //    Input arrays to compare.
            //rtol : float
            //    The relative tolerance parameter(see Notes).
            //atol : float
            //    The absolute tolerance parameter(see Notes).
            //equal_nan : bool
            //    Whether to compare NaN's as equal.  If True, NaN's in `a` will be
            //    considered equal to NaN's in `b` in the output array.
            //Returns
            //------ -
            //y : array_like
            //    Returns a boolean array of where `a` and `b` are equal within the
            //    given tolerance.If both `a` and `b` are scalars, returns a single
            //    boolean value.
            //See Also
            //--------
            //allclose
            //Notes
            //---- -
            //..versionadded:: 1.7.0
            //For finite values, isclose uses the following equation to test whether
            //two floating point values are equivalent.
            // absolute(`a` - `b`) <= (`atol` + `rtol` *absolute(`b`))
            //             Unlike the built -in `math.isclose`, the above equation is not symmetric
            //
            // in `a` and `b` --it assumes `b` is the reference value-- so that
            // `isclose(a, b)` might be different from `isclose(b, a)`. Furthermore,
            // the default value of atol is not zero, and is used to determine what
            // small values should be considered close to zero. The default value is
            // appropriate for expected values of order unity: if the expected values
            // are significantly smaller than one, it can result in false positives.
            // `atol` should be carefully selected for the use case at hand.A zero value
            // for `atol` will result in `False` if either `a` or `b` is zero.
            //
            // Examples
            // --------
            // >>> np.isclose([1e10, 1e-7], [1.00001e10,1e-8])
            // array([True, False])
            // >>> np.isclose([1e10, 1e-8], [1.00001e10, 1e-9])
            // array([True, True])
            // >>> np.isclose([1e10, 1e-8], [1.0001e10, 1e-9])
            // array([False, True])
            // >>> np.isclose([1.0, np.nan], [1.0, np.nan])
            // array([True, False])
            // >>> np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
            // array([True, True])
            // >>> np.isclose([1e-8, 1e-7], [0.0, 0.0])
            // array([True, False], dtype= bool)
            // >>> np.isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0)
            // array([False, False], dtype= bool)
            // >>> np.isclose([1e-10, 1e-10], [1e-20, 0.0])
            // array([True, True], dtype= bool)
            // >>> np.isclose([1e-10, 1e-10], [1e-20, 0.999999e-10], atol=0.0)
            // array([False, True], dtype= bool)

            ndarray x = asanyarray(a);
            ndarray y = asanyarray(b);

            dtype dt = np.result_type(y, 1.0);
            y = array(y, dtype: dt, copy: false, subok: true);

            var xfin = isfinite(x);
            var yfin = isfinite(y);
            if ((bool)(all(xfin).GetItem(0)) && (bool)(all(yfin).GetItem(0)))
            {
                return within_tol(x, y, atol, rtol);
            }
            else
            {
                ndarray finite = xfin & yfin;
                ndarray cond = zeros_like(finite, subok: true);
                // Because we're using boolean indexing, x & y must be the same shape.
                // Ideally, we'd just do x, y = broadcast_arrays(x, y). It's in
                // lib.stride_tricks, though, so we can't import it here.
                x = x * ones_like(cond);
                y = y * ones_like(cond);
                // Avoid subtraction with infinite/nan values...
                cond[finite] = within_tol(x.A(finite), y.A(finite), atol, rtol);
                // Check for equality of infinite values...

                cond[~finite] = x.A(~finite).Equals(y.A(~finite));

                if (equal_nan)
                {
                    // Make NaN == NaN
                    ndarray both_nan = isnan(x) & isnan(y);

                    // Needed to treat masked arrays correctly. = True would not work.
                    cond[both_nan] = both_nan[both_nan];
                }
   
                return cond.astype(np.Bool);        // Flatten 0d arrays to scalars
            }

        }

        private static ndarray within_tol(ndarray x, ndarray y, double atol, double rtol)
        {
            try
            {
                return less_equal(absolute(x - y), atol + rtol * absolute(y));
            }
            catch
            {
                throw new Exception("Exception calculating differences between arrays");
            }
        }

        #endregion

        #region array_equal
        /// <summary>
        /// True if two arrays have the same shape and elements, False otherwise.
        /// </summary>
        /// <param name="a1">Input array</param>
        /// <param name="a2">Input array</param>
        /// <returns></returns>
        public static bool array_equal(object a1, object a2)
        {
            // True if two arrays have the same shape and elements, False otherwise.

            // Parameters
            // ----------
            // a1, a2: array_like
            //    Input arrays.

            //Returns
            //------ -
            //b : bool
            //    Returns True if the arrays are equal.

            // See Also
            // --------
            // allclose: Returns True if two arrays are element - wise equal within a
            //           tolerance.
            // array_equiv: Returns True if input arrays are shape consistent and all
            //              elements equal.

            // Examples
            // --------
            // >>> np.array_equal([1, 2], [1, 2])
            // True
            // >>> np.array_equal(np.array([1, 2]), np.array([1, 2]))
            // True
            // >>> np.array_equal([1, 2], [1, 2, 3])
            // False
            // >>> np.array_equal([1, 2], [1, 4])
            // False

            ndarray arr1 = null;
            ndarray arr2 = null;

            try
            {
                arr1 = asanyarray(a1);
                arr2 = asanyarray(a2);
            }
            catch (Exception ex)
            {
                return false;
            }

            if (arr1.shape != arr2.shape)
            {
                return false;
            }

            return np.allb(arr1.Equals(arr2));
        }
        #endregion

        #region array_equiv
        /// <summary>
        /// Returns True if input arrays are shape consistent and all elements equal.
        /// </summary>
        /// <param name="a1">Input array</param>
        /// <param name="a2">Input array</param>
        /// <returns></returns>
        public static bool array_equiv(object a1, object a2)
        {
            // Returns True if input arrays are shape consistent and all elements equal.

            // Shape consistent means they are either the same shape, or one input array
            // can be broadcasted to create the same shape as the other one.

            // Parameters
            // ----------
            // a1, a2: array_like
            //    Input arrays.

            //Returns
            //------ -
            // out : bool
            //     True if equivalent, False otherwise.

            // Examples
            // --------
            // >>> np.array_equiv([1, 2], [1, 2])
            // True
            // >>> np.array_equiv([1, 2], [1, 3])
            // False

            // Showing the shape equivalence:

            // >>> np.array_equiv([1, 2], [[1, 2], [1, 2]])
            // True
            // >>> np.array_equiv([1, 2], [[1, 2, 1, 2], [1, 2, 1, 2]])
            // False

            // >>> np.array_equiv([1, 2], [[1, 2], [1, 3]])
            // False


            ndarray arr1 = null;
            ndarray arr2 = null;

            try
            {
                arr1 = asanyarray(a1);
                arr2 = asanyarray(a2);
            }
            catch (Exception ex)
            {
                return false;
            }

            try
            {
                np.broadcast(arr1, arr2);
            }
            catch
            {
                return false;
            }


            return np.allb(arr1.Equals(arr2));
        }
        #endregion



    }
}