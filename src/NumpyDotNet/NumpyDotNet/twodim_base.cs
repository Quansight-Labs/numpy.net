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
    public static partial class np
    {

        private static dtype _min_int(Int64 low, Int64 high)
        {
            // get smallest int that fits the range
            if (high <= System.SByte.MaxValue && low >= System.SByte.MinValue)
                return np.Int8;

            if (high <= System.Int16.MaxValue && low >= System.Int16.MinValue)
                return np.Int16;

            if (high <= System.Int32.MaxValue && low >= System.Int32.MinValue)
                return np.Int32;

            return np.Int64;
        }

        #region fliplr
        public static ndarray fliplr(ndarray m)
        {
            /*
            Flip array in the left/right direction.

            Flip the entries in each row in the left/right direction.
            Columns are preserved, but appear in a different order than before.

            Parameters
            ----------
            m : array_like
                Input array, must be at least 2-D.

            Returns
            -------
            f : ndarray
                A view of `m` with the columns reversed.  Since a view
                is returned, this operation is :math:`\\mathcal O(1)`.

            See Also
            --------
            flipud : Flip array in the up/down direction.
            rot90 : Rotate array counterclockwise.

            Notes
            -----
            Equivalent to m[:,::-1]. Requires the array to be at least 2-D.

            Examples
            --------
            >>> A = np.diag([1.,2.,3.])
            >>> A
            array([[ 1.,  0.,  0.],
                   [ 0.,  2.,  0.],
                   [ 0.,  0.,  3.]])
            >>> np.fliplr(A)
            array([[ 0.,  0.,  1.],
                   [ 0.,  2.,  0.],
                   [ 3.,  0.,  0.]])

            >>> A = np.random.randn(2,3,5)
            >>> np.all(np.fliplr(A) == A[:,::-1,...])
            True             
            */

            m = asanyarray(m);
            if (m.ndim < 2)
            {
                throw new ValueError("Input must be >= 2-d.");
            }
            return m.A(":", "::-1");
        }
        #endregion

        #region flipud
        public static ndarray flipud(ndarray m)
        {
            /*
            Flip array in the up/down direction.

            Flip the entries in each column in the up/down direction.
            Rows are preserved, but appear in a different order than before.

            Parameters
            ----------
            m : array_like
                Input array.

            Returns
            -------
            out : array_like
                A view of `m` with the rows reversed.  Since a view is
                returned, this operation is :math:`\\mathcal O(1)`.

            See Also
            --------
            fliplr : Flip array in the left/right direction.
            rot90 : Rotate array counterclockwise.

            Notes
            -----
            Equivalent to ``m[::-1,...]``.
            Does not require the array to be two-dimensional.

            Examples
            --------
            >>> A = np.diag([1.0, 2, 3])
            >>> A
            array([[ 1.,  0.,  0.],
                   [ 0.,  2.,  0.],
                   [ 0.,  0.,  3.]])
            >>> np.flipud(A)
            array([[ 0.,  0.,  3.],
                   [ 0.,  2.,  0.],
                   [ 1.,  0.,  0.]])

            >>> A = np.random.randn(2,3,5)
            >>> np.all(np.flipud(A) == A[::-1,...])
            True

            >>> np.flipud([1,2])
            array([2, 1])
            */

            m = asanyarray(m);
            if (m.ndim < 1)
            {
                throw new ValueError("Input must be >= 1-d.");
            }
            return m.A("::-1", "...");
        }
        #endregion

        #region eye
        /// <summary>
        /// Return a 2-D array with ones on the diagonal and zeros elsewhere
        /// </summary>
        /// <param name="N">Number of rows in the output</param>
        /// <param name="M">(optional) Number of columns in the output. If None, defaults to N</param>
        /// <param name="k">(optional) Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal.</param>
        /// <param name="dtype">(optional) Data-type of the returned array</param>
        /// <param name="order">(optional)</param>
        /// <returns>An array where all elements are equal to zero, except for the k-th diagonal, whose values are equal to one</returns>
        public static ndarray eye(int N, int? M = null, int k = 0, dtype dtype = null, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            /*
                Return a 2-D array with ones on the diagonal and zeros elsewhere.

                Parameters
                ----------
                N : int
                  Number of rows in the output.
                M : int, optional
                  Number of columns in the output. If None, defaults to `N`.
                k : int, optional
                  Index of the diagonal: 0 (the default) refers to the main diagonal,
                  a positive value refers to an upper diagonal, and a negative value
                  to a lower diagonal.
                dtype : data-type, optional
                  Data-type of the returned array.
                order : {'C', 'F'}, optional
                    Whether the output should be stored in row-major (C-style) or
                    column-major (Fortran-style) order in memory.

                    .. versionadded:: 1.14.0

                Returns
                -------
                I : ndarray of shape (N,M)
                  An array where all elements are equal to zero, except for the `k`-th
                  diagonal, whose values are equal to one.

                See Also
                --------
                identity : (almost) equivalent function
                diag : diagonal 2-D array from a 1-D array specified by the user.

                Examples
                --------
                >>> np.eye(2, dtype=int)
                array([[1, 0],
                       [0, 1]])
                >>> np.eye(3, k=1)
                array([[ 0.,  1.,  0.],
                       [ 0.,  0.,  1.],
                       [ 0.,  0.,  0.]])

             */

            int i;

            if (M == null)
            {
                M = N;
            }
            ndarray m = zeros(new shape(N, (int)M), dtype: dtype, order: order);
            if (k >= M)
            {
                return m;
            }
            if (k >= 0)
                i = k;
            else
                i = (-k) * (int)M;

            m.A(":" + (M - k).ToString()).Flat[i.ToString() + "::" + (M + 1).ToString()] = 1;
            return m;
        }

        #endregion

        #region diag

        public static ndarray diag(ndarray v, int k=0)
        {
            /*
            Extract a diagonal or construct a diagonal array.

            See the more detailed documentation for ``numpy.diagonal`` if you use this
            function to extract a diagonal and wish to write to the resulting array;
            whether it returns a copy or a view depends on what version of numpy you
            are using.

            Parameters
            ----------
            v : array_like
                If `v` is a 2-D array, return a copy of its `k`-th diagonal.
                If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
                diagonal.
            k : int, optional
                Diagonal in question. The default is 0. Use `k>0` for diagonals
                above the main diagonal, and `k<0` for diagonals below the main
                diagonal.

            Returns
            -------
            out : ndarray
                The extracted diagonal or constructed diagonal array.

            See Also
            --------
            diagonal : Return specified diagonals.
            diagflat : Create a 2-D array with the flattened input as a diagonal.
            trace : Sum along diagonals.
            triu : Upper triangle of an array.
            tril : Lower triangle of an array.

            Examples
            --------
            >>> x = np.arange(9).reshape((3,3))
            >>> x
            array([[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]])

            >>> np.diag(x)
            array([0, 4, 8])
            >>> np.diag(x, k=1)
            array([1, 5])
            >>> np.diag(x, k=-1)
            array([3, 7])

            >>> np.diag(np.diag(x))
            array([[0, 0, 0],
                   [0, 4, 0],
                   [0, 0, 8]])             
            */

            int i;

            v = asanyarray(v);
            var s = v.dims;
            if (len(s) == 1)
            {
                int n = (int)(s[0] + Math.Abs(k));
                var res = zeros(new shape(n, n), dtype: v.Dtype);
                if (k >= 0)
                {
                    i = k;
                }
                else
                {
                    i = (-k) * n;
                }

                res.A(":" + (n - k).ToString()).Flat[i.ToString() + "::" + (n + 1).ToString()] = v;
                return res;
            }
            else if (len(s) == 2)
            {
                return v.diagonal(k);
            }
            else
            {
                throw new ValueError("Input must be 1- or 2-d.");
            }

        }

        #endregion

        #region diagflat

        public static ndarray diagflat(ndarray v, int k = 0)
        {
            /*
            Create a two-dimensional array with the flattened input as a diagonal.

            Parameters
            ----------
            v : array_like
                Input data, which is flattened and set as the `k`-th
                diagonal of the output.
            k : int, optional
                Diagonal to set; 0, the default, corresponds to the "main" diagonal,
                a positive (negative) `k` giving the number of the diagonal above
                (below) the main.

            Returns
            -------
            out : ndarray
                The 2-D output array.

            See Also
            --------
            diag : MATLAB work-alike for 1-D and 2-D arrays.
            diagonal : Return specified diagonals.
            trace : Sum along diagonals.

            Examples
            --------
            >>> np.diagflat([[1,2], [3,4]])
            array([[1, 0, 0, 0],
                   [0, 2, 0, 0],
                   [0, 0, 3, 0],
                   [0, 0, 0, 4]])

            >>> np.diagflat([1,2], 1)
            array([[0, 1, 0],
                   [0, 0, 2],
                   [0, 0, 0]])             
            */

            ndarray fi;

            WrapDelegate wrap = null;

            v = asarray(v).ravel();
            int s = len(v);
            int n = s + Math.Abs(k);
            ndarray res = zeros(new shape(n, n), dtype: v.Dtype);
            if (k >= 0)
            {
                ndarray i = arange(0, n - k, null, null);
                fi = i + k + i * n;
            }
            else
            {
                ndarray i = arange(0, n + k, null, null);
                fi = i + (i - k) * n;
            }
            res.Flat[fi] = v;
            if (wrap == null)
                return res;
            return wrap(res);
        }

        #endregion

        #region tri

        public static ndarray tri(int N, int? M = null, int k = 0, dtype dtype=null)
        {
            /*
            An array with ones at and below the given diagonal and zeros elsewhere.

            Parameters
            ----------
            N : int
                Number of rows in the array.
            M : int, optional
                Number of columns in the array.
                By default, `M` is taken equal to `N`.
            k : int, optional
                The sub-diagonal at and below which the array is filled.
                `k` = 0 is the main diagonal, while `k` < 0 is below it,
                and `k` > 0 is above.  The default is 0.
            dtype : dtype, optional
                Data type of the returned array.  The default is float.

            Returns
            -------
            tri : ndarray of shape (N, M)
                Array with its lower triangle filled with ones and zero elsewhere;
                in other words ``T[i,j] == 1`` for ``i <= j + k``, 0 otherwise.

            Examples
            --------
            >>> np.tri(3, 5, 2, dtype=int)
            array([[1, 1, 1, 0, 0],
                   [1, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1]])

            >>> np.tri(3, 5, -1)
            array([[ 0.,  0.,  0.,  0.,  0.],
                   [ 1.,  0.,  0.,  0.,  0.],
                   [ 1.,  1.,  0.,  0.,  0.]])
             */

            if (dtype == null)
                dtype = np.Float32;

            if (M == null)
                M = N;


            ndarray m = ufunc.outer(UFuncOperation.greater_equal, np.Bool, arange(N, dtype: _min_int(0, N)),
                                                  arange(-k, M-k, dtype: _min_int(-k, (int)M - k)));

            // Avoid making a copy if the requested type is already bool
            m = m.astype(dtype,  copy : false);

            return m;

        }
        #endregion

        #region tril

        public static ndarray tril(ndarray m, int k=0)
        {
            /*
            Lower triangle of an array.

            Return a copy of an array with elements above the `k`-th diagonal zeroed.

            Parameters
            ----------
            m : array_like, shape (M, N)
                Input array.
            k : int, optional
                Diagonal above which to zero elements.  `k = 0` (the default) is the
                main diagonal, `k < 0` is below it and `k > 0` is above.

            Returns
            -------
            tril : ndarray, shape (M, N)
                Lower triangle of `m`, of same shape and data-type as `m`.

            See Also
            --------
            triu : same thing, only for the upper triangle

            Examples
            --------
            >>> np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
            array([[ 0,  0,  0],
                   [ 4,  0,  0],
                   [ 7,  8,  0],
                   [10, 11, 12]])
            */

            m = asanyarray(m);
            ndarray m1 = array(m.dims).A("-2:");
            ndarray mask = tri(N: Convert.ToInt32(m1[0]), M: Convert.ToInt32(m1[1]), k: k, dtype: np.Bool);

            ndarray results = where(mask, m, zeros(new shape(1), dtype: m.Dtype)) as ndarray;
            return results;
        }

        #endregion

        #region triu

        public static ndarray triu(ndarray m, int k = 0)
        {
            /*
            Upper triangle of an array.

            Return a copy of a matrix with the elements below the `k`-th diagonal
            zeroed.

            Please refer to the documentation for `tril` for further details.

            See Also
            --------
            tril : lower triangle of an array

            Examples
            --------
            >>> np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
            array([[ 1,  2,  3],
                   [ 4,  5,  6],
                   [ 0,  8,  9],
                   [ 0,  0, 12]])
            */

            m = asanyarray(m);
            ndarray m1 = array(m.dims).A("-2:");
            ndarray mask = tri(N: Convert.ToInt32(m1[0]), M: Convert.ToInt32(m1[1]), k: k-1, dtype: np.Bool);

            return where(mask, zeros(new shape(1), dtype: m.Dtype), m) as ndarray;

        }

        #endregion

        #region vander

        public static ndarray vander(object x, int? N = null, bool increasing = false)
        {
            /*
            Generate a Vandermonde matrix.

            The columns of the output matrix are powers of the input vector. The
            order of the powers is determined by the `increasing` boolean argument.
            Specifically, when `increasing` is False, the `i`-th output column is
            the input vector raised element-wise to the power of ``N - i - 1``. Such
            a matrix with a geometric progression in each row is named for Alexandre-
            Theophile Vandermonde.

            Parameters
            ----------
            x : array_like
                1-D input array.
            N : int, optional
                Number of columns in the output.  If `N` is not specified, a square
                array is returned (``N = len(x)``).
            increasing : bool, optional
                Order of the powers of the columns.  If True, the powers increase
                from left to right, if False (the default) they are reversed.

                .. versionadded:: 1.9.0

            Returns
            -------
            out : ndarray
                Vandermonde matrix.  If `increasing` is False, the first column is
                ``x^(N-1)``, the second ``x^(N-2)`` and so forth. If `increasing` is
                True, the columns are ``x^0, x^1, ..., x^(N-1)``.

            See Also
            --------
            polynomial.polynomial.polyvander

            Examples
            --------
            >>> x = np.array([1, 2, 3, 5])
            >>> N = 3
            >>> np.vander(x, N)
            array([[ 1,  1,  1],
                   [ 4,  2,  1],
                   [ 9,  3,  1],
                   [25,  5,  1]])

            >>> np.column_stack([x**(N-1-i) for i in range(N)])
            array([[ 1,  1,  1],
                   [ 4,  2,  1],
                   [ 9,  3,  1],
                   [25,  5,  1]])

            >>> x = np.array([1, 2, 3, 5])
            >>> np.vander(x)
            array([[  1,   1,   1,   1],
                   [  8,   4,   2,   1],
                   [ 27,   9,   3,   1],
                   [125,  25,   5,   1]])
            >>> np.vander(x, increasing=True)
            array([[  1,   1,   1,   1],
                   [  1,   2,   4,   8],
                   [  1,   3,   9,  27],
                   [  1,   5,  25, 125]])

            The determinant of a square Vandermonde matrix is the product
            of the differences between the values of the input vector:

            >>> np.linalg.det(np.vander(x))
            48.000000000000043
            >>> (5-3)*(5-2)*(5-1)*(3-2)*(3-1)*(2-1)
            48             
            */


            var array = asanyarray(x);
            if (array.ndim != 1)
            {
                throw new ValueError("x must be a one-dimensional array or sequence.");
            }

            if (N == null)
            {
                N = len(array);
            }

            ndarray tmp = null;
            var v = empty((len(array), N.Value), dtype: promote_types(array.Dtype, np.Int32));
            if (!increasing)
                tmp = v[":", "::-1"] as ndarray;
            else
                tmp = v;

            if (N > 0)
            {
                tmp[":", 0] = 1;
            }
            if (N > 1)
            {
                tmp[":", "1:"] = array[":", null];
                ufunc.accumulate(UFuncOperation.multiply, tmp[":", "1:"] as ndarray, @out : tmp[":", "1:"] as ndarray, axis : 1);
            }


            return v;

        }


   
        #endregion

 

        #region mask_indices

        public delegate ndarray mask_indices_delegate(ndarray m, int k = 0);


        public static ndarray[] mask_indices(int n, mask_indices_delegate mask_func, int k=0)
        {
            /*
            Return the indices to access (n, n) arrays, given a masking function.

            Assume `mask_func` is a function that, for a square array a of size
            ``(n, n)`` with a possible offset argument `k`, when called as
            ``mask_func(a, k)`` returns a new array with zeros in certain locations
            (functions like `triu` or `tril` do precisely this). Then this function
            returns the indices where the non-zero values would be located.

            Parameters
            ----------
            n : int
                The returned indices will be valid to access arrays of shape (n, n).
            mask_func : callable
                A function whose call signature is similar to that of `triu`, `tril`.
                That is, ``mask_func(x, k)`` returns a boolean array, shaped like `x`.
                `k` is an optional argument to the function.
            k : scalar
                An optional argument which is passed through to `mask_func`. Functions
                like `triu`, `tril` take a second argument that is interpreted as an
                offset.

            Returns
            -------
            indices : tuple of arrays.
                The `n` arrays of indices corresponding to the locations where
                ``mask_func(np.ones((n, n)), k)`` is True.

            See Also
            --------
            triu, tril, triu_indices, tril_indices

            Notes
            -----
            .. versionadded:: 1.4.0

            Examples
            --------
            These are the indices that would allow you to access the upper triangular
            part of any 3x3 array:

            >>> iu = np.mask_indices(3, np.triu)

            For example, if `a` is a 3x3 array:

            >>> a = np.arange(9).reshape(3, 3)
            >>> a
            array([[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]])
            >>> a[iu]
            array([0, 1, 2, 4, 5, 8])

            An offset can be passed also to the masking function.  This gets us the
            indices starting on the first diagonal right of the main one:

            >>> iu1 = np.mask_indices(3, np.triu, 1)

            with which we now extract only three elements:

            >>> a[iu1]
            array([1, 2, 5])             
            */

            ndarray m = ones(new shape(n, n), dtype: np.Int32);
            ndarray a = mask_func(m, k);
            return nonzero(a != 0);
        }


        #endregion

        #region tril_indices

        public static ndarray[] tril_indices(int n, int k=0, int? m = null)
        {
            /*
            Return the indices for the lower-triangle of an (n, m) array.

            Parameters
            ----------
            n : int
                The row dimension of the arrays for which the returned
                indices will be valid.
            k : int, optional
                Diagonal offset (see `tril` for details).
            m : int, optional
                .. versionadded:: 1.9.0

                The column dimension of the arrays for which the returned
                arrays will be valid.
                By default `m` is taken equal to `n`.


            Returns
            -------
            inds : tuple of arrays
                The indices for the triangle. The returned tuple contains two arrays,
                each with the indices along one dimension of the array.

            See also
            --------
            triu_indices : similar function, for upper-triangular.
            mask_indices : generic function accepting an arbitrary mask function.
            tril, triu

            Notes
            -----
            .. versionadded:: 1.4.0

            Examples
            --------
            Compute two different sets of indices to access 4x4 arrays, one for the
            lower triangular part starting at the main diagonal, and one starting two
            diagonals further right:

            >>> il1 = np.tril_indices(4)
            >>> il2 = np.tril_indices(4, 2)

            Here is how they can be used with a sample array:

            >>> a = np.arange(16).reshape(4, 4)
            >>> a
            array([[ 0,  1,  2,  3],
                   [ 4,  5,  6,  7],
                   [ 8,  9, 10, 11],
                   [12, 13, 14, 15]])

            Both for indexing:

            >>> a[il1]
            array([ 0,  4,  5,  8,  9, 10, 12, 13, 14, 15])

            And for assigning values:

            >>> a[il1] = -1
            >>> a
            array([[-1,  1,  2,  3],
                   [-1, -1,  6,  7],
                   [-1, -1, -1, 11],
                   [-1, -1, -1, -1]])

            These cover almost the whole array (two diagonals right of the main one):

            >>> a[il2] = -10
            >>> a
            array([[-10, -10, -10,   3],
                   [-10, -10, -10, -10],
                   [-10, -10, -10, -10],
                   [-10, -10, -10, -10]])
             */

            return nonzero(tri(n, m, k : k, dtype : np.Bool));
        }

        #endregion

        #region tril_indices_from

        public static ndarray[] tril_indices_from(ndarray arr, int k = 0)
        {
            /*
            Return the indices for the lower-triangle of arr.

            See `tril_indices` for full details.

            Parameters
            ----------
            arr : array_like
                The indices will be valid for square arrays whose dimensions are
                the same as arr.
            k : int, optional
                Diagonal offset (see `tril` for details).

            See Also
            --------
            tril_indices, tril             
            */

            if (arr.ndim != 2)
            {
                throw new ValueError("input array must be 2-d");
            }
            ndarray m1 = array(arr.dims).A("-2");
            ndarray m2 = array(arr.dims).A("-1");

            return tril_indices(n: Convert.ToInt32(m1[0]), k: k, m: Convert.ToInt32(m2[0]));

        }

        #endregion

        #region triu_indices

        public static ndarray[] triu_indices(int n, int k=0, int? m = null)
        {
            /*
            Return the indices for the upper-triangle of an (n, m) array.

            Parameters
            ----------
            n : int
                The size of the arrays for which the returned indices will
                be valid.
            k : int, optional
                Diagonal offset (see `triu` for details).
            m : int, optional
                .. versionadded:: 1.9.0

                The column dimension of the arrays for which the returned
                arrays will be valid.
                By default `m` is taken equal to `n`.


            Returns
            -------
            inds : tuple, shape(2) of ndarrays, shape(`n`)
                The indices for the triangle. The returned tuple contains two arrays,
                each with the indices along one dimension of the array.  Can be used
                to slice a ndarray of shape(`n`, `n`).

            See also
            --------
            tril_indices : similar function, for lower-triangular.
            mask_indices : generic function accepting an arbitrary mask function.
            triu, tril

            Notes
            -----
            .. versionadded:: 1.4.0

            Examples
            --------
            Compute two different sets of indices to access 4x4 arrays, one for the
            upper triangular part starting at the main diagonal, and one starting two
            diagonals further right:

            >>> iu1 = np.triu_indices(4)
            >>> iu2 = np.triu_indices(4, 2)

            Here is how they can be used with a sample array:

            >>> a = np.arange(16).reshape(4, 4)
            >>> a
            array([[ 0,  1,  2,  3],
                   [ 4,  5,  6,  7],
                   [ 8,  9, 10, 11],
                   [12, 13, 14, 15]])

            Both for indexing:

            >>> a[iu1]
            array([ 0,  1,  2,  3,  5,  6,  7, 10, 11, 15])

            And for assigning values:

            >>> a[iu1] = -1
            >>> a
            array([[-1, -1, -1, -1],
                   [ 4, -1, -1, -1],
                   [ 8,  9, -1, -1],
                   [12, 13, 14, -1]])

            These cover only a small part of the whole array (two diagonals right
            of the main one):

            >>> a[iu2] = -10
            >>> a
            array([[ -1,  -1, -10, -10],
                   [  4,  -1,  -1, -10],
                   [  8,   9,  -1,  -1],
                   [ 12,  13,  14,  -1]])
            */

            return nonzero(~tri(n, m, k:k - 1, dtype: np.Bool));
        }

        #endregion

        #region triu_indices_from

        public static ndarray[] triu_indices_from(ndarray arr, int k = 0)
        {
            /*
            Return the indices for the upper-triangle of arr.

            See `triu_indices` for full details.

            Parameters
            ----------
            arr : ndarray, shape(N, N)
                The indices will be valid for square arrays.
            k : int, optional
                Diagonal offset (see `triu` for details).

            Returns
            -------
            triu_indices_from : tuple, shape(2) of ndarray, shape(N)
                Indices for the upper-triangle of `arr`.

            See Also
            --------
            triu_indices, triu
            */

            if (arr.ndim != 2)
            {
                throw new ValueError("input array must be 2-d");
            }

            ndarray m1 = array(arr.dims).A("-2");
            ndarray m2 = array(arr.dims).A("-1");

            return triu_indices(Convert.ToInt32(m1[0]), k: k, m: Convert.ToInt32(m2[0]));
        }

        #endregion

 
    }
}
