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

        #region rot90
        /// <summary>
        /// Rotate an array by 90 degrees in the plane specified by axes.
        /// </summary>
        /// <param name="m">Array of two or more dimensions.</param>
        /// <param name="k">Number of times the array is rotated by 90 degrees.</param>
        /// <param name="axes">The array is rotated in the plane defined by the axes. Axes must be different.</param>
        /// <returns></returns>
        public static ndarray rot90(ndarray m, int k = 1, int[] axes = null)
        {
            /*
            Rotate an array by 90 degrees in the plane specified by axes.

            Rotation direction is from the first towards the second axis.

            Parameters
            ----------
            m : array_like
                Array of two or more dimensions.
            k : integer
                Number of times the array is rotated by 90 degrees.
            axes: (2,) array_like
                The array is rotated in the plane defined by the axes.
                Axes must be different.

                .. versionadded:: 1.12.0

            Returns
            -------
            y : ndarray
                A rotated view of `m`.

            See Also
            --------
            flip : Reverse the order of elements in an array along the given axis.
            fliplr : Flip an array horizontally.
            flipud : Flip an array vertically.

            Notes
            -----
            rot90(m, k=1, axes=(1,0)) is the reverse of rot90(m, k=1, axes=(0,1))
            rot90(m, k=1, axes=(1,0)) is equivalent to rot90(m, k=-1, axes=(0,1))

            Examples
            --------
            >>> m = np.array([[1,2],[3,4]], int)
            >>> m
            array([[1, 2],
                   [3, 4]])
            >>> np.rot90(m)
            array([[2, 4],
                   [1, 3]])
            >>> np.rot90(m, 2)
            array([[4, 3],
                   [2, 1]])
            >>> m = np.arange(8).reshape((2,2,2))
            >>> np.rot90(m, 1, (1,2))
            array([[[1, 3],
                    [0, 2]],

                  [[5, 7],
                   [4, 6]]])

            */

            if (axes == null)
            {
                axes = new int[] { 0, 1 };
            }

            //axes = tuple(axes);
            if (len(axes) != 2)
            {
                throw new ValueError("len(axes) must be 2.");
            }

            m = asanyarray(m);

            if ((axes[0] == axes[1]) || Math.Abs(axes[0] - axes[1]) == m.ndim)
            {
                throw new ValueError("Axes must be different.");
            }

            if (axes[0] >= m.ndim || axes[0] < -m.ndim
                || axes[1] >= m.ndim || axes[1] < -m.ndim)
            {
                throw new ValueError(string.Format("Axes={0},{1} out of range for array of ndim={2}.", axes[0], axes[1], m.ndim));
            }

            k %= 4;

            if (k == 0)
            {
                return m.A(":");
            }
            if (k == 2)
            {
                return flip(flip(m, axes[0]), axes[1]);
            }

            npy_intp[] axes_list = PythonFunction.range(0, m.ndim);
            npy_intp temp = axes_list[axes[0]];
            axes_list[axes[0]] = axes_list[axes[1]];
            axes_list[axes[1]] = temp;

            if (k == 1)
            {
                return transpose(flip(m, axes[1]), axes_list);
            }
            else
            {
                //k == 3
                return flip(transpose(m, axes_list), axes[1]);
            }

        }
        #endregion

        #region flip
        /// <summary>
        /// Reverse the order of elements in an array along the given axis.
        /// </summary>
        /// <param name="m">Input array.</param>
        /// <param name="axis">Axis in array, which entries are reversed.</param>
        /// <returns></returns>
        public static ndarray flip(ndarray m, int axis)
        {
            /*
            Reverse the order of elements in an array along the given axis.

            The shape of the array is preserved, but the elements are reordered.

            .. versionadded:: 1.12.0

            Parameters
            ----------
            m : array_like
                Input array.
            axis : integer
                Axis in array, which entries are reversed.


            Returns
            -------
            out : array_like
                A view of `m` with the entries of axis reversed.  Since a view is
                returned, this operation is done in constant time.

            See Also
            --------
            flipud : Flip an array vertically (axis=0).
            fliplr : Flip an array horizontally (axis=1).

            Notes
            -----
            flip(m, 0) is equivalent to flipud(m).
            flip(m, 1) is equivalent to fliplr(m).
            flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.

            Examples
            --------
            >>> A = np.arange(8).reshape((2,2,2))
            >>> A
            array([[[0, 1],
                    [2, 3]],

                   [[4, 5],
                    [6, 7]]])

            >>> flip(A, 0)
            array([[[4, 5],
                    [6, 7]],

                   [[0, 1],
                    [2, 3]]])

            >>> flip(A, 1)
            array([[[2, 3],
                    [0, 1]],

                   [[6, 7],
                    [4, 5]]])

            >>> A = np.random.randn(3,4,5)
            >>> np.all(flip(A,2) == A[:,:,::-1,...])
            True
            */

            if (!hasattr(m, "ndim"))
            {
                m = asarray(m);
            }
            object[] indexer = BuildSliceArray(new Slice(null), m.ndim);

            try
            {
                indexer[axis] = new Slice(null, null, -1);
            }
            catch (Exception ex)
            {
                throw new ValueError(
                    string.Format("axis={0} is invalid for the {1}-dimensional input array",
                    axis, m.ndim));
            }

            return m.A(indexer);
        }
        #endregion

        #region iterable
        /// <summary>
        /// Check whether or not an object can be iterated over.
        /// </summary>
        /// <param name="o"></param>
        /// <returns></returns>
        public static bool iterable(object o)
        {
            /*
             Check whether or not an object can be iterated over.

            Parameters
            ----------
            y : object
              Input object.

            Returns
            -------
            b : bool
              Return ``True`` if the object has an iterator method or is a
              sequence and ``False`` otherwise.


            Examples
            --------
            >>> np.iterable([1, 2, 3])
            True
            >>> np.iterable(2)
            False            
             */
            if (o is System.Collections.IEnumerable)
            {
                return true;
            }


            return false;
        }
        #endregion

        #region asarray_chkfinite
        /// <summary>
        /// onvert the input to an array, checking for NaNs or Infs.
        /// </summary>
        /// <param name="a">Input data, in any form that can be converted to an array</param>
        /// <param name="dtype">data-type, optional</param>
        /// <param name="order">{'C', 'F'}, optional</param>
        /// <returns></returns>
        public static ndarray asarray_chkfinite(ndarray a, dtype dtype = null, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            /*
            Convert the input to an array, checking for NaNs or Infs.

            Parameters
            ----------
            a : array_like
                Input data, in any form that can be converted to an array.  This
                includes lists, lists of tuples, tuples, tuples of tuples, tuples
                of lists and ndarrays.  Success requires no NaNs or Infs.
            dtype : data-type, optional
                By default, the data-type is inferred from the input data.
            order : {'C', 'F'}, optional
                 Whether to use row-major (C-style) or
                 column-major (Fortran-style) memory representation.
                 Defaults to 'C'.

            Returns
            -------
            out : ndarray
                Array interpretation of `a`.  No copy is performed if the input
                is already an ndarray.  If `a` is a subclass of ndarray, a base
                class ndarray is returned.

            Raises
            ------
            ValueError
                Raises ValueError if `a` contains NaN (Not a Number) or Inf (Infinity).

            See Also
            --------
            asarray : Create and array.
            asanyarray : Similar function which passes through subclasses.
            ascontiguousarray : Convert input to a contiguous array.
            asfarray : Convert input to a floating point ndarray.
            asfortranarray : Convert input to an ndarray with column-major
                             memory order.
            fromiter : Create an array from an iterator.
            fromfunction : Construct an array by executing a function on grid
                           positions.

            Examples
            --------
            Convert a list into an array.  If all elements are finite
            ``asarray_chkfinite`` is identical to ``asarray``.

            >>> a = [1, 2]
            >>> np.asarray_chkfinite(a, dtype=float)
            array([1., 2.])

            Raises ValueError if array_like contains Nans or Infs.

            >>> a = [1, 2, np.inf]
            >>> try:
            ...     np.asarray_chkfinite(a)
            ... except ValueError:
            ...     print('ValueError')
            ...
            ValueError
             */

            throw new NotImplementedException();
        }
        #endregion

        #region piecewise
        /// <summary>
        /// Evaluate a piecewise-defined function.
        /// </summary>
        /// <param name="x">The input domain.</param>
        /// <param name="condlist">list of bool arrays or bool scalars</param>
        /// <param name="funclist">list of callables, f(x,*args,**kw), or scalars</param>
        /// <param name="args">Any further arguments given to `piecewise` are passed to the functions upon execution</param>
        /// <param name="kw">Keyword arguments used in calling `piecewise` are passed to the functions upon execution</param>
        /// <returns></returns>
        public static ndarray piecewise(ndarray x, bool[] condlist, object[] funclist, string[] args, string[] kw)
        {
            /*
            Evaluate a piecewise-defined function.

            Given a set of conditions and corresponding functions, evaluate each
            function on the input data wherever its condition is true.

            Parameters
            ----------
            x : ndarray or scalar
                The input domain.
            condlist : list of bool arrays or bool scalars
                Each boolean array corresponds to a function in `funclist`.  Wherever
                `condlist[i]` is True, `funclist[i](x)` is used as the output value.

                Each boolean array in `condlist` selects a piece of `x`,
                and should therefore be of the same shape as `x`.

                The length of `condlist` must correspond to that of `funclist`.
                If one extra function is given, i.e. if
                ``len(funclist) == len(condlist) + 1``, then that extra function
                is the default value, used wherever all conditions are false.
            funclist : list of callables, f(x,*args,**kw), or scalars
                Each function is evaluated over `x` wherever its corresponding
                condition is True.  It should take a 1d array as input and give an 1d
                array or a scalar value as output.  If, instead of a callable,
                a scalar is provided then a constant function (``lambda x: scalar``) is
                assumed.
            args : tuple, optional
                Any further arguments given to `piecewise` are passed to the functions
                upon execution, i.e., if called ``piecewise(..., ..., 1, 'a')``, then
                each function is called as ``f(x, 1, 'a')``.
            kw : dict, optional
                Keyword arguments used in calling `piecewise` are passed to the
                functions upon execution, i.e., if called
                ``piecewise(..., ..., alpha=1)``, then each function is called as
                ``f(x, alpha=1)``.

            Returns
            -------
            out : ndarray
                The output is the same shape and type as x and is found by
                calling the functions in `funclist` on the appropriate portions of `x`,
                as defined by the boolean arrays in `condlist`.  Portions not covered
                by any condition have a default value of 0.


            See Also
            --------
            choose, select, where

            Notes
            -----
            This is similar to choose or select, except that functions are
            evaluated on elements of `x` that satisfy the corresponding condition from
            `condlist`.

            The result is::

                    |--
                    |funclist[0](x[condlist[0]])
              out = |funclist[1](x[condlist[1]])
                    |...
                    |funclist[n2](x[condlist[n2]])
                    |--

            Examples
            --------
            Define the sigma function, which is -1 for ``x < 0`` and +1 for ``x >= 0``.

            >>> x = np.linspace(-2.5, 2.5, 6)
            >>> np.piecewise(x, [x < 0, x >= 0], [-1, 1])
            array([-1., -1., -1.,  1.,  1.,  1.])

            Define the absolute value, which is ``-x`` for ``x <0`` and ``x`` for
            ``x >= 0``.

            >>> np.piecewise(x, [x < 0, x >= 0], [lambda x: -x, lambda x: x])
            array([ 2.5,  1.5,  0.5,  0.5,  1.5,  2.5])

            Apply the same function to a scalar value.

            >>> y = -2
            >>> np.piecewise(y, [y < 0, y >= 0], [lambda x: -x, lambda x: x])
            array(2)

         */

            throw new NotImplementedException();
        }
        #endregion

        #region select
        /// <summary>
        /// Return an array drawn from elements in choicelist, depending on conditions.
        /// </summary>
        /// <param name="condlist">The list of conditions which determine from which array in `choicelist` the output elements are taken</param>
        /// <param name="choicelist">The list of arrays from which the output elements are taken. It has to be of the same length as `condlist`.</param>
        /// <param name="_default">The element inserted in `output` when all conditions evaluate to False.</param>
        /// <returns></returns>
        public static ndarray select(ndarray[] condlist, ndarray[] choicelist, int _default = 0)
        {
            /*
            Return an array drawn from elements in choicelist, depending on conditions.

            Parameters
            ----------
            condlist : list of bool ndarrays
                The list of conditions which determine from which array in `choicelist`
                the output elements are taken. When multiple conditions are satisfied,
                the first one encountered in `condlist` is used.
            choicelist : list of ndarrays
                The list of arrays from which the output elements are taken. It has
                to be of the same length as `condlist`.
            default : scalar, optional
                The element inserted in `output` when all conditions evaluate to False.

            Returns
            -------
            output : ndarray
                The output at position m is the m-th element of the array in
                `choicelist` where the m-th element of the corresponding array in
                `condlist` is True.

            See Also
            --------
            where : Return elements from one of two arrays depending on condition.
            take, choose, compress, diag, diagonal

            Examples
            --------
            >>> x = np.arange(10)
            >>> condlist = [x<3, x>5]
            >>> choicelist = [x, x**2]
            >>> np.select(condlist, choicelist)
            array([ 0,  1,  2,  0,  0,  0, 36, 49, 64, 81])
             
            */

            var n = condlist.Length;
            var n2 = choicelist.Length;

            if (n2 != n)
            {
                throw new ValueError("list of cases must be same length as list of conditions");
            }

            int condlistcount = condlist.Count(t => t.TypeNum == NPY_TYPES.NPY_BOOL);
            if (condlistcount != condlist.Count())
            {
                throw new ValueError("condlist must be all boolean arrays");
            }

            int choicelistcount = choicelist.Count(t => t.TypeNum == choicelist[0].TypeNum);
            if (choicelistcount != choicelist.Count())
            {
                throw new ValueError("choicelist must be all of the same data type");
            }

            var ExpandedChoiceList = new ndarray[choicelist.Length + 1];
            ExpandedChoiceList[0] = array(new int[] { 0 }).astype(choicelist[0].Dtype);
            Array.Copy(choicelist, 0, ExpandedChoiceList, 1, choicelist.Length);
            choicelist = ExpandedChoiceList;

            ndarray S = array(new int[] { 0 });
            ndarray pfac = array(new int[] { 1 });
            for (int k = 1; k < n + 1; k++)
            {
                S += k * pfac * asarray(condlist[k - 1]);
                if (k < n)
                {
                    pfac *= (1 - asarray(condlist[k - 1]));
                }

            }

            // handle special case of a 1-element condition but
            // a multi-element choice
            if (S.size == 1 || ConvertToIntp(np._max(asanyarray(S.shape.iDims))) == 1)
            {
                pfac = asarray(1);
                for (int k = 0; k < n2 + 1; k++)
                {
                    pfac = pfac + asarray(choicelist[k]);
                }
                if (S.size == 1)
                {
                    S = S * ones(asarray(pfac).shape, S.Dtype);
                }
                else
                {
                    S = S * ones(asarray(pfac).shape, S.Dtype);
                }
            }

            return choose(S, choicelist);
        }
        #endregion

        #region copy
        /// <summary>
        /// Return an array copy of the given object.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="order">{'C', 'F', 'A', 'K'}, optional</param>
        /// <returns></returns>
        public static ndarray copy(ndarray a, NPY_ORDER order = NPY_ORDER.NPY_KORDER)
        {
            /*
           Return an array copy of the given object.

            Parameters
            ----------
            a : array_like
                Input data.
            order : {'C', 'F', 'A', 'K'}, optional
                Controls the memory layout of the copy. 'C' means C-order,
                'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
                'C' otherwise. 'K' means match the layout of `a` as closely
                as possible. (Note that this function and :meth:`ndarray.copy` are very
                similar, but have different default values for their order=
                arguments.)

            Returns
            -------
            arr : ndarray
                Array interpretation of `a`.

            Notes
            -----
            This is equivalent to:

            >>> np.array(a, copy=True)  #doctest: +SKIP

            Examples
            --------
            Create an array x, with a reference y and a copy z:

            >>> x = np.array([1, 2, 3])
            >>> y = x
            >>> z = np.copy(x)

            Note that, when we modify x, y changes, but not z:

            >>> x[0] = 10
            >>> x[0] == y[0]
            True
            >>> x[0] == z[0]
            False
            */

            return a.Copy(order);
        }
        #endregion

        #region gradient
        /// <summary>
        /// Return the gradient of an N-dimensional array.
        /// </summary>
        /// <param name="f">An N-dimensional array containing samples of a scalar function.</param>
        /// <param name="varargs">Spacing between f values. </param>
        /// <param name="axes">Gradient is calculated only along the given axis or axes</param>
        /// <param name="edge_order">{1, 2}, optional</param>
        /// <returns></returns>
        public static ndarray[] gradient(object f, IList<object> varargs = null, int? axes = null, int edge_order = 1)
        {
            return gradient(f, varargs, axes != null ? new int[] { axes.Value } : null, edge_order);
        }
        /// <summary>
        /// Return the gradient of an N-dimensional array.
        /// </summary>
        /// <param name="f">An N-dimensional array containing samples of a scalar function.</param>
        /// <param name="varargs">Spacing between f values. </param>
        /// <param name="axes">Gradient is calculated only along the given axis or axes</param>
        /// <param name="edge_order">{1, 2}, optional</param>
        /// <returns></returns>
        public static ndarray[] gradient(object f, IList<object> varargs, IList<int> axes, int edge_order = 1)
        {
            var f1 = np.asanyarray(f);
            var N = f1.ndim;      // number of dimensions

            if (axes == null)
            {
                List<int> tempAxes = new List<int>();
                for (int i = 0; i < N; i++)
                    tempAxes.Add(i);
                axes = tempAxes.ToArray();
            }
            else
            {
                axes = normalize_axis_tuple(axes, N);
            }

            var len_axes = axes.Count();
            var n = varargs != null ? varargs.Count() : 0;

            List<object> dx = new List<object>();
            if (n == 0)
            {
                // no spacing argument - use 1 in all axes
                for (int i = 0; i < len_axes; i++)
                    dx.Add(1.0f);
            }
            else if (n == 1 && asanyarray(varargs[0]).IsAScalar)
            {
                // single scalar for all axes
                for (int i = 0; i < len_axes; i++)
                    dx.Add(varargs[0]);
            }
            else if (n == len_axes)
            {
                dx.AddRange(varargs);

                for (int i = 0; i < dx.Count; i++)
                {
                    var distances = asanyarray(dx[i]);
                    if (distances.IsAScalar)
                    {
                        continue;
                    }
                    else if (distances.ndim != 1)
                    {
                        throw new ValueError("distances must be either scalars or 1d");
                    }

                    if (len(distances) != f1.dims[axes[i]])
                    {
                        throw new ValueError("when 1d, distances must match the length of the corresponding dimension");
                    }

                    var diffx = np.diff(distances);
                    // if distances are constant reduce to the scalar case
                    // since it brings a consistent speedup


                    bool AllSame = true;
                    object diff0 = diffx[0];
                    foreach (var dd in diffx)
                    {
                        if (diff0.ToString() != dd.ToString())
                        {
                            AllSame = false;
                            break;
                        }
                    }
                    if (AllSame)
                    {
                        dx[i] = diff0;
                    }
                    else
                    {
                        dx[i] = diffx;
                    }
                }
            }
            else
            {
                throw new TypeError("invalid number of arguments");
            }


            if (edge_order > 2)
            {
                throw new ValueError("'edge_order' greater than 2 not supported");
            }

            // use central differences on interior and one-sided differences on the
            // endpoints. This preserves second order-accuracy over the full domain.

            List<ndarray> outvals = new List<ndarray>();

            // create slice objects --- initially all are [:, :, ..., :]

            var slice1 = BuildSliceArray(new Slice(null), N);
            var slice2 = BuildSliceArray(new Slice(null), N);
            var slice3 = BuildSliceArray(new Slice(null), N);
            var slice4 = BuildSliceArray(new Slice(null), N);

            dtype otype = f1.Dtype;
    
            if (f1.IsInexact)
            {
                ;
            }
            else
            {
                //all other types convert to floating point
                otype = np.Float64;
            }

            f1 = f1.astype(otype);

            for (int i = 0; i < axes.Count; i++)
            {
                int axis = axes[i];
                object ax_dx = dx[i];

                if (f1.dims[axis] < edge_order + 1)
                {
                    throw new ValueError(
                        "Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required.");
                }

                // result allocation
                var _out = np.empty_like(f, dtype: otype);

                // spacing for the current axis
                bool uniform_spacing = asanyarray(ax_dx).IsAScalar;

                // Numerical differentiation: 2nd order interior
                slice1[axis] = new Slice(1, -1);
                slice2[axis] = new Slice(null, -2);
                slice3[axis] = new Slice(1, -1);
                slice4[axis] = new Slice(2, null);

                if (uniform_spacing)
                {
                    _out[slice1] = (f1.A(slice4) - f1.A(slice2)) / (2.0 * asanyarray(ax_dx));
                }
                else
                {
                    ndarray ax_dxx = asanyarray(ax_dx);

                    ndarray dx1 = ax_dxx.A("0:-1");
                    ndarray dx2 = ax_dxx.A("1:");
                    ndarray a = (ndarray)(-(dx2) / (dx1 * (dx1 + dx2)));
                    ndarray b = (ndarray)((dx2 - dx1) / (dx1 * dx2));
                    ndarray c = (ndarray)(dx1 / (dx2 * (dx1 + dx2)));
                    // fix the shape for broadcasting
                    var shape = np.ones(N, dtype : np.Int32);
                    shape[axis] = -1;

                    //a = a.reshape(new shape(shape));
                   //a.shape = b.shape = c.shape = shape;
                    // 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
                    _out[slice1] = (a * f1.A(slice2)) + (b * f1.A(slice3)) + (c * f1.A(slice4));
                }

                ndarray dx_0;
                ndarray dx_n;

                // Numerical differentiation: 1st order edges
                if (edge_order == 1)
                {
                    slice1[axis] = new Slice(0, 1);
                    slice2[axis] = new Slice(1, 2);
                    slice3[axis] = new Slice(0, 1);

                    if (uniform_spacing)
                    {
                        dx_0 = asanyarray(ax_dx);
                    }
                    else
                    {
                        dx_0 = asanyarray(asanyarray(ax_dx)[0]);
                    }

                    // 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
                    _out[slice1] = (f1.A(slice2) - f1.A(slice3)) / dx_0;

                    slice1[axis] = new Slice(-1,-2,-1);
                    slice2[axis] = new Slice(-1,-2,-1);
                    slice3[axis] = new Slice(-2,-3,-1);

                    if (uniform_spacing)
                    {
                        dx_n = asanyarray(ax_dx);
                    }
                    else
                    {
                        dx_n = asanyarray(asanyarray(ax_dx)[-1]);
                    }
                    // 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
                    _out[slice1] = (f1.A(slice2) - f1.A(slice3)) / dx_n;
                }
                else // Numerical differentiation: 2nd order edges
                {
                    ndarray a, b, c;


                    slice1[axis] = new Slice(0, 1);
                    slice2[axis] = new Slice(0, 1);
                    slice3[axis] = new Slice(1, 2);
                    slice4[axis] = new Slice(2, 3);
                    if (uniform_spacing)
                    {
                        double ax_dxd = Convert.ToDouble(ax_dx);
                        a = asanyarray(-1.5 / ax_dxd);
                        b = asanyarray(2.0 / ax_dxd);
                        c = asanyarray(-0.5 / ax_dxd);
                    }
                    else
                    {
                        ndarray ax_dxx = asanyarray(ax_dx);

                        var dx1 = asanyarray(ax_dxx[0]);
                        var dx2 = asanyarray(ax_dxx[1]);
                        a = (ndarray)(-(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2)));
                        b = (ndarray)((dx1 + dx2) / (dx1 * dx2));
                        c = (ndarray)(-dx1 / (dx2 * (dx1 + dx2)));
                    }

                    // 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
                    _out[slice1] = (a * f1.A(slice2)) + (b * f1.A(slice3)) + (c * f1.A(slice4));

                    slice1[axis] = new Slice(-1, -2, -1);
                    slice2[axis] = new Slice(-3, -4, -1);
                    slice3[axis] = new Slice(-2, -3, -1);
                    slice4[axis] = new Slice(-1, -2, -1);
                    if (uniform_spacing)
                    {
                        double ax_dxd = Convert.ToDouble(ax_dx);

                        a = asanyarray(0.5 / ax_dxd);
                        b = asanyarray(-2.0 / ax_dxd);
                        c = asanyarray(1.5 / ax_dxd);
                    }
                    else
                    {
                        ndarray ax_dxx = asanyarray(ax_dx);

                        var dx1 = asanyarray(ax_dxx[-2]);
                        var dx2 = asanyarray(ax_dxx[-1]);
                        a = (ndarray)((dx2) / (dx1 * (dx1 + dx2)));
                        b = (ndarray)(-(dx2 + dx1) / (dx1 * dx2));
                        c = (ndarray)((2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2)));
                    }

                    // 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
                    _out[slice1] = (a * f1.A(slice2)) + (b * f1.A(slice3)) + (c * f1.A(slice4));
                }

                outvals.Add(_out);

                // reset the slice object in this dimension to ":"
                slice1[axis] = new Slice(null);
                slice2[axis] = new Slice(null);
                slice3[axis] = new Slice(null);
                slice4[axis] = new Slice(null);
            }

            if (len_axes == 1)
                return new ndarray[] { outvals[0] };
            else
                return outvals.ToArray();
        }
        #endregion

        #region diff
        /// <summary>
        /// Calculate the n - th discrete difference along the given axis.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <param name="n">The number of times values are differenced.If zero, the input is returned as-is.</param>
        /// <param name="axis">The axis along which the difference is taken, default is the last axis.</param>
        /// <returns></returns>
        public static ndarray diff(object a, int n = 1, int axis = -1)
        {
            //    Calculate the n - th discrete difference along the given axis.

            //    The first difference is given by ``out[n] = a[n + 1] - a[n]`` along
            //    the given axis, higher differences are calculated by using `diff`
            //    recursively.

            //    Parameters
            //    ----------
            //    a : array_like
            //        Input array
            //    n : int, optional
            //        The number of times values are differenced.If zero, the input
            //        is returned as-is.
            //    axis : int, optional
            //        The axis along which the difference is taken, default is the
            //        last axis.

            //    Returns
            //    -------
            //    diff : ndarray
            //        The n-th differences. The shape of the output is the same as `a`
            //        except along `axis` where the dimension is smaller by `n`. The
            //        type of the output is the same as the type of the difference
            //        between any two elements of `a`. This is the same as the type of
            //        `a` in most cases. A notable exception is `datetime64`, which
            //        results in a `timedelta64` output array.

            //    See Also
            //    --------
            //    gradient, ediff1d, cumsum

            //    Notes
            //    -----
            //    Type is preserved for boolean arrays, so the result will contain
            //    `False` when consecutive elements are the same and `True` when they
            //    differ.

            //    For unsigned integer arrays, the results will also be unsigned. This
            //    should not be surprising, as the result is consistent with
            //    calculating the difference directly:

            //    >>> u8_arr = np.array([1, 0], dtype= np.uint8)
            //    >>> np.diff(u8_arr)
            //    array([255], dtype= uint8)
            //    >>> u8_arr[1,...] - u8_arr[0,...]
            //    array(255, np.uint8)

            //    If this is not desirable, then the array should be cast to a larger
            //    integer type first:

            //    >>> i16_arr = u8_arr.astype(np.int16)
            //    >>> np.diff(i16_arr)
            //    array([-1], dtype= int16)

            //    Examples
            //    --------
            //    >>> x = np.array([1, 2, 4, 7, 0])
            //    >>> np.diff(x)
            //    array([ 1,  2,  3, -7])
            //    >>> np.diff(x, n=2)
            //    array([  1,   1, -10])

            //    >>> x = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
            //    >>> np.diff(x)
            //    array([[2, 3, 4],
            //           [5, 1, 2]])
            //    >>> np.diff(x, axis=0)
            //    array([[-1,  2,  0, -2]])

            //    >>> x = np.arange('1066-10-13', '1066-10-16', dtype=np.datetime64)
            //    >>> np.diff(x)
            //    array([1, 1], dtype='timedelta64[D]')

            var srcArray = asanyarray(a);

            if (n == 0)
            {
                return srcArray;
            }
            if (n < 0)
            {
                throw new Exception(string.Format("order must be non-negative but got {0}", n));
            }

            ndarray x = srcArray;
            ndarray y = srcArray;

            int nd = srcArray.ndim;

            axis = normalize_axis_index(axis, nd);

            object[] slice1 = BuildSliceArray(new Slice(null), nd);
            object[] slice2 = BuildSliceArray(new Slice(null), nd);

            slice1[axis] = new Slice(1, null);
            slice2[axis] = new Slice(null, -1);

            x = srcArray.A(slice1);
            y = srcArray.A(slice2);

            if (srcArray.TypeNum == NPY_TYPES.NPY_BOOL)
            {
                ndarray diff = x.NotEquals(y);
                return diff;
            }
            else
            {
                ndarray diff = x - y;
                return diff;
            }
        }

        #endregion

        #region interp
   
        /// <summary>
        /// One-dimensional linear interpolation.
        /// </summary>
        /// <param name="x">The x-coordinates of the interpolated values.</param>
        /// <param name="xp">1-D sequence of floats, the x-coordinates of the data points</param>
        /// <param name="fp">1-D sequence of float or complex, the y-coordinates of the data points</param>
        /// <param name="left"> optional float or complex corresponding to fp value to return for "x LT xp[0]"</param>
        /// <param name="right">optional float or complex corresponding to fp value to return for "x GT xp[-1]"</param>
        /// <param name="period">A period for the x-coordinates. This parameter allows the proper interpolation of angular x-coordinates.</param>
        /// <returns></returns>
        public static ndarray interp(object x, object xp, object fp, double? left = null, double? right = null, double? period = null)
        {
            /*
            One-dimensional linear interpolation.

            Returns the one-dimensional piecewise linear interpolant to a function
            with given values at discrete data-points.

            Parameters
            ----------
            x : array_like
                The x-coordinates of the interpolated values.

            xp : 1-D sequence of floats
                The x-coordinates of the data points, must be increasing if argument
                `period` is not specified. Otherwise, `xp` is internally sorted after
                normalizing the periodic boundaries with ``xp = xp % period``.

            fp : 1-D sequence of float or complex
                The y-coordinates of the data points, same length as `xp`.

            left : optional float or complex corresponding to fp
                Value to return for `x < xp[0]`, default is `fp[0]`.

            right : optional float or complex corresponding to fp
                Value to return for `x > xp[-1]`, default is `fp[-1]`.

            period : None or float, optional
                A period for the x-coordinates. This parameter allows the proper
                interpolation of angular x-coordinates. Parameters `left` and `right`
                are ignored if `period` is specified.

                .. versionadded:: 1.10.0

            Returns
            -------
            y : float or complex (corresponding to fp) or ndarray
                The interpolated values, same shape as `x`.

            Raises
            ------
            ValueError
                If `xp` and `fp` have different length
                If `xp` or `fp` are not 1-D sequences
                If `period == 0`

            Notes
            -----
            Does not check that the x-coordinate sequence `xp` is increasing.
            If `xp` is not increasing, the results are nonsense.
            A simple check for increasing is::

                np.all(np.diff(xp) > 0)

            Examples
            --------
            >>> xp = [1, 2, 3]
            >>> fp = [3, 2, 0]
            >>> np.interp(2.5, xp, fp)
            1.0
            >>> np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
            array([ 3. ,  3. ,  2.5 ,  0.56,  0. ])
            >>> UNDEF = -99.0
            >>> np.interp(3.14, xp, fp, right=UNDEF)
            -99.0

            Plot an interpolant to the sine function:

            >>> x = np.linspace(0, 2*np.pi, 10)
            >>> y = np.sin(x)
            >>> xvals = np.linspace(0, 2*np.pi, 50)
            >>> yinterp = np.interp(xvals, x, y)
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(x, y, 'o')
            [<matplotlib.lines.Line2D object at 0x...>]
            >>> plt.plot(xvals, yinterp, '-x')
            [<matplotlib.lines.Line2D object at 0x...>]
            >>> plt.show()

            Interpolation with periodic x-coordinates:

            >>> x = [-180, -170, -185, 185, -10, -5, 0, 365]
            >>> xp = [190, -190, 350, -350]
            >>> fp = [5, 10, 3, 4]
            >>> np.interp(x, xp, fp, period=360)
            array([7.5, 5., 8.75, 6.25, 3., 3.25, 3.5, 3.75])

            Complex interpolation:

            >>> x = [1.5, 4.0]
            >>> xp = [2,3,5]
            >>> fp = [1.0j, 0, 2+3j]
            >>> np.interp(x, xp, fp)
            array([ 0.+1.j ,  1.+1.5j])
             */
            var dzz = asarray(x) ?? throw new ArgumentNullException("x");
            var dxx = asarray(xp) ?? throw new ArgumentNullException("xp");
            var dyy = asarray(fp) ?? throw new ArgumentNullException("fp");
            if (dzz.ndim > 1 || dxx.ndim != 1 || dyy.ndim != 1) throw new ArgumentException("x, xp, fp must be 1-D array");
            if (dzz.size == 0 || dxx.size == 0 || dyy.size == 0) throw new ArgumentException("x, xp, fp can't be empty array");
            if (dxx.size != dyy.size) throw new ArgumentException("xp and fp must have equal size");
            if (period != null)
            {
                if (period == 0) throw new ArgumentException("period must be a non-zero value");
                period = period < 0 ? -period : period;
                dzz %= period;
                dxx %= period;
                var asort_xp = np.argsort(dxx);
                dxx = dxx[asort_xp] as ndarray;
                dyy = dyy[asort_xp] as ndarray;
                dxx = np.concatenate((dxx["-1:"] as ndarray - period, dxx, dxx["0:1"] as ndarray + period));
                dyy = np.concatenate((dyy["-1:"], dyy, dyy["0:1"]));
            }

            if (dzz.IsComplex || dxx.IsComplex || dyy.IsComplex)
            {
                throw new Exception("This function does not currently support complex numbers");
            }
            
            return interp_func(x: dzz, xp: dxx, fp: dyy, left: left, right: right);
        }

        private static ndarray interp_func(ndarray x, ndarray xp, ndarray fp, double? left = null, double? right = null)
        {
            var dx = xp.AsDoubleArray();
            var dy = fp.AsDoubleArray();
            var dz = x.AsDoubleArray();
            int lenxp = dx.Length; int lenx = dz.Length;
            var lval = left is null ? dy[0] : (double)left;
            var rval = right is null ? dy[lenxp - 1] : (double)right;
            var dres = new double[lenx];
            double[] slopes = null;

            if (lenxp == 1)
            {
                var xp_val = dx[0];
                var fp_val = dy[0];
                for (int i = 0; i < lenx; i++)
                {
                    var x_val = dz[i];
                    dres[i] = (x_val < xp_val) ? lval : ((x_val > xp_val) ? rval : fp_val);
                }
            }
            else
            {
                int j = 0;
                if (lenxp <= lenx)
                {
                    slopes = new double[lenxp - 1];
                    for (int i = 0; i < lenxp - 1; i++)
                        slopes[i] = (dy[i + 1] - dy[i]) / (dx[i + 1] - dx[i]);
                }

                for (int i = 0; i < lenx; i++)
                {
                    var x_val = dz[i];
                    if (x_val == double.NaN)
                    {
                        dres[i] = x_val;
                        continue;
                    }

                    j = binary_search_with_guess(x_val, dx, j);
                    if (j == -1) dres[i] = lval;
                    else if (j == lenxp) dres[i] = rval;
                    else if (j == lenxp - 1) dres[i] = dy[j];
                    else if (dx[j] == x_val) dres[i] = dy[j];
                    else
                    {
                        var slope = (slopes != null) ? slopes[j] : (dy[j + 1] - dy[j]) / (dx[j + 1] - dx[j]);
                        /* If we get nan in one direction, try the other */
                        dres[i] = slope * (x_val - dx[j]) + dy[j];
                        /* NPY_UNLIKELY() seems to be used to tell the compiler 
                         * the condition not likely to go, ignored here
                         * 2023.4.1
                         */
                        if (double.IsNaN(dres[i]))
                        {
                            dres[i] = slope * (x_val - dx[j + 1]) + dy[j + 1];
                            if (double.IsNaN(dres[i]))
                                dres[i] = dy[j];
                        }
                    }
                }
            }
            return asarray(dres, np.Float64);
        }

        private static int binary_search_with_guess(double key, double[] arr, int guess)
        {
            int imin = 0, imax = arr.Length, len = arr.Length;
            int LIKELY_IN_CACHE_SIZE = 8;
            if (key > arr[len - 1]) return len;
            if (key < arr[0]) return -1;
            /*
             * If len <= 4 use linear search.
             * From above we know key >= arr[0] when we start.
             */
            if (len <= 4)
            {
                int i = 1;
                for (; i < len && key >= arr[i]; i++) ;
                return i - 1;
            }
            guess = guess > len - 3 ? len - 3 : ((guess < 1) ? 1 : guess);

            /* check most likely values: guess - 1, guess, guess + 1 */
            if (key < arr[guess])
            {
                if (key < arr[guess - 1])
                {
                    imax = guess - 1;
                    /* last attempt to restrict search to items in cache */
                    if (guess > LIKELY_IN_CACHE_SIZE && key >= arr[guess - LIKELY_IN_CACHE_SIZE])
                        imin = guess - LIKELY_IN_CACHE_SIZE;
                }
                else
                    /* key >= arr[guess - 1] */
                    return guess - 1;
            }
            else
            {
                /* key >= arr[guess] */
                if (key < arr[guess + 1]) return guess;
                /* key >= arr[guess + 1] */
                if (key < arr[guess + 2]) return guess + 1;
                /* key >= arr[guess + 2] */
                imin = guess + 2;
                /* last attempt to restrict search to items in cache */
                if (guess < len - LIKELY_IN_CACHE_SIZE - 1 && key < arr[guess + LIKELY_IN_CACHE_SIZE])
                    imax = guess + LIKELY_IN_CACHE_SIZE;
            }

            /* finally, find index by bisection */
            while (imin < imax)
            {
                int imid = imin + ((imax - imin) >> 1);
                if (key >= arr[imid]) imin = imid + 1;
                else imax = imid;
            }
            return imin - 1;
        }
        #endregion

        #region angle
        /// <summary>
        /// Return the angle of the complex argument.
        /// </summary>
        /// <param name="z"> A complex number or sequence of complex numbers.</param>
        /// <param name="deg">Return angle in degrees if True, radians if False (default).</param>
        /// <returns></returns>
        public static ndarray angle(object z, bool deg = false)
        {
            /*
            Return the angle of the complex argument.

            Parameters
            ----------
            z : array_like
                A complex number or sequence of complex numbers.
            deg : bool, optional
                Return angle in degrees if True, radians if False (default).

            Returns
            -------
            angle : ndarray or scalar
                The counterclockwise angle from the positive real axis on
                the complex plane, with dtype as numpy.float64.

            See Also
            --------
            arctan2
            absolute

            Examples
            --------
            >>> np.angle([1.0, 1.0j, 1+1j])               # in radians
            array([ 0.        ,  1.57079633,  0.78539816])
            >>> np.angle(1+1j, deg=True)                  # in degrees
            45.0             
            */


            double fact;

            if (deg)
            {
                fact = 180 / Math.PI;
            }
            else
            {
                fact = 1.0;
            }
            var za = asanyarray(z);

            ndarray zimag = null;
            ndarray zreal = null;

            if (za.IsComplex)
            {
                zimag = za.Imag;
                zreal = za.Real;
            }
            else
            {
                za = za.astype(np.Float64);
                zimag = np.zeros_like(za);
                zreal = za;
            }
            return arctan2(zimag, zreal) * fact;
        }


        #endregion

        #region unwrap
        /// <summary>
        /// Unwrap by changing deltas between values to 2*pi complement.
        /// </summary>
        /// <param name="p">Input array.</param>
        /// <param name="discont">Maximum discontinuity between values, default is "pi"</param>
        /// <param name="axis">Axis along which unwrap will operate, default is the last axis.</param>
        /// <returns></returns>
        public static ndarray unwrap(object p, double discont = Math.PI, int axis = -1)
        {
            /*
            Unwrap by changing deltas between values to 2*pi complement.

            Unwrap radian phase `p` by changing absolute jumps greater than
            `discont` to their 2*pi complement along the given axis.

            Parameters
            ----------
            p : array_like
                Input array.
            discont : float, optional
                Maximum discontinuity between values, default is ``pi``.
            axis : int, optional
                Axis along which unwrap will operate, default is the last axis.

            Returns
            -------
            out : ndarray
                Output array.

            See Also
            --------
            rad2deg, deg2rad

            Notes
            -----
            If the discontinuity in `p` is smaller than ``pi``, but larger than
            `discont`, no unwrapping is done because taking the 2*pi complement
            would only make the discontinuity larger.

            Examples
            --------
            >>> phase = np.linspace(0, np.pi, num=5)
            >>> phase[3:] += np.pi
            >>> phase
            array([ 0.        ,  0.78539816,  1.57079633,  5.49778714,  6.28318531])
            >>> np.unwrap(phase)
            array([ 0.        ,  0.78539816,  1.57079633, -0.78539816,  0.        ])
            */

     

            var arrp = asanyarray(p);
            var nd = arrp.ndim;

            if (axis < 0)
                axis += nd;

            var dd = diff(arrp, axis: axis);
            var slice1 = BuildSliceArray(new Slice(null, null), nd);    // full slices
            slice1[axis] = new Slice(1, null);
            var ddmod = mod(dd + Math.PI, 2 * Math.PI) - Math.PI;
            copyto(ddmod, Math.PI, where : (ddmod == -Math.PI) & (dd > 0));
            var ph_correct = ddmod - dd.astype(ddmod.Dtype);
            copyto(ph_correct, 0, where: absolute(dd) < discont);


            var dtype_typenum = numpyAPI.GetArrayHandler(arrp.TypeNum).MathOpReturnType(UFuncOperation.multiply);
            dtype dtype = new dtype(numpyAPI.NpyArray_DescrFromType(dtype_typenum));
  
            var up = array(p, copy: true, dtype: dtype);

            up[slice1] = arrp.A(slice1) + ph_correct.cumsum(axis);
            return up;
        }
        #endregion

        #region sort_complex

        /*
            Sort a complex array using the real part first, then the imaginary part.

            Parameters
            ----------
            o : array_like
                Input array

            Returns
            -------
            out : complex ndarray
                Always returns a sorted complex array.

            Examples
            --------
            >>> np.sort_complex([5, 3, 6, 2, 1])
            array([ 1.+0.j,  2.+0.j,  3.+0.j,  5.+0.j,  6.+0.j])

            >>> np.sort_complex([1 + 2j, 2 - 1j, 3 - 2j, 3 - 3j, 3 + 5j])
            array([ 1.+2.j,  2.-1.j,  3.-3.j,  3.-2.j,  3.+5.j])
        */

        /// <summary>
        /// Sort a complex array using the real part first, then the imaginary part.
        /// </summary>
        /// <param name="oa">Input array</param>
        /// <returns></returns>
        public static ndarray sort_complex(object oa)
        {
            var a = asanyarray(oa);
            var b = np.array(a, copy: true);
            b = b.Sort();

            if (!b.IsComplex)
            {
                b = b.astype(np.Complex);
            }
            return b;
        }

        #endregion

        #region trim_zero
        /// <summary>
        /// Trim the leading and/or trailing zeros from a 1-D array or sequence.
        /// </summary>
        /// <param name="filt">1-D array or sequence Input array.</param>
        /// <param name="trim">A string with 'f' representing trim from front and 'b' to trim from back.</param>
        /// <returns></returns>
        public static ndarray trim_zeros(ndarray filt, string trim = "fb")
        {
            /*
            Trim the leading and/or trailing zeros from a 1-D array or sequence.

            Parameters
            ----------
            filt : 1-D array or sequence
                Input array.
            trim : str, optional
                A string with 'f' representing trim from front and 'b' to trim from
                back. Default is 'fb', trim zeros from both front and back of the
                array.

            Returns
            -------
            trimmed : 1-D array or sequence
                The result of trimming the input. The input data type is preserved.

            Examples
            --------
            >>> a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
            >>> np.trim_zeros(a)
            array([1, 2, 3, 0, 2, 1])

            >>> np.trim_zeros(a, 'b')
            array([0, 0, 0, 1, 2, 3, 0, 2, 1])

            The input data type is preserved, list/tuple in means list/tuple out.

            >>> np.trim_zeros([0, 1, 2, 0])
            [1, 2]             
            */

            int first = 0;
            trim = trim.ToUpper();
            if (trim.Contains("F"))
            {
                var ArrayHandler = DefaultArrayHandlers.GetArrayHandler(filt.TypeNum);

                foreach (var ii in filt)
                {
                    if (ArrayHandler.NonZero(filt.Array.data, first))
                        break;
                    else
                        first = first + 1;
                }

            }

            int last = len(filt);
            if (trim.Contains("B"))
            {
                ndarray tempFilt = filt.A("::-1").ravel();
                foreach (var ii in tempFilt)
                {
                    var ArrayHandler = DefaultArrayHandlers.GetArrayHandler(filt.TypeNum);

                    if (ArrayHandler.NonZero(filt.Array.data, last-1))
                        break;
                    else
                        last = last - 1;
                }

            }
            return filt.A(first.ToString() + ":" + last.ToString());
        }

        #endregion

        #region extract
        /// <summary>
        /// Return the elements of an array that satisfy some condition.
        /// </summary>
        /// <param name="condition">An array whose nonzero or True entries indicate the elements of 'arr' to extract.</param>
        /// <param name="arr">Input array of the same size as 'condition'</param>
        /// <returns></returns>
        public static ndarray extract(ndarray condition, ndarray arr)
        {
            /*
            Return the elements of an array that satisfy some condition.

            This is equivalent to ``np.compress(ravel(condition), ravel(arr))``.  If
            `condition` is boolean ``np.extract`` is equivalent to ``arr[condition]``.

            Note that `place` does the exact opposite of `extract`.

            Parameters
            ----------
            condition : array_like
                An array whose nonzero or True entries indicate the elements of `arr`
                to extract.
            arr : array_like
                Input array of the same size as `condition`.

            Returns
            -------
            extract : ndarray
                Rank 1 array of values from `arr` where `condition` is True.

            See Also
            --------
            take, put, copyto, compress, place

            Examples
            --------
            >>> arr = np.arange(12).reshape((3, 4))
            >>> arr
            array([[ 0,  1,  2,  3],
                   [ 4,  5,  6,  7],
                   [ 8,  9, 10, 11]])
            >>> condition = np.mod(arr, 3)==0
            >>> condition
            array([[ True, False, False,  True],
                   [False, False,  True, False],
                   [False,  True, False, False]])
            >>> np.extract(condition, arr)
            array([0, 3, 6, 9])


            If `condition` is boolean:

            >>> arr[condition]
            array([0, 3, 6, 9])             
            */

            return take(ravel(arr), nonzero(ravel(condition))[0]);
        }
        #endregion

        #region place
        /// <summary>
        /// Change elements of an array based on conditional and input values.
        /// </summary>
        /// <param name="arr">Array to put data into.</param>
        /// <param name="mask"> Boolean mask array. Must have the same size as 'arr'.</param>
        /// <param name="vals">Values to put into 'arr'. Only the first N elements are used, where N is the number of True values in 'mask'</param>
        public static void place(object arr, object mask, object vals)
        {
            /*
            Change elements of an array based on conditional and input values.

            Similar to ``np.copyto(arr, vals, where=mask)``, the difference is that
            `place` uses the first N elements of `vals`, where N is the number of
            True values in `mask`, while `copyto` uses the elements where `mask`
            is True.

            Note that `extract` does the exact opposite of `place`.

            Parameters
            ----------
            arr : ndarray
                Array to put data into.
            mask : array_like
                Boolean mask array. Must have the same size as `a`.
            vals : 1-D sequence
                Values to put into `a`. Only the first N elements are used, where
                N is the number of True values in `mask`. If `vals` is smaller
                than N, it will be repeated, and if elements of `a` are to be masked,
                this sequence must be non-empty.

            See Also
            --------
            copyto, put, take, extract

            Examples
            --------
            >>> arr = np.arange(6).reshape(2, 3)
            >>> np.place(arr, arr>2, [44, 55])
            >>> arr
            array([[ 0,  1,  2],
                   [44, 55, 44]])             
            */

            ndarray array = asanyarray(arr);
            if (array == null)
                throw new Exception("arr must not be null");

            ndarray maskarray = asanyarray(mask);
            if (maskarray == null)
                throw new Exception("mask must not be null");

            ndarray valsarray = asanyarray(vals);
            if (valsarray == null)
                throw new Exception("vals must not be null");


            NpyCoreApi.Place(array, maskarray, valsarray);
     
            return;
        }
        #endregion

        #region disp
        public static void disp(string mesg, object device = null, bool linefeed = true)
        {
            if (linefeed)
                Console.WriteLine(mesg);
            else
                Console.Write(mesg);
            return;
        }

        #endregion

        #region cov
        /// <summary>
        /// Estimate a covariance matrix, given data and weights.
        /// </summary>
        /// <param name="m">A 1-D or 2-D array containing multiple variables and observations.</param>
        /// <param name="y">An additional set of variables and observations.</param>
        /// <param name="rowvar">If `rowvar` is True (default), then each row represents a variable, with observations in the columns.</param>
        /// <param name="bias">Default normalization (False) is by '(N - 1)', where 'N' is the number of observations given(unbiased estimate)</param>
        /// <param name="ddof">If not null the default value implied by 'bias' is overridden.</param>
        /// <param name="fweights">1-D array of integer frequency weights</param>
        /// <param name="aweights">1-D array of observation vector weights.</param>
        /// <returns></returns>
        public static ndarray cov(object m, object y = null, bool rowvar = true,
            bool bias = false, int? ddof = null, object fweights = null, object aweights = null)
        {
            /*
            Estimate a covariance matrix, given data and weights.

            Covariance indicates the level to which two variables vary together.
            If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
            then the covariance matrix element :math:`C_{ij}` is the covariance of
            :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
            of :math:`x_i`.

            See the notes for an outline of the algorithm.

            Parameters
            ----------
            m : array_like
                A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables. Also see `rowvar` below.
            y : array_like, optional
                An additional set of variables and observations. `y` has the same form
                as that of `m`.
            rowvar : bool, optional
                If `rowvar` is True (default), then each row represents a
                variable, with observations in the columns. Otherwise, the relationship
                is transposed: each column represents a variable, while the rows
                contain observations.
            bias : bool, optional
                Default normalization (False) is by ``(N - 1)``, where ``N`` is the
                number of observations given (unbiased estimate). If `bias` is True,
                then normalization is by ``N``. These values can be overridden by using
                the keyword ``ddof`` in numpy versions >= 1.5.
            ddof : int, optional
                If not ``None`` the default value implied by `bias` is overridden.
                Note that ``ddof=1`` will return the unbiased estimate, even if both
                `fweights` and `aweights` are specified, and ``ddof=0`` will return
                the simple average. See the notes for the details. The default value
                is ``None``.

                .. versionadded:: 1.5
            fweights : array_like, int, optional
                1-D array of integer frequency weights; the number of times each
                observation vector should be repeated.

                .. versionadded:: 1.10
            aweights : array_like, optional
                1-D array of observation vector weights. These relative weights are
                typically large for observations considered "important" and smaller for
                observations considered less "important". If ``ddof=0`` the array of
                weights can be used to assign probabilities to observation vectors.

                .. versionadded:: 1.10

            Returns
            -------
            out : ndarray
                The covariance matrix of the variables.

            See Also
            --------
            corrcoef : Normalized covariance matrix

            Notes
            -----
            Assume that the observations are in the columns of the observation
            array `m` and let ``f = fweights`` and ``a = aweights`` for brevity. The
            steps to compute the weighted covariance are as follows::

                >>> w = f * a
                >>> v1 = np.sum(w)
                >>> v2 = np.sum(w * a)
                >>> m -= np.sum(m * w, axis=1, keepdims=True) / v1
                >>> cov = np.dot(m * w, m.T) * v1 / (v1**2 - ddof * v2)

            Note that when ``a == 1``, the normalization factor
            ``v1 / (v1**2 - ddof * v2)`` goes over to ``1 / (np.sum(f) - ddof)``
            as it should.

            Examples
            --------
            Consider two variables, :math:`x_0` and :math:`x_1`, which
            correlate perfectly, but in opposite directions:

            >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T
            >>> x
            array([[0, 1, 2],
                   [2, 1, 0]])

            Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
            matrix shows this clearly:

            >>> np.cov(x)
            array([[ 1., -1.],
                   [-1.,  1.]])

            Note that element :math:`C_{0,1}`, which shows the correlation between
            :math:`x_0` and :math:`x_1`, is negative.

            Further, note how `x` and `y` are combined:

            >>> x = [-2.1, -1,  4.3]
            >>> y = [3,  1.1,  0.12]
            >>> X = np.stack((x, y), axis=0)
            >>> print(np.cov(X))
            [[ 11.71        -4.286     ]
             [ -4.286        2.14413333]]
            >>> print(np.cov(x, y))
            [[ 11.71        -4.286     ]
             [ -4.286        2.14413333]]
            >>> print(np.cov(x))
            11.71             */

            dtype dtype = null;
            ndarray yarr = null;

            if (ddof != null && ddof != Convert.ToInt32(ddof))
            {
                throw new ValueError("ddof must be integer");
            }


            // Handles complex arrays too
            var marr = np.asarray(m);
            if (marr.ndim > 2)
            {
                throw new ValueError("m has more than 2 dimensions");
            }

            if (y == null)
            {
                dtype = np.result_type(marr, marr, np.Float64);
            }
            else
            {
                yarr = np.asarray(y);
                if (yarr.ndim > 2)
                {
                    throw new ValueError("y has more than 2 dimensions");
                }
                dtype = np.result_type(marr.TypeNum);
            }

            var X = array(marr, ndmin : 2, dtype : dtype);
            if (!rowvar && X.shape.iDims[0] != 1)
            {
                X = X.T;
            }
            if (X.shape.iDims[0] == 0)
            {
                return np.array(new int[] { }).reshape(0, 0);
            }

            if (yarr != null)
            {
                yarr = array(yarr, copy: false, ndmin: 2, dtype: dtype);
                if (!rowvar && yarr.shape.iDims[0] != 1)
                    yarr = yarr.T;
                X = np.concatenate(X, yarr, axis : 0);
            }

            if (ddof == null)
            {
                if (bias == false)
                    ddof = 1;
                else
                    ddof = 0;
            }
    
            ndarray w = null;
            ndarray afweights = null;
            ndarray aaweights = null;

            if (fweights != null)
            {
                afweights = np.asarray(fweights, dtype: np.Float64);
                if ((bool)np.all(afweights.Equals(np.around(afweights))).GetItem(0) == false)
                {
                    throw new TypeError("fweights must be integer");
                }
 
                if (afweights.ndim > 1)
                {
                    throw new RuntimeError("cannot handle multidimensional fweights");
                }

                if (afweights.shape.iDims[0] != X.shape.iDims[1])
                {
                    throw new RuntimeError("incompatible numbers of samples and fweights");

                }
                if (anyb(afweights < 0))
                {
                    throw new ValueError("fweights cannot be negative");
                }

                w = afweights;
            }

            if (aweights != null)
            {
                aaweights = np.asarray(aweights, dtype : np.Float64);
                if (aaweights.ndim > 1)
                {
                    throw new RuntimeError("cannot handle multidimensional aweights");
                }
   
                if (aaweights.shape.iDims[0] != X.shape.iDims[1])
                {
                    throw new RuntimeError("incompatible numbers of samples and aweights");

                }
                if (np.anyb(aaweights < 0))
                {
                    throw new ValueError("aweights cannot be negative");
                }

                if (w == null)
                    w = aaweights;
                else
                    w *= aaweights;
            }

            var avg_result = average(X, axis: 1, weights: w, returned: true);
            double w_sum = ConvertToDouble(avg_result.sum_of_weights[0]);

            // Determine the normalization
            double fact;
            if (w == null)
            {
                fact = X.shape.iDims[1] - ddof.Value;
            }
            else if (ddof == 0)
            {
                fact = w_sum;
            }
            else if (aaweights == null)
            {
                fact = w_sum - ddof.Value;
            }
            else
            {
                fact = w_sum - ddof.Value * Convert.ToDouble(np.sum(w * aaweights).GetItem(0)) / w_sum;
            }

            if (fact <= 0)
            {
                //warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel = 2);
                fact = 0.0;
            }


            ndarray X_T = null;
            X -= avg_result.retval.A(":", null);
            if (w == null)
            {
                X_T = X.T;
            }
            else
            {
                X_T = (X * w).T;
            }
            var c = np.dot(X, np.conj(X_T));

            if (c.IsDecimal)
            {
                c *= 1.0m / Convert.ToDecimal(fact);
            }
            else if (c.IsComplex)
            {
                c *= (System.Numerics.Complex)1.0 / new System.Numerics.Complex(fact, 0);
            }
            else
            {
                c *= 1.0 / fact;
            }
            return c.Squeeze();

        }

        private static double ConvertToDouble(object value)
        {
            if (value is System.Numerics.Complex)
            {
                System.Numerics.Complex c = (System.Numerics.Complex)value;
                return c.Real;
            }
            if (value is System.Numerics.BigInteger)
            {
                System.Numerics.BigInteger c = (System.Numerics.BigInteger)value;
                return (double)c;
            }

            return Convert.ToDouble(value);
        }
        #endregion

        #region corrcoef

        /// <summary>
        /// Return Pearson product-moment correlation coefficients.
        /// </summary>
        /// <param name="x">A 1-D or 2-D array containing multiple variables and observations.</param>
        /// <param name="y">An additional set of variables and observations.</param>
        /// <param name="rowvar">If `rowvar` is True (default), then each row represents a variable, with observations in the columns.</param>
        /// <returns></returns>
        public static ndarray corrcoef(object x, object y = null, bool rowvar = true)
        {
            /*
            Return Pearson product-moment correlation coefficients.

            Please refer to the documentation for `cov` for more detail.  The
            relationship between the correlation coefficient matrix, `R`, and the
            covariance matrix, `C`, is

            .. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} * C_{jj} } }

            The values of `R` are between -1 and 1, inclusive.

            Parameters
            ----------
            x : array_like
                A 1-D or 2-D array containing multiple variables and observations.
                Each row of `x` represents a variable, and each column a single
                observation of all those variables. Also see `rowvar` below.
            y : array_like, optional
                An additional set of variables and observations. `y` has the same
                shape as `x`.
            rowvar : bool, optional
                If `rowvar` is True (default), then each row represents a
                variable, with observations in the columns. Otherwise, the relationship
                is transposed: each column represents a variable, while the rows
                contain observations.
            bias : _NoValue, optional
                Has no effect, do not use.

                .. deprecated:: 1.10.0
            ddof : _NoValue, optional
                Has no effect, do not use.

                .. deprecated:: 1.10.0

            Returns
            -------
            R : ndarray
                The correlation coefficient matrix of the variables.

            See Also
            --------
            cov : Covariance matrix

            Notes
            -----
            Due to floating point rounding the resulting array may not be Hermitian,
            the diagonal elements may not be 1, and the elements may not satisfy the
            inequality abs(a) <= 1. The real and imaginary parts are clipped to the
            interval [-1,  1] in an attempt to improve on that situation but is not
            much help in the complex case.

            This function accepts but discards arguments `bias` and `ddof`.  This is
            for backwards compatibility with previous versions of this function.  These
            arguments had no effect on the return values of the function and can be
            safely ignored in this and previous versions of numpy.
            */

            ndarray d = null;

            var c = np.cov(x, y, rowvar);
            try
            {
                d = diag(c);
            }
            catch (Exception ex)
            {
                // scalar covariance
                // nan if incorrect value (nan, inf, 0), 1 otherwise
                return np.divide(c , c);
            }

            var stddev = np.sqrt(d);

            //var s1 = stddev[":", null, null, null];
            //var s2 = stddev[null, ":"];

            c = np.divide(c, stddev.A(":", null));
            c = np.divide(c, stddev.A(null, ":"));

            return c;
        }

        #endregion

        #region blackman
        /// <summary>
        /// Return the Blackman window.
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <returns></returns>
        public static ndarray blackman(int M)
        {
            /*
            Return the Blackman window.

            The Blackman window is a taper formed by using the first three
            terms of a summation of cosines. It was designed to have close to the
            minimal leakage possible.  It is close to optimal, only slightly worse
            than a Kaiser window.

            Parameters
            ----------
            M : int
                Number of points in the output window. If zero or less, an empty
                array is returned.

            Returns
            -------
            out : ndarray
                The window, with the maximum value normalized to one (the value one
                appears only if the number of samples is odd).

            See Also
            --------
            bartlett, hamming, hanning, kaiser

            Notes
            -----
            The Blackman window is defined as

            .. math::  w(n) = 0.42 - 0.5 \\cos(2\\pi n/M) + 0.08 \\cos(4\\pi n/M)

            Most references to the Blackman window come from the signal processing
            literature, where it is used as one of many windowing functions for
            smoothing values.  It is also known as an apodization (which means
            "removing the foot", i.e. smoothing discontinuities at the beginning
            and end of the sampled signal) or tapering function. It is known as a
            "near optimal" tapering function, almost as good (by some measures)
            as the kaiser window.

            References
            ----------
            Blackman, R.B. and Tukey, J.W., (1958) The measurement of power spectra,
            Dover Publications, New York.

            Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.
            Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.

            Examples
            --------
            >>> np.blackman(12)
            array([ -1.38777878e-17,   3.26064346e-02,   1.59903635e-01,
                     4.14397981e-01,   7.36045180e-01,   9.67046769e-01,
                     9.67046769e-01,   7.36045180e-01,   4.14397981e-01,
                     1.59903635e-01,   3.26064346e-02,  -1.38777878e-17])


            Plot the window and the frequency response:

            >>> from numpy.fft import fft, fftshift
            >>> window = np.blackman(51)
            >>> plt.plot(window)
            [<matplotlib.lines.Line2D object at 0x...>]
            >>> plt.title("Blackman window")
            <matplotlib.text.Text object at 0x...>
            >>> plt.ylabel("Amplitude")
            <matplotlib.text.Text object at 0x...>
            >>> plt.xlabel("Sample")
            <matplotlib.text.Text object at 0x...>
            >>> plt.show()

            >>> plt.figure()
            <matplotlib.figure.Figure object at 0x...>
            >>> A = fft(window, 2048) / 25.5
            >>> mag = np.abs(fftshift(A))
            >>> freq = np.linspace(-0.5, 0.5, len(A))
            >>> response = 20 * np.log10(mag)
            >>> response = np.clip(response, -100, 100)
            >>> plt.plot(freq, response)
            [<matplotlib.lines.Line2D object at 0x...>]
            >>> plt.title("Frequency response of Blackman window")
            <matplotlib.text.Text object at 0x...>
            >>> plt.ylabel("Magnitude [dB]")
            <matplotlib.text.Text object at 0x...>
            >>> plt.xlabel("Normalized frequency [cycles per sample]")
            <matplotlib.text.Text object at 0x...>
            >>> plt.axis('tight')
            (-0.5, 0.5, -100.0, ...)
            >>> plt.show()
             */

            if (M < 1)
                return array(new int[] { });
            if (M == 1)
                return ones(1, np.Float32);

            var n = np.arange(0, M);

            return asanyarray(0.42) - asanyarray(0.5) * cos(asanyarray(2.0) * asanyarray(Math.PI) * n / (M - 1)) + 0.08 * cos(asanyarray(4.0) * asanyarray(Math.PI) * n / (M - 1));

        }

        #endregion

        #region bartlett
        /// <summary>
        /// Return the Bartlett window.
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <returns></returns>
        public static ndarray bartlett(int M)
        {
            /*
               Return the Bartlett window.

                The Bartlett window is very similar to a triangular window, except
                that the end points are at zero.  It is often used in signal
                processing for tapering a signal, without generating too much
                ripple in the frequency domain.

                Parameters
                ----------
                M : int
                    Number of points in the output window. If zero or less, an
                    empty array is returned.

                Returns
                -------
                out : array
                    The triangular window, with the maximum value normalized to one
                    (the value one appears only if the number of samples is odd), with
                    the first and last samples equal to zero.

                See Also
                --------
                blackman, hamming, hanning, kaiser

                Notes
                -----
                The Bartlett window is defined as

                .. math:: w(n) = \\frac{2}{M-1} \\left(
                          \\frac{M-1}{2} - \\left|n - \\frac{M-1}{2}\\right|
                          \\right)

                Most references to the Bartlett window come from the signal
                processing literature, where it is used as one of many windowing
                functions for smoothing values.  Note that convolution with this
                window produces linear interpolation.  It is also known as an
                apodization (which means"removing the foot", i.e. smoothing
                discontinuities at the beginning and end of the sampled signal) or
                tapering function. The fourier transform of the Bartlett is the product
                of two sinc functions.
                Note the excellent discussion in Kanasewich.

                References
                ----------
                .. [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
                       Biometrika 37, 1-16, 1950.
                .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
                       The University of Alberta Press, 1975, pp. 109-110.
                .. [3] A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal
                       Processing", Prentice-Hall, 1999, pp. 468-471.
                .. [4] Wikipedia, "Window function",
                       http://en.wikipedia.org/wiki/Window_function
                .. [5] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
                       "Numerical Recipes", Cambridge University Press, 1986, page 429.

                Examples
                --------
                >>> np.bartlett(12)
                array([ 0.        ,  0.18181818,  0.36363636,  0.54545455,  0.72727273,
                        0.90909091,  0.90909091,  0.72727273,  0.54545455,  0.36363636,
                        0.18181818,  0.        ])

                Plot the window and its frequency response (requires SciPy and matplotlib):

                >>> from numpy.fft import fft, fftshift
                >>> window = np.bartlett(51)
                >>> plt.plot(window)
                [<matplotlib.lines.Line2D object at 0x...>]
                >>> plt.title("Bartlett window")
                <matplotlib.text.Text object at 0x...>
                >>> plt.ylabel("Amplitude")
                <matplotlib.text.Text object at 0x...>
                >>> plt.xlabel("Sample")
                <matplotlib.text.Text object at 0x...>
                >>> plt.show()

                >>> plt.figure()
                <matplotlib.figure.Figure object at 0x...>
                >>> A = fft(window, 2048) / 25.5
                >>> mag = np.abs(fftshift(A))
                >>> freq = np.linspace(-0.5, 0.5, len(A))
                >>> response = 20 * np.log10(mag)
                >>> response = np.clip(response, -100, 100)
                >>> plt.plot(freq, response)
                [<matplotlib.lines.Line2D object at 0x...>]
                >>> plt.title("Frequency response of Bartlett window")
                <matplotlib.text.Text object at 0x...>
                >>> plt.ylabel("Magnitude [dB]")
                <matplotlib.text.Text object at 0x...>
                >>> plt.xlabel("Normalized frequency [cycles per sample]")
                <matplotlib.text.Text object at 0x...>
                >>> plt.axis('tight')
                (-0.5, 0.5, -100.0, ...)
                >>> plt.show()             
             */

            if (M < 1)
                return array(new int[] { });
            if (M == 1)
                return ones(1, np.Float32);

            var n = np.arange(0, M);

            return where(less_equal(n, (M - 1) / 2.0), asanyarray(2.0) * n / (M - 1), asanyarray(2.0) - asanyarray(2.0) * n / (M - 1)) as ndarray;

        }
        #endregion

        #region hanning

        /// <summary>
        /// Return the Hanning window.
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <returns></returns>
        public static ndarray hanning(int M)
        {
            /*
            Return the Hanning window.

            The Hanning window is a taper formed by using a weighted cosine.

            Parameters
            ----------
            M : int
                Number of points in the output window. If zero or less, an
                empty array is returned.

            Returns
            -------
            out : ndarray, shape(M,)
                The window, with the maximum value normalized to one (the value
                one appears only if `M` is odd).

            See Also
            --------
            bartlett, blackman, hamming, kaiser

            Notes
            -----
            The Hanning window is defined as

            .. math::  w(n) = 0.5 - 0.5cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
                       \\qquad 0 \\leq n \\leq M-1

            The Hanning was named for Julius von Hann, an Austrian meteorologist.
            It is also known as the Cosine Bell. Some authors prefer that it be
            called a Hann window, to help avoid confusion with the very similar
            Hamming window.

            Most references to the Hanning window come from the signal processing
            literature, where it is used as one of many windowing functions for
            smoothing values.  It is also known as an apodization (which means
            "removing the foot", i.e. smoothing discontinuities at the beginning
            and end of the sampled signal) or tapering function.

            References
            ----------
            .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
                   spectra, Dover Publications, New York.
            .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
                   The University of Alberta Press, 1975, pp. 106-108.
            .. [3] Wikipedia, "Window function",
                   http://en.wikipedia.org/wiki/Window_function
            .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
                   "Numerical Recipes", Cambridge University Press, 1986, page 425.

            Examples
            --------
            >>> np.hanning(12)
            array([ 0.        ,  0.07937323,  0.29229249,  0.57115742,  0.82743037,
                    0.97974649,  0.97974649,  0.82743037,  0.57115742,  0.29229249,
                    0.07937323,  0.        ])

            Plot the window and its frequency response:

            >>> from numpy.fft import fft, fftshift
            >>> window = np.hanning(51)
            >>> plt.plot(window)
            [<matplotlib.lines.Line2D object at 0x...>]
            >>> plt.title("Hann window")
            <matplotlib.text.Text object at 0x...>
            >>> plt.ylabel("Amplitude")
            <matplotlib.text.Text object at 0x...>
            >>> plt.xlabel("Sample")
            <matplotlib.text.Text object at 0x...>
            >>> plt.show()

            >>> plt.figure()
            <matplotlib.figure.Figure object at 0x...>
            >>> A = fft(window, 2048) / 25.5
            >>> mag = np.abs(fftshift(A))
            >>> freq = np.linspace(-0.5, 0.5, len(A))
            >>> response = 20 * np.log10(mag)
            >>> response = np.clip(response, -100, 100)
            >>> plt.plot(freq, response)
            [<matplotlib.lines.Line2D object at 0x...>]
            >>> plt.title("Frequency response of the Hann window")
            <matplotlib.text.Text object at 0x...>
            >>> plt.ylabel("Magnitude [dB]")
            <matplotlib.text.Text object at 0x...>
            >>> plt.xlabel("Normalized frequency [cycles per sample]")
            <matplotlib.text.Text object at 0x...>
            >>> plt.axis('tight')
            (-0.5, 0.5, -100.0, ...)
            >>> plt.show()             
            */

            if (M < 1)
                return array(new int[] { });
            if (M == 1)
                return ones(1, np.Float32);

            var n = np.arange(0, M);

            return asanyarray(0.5) - asanyarray(0.5) * cos(asanyarray(2.0) * asanyarray(Math.PI) * n / (M - 1));
        }

        #endregion

        #region hamming
        /// <summary>
        /// Return the Hamming window.
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <returns></returns>
        public static ndarray hamming(int M)
        {
            /*
            Return the Hamming window.

            The Hamming window is a taper formed by using a weighted cosine.

            Parameters
            ----------
            M : int
                Number of points in the output window. If zero or less, an
                empty array is returned.

            Returns
            -------
            out : ndarray
                The window, with the maximum value normalized to one (the value
                one appears only if the number of samples is odd).

            See Also
            --------
            bartlett, blackman, hanning, kaiser

            Notes
            -----
            The Hamming window is defined as

            .. math::  w(n) = 0.54 - 0.46cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
                       \\qquad 0 \\leq n \\leq M-1

            The Hamming was named for R. W. Hamming, an associate of J. W. Tukey
            and is described in Blackman and Tukey. It was recommended for
            smoothing the truncated autocovariance function in the time domain.
            Most references to the Hamming window come from the signal processing
            literature, where it is used as one of many windowing functions for
            smoothing values.  It is also known as an apodization (which means
            "removing the foot", i.e. smoothing discontinuities at the beginning
            and end of the sampled signal) or tapering function.

            References
            ----------
            .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
                   spectra, Dover Publications, New York.
            .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
                   University of Alberta Press, 1975, pp. 109-110.
            .. [3] Wikipedia, "Window function",
                   http://en.wikipedia.org/wiki/Window_function
            .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
                   "Numerical Recipes", Cambridge University Press, 1986, page 425.

            Examples
            --------
            >>> np.hamming(12)
            array([ 0.08      ,  0.15302337,  0.34890909,  0.60546483,  0.84123594,
                    0.98136677,  0.98136677,  0.84123594,  0.60546483,  0.34890909,
                    0.15302337,  0.08      ])

            Plot the window and the frequency response:

            >>> from numpy.fft import fft, fftshift
            >>> window = np.hamming(51)
            >>> plt.plot(window)
            [<matplotlib.lines.Line2D object at 0x...>]
            >>> plt.title("Hamming window")
            <matplotlib.text.Text object at 0x...>
            >>> plt.ylabel("Amplitude")
            <matplotlib.text.Text object at 0x...>
            >>> plt.xlabel("Sample")
            <matplotlib.text.Text object at 0x...>
            >>> plt.show()

            >>> plt.figure()
            <matplotlib.figure.Figure object at 0x...>
            >>> A = fft(window, 2048) / 25.5
            >>> mag = np.abs(fftshift(A))
            >>> freq = np.linspace(-0.5, 0.5, len(A))
            >>> response = 20 * np.log10(mag)
            >>> response = np.clip(response, -100, 100)
            >>> plt.plot(freq, response)
            [<matplotlib.lines.Line2D object at 0x...>]
            >>> plt.title("Frequency response of Hamming window")
            <matplotlib.text.Text object at 0x...>
            >>> plt.ylabel("Magnitude [dB]")
            <matplotlib.text.Text object at 0x...>
            >>> plt.xlabel("Normalized frequency [cycles per sample]")
            <matplotlib.text.Text object at 0x...>
            >>> plt.axis('tight')
            (-0.5, 0.5, -100.0, ...)
            >>> plt.show()             
            */

            if (M < 1)
                return array(new int[] { });
            if (M == 1)
                return ones(1, np.Float32);

            var n = np.arange(0, M);

            return asanyarray(0.54) - asanyarray(0.46) * cos(asanyarray(2.0) * asanyarray(Math.PI) * n / (M - 1));
        }
        #endregion

        #region i0

        private static double[] _i0A =
        {
                -4.41534164647933937950E-18,
                3.33079451882223809783E-17,
                -2.43127984654795469359E-16,
                1.71539128555513303061E-15,
                -1.16853328779934516808E-14,
                7.67618549860493561688E-14,
                -4.85644678311192946090E-13,
                2.95505266312963983461E-12,
                -1.72682629144155570723E-11,
                9.67580903537323691224E-11,
                -5.18979560163526290666E-10,
                2.65982372468238665035E-9,
                -1.30002500998624804212E-8,
                6.04699502254191894932E-8,
                -2.67079385394061173391E-7,
                1.11738753912010371815E-6,
                -4.41673835845875056359E-6,
                1.64484480707288970893E-5,
                -5.75419501008210370398E-5,
                1.88502885095841655729E-4,
                -5.76375574538582365885E-4,
                1.63947561694133579842E-3,
                -4.32430999505057594430E-3,
                1.05464603945949983183E-2,
                -2.37374148058994688156E-2,
                4.93052842396707084878E-2,
                -9.49010970480476444210E-2,
                1.71620901522208775349E-1,
                -3.04682672343198398683E-1,
                6.76795274409476084995E-1
        };

        private static double[] _i0B =
        {
            -7.23318048787475395456E-18,
            -4.83050448594418207126E-18,
            4.46562142029675999901E-17,
            3.46122286769746109310E-17,
            -2.82762398051658348494E-16,
            -3.42548561967721913462E-16,
            1.77256013305652638360E-15,
            3.81168066935262242075E-15,
            -9.55484669882830764870E-15,
            -4.15056934728722208663E-14,
            1.54008621752140982691E-14,
            3.85277838274214270114E-13,
            7.18012445138366623367E-13,
            -1.79417853150680611778E-12,
            -1.32158118404477131188E-11,
            -3.14991652796324136454E-11,
            1.18891471078464383424E-11,
            4.94060238822496958910E-10,
            3.39623202570838634515E-9,
            2.26666899049817806459E-8,
            2.04891858946906374183E-7,
            2.89137052083475648297E-6,
            6.88975834691682398426E-5,
            3.36911647825569408990E-3,
            8.04490411014108831608E-1
        };

        private static ndarray _chbevl(ndarray x, double[] vals)
        {
            ndarray b0 = asanyarray(vals[0]);
            ndarray b1 = asanyarray(new double[] { 0.0 });
            ndarray b2 = asanyarray(new double[] { 0.0 });

            for (int i = 1; i < vals.Length; i++)
            {
                b2 = b1;
                b1 = b0;
                b0 = x * b1 - b2 + vals[i];

            }

            return asanyarray(0.5) * (b0 - b2);
        }


        private static ndarray _i0_1(ndarray x)
        {
            return exp(x) * _chbevl(np.divide(x,2.0) - 2, _i0A);
        }

        private static ndarray _i0_2(ndarray x)
        {
            return (exp(x) * _chbevl(np.divide(asanyarray(32.0), x) - 2.0, _i0B) / sqrt(x)) as ndarray;
        }

        /// <summary>
        /// Modified Bessel function of the first kind, order 0.
        /// </summary>
        /// <param name="x">array_like, dtype float or complex Argument of the Bessel function.</param>
        /// <returns></returns>
        public static ndarray i0(object x)
        {
            /*
            Modified Bessel function of the first kind, order 0.

            Usually denoted :math:`I_0`.  This function does broadcast, but will *not*
            "up-cast" int dtype arguments unless accompanied by at least one float or
            complex dtype argument (see Raises below).

            Parameters
            ----------
            x : array_like, dtype float or complex
                Argument of the Bessel function.

            Returns
            -------
            out : ndarray, shape = x.shape, dtype = x.dtype
                The modified Bessel function evaluated at each of the elements of `x`.

            Raises
            ------
            TypeError: array cannot be safely cast to required type
                If argument consists exclusively of int dtypes.

            See Also
            --------
            scipy.special.iv, scipy.special.ive

            Notes
            -----
            We use the algorithm published by Clenshaw [1]_ and referenced by
            Abramowitz and Stegun [2]_, for which the function domain is
            partitioned into the two intervals [0,8] and (8,inf), and Chebyshev
            polynomial expansions are employed in each interval. Relative error on
            the domain [0,30] using IEEE arithmetic is documented [3]_ as having a
            peak of 5.8e-16 with an rms of 1.4e-16 (n = 30000).

            References
            ----------
            .. [1] C. W. Clenshaw, "Chebyshev series for mathematical functions", in
                   *National Physical Laboratory Mathematical Tables*, vol. 5, London:
                   Her Majesty's Stationery Office, 1962.
            .. [2] M. Abramowitz and I. A. Stegun, *Handbook of Mathematical
                   Functions*, 10th printing, New York: Dover, 1964, pp. 379.
                   http://www.math.sfu.ca/~cbm/aands/page_379.htm
            .. [3] http://kobesearch.cpan.org/htdocs/Math-Cephes/Math/Cephes.html

            Examples
            --------
            >>> np.i0([0.])
            array(1.0)
            >>> np.i0([0., 1. + 2j])
            array([ 1.00000000+0.j        ,  0.18785373+0.64616944j])             
            */

            var xa = asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            dtype result_dtype = result_type(xa.TypeNum); 

            xa = atleast_1d(xa.astype(result_dtype)).ElementAt(0).Copy();
            var y = empty_like(xa);
            var ind = (xa < 0);
            xa[ind] = -xa.A(ind);
            ind = (xa <= 8.0);
            y[ind] = _i0_1(xa.A(ind));
            var ind2 = ~ind;
            y[ind2] = _i0_2(xa.A(ind2));
            return y.Squeeze().astype(xa.Dtype);
        }

        #endregion

        #region kaiser
        /// <summary>
        /// Return the Kaiser window.
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <param name="beta">Shape parameter for window.</param>
        /// <returns></returns>
        public static ndarray kaiser(int M, double beta)
        {
            /*
            Return the Kaiser window.

            The Kaiser window is a taper formed by using a Bessel function.

            Parameters
            ----------
            M : int
                Number of points in the output window. If zero or less, an
                empty array is returned.
            beta : float
                Shape parameter for window.

            Returns
            -------
            out : array
                The window, with the maximum value normalized to one (the value
                one appears only if the number of samples is odd).

            See Also
            --------
            bartlett, blackman, hamming, hanning

            Notes
            -----
            The Kaiser window is defined as

            .. math::  w(n) = I_0\\left( \\beta \\sqrt{1-\\frac{4n^2}{(M-1)^2}}
                       \\right)/I_0(\\beta)

            with

            .. math:: \\quad -\\frac{M-1}{2} \\leq n \\leq \\frac{M-1}{2},

            where :math:`I_0` is the modified zeroth-order Bessel function.

            The Kaiser was named for Jim Kaiser, who discovered a simple
            approximation to the DPSS window based on Bessel functions.  The Kaiser
            window is a very good approximation to the Digital Prolate Spheroidal
            Sequence, or Slepian window, which is the transform which maximizes the
            energy in the main lobe of the window relative to total energy.

            The Kaiser can approximate many other windows by varying the beta
            parameter.

            ====  =======================
            beta  Window shape
            ====  =======================
            0     Rectangular
            5     Similar to a Hamming
            6     Similar to a Hanning
            8.6   Similar to a Blackman
            ====  =======================

            A beta value of 14 is probably a good starting point. Note that as beta
            gets large, the window narrows, and so the number of samples needs to be
            large enough to sample the increasingly narrow spike, otherwise NaNs will
            get returned.

            Most references to the Kaiser window come from the signal processing
            literature, where it is used as one of many windowing functions for
            smoothing values.  It is also known as an apodization (which means
            "removing the foot", i.e. smoothing discontinuities at the beginning
            and end of the sampled signal) or tapering function.

            References
            ----------
            .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
                   digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
                   John Wiley and Sons, New York, (1966).
            .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
                   University of Alberta Press, 1975, pp. 177-178.
            .. [3] Wikipedia, "Window function",
                   http://en.wikipedia.org/wiki/Window_function

            Examples
            --------
            >>> np.kaiser(12, 14)
            array([  7.72686684e-06,   3.46009194e-03,   4.65200189e-02,
                     2.29737120e-01,   5.99885316e-01,   9.45674898e-01,
                     9.45674898e-01,   5.99885316e-01,   2.29737120e-01,
                     4.65200189e-02,   3.46009194e-03,   7.72686684e-06])


            Plot the window and the frequency response:

            >>> from numpy.fft import fft, fftshift
            >>> window = np.kaiser(51, 14)
            >>> plt.plot(window)
            [<matplotlib.lines.Line2D object at 0x...>]
            >>> plt.title("Kaiser window")
            <matplotlib.text.Text object at 0x...>
            >>> plt.ylabel("Amplitude")
            <matplotlib.text.Text object at 0x...>
            >>> plt.xlabel("Sample")
            <matplotlib.text.Text object at 0x...>
            >>> plt.show()

            >>> plt.figure()
            <matplotlib.figure.Figure object at 0x...>
            >>> A = fft(window, 2048) / 25.5
            >>> mag = np.abs(fftshift(A))
            >>> freq = np.linspace(-0.5, 0.5, len(A))
            >>> response = 20 * np.log10(mag)
            >>> response = np.clip(response, -100, 100)
            >>> plt.plot(freq, response)
            [<matplotlib.lines.Line2D object at 0x...>]
            >>> plt.title("Frequency response of Kaiser window")
            <matplotlib.text.Text object at 0x...>
            >>> plt.ylabel("Magnitude [dB]")
            <matplotlib.text.Text object at 0x...>
            >>> plt.xlabel("Normalized frequency [cycles per sample]")
            <matplotlib.text.Text object at 0x...>
            >>> plt.axis('tight')
            (-0.5, 0.5, -100.0, ...)
            >>> plt.show()             
            */

            if (M == 1)
                return np.array(new double[] { 1.0 });

            var n = arange(0, M);
            double alpha = (M - 1) / 2.0;
 
            return i0(asanyarray(beta) * sqrt(asanyarray(1) - np.power(np.divide(n - asanyarray(alpha), alpha), 2.0))) / i0(asanyarray(beta)) as ndarray;
        }

        #endregion

        #region sinc
        /// <summary>
        /// Return the sinc function.
        /// </summary>
        /// <param name="x">Array (possibly multi-dimensional) of values for which to to calculate 'sinc(x)'.</param>
        /// <returns></returns>
        public static ndarray sinc(ndarray x)
        {
            /*
            Return the sinc function.

            The sinc function is :math:`\\sin(\\pi x)/(\\pi x)`.

            Parameters
            ----------
            x : ndarray
                Array (possibly multi-dimensional) of values for which to to
                calculate ``sinc(x)``.

            Returns
            -------
            out : ndarray
                ``sinc(x)``, which has the same shape as the input.

            Notes
            -----
            ``sinc(0)`` is the limit value 1.

            The name sinc is short for "sine cardinal" or "sinus cardinalis".

            The sinc function is used in various signal processing applications,
            including in anti-aliasing, in the construction of a Lanczos resampling
            filter, and in interpolation.

            For bandlimited interpolation of discrete-time signals, the ideal
            interpolation kernel is proportional to the sinc function.

            References
            ----------
            .. [1] Weisstein, Eric W. "Sinc Function." From MathWorld--A Wolfram Web
                   Resource. http://mathworld.wolfram.com/SincFunction.html
            .. [2] Wikipedia, "Sinc function",
                   http://en.wikipedia.org/wiki/Sinc_function

            Examples
            --------
            >>> x = np.linspace(-4, 4, 41)
            >>> np.sinc(x)
            array([ -3.89804309e-17,  -4.92362781e-02,  -8.40918587e-02,
                    -8.90384387e-02,  -5.84680802e-02,   3.89804309e-17,
                     6.68206631e-02,   1.16434881e-01,   1.26137788e-01,
                     8.50444803e-02,  -3.89804309e-17,  -1.03943254e-01,
                    -1.89206682e-01,  -2.16236208e-01,  -1.55914881e-01,
                     3.89804309e-17,   2.33872321e-01,   5.04551152e-01,
                     7.56826729e-01,   9.35489284e-01,   1.00000000e+00,
                     9.35489284e-01,   7.56826729e-01,   5.04551152e-01,
                     2.33872321e-01,   3.89804309e-17,  -1.55914881e-01,
                    -2.16236208e-01,  -1.89206682e-01,  -1.03943254e-01,
                    -3.89804309e-17,   8.50444803e-02,   1.26137788e-01,
                     1.16434881e-01,   6.68206631e-02,   3.89804309e-17,
                    -5.84680802e-02,  -8.90384387e-02,  -8.40918587e-02,
                    -4.92362781e-02,  -3.89804309e-17])

            >>> plt.plot(x, np.sinc(x))
            [<matplotlib.lines.Line2D object at 0x...>]
            >>> plt.title("Sinc Function")
            <matplotlib.text.Text object at 0x...>
            >>> plt.ylabel("Amplitude")
            <matplotlib.text.Text object at 0x...>
            >>> plt.xlabel("X")
            <matplotlib.text.Text object at 0x...>
            >>> plt.show()

            It works in 2-D as well:

            >>> x = np.linspace(-4, 4, 401)
            >>> xx = np.outer(x, x)
            >>> plt.imshow(np.sinc(xx))
            <matplotlib.image.AxesImage object at 0x...>             
            */

            var xa = np.asanyarray(x);

            if (!xa.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(xa);
            }

            ndarray y;

            if (xa.IsComplex)
            {
                y = asanyarray((System.Numerics.Complex)Math.PI) * (ndarray)np.where(xa == 0, (System.Numerics.Complex)1.0e-20, xa);
            }
            else
            {
                y = asanyarray(Math.PI) * (ndarray)np.where(xa == 0, 1.0e-20, xa);
            }
            return sin(y) / y as ndarray;

        }

        #endregion

        #region msort
        /// <summary>
        /// Return a copy of an array sorted along the first axis.
        /// </summary>
        /// <param name="a">Array to be sorted.</param>
        /// <returns></returns>
        public static ndarray msort(ndarray a)
        {
            // Return a copy of an array sorted along the first axis.

            // Parameters
            // ----------
            // a: array_like
            //    Array to be sorted.

            //Returns
            //------ -
            //sorted_array : ndarray
            //    Array of the same type and shape as `a`.

            // See Also
            // --------
            // sort

            // Notes
            // ---- -
            // ``np.msort(a)`` is equivalent to  ``np.sort(a, axis = 0)``.

            ndarray b = array(a, subok: true, copy: true);
            b = b.Sort(0);
            return b;
        }

        #endregion

        #region _ureduce

        private delegate ndarray _ureduce_func(ndarray a, ndarray q, bool IsQArray, int? axis = null, ndarray @out = null,
                                                bool overwrite_input = false, string interpolation = "linear", bool keepdims = false);

        private static (ndarray r, List<npy_intp> keepdims) _ureduce(ndarray a, _ureduce_func func, ndarray q, bool IsQarray, int[] axisarray = null,
            ndarray @out = null, bool overwrite_input = false, string interpolation = "linear", bool keepdims = false)
        {

            //Internal Function.
            //Call `func` with `a` as first argument swapping the axes to use extended
            //axis on functions that don't support it natively.

            //Returns result and a.shape with axis dims set to 1.

            //Parameters
            //----------
            //a: array_like
            //   Input array or object that can be converted to an array.
            //func: callable
            //   Reduction function capable of receiving a single axis argument.
            //    It is called with `a` as first argument followed by `kwargs`.
            //kwargs: keyword arguments
            //    additional keyword arguments to pass to `func`.

            //Returns
            //------ -
            //result : tuple
            //    Result of func(a, ** kwargs) and a.shape with axis dims set to 1
            //    which can be used to reshape the result to the same shape a ufunc with
            //    keepdims = True would produce.

            List<npy_intp> keepdim = null;
            int nd;
            int? axis = null;

            a = np.asanyarray(a);
            if (axisarray != null)
            {
                keepdim = a.shape.iDims.ToList();
                nd = a.ndim;
                axisarray = normalize_axis_tuple(axisarray, a.ndim);

                foreach (var ax in axisarray)
                    keepdim[ax] = 1;

                if (len(axisarray) == 1)
                {
                    axis = axisarray[0];
                }
                else
                {
                    // keep = set(range(nd)) - set(axis)

                    List<int> keep = new List<int>();
                    for (int i = 0; i < nd; i++)
                        keep.Add(i);

                    foreach (var aa in axisarray)
                    {
                        if (keep.Contains(aa))
                        {
                            keep.Remove(aa);
                        }
                    }

                    var nkeep = keep.Count;

                    // swap axis that should not be reduced to front
                    keep.Sort();
                    for (int i = 0; i < keep.Count; i++)
                    {
                        a = a.SwapAxes(i, keep[i]);
                    }

                    // merge reduced axis

                    var newShape = new npy_intp[nkeep + 1];
                    for (int i = 0; i < nkeep; i++)
                    {
                        newShape[i] = a.shape.iDims[i];
                    }
                    newShape[nkeep] = -1;
                    a = a.reshape(new shape(newShape));
                    axis = -1;
                }

            }
            else
            {
                keepdim = new List<npy_intp>();
                for (int i = 0; i < a.ndim; i++)
                {
                    keepdim.Add(1);
                }
            }

            ndarray r = func(a, q, IsQarray, axis, @out, overwrite_input, interpolation, keepdims);
            return (r, keepdim);

        }
        #endregion

        #region median
        /// <summary>
        /// Compute the median along the specified axis.
        /// </summary>
        /// <param name="a">Input array or object that can be converted to an array.</param>
        /// <param name="axis">Axis or axes along which the medians are computed.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray median(ndarray a, int[] axis = null, bool keepdims = false)
        {
            /*
            Compute the median along the specified axis.

            Returns the median of the array elements.

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
                the result will broadcast correctly against the original `arr`.

                .. versionadded:: 1.9.0

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
            mean, percentile

            Notes
            -----
            Given a vector ``V`` of length ``N``, the median of ``V`` is the
            middle value of a sorted copy of ``V``, ``V_sorted`` - i
            e., ``V_sorted[(N-1)/2]``, when ``N`` is odd, and the average of the
            two middle values of ``V_sorted`` when ``N`` is even.

            Examples
            --------
            >>> a = np.array([[10, 7, 4], [3, 2, 1]])
            >>> a
            array([[10,  7,  4],
                   [ 3,  2,  1]])
            >>> np.median(a)
            3.5
            >>> np.median(a, axis=0)
            array([ 6.5,  4.5,  2.5])
            >>> np.median(a, axis=1)
            array([ 7.,  2.])
            >>> m = np.median(a, axis=0)
            >>> out = np.zeros_like(m)
            >>> np.median(a, axis=0, out=m)
            array([ 6.5,  4.5,  2.5])
            >>> m
            array([ 6.5,  4.5,  2.5])
            >>> b = a.copy()
            >>> np.median(b, axis=1, overwrite_input=True)
            array([ 7.,  2.])
            >>> assert not np.all(a==b)
            >>> b = a.copy()
            >>> np.median(b, axis=None, overwrite_input=True)
            3.5
            >>> assert not np.all(a==b)
             */

            var _ureduce_ret = _ureduce(a, func: _median, q: null, IsQarray: false, axisarray: axis);

            if (keepdims)
                return _ureduce_ret.r.reshape(_ureduce_ret.keepdims);
            else
                return _ureduce_ret.r;

        }

        /// <summary>
        /// Compute the median along the specified axis.
        /// </summary>
        /// <param name="a">Input array or object that can be converted to an array.</param>
        /// <param name="axis">Axis or axes along which the medians are computed.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray median(ndarray a, int axis, bool keepdims = false)
        {
            return median(a, new int[] { axis }, keepdims);
        }

        private static ndarray _median(ndarray a, ndarray q, bool IsQarray, int? axis = null, ndarray @out = null,
           bool overwrite_input = false, string interpolation = "linear", bool keepdims = false)
        {
            // can't be reasonably be implemented in terms of percentile as we have to
            // call mean to not break astropy
            a = np.asanyarray(a);

            npy_intp sz;
            List<npy_intp> kth = new List<npy_intp>();
            npy_intp szh;
            ndarray part;

            //Set the partition indexes
            if (axis == null)
            {
                sz = a.size;
            }
            else
            {
                if (axis.Value < 0)
                    axis += a.shape.iDims.Length;

                sz = a.shape.iDims[axis.Value];
            }

            if (sz % 2 == 0)
            {
                szh = sz / 2;
                kth.AddRange(new npy_intp[] { szh - 1, szh });
            }
            else
            {
                kth.AddRange(new npy_intp[] { (sz - 1) / 2 });
            }

            // Check if the array contains any nan's
            if (a.IsInexact)
            {
                kth.Add(-1);
            }


            if (overwrite_input)
            {
                if (axis == null)
                {
                    part = a.ravel();
                    part = part.partition(kth.ToArray());
                }
                else
                {
                    part = a.partition(kth.ToArray(), axis: axis);
                }
            }
            else
            {
                part = partition(a, kth.ToArray(), axis: axis);
            }

            if (part.ndim == 0)
            {
                //make 0 - D arrays work
                return asanyarray(part.GetItem(0));
            }

            if (axis == null)
            {
                axis = 0;
            }

            var indexer = BuildSliceArray(new Slice(null), part.ndim);

            var index = part.shape.iDims[axis.Value] / 2;
            if (part.shape.iDims[axis.Value] % 2 == 1)
            {
                // index with slice to allow mean (below) to work
                indexer[axis.Value] = new Slice(index, index + 1);
            }
            else
            {
                indexer[axis.Value] = new Slice(index - 1, index + 1);
            }

            // Check if the array contains any nan's
            if (a.IsInexact && sz > 0)
            {
                // warn and return nans like mean would
                var rout = mean(part[indexer], axis: axis);
                return _median_nancheck(part, rout, axis);
            }
            else
            {
                // if there are no nans
                // Use mean in odd and even case to coerce data type
                // and check, use out array.
                return mean(part[indexer], axis: axis);
            }

        }

        private static ndarray _median_nancheck(ndarray data, ndarray result, int? axis = null)
        {
            // Utility function to check median result from data for NaN values at the end
            // and return NaN in that case. Input result can also be a MaskedArray.

            // Parameters
            // ----------
            // data: array
            //    Input data to median function
            // result: Array or MaskedArray
            //    Result of median function
            //axis : { int, sequence of int, None}, optional
            //     Axis or axes along which the median was computed.
            // out : ndarray, optional
            //     Output array in which to place the result.
            // Returns
            // ------ -
            // median : scalar or ndarray
            //     Median or NaN in axes which contained NaN in the input.

            if (data.size == 0)
                return result;
            data = np.moveaxis(data, axis, -1);
            var n = np.isnan(data.A("...", -1));
            // masked NaN values are ok
            //if (np.ma.isMaskedArray(n))
            //    n = n.filled(false);

            if (result.ndim == 0)
            {
                if ((bool)n.GetItem(0) == true)
                {
                    //warnings.warn("Invalid value encountered in median",  RuntimeWarning, stacklevel = 3)

                    result = array(np.NaN, dtype: data.Dtype);
                }
   
            }
            else if ((int)np.count_nonzero(n.ravel()) > 0)
            {
                // warnings.warn("Invalid value encountered in median for" + " %d results" % np.count_nonzero(n.ravel()), RuntimeWarning, stacklevel=3)
                result[n] = np.NaN;
            }
            return result;
        }

        #endregion

        #region percentile
        /// <summary>
        /// Compute the qth percentile of the data along the specified axis.
        /// </summary>
        /// <param name="a">Input array or object that can be converted to an array.</param>
        /// <param name="q">Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.</param>
        /// <param name="axis">Axis or axes along which the percentiles are computed.</param>
        /// <param name="interpolation">{'linear', 'lower', 'higher', 'midpoint', 'nearest'}</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray percentile(object a, object q, int? axis = null,  
                string interpolation = "linear", bool keepdims=false)
        {
            /*
            Compute the qth percentile of the data along the specified axis.

            Returns the qth percentile(s) of the array elements.

            Parameters
            ----------
            a : array_like
                Input array or object that can be converted to an array.
            q : array_like of float
                Percentile or sequence of percentiles to compute, which must be between
                0 and 100 inclusive.
            axis : {int, tuple of int, None}, optional
                Axis or axes along which the percentiles are computed. The
                default is to compute the percentile(s) along a flattened
                version of the array.

                .. versionchanged:: 1.9.0
                    A tuple of axes is supported
          
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

                .. versionadded:: 1.9.0
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left in
                the result as dimensions with size one. With this option, the
                result will broadcast correctly against the original array `a`.

                .. versionadded:: 1.9.0

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
            mean
            median : equivalent to ``percentile(..., 50)``
            nanpercentile

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
            >>> a = np.array([[10, 7, 4], [3, 2, 1]])
            >>> a
            array([[10,  7,  4],
                   [ 3,  2,  1]])
            >>> np.percentile(a, 50)
            3.5
            >>> np.percentile(a, 50, axis=0)
            array([[ 6.5,  4.5,  2.5]])
            >>> np.percentile(a, 50, axis=1)
            array([ 7.,  2.])
            >>> np.percentile(a, 50, axis=1, keepdims=True)
            array([[ 7.],
                   [ 2.]])

            >>> m = np.percentile(a, 50, axis=0)
            >>> out = np.zeros_like(m)
            >>> np.percentile(a, 50, axis=0, out=out)
            array([[ 6.5,  4.5,  2.5]])
            >>> m
            array([[ 6.5,  4.5,  2.5]])

            >>> b = a.copy()
            >>> np.percentile(b, 50, axis=1, overwrite_input=True)
            array([ 7.,  2.])
            >>> assert not np.all(a == b)
            */

            bool overwrite_input = false;

            bool IsQArray = false;
            if (q.GetType().IsArray)
            {
                IsQArray = true;
            }


            var qa = asanyarray(q);
            var d = result_type(qa.TypeNum);
            qa = qa.astype(d);

            var aa = asanyarray(a);
            aa = aa.astype(d);

            var q1 = np.true_divide(qa, 100.0);  // handles the asarray for us too
            if (!_quantile_is_valid(q1))
            {
                throw new ValueError("Percentiles must be in the range [0, 100]");
            }
            return _quantile_unchecked(aa, q1,IsQArray, axis, overwrite_input, interpolation, keepdims);
        }

        /// <summary>
        /// Compute the q-th quantile of the data along the specified axis.
        /// </summary>
        /// <param name="a">Input array or object that can be converted to an array.</param>
        /// <param name="q">Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive.</param>
        /// <param name="axis">Axis or axes along which the quantiles are computed.</param>
        /// <param name="interpolation">{'linear', 'lower', 'higher', 'midpoint', 'nearest'}</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. </param>
        /// <returns></returns>
        public static ndarray quantile(object a, object q, int? axis = null,
            string interpolation = "linear", bool keepdims = false)
        {
            /*
            Compute the q-th quantile of the data along the specified axis.
            Parameters
            ----------
            a : array_like
                Input array or object that can be converted to an array.
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

            Returns
            -------
            quantile : scalar or ndarray
                If `q` is a single quantile and `axis=None`, then the result
                is a scalar. If multiple quantiles are given, first axis of
                the result corresponds to the quantiles. The other axes are
                the axes that remain after the reduction of `a`. If the input
                contains integers or floats smaller than ``float64``, the output
                data-type is ``float64``. Otherwise, the output data-type is the
                same as that of the input. If `out` is specified, that array is
                returned instead.

            See Also
            --------
            mean
            percentile : equivalent to quantile, but with q in the range [0, 100].
            median : equivalent to ``quantile(..., 0.5)``
            nanquantile

            Notes
            -----
            Given a vector ``V`` of length ``N``, the q-th quantile of
            ``V`` is the value ``q`` of the way from the minimum to the
            maximum in a sorted copy of ``V``. The values and distances of
            the two nearest neighbors as well as the `interpolation` parameter
            will determine the quantile if the normalized ranking does not
            match the location of ``q`` exactly. This function is the same as
            the median if ``q=0.5``, the same as the minimum if ``q=0.0`` and the
            same as the maximum if ``q=1.0``.

            Examples
            --------
            >>> a = np.array([[10, 7, 4], [3, 2, 1]])
            >>> a

            array([[10,  7,  4],
                   [ 3,  2,  1]])

            >>> np.quantile(a, 0.5)
            3.5

            >>> np.quantile(a, 0.5, axis=0)
            array([6.5, 4.5, 2.5])

            >>> np.quantile(a, 0.5, axis=1)
            array([7.,  2.])

            >>> np.quantile(a, 0.5, axis=1, keepdims=True)
            array([[7.],
                   [2.]])

            >>> m = np.quantile(a, 0.5, axis=0)
            >>> out = np.zeros_like(m)
            >>> np.quantile(a, 0.5, axis=0, out=out)
            array([6.5, 4.5, 2.5])

            >>> m
            array([6.5, 4.5, 2.5])

            >>> b = a.copy()
            >>> np.quantile(b, 0.5, axis=1, overwrite_input=True)
            array([7.,  2.])

            >>> assert not np.all(a == b)        
             */

            bool overwrite_input = false;

            bool IsQArray = false;
            if (q.GetType().IsArray)
            {
                IsQArray = true;
            }

            var qa = asanyarray(q);
            var d = result_type(qa.TypeNum);
            qa = qa.astype(d);

            var aa = asanyarray(a);
            aa = aa.astype(d);


            if (!_quantile_is_valid(qa))
            {
                throw new ValueError("Quantiles must be in the range [0, 1]");
            }

            return _quantile_unchecked(aa, qa, IsQArray, axis, overwrite_input, interpolation, keepdims);
        }

        private static ndarray _quantile_unchecked(ndarray a, ndarray q, bool IsQArray, int? axis = null,
                bool overwrite_input = false, string interpolation = "linear", bool keepdims = false)
        {
            int[] axisarray = null;
            if (axis != null)
            {
                axisarray = new int[] { axis.Value };
            }

            // Assumes that q is in [0, 1], and is an ndarray
            var _ureduce_ret = _ureduce(a, func: _quantile_ureduce_func, q: q, IsQarray : IsQArray,
                    axisarray: axisarray, 
                    overwrite_input: overwrite_input,
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

        private static bool _quantile_is_valid(ndarray q)
        {
            int nz1 = (int)np.count_nonzero(q < 0.0);
            int nz2 = (int)np.count_nonzero(q > 1.0);
            if (nz1 > 0 || nz2 > 0)
            {
                return false;
            }

            return true;
        }


        private static ndarray _quantile_ureduce_func(ndarray a, ndarray q, bool IsQarray, int? axis = null, ndarray @out = null,
                bool overwrite_input = false, string interpolation = "linear", bool keepdims = false)
        {
            bool zerod = false;
            ndarray ap;
            ndarray r;

            a = asarray(a);
            if (!IsQarray)
            {
                // Do not allow 0-d arrays because following code fails for scalar
                zerod = true;
                //q = q[null] as ndarray;
            }
            else
            {
                zerod = false;
            }

            // prepare a for partitioning
            if (overwrite_input)
            {
                if (axis == null)
                    ap = a.ravel();
                else
                    ap = a;
            }
            else
            {
                if (axis == null)
                    ap = a.flatten();
                else
                    ap = a.Copy();
            }


            if (axis == null)
                axis = 0;

            var Nx = ap.shape.iDims[axis.Value];
            var indices = q * (Nx - 1);

            // round fractional indices according to interpolation method

            switch (interpolation)
            {
                case "lower":
                    indices = np.floor(indices).astype(np.intp);
                    break;
                case "higher":
                    indices = np.ceil(indices).astype(np.intp);
                    break;
                case "midpoint":
                    indices = 0.5 * (np.floor(indices) + np.ceil(indices));
                    break;
                case "nearest":
                    indices = np.around(indices).astype(np.intp);
                    break;
                case "linear":
                    //pass  - keep index as fraction and interpolate
                    break;
                default:
                    throw new ValueError("interpolation can only be 'linear', 'lower' 'higher', 'midpoint', or 'nearest'");
            }


            var n = np.array(false, dtype: np.Bool);            // check for nan's flag
            if (indices.TypeNum == NPY_TYPES.NPY_INTP)    // take the points along axis
            {
                // Check if the array contains any nan's
                if (a.IsInexact)
                {
                    indices = concatenate((new ndarray[] { indices, np.array(new int[] { -1 }) }));
                }

                ap = ap.partition((npy_intp[])indices.ToArray(), axis: axis);
                // ensure axis with qth is first
                ap = np.moveaxis(ap, axis, 0);
                axis = 0;

                // Check if the array contains any nan's
                if (a.IsInexact)
                {
                    indices = indices.A(":-1");
                    n = np.isnan(ap.A("-1:", "..."));
                }

                if (zerod)
                {
                    indices = indices.A(0);
                }
                r = take(ap, indices, axis: axis, _out: @out);
            }
            else // weight the points above and below the indices
            {
                var indices_below = floor(indices).astype(intp);
                var indices_above = indices_below + 1;
                indices_above[indices_above > Nx - 1] = Nx - 1;

                // Check if the array contains any nan's
                if (a.IsInexact)
                {
                    indices_above = concatenate((indices_above, new int[] { -1 }));
                }

                var weights_above = indices - indices_below;
                var weights_below = asanyarray(1.0) - weights_above;

                npy_intp[] weights_shape = new npy_intp[ap.ndim];
                for (int i = 0; i < weights_shape.Length; i++)
                    weights_shape[i] = 1;

                weights_shape[axis.Value] = len(indices);
                weights_below = weights_below.reshape(new shape(weights_shape));
                weights_above = weights_above.reshape(new shape(weights_shape));

                ap = ap.partition((npy_intp[])concatenate(((object)indices_below, indices_above)).ToArray(), axis: axis);

                // ensure axis with qth is first
                ap = np.moveaxis(ap, axis, 0);
                weights_below = np.moveaxis(weights_below, axis, 0);
                weights_above = np.moveaxis(weights_above, axis, 0);
                axis = 0;

                // Check if the array contains any nan's
                if (a.IsInexact)
                {
                    indices_above = indices_above.A(":-1");
                    n = np.isnan(ap.A("-1:", "..."));
                }

                var x1 = take(ap, indices_below, axis: axis) * weights_below;
                var x2 = take(ap, indices_above, axis: axis) * weights_above;

                // ensure axis with qth is first
                x1 = np.moveaxis(x1, axis, 0);
                x2 = np.moveaxis(x2, axis, 0);

                if (zerod)
                {
                    x1 = np.squeeze(x1, 0);
                    x2 = np.squeeze(x2, 0);
                }


                r = np.add(x1, x2);

                if (np.anyb(n))
                {
                    //warnings.warn("Invalid value encountered in percentile", RuntimeWarning, stacklevel = 3);
                    if (zerod)
                    {
                        if (ap.ndim == 1)
                        {
                            r = np.array(double.NaN);
                        }
                        else
                        {
                            r["...", np.squeeze(n, 0)] = double.NaN;
                        }
                    }

                    else
                    {
                        if (r.ndim == 1)
                        {
                            r[":"] = double.NaN;
                        }
                        else
                        {
                            r["...", np.repeat(n, q.size, 0)] = double.NaN;
                        }

                    }

                }
            }

            return r;
        }

        #endregion

        #region trapz
        /// <summary>
        /// Integrate along the given axis using the composite trapezoidal rule.
        /// </summary>
        /// <param name="y">Input array to integrate.</param>
        /// <param name="x">The sample points corresponding to the 'y' values</param>
        /// <param name="dx">The spacing between sample points when 'x' is null</param>
        /// <param name="axis">The axis along which to integrate.</param>
        /// <returns></returns>
        public static ndarray trapz(object y, object x = null, double dx=1.0, int axis= -1)
        {
            /*
            Integrate along the given axis using the composite trapezoidal rule.

            Integrate `y` (`x`) along given axis.

            Parameters
            ----------
            y : array_like
                Input array to integrate.
            x : array_like, optional
                The sample points corresponding to the `y` values. If `x` is None,
                the sample points are assumed to be evenly spaced `dx` apart. The
                default is None.
            dx : scalar, optional
                The spacing between sample points when `x` is None. The default is 1.
            axis : int, optional
                The axis along which to integrate.

            Returns
            -------
            trapz : float
                Definite integral as approximated by trapezoidal rule.

            See Also
            --------
            sum, cumsum

            Notes
            -----
            Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
            will be taken from `y` array, by default x-axis distances between
            points will be 1.0, alternatively they can be provided with `x` array
            or with `dx` scalar.  Return value will be equal to combined area under
            the red lines.


            References
            ----------
            .. [1] Wikipedia page: http://en.wikipedia.org/wiki/Trapezoidal_rule

            .. [2] Illustration image:
                   http://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

            Examples
            --------
            >>> np.trapz([1,2,3])
            4.0
            >>> np.trapz([1,2,3], x=[4,6,8])
            8.0
            >>> np.trapz([1,2,3], dx=2)
            8.0
            >>> a = np.arange(6).reshape(2, 3)
            >>> a
            array([[0, 1, 2],
                   [3, 4, 5]])
            >>> np.trapz(a, axis=0)
            array([ 1.5,  2.5,  3.5])
            >>> np.trapz(a, axis=1)
            array([ 2.,  8.])             
            */

            ndarray d;
            ndarray yarr;
            ndarray xarr;

     
            yarr = asanyarray(y);

            if (!yarr.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(yarr);
            }

            yarr = yarr.astype(result_type(yarr.TypeNum));

            if (axis < 0)
                axis += yarr.ndim;

            if (x is null)
            {
                d = asanyarray(dx);
            }
            else
            {
                xarr = asanyarray(x);
                if (xarr.ndim == 1)
                {
                    d = diff(xarr);
                    // reshape to correct shape
                    List<npy_intp> shape = new List<npy_intp>();
                    for (int i = 0; i < yarr.ndim; i++)
                        shape.Add(1);

                    shape[axis] = d.shape.iDims[0];
                    d = d.reshape(shape);
                }
                else
                {
                    d = diff(xarr, axis: axis);
                }
            }

            var nd = yarr.ndim;
            var slice1 = BuildSliceArray(new Slice(null), nd);
            var slice2 = BuildSliceArray(new Slice(null), nd);

 

            slice1[axis] = new Slice(1, null);
            slice2[axis] = new Slice(null, -1);

            ndarray ret = null;

            try
            {
                ret = (d * (yarr.A(slice1) + yarr.A(slice2)) / 2.0).Sum(axis);
            }
            catch (Exception ex)
            {
                // Operations didn't work, cast to ndarray
                d = np.asarray(d);
                yarr = np.asarray(yarr);
                ret = ufunc.reduce(UFuncOperation.add, d * (yarr.A(slice1) + yarr.A(slice2)) / 2.0, axis: axis);
            }

            return ret;

        }

        #endregion

        #region add_newdoc

        public static void add_newdoc(object place, object obj, object doc)
        {

        }

        #endregion

        #region meshgrid

        /// <summary>
        /// Return coordinate matrices from coordinate vectors.
        /// </summary>
        /// <param name="xi">1-D arrays representing the coordinates of a grid.</param>
        /// <param name="indexing">Cartesian ('xy', default) or matrix ('ij') indexing of output.</param>
        /// <param name="sparse">If True a sparse grid is returned in order to conserve memory.</param>
        /// <param name="copy">If False, a view into the original arrays are returned in order to conserve memory</param>
        /// <returns></returns>
        public static ndarray[] meshgrid(ndarray []xi, string indexing = "xy", bool sparse = false, bool copy = true)
        {
            /*
            Return coordinate matrices from coordinate vectors.

            Make N-D coordinate arrays for vectorized evaluations of
            N-D scalar/vector fields over N-D grids, given
            one-dimensional coordinate arrays x1, x2,..., xn.

            .. versionchanged:: 1.9
               1-D and 0-D cases are allowed.

            Parameters
            ----------
            x1, x2,..., xn : array_like
                1-D arrays representing the coordinates of a grid.
            indexing : {'xy', 'ij'}, optional
                Cartesian ('xy', default) or matrix ('ij') indexing of output.
                See Notes for more details.

                .. versionadded:: 1.7.0
            sparse : bool, optional
                If True a sparse grid is returned in order to conserve memory.
                Default is False.

                .. versionadded:: 1.7.0
            copy : bool, optional
                If False, a view into the original arrays are returned in order to
                conserve memory.  Default is True.  Please note that
                ``sparse=False, copy=False`` will likely return non-contiguous
                arrays.  Furthermore, more than one element of a broadcast array
                may refer to a single memory location.  If you need to write to the
                arrays, make copies first.

                .. versionadded:: 1.7.0

            Returns
            -------
            X1, X2,..., XN : ndarray
                For vectors `x1`, `x2`,..., 'xn' with lengths ``Ni=len(xi)`` ,
                return ``(N1, N2, N3,...Nn)`` shaped arrays if indexing='ij'
                or ``(N2, N1, N3,...Nn)`` shaped arrays if indexing='xy'
                with the elements of `xi` repeated to fill the matrix along
                the first dimension for `x1`, the second for `x2` and so on.

            Notes
            -----
            This function supports both indexing conventions through the indexing
            keyword argument.  Giving the string 'ij' returns a meshgrid with
            matrix indexing, while 'xy' returns a meshgrid with Cartesian indexing.
            In the 2-D case with inputs of length M and N, the outputs are of shape
            (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.  In the 3-D case
            with inputs of length M, N and P, outputs are of shape (N, M, P) for
            'xy' indexing and (M, N, P) for 'ij' indexing.  The difference is
            illustrated by the following code snippet::

                xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
                for i in range(nx):
                    for j in range(ny):
                 treat xv[i,j], yv[i,j]

                xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
                for i in range(nx):
                    for j in range(ny):
                treat xv[j,i], yv[j,i]

            In the 1-D and 0-D case, the indexing and sparse keywords have no effect.

            See Also
            --------
            index_tricks.mgrid : Construct a multi-dimensional "meshgrid"
                             using indexing notation.
            index_tricks.ogrid : Construct an open multi-dimensional "meshgrid"
                             using indexing notation.

            Examples
            --------
            >>> nx, ny = (3, 2)
            >>> x = np.linspace(0, 1, nx)
            >>> y = np.linspace(0, 1, ny)
            >>> xv, yv = np.meshgrid(x, y)
            >>> xv
            array([[ 0. ,  0.5,  1. ],
                   [ 0. ,  0.5,  1. ]])
            >>> yv
            array([[ 0.,  0.,  0.],
                   [ 1.,  1.,  1.]])
            >>> xv, yv = np.meshgrid(x, y, sparse=True)  # make sparse output arrays
            >>> xv
            array([[ 0. ,  0.5,  1. ]])
            >>> yv
            array([[ 0.],
                   [ 1.]])

            `meshgrid` is very useful to evaluate functions on a grid.

            >>> x = np.arange(-5, 5, 0.1)
            >>> y = np.arange(-5, 5, 0.1)
            >>> xx, yy = np.meshgrid(x, y, sparse=True)
            >>> z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
            >>> h = plt.contourf(x,y,z)             
           */


            int ndim = xi.Length;

            switch (indexing)
            {
                case "xy":
                case "ij":
                    break;
                default:
                    throw new TypeError("Valid values for `indexing` are 'xy' and 'ij'.");
            }

            List<ndarray> output = new List<ndarray>();

            npy_intp[] s0 = new npy_intp[ndim];
            for (int i = 0; i < ndim; i++) s0[i] = 1;

            for (int i = 0; i < ndim; i++)
            {
                npy_intp[] newshape = new npy_intp[s0.Length];
                Array.Copy(s0, 0, newshape, 0, s0.Length);
                newshape[i] = -1;
                output.Add(np.asanyarray(xi[i]).reshape(newshape));
            }

            if (indexing == "xy" && ndim > 1)
            {
                npy_intp[] newshape = new npy_intp[s0.Length];
                Array.Copy(s0, 0, newshape, 0, s0.Length);

                // switch first and second axis
                newshape[0] = 1; newshape[1] = -1;
                output[0] = output[0].reshape(newshape);

                newshape[0] = -1; newshape[1] = 1;
                output[1] = output[1].reshape(newshape);
            }

            if (output.Count() > 1 && sparse == false)
            {
                output = np.broadcast_arrays(true, output).ToList();
            }

            if (copy)
            {
                for (int i = 0; i < output.Count; i++)
                {
                    output[i] = np.copy(output[i]);
                }
            }

            return output.ToArray();
        }

        #endregion

        #region delete

        /// <summary>
        /// Return a new array with sub-arrays along an axis deleted. For a one dimensional array, this returns those entries not returned by 'arr[obj]'.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="slice">slice, int or array of ints. Indicate which sub-arrays to remove.</param>
        /// <param name="axis">The axis along which to delete the subarray defined by 'slice'</param>
        /// <returns></returns>
        public static ndarray delete(ndarray arr, Slice slice, int axis)
        {
            /*
            Return a new array with sub-arrays along an axis deleted. For a one
            dimensional array, this returns those entries not returned by
            `arr[obj]`.

            Parameters
            ----------
            arr : array_like
              Input array.
            obj : slice, int or array of ints
              Indicate which sub-arrays to remove.
            axis : int, optional
              The axis along which to delete the subarray defined by `obj`.
              If `axis` is None, `obj` is applied to the flattened array.

            Returns
            -------
            out : ndarray
                A copy of `arr` with the elements specified by `obj` removed. Note
                that `delete` does not occur in-place. If `axis` is None, `out` is
                a flattened array.

            See Also
            --------
            insert : Insert elements into an array.
            append : Append elements at the end of an array.

            Notes
            -----
            Often it is preferable to use a boolean mask. For example:

            >>> mask = np.ones(len(arr), dtype=bool)
            >>> mask[[0,2,4]] = False
            >>> result = arr[mask,...]

            Is equivalent to `np.delete(arr, [0,2,4], axis=0)`, but allows further
            use of `mask`.

            Examples
            --------
            >>> arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
            >>> arr
            array([[ 1,  2,  3,  4],
                   [ 5,  6,  7,  8],
                   [ 9, 10, 11, 12]])
            >>> np.delete(arr, 1, 0)
            array([[ 1,  2,  3,  4],
                   [ 9, 10, 11, 12]])

            >>> np.delete(arr, np.s_[::2], 1)
            array([[ 2,  4],
                   [ 6,  8],
                   [10, 12]])
            >>> np.delete(arr, [1,3,5], None)
            array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])
             */

            // create a bool mask to indicate which of the fields to delete (false == delete)
            var mask = np.ones_like(arr, dtype: np.Bool);

            var maskIndex = new object[mask.ndim];
            for (int i = 0; i < maskIndex.Length; i++)
                maskIndex[i] = i == axis ? (object)slice : ":";
            mask[maskIndex] = false;

            var reshapeDims = arr.dims;
            reshapeDims[axis] = -1; // Unknown dimension to be resolved by reshape
            // use the "fancy index" feature to get only the data items marked true
            return arr.A(mask).reshape(reshapeDims);
        }
        /// <summary>
        /// Return a new array with sub-arrays along an axis deleted. For a one dimensional array, this returns those entries not returned by 'arr[obj]'
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="index">Indicate which sub-arrays to remove.</param>
        /// <param name="axis">The axis along which to delete the subarray defined by 'index'</param>
        /// <returns></returns>
        public static ndarray delete(ndarray arr, int index, int axis)
        {
            /*
            Return a new array with sub-arrays along an axis deleted. For a one
            dimensional array, this returns those entries not returned by
            `arr[obj]`.

            Parameters
            ----------
            arr : array_like
              Input array.
            obj : slice, int or array of ints
              Indicate which sub-arrays to remove.
            axis : int, optional
              The axis along which to delete the subarray defined by `obj`.
              If `axis` is None, `obj` is applied to the flattened array.

            Returns
            -------
            out : ndarray
                A copy of `arr` with the elements specified by `obj` removed. Note
                that `delete` does not occur in-place. If `axis` is None, `out` is
                a flattened array.

            See Also
            --------
            insert : Insert elements into an array.
            append : Append elements at the end of an array.

            Notes
            -----
            Often it is preferable to use a boolean mask. For example:

            >>> mask = np.ones(len(arr), dtype=bool)
            >>> mask[[0,2,4]] = False
            >>> result = arr[mask,...]

            Is equivalent to `np.delete(arr, [0,2,4], axis=0)`, but allows further
            use of `mask`.

            Examples
            --------
            >>> arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
            >>> arr
            array([[ 1,  2,  3,  4],
                   [ 5,  6,  7,  8],
                   [ 9, 10, 11, 12]])
            >>> np.delete(arr, 1, 0)
            array([[ 1,  2,  3,  4],
                   [ 9, 10, 11, 12]])

            >>> np.delete(arr, np.s_[::2], 1)
            array([[ 2,  4],
                   [ 6,  8],
                   [10, 12]])
            >>> np.delete(arr, [1,3,5], None)
            array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])
             */
            // create a bool mask to indicate which of the fields to delete (false == delete)
            var mask = np.ones_like(arr, dtype: np.Bool);

            var maskIndex = new object[mask.ndim];
            for (int i = 0; i < maskIndex.Length; i++)
                maskIndex[i] = i == axis ? (object)index : ":";
            mask[maskIndex] = false;

            var reshapeDims = arr.dims;
            reshapeDims[axis]--;
            // use the "fancy index" feature to get only the data items marked true
            return arr.A(mask).reshape(reshapeDims);
        }

        #endregion

        #region insert
        /// <summary>
        /// Insert values along the given axis before the given indices.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj"> Object that defines the index or indices before which 'values' is inserted.</param>
        /// <param name="_invalues">Values to insert into 'arr'.</param>
        /// <param name="axis">Axis along which to insert 'values'.</param>
        /// <returns></returns>
        public static ndarray insert(ndarray arr, object obj, dynamic _invalues, int? axis = null)
        {
            // Insert values along the given axis before the given indices.

            // Parameters
            // ----------
            // arr: array_like
            //    Input array.
            //obj : int, slice or sequence of ints
            //     Object that defines the index or indices before which `values` is
            //     inserted.

            //     ..versionadded:: 1.8.0

            //     Support for multiple insertions when `obj` is a single scalar or a
            //     sequence with one element(similar to calling insert multiple
            //     times).
            // values : array_like
            //     Values to insert into `arr`. If the type of `values` is different
            //     from that of `arr`, `values` is converted to the type of `arr`.
            //     `values` should be shaped so that ``arr[..., obj,...] = values``
            //     is legal.
            // axis : int, optional
            //     Axis along which to insert `values`.  If `axis` is None then `arr`
            //     is flattened first.

            // Returns
            // ------ -
            // out : ndarray
            //     A copy of `arr` with `values` inserted.Note that `insert`
            //     does not occur in-place: a new array is returned.If
            //     `axis` is None, `out` is a flattened array.

            // See Also
            // --------
            // append : Append elements at the end of an array.
            // concatenate : Join a sequence of arrays along an existing axis.
            // delete : Delete elements from an array.

            // Notes
            // ---- -
            // Note that for higher dimensional inserts `obj = 0` behaves very different
            // from `obj =[0]` just like `arr[:, 0,:] = values` is different from
            // `arr[:,[0],:] = values`.

            // Examples
            // --------
            // >>> a = np.array([[1, 1], [2, 2], [3, 3]])
            // >>> a
            // array([[1, 1],
            //        [2, 2],
            //        [3, 3]])
            // >>> np.insert(a, 1, 5)
            // array([1, 5, 1, 2, 2, 3, 3])
            // >>> np.insert(a, 1, 5, axis=1)
            // array([[1, 5, 1],
            //        [2, 5, 2],
            //        [3, 5, 3]])

            // Difference between sequence and scalars:

            // >>> np.insert(a, [1], [[1],[2],[3]], axis=1)
            // array([[1, 1, 1],
            //        [2, 2, 2],
            //        [3, 3, 3]])
            // >>> np.array_equal(np.insert(a, 1, [1, 2, 3], axis=1),
            // ...                np.insert(a, [1], [[1],[2],[3]], axis=1))
            // True

            // >>> b = a.flatten()
            // >>> b
            // array([1, 1, 2, 2, 3, 3])
            // >>> np.insert(b, [2, 2], [5, 6])
            // array([1, 1, 5, 6, 2, 2, 3, 3])

            // >>> np.insert(b, slice(2, 4), [5, 6])
            // array([1, 1, 5, 2, 6, 2, 3, 3])

            // >>> np.insert(b, [2, 2], [7.13, False]) # type casting
            // array([1, 1, 7, 0, 2, 2, 3, 3])

            // >>> x = np.arange(8).reshape(2, 4)
            // >>> idx = (1, 3)
            // >>> np.insert(x, idx, 999, axis=1)
            // array([[  0, 999,   1,   2, 999,   3],
            //        [  4, 999,   5,   6, 999,   7]])

            WrapDelegate wrap = null;
            npy_intp numnew;
            ndarray newarray;

            int ndim = arr.ndim;

            NPY_ORDER arrorder = NPY_ORDER.NPY_CORDER;
            if (arr.flags.fnc)
            {
                arrorder = NPY_ORDER.NPY_FORTRANORDER; 
            }

            if (axis == null)
            {
                if (ndim != 1)
                {
                    arr = arr.ravel();
                }
                ndim = arr.ndim;
                axis = ndim - 1;
            }
            else if (ndim == 0)
            {
                //2013-09-24, 1.9
                //warnings.warn(string.Format("in the future the special handling of scalars will be removed from insert and raise an error"));
                arr = arr.Copy(order: arrorder);
                arr["..."] = _invalues;
                if (wrap != null)
                {
                    return wrap(arr);
                }
                else
                {
                    return arr;
                }
            }
            else
            {
                axis = normalize_axis_index((int)axis, ndim);
            }

            object[] slobj = BuildSliceArray(new Slice(null), ndim);
            object[] slobj2;

            long N = arr.Dim((int)axis);
            var newshape = list(arr.dims);


            ndarray indices;

            if (obj is Slice)
            {
                var _obj = obj as Slice;

                int start = _obj.start != null ? (int)_obj.start : 0;
                int? stop = _obj.stop != null ? (int)_obj.stop : (int)N;
                int? step = _obj.step != null ? (int)_obj.step : 1;

                indices = arange(start, stop, step, dtype: np.intp);
            }
            else
            {
                indices = np.array(obj);
                if (indices.TypeNum == NPY_TYPES.NPY_BOOL)
                {
                    // See also delete
                    //warnings.warn("in the future insert will treat boolean arrays and array-likes as a boolean index instead of casting it to integer");
                    indices = indices.astype(np.intp);
                    // Code after warning period:
                    //if (obj.ndim != 1)
                    //{
                    //    throw new Exception("boolean array argument obj to insert must be one dimensional");
                    //}
                    // indices = np.flatnonzero(obj);
                }
                else if (indices.ndim > 0)
                {
                    throw new Exception("index array argument obj to insert must be one dimensional or scalar");
                }
            }

            if (indices.size == 1)
            {
                dynamic index = indices.GetItem(0);

                if (index < -N || index > N)
                {
                    throw new Exception(string.Format("index {0} is out of bounds for axis {1} with size {2}", obj, axis, N));
                }
                if (index < 0)
                {
                    index += N;
                }

                // There are some object array corner cases here, but we cannot avoid
                // that:
                var values = array(_invalues, copy: false, ndmin: arr.ndim, dtype: arr.Dtype);
                if (indices.ndim == 0)
                {
                    // broadcasting is very different here, since a[:,0,:] = ... behaves
                    // very different from a[:,[0],:] = ...! This changes values so that
                    // it works likes the second case. (here a[:,0:1,:])
                    //values = np.moveaxis(values, 0, axis);
                }

                numnew = values.Dim((int)axis);
                newshape[(int)axis] += numnew;
                newarray = empty(new shape(newshape), dtype: arr.Dtype, order: arrorder);
                slobj[(int)axis] = new Slice(null, index);
                newarray[slobj] = arr[slobj];
                slobj[(int)axis] = new Slice(index, index + numnew);
                newarray[slobj] = values;
                slobj[(int)axis] = new Slice(index + numnew, None);
                slobj2 = BuildSliceArray(new Slice(null), ndim);
                slobj2[(int)axis] = new Slice(index, None);
                newarray[slobj] = arr[slobj2];
                if (wrap != null)
                {
                    return wrap(newarray);
                }
                return newarray;
            }
            else if (indices.size == 0 && !(obj is ndarray))
            {
                indices = indices.astype(np.intp);
            }

            if (!np.can_cast(indices, np.intp, "same_kind"))
            {
                // 2013-09-24, 1.9
                //warnings.warn("using a non-integer array as obj in insert will result in an error in the future");
                indices = indices.astype(np.intp);
            }

            indices[indices < 0] = indices.A(indices < 0) + N;

            numnew = len(indices);
            ndarray order = indices.ArgSort(kind: NPY_SORTKIND.NPY_MERGESORT); // stable sort
            indices[order] = indices.A(order) + np.arange(numnew, dtype:np.intp);

            newshape[(int)axis] += numnew;
            var old_mask = ones(new shape((int)newshape[(int)axis]), dtype: np.Bool);
            old_mask[indices] = false;

            newarray = empty(new shape(newshape), dtype: arr.Dtype, order: arrorder);
            slobj2 = BuildSliceArray(new Slice(null), ndim);

            slobj[(int)axis] = null;
            slobj[(int)axis] = indices;
            slobj2[(int)axis] = null;
            slobj2[(int)axis] = old_mask;
            newarray[slobj] = _invalues;
            newarray[slobj2] = arr;

            if (wrap != null)
            {
                return wrap(newarray);
            }
            return newarray;
        }

        #endregion

        #region append
        /// <summary>
        /// Append values to the end of an array.
        /// </summary>
        /// <param name="arr">Values are appended to a copy of this array.</param>
        /// <param name="values">These values are appended to a copy of 'arr'.</param>
        /// <param name="axis">The axis along which 'values' are appended.</param>
        /// <returns></returns>
        public static ndarray append(ndarray arr, dynamic values, int? axis = null)
        {
            // Append values to the end of an array.

            // Parameters
            // ----------
            // arr: array_like
            //    Values are appended to a copy of this array.
            // values : array_like
            //    These values are appended to a copy of `arr`.  It must be of the
            //    correct shape(the same shape as `arr`, excluding `axis`).If
            //     `axis` is not specified, `values` can be any shape and will be
            //     flattened before use.
            // axis: int, optional
            //     The axis along which `values` are appended.  If `axis` is not
            //     given, both `arr` and `values` are flattened before use.

            // Returns
            // ------ -
            // append : ndarray
            //     A copy of `arr` with `values` appended to `axis`.  Note that
            //     `append` does not occur in-place: a new array is allocated and
            //     filled.If `axis` is None, `out` is a flattened array.

            // See Also
            // --------
            // insert : Insert elements into an array.
            // delete : Delete elements from an array.

            // Examples
            // --------
            // >>> np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
            // array([1, 2, 3, 4, 5, 6, 7, 8, 9])

            // When `axis` is specified, `values` must have the correct shape.

            // >>> np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
            // array([[1, 2, 3],
            //        [4, 5, 6],
            //        [7, 8, 9]])
            // >>> np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)
            // Traceback(most recent call last) :
            // ...
            // ValueError: arrays must have same number of dimensions

            arr = asanyarray(arr);
            if (axis == null)
            {
                if (arr.ndim != 1)
                {
                    arr = arr.ravel();
                }
                values = ravel(values);
                axis = arr.ndim - 1;
            }

            return concatenate(new ndarray[] { arr, values }, axis: (int)axis);

        }

        #endregion
    }


}
