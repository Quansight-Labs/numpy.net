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
        #region take

        /// <summary>
        /// Take elements from an array along an axis.
        /// </summary>
        /// <param name="a">The source array.</param>
        /// <param name="indices">The indices of the values to extract.</param>
        /// <param name="axis">The axis over which to select values. By default, the flattened input array is used.</param>
        /// <param name="_out">If provided, the result will be placed in this array.</param>
        /// <param name="mode">{'raise', 'wrap', 'clip'}, optional</param>
        /// <returns></returns>
        public static ndarray take(ndarray a, ndarray indices, int? axis = null, ndarray _out = null, NPY_CLIPMODE mode = NPY_CLIPMODE.NPY_RAISE)
        {
            /*
            Take elements from an array along an axis.

            When axis is not None, this function does the same thing as "fancy"
            indexing (indexing arrays using arrays); however, it can be easier to use
            if you need elements along a given axis. A call such as
            ``np.take(arr, indices, axis=3)`` is equivalent to
            ``arr[:,:,:,indices,...]``.

            Explained without fancy indexing, this is equivalent to the following use
            of `ndindex`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of
            indices::

                Ni, Nk = a.shape[:axis], a.shape[axis+1:]
                Nj = indices.shape
                for ii in ndindex(Ni):
                    for jj in ndindex(Nj):
                        for kk in ndindex(Nk):
                            out[ii + jj + kk] = a[ii + (indices[jj],) + kk]

            Parameters
            ----------
            a : array_like (Ni..., M, Nk...)
                The source array.
            indices : array_like (Nj...)
                The indices of the values to extract.

                .. versionadded:: 1.8.0

                Also allow scalars for indices.
            axis : int, optional
                The axis over which to select values. By default, the flattened
                input array is used.
            out : ndarray, optional (Ni..., Nj..., Nk...)
                If provided, the result will be placed in this array. It should
                be of the appropriate shape and dtype.
            mode : {'raise', 'wrap', 'clip'}, optional
                Specifies how out-of-bounds indices will behave.

                * 'raise' -- raise an error (default)
                * 'wrap' -- wrap around
                * 'clip' -- clip to the range

                'clip' mode means that all indices that are too large are replaced
                by the index that addresses the last element along that axis. Note
                that this disables indexing with negative numbers.

            Returns
            -------
            out : ndarray (Ni..., Nj..., Nk...)
                The returned array has the same type as `a`.

            See Also
            --------
            compress : Take elements using a boolean mask
            ndarray.take : equivalent method

            Notes
            -----

            By eliminating the inner loop in the description above, and using `s_` to
            build simple slice objects, `take` can be expressed  in terms of applying
            fancy indexing to each 1-d slice::

                Ni, Nk = a.shape[:axis], a.shape[axis+1:]
                for ii in ndindex(Ni):
                    for kk in ndindex(Nj):
                        out[ii + s_[...,] + kk] = a[ii + s_[:,] + kk][indices]

            For this reason, it is equivalent to (but faster than) the following use
            of `apply_along_axis`::

                out = np.apply_along_axis(lambda a_1d: a_1d[indices], axis, a)

            Examples
            --------
            >>> a = [4, 3, 5, 7, 6, 8]
            >>> indices = [0, 1, 4]
            >>> np.take(a, indices)
            array([4, 3, 6])

            In this example if `a` is an ndarray, "fancy" indexing can be used.

            >>> a = np.array(a)
            >>> a[indices]
            array([4, 3, 6])

            If `indices` is not one dimensional, the output also has these dimensions.

            >>> np.take(a, [[0, 1], [2, 3]])
            array([[4, 3],
                   [5, 7]])             
            */
            
            if (axis == null)
            {
                a = a.ravel();
                axis = 0;
            }

            return NpyCoreApi.TakeFrom(a, indices, axis.Value, _out, mode);
        }

        #endregion

        #region reshape


        /// <summary>
        /// Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="a">Array to be reshaped</param>
        /// <param name="newshape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <param name="order">(Optional) {‘C’, ‘F’, ‘A’} ,Read the elements of a using this index order, and place the elements into the reshaped array using this index order. ‘C’ means to read / write the elements using C-like index order </param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy.Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        public static ndarray reshape(ndarray a, shape newshape, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            if (a == null)
            {
                throw new Exception("ndarray can't be null in reshape");
            }

            NpyArray_Dims newDims = ConvertTo_NpyArray_Dims(newshape);

            return new ndarray(numpyAPI.NpyArray_Newshape(a.Array, newDims, order));
        }

        private static NpyArray_Dims ConvertTo_NpyArray_Dims(shape newshape)
        {
            if (newshape.iDims != null)
            {
                NpyArray_Dims newDims = new NpyArray_Dims();
                newDims.ptr = new npy_intp[newshape.iDims.Length];
                newDims.len = newshape.iDims.Length;
                for (int i = 0; i < newDims.len; i++)
                {
                    newDims.ptr[i] = newshape.iDims[i];
                }
                return newDims;
            }

            return null;
        }

        #endregion

        #region choose
        /// <summary>
        /// Construct an array from an index array and a set of arrays to choose from.
        /// </summary>
        /// <param name="a">This array must contain integers in `[0, n-1]`, where `n` is the number of choices,</param>
        /// <param name="choices">sequence of arrays Choice arrays. `a` and all of the choices must be broadcastable to the same shape</param>
        /// <param name="out"> If provided, the result will be inserted into this array.</param>
        /// <param name="mode">{'raise' (default), 'wrap', 'clip'}, optional</param>
        /// <returns></returns>
        public static ndarray choose(ndarray a, IEnumerable<ndarray> choices, ndarray @out = null, NPY_CLIPMODE mode= NPY_CLIPMODE.NPY_RAISE)
        {
            /*
            Construct an array from an index array and a set of arrays to choose from.

            First of all, if confused or uncertain, definitely look at the Examples -
            in its full generality, this function is less simple than it might
            seem from the following code description (below ndi =
            `numpy.lib.index_tricks`):

            ``np.choose(a,c) == np.array([c[a[I]][I] for I in ndi.ndindex(a.shape)])``.

            But this omits some subtleties.  Here is a fully general summary:

            Given an "index" array (`a`) of integers and a sequence of `n` arrays
            (`choices`), `a` and each choice array are first broadcast, as necessary,
            to arrays of a common shape; calling these *Ba* and *Bchoices[i], i =
            0,...,n-1* we have that, necessarily, ``Ba.shape == Bchoices[i].shape``
            for each `i`.  Then, a new array with shape ``Ba.shape`` is created as
            follows:

            * if ``mode=raise`` (the default), then, first of all, each element of
              `a` (and thus `Ba`) must be in the range `[0, n-1]`; now, suppose that
              `i` (in that range) is the value at the `(j0, j1, ..., jm)` position
              in `Ba` - then the value at the same position in the new array is the
              value in `Bchoices[i]` at that same position;

            * if ``mode=wrap``, values in `a` (and thus `Ba`) may be any (signed)
              integer; modular arithmetic is used to map integers outside the range
              `[0, n-1]` back into that range; and then the new array is constructed
              as above;

            * if ``mode=clip``, values in `a` (and thus `Ba`) may be any (signed)
              integer; negative integers are mapped to 0; values greater than `n-1`
              are mapped to `n-1`; and then the new array is constructed as above.

            Parameters
            ----------
            a : int array
                This array must contain integers in `[0, n-1]`, where `n` is the number
                of choices, unless ``mode=wrap`` or ``mode=clip``, in which cases any
                integers are permissible.
            choices : sequence of arrays
                Choice arrays. `a` and all of the choices must be broadcastable to the
                same shape.  If `choices` is itself an array (not recommended), then
                its outermost dimension (i.e., the one corresponding to
                ``choices.shape[0]``) is taken as defining the "sequence".
            out : array, optional
                If provided, the result will be inserted into this array. It should
                be of the appropriate shape and dtype.
            mode : {'raise' (default), 'wrap', 'clip'}, optional
                Specifies how indices outside `[0, n-1]` will be treated:

                  * 'raise' : an exception is raised
                  * 'wrap' : value becomes value mod `n`
                  * 'clip' : values < 0 are mapped to 0, values > n-1 are mapped to n-1

            Returns
            -------
            merged_array : array
                The merged result.

            Raises
            ------
            ValueError: shape mismatch
                If `a` and each choice array are not all broadcastable to the same
                shape.

            See Also
            --------
            ndarray.choose : equivalent method

            Notes
            -----
            To reduce the chance of misinterpretation, even though the following
            "abuse" is nominally supported, `choices` should neither be, nor be
            thought of as, a single array, i.e., the outermost sequence-like container
            should be either a list or a tuple.

            Examples
            --------

            >>> choices = [[0, 1, 2, 3], [10, 11, 12, 13],
            ...   [20, 21, 22, 23], [30, 31, 32, 33]]
            >>> np.choose([2, 3, 1, 0], choices
            ... # the first element of the result will be the first element of the
            ... # third (2+1) "array" in choices, namely, 20; the second element
            ... # will be the second element of the fourth (3+1) choice array, i.e.,
            ... # 31, etc.
            ... )
            array([20, 31, 12,  3])
            >>> np.choose([2, 4, 1, 0], choices, mode='clip') # 4 goes to 3 (4-1)
            array([20, 31, 12,  3])
            >>> # because there are 4 choice arrays
            >>> np.choose([2, 4, 1, 0], choices, mode='wrap') # 4 goes to (4 mod 4)
            array([20,  1, 12,  3])
            >>> # i.e., 0

            A couple examples illustrating how choose broadcasts:

            >>> a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
            >>> choices = [-10, 10]
            >>> np.choose(a, choices)
            array([[ 10, -10,  10],
                   [-10,  10, -10],
                   [ 10, -10,  10]])

            >>> # With thanks to Anne Archibald
            >>> a = np.array([0, 1]).reshape((2,1,1))
            >>> c1 = np.array([1, 2, 3]).reshape((1,3,1))
            >>> c2 = np.array([-1, -2, -3, -4, -5]).reshape((1,1,5))
            >>> np.choose(a, (c1, c2)) # result is 2x3x5, res[0,:,:]=c1, res[1,:,:]=c2
            array([[[ 1,  1,  1,  1,  1],
                    [ 2,  2,  2,  2,  2],
                    [ 3,  3,  3,  3,  3]],
                   [[-1, -2, -3, -4, -5],
                    [-1, -2, -3, -4, -5],
                    [-1, -2, -3, -4, -5]]])             
             */

            return NpyCoreApi.Choose(a, choices.ToArray(), @out, mode);
        }

        #endregion

        #region repeat

        /// <summary>
        /// Repeat elements of an array
        /// </summary>
        /// <param name="a">Input array</param>
        /// <param name="repeats">The number of repetitions for each element.</param>
        /// <param name="axis">The axis along which to repeat values.</param>
        /// <returns></returns>
        public static ndarray repeat(object a, object repeats, int? axis = null)
        {
            /*
            Repeat elements of an array.

            Parameters
            ----------
            a : array_like
                Input array.
            repeats : int or array of ints
                The number of repetitions for each element.  `repeats` is broadcasted
                to fit the shape of the given axis.
            axis : int, optional
                The axis along which to repeat values.  By default, use the
                flattened input array, and return a flat output array.

            Returns
            -------
            repeated_array : ndarray
                Output array which has the same shape as `a`, except along
                the given axis.

            See Also
            --------
            tile : Tile an array.

            Examples
            --------
            >>> np.repeat(3, 4)
            array([3, 3, 3, 3])
            >>> x = np.array([[1,2],[3,4]])
            >>> np.repeat(x, 2)
            array([1, 1, 2, 2, 3, 3, 4, 4])
            >>> np.repeat(x, 3, axis=1)
            array([[1, 1, 1, 2, 2, 2],
                   [3, 3, 3, 4, 4, 4]])
            >>> np.repeat(x, [1, 2], axis=0)
            array([[1, 2],
                   [3, 4],
                   [3, 4]])             
            */

            var _a = asanyarray(a);
            var _repeats = asanyarray(repeats);

            if (axis == null)
            {
                _a = _a.ravel();
            }

            return NpyCoreApi.Repeat(_a, _repeats, axis.HasValue  ? axis.Value : -1);
        }

        #endregion

        #region put
        /// <summary>
        /// Replaces specified elements of an array with given values.
        /// </summary>
        /// <param name="a">Target array.</param>
        /// <param name="ind">Target indices, interpreted as integers.</param>
        /// <param name="v">Values to place in `a` at target indices.</param>
        /// <param name="mode">{'raise', 'wrap', 'clip'}, optional</param>
        /// <returns></returns>
        public static int put(ndarray a, ndarray ind, ndarray v, NPY_CLIPMODE mode = NPY_CLIPMODE.NPY_RAISE)
        {
            /*
            Replaces specified elements of an array with given values.

            The indexing works on the flattened target array. `put` is roughly
            equivalent to:

            ::

                a.flat[ind] = v

            Parameters
            ----------
            a : ndarray
                Target array.
            ind : array_like
                Target indices, interpreted as integers.
            v : array_like
                Values to place in `a` at target indices. If `v` is shorter than
                `ind` it will be repeated as necessary.
            mode : {'raise', 'wrap', 'clip'}, optional
                Specifies how out-of-bounds indices will behave.

                * 'raise' -- raise an error (default)
                * 'wrap' -- wrap around
                * 'clip' -- clip to the range

                'clip' mode means that all indices that are too large are replaced
                by the index that addresses the last element along that axis. Note
                that this disables indexing with negative numbers.

            See Also
            --------
            putmask, place

            Examples
            --------
            >>> a = np.arange(5)
            >>> np.put(a, [0, 2], [-44, -55])
            >>> a
            array([-44,   1, -55,   3,   4])

            >>> a = np.arange(5)
            >>> np.put(a, 22, -5, mode='clip')
            >>> a
            array([ 0,  1,  2,  3, -5])
            */

            int ret = NpyCoreApi.PutTo(a, v, ind, mode);
            if (ret < 0)
            {
                NpyCoreApi.CheckError();
            }

            return ret;
        }
        
        /// <summary>
        /// Replaces specified elements of an array with given values.
        /// </summary>
        /// <param name="a">Target array.</param>
        /// <param name="ind">Target indices, interpreted as integers.</param>
        /// <param name="v">Values to place in `a` at target indices.</param>
        /// <param name="mode">{'raise', 'wrap', 'clip'}, optional</param>
        /// <returns></returns>
        public static int put(ndarray a, object indices, object values, NPY_CLIPMODE mode = NPY_CLIPMODE.NPY_RAISE)
        {
            ndarray aIndices;
            ndarray aValues;

            aIndices = asanyarray(indices);
            if (aIndices == null)
            {
                aIndices = np.FromAny(indices, NpyCoreApi.DescrFromType(NPY_TYPES.NPY_INTP),
                    0, 0, NPYARRAYFLAGS.NPY_CARRAY, null);
            }
            aValues = asanyarray(values);
            if (aValues == null)
            {
                aValues = np.FromAny(values, a.Dtype, 0, 0, NPYARRAYFLAGS.NPY_CARRAY, null);
            }
            return np.put(a, aIndices, aValues, mode);
        }

        #endregion

        #region swapaxes
        /// <summary>
        /// Interchange two axes of an array.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="axis1">First axis.</param>
        /// <param name="axis2">Second axis.</param>
        /// <returns></returns>
        public static ndarray swapaxes(ndarray a, int axis1, int axis2)
        {
            /*
            Interchange two axes of an array.

            Parameters
            ----------
            a : array_like
                Input array.
            axis1 : int
                First axis.
            axis2 : int
                Second axis.

            Returns
            -------
            a_swapped : ndarray
                For NumPy >= 1.10.0, if `a` is an ndarray, then a view of `a` is
                returned; otherwise a new array is created. For earlier NumPy
                versions a view of `a` is returned only if the order of the
                axes is changed, otherwise the input array is returned.

            Examples
            --------
            >>> x = np.array([[1,2,3]])
            >>> np.swapaxes(x,0,1)
            array([[1],
                   [2],
                   [3]])

            >>> x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
            >>> x
            array([[[0, 1],
                    [2, 3]],
                   [[4, 5],
                    [6, 7]]])

            >>> np.swapaxes(x,0,2)
            array([[[0, 4],
                    [2, 6]],
                   [[1, 5],
                    [3, 7]]])             
            */

            return NpyCoreApi.SwapAxis(a, axis1, axis2);
        }

        #endregion

        #region transpose
        /// <summary>
        /// Permute the dimensions of an array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axes">list of ints, optional.By default, reverse the dimensions, otherwise permute the axes according to the values given.</param>
        /// <returns></returns>
        public static ndarray transpose(ndarray a, npy_intp[] axes = null)
        {
            /*
            Permute the dimensions of an array.

            Parameters
            ----------
            a : array_like
                Input array.
            axes : list of ints, optional
                By default, reverse the dimensions, otherwise permute the axes
                according to the values given.

            Returns
            -------
            p : ndarray
                `a` with its axes permuted.  A view is returned whenever
                possible.

            See Also
            --------
            moveaxis
            argsort

            Notes
            -----
            Use `transpose(a, argsort(axes))` to invert the transposition of tensors
            when using the `axes` keyword argument.

            Transposing a 1-D array returns an unchanged view of the original array.

            Examples
            --------
            >>> x = np.arange(4).reshape((2,2))
            >>> x
            array([[0, 1],
                   [2, 3]])

            >>> np.transpose(x)
            array([[0, 2],
                   [1, 3]])

            >>> x = np.ones((1, 2, 3))
            >>> np.transpose(x, (1, 0, 2)).shape
            (2, 1, 3)
            */


            return NpyCoreApi.Transpose(a, axes == null ? null : axes.ToArray());
        }

        /// <summary>
        /// Permute the dimensions of an array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axes">list of ints, optional.By default, reverse the dimensions, otherwise permute the axes according to the values given.</param>
        /// <returns></returns>
        public static ndarray transpose(ndarray []a, npy_intp[] axes = null)
        {
            return transpose(vstack(a), axes);
        }

#if NPY_INTP_64
        /// <summary>
        /// Permute the dimensions of an array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axes">list of ints, optional.By default, reverse the dimensions, otherwise permute the axes according to the values given.</param>
        /// <returns></returns>
        public static ndarray transpose(ndarray a, int[] axes)
        {
            if (axes != null)
            {
                List<npy_intp> _axes = new List<npy_intp>();
                foreach (var axis in axes)
                {
                    _axes.Add(axis);
                }

                return transpose(a, _axes.ToArray());
            }
            return transpose(a, (npy_intp[])null);
        }
#else
        /// <summary>
        /// Permute the dimensions of an array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axes">list of ints, optional.By default, reverse the dimensions, otherwise permute the axes according to the values given.</param>
        /// <returns></returns>
        public static ndarray transpose(ndarray a, long[] axes)
        {
            if (axes != null)
            {
                List<npy_intp> _axes = new List<npy_intp>();
                foreach (var axis in axes)
                {
                    _axes.Add((npy_intp)axis);
                }

                return transpose(a, _axes.ToArray());
            }
            return transpose(a, (npy_intp[])null);
        }
#endif
        #endregion

        #region partition

        /// <summary>
        /// Return a partitioned copy of an array.
        /// </summary>
        /// <param name="a">Array to be sorted.</param>
        /// <param name="kth">Element index to partition by.</param>
        /// <param name="axis">Axis along which to sort.</param>
        /// <param name="kind">Selection algorithm.</param>
        /// <param name="order">str or list of str, optional</param>
        /// <returns></returns>
        public static ndarray partition(ndarray a, IEnumerable<npy_intp> kth, int? axis = null, string kind = "introselect", IEnumerable<string> order = null)
        {
            /*
            Return a partitioned copy of an array.

            Creates a copy of the array with its elements rearranged in such a
            way that the value of the element in k-th position is in the
            position it would be in a sorted array. All elements smaller than
            the k-th element are moved before this element and all equal or
            greater are moved behind it. The ordering of the elements in the two
            partitions is undefined.

            .. versionadded:: 1.8.0

            Parameters
            ----------
            a : array_like
                Array to be sorted.
            kth : int or sequence of ints
                Element index to partition by. The k-th value of the element
                will be in its final sorted position and all smaller elements
                will be moved before it and all equal or greater elements behind
                it. The order of all elements in the partitions is undefined. If
                provided with a sequence of k-th it will partition all elements
                indexed by k-th  of them into their sorted position at once.
            axis : int or None, optional
                Axis along which to sort. If None, the array is flattened before
                sorting. The default is -1, which sorts along the last axis.
            kind : {'introselect'}, optional
                Selection algorithm. Default is 'introselect'.
            order : str or list of str, optional
                When `a` is an array with fields defined, this argument
                specifies which fields to compare first, second, etc.  A single
                field can be specified as a string.  Not all fields need be
                specified, but unspecified fields will still be used, in the
                order in which they come up in the dtype, to break ties.

            Returns
            -------
            partitioned_array : ndarray
                Array of the same type and shape as `a`.

            See Also
            --------
            ndarray.partition : Method to sort an array in-place.
            argpartition : Indirect partition.
            sort : Full sorting

            Notes
            -----
            The various selection algorithms are characterized by their average
            speed, worst case performance, work space size, and whether they are
            stable. A stable sort keeps items with the same key in the same
            relative order. The available algorithms have the following
            properties:

            ================= ======= ============= ============ =======
               kind            speed   worst case    work space  stable
            ================= ======= ============= ============ =======
            'introselect'        1        O(n)           0         no
            ================= ======= ============= ============ =======

            All the partition algorithms make temporary copies of the data when
            partitioning along any but the last axis.  Consequently,
            partitioning along the last axis is faster and uses less space than
            partitioning along any other axis.

            The sort order for complex numbers is lexicographic. If both the
            real and imaginary parts are non-nan then the order is determined by
            the real parts except when they are equal, in which case the order
            is determined by the imaginary parts.

            Examples
            --------
            >>> a = np.array([3, 4, 2, 1])
            >>> np.partition(a, 3)
            array([2, 1, 3, 4])

            >>> np.partition(a, (1, 3))
            array([1, 2, 3, 4])
             */

            if (axis == null)
            {
                a = a.flatten();
                axis = -1;
            }
            else
            {
                a = asanyarray(a).Copy(order: NPY_ORDER.NPY_KORDER);
            }

            int ret = NpyCoreApi.Partition(a, asanyarray(kth), axis.Value, which: NPY_SELECTKIND.NPY_INTROSELECT);

            return a;
        }

        /// <summary>
        /// Return a partitioned copy of an array.
        /// </summary>
        /// <param name="a">Array to be sorted.</param>
        /// <param name="kth">Element index to partition by.</param>
        /// <param name="axis">Axis along which to sort.</param>
        /// <param name="kind">Selection algorithm.</param>
        /// <param name="order">str or list of str, optional</param>
        /// <returns></returns>
        public static ndarray partition(ndarray a, int kth, int? axis = null, string kind = "introselect", IEnumerable<string> order = null)
        {
            return partition(a, new npy_intp[1] { kth }, axis, kind, order);
        }

        #endregion

        #region argpartition
        /// <summary>
        /// Perform an indirect partition along the given axis using the algorithm specified by the `kind` keyword.
        /// </summary>
        /// <param name="a">Array to sort</param>
        /// <param name="kth">Element index to partition by.</param>
        /// <param name="axis">Axis along which to sort.</param>
        /// <param name="kind">Selection algorithm</param>
        /// <param name="order">str or list of str, optional</param>
        /// <returns></returns>
        public static ndarray argpartition(ndarray a, IEnumerable<int> kth, int? axis = null, string kind = "introselect", IEnumerable<string> order = null)
        {
            /*
            Perform an indirect partition along the given axis using the
            algorithm specified by the `kind` keyword. It returns an array of
            indices of the same shape as `a` that index data along the given
            axis in partitioned order.

            .. versionadded:: 1.8.0

            Parameters
            ----------
            a : array_like
                Array to sort.
            kth : int or sequence of ints
                Element index to partition by. The k-th element will be in its
                final sorted position and all smaller elements will be moved
                before it and all larger elements behind it. The order all
                elements in the partitions is undefined. If provided with a
                sequence of k-th it will partition all of them into their sorted
                position at once.
            axis : int or None, optional
                Axis along which to sort. The default is -1 (the last axis). If
                None, the flattened array is used.
            kind : {'introselect'}, optional
                Selection algorithm. Default is 'introselect'
            order : str or list of str, optional
                When `a` is an array with fields defined, this argument
                specifies which fields to compare first, second, etc. A single
                field can be specified as a string, and not all fields need be
                specified, but unspecified fields will still be used, in the
                order in which they come up in the dtype, to break ties.

            Returns
            -------
            index_array : ndarray, int
                Array of indices that partition `a` along the specified axis.
                In other words, ``a[index_array]`` yields a partitioned `a`.

            See Also
            --------
            partition : Describes partition algorithms used.
            ndarray.partition : Inplace partition.
            argsort : Full indirect sort

            Notes
            -----
            See `partition` for notes on the different selection algorithms.

            Examples
            --------
            One dimensional array:

            >>> x = np.array([3, 4, 2, 1])
            >>> x[np.argpartition(x, 3)]
            array([2, 1, 3, 4])
            >>> x[np.argpartition(x, (1, 3))]
            array([1, 2, 3, 4])

            >>> x = [3, 4, 2, 1]
            >>> np.array(x)[np.argpartition(x, 3)]
            array([2, 1, 3, 4])             
            */


            if (axis == null)
            {
                a = a.ravel();
                axis = -1;
            }
            else
            {
                a = asanyarray(a).Copy(order : NPY_ORDER.NPY_KORDER);
            }
            
            ndarray ret = NpyCoreApi.ArgPartition(a, asanyarray(kth), axis.Value, which: NPY_SELECTKIND.NPY_INTROSELECT);
            return ret;
        }
        /// <summary>
        /// Perform an indirect partition along the given axis using the algorithm specified by the `kind` keyword.
        /// </summary>
        /// <param name="a">Array to sort</param>
        /// <param name="kth">Element index to partition by.</param>
        /// <param name="axis">Axis along which to sort.</param>
        /// <param name="kind">Selection algorithm</param>
        /// <param name="order">str or list of str, optional</param>
        /// <returns></returns>
        public static ndarray argpartition(ndarray a, int kth, int? axis = null, string kind = "introselect", IEnumerable<string> order = null)
        {
            return argpartition(a, new Int32[1] { kth }, axis, kind, order);
        }

        #endregion

        #region sort

        /// <summary>
        /// Return a sorted copy of an array.
        /// </summary>
        /// <param name="a">Array to be sorted.</param>
        /// <param name="axis">Axis along which to sort</param>
        /// <param name="kind">{'quicksort', 'mergesort', 'heapsort'}, optional</param>
        /// <param name="order">str or list of str, optional</param>
        /// <returns></returns>
        public static ndarray sort(object a, int? axis = -1, NPY_SORTKIND kind = NPY_SORTKIND.NPY_QUICKSORT, IEnumerable<string> order= null)
        {
            /*
            Return a sorted copy of an array.

            Parameters
            ----------
            a : array_like
                Array to be sorted.
            axis : int or None, optional
                Axis along which to sort. If None, the array is flattened before
                sorting. The default is -1, which sorts along the last axis.
            kind : {'quicksort', 'mergesort', 'heapsort'}, optional
                Sorting algorithm. Default is 'quicksort'.
            order : str or list of str, optional
                When `a` is an array with fields defined, this argument specifies
                which fields to compare first, second, etc.  A single field can
                be specified as a string, and not all fields need be specified,
                but unspecified fields will still be used, in the order in which
                they come up in the dtype, to break ties.

            Returns
            -------
            sorted_array : ndarray
                Array of the same type and shape as `a`.

            See Also
            --------
            ndarray.sort : Method to sort an array in-place.
            argsort : Indirect sort.
            lexsort : Indirect stable sort on multiple keys.
            searchsorted : Find elements in a sorted array.
            partition : Partial sort.

            Notes
            -----
            The various sorting algorithms are characterized by their average speed,
            worst case performance, work space size, and whether they are stable. A
            stable sort keeps items with the same key in the same relative
            order. The three available algorithms have the following
            properties:

            =========== ======= ============= ============ =======
               kind      speed   worst case    work space  stable
            =========== ======= ============= ============ =======
            'quicksort'    1     O(n^2)            0          no
            'mergesort'    2     O(n*log(n))      ~n/2        yes
            'heapsort'     3     O(n*log(n))       0          no
            =========== ======= ============= ============ =======

            All the sort algorithms make temporary copies of the data when
            sorting along any but the last axis.  Consequently, sorting along
            the last axis is faster and uses less space than sorting along
            any other axis.

            The sort order for complex numbers is lexicographic. If both the real
            and imaginary parts are non-nan then the order is determined by the
            real parts except when they are equal, in which case the order is
            determined by the imaginary parts.

            Previous to numpy 1.4.0 sorting real and complex arrays containing nan
            values led to undefined behaviour. In numpy versions >= 1.4.0 nan
            values are sorted to the end. The extended sort order is:

              * Real: [R, nan]
              * Complex: [R + Rj, R + nanj, nan + Rj, nan + nanj]

            where R is a non-nan real value. Complex values with the same nan
            placements are sorted according to the non-nan part if it exists.
            Non-nan values are sorted as before.

            .. versionadded:: 1.12.0

            quicksort has been changed to an introsort which will switch
            heapsort when it does not make enough progress. This makes its
            worst case O(n*log(n)).

            Examples
            --------
            >>> a = np.array([[1,4],[3,1]])
            >>> np.sort(a)                # sort along the last axis
            array([[1, 4],
                   [1, 3]])
            >>> np.sort(a, axis=None)     # sort the flattened array
            array([1, 1, 3, 4])
            >>> np.sort(a, axis=0)        # sort along the first axis
            array([[1, 1],
                   [3, 4]])

            Use the `order` keyword to specify a field to use when sorting a
            structured array:

            >>> dtype = [('name', 'S10'), ('height', float), ('age', int)]
            >>> values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),
            ...           ('Galahad', 1.7, 38)]
            >>> a = np.array(values, dtype=dtype)       # create a structured array
            >>> np.sort(a, order='height')                        # doctest: +SKIP
            array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
                   ('Lancelot', 1.8999999999999999, 38)],
                  dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])

            Sort by age, then height if ages are equal:

            >>> np.sort(a, order=['age', 'height'])               # doctest: +SKIP
            array([('Galahad', 1.7, 38), ('Lancelot', 1.8999999999999999, 38),
                   ('Arthur', 1.8, 41)],
                  dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])
            */

            ndarray _a = null;
            if (axis == null)
            {
                axis = 0;
                _a = asanyarray(a).flatten();
            }
            else
            {
                _a = asanyarray(a).Copy();
            }

            NpyCoreApi.Sort(_a, axis.Value, kind);
            return _a;
        }

        #endregion

        #region argsort
        /// <summary>
        /// Returns the indices that would sort an array.
        /// </summary>
        /// <param name="a">Array to sort.</param>
        /// <param name="axis">Axis along which to sort.</param>
        /// <param name="kind">{'quicksort', 'mergesort', 'heapsort'}, optional</param>
        /// <param name="order">str or list of str, optional</param>
        /// <returns></returns>
        public static ndarray argsort(ndarray a, int? axis = -1, NPY_SORTKIND kind = NPY_SORTKIND.NPY_QUICKSORT, IEnumerable<string> order = null)
        {
            /*
            Returns the indices that would sort an array.

            Perform an indirect sort along the given axis using the algorithm specified
            by the `kind` keyword. It returns an array of indices of the same shape as
            `a` that index data along the given axis in sorted order.

            Parameters
            ----------
            a : array_like
                Array to sort.
            axis : int or None, optional
                Axis along which to sort.  The default is -1 (the last axis). If None,
                the flattened array is used.
            kind : {'quicksort', 'mergesort', 'heapsort'}, optional
                Sorting algorithm.
            order : str or list of str, optional
                When `a` is an array with fields defined, this argument specifies
                which fields to compare first, second, etc.  A single field can
                be specified as a string, and not all fields need be specified,
                but unspecified fields will still be used, in the order in which
                they come up in the dtype, to break ties.

            Returns
            -------
            index_array : ndarray, int
                Array of indices that sort `a` along the specified axis.
                If `a` is one-dimensional, ``a[index_array]`` yields a sorted `a`.

            See Also
            --------
            sort : Describes sorting algorithms used.
            lexsort : Indirect stable sort with multiple keys.
            ndarray.sort : Inplace sort.
            argpartition : Indirect partial sort.

            Notes
            -----
            See `sort` for notes on the different sorting algorithms.

            As of NumPy 1.4.0 `argsort` works with real/complex arrays containing
            nan values. The enhanced sort order is documented in `sort`.

            Examples
            --------
            One dimensional array:

            >>> x = np.array([3, 1, 2])
            >>> np.argsort(x)
            array([1, 2, 0])

            Two-dimensional array:

            >>> x = np.array([[0, 3], [2, 2]])
            >>> x
            array([[0, 3],
                   [2, 2]])

            >>> np.argsort(x, axis=0)  # sorts along first axis (down)
            array([[0, 1],
                   [1, 0]])

            >>> np.argsort(x, axis=1)  # sorts along last axis (across)
            array([[0, 1],
                   [0, 1]])

            Indices of the sorted elements of a N-dimensional array:

            >>> ind = np.unravel_index(np.argsort(x, axis=None), x.shape)
            >>> ind
            (array([0, 1, 1, 0]), array([0, 0, 1, 1]))
            >>> x[ind]  # same as np.sort(x, axis=None)
            array([0, 2, 2, 3])

            Sorting with keys:

            >>> x = np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
            >>> x
            array([(1, 0), (0, 1)],
                  dtype=[('x', '<i4'), ('y', '<i4')])

            >>> np.argsort(x, order=('x','y'))
            array([1, 0])

            >>> np.argsort(x, order=('y','x'))
            array([0, 1])
            */

            if (axis == null)
            {
                axis = 0;
                a = a.ravel();
            }

            return NpyCoreApi.ArgSort(a, axis.Value, kind);

        }

        #endregion

        #region argmax

        /// <summary>
        /// Returns the indices of the maximum values along an axis.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">By default, the index is into the flattened array, otherwise along the specified axis.</param>
        /// <param name="out">If provided, the result will be inserted into this array.</param>
        /// <returns></returns>
        public static ndarray argmax(ndarray a, int? axis= null, ndarray @out = null)
        {
            /*
            Returns the indices of the maximum values along an axis.

            Parameters
            ----------
            a : array_like
                Input array.
            axis : int, optional
                By default, the index is into the flattened array, otherwise
                along the specified axis.
            out : array, optional
                If provided, the result will be inserted into this array. It should
                be of the appropriate shape and dtype.

            Returns
            -------
            index_array : ndarray of ints
                Array of indices into the array. It has the same shape as `a.shape`
                with the dimension along `axis` removed.

            See Also
            --------
            ndarray.argmax, argmin
            amax : The maximum value along a given axis.
            unravel_index : Convert a flat index into an index tuple.

            Notes
            -----
            In case of multiple occurrences of the maximum values, the indices
            corresponding to the first occurrence are returned.

            Examples
            --------
            >>> a = np.arange(6).reshape(2,3)
            >>> a
            array([[0, 1, 2],
                   [3, 4, 5]])
            >>> np.argmax(a)
            5
            >>> np.argmax(a, axis=0)
            array([1, 1, 1])
            >>> np.argmax(a, axis=1)
            array([2, 2])

            Indexes of the maximal elements of a N-dimensional array:

            >>> ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
            >>> ind
            (1, 2)
            >>> a[ind]
            5

            >>> b = np.arange(6)
            >>> b[1] = 5
            >>> b
            array([0, 5, 2, 3, 4, 5])
            >>> np.argmax(b)  # Only the first occurrence is returned.
            1             
            */

            if (axis == null)
            {
                axis = 0;
                a = a.ravel();
            }

            return NpyCoreApi.ArrayArgMax(a, axis.Value, @out);
        }

        #endregion

        #region argmin
        /// <summary>
        /// Returns the indices of the minimum values along an axis.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <param name="axis">By default, the index is into the flattened array, otherwise along the specified axis.</param>
        /// <param name="out">If provided, the result will be inserted into this array</param>
        /// <returns></returns>
        public static ndarray argmin(ndarray a, int? axis = null, ndarray @out = null)
        {
            /*
            Returns the indices of the minimum values along an axis.

            Parameters
            ----------
            a : array_like
                Input array.
            axis : int, optional
                By default, the index is into the flattened array, otherwise
                along the specified axis.
            out : array, optional
                If provided, the result will be inserted into this array. It should
                be of the appropriate shape and dtype.

            Returns
            -------
            index_array : ndarray of ints
                Array of indices into the array. It has the same shape as `a.shape`
                with the dimension along `axis` removed.

            See Also
            --------
            ndarray.argmin, argmax
            amin : The minimum value along a given axis.
            unravel_index : Convert a flat index into an index tuple.

            Notes
            -----
            In case of multiple occurrences of the minimum values, the indices
            corresponding to the first occurrence are returned.

            Examples
            --------
            >>> a = np.arange(6).reshape(2,3)
            >>> a
            array([[0, 1, 2],
                   [3, 4, 5]])
            >>> np.argmin(a)
            0
            >>> np.argmin(a, axis=0)
            array([0, 0, 0])
            >>> np.argmin(a, axis=1)
            array([0, 0])

            Indices of the minimum elements of a N-dimensional array:

            >>> ind = np.unravel_index(np.argmin(a, axis=None), a.shape)
            >>> ind
            (0, 0)
            >>> a[ind]
            0

            >>> b = np.arange(6)
            >>> b[4] = 0
            >>> b
            array([0, 1, 2, 3, 0, 5])
            >>> np.argmin(b)  # Only the first occurrence is returned.
            0             
            */

            if (axis == null)
            {
                axis = 0;
                a = a.ravel();
            }

            return NpyCoreApi.ArrayArgMin(a, axis.Value, @out);

        }

        #endregion

        #region searchsorted
        /// <summary>
        /// Find indices where elements should be inserted to maintain order.
        /// </summary>
        /// <param name="a">1-D array_like</param>
        /// <param name="v">Values to insert into `a`</param>
        /// <param name="side"> {'left', 'right'}</param>
        /// <param name="sorter">Optional array of integer indices that sort array a into ascending order.They are typically the result of argsort.</param>
        /// <returns></returns>
        public static ndarray searchsorted(ndarray a, ndarray v, NPY_SEARCHSIDE side = NPY_SEARCHSIDE.NPY_SEARCHLEFT, ndarray sorter = null)
        {
            /*
            Find indices where elements should be inserted to maintain order.

            Find the indices into a sorted array `a` such that, if the
            corresponding elements in `v` were inserted before the indices, the
            order of `a` would be preserved.

            Assuming that `a` is sorted:

            ======  ============================
            `side`  returned index `i` satisfies
            ======  ============================
            left    ``a[i-1] < v <= a[i]``
            right   ``a[i-1] <= v < a[i]``
            ======  ============================

            Parameters
            ----------
            a : 1-D array_like
                Input array. If `sorter` is None, then it must be sorted in
                ascending order, otherwise `sorter` must be an array of indices
                that sort it.
            v : array_like
                Values to insert into `a`.
            side : {'left', 'right'}, optional
                If 'left', the index of the first suitable location found is given.
                If 'right', return the last such index.  If there is no suitable
                index, return either 0 or N (where N is the length of `a`).
            sorter : 1-D array_like, optional
                Optional array of integer indices that sort array a into ascending
                order. They are typically the result of argsort.

                .. versionadded:: 1.7.0

            Returns
            -------
            indices : array of ints
                Array of insertion points with the same shape as `v`.

            See Also
            --------
            sort : Return a sorted copy of an array.
            histogram : Produce histogram from 1-D data.

            Notes
            -----
            Binary search is used to find the required insertion points.

            As of NumPy 1.4.0 `searchsorted` works with real/complex arrays containing
            `nan` values. The enhanced sort order is documented in `sort`.

            This function is a faster version of the builtin python `bisect.bisect_left`
            (``side='left'``) and `bisect.bisect_right` (``side='right'``) functions,
            which is also vectorized in the `v` argument.

            Examples
            --------
            >>> np.searchsorted([1,2,3,4,5], 3)
            2
            >>> np.searchsorted([1,2,3,4,5], 3, side='right')
            3
            >>> np.searchsorted([1,2,3,4,5], [-10, 10, 2, 3])
            array([0, 5, 1, 2])
            */


            return NpyCoreApi.Searchsorted(a, v, side);
        }
        
        /// <summary>
        /// Find indices where elements should be inserted to maintain order.
        /// </summary>
        /// <param name="a">1-D array_like</param>
        /// <param name="v">Values to insert into `a`</param>
        /// <param name="side"> {'left', 'right'}</param>
        /// <param name="sorter">Optional array of integer indices that sort array a into ascending order.They are typically the result of argsort.</param>
        /// <returns></returns>
        public static ndarray searchsorted(this ndarray a, object keys, NPY_SEARCHSIDE side = NPY_SEARCHSIDE.NPY_SEARCHLEFT)
        {
            ndarray aKeys = (keys as ndarray);
            if (aKeys == null)
            {
                aKeys = np.FromAny(keys, a.Dtype, 0, 0, NPYARRAYFLAGS.NPY_CARRAY, null);
            }

            return np.searchsorted(a, aKeys, side);
        }

        #endregion

        #region resize
        /// <summary>
        /// Return a new array with the specified shape.
        /// </summary>
        /// <param name="a">Array to be resized</param>
        /// <param name="new_shape">Shape of resized array</param>
        /// <returns></returns>
        public static ndarray resize(ndarray a, shape new_shape)
        {
            return np.resize(a, new_shape.iDims);
        }

        /// <summary>
        /// Return a new array with the specified shape.
        /// </summary>
        /// <param name="a">Array to be resized</param>
        /// <param name="newdims">Shape of resized array</param>
        /// <returns></returns>
        public static ndarray resize(ndarray a, npy_intp[] newdims)
        {
            /*
             Return a new array with the specified shape.

             If the new array is larger than the original array, then the new
             array is filled with repeated copies of `a`.  Note that this behavior
             is different from a.resize(new_shape) which fills with zeros instead
             of repeated copies of `a`.

             Parameters
             ----------
             a : array_like
                 Array to be resized.

             new_shape : int or tuple of int
                 Shape of resized array.

             Returns
             -------
             reshaped_array : ndarray
                 The new array is formed from the data in the old array, repeated
                 if necessary to fill out the required number of elements.  The
                 data are repeated in the order that they are stored in memory.

             See Also
             --------
             ndarray.resize : resize an array in-place.

             Examples
             --------
             >>> a=np.array([[0,1],[2,3]])
             >>> np.resize(a,(2,3))
             array([[0, 1, 2],
                    [3, 0, 1]])
             >>> np.resize(a,(1,4))
             array([[0, 1, 2, 3]])
             >>> np.resize(a,(2,4))
             array([[0, 1, 2, 3],
                    [0, 1, 2, 3]])             
             */


            shape new_shape = new shape(newdims);

            a = a.ravel();
            int Na = len(a);

            long total_size = 1;
            foreach (npy_intp dim in newdims)
            {
                total_size *= dim;
            }

            if (Na == 0 || total_size == 0)
            {
                return np.zeros(new_shape, a.Dtype);
            }

            int n_copies = Convert.ToInt32(total_size / Na);
            long extra = total_size % Na;

            if (extra != 0)
            {
                n_copies = n_copies + 1;
                extra = Na - extra;
            }

            List<ndarray> alist = new List<ndarray>();
            for (int i = 0; i < n_copies; i++)
            {
                alist.Add(a);
            }

            a = concatenate(alist);
            if (extra > 0)
            {
                a = a.A(":" + (-extra).ToString());
            }

            return reshape(a, new_shape);
        }

        #endregion

        #region squeeze

        /// <summary>
        /// Remove single-dimensional entries from the shape of an array
        /// </summary>
        /// <param name="a">Input data</param>
        /// <param name="axis">Selects a subset of the single-dimensional entries in the shape</param>
        /// <returns></returns>
        public static ndarray squeeze(ndarray a, int? axis = null)
        {
            /*
            Remove single-dimensional entries from the shape of an array.

            Parameters
            ----------
            a : array_like
                Input data.
            axis : None or int or tuple of ints, optional
                .. versionadded:: 1.7.0

                Selects a subset of the single-dimensional entries in the
                shape. If an axis is selected with shape entry greater than
                one, an error is raised.

            Returns
            -------
            squeezed : ndarray
                The input array, but with all or a subset of the
                dimensions of length 1 removed. This is always `a` itself
                or a view into `a`.

            Raises
            ------
            ValueError
                If `axis` is not `None`, and an axis being squeezed is not of length 1

            See Also
            --------
            expand_dims : The inverse operation, adding singleton dimensions
            reshape : Insert, remove, and combine dimensions, and resize existing ones

            Examples
            --------
            >>> x = np.array([[[0], [1], [2]]])
            >>> x.shape
            (1, 3, 1)
            >>> np.squeeze(x).shape
            (3,)
            >>> np.squeeze(x, axis=0).shape
            (3, 1)
            >>> np.squeeze(x, axis=1).shape
            Traceback (most recent call last):
            ...
            ValueError: cannot select an axis to squeeze out which has size not equal to one
            >>> np.squeeze(x, axis=2).shape
            (1, 3)             
            */
            if (axis != null)
            {
                return NpyCoreApi.SqueezeSelected(a, axis.Value);
            }
            else
            {
                return NpyCoreApi.Squeeze(a);
            }
        }

        #endregion

        #region diagonal
        /// <summary>
        /// Return specified diagonals
        /// </summary>
        /// <param name="a">Array from which the diagonals are taken</param>
        /// <param name="offset">Offset of the diagonal from the main diagonal</param>
        /// <param name="axis1">Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken</param>
        /// <param name="axis2">Axis to be used as the second axis of the 2-D sub-arrays from which the diagonals should be taken</param>
        /// <returns></returns>
        public static ndarray diagonal(ndarray a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            /*
            Return specified diagonals.

            If `a` is 2-D, returns the diagonal of `a` with the given offset,
            i.e., the collection of elements of the form ``a[i, i+offset]``.  If
            `a` has more than two dimensions, then the axes specified by `axis1`
            and `axis2` are used to determine the 2-D sub-array whose diagonal is
            returned.  The shape of the resulting array can be determined by
            removing `axis1` and `axis2` and appending an index to the right equal
            to the size of the resulting diagonals.

            In versions of NumPy prior to 1.7, this function always returned a new,
            independent array containing a copy of the values in the diagonal.

            In NumPy 1.7 and 1.8, it continues to return a copy of the diagonal,
            but depending on this fact is deprecated. Writing to the resulting
            array continues to work as it used to, but a FutureWarning is issued.

            Starting in NumPy 1.9 it returns a read-only view on the original array.
            Attempting to write to the resulting array will produce an error.

            In some future release, it will return a read/write view and writing to
            the returned array will alter your original array.  The returned array
            will have the same type as the input array.

            If you don't write to the array returned by this function, then you can
            just ignore all of the above.

            If you depend on the current behavior, then we suggest copying the
            returned array explicitly, i.e., use ``np.diagonal(a).copy()`` instead
            of just ``np.diagonal(a)``. This will work with both past and future
            versions of NumPy.

            Parameters
            ----------
            a : array_like
                Array from which the diagonals are taken.
            offset : int, optional
                Offset of the diagonal from the main diagonal.  Can be positive or
                negative.  Defaults to main diagonal (0).
            axis1 : int, optional
                Axis to be used as the first axis of the 2-D sub-arrays from which
                the diagonals should be taken.  Defaults to first axis (0).
            axis2 : int, optional
                Axis to be used as the second axis of the 2-D sub-arrays from
                which the diagonals should be taken. Defaults to second axis (1).

            Returns
            -------
            array_of_diagonals : ndarray
                If `a` is 2-D and not a `matrix`, a 1-D array of the same type as `a`
                containing the diagonal is returned. If `a` is a `matrix`, a 1-D
                array containing the diagonal is returned in order to maintain
                backward compatibility.
                If ``a.ndim > 2``, then the dimensions specified by `axis1` and `axis2`
                are removed, and a new axis inserted at the end corresponding to the
                diagonal.

            Raises
            ------
            ValueError
                If the dimension of `a` is less than 2.

            See Also
            --------
            diag : MATLAB work-a-like for 1-D and 2-D arrays.
            diagflat : Create diagonal arrays.
            trace : Sum along diagonals.

            Examples
            --------
            >>> a = np.arange(4).reshape(2,2)
            >>> a
            array([[0, 1],
                   [2, 3]])
            >>> a.diagonal()
            array([0, 3])
            >>> a.diagonal(1)
            array([1])

            A 3-D example:

            >>> a = np.arange(8).reshape(2,2,2); a
            array([[[0, 1],
                    [2, 3]],
                   [[4, 5],
                    [6, 7]]])
            >>> a.diagonal(0, # Main diagonals of two arrays created by skipping
            ...            0, # across the outer(left)-most axis last and
            ...            1) # the "middle" (row) axis first.
            array([[0, 6],
                   [1, 7]])

            The sub-arrays whose main diagonals we just obtained; note that each
            corresponds to fixing the right-most (column) axis, and that the
            diagonals are "packed" in rows.

            >>> a[:,:,0] # main diagonal is [0 6]
            array([[0, 2],
                   [4, 6]])
            >>> a[:,:,1] # main diagonal is [1 7]
            array([[1, 3],
                   [5, 7]])             
            */

            int n = a.ndim;
            npy_intp[] newaxes = new npy_intp[n];

            // Check the axes
            if (n < 2)
            {
                throw new ArgumentException("array.ndim must be >= 2");
            }
            if (axis1 < 0)
            {
                axis1 += n;
            }
            if (axis2 < 0)
            {
                axis2 += n;
            }
            if (axis1 == axis2 ||
                axis1 < 0 || axis1 >= n ||
                axis2 < 0 || axis2 >= n)
            {
                throw new ArgumentException(
                    String.Format("axis1(={0}) and axis2(={1}) must be different and within range (nd={2})",
                        axis1, axis2, n));
            }

            // Transpose to be axis1 and axis2 at the end
            newaxes[n - 2] = (npy_intp)axis1;
            newaxes[n - 1] = (npy_intp)axis2;
            int pos = 0;
            for (int i = 0; i < n; i++)
            {
                if (i != axis1 && i != axis2)
                {
                    newaxes[pos++] = (npy_intp)i;
                }
            }
            ndarray newarray = np.transpose(a, newaxes);

            if (n == 2)
            {
                long n1, n2, start, stop, step;

                n1 = newarray.Dim(0);
                n2 = newarray.Dim(1);
                step = n2 + 1;
                if (offset < 0)
                {
                    start = -n2 * offset;
                    stop = Math.Min(2, n1 + offset) * (n2 + 1) - n2 * offset;
                }
                else
                {
                    start = offset;
                    stop = Math.Min(n1, n2 - offset) * (n2 + 1) + offset;
                }

                Slice slice = new Slice(start, stop, step);
                return (ndarray)newarray.Flat[slice];
            }
            else
            {
                // my_diagonal = []
                // for i in range(s[0]):
                // my_diagonal.append(diagonal(a[i], offset))
                // return array(my_diagonal);
                dtype typecode = newarray.Dtype;
                List<ndarray> my_diagonal = new List<ndarray>();
                long n1 = newarray.Dim(0);
                for (long i = 0; i < n1; i++)
                {
                    ndarray sub = newarray.A(i);
                    my_diagonal.Add(sub.diagonal(offset, n - 3, n - 2));
                }
                return np.FromAny(my_diagonal, typecode, 0, 0, 0, null);
            }
        }

        #endregion

        #region trace
        /// <summary>
        /// Return the sum along diagonals of the array.
        /// </summary>
        /// <param name="a">Input array, from which the diagonals are taken.</param>
        /// <param name="offset">Offset of the diagonal from the main diagonal.</param>
        /// <param name="axis1">Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken</param>
        /// <param name="axis2">Axis to be used as the second axis of the 2-D sub-arrays from which the diagonals should be taken</param>
        /// <param name="dtype">Determines the data-type of the returned array and of the accumulator where the elements are summed.</param>
        /// <param name="out">Array into which the output is placed</param>
        /// <returns></returns>
        public static ndarray trace(ndarray a, int offset = 0, int axis1 = 0, int axis2 = 1, dtype dtype = null, ndarray @out = null)
        {
            /*
            Return the sum along diagonals of the array.

            If `a` is 2-D, the sum along its diagonal with the given offset
            is returned, i.e., the sum of elements ``a[i,i+offset]`` for all i.

            If `a` has more than two dimensions, then the axes specified by axis1 and
            axis2 are used to determine the 2-D sub-arrays whose traces are returned.
            The shape of the resulting array is the same as that of `a` with `axis1`
            and `axis2` removed.

            Parameters
            ----------
            a : array_like
                Input array, from which the diagonals are taken.
            offset : int, optional
                Offset of the diagonal from the main diagonal. Can be both positive
                and negative. Defaults to 0.
            axis1, axis2 : int, optional
                Axes to be used as the first and second axis of the 2-D sub-arrays
                from which the diagonals should be taken. Defaults are the first two
                axes of `a`.
            dtype : dtype, optional
                Determines the data-type of the returned array and of the accumulator
                where the elements are summed. If dtype has the value None and `a` is
                of integer type of precision less than the default integer
                precision, then the default integer precision is used. Otherwise,
                the precision is the same as that of `a`.
            out : ndarray, optional
                Array into which the output is placed. Its type is preserved and
                it must be of the right shape to hold the output.

            Returns
            -------
            sum_along_diagonals : ndarray
                If `a` is 2-D, the sum along the diagonal is returned.  If `a` has
                larger dimensions, then an array of sums along diagonals is returned.

            See Also
            --------
            diag, diagonal, diagflat

            Examples
            --------
            >>> np.trace(np.eye(3))
            3.0
            >>> a = np.arange(8).reshape((2,2,2))
            >>> np.trace(a)
            array([6, 8])

            >>> a = np.arange(24).reshape((2,2,2,3))
            >>> np.trace(a).shape
            (2, 3)             
            */

            var D = np.diagonal(a, offset, axis1, axis2);
            var S = D.Sum(axis:-1);
            return S;

        }

        #endregion

        #region ravel 
        /// <summary>
        /// Return a contiguous flattened array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="order">{'C','F', 'A', 'K'}, optional</param>
        /// <returns></returns>
        public static ndarray ravel(ndarray a, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            /*
            Return a contiguous flattened array.

            A 1-D array, containing the elements of the input, is returned.  A copy is
            made only if needed.

            As of NumPy 1.10, the returned array will have the same type as the input
            array. (for example, a masked array will be returned for a masked array
            input)

            Parameters
            ----------
            a : array_like
                Input array.  The elements in `a` are read in the order specified by
                `order`, and packed as a 1-D array.
            order : {'C','F', 'A', 'K'}, optional

                The elements of `a` are read using this index order. 'C' means
                to index the elements in row-major, C-style order,
                with the last axis index changing fastest, back to the first
                axis index changing slowest.  'F' means to index the elements
                in column-major, Fortran-style order, with the
                first index changing fastest, and the last index changing
                slowest. Note that the 'C' and 'F' options take no account of
                the memory layout of the underlying array, and only refer to
                the order of axis indexing.  'A' means to read the elements in
                Fortran-like index order if `a` is Fortran *contiguous* in
                memory, C-like order otherwise.  'K' means to read the
                elements in the order they occur in memory, except for
                reversing the data when strides are negative.  By default, 'C'
                index order is used.

            Returns
            -------
            y : array_like
                If `a` is a matrix, y is a 1-D ndarray, otherwise y is an array of
                the same subtype as `a`. The shape of the returned array is
                ``(a.size,)``. Matrices are special cased for backward
                compatibility.

            See Also
            --------
            ndarray.flat : 1-D iterator over an array.
            ndarray.flatten : 1-D array copy of the elements of an array
                              in row-major order.
            ndarray.reshape : Change the shape of an array without changing its data.

            Notes
            -----
            In row-major, C-style order, in two dimensions, the row index
            varies the slowest, and the column index the quickest.  This can
            be generalized to multiple dimensions, where row-major order
            implies that the index along the first axis varies slowest, and
            the index along the last quickest.  The opposite holds for
            column-major, Fortran-style index ordering.

            When a view is desired in as many cases as possible, ``arr.reshape(-1)``
            may be preferable.

            Examples
            --------
            It is equivalent to ``reshape(-1, order=order)``.

            >>> x = np.array([[1, 2, 3], [4, 5, 6]])
            >>> print(np.ravel(x))
            [1 2 3 4 5 6]

            >>> print(x.reshape(-1))
            [1 2 3 4 5 6]

            >>> print(np.ravel(x, order='F'))
            [1 4 2 5 3 6]

            When ``order`` is 'A', it will preserve the array's 'C' or 'F' ordering:

            >>> print(np.ravel(x.T))
            [1 4 2 5 3 6]
            >>> print(np.ravel(x.T, order='A'))
            [1 2 3 4 5 6]

            When ``order`` is 'K', it will preserve orderings that are neither 'C'
            nor 'F', but won't reverse axes:

            >>> a = np.arange(3)[::-1]; a
            array([2, 1, 0])
            >>> a.ravel(order='C')
            array([2, 1, 0])
            >>> a.ravel(order='K')
            array([2, 1, 0])

            >>> a = np.arange(12).reshape(2,3,2).swapaxes(1,2); a
            array([[[ 0,  2,  4],
                    [ 1,  3,  5]],
                   [[ 6,  8, 10],
                    [ 7,  9, 11]]])
            >>> a.ravel(order='C')
            array([ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11])
            >>> a.ravel(order='K')
            array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])             
            */

            return NpyCoreApi.Ravel(a, order);
        }

        /// <summary>
        /// Return a contiguous flattened array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="order">{'C','F', 'A', 'K'}, optional</param>
        /// <returns></returns>
        public static ndarray ravel(System.Array a, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            return ravel(asanyarray(a), order);
        }

        /// <summary>
        /// Return a contiguous flattened array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="order">{'C','F', 'A', 'K'}, optional</param>
        /// <returns></returns>
        private static ndarray ravel(dynamic values, dtype dtype = null)
        {
            return np.array(values, dtype: dtype, copy: true, order: NPY_ORDER.NPY_ANYORDER).flatten();
        }

        #endregion

        #region nonzero
        /// <summary>
        /// Return the indices of the elements that are non-zero.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <returns></returns>
        public static ndarray[] nonzero(ndarray a)
        {
            /*
            Return the indices of the elements that are non-zero.

            Returns a tuple of arrays, one for each dimension of `a`,
            containing the indices of the non-zero elements in that
            dimension. The values in `a` are always tested and returned in
            row-major, C-style order. The corresponding non-zero
            values can be obtained with::

                a[nonzero(a)]

            To group the indices by element, rather than dimension, use::

                transpose(nonzero(a))

            The result of this is always a 2-D array, with a row for
            each non-zero element.

            Parameters
            ----------
            a : array_like
                Input array.

            Returns
            -------
            tuple_of_arrays : tuple
                Indices of elements that are non-zero.

            See Also
            --------
            flatnonzero :
                Return indices that are non-zero in the flattened version of the input
                array.
            ndarray.nonzero :
                Equivalent ndarray method.
            count_nonzero :
                Counts the number of non-zero elements in the input array.

            Examples
            --------
            >>> x = np.array([[1,0,0], [0,2,0], [1,1,0]])
            >>> x
            array([[1, 0, 0],
                   [0, 2, 0],
                   [1, 1, 0]])
            >>> np.nonzero(x)
            (array([0, 1, 2, 2]), array([0, 1, 0, 1]))

            >>> x[np.nonzero(x)]
            array([1, 2, 1, 1])
            >>> np.transpose(np.nonzero(x))
            array([[0, 0],
                   [1, 1],
                   [2, 0],
                   [2, 1])

            A common use for ``nonzero`` is to find the indices of an array, where
            a condition is True.  Given an array `a`, the condition `a` > 3 is a
            boolean array and since False is interpreted as 0, np.nonzero(a > 3)
            yields the indices of the `a` where the condition is true.

            >>> a = np.array([[1,2,3],[4,5,6],[7,8,9]])
            >>> a > 3
            array([[False, False, False],
                   [ True,  True,  True],
                   [ True,  True,  True]])
            >>> np.nonzero(a > 3)
            (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

            The ``nonzero`` method of the boolean array can also be called.

            >>> (a > 3).nonzero()
            (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
            */

            return NpyCoreApi.NonZero(a);
        }

        #endregion

        #region shape
        /// <summary>
        /// Return the shape of an array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns></returns>
        public static NumpyDotNet.shape shape(ndarray a)
        {
            /*
            Return the shape of an array.

            Parameters
            ----------
            a : array_like
                Input array.

            Returns
            -------
            shape : tuple of ints
                The elements of the shape tuple give the lengths of the
                corresponding array dimensions.

            See Also
            --------
            alen
            ndarray.shape : Equivalent array method.

            Examples
            --------
            >>> np.shape(np.eye(3))
            (3, 3)
            >>> np.shape([[1, 2]])
            (1, 2)
            >>> np.shape([0])
            (1,)
            >>> np.shape(0)
            ()

            >>> a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
            >>> np.shape(a)
            (2,)
            >>> a.shape
            (2,)             
            */

            return new shape(a.Array.dimensions, a.Array.nd);
        }

        #endregion

        #region compress
        /// <summary>
        /// Return selected slices of an array along given axis.
        /// </summary>
        /// <param name="condition">Array that selects which entries to return.</param>
        /// <param name="a">Array from which to extract a part.</param>
        /// <param name="axis">Axis along which to take slices.</param>
        /// <param name="out">Output array, if specified</param>
        /// <returns></returns>
        public static ndarray compress(ndarray condition, ndarray a, int? axis = null, ndarray @out = null)
        {
            /*
            Return selected slices of an array along given axis.

            When working along a given axis, a slice along that axis is returned in
            `output` for each index where `condition` evaluates to True. When
            working on a 1-D array, `compress` is equivalent to `extract`.

            Parameters
            ----------
            condition : 1-D array of bools
                Array that selects which entries to return. If len(condition)
                is less than the size of `a` along the given axis, then output is
                truncated to the length of the condition array.
            a : array_like
                Array from which to extract a part.
            axis : int, optional
                Axis along which to take slices. If None (default), work on the
                flattened array.
            out : ndarray, optional
                Output array.  Its type is preserved and it must be of the right
                shape to hold the output.

            Returns
            -------
            compressed_array : ndarray
                A copy of `a` without the slices along axis for which `condition`
                is false.

            See Also
            --------
            take, choose, diag, diagonal, select
            ndarray.compress : Equivalent method in ndarray
            np.extract: Equivalent method when working on 1-D arrays
            numpy.doc.ufuncs : Section "Output arguments"

            Examples
            --------
            >>> a = np.array([[1, 2], [3, 4], [5, 6]])
            >>> a
            array([[1, 2],
                   [3, 4],
                   [5, 6]])
            >>> np.compress([0, 1], a, axis=0)
            array([[3, 4]])
            >>> np.compress([False, True, True], a, axis=0)
            array([[3, 4],
                   [5, 6]])
            >>> np.compress([False, True], a, axis=1)
            array([[2],
                   [4],
                   [6]])

            Working on the flattened array does not return slices along an axis but
            selects elements.

            >>> np.compress([False, True], a)
            array([2])             
            */
   

            if (condition.ndim != 1)
            {
                throw new ArgumentException("condition must be 1-d array");
            }

            ndarray indexes = np.nonzero(condition)[0];

            return np.take(a, indexes, axis, @out, NPY_CLIPMODE.NPY_RAISE);
        }
        /// <summary>
        /// Return selected slices of an array along given axis.
        /// </summary>
        /// <param name="condition">Array that selects which entries to return.</param>
        /// <param name="a">Array from which to extract a part.</param>
        /// <param name="axis">Axis along which to take slices.</param>
        /// <param name="out">Output array, if specified</param>
        /// <returns></returns>
        public static ndarray compress(object condition, ndarray a, object axis = null, ndarray @out = null)
        {
            ndarray aCondition = np.FromAny(condition, null, 0, 0, 0, null);
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            return np.compress(aCondition, a, iAxis, @out);
        }

        #endregion

        #region clip
        /// <summary>
        /// Clip (limit) the values in an array.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_min">Minimum value</param>
        /// <param name="a_max">maximum  value</param>
        /// <param name="out">The results will be placed in this array.</param>
        /// <returns></returns>
        public static ndarray clip(ndarray a, object a_min, object a_max, ndarray @out = null)
        {
            return a.Clip(a_min, a_max, @out);
        }


        #endregion

        #region sum
        /// <summary>
        /// Sum of array elements over a given axis.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis or axes along which a sum is performed.</param>
        /// <param name="dtype">The type of the returned array and of the accumulator in which the elements are summed.</param>
        /// <param name="ret">Alternative output array in which to place the result</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray sum(ndarray a, int? axis = null, dtype dtype = null, ndarray ret = null, bool keepdims = false)
        {
            /*
            Sum of array elements over a given axis.

            Parameters
            ----------
            a : array_like
                Elements to sum.
            axis : None or int or tuple of ints, optional
                Axis or axes along which a sum is performed.  The default,
                axis=None, will sum all of the elements of the input array.  If
                axis is negative it counts from the last to the first axis.

                .. versionadded:: 1.7.0

                If axis is a tuple of ints, a sum is performed on all of the axes
                specified in the tuple instead of a single axis or all the axes as
                before.
            dtype : dtype, optional
                The type of the returned array and of the accumulator in which the
                elements are summed.  The dtype of `a` is used by default unless `a`
                has an integer dtype of less precision than the default platform
                integer.  In that case, if `a` is signed then the platform integer
                is used while if `a` is unsigned then an unsigned integer of the
                same precision as the platform integer is used.
            out : ndarray, optional
                Alternative output array in which to place the result. It must have
                the same shape as the expected output, but the type of the output
                values will be cast if necessary.
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the input array.

                If the default value is passed, then `keepdims` will not be
                passed through to the `sum` method of sub-classes of
                `ndarray`, however any non-default value will be.  If the
                sub-class' method does not implement `keepdims` any
                exceptions will be raised.

            Returns
            -------
            sum_along_axis : ndarray
                An array with the same shape as `a`, with the specified
                axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
                is returned.  If an output array is specified, a reference to
                `out` is returned.

            See Also
            --------
            ndarray.sum : Equivalent method.

            cumsum : Cumulative sum of array elements.

            trapz : Integration of array values using the composite trapezoidal rule.

            mean, average

            Notes
            -----
            Arithmetic is modular when using integer types, and no error is
            raised on overflow.

            The sum of an empty array is the neutral element 0:

            >>> np.sum([])
            0.0

            Examples
            --------
            >>> np.sum([0.5, 1.5])
            2.0
            >>> np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
            1
            >>> np.sum([[0, 1], [0, 5]])
            6
            >>> np.sum([[0, 1], [0, 5]], axis=0)
            array([0, 6])
            >>> np.sum([[0, 1], [0, 5]], axis=1)
            array([1, 5])

            If the accumulator is too small, overflow occurs:

            >>> np.ones(128, dtype=np.int8).sum(dtype=np.int8)
            -128             
            */

            if (axis == null)
            {
                a = a.ravel();
                axis = 0;
            }

            if (dtype == null)
            {
                dtype = a.Dtype;
            }

            return NpyCoreApi.Sum(a, axis.Value, dtype, ret, keepdims);
        }

        #endregion

        #region any
        /// <summary>
        /// helper function to convert the results of np.any to a boolean value
        /// </summary>
        /// <param name="a"></param>
        /// <param name="axis"></param>
        /// <param name="out"></param>
        /// <param name="keepdims"></param>
        /// <returns></returns>
        public static bool anyb(object a, object axis = null, ndarray @out = null, bool keepdims = false)
        {
            var any = np.any(a, axis, @out, keepdims);
            if (any != null)
            {
                if (any.size > 1)
                {
                    throw new ValueError("Attempt to take boolean from array");
                }
                bool b = (bool)any.GetItem(0);
                return b;
            }
            return false;
        }
        /// <summary>
        /// Test whether any array element along a given axis evaluates to True.
        /// </summary>
        /// <param name="a">Input array or object</param>
        /// <param name="axis">Axis or axes along which a logical OR reduction is performed.</param>
        /// <param name="out">Alternate output array in which to place the result.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray any(object a, object axis = null, ndarray @out=null, bool keepdims = false)
        {
            /*
            Test whether any array element along a given axis evaluates to True.

            Returns single boolean unless `axis` is not ``None``

            Parameters
            ----------
            a : array_like
                Input array or object that can be converted to an array.
            axis : None or int or tuple of ints, optional
                Axis or axes along which a logical OR reduction is performed.
                The default (`axis` = `None`) is to perform a logical OR over all
                the dimensions of the input array. `axis` may be negative, in
                which case it counts from the last to the first axis.

                .. versionadded:: 1.7.0

                If this is a tuple of ints, a reduction is performed on multiple
                axes, instead of a single axis or all the axes as before.
            out : ndarray, optional
                Alternate output array in which to place the result.  It must have
                the same shape as the expected output and its type is preserved
                (e.g., if it is of type float, then it will remain so, returning
                1.0 for True and 0.0 for False, regardless of the type of `a`).
                See `doc.ufuncs` (Section "Output arguments") for details.

            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the input array.

                If the default value is passed, then `keepdims` will not be
                passed through to the `any` method of sub-classes of
                `ndarray`, however any non-default value will be.  If the
                sub-class' method does not implement `keepdims` any
                exceptions will be raised.

            Returns
            -------
            any : bool or ndarray
                A new boolean or `ndarray` is returned unless `out` is specified,
                in which case a reference to `out` is returned.

            See Also
            --------
            ndarray.any : equivalent method

            all : Test whether all elements along a given axis evaluate to True.

            Notes
            -----
            Not a Number (NaN), positive infinity and negative infinity evaluate
            to `True` because these are not equal to zero.

            Examples
            --------
            >>> np.any([[True, False], [True, True]])
            True

            >>> np.any([[True, False], [False, False]], axis=0)
            array([ True, False])

            >>> np.any([-1, 0, 5])
            True

            >>> np.any(np.nan)
            True

            >>> o=np.array([False])
            >>> z=np.any([-1, 4, 5], out=o)
            >>> z, o
            (array([ True]), array([ True]))
            >>> # Check now that z is a reference to o
            >>> z is o
            True
            >>> id(z), id(o) # identity of z and o              # doctest: +SKIP
            (191614240, 191614240)
            */

            ndarray aa = np.FromAny(a, null, 0, 0, 0, null);
            if (aa.Size == 0)
            {
                return np.array(new bool[] { false });
            }


            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            return NpyCoreApi.ArrayAny(aa, iAxis, @out);
        }

        #endregion

        #region all
        /// <summary>
        /// helper function to convert the results of np.all to a boolean value
        /// </summary>
        /// <param name="a"></param>
        /// <param name="axis"></param>
        /// <param name="out"></param>
        /// <param name="keepdims"></param>
        /// <returns></returns>
        public static bool allb(object a, object axis = null, ndarray @out = null, bool keepdims = false)
        {
            var all = np.all(a, axis, @out, keepdims);
            if (all != null)
            {
                if (all.size > 1)
                {
                    throw new ValueError("Attempt to take boolean from array");
                }
                bool b = (bool)all.GetItem(0);
                return b;
            }
            return false;
        }
        /// <summary>
        /// Test whether all array elements along a given axis evaluate to True.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <param name="axis">Axis or axes along which a logical AND reduction is performed.</param>
        /// <param name="out">Alternate output array in which to place the result.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray all(object a, object axis = null, ndarray @out = null, bool keepdims = false)
        {
            /*
            Test whether all array elements along a given axis evaluate to True.

            Parameters
            ----------
            a : array_like
                Input array or object that can be converted to an array.
            axis : None or int or tuple of ints, optional
                Axis or axes along which a logical AND reduction is performed.
                The default (`axis` = `None`) is to perform a logical AND over all
                the dimensions of the input array. `axis` may be negative, in
                which case it counts from the last to the first axis.

                .. versionadded:: 1.7.0

                If this is a tuple of ints, a reduction is performed on multiple
                axes, instead of a single axis or all the axes as before.
            out : ndarray, optional
                Alternate output array in which to place the result.
                It must have the same shape as the expected output and its
                type is preserved (e.g., if ``dtype(out)`` is float, the result
                will consist of 0.0's and 1.0's).  See `doc.ufuncs` (Section
                "Output arguments") for more details.

            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the input array.

                If the default value is passed, then `keepdims` will not be
                passed through to the `all` method of sub-classes of
                `ndarray`, however any non-default value will be.  If the
                sub-class' method does not implement `keepdims` any
                exceptions will be raised.

            Returns
            -------
            all : ndarray, bool
                A new boolean or array is returned unless `out` is specified,
                in which case a reference to `out` is returned.

            See Also
            --------
            ndarray.all : equivalent method

            any : Test whether any element along a given axis evaluates to True.

            Notes
            -----
            Not a Number (NaN), positive infinity and negative infinity
            evaluate to `True` because these are not equal to zero.

            Examples
            --------
            >>> np.all([[True,False],[True,True]])
            False

            >>> np.all([[True,False],[True,True]], axis=0)
            array([ True, False])

            >>> np.all([-1, 4, 5])
            True

            >>> np.all([1.0, np.nan])
            True

            >>> o=np.array([False])
            >>> z=np.all([-1, 4, 5], out=o)
            >>> id(z), id(o), z                             # doctest: +SKIP
            (28293632, 28293632, array([ True]))             
            */

            ndarray aa = np.FromAny(a, null, 0, 0, 0, null);
            if (aa.Size == 0)
            {
                return np.array(new bool[]{true});
            }

            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);

            return NpyCoreApi.ArrayAll(aa, iAxis, @out);
        }

        #endregion

        #region cumsum
        /// <summary>
        /// Return the cumulative sum of the elements along a given axis.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">Axis along which the cumulative sum is computed.</param>
        /// <param name="dtype">Type of the returned array and of the accumulator in which the elements are summed.</param>
        /// <param name="ret">Alternative output array in which to place the result</param>
        /// <returns></returns>
        public static ndarray cumsum(ndarray a, int? axis = null, dtype dtype = null, ndarray ret = null)
        {
            /*
            Return the cumulative sum of the elements along a given axis.

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
            cumsum_along_axis : ndarray.
                A new array holding the result is returned unless `out` is
                specified, in which case a reference to `out` is returned. The
                result has the same size as `a`, and the same shape as `a` if
                `axis` is not None or `a` is a 1-d array.


            See Also
            --------
            sum : Sum array elements.

            trapz : Integration of array values using the composite trapezoidal rule.

            diff :  Calculate the n-th discrete difference along given axis.

            Notes
            -----
            Arithmetic is modular when using integer types, and no error is
            raised on overflow.

            Examples
            --------
            >>> a = np.array([[1,2,3], [4,5,6]])
            >>> a
            array([[1, 2, 3],
                   [4, 5, 6]])
            >>> np.cumsum(a)
            array([ 1,  3,  6, 10, 15, 21])
            >>> np.cumsum(a, dtype=float)     # specifies type of output value(s)
            array([  1.,   3.,   6.,  10.,  15.,  21.])

            >>> np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
            array([[1, 2, 3],
                   [5, 7, 9]])
            >>> np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
            array([[ 1,  3,  6],
                   [ 4,  9, 15]])
            */

            if (axis == null)
            {
                axis = 0;
                a = a.ravel();
            }

            return NpyCoreApi.CumSum(a, axis.Value, dtype, ret);
        }

        #endregion

        #region ptp
        /// <summary>
        /// Range of values (maximum - minimum) along an axis.
        /// </summary>
        /// <param name="a">Input values.</param>
        /// <param name="axis">Axis along which to find the peaks.</param>
        /// <param name="out">Alternative output array in which to place the result.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. </param>
        /// <returns></returns>
        public static ndarray ptp(ndarray a, object axis = null, ndarray @out=null, bool keepdims = false)
        {
            /*
            Range of values (maximum - minimum) along an axis.

            The name of the function comes from the acronym for 'peak to peak'.

            Parameters
            ----------
            a : array_like
                Input values.
            axis : None or int or tuple of ints, optional
                Axis along which to find the peaks.  By default, flatten the
                array.  `axis` may be negative, in
                which case it counts from the last to the first axis.

                .. versionadded:: 1.15.0

                If this is a tuple of ints, a reduction is performed on multiple
                axes, instead of a single axis or all the axes as before.
            out : array_like
                Alternative output array in which to place the result. It must
                have the same shape and buffer length as the expected output,
                but the type of the output values will be cast if necessary.

            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the input array.

                If the default value is passed, then `keepdims` will not be
                passed through to the `ptp` method of sub-classes of
                `ndarray`, however any non-default value will be.  If the
                sub-class' method does not implement `keepdims` any
                exceptions will be raised.

            Returns
            -------
            ptp : ndarray
                A new array holding the result, unless `out` was
                specified, in which case a reference to `out` is returned.

            Examples
            --------
            >>> x = np.arange(4).reshape((2,2))
            >>> x
            array([[0, 1],
                   [2, 3]])

            >>> np.ptp(x, axis=0)
            array([2, 2])

            >>> np.ptp(x, axis=1)
            array([1, 1])
            */

            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);

            ndarray arr = NpyCoreApi.CheckAxis(a, ref iAxis, 0);
            ndarray a1 = np.amax(arr, iAxis, null, keepdims:keepdims);
            ndarray a2 = np.amin(arr, iAxis, null, keepdims:keepdims);

            ndarray ret = NpyCoreApi.PerformUFUNC(UFuncOperation.subtract, a1, a2, @out, null);
            return ret;
        }

        #endregion

        #region amax
        /// <summary>
        /// Return the maximum of an array or maximum along an axis.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis or axes along which to operate.</param>
        /// <param name="out">Alternative output array in which to place the result.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray amax(object a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {

            /*
            Return the maximum of an array or maximum along an axis.

            Parameters
            ----------
            a : array_like
                Input data.
            axis : None or int or tuple of ints, optional
                Axis or axes along which to operate.  By default, flattened input is
                used.

                .. versionadded:: 1.7.0

                If this is a tuple of ints, the maximum is selected over multiple axes,
                instead of a single axis or all the axes as before.
            out : ndarray, optional
                Alternative output array in which to place the result.  Must
                be of the same shape and buffer length as the expected output.
                See `doc.ufuncs` (Section "Output arguments") for more details.

            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the input array.

                If the default value is passed, then `keepdims` will not be
                passed through to the `amax` method of sub-classes of
                `ndarray`, however any non-default value will be.  If the
                sub-class' method does not implement `keepdims` any
                exceptions will be raised.

            Returns
            -------
            amax : ndarray or scalar
                Maximum of `a`. If `axis` is None, the result is a scalar value.
                If `axis` is given, the result is an array of dimension
                ``a.ndim - 1``.

            See Also
            --------
            amin :
                The minimum value of an array along a given axis, propagating any NaNs.
            nanmax :
                The maximum value of an array along a given axis, ignoring any NaNs.
            maximum :
                Element-wise maximum of two arrays, propagating any NaNs.
            fmax :
                Element-wise maximum of two arrays, ignoring any NaNs.
            argmax :
                Return the indices of the maximum values.

            nanmin, minimum, fmin

            Notes
            -----
            NaN values are propagated, that is if at least one item is NaN, the
            corresponding max value will be NaN as well. To ignore NaN values
            (MATLAB behavior), please use nanmax.

            Don't use `amax` for element-wise comparison of 2 arrays; when
            ``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than
            ``amax(a, axis=0)``.

            Examples
            --------
            >>> a = np.arange(4).reshape((2,2))
            >>> a
            array([[0, 1],
                   [2, 3]])
            >>> np.amax(a)           # Maximum of the flattened array
            3
            >>> np.amax(a, axis=0)   # Maxima along the first axis
            array([2, 3])
            >>> np.amax(a, axis=1)   # Maxima along the second axis
            array([1, 3])

            >>> b = np.arange(5, dtype=float)
            >>> b[2] = np.NaN
            >>> np.amax(b)
            nan
            >>> np.nanmax(b)
            4.0             
            */

            var arr = asanyarray(a);

            if (axis == null)
            {
                arr = arr.ravel();
                axis = 0;
                
            }

            var resultArray = NpyCoreApi.ArrayMax(arr, axis.Value, @out, keepdims);
            return resultArray;
        }

        /// <summary>
        /// Return the maximum of an array or maximum along an axis.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis or axes along which to operate.</param>
        /// <param name="out">Alternative output array in which to place the result.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray max(object a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.amax(a, axis, @out, keepdims);
        }

        private static object _max(object arr)
        {
            var resultArray = np.amax(arr);
            return DefaultArrayHandlers.GetArrayHandler(resultArray.TypeNum).ConvertToUpgradedValue(resultArray.GetItem(0));
        }

        #endregion

        #region amin
        /// <summary>
        /// Return the minimum of an array or minimum along an axis.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis or axes along which to operate.</param>
        /// <param name="out">Alternative output array in which to place the result.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray amin(object a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {
            /*
            Return the minimum of an array or minimum along an axis.

            Parameters
            ----------
            a : array_like
                Input data.
            axis : None or int or tuple of ints, optional
                Axis or axes along which to operate.  By default, flattened input is
                used.

                .. versionadded:: 1.7.0

                If this is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before.
            out : ndarray, optional
                Alternative output array in which to place the result.  Must
                be of the same shape and buffer length as the expected output.
                See `doc.ufuncs` (Section "Output arguments") for more details.

            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the input array.

                If the default value is passed, then `keepdims` will not be
                passed through to the `amin` method of sub-classes of
                `ndarray`, however any non-default value will be.  If the
                sub-class' method does not implement `keepdims` any
                exceptions will be raised.

            Returns
            -------
            amin : ndarray or scalar
                Minimum of `a`. If `axis` is None, the result is a scalar value.
                If `axis` is given, the result is an array of dimension
                ``a.ndim - 1``.

            See Also
            --------
            amax :
                The maximum value of an array along a given axis, propagating any NaNs.
            nanmin :
                The minimum value of an array along a given axis, ignoring any NaNs.
            minimum :
                Element-wise minimum of two arrays, propagating any NaNs.
            fmin :
                Element-wise minimum of two arrays, ignoring any NaNs.
            argmin :
                Return the indices of the minimum values.

            nanmax, maximum, fmax

            Notes
            -----
            NaN values are propagated, that is if at least one item is NaN, the
            corresponding min value will be NaN as well. To ignore NaN values
            (MATLAB behavior), please use nanmin.

            Don't use `amin` for element-wise comparison of 2 arrays; when
            ``a.shape[0]`` is 2, ``minimum(a[0], a[1])`` is faster than
            ``amin(a, axis=0)``.

            Examples
            --------
            >>> a = np.arange(4).reshape((2,2))
            >>> a
            array([[0, 1],
                   [2, 3]])
            >>> np.amin(a)           # Minimum of the flattened array
            0
            >>> np.amin(a, axis=0)   # Minima along the first axis
            array([0, 1])
            >>> np.amin(a, axis=1)   # Minima along the second axis
            array([0, 2])

            >>> b = np.arange(5, dtype=float)
            >>> b[2] = np.NaN
            >>> np.amin(b)
            nan
            >>> np.nanmin(b)
            0.0             
            */

            var arr = asanyarray(a);

            if (axis == null)
            {
                arr = arr.ravel();
                axis = 0;

            }

            var resultArray = NpyCoreApi.ArrayMin(arr, axis.Value, @out, keepdims);
            return resultArray;
        }

        /// <summary>
        /// Return the minimum of an array or minimum along an axis.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis or axes along which to operate.</param>
        /// <param name="out">Alternative output array in which to place the result.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray min(object a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.amin(a, axis, @out, keepdims);
        }

        private static object _min(object arr)
        {
            var resultArray = np.amin(arr);
            return DefaultArrayHandlers.GetArrayHandler(resultArray.TypeNum).ConvertToUpgradedValue(resultArray.GetItem(0));
        }

        #endregion

        #region alen
        /// <summary>
        /// Return the length of the first dimension of the input array.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <returns></returns>
        public static int alen(ndarray a)
        {
            /*
            Return the length of the first dimension of the input array.

            Parameters
            ----------
            a : array_like
               Input array.

            Returns
            -------
            alen : int
               Length of the first dimension of `a`.

            See Also
            --------
            shape, size

            Examples
            --------
            >>> a = np.zeros((7,4,5))
            >>> a.shape[0]
            7
            >>> np.alen(a)
            7             
            */

            return len(a);
        }

        #endregion

        #region prod
        /// <summary>
        /// Return the product of array elements over a given axis.
        /// </summary>
        /// <param name="a">Input data</param>
        /// <param name="axis">Axis or axes along which a product is performed.</param>
        /// <param name="dtype">The type of the returned array, as well as of the accumulator in which the elements are multiplied.</param>
        /// <param name="out">Alternative output array in which to place the result.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray prod(ndarray a, int? axis = null, dtype dtype = null, ndarray @out = null, bool keepdims = false)
        {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            return NpyCoreApi.Prod(a, iAxis, dtype, @out, keepdims);
        }

        #endregion

        #region cumprod
        /// <summary>
        /// Return the cumulative product of elements along a given axis.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">Axis along which the cumulative product is computed.</param>
        /// <param name="dtype">Type of the returned array, as well as of the accumulator in which the elements are multiplied. </param>
        /// <param name="out">Alternative output array in which to place the result. </param>
        /// <returns></returns>
        public static ndarray cumprod(object a, int? axis = null, dtype dtype = null, ndarray @out = null)
        {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);

            return NpyCoreApi.CumProd(asanyarray(a), iAxis, dtype, @out);
        }
        #endregion

        #region ndim
        /// <summary>
        /// Return the number of dimensions of an array.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <returns></returns>
        public static int ndim(ndarray a)
        {
            /*
            Return the number of dimensions of an array.

            Parameters
            ----------
            a : array_like
                Input array.  If it is not already an ndarray, a conversion is
                attempted.

            Returns
            -------
            number_of_dimensions : int
                The number of dimensions in `a`.  Scalars are zero-dimensional.

            See Also
            --------
            ndarray.ndim : equivalent method
            shape : dimensions of array
            ndarray.shape : dimensions of array

            Examples
            --------
            >>> np.ndim([[1,2,3],[4,5,6]])
            2
            >>> np.ndim(np.array([[1,2,3],[4,5,6]]))
            2
            >>> np.ndim(1)
            0             
            */

            return a.ndim;
        }

        #endregion

        #region size
        /// <summary>
        /// Return the number of elements along a given axis.
        /// </summary>
        /// <param name="a">Input data</param>
        /// <param name="axis">Axis along which the elements are counted.</param>
        /// <returns></returns>
        public static npy_intp size(ndarray a, int? axis = null)
        {
            /*
            Return the number of elements along a given axis.

            Parameters
            ----------
            a : array_like
                Input data.
            axis : int, optional
                Axis along which the elements are counted.  By default, give
                the total number of elements.

            Returns
            -------
            element_count : int
                Number of elements along the specified axis.

            See Also
            --------
            shape : dimensions of array
            ndarray.shape : dimensions of array
            ndarray.size : number of elements in array

            Examples
            --------
            >>> a = np.array([[1,2,3],[4,5,6]])
            >>> np.size(a)
            6
            >>> np.size(a,1)
            3
            >>> np.size(a,0)
            2             
            */

            if (axis == null)
            {
                return a.size;
            }
            else
            {
                if (axis.Value < 0 || axis.Value >= a.ndim)
                {
                    throw new Exception(string.Format("np.size: axis {0} out of range for array with {1} dimenstions", axis.Value, a.ndim));
                }
                return a.Dim(axis.Value);
            }
        }

        #endregion

        #region around
        /// <summary>
        /// Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="a">Input data</param>
        /// <param name="decimals">Number of decimal places to round to (default: 0).</param>
        /// <param name="out">Alternative output array in which to place the result.</param>
        /// <returns></returns>
        public static ndarray around(ndarray a, int decimals = 0, ndarray @out = null)
        {
            /*
            Evenly round to the given number of decimals.

            Parameters
            ----------
            a : array_like
                Input data.
            decimals : int, optional
                Number of decimal places to round to (default: 0).  If
                decimals is negative, it specifies the number of positions to
                the left of the decimal point.
            out : ndarray, optional
                Alternative output array in which to place the result. It must have
                the same shape as the expected output, but the type of the output
                values will be cast if necessary. See `doc.ufuncs` (Section
                "Output arguments") for details.

            Returns
            -------
            rounded_array : ndarray
                An array of the same type as `a`, containing the rounded values.
                Unless `out` was specified, a new array is created.  A reference to
                the result is returned.

                The real and imaginary parts of complex numbers are rounded
                separately.  The result of rounding a float is a float.

            See Also
            --------
            ndarray.round : equivalent method

            ceil, fix, floor, rint, trunc


            Notes
            -----
            For values exactly halfway between rounded decimal values, NumPy
            rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0,
            -0.5 and 0.5 round to 0.0, etc. Results may also be surprising due
            to the inexact representation of decimal fractions in the IEEE
            floating point standard [1]_ and errors introduced when scaling
            by powers of ten.

            References
            ----------
            .. [1] "Lecture Notes on the Status of  IEEE 754", William Kahan,
                   http://www.cs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF
            .. [2] "How Futile are Mindless Assessments of
                   Roundoff in Floating-Point Computation?", William Kahan,
                   http://www.cs.berkeley.edu/~wkahan/Mindless.pdf

            Examples
            --------
            >>> np.around([0.37, 1.64])
            array([ 0.,  2.])
            >>> np.around([0.37, 1.64], decimals=1)
            array([ 0.4,  1.6])
            >>> np.around([.5, 1.5, 2.5, 3.5, 4.5]) # rounds to nearest even value
            array([ 0.,  2.,  2.,  4.,  4.])
            >>> np.around([1,2,3,11], decimals=1) # ndarray of ints is returned
            array([ 1,  2,  3, 11])
            >>> np.around([1,2,3,11], decimals=-1)
            array([ 0,  0,  0, 10])
            */

            if (!a.IsMathFunctionCapable)
            {
                ArrayTypeNotSupported(a);
            }


            return a.Round(decimals, @out) as ndarray;
        }
        /// <summary>
        /// Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="a">Input data</param>
        /// <param name="decimals">Number of decimal places to round to (default: 0).</param>
        /// <param name="out">Alternative output array in which to place the result.</param>
        /// <returns></returns>
        public static ndarray round_(ndarray a, int decimals = 0, ndarray @out = null)
        {
            return np.around(a, decimals, @out);
        }
        /// <summary>
        /// Evenly round to the given number of decimals.
        /// </summary>
        /// <param name="a">Input data</param>
        /// <param name="decimals">Number of decimal places to round to (default: 0).</param>
        /// <param name="out">Alternative output array in which to place the result.</param>
        /// <returns></returns>
        public static ndarray round(ndarray a, int decimals = 0, ndarray @out = null)
        {
            return np.around(a, decimals, @out);
        }
        #endregion

        #region mean

        /// <summary>
        /// Compute the arithmetic mean along the specified axis.
        /// </summary>
        /// <param name="input">Array containing numbers whose mean is desired</param>
        /// <param name="axis">Axis or axes along which the means are computed. The default is to compute the mean of the flattened arra</param>
        /// <param name="dtype">Type to use in computing the mean. For integer inputs, the default is float64; for floating point inputs, it is the same as the input dtype.</param>
        /// <returns>ndarray, see dtype parameter above</returns>
        public static ndarray mean(object a, int? axis = null, dtype dtype = null, bool keepdims = false)
        {
            /*
            Compute the arithmetic mean along the specified axis.

            Returns the average of the array elements.  The average is taken over
            the flattened array by default, otherwise over the specified axis.
            `float64` intermediate and return values are used for integer inputs.

            Parameters
            ----------
            a : array_like
                Array containing numbers whose mean is desired. If `a` is not an
                array, a conversion is attempted.
            axis : None or int or tuple of ints, optional
                Axis or axes along which the means are computed. The default is to
                compute the mean of the flattened array.

                .. versionadded:: 1.7.0

                If this is a tuple of ints, a mean is performed over multiple axes,
                instead of a single axis or all the axes as before.
            dtype : data-type, optional
                Type to use in computing the mean.  For integer inputs, the default
                is `float64`; for floating point inputs, it is the same as the
                input dtype.
            out : ndarray, optional
                Alternate output array in which to place the result.  The default
                is ``None``; if provided, it must have the same shape as the
                expected output, but the type will be cast if necessary.
                See `doc.ufuncs` for details.

            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the input array.

                If the default value is passed, then `keepdims` will not be
                passed through to the `mean` method of sub-classes of
                `ndarray`, however any non-default value will be.  If the
                sub-class' method does not implement `keepdims` any
                exceptions will be raised.

            Returns
            -------
            m : ndarray, see dtype parameter above
                If `out=None`, returns a new array containing the mean values,
                otherwise a reference to the output array is returned.

            See Also
            --------
            average : Weighted average
            std, var, nanmean, nanstd, nanvar

            Notes
            -----
            The arithmetic mean is the sum of the elements along the axis divided
            by the number of elements.

            Note that for floating-point input, the mean is computed using the
            same precision the input has.  Depending on the input data, this can
            cause the results to be inaccurate, especially for `float32` (see
            example below).  Specifying a higher-precision accumulator using the
            `dtype` keyword can alleviate this issue.

            By default, `float16` results are computed using `float32` intermediates
            for extra precision.

            Examples
            --------
            >>> a = np.array([[1, 2], [3, 4]])
            >>> np.mean(a)
            2.5
            >>> np.mean(a, axis=0)
            array([ 2.,  3.])
            >>> np.mean(a, axis=1)
            array([ 1.5,  3.5])

            In single precision, `mean` can be inaccurate:

            >>> a = np.zeros((2, 512*512), dtype=np.float32)
            >>> a[0, :] = 1.0
            >>> a[1, :] = 0.1
            >>> np.mean(a)
            0.54999924

            Computing the mean in float64 is more accurate:

            >>> np.mean(a, dtype=np.float64)
            0.55000000074505806
            */

            var arr = asanyarray(a);

            long rcount = _count_reduce_items(arr, axis);
            if (rcount == 0)
            {
                Console.WriteLine("mean of empty slice");
            }

            if (dtype == null)
            {
                dtype = result_type(arr.TypeNum);
            }

            var ret = np.sum(arr, axis, dtype, keepdims: keepdims);
            ret = np.true_divide(ret, rcount);
            return ret;
   
        }

        /// <summary>
        /// Compute the weighted average along the specified axis.
        /// </summary>
        /// <param name="a">Array containing data to be averaged.</param>
        /// <param name="axis">Axis or axes along which to average a.</param>
        /// <param name="weights">An array of weights associated with the values in a. Each value in a contributes to the average according to its associated weight.</param>
        /// <returns></returns>
        public static ndarray average(object a, int? axis = null, object weights = null)
        {
            var average_result = average(a, axis, weights, false);
            return average_result.retval;
        }

        /// <summary>
        /// Compute the weighted average along the specified axis.
        /// </summary>
        /// <param name="a">Array containing data to be averaged.</param>
        /// <param name="axis">Axis or axes along which to average a.</param>
        /// <param name="weights">An array of weights associated with the values in a. Each value in a contributes to the average according to its associated weight.</param>
        /// <returns></returns>
        public static (ndarray retval, ndarray sum_of_weights) average(object a, int? axis, object weights, bool returned = false)
        {
            ndarray avg = null;
            ndarray scl = null;

            var arr = asanyarray(a);
            dtype result_dtype = result_type(arr.TypeNum);

            if (weights == null)
            {
                avg =  mean(arr, axis);

                scl = asanyarray(arr.size / avg.size).astype(result_dtype);
            }
            else
            {
                var wgt = np.asanyarray(weights);

                // Sanity checks
                if (arr.shape != wgt.shape)
                {
                    if (axis == null)
                    {
                        throw new TypeError("Axis must be specified when shapes of a and weights differ.");
                    }

                    if (wgt.ndim != 1)
                    {
                        throw new TypeError("1D weights expected when shapes of a and weights differ.");
                    }
                    if (wgt.shape.iDims[0] != arr.shape.iDims[axis.Value])
                    {
                        throw new ValueError("Length of weights not compatible with specified axis.");

                    }
                    // setup wgt to broadcast along axis

                    List<npy_intp> newShape = new List<npy_intp>();
                    for (int i = 0; i < arr.ndim-1; i++)
                        newShape.Add(1);
                    newShape.AddRange(wgt.shape.iDims);
  
                    wgt = np.broadcast_to(wgt, new shape(newShape.ToArray()));
                    wgt = wgt.SwapAxes(-1, axis.Value);
                }

                scl = wgt.Sum(axis: axis, dtype: result_dtype);
                if (np.anyb(scl == 0.0))
                {
                    throw new ZeroDivisionError("Weights sum to zero, can't be normalized");

                }

                var ax = np.multiply(a, wgt).Sum(axis).astype(result_dtype);
                avg = np.divide(ax, scl);

            }

            if (returned)
            {
                if (scl.shape != avg.shape)
                {
                    scl = np.broadcast_to(scl, avg.shape).Copy();
                }

                return (avg, scl);
            }
            else
            {
                return (avg, null);
            }

        }

        #endregion

        #region std

        /// <summary>
        /// Compute the standard deviation along the specified axis.
        /// </summary>
        /// <param name="a">Calculate the standard deviation of these values.</param>
        /// <param name="axis">Axis or axes along which the standard deviation is computed.</param>
        /// <param name="dtype">Type to use in computing the standard deviation.</param>
        /// <param name="ddof">Means Delta Degrees of Freedom.</param>
        /// <param name="keep_dims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray std(ndarray a, int? axis = null, dtype dtype = null, int ddof = 0, bool keep_dims = false)
        {
            /*
            Compute the standard deviation along the specified axis.

            Returns the standard deviation, a measure of the spread of a distribution,
            of the array elements. The standard deviation is computed for the
            flattened array by default, otherwise over the specified axis.

            Parameters
            ----------
            a : array_like
                Calculate the standard deviation of these values.
            axis : None or int or tuple of ints, optional
                Axis or axes along which the standard deviation is computed. The
                default is to compute the standard deviation of the flattened array.

                .. versionadded:: 1.7.0

                If this is a tuple of ints, a standard deviation is performed over
                multiple axes, instead of a single axis or all the axes as before.
            dtype : dtype, optional
                Type to use in computing the standard deviation. For arrays of
                integer type the default is float64, for arrays of float types it is
                the same as the array type.
            out : ndarray, optional
                Alternative output array in which to place the result. It must have
                the same shape as the expected output but the type (of the calculated
                values) will be cast if necessary.
            ddof : int, optional
                Means Delta Degrees of Freedom.  The divisor used in calculations
                is ``N - ddof``, where ``N`` represents the number of elements.
                By default `ddof` is zero.
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the input array.

                If the default value is passed, then `keepdims` will not be
                passed through to the `std` method of sub-classes of
                `ndarray`, however any non-default value will be.  If the
                sub-class' method does not implement `keepdims` any
                exceptions will be raised.

            Returns
            -------
            standard_deviation : ndarray, see dtype parameter above.
                If `out` is None, return a new array containing the standard deviation,
                otherwise return a reference to the output array.

            See Also
            --------
            var, mean, nanmean, nanstd, nanvar
            numpy.doc.ufuncs : Section "Output arguments"

            Notes
            -----
            The standard deviation is the square root of the average of the squared
            deviations from the mean, i.e., ``std = sqrt(mean(abs(x - x.mean())**2))``.

            The average squared deviation is normally calculated as
            ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is specified,
            the divisor ``N - ddof`` is used instead. In standard statistical
            practice, ``ddof=1`` provides an unbiased estimator of the variance
            of the infinite population. ``ddof=0`` provides a maximum likelihood
            estimate of the variance for normally distributed variables. The
            standard deviation computed in this function is the square root of
            the estimated variance, so even with ``ddof=1``, it will not be an
            unbiased estimate of the standard deviation per se.

            Note that, for complex numbers, `std` takes the absolute
            value before squaring, so that the result is always real and nonnegative.

            For floating-point input, the *std* is computed using the same
            precision the input has. Depending on the input data, this can cause
            the results to be inaccurate, especially for float32 (see example below).
            Specifying a higher-accuracy accumulator using the `dtype` keyword can
            alleviate this issue.

            Examples
            --------
            >>> a = np.array([[1, 2], [3, 4]])
            >>> np.std(a)
            1.1180339887498949
            >>> np.std(a, axis=0)
            array([ 1.,  1.])
            >>> np.std(a, axis=1)
            array([ 0.5,  0.5])

            In single precision, std() can be inaccurate:

            >>> a = np.zeros((2, 512*512), dtype=np.float32)
            >>> a[0, :] = 1.0
            >>> a[1, :] = 0.1
            >>> np.std(a)
            0.45000005

            Computing the standard deviation in float64 is more accurate:

            >>> np.std(a, dtype=np.float64)
            0.44999999925494177             
            */

            ndarray ret = var(a, axis, dtype, ddof, keep_dims);
            return np.sqrt(ret);

#if OLDER_VERSION
            ndarray x, mean, tmp;
            object result;
            long n;
            npy_intp[] newshape;

            int _axis = 0;
            if (axis == null)
            {
                a = a.flatten();
                axis = 0;
            }
            else
            {
                _axis = axis.Value;
            }


            // Reshape and get axis
            x = NpyCoreApi.CheckAxis(a, ref _axis, 0);
            // Compute the mean
            mean = np.FromAny(np.mean(x,_axis, dtype), dtype);
            // Add an axis back to the mean so it will broadcast correctly
            newshape = x.dims.Select(y => (npy_intp)y).ToArray();
            newshape[_axis] = (npy_intp)1;
            tmp = NpyCoreApi.Newshape(mean, newshape, NPY_ORDER.NPY_CORDER);
            // Compute x - mean
            tmp = np.FromAny(x - tmp);
            // Square the difference
            if (tmp.IsComplex)
            {
                tmp = np.FromAny(tmp * tmp.Conjugate());
            }
            else
            {
                tmp = np.FromAny(tmp * tmp);
            }

            // Sum the square
            tmp = tmp.Sum(_axis, dtype);

            // Divide by n (or n-ddof) and maybe take the sqrt
            n = x.dims[_axis] - ddof;
            if (n == 0) n = 1;
            if (true)
            {
                result = np.sqrt(np.FromAny(tmp * (1.0 / n)));
            }
            else
            {
                result = tmp * (1.0 / n);
            }

            // Deal with subclasses
            if (result is ndarray && result.GetType() != a.GetType())
            {
                ndarray aresult;
                if (result.GetType() != typeof(ndarray))
                {
                    aresult = np.FromAny(result, flags: NPYARRAYFLAGS.NPY_ENSUREARRAY);
                }
                else
                {
                    aresult = (ndarray)result;
                }
                if (a.GetType() != typeof(ndarray))
                {
                    throw new Exception("not an ndarray");
                }
                result = aresult;
            }

            // Copy into ret, if necessary
            if (ret != null)
            {
                NpyCoreApi.CopyAnyInto(ret, np.FromAny(result));
                return ret;
            }
            return result as ndarray;
#endif
        }


        #endregion

        #region var

        /// <summary>
        /// Compute the variance along the specified axis.
        /// </summary>
        /// <param name="a">Array containing numbers whose variance is desired.</param>
        /// <param name="axis">Axis or axes along which the variance is computed.</param>
        /// <param name="dtype">Type to use in computing the variance.</param>
        /// <param name="ddof">"Delta Degrees of Freedom": the divisor used in the calculation</param>
        /// <param name="keep_dims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns></returns>
        public static ndarray var(object a, int? axis = null, dtype dtype = null, int ddof = 0, bool keep_dims = false)
        {
            /*
            Compute the variance along the specified axis.

            Returns the variance of the array elements, a measure of the spread of a
            distribution.  The variance is computed for the flattened array by
            default, otherwise over the specified axis.

            Parameters
            ----------
            a : array_like
                Array containing numbers whose variance is desired.  If `a` is not an
                array, a conversion is attempted.
            axis : None or int or tuple of ints, optional
                Axis or axes along which the variance is computed.  The default is to
                compute the variance of the flattened array.

                .. versionadded:: 1.7.0

                If this is a tuple of ints, a variance is performed over multiple axes,
                instead of a single axis or all the axes as before.
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
                ``N - ddof``, where ``N`` represents the number of elements. By
                default `ddof` is zero.
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the input array.

                If the default value is passed, then `keepdims` will not be
                passed through to the `var` method of sub-classes of
                `ndarray`, however any non-default value will be.  If the
                sub-class' method does not implement `keepdims` any
                exceptions will be raised.

            Returns
            -------
            variance : ndarray, see dtype parameter above
                If ``out=None``, returns a new array containing the variance;
                otherwise, a reference to the output array is returned.

            See Also
            --------
            std , mean, nanmean, nanstd, nanvar
            numpy.doc.ufuncs : Section "Output arguments"

            Notes
            -----
            The variance is the average of the squared deviations from the mean,
            i.e.,  ``var = mean(abs(x - x.mean())**2)``.

            The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
            If, however, `ddof` is specified, the divisor ``N - ddof`` is used
            instead.  In standard statistical practice, ``ddof=1`` provides an
            unbiased estimator of the variance of a hypothetical infinite population.
            ``ddof=0`` provides a maximum likelihood estimate of the variance for
            normally distributed variables.

            Note that for complex numbers, the absolute value is taken before
            squaring, so that the result is always real and nonnegative.

            For floating-point input, the variance is computed using the same
            precision the input has.  Depending on the input data, this can cause
            the results to be inaccurate, especially for `float32` (see example
            below).  Specifying a higher-accuracy accumulator using the ``dtype``
            keyword can alleviate this issue.

            Examples
            --------
            >>> a = np.array([[1, 2], [3, 4]])
            >>> np.var(a)
            1.25
            >>> np.var(a, axis=0)
            array([ 1.,  1.])
            >>> np.var(a, axis=1)
            array([ 0.25,  0.25])

            In single precision, var() can be inaccurate:

            >>> a = np.zeros((2, 512*512), dtype=np.float32)
            >>> a[0, :] = 1.0
            >>> a[1, :] = 0.1
            >>> np.var(a)
            0.20250003

            Computing the variance in float64 is more accurate:

            >>> np.var(a, dtype=np.float64)
            0.20249999932944759
            >>> ((1-0.55)**2 + (0.1-0.55)**2)/2
            0.2025             
            */

            var arr = asanyarray(a);

            long rcount = _count_reduce_items(arr, axis);
            if (ddof >= rcount)
            {
                Console.WriteLine("Degrees of freedom <= 0 for slice");
            }

            if (dtype == null)
            {
                if (NpyDefs.IsBool(arr.TypeNum))
                {
                    dtype = np.Float32;
                }
                else
                {
                    dtype = result_type(arr.TypeNum);
                }
            }

            // Compute the mean.
            var arrmean = np.sum(arr, axis, dtype, keepdims: true);
            arrmean = np.true_divide(arrmean, rcount);

            // Compute sum of squared deviations from mean

            var x = asanyarray(arr.astype(dtype) - arrmean);
            x = np.multiply(x, x);

            var ret = np.sum(x, axis, dtype, keepdims: keep_dims);

            // Compute degrees of freedom and make sure it is not negative.
            rcount = Math.Max(rcount - ddof, 0);

            //divide by degrees of freedom
            ret = np.true_divide(ret, rcount);

            return ret;
        }

        #endregion

        private static long _count_reduce_items(ndarray arr, int? axis)
        {
            int[] dimArray = null;
            if (axis == null)
            {
                dimArray = new int[arr.ndim];
                for (int i = 0; i < arr.ndim; i++)
                    dimArray[i] = i;
            }
            else
            {
                dimArray = new int[1] { axis.Value };
            }

            long items = 1;
            foreach (int index in dimArray)
            {
                items *= arr.shape.iDims[index];
            }

            return items;
        }
    }
}
