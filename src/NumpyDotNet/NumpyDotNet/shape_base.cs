using System;
using System.Collections.Generic;
using System.Text;

namespace NumpyDotNet
{
    public static partial class np
    {
        public static ICollection<ndarray> atleast_1d(ICollection<object> arys)
        {
            //
            //Convert inputs to arrays with at least one dimension.

            //Scalar inputs are converted to 1 - dimensional arrays, whilst
            //higher - dimensional inputs are preserved.

            //  Parameters
            //  ----------
            //arys1, arys2, ... : array_like
            //    One or more input arrays.

            //Returns
            //------ -
            //ret : ndarray
            //    An array, or list of arrays, each with ``a.ndim >= 1``.
            //    Copies are made only if necessary.

            //See Also
            //--------
            //atleast_2d, atleast_3d

            //Examples
            //--------
            //>>> np.atleast_1d(1.0)
            //array([1.])

            //>>> x = np.arange(9.0).reshape(3, 3)
            //>>> np.atleast_1d(x)
            //array([[0., 1., 2.],
            //       [ 3.,  4.,  5.],
            //       [ 6.,  7.,  8.]])
            //>>> np.atleast_1d(x) is x
            //True

            //>>> np.atleast_1d(1, [3, 4])
            //[array([1]), array([3, 4])]

            //

            throw new NotImplementedException();
        }

        public static ICollection<ndarray> atleast_2d(ICollection<object> arys)
        {
            //
            //View inputs as arrays with at least two dimensions.

            //Parameters
            //----------
            //arys1, arys2, ... : array_like
            //    One or more array - like sequences.Non - array inputs are converted
            //      to arrays.Arrays that already have two or more dimensions are
            //    preserved.

            //Returns
            //------ -
            //res, res2, ... : ndarray
            //    An array, or list of arrays, each with ``a.ndim >= 2``.
            //    Copies are avoided where possible, and views with two or more
            //    dimensions are returned.

            //See Also
            //--------
            //atleast_1d, atleast_3d

            //Examples
            //--------
            //>>> np.atleast_2d(3.0)
            //array([[3.]])

            //>>> x = np.arange(3.0)
            //>>> np.atleast_2d(x)
            //array([[0., 1., 2.]])
            //>>> np.atleast_2d(x).base is x
            //True

            //>>> np.atleast_2d(1, [1, 2], [[1, 2]])
            //[array([[1]]), array([[1, 2]]), array([[1, 2]])]

            //

            throw new NotImplementedException();
        }

        public static ICollection<ndarray> atleast_3d(ICollection<object> arys)
        {
            //
            //View inputs as arrays with at least three dimensions.

            //Parameters
            //----------
            //arys1, arys2, ... : array_like
            //    One or more array - like sequences.Non - array inputs are converted to
            //    arrays.Arrays that already have three or more dimensions are
            //    preserved.

            //Returns
            //------ -
            //res1, res2, ... : ndarray
            //    An array, or list of arrays, each with ``a.ndim >= 3``.  Copies are
            //    avoided where possible, and views with three or more dimensions are
            //    returned.For example, a 1 - D array of shape ``(N,)`` becomes a view
            //    of shape ``(1, N, 1)``, and a 2 - D array of shape ``(M, N)`` becomes a
            //    view of shape ``(M, N, 1)``.

            //See Also
            //--------
            //atleast_1d, atleast_2d

            //Examples
            //--------
            //>>> np.atleast_3d(3.0)
            //array([[[3.]]])

            //>>> x = np.arange(3.0)
            //>>> np.atleast_3d(x).shape
            //(1, 3, 1)

            //>>> x = np.arange(12.0).reshape(4, 3)
            //>>> np.atleast_3d(x).shape
            //(4, 3, 1)
            //>>> np.atleast_3d(x).base is x.base  # x is a reshape, so not base itself
            //True

            //>>> for arr in np.atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):
            //...     print(arr, arr.shape)
            //...
            //[[[1]
            //  [2]]] (1, 2, 1)
            //[[[1]
            //  [2]]] (1, 2, 1)
            //[[[1 2]]] (1, 1, 2)

            //

            throw new NotImplementedException();
        }

        public static ndarray vstack(ICollection<object> tup)
        {
            //
            //Stack arrays in sequence vertically(row wise).

            //This is equivalent to concatenation along the first axis after 1 - D arrays
            //  of shape `(N,)` have been reshaped to `(1, N)`. Rebuilds arrays divided by
            //`vsplit`.

            //This function makes most sense for arrays with up to 3 dimensions.For
            //instance, for pixel - data with a height(first axis), width(second axis),
            //and r / g / b channels(third axis).The functions `concatenate`, `stack` and
            //`block` provide more general stacking and concatenation operations.

            //Parameters
            //----------
            //tup : sequence of ndarrays
            //    The arrays must have the same shape along all but the first axis.
            //    1 - D arrays must have the same length.

            //Returns
            //------ -
            //stacked : ndarray
            //    The array formed by stacking the given arrays, will be at least 2 - D.

            //See Also
            //--------
            //stack : Join a sequence of arrays along a new axis.
            //hstack : Stack arrays in sequence horizontally(column wise).
            //dstack : Stack arrays in sequence depth wise(along third dimension).
            //concatenate : Join a sequence of arrays along an existing axis.
            //vsplit : Split array into a list of multiple sub - arrays vertically.
            //block : Assemble arrays from blocks.

            //Examples
            //--------
            //>>> a = np.array([1, 2, 3])
            //>>> b = np.array([2, 3, 4])
            //>>> np.vstack((a, b))
            //array([[1, 2, 3],
            //       [2, 3, 4]])

            //>>> a = np.array([[1], [2], [3]])
            //>>> b = np.array([[2], [3], [4]])
            //>>> np.vstack((a, b))
            //array([[1],
            //       [2],
            //       [3],
            //       [2],
            //       [3],
            //       [4]])

            //

            throw new NotImplementedException();
        }

        public static ndarray hstack(ICollection<object> tup)
        {
            //
            //Stack arrays in sequence horizontally(column wise).

            //This is equivalent to concatenation along the second axis, except for 1 - D
            //arrays where it concatenates along the first axis.Rebuilds arrays divided
            //by `hsplit`.

            //This function makes most sense for arrays with up to 3 dimensions.For
            //instance, for pixel - data with a height(first axis), width(second axis),
            //and r / g / b channels(third axis).The functions `concatenate`, `stack` and
            //`block` provide more general stacking and concatenation operations.

            //Parameters
            //----------
            //tup : sequence of ndarrays
            //    The arrays must have the same shape along all but the second axis,
            //    except 1 - D arrays which can be any length.

            //Returns
            //------ -
            //stacked : ndarray
            //    The array formed by stacking the given arrays.

            //See Also
            //--------
            //stack : Join a sequence of arrays along a new axis.
            //vstack : Stack arrays in sequence vertically(row wise).
            //dstack : Stack arrays in sequence depth wise(along third axis).
            //concatenate : Join a sequence of arrays along an existing axis.
            //hsplit : Split array along second axis.
            //block : Assemble arrays from blocks.

            //Examples
            //--------
            //>>> a = np.array((1, 2, 3))
            //>>> b = np.array((2, 3, 4))
            //>>> np.hstack((a, b))
            //array([1, 2, 3, 2, 3, 4])
            //>>> a = np.array([[1],[2],[3]])
            //>>> b = np.array([[2],[3],[4]])
            //>>> np.hstack((a, b))
            //array([[1, 2],
            //       [2, 3],
            //       [3, 4]])

            //

            throw new NotImplementedException();
        }

        public static ndarray stack(ICollection<object> tup, int? axis = null, ndarray _out = null)
        {
            //
            //Join a sequence of arrays along a new axis.

            //The `axis` parameter specifies the index of the new axis in the dimensions
            //of the result.For example, if ``axis = 0`` it will be the first dimension
            //and if ``axis = -1`` it will be the last dimension.

            //.. versionadded:: 1.10.0

            //Parameters
            //----------
            //arrays: sequence of array_like
            //   Each array must have the same shape.
            //axis: int, optional
            //    The axis in the result array along which the input arrays are stacked.
            //out : ndarray, optional
            //    If provided, the destination to place the result. The shape must be
            //    correct, matching that of what stack would have returned if no
            //    out argument were specified.

            //Returns
            //------ -
            //stacked : ndarray
            //    The stacked array has one more dimension than the input arrays.

            //See Also
            //--------
            //concatenate : Join a sequence of arrays along an existing axis.
            //split : Split array into a list of multiple sub-arrays of equal size.
            //block : Assemble arrays from blocks.

            //Examples
            //--------
            //>>> arrays = [np.random.randn(3, 4) for _ in range(10)]
            //>>> np.stack(arrays, axis = 0).shape
            //(10, 3, 4)

            //>>> np.stack(arrays, axis = 1).shape
            //(3, 10, 4)

            //>>> np.stack(arrays, axis = 2).shape
            //(3, 4, 10)

            //>>> a = np.array([1, 2, 3])
            //>>> b = np.array([2, 3, 4])
            //>>> np.stack((a, b))
            //array([[1, 2, 3],
            //       [2, 3, 4]])

            //>>> np.stack((a, b), axis=-1)
            //array([[1, 2],
            //       [2, 3],
            //       [3, 4]])

            //

            throw new NotImplementedException();
        }

        public static ndarray column_stack(ICollection<object> tup)
        {
           //
           // Stack 1 - D arrays as columns into a 2 - D array.

           // Take a sequence of 1 - D arrays and stack them as columns
           // to make a single 2 - D array. 2 - D arrays are stacked as-is,
           // just like with `hstack`.  1 - D arrays are turned into 2 - D columns
           // first.

           // Parameters
           // ----------
           // tup: sequence of 1 - D or 2 - D arrays.
           //     Arrays to stack.All of them must have the same first dimension.

           //Returns
           //------ -
           //stacked : 2 - D array
           //      The array formed by stacking the given arrays.

           // See Also
           // --------
           // stack, hstack, vstack, concatenate

           // Examples
           // --------
           // >>> a = np.array((1, 2, 3))
           // >>> b = np.array((2, 3, 4))
           // >>> np.column_stack((a, b))
           // array([[1, 2],
           //        [2, 3],
           //        [3, 4]])
           //

            throw new NotImplementedException();
        }

        public static ndarray dstack(ICollection<object> tup)
        {
            //
            //Stack arrays in sequence depth wise(along third axis).

            //This is equivalent to concatenation along the third axis after 2 - D arrays
            //  of shape `(M, N)` have been reshaped to `(M, N, 1)` and 1 - D arrays of shape
            //`(N,)` have been reshaped to `(1, N, 1)`. Rebuilds arrays divided by
            //`dsplit`.

            //This function makes most sense for arrays with up to 3 dimensions.For
            //instance, for pixel - data with a height(first axis), width(second axis),
            //and r / g / b channels(third axis).The functions `concatenate`, `stack` and
            //`block` provide more general stacking and concatenation operations.

            //Parameters
            //----------
            //tup : sequence of arrays
            //    The arrays must have the same shape along all but the third axis.
            //    1 - D or 2 - D arrays must have the same shape.

            //Returns
            //------ -
            //stacked : ndarray
            //    The array formed by stacking the given arrays, will be at least 3 - D.

            //See Also
            //--------
            //stack : Join a sequence of arrays along a new axis.
            //vstack : Stack along first axis.
            //hstack : Stack along second axis.
            //concatenate : Join a sequence of arrays along an existing axis.
            //dsplit : Split array along third axis.

            //Examples
            //--------
            //>>> a = np.array((1, 2, 3))
            //>>> b = np.array((2, 3, 4))
            //>>> np.dstack((a, b))
            //array([[[1, 2],
            //        [2, 3],
            //        [3, 4]]])

            //>>> a = np.array([[1],[2],[3]])
            //>>> b = np.array([[2],[3],[4]])
            //>>> np.dstack((a, b))
            //array([[[1, 2]],
            //       [[2, 3]],
            //       [[3, 4]]])

            //   

            throw new NotImplementedException();
        }

        public static ICollection<ndarray> array_split(ndarray ary, int indices_or_sections, int? axis= null)
        {
          //
          //  Split an array into multiple sub-arrays.

          //  Please refer to the ``split`` documentation.The only difference
          //between these functions is that ``array_split`` allows
          //  `indices_or_sections` to be an integer that does *not * equally
          //  divide the axis.For an array of length l that should be split
          //  into n sections, it returns l % n sub - arrays of size l//n + 1
          //    and the rest of size l//n.

          //    See Also
          //    --------
          //  split: Split array into multiple sub - arrays of equal size.

          //   Examples
          //   --------
          //   >>> x = np.arange(8.0)
          //   >>> np.array_split(x, 3)
          //       [array([0., 1., 2.]), array([3., 4., 5.]), array([6., 7.])]

          //  >>> x = np.arange(7.0)
          //  >>> np.array_split(x, 3)
          //      [array([0., 1., 2.]), array([3., 4.]), array([5., 6.])]

          //

            throw new NotImplementedException();
        }

        public static ICollection<ndarray> split(ndarray ary, int indices_or_sections, int? axis = null)
        {
           //
           // Split an array into multiple sub-arrays.

           // Parameters
           // ----------
           // ary: ndarray
           //    Array to be divided into sub - arrays.
           //indices_or_sections : int or 1 - D array
           //      If `indices_or_sections` is an integer, N, the array will be divided
           //      into N equal arrays along `axis`.  If such a split is not possible,
           //     an error is raised.

           //     If `indices_or_sections` is a 1 - D array of sorted integers, the entries
           //       indicate where along `axis` the array is split.For example,
           //     ``[2, 3]`` would, for ``axis=0``, result in

           //       - ary[:2]
           //       - ary[2:3]
           //       - ary[3:]

           //     If an index exceeds the dimension of the array along `axis`,
           //     an empty sub-array is returned correspondingly.
           // axis : int, optional
           //     The axis along which to split, default is 0.

           // Returns
           // -------
           // sub-arrays : list of ndarrays
           //     A list of sub-arrays.

           // Raises
           // ------
           // ValueError
           //     If `indices_or_sections` is given as an integer, but
           //     a split does not result in equal division.

           // See Also
           // --------
           // array_split : Split an array into multiple sub-arrays of equal or
           //               near-equal size.  Does not raise an exception if
           //               an equal division cannot be made.
           // hsplit : Split array into multiple sub-arrays horizontally (column-wise).
           // vsplit : Split array into multiple sub-arrays vertically(row wise).
           // dsplit : Split array into multiple sub-arrays along the 3rd axis(depth).
           // concatenate : Join a sequence of arrays along an existing axis.
           // stack : Join a sequence of arrays along a new axis.
           // hstack : Stack arrays in sequence horizontally(column wise).
           // vstack : Stack arrays in sequence vertically(row wise).
           // dstack : Stack arrays in sequence depth wise(along third dimension).

           // Examples
           // --------
           // >>> x = np.arange(9.0)
           // >>> np.split(x, 3)
           // [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.,  8.])]

           // >>> x = np.arange(8.0)
           // >>> np.split(x, [3, 5, 6, 10])
           // [array([0., 1., 2.]),
           //  array([3., 4.]),
           //  array([5.]),
           //  array([6., 7.]),
           //  array([], dtype = float64)]
           //

            throw new NotImplementedException();
        }

        public static ICollection<ndarray> hsplit(ndarray ary, int indices_or_sections)
        {
            //
            //Split an array into multiple sub-arrays horizontally(column - wise).

            //Please refer to the `split` documentation.  `hsplit` is equivalent
            //to `split` with ``axis = 1``, the array is always split along the second
            //  axis regardless of the array dimension.

            //See Also
            //--------
            //split : Split an array into multiple sub-arrays of equal size.

            //Examples
            //--------
            //>>> x = np.arange(16.0).reshape(4, 4)
            //>>> x
            //array([[0., 1., 2., 3.],
            //       [  4.,   5.,   6.,   7.],
            //       [  8.,   9.,  10.,  11.],
            //       [ 12.,  13.,  14.,  15.]])
            //>>> np.hsplit(x, 2)
            //[array([[  0.,   1.],
            //       [  4.,   5.],
            //       [  8.,   9.],
            //       [ 12.,  13.]]),
            // array([[  2.,   3.],
            //       [  6.,   7.],
            //       [ 10.,  11.],
            //       [ 14.,  15.]])]
            //>>> np.hsplit(x, np.array([3, 6]))
            //[array([[  0.,   1.,   2.],
            //       [  4.,   5.,   6.],
            //       [  8.,   9.,  10.],
            //       [ 12.,  13.,  14.]]),
            // array([[  3.],
            //       [  7.],
            //       [ 11.],
            //       [ 15.]]),
            // array([], dtype= float64)]

            //With a higher dimensional array the split is still along the second axis.

            //>>> x = np.arange(8.0).reshape(2, 2, 2)
            //>>> x
            //array([[[ 0.,  1.],
            //        [ 2.,  3.]],
            //       [[ 4.,  5.],
            //        [ 6.,  7.]]])
            //>>> np.hsplit(x, 2)
            //[array([[[ 0.,  1.]],
            //       [[ 4.,  5.]]]),
            // array([[[ 2.,  3.]],
            //       [[ 6.,  7.]]])]

            //

 
            throw new NotImplementedException();
        }

        public static ICollection<ndarray> vsplit(ndarray ary, int indices_or_sections)
        {
            //
            //Split an array into multiple sub-arrays vertically(row - wise).

            //Please refer to the ``split`` documentation.  ``vsplit`` is equivalent
            //to ``split`` with `axis = 0` (default), the array is always split along the
            //first axis regardless of the array dimension.

            //See Also
            //--------
            //split : Split an array into multiple sub-arrays of equal size.

            //Examples
            //--------
            //>>> x = np.arange(16.0).reshape(4, 4)
            //>>> x
            //array([[0., 1., 2., 3.],
            //       [  4.,   5.,   6.,   7.],
            //       [  8.,   9.,  10.,  11.],
            //       [ 12.,  13.,  14.,  15.]])
            //>>> np.vsplit(x, 2)
            //[array([[ 0.,  1.,  2.,  3.],
            //       [ 4.,  5.,  6.,  7.]]),
            // array([[  8.,   9.,  10.,  11.],
            //       [ 12.,  13.,  14.,  15.]])]
            //>>> np.vsplit(x, np.array([3, 6]))
            //[array([[  0.,   1.,   2.,   3.],
            //       [  4.,   5.,   6.,   7.],
            //       [  8.,   9.,  10.,  11.]]),
            // array([[ 12.,  13.,  14.,  15.]]),
            // array([], dtype= float64)]

            //With a higher dimensional array the split is still along the first axis.

            //>>> x = np.arange(8.0).reshape(2, 2, 2)
            //>>> x
            //array([[[ 0.,  1.],
            //        [ 2.,  3.]],
            //       [[ 4.,  5.],
            //        [ 6.,  7.]]])
            //>>> np.vsplit(x, 2)
            //[array([[[ 0.,  1.],
            //        [ 2.,  3.]]]),
            // array([[[ 4.,  5.],
            //        [ 6.,  7.]]])]

            //

            throw new NotImplementedException();
        }

        public static ICollection<ndarray> dsplit(ndarray ary, int indices_or_sections)
        {
            //
            //Split array into multiple sub - arrays along the 3rd axis(depth).

            //Please refer to the `split` documentation.  `dsplit` is equivalent
            //to `split` with ``axis = 2``, the array is always split along the third
            //  axis provided the array dimension is greater than or equal to 3.

            //See Also
            //--------
            //split : Split an array into multiple sub-arrays of equal size.

            //Examples
            //--------
            //>>> x = np.arange(16.0).reshape(2, 2, 4)
            //>>> x
            //array([[[0., 1., 2., 3.],
            //        [  4.,   5.,   6.,   7.]],
            //       [[  8.,   9.,  10.,  11.],
            //        [ 12.,  13.,  14.,  15.]]])
            //>>> np.dsplit(x, 2)
            //[array([[[  0.,   1.],
            //        [  4.,   5.]],
            //       [[  8.,   9.],
            //        [ 12.,  13.]]]),
            // array([[[  2.,   3.],
            //        [  6.,   7.]],
            //       [[ 10.,  11.],
            //        [ 14.,  15.]]])]
            //>>> np.dsplit(x, np.array([3, 6]))
            //[array([[[  0.,   1.,   2.],
            //        [  4.,   5.,   6.]],
            //       [[  8.,   9.,  10.],
            //        [ 12.,  13.,  14.]]]),
            // array([[[  3.],
            //        [  7.]],
            //       [[ 11.],
            //        [ 15.]]]),
            // array([], dtype= float64)]

            //

            throw new NotImplementedException();
        }

        public static ndarray kron(ndarray a, ndarray b)
        {
           //
           // Kronecker product of two arrays.

           // Computes the Kronecker product, a composite array made of blocks of the
           // second array scaled by the first.

           // Parameters
           // ----------
           // a, b: array_like

           //Returns
           // -------
           // out : ndarray

           // See Also
           // --------
           // outer: The outer product

           //Notes
           // -----
           // The function assumes that the number of dimensions of `a` and `b`
           // are the same, if necessary prepending the smallest with ones.
           // If `a.shape = (r0, r1,.., rN)` and `b.shape = (s0, s1,..., sN)`,
           // the Kronecker product has shape `(r0 * s0, r1 * s1, ..., rN * SN)`.
           // The elements are products of elements from `a` and `b`, organized
           // explicitly by::

           //     kron(a, b)[k0, k1,..., kN] = a[i0, i1,..., iN] * b[j0, j1,..., jN]

           // where::

           //     kt = it * st + jt,  t = 0,...,N

           // In the common 2 - D case (N = 1), the block structure can be visualized::
    

           //         [[a[0, 0] * b, a[0, 1] * b,  ... , a[0, -1] * b],
           //      [  ...                              ...   ],
           //      [a[-1,0]* b, a[-1, 1]*b, ... , a[-1, -1]*b ]]


           // Examples
           // --------
           // >>> np.kron([1, 10, 100], [5, 6, 7])
           // array([  5,   6,   7,  50,  60,  70, 500, 600, 700])
           // >>> np.kron([5, 6, 7], [1, 10, 100])
           // array([  5,  50, 500,   6,  60, 600,   7,  70, 700])

           // >>> np.kron(np.eye(2), np.ones((2,2)))
           // array([[ 1.,  1.,  0.,  0.],
           //        [ 1.,  1.,  0.,  0.],
           //        [ 0.,  0.,  1.,  1.],
           //        [ 0.,  0.,  1.,  1.]])

           // >>> a = np.arange(100).reshape((2,5,2,5))
           // >>> b = np.arange(24).reshape((2,3,4))
           // >>> c = np.kron(a, b)
           // >>> c.shape
           // (2, 10, 6, 20)
           // >>> I = (1,3,0,2)
           // >>> J = (0,2,1)
           // >>> J1 = (0,) + J             # extend to ndim=4
           // >>> S1 = (1,) + b.shape
           // >>> K = tuple(np.array(I) * np.array(S1) + np.array(J1))
           // >>> c[K] == a[I]*b[J]
           // True

           //
            throw new NotImplementedException();
        }

        public static ndarray tile(ndarray a, Int32[] reps)
        {
         //
         //   Construct an array by repeating A the number of times given by reps.

         //   If `reps` has length ``d``, the result will have dimension of
         //   ``max(d, A.ndim)``.

         //   If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
         //   axes.So a shape(3,) array is promoted to (1, 3) for 2 - D replication,
         // or shape(1, 1, 3) for 3 - D replication.If this is not the desired

         //behavior, promote `A` to d - dimensions manually before calling this

         //function.

         //If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre - pending 1's to it.

         //Thus for an `A` of shape(2, 3, 4, 5), a `reps` of(2, 2) is treated as
         //(1, 1, 2, 2).

         //Note : Although tile may be used for broadcasting, it is strongly
         //recommended to use numpy's broadcasting operations and functions.


         //Parameters
         //----------

         //A : array_like

         //    The input array.
         //reps : array_like

         //    The number of repetitions of `A` along each axis.

         //Returns
         //------ -
         //c : ndarray

         //    The tiled output array.


         //See Also
         //--------

         //repeat : Repeat elements of an array.
         //broadcast_to : Broadcast an array to a new shape


         //Examples
         //--------
         //>>> a = np.array([0, 1, 2])
         //   >>> np.tile(a, 2)
         //   array([0, 1, 2, 0, 1, 2])
         //   >>> np.tile(a, (2, 2))
         //   array([[0, 1, 2, 0, 1, 2],
         //          [0, 1, 2, 0, 1, 2]])
         //   >>> np.tile(a, (2, 1, 2))
         //   array([[[0, 1, 2, 0, 1, 2]],
         //          [[0, 1, 2, 0, 1, 2]]])

         //   >>> b = np.array([[1, 2], [3, 4]])
         //   >>> np.tile(b, 2)
         //   array([[1, 2, 1, 2],
         //          [3, 4, 3, 4]])
         //   >>> np.tile(b, (2, 1))
         //   array([[1, 2],
         //          [3, 4],
         //          [1, 2],
         //          [3, 4]])

         //   >>> c = np.array([1, 2, 3, 4])
         //   >>> np.tile(c, (4,1))
         //   array([[1, 2, 3, 4],
         //          [1, 2, 3, 4],
         //          [1, 2, 3, 4],
         //          [1, 2, 3, 4]])
         // 

            throw new NotImplementedException();
        }

    }
}
