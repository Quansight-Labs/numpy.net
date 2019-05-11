using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
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

            List<ndarray> res = new List<ndarray>();

            foreach (var ary in arys)
            {
                ndarray result = null;
                ndarray array = asanyarray(ary);
                if (array.ndim == 0)
                {
                    result = array.reshape(new shape(1));
                }
                else
                {
                    result = array;
                }
                res.Add(result);
            }

            return res;
        }

        public static ICollection<ndarray> atleast_1d(object array)
        {
            return atleast_1d(new object[] { array });
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

            List<ndarray> res = new List<ndarray>();

            foreach (var ary in arys)
            {
                ndarray result = null;
                ndarray array = asanyarray(ary);
                if (array.ndim == 0)
                {
                    result = array.reshape(new shape(1, 1));
                }
                else if (array.ndim == 1)
                {
                    result = array.A(newaxis, ":");
                }
                else
                {
                    result = array;
                }
                res.Add(result);
            }

            return res;
        }

        public static ICollection<ndarray> atleast_2d(object array)
        {
            return atleast_2d(new object[] { array });
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

            List<ndarray> res = new List<ndarray>();

            foreach (var ary in arys)
            {
                ndarray result = null;
                ndarray array = asanyarray(ary);
                if (array.ndim == 0)
                {
                    result = array.reshape(new shape(1, 1, 1));
                }
                else if (array.ndim == 1)
                {
                    result = array.A(newaxis, ":", newaxis);
                }
                else if (array.ndim == 2)
                {
                    result = array.A(":", ":", newaxis);
                }
                else
                {
                    result = array;
                }
                res.Add(result);
            }

            return res;
        }

        public static ICollection<ndarray> atleast_3d(object array)
        {
            return atleast_3d(new object[] { array });
        }

        #region apply_along_axis

        public delegate ndarray apply_along_axis_view(ndarray a, ndarray view);
        public delegate ndarray apply_along_axis_indices(ndarray a, IList<npy_intp> indices);
        public delegate object apply_along_axis_fn(ndarray a, params object[] args);

        public static ndarray apply_along_axis(apply_along_axis_indices fn, int axis, ndarray arr)
        {
            if (fn == null)
            {
                throw new Exception("function can't be null");
            }
            var indices = IndicesFromAxis(arr, axis);

            try
            {
                var ret = fn(arr, indices);
                return ret;
            }
            catch (Exception ex)
            {
                throw;
            }

        }

        public static ndarray apply_along_axis(apply_along_axis_view fn, int axis, ndarray arr)
        {
            if (fn == null)
            {
                throw new Exception("function can't be null");
            }
            var view = ViewFromAxis(arr, axis);

            try
            {
                var ret = fn(arr, view);
                return ret;
            }
            catch (Exception ex)
            {
                throw;
            }
        }

        public static ndarray apply_along_axis(apply_along_axis_fn func1d, int axis, ndarray arr, params object[] args)
        {
            if (func1d == null)
            {
                throw new Exception("function can't be null");
            }

            // handle negative axes
            arr = asanyarray(arr);
            var nd = arr.ndim;
            axis = normalize_axis_index(axis, nd);

            // arr, with the iteration axis at the end
            var in_dims = PythonFunction.range(0, nd);
            List<npy_intp> in_dims2 = new List<npy_intp>();
            for (int i = 0; i < axis; i++)
                in_dims2.Add(in_dims[i]);
            for (int i = axis+1; i < in_dims.Length; i++)
                in_dims2.Add(in_dims[i]);
            in_dims2.Add(axis);
            in_dims = in_dims2.ToArray();

            var inarr_view = transpose(arr, in_dims);

            // compute indices for the iteration axes, and append a trailing ellipsis to
            // prevent 0d arrays decaying to scalars, which fixes gh-8642

            npy_intp[] newshape = new npy_intp[inarr_view.shape.iDims.Length - 1];
            Array.Copy(inarr_view.shape.iDims, 0, newshape, 0, inarr_view.shape.iDims.Length - 1);
            var indices = new ndindex(new shape(newshape));

            List<object> inds = new List<object>();
            foreach (var _ind in indices)
            {
                npy_intp[] ind = _ind as npy_intp[];
                inds.Add(new object[] { ind[0], new Ellipsis() });
            }

            object ind0 = null;
            ndarray res = null;
            int res_dimadjust = 0;
            foreach (var _ind0 in inds)
            {
                ind0 = _ind0;
                var f1 = func1d(inarr_view.A(ind0), args);
                if (np.IsNumericType(f1))
                {
                    res_dimadjust = 1;
                }
                res = asanyarray(f1);
                break;
            }


            // build a buffer for storing evaluations of func1d.
            // remove the requested axis, and add the new ones on the end.
            // laid out so that each write is contiguous.
            // for a tuple index inds, buff[inds] = func1d(inarr_view[inds])

            List<npy_intp> buffShape = new List<npy_intp>();
            for (int i = 0; i < inarr_view.shape.iDims.Length - 1; i++)
                buffShape.Add(inarr_view.shape.iDims[i]);
            for (int i = 0; i < (res.shape.iDims.Length-res_dimadjust); i++)
                buffShape.Add(res.shape.iDims[i]);
            var buff = zeros(new shape(buffShape), dtype: res.Dtype);


            // permutation of axes such that out = buff.transpose(buff_permute)
            var buff_dims = PythonFunction.range(0, buff.ndim);
            List<npy_intp> buff_permute = new List<npy_intp>();
            for (int i = 0; i < axis; i++)
                buff_permute.Add(buff_dims[i]);
            for (int i = buff.ndim - (res.ndim-res_dimadjust); i < buff.ndim; i++)
                buff_permute.Add(buff_dims[i]);
            for (int i = axis; i < buff.ndim - (res.ndim-res_dimadjust); i++)
                buff_permute.Add(buff_dims[i]);

            // matrices have a nasty __array_prepare__ and __array_wrap__
            //if not isinstance(res, matrix):
            //    buff = res.__array_prepare__(buff)

            buff[ind0] =  res;
            foreach (var ind in inds)
            {
                buff[ind] = asanyarray(func1d(inarr_view.A(ind), args));
            }

            if (!res.IsMatrix)
            {
                // wrap the array, to preserve subclasses
                buff = res.__array_wrap__(buff);

                // finally, rotate the inserted axes back to where they belong
                return transpose(buff, buff_permute.ToArray());
            }
            else
            {
                // matrices have to be transposed first, because they collapse dimensions!
                var out_arr = transpose(buff, buff_permute.ToArray());
                return res.__array_wrap__(out_arr);
            }

        }

        #endregion

        #region apply_over_axis

        public delegate ndarray apply_over_axes_fn(ndarray a, params object[] args);

        public static ndarray apply_over_axes(apply_over_axes_fn func, object a, object axes)
        {
            var val = asarray(a);
            var N = val.ndim;

            int[] _axes = null;

            var axesarr = asanyarray(axes);
            if (axesarr.ndim == 0)
            {
                _axes = axesarr.AsInt32Array();
                if (_axes == null)
                {
                    throw new Exception("axes must be in int[]");
                }
            }
            else
            {
                _axes = axesarr.AsInt32Array();
            }

            for (int i= 0; i < _axes.Length; i++)
            {
                int axis = _axes[i];

                if (axis < 0)
                {
                    axis = N + axis;
                }

                var res = func(val, axis);
                if (res.ndim == val.ndim)
                {
                    val = res;
                }
                else
                {
                    res = expand_dims(res, axis);
                    if (res.ndim == val.ndim)
                    {
                        val = res;
                    }
                    else
                    {
                        throw new ValueError("function is not returning an array of the correct shape");
                    }
 
                }
  
            }

            return val;
        }

        #endregion

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

            return np.concatenate(atleast_2d(tup));
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

            var arrs = atleast_1d(tup);
            if (arrs != null && arrs.ElementAt(0).ndim == 1)
            {
                return np.concatenate(arrs, 0);
            }
            else
            {
                return np.concatenate(arrs, 1);
            }

        }

        public static ndarray stack(ICollection<object> arrays, int axis = 0, ndarray _out = null)
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

            List<ndarray> _arrays = new List<ndarray>();
            foreach (var _arr in arrays)
            {
                _arrays.Add(asanyarray(_arr));
            }
            if (_arrays == null || _arrays.Count() == 0)
            {
                throw new ValueError("need at least one array to stack");
            }

            if (!ValidateSameShapes(_arrays))
            {
                throw new ValueError("all input arrays must have the same shape");
            }

            var result_ndim = _arrays[0].ndim + 1;
            axis = normalize_axis_index(axis, result_ndim);

            List<ndarray> expanded_arrays = new List<ndarray>();

            foreach (var arr in _arrays)
            {
                expanded_arrays.Add(expand_dims(arr, axis));
            }

            return np.concatenate(expanded_arrays, axis: axis);
        }

        private static (object[] bottom_index, int arr_ndim) _block_check_depths_match(object arrays, List<object> parent_index = null)
        {
           // Recursive function checking that the depths of nested lists in `arrays`
           // all match. Mismatch raises a ValueError as described in the block
           // docstring below.

           // The entire index(rather than just the depth) needs to be calculated
           // for each innermost list, in case an error needs to be raised, so that
           // the index of the offending list can be printed as part of the error.

           // The parameter `parent_index` is the full index of `arrays` within the
           // nested lists passed to _block_check_depths_match at the top of the
           // recursion.
           // The return value is a pair. The first item returned is the full index
           // of an element(specifically the first element) from the bottom of the
           //nesting in `arrays`. An empty list at the bottom of the nesting is
           //represented by a `None` index.
           // The second item is the maximum of the ndims of the arrays nested in
           // `arrays`.

            if (arrays is Tuple)
            {
                // not strictly necessary, but saves us from:
                //  - more than one way to do things - no point treating tuples like
                // lists
                // - horribly confusing behaviour that results when tuples are
                // treated like ndarray
                throw new TypeError("{} is a tuple. Only lists can be used to arrange blocks, and np.block does not allow implicit conversion from tuple to ndarray.");
            }

            if (parent_index == null)
                parent_index = new List<object>();

            if (arrays is object[])
            {
                object[] _arrays = (object[])arrays;
                if (_arrays.Length > 0)
                {
                    List<(object[], int)> idxs_ndims = new List<(object[], int)>();


 
                    for (int i = 0; i < _arrays.Length; i++)
                    {
                        List<object> temp_parent_index = new List<object>();
                        temp_parent_index.AddRange(parent_index);
                        temp_parent_index.Add(i);
                        idxs_ndims.Add(_block_check_depths_match(_arrays[i], temp_parent_index));
                    }

                    object[] first_index = (object[])idxs_ndims[0].Item1;
                    int max_arr_ndim = idxs_ndims[0].Item2;

                    for (int i = 1; i < idxs_ndims.Count; i++)
                    {
                        object[] index = idxs_ndims[i].Item1;
                        int ndim = idxs_ndims[i].Item2;

                        if (ndim > max_arr_ndim)
                            max_arr_ndim = ndim;

                        if (index.Length != first_index.Length)
                        {
                            throw new ValueError("List depths are mismatched. First element was at depth {}, but there is an element at depth {}");
                        }

                    }

                    return (first_index, max_arr_ndim);
                }
                else
                {
                    // We've 'bottomed out' on an empty list
                    parent_index.Add(null);
                    return (parent_index.ToArray(), 0);
                }
      
            }
            else
            {
                // We've 'bottomed out' - arrays is either a scalar or an array
                return (parent_index.ToArray(), np.ndim(asanyarray(arrays)));
            }
        }

        private static ndarray _block(object[] arrays, int max_depth, int result_ndim)
        {
            //Internal implementation of block. `arrays` is the argument passed to
            //block. `max_depth` is the depth of nested lists within `arrays` and
            //`result_ndim` is the greatest of the dimensions of the arrays in
            //`arrays` and the depth of the lists in `arrays` (see block docstring
            //for details).

            ndarray atleast_nd(object a, int ndim)
            {
                // Ensures `a` has at least `ndim` dimensions by prepending
                // ones to `a.shape` as necessary
                return array(a, ndmin: ndim, copy:false, subok: true);
            }

            ndarray block_recursion(object[] _arrays, int depth = 0)
            {
                if (depth < max_depth)
                {
                    if (_arrays.Length == 0)
                    {
                        throw new ValueError("Lists cannot be empty");
                    }

                    List<ndarray> arrs = new List<ndarray>();
                    foreach (var arr in _arrays)
                    {
                        if (arr.GetType().IsArray)
                        {
                            arrs.Add(block_recursion( arr as object[], depth + 1));
                        }
                        else
                        {
                            arrs.Add(block_recursion(new object[] { arr }, depth + 1));
                        }
                    }

                    return np.Concatenate(arrs, axis: -(max_depth - depth));

                }
                else
                {
                    // We've 'bottomed out' - arrays is either a scalar or an array
                    // type(arrays) is not list
                    return atleast_nd(_arrays[0], result_ndim);
                }

            }


            try
            {
                return block_recursion(arrays);
            }
            catch (Exception ex)
            {
                return null;
            }
  
        }

        public static ndarray block(object[] arrays)
        {
            //Assemble an nd - array from nested lists of blocks.

            //  Blocks in the innermost lists are concatenated(see `concatenate`) along
            // the last dimension(-1), then these are concatenated along the
            //second - last dimension(-2), and so on until the outermost list is reached.

            // Blocks can be of any dimension, but will not be broadcasted using the normal
            // rules.Instead, leading axes of size 1 are inserted, to make ``block.ndim``
            //the same for all blocks. This is primarily useful for working with scalars,
            //and means that code like ``np.block([v, 1])`` is valid, where
            //``v.ndim == 1``.

            //When the nested list is two levels deep, this allows block matrices to be
            //constructed from their components.

            //..versionadded:: 1.13.0

            //Parameters
            //----------
            //arrays: nested list of array_like or scalars(but not tuples)
            //    If passed a single ndarray or scalar(a nested list of depth 0), this
            //   is returned unmodified (and not copied).

            //    Elements shapes must match along the appropriate axes(without
            //    broadcasting), but leading 1s will be prepended to the shape as
            //    necessary to make the dimensions match.

            //Returns
            //------ -
            //block_array : ndarray
            //    The array assembled from the given blocks.

            //    The dimensionality of the output is equal to the greatest of:
            //        *the dimensionality of all the inputs
            //       * the depth to which the input list is nested

            //   Raises
            //------
            //ValueError
            //    * If list depths are mismatched - for instance, ``[[a, b], c]`` is
            //      illegal, and should be spelt ``[[a, b], [c]]``
            //    * If lists are empty - for instance, ``[[a, b], []]``

            //See Also
            //--------
            //concatenate : Join a sequence of arrays together.
            //stack : Stack arrays in sequence along a new dimension.
            //hstack : Stack arrays in sequence horizontally(column wise).
            //vstack : Stack arrays in sequence vertically(row wise).
            //dstack : Stack arrays in sequence depth wise(along third dimension).
            //vsplit : Split array into a list of multiple sub-arrays vertically.

            //Notes
            //-----

            //When called with only scalars, ``np.block`` is equivalent to an ndarray
            //call.So ``np.block([[1, 2], [3, 4]])`` is equivalent to
            //``np.array([[1, 2], [3, 4]])``.

            //This function does not enforce that the blocks lie on a fixed grid.
            //``np.block([[a, b], [c, d]])`` is not restricted to arrays of the form::

            //    AAAbb
            //    AAAbb
            //    cccDD

            //But is also allowed to produce, for some ``a, b, c, d``::

            //    AAAbb
            //    AAAbb
            //    cDDDD

            //Since concatenation happens along the last axis first, `block` is _not_
            //capable of producing the following directly::

            //    AAAbb
            //    cccbb
            //    cccDD

            //Matlab's "square bracket stacking", ``[A, B, ...; p, q, ...]``, is
            //equivalent to ``np.block([[A, B, ...], [p, q, ...]])``.

            //Examples
            //--------
            //The most common use of this function is to build a block matrix

            //>>> A = np.eye(2) * 2
            //>>> B = np.eye(3) * 3
            //>>> np.block([
            //...     [A, np.zeros((2, 3))],
            //...     [np.ones((3, 2)), B]
            //... ])
            //array([[2., 0., 0., 0., 0.],
            //       [0., 2., 0., 0., 0.],
            //       [1., 1., 3., 0., 0.],
            //       [1., 1., 0., 3., 0.],
            //       [1., 1., 0., 0., 3.]])

            //With a list of depth 1, `block` can be used as `hstack`

            //>>> np.block([1, 2, 3])              # hstack([1, 2, 3])
            //array([1, 2, 3])

            //>>> a = np.array([1, 2, 3])
            //>>> b = np.array([2, 3, 4])
            //>>> np.block([a, b, 10])             # hstack([a, b, 10])
            //array([1, 2, 3, 2, 3, 4, 10])

            //>>> A = np.ones((2, 2), int)
            //>>> B = 2 * A
            //>>> np.block([A, B])                 # hstack([A, B])
            //array([[1, 1, 2, 2],
            //       [1, 1, 2, 2]])

            //With a list of depth 2, `block` can be used in place of `vstack`:

            //>>> a = np.array([1, 2, 3])
            //>>> b = np.array([2, 3, 4])
            //>>> np.block([[a], [b]])             # vstack([a, b])
            //array([[1, 2, 3],
            //       [2, 3, 4]])

            //>>> A = np.ones((2, 2), int)
            //>>> B = 2 * A
            //>>> np.block([[A], [B]])             # vstack([A, B])
            //array([[1, 1],
            //       [1, 1],
            //       [2, 2],
            //       [2, 2]])

            //It can also be used in places of `atleast_1d` and `atleast_2d`

            //>>> a = np.array(0)
            //>>> b = np.array([1])
            //>>> np.block([a])                    # atleast_1d(a)
            //array([0])
            //>>> np.block([b])                    # atleast_1d(b)
            //array([1])

            //>>> np.block([[a]])                  # atleast_2d(a)
            //array([[0]])
            //>>> np.block([[b]])                  # atleast_2d(b)
            //array([[1]])


            var bcdm_return = _block_check_depths_match(arrays);
            object[] list_ndim = bcdm_return.bottom_index;
            return _block(arrays, list_ndim.Length, Math.Max(bcdm_return.arr_ndim, list_ndim.Length));
        }

   

        public static ndarray expand_dims(ndarray a, int axis)
        {
            // Expand the shape of an array.

            // Insert a new axis that will appear at the `axis` position in the expanded
            // array shape.

            // .. note::Previous to NumPy 1.13.0, neither ``axis < -a.ndim - 1`` nor
            //    ``axis > a.ndim`` raised errors or put the new axis where documented.
            //    Those axis values are now deprecated and will raise an AxisError in the
            //    future.

            // Parameters
            // ----------
            // a: array_like
            //    Input array.
            //axis : int
            //    Position in the expanded axes where the new axis is placed.

            //Returns
            // -------
            // res : ndarray
            //     Output array.The number of dimensions is one greater than that of
            //    the input array.

            // See Also
            // --------
            // squeeze : The inverse operation, removing singleton dimensions
            // reshape : Insert, remove, and combine dimensions, and resize existing ones
            // doc.indexing, atleast_1d, atleast_2d, atleast_3d

            // Examples
            // --------
            // >>> x = np.array([1, 2])
            // >>> x.shape
            // (2,)

            // The following is equivalent to ``x[np.newaxis,:]`` or ``x[np.newaxis]``:

            // >>> y = np.expand_dims(x, axis = 0)
            // >>> y
            // array([[1, 2]])
            // >>> y.shape
            // (1, 2)

            // >>> y = np.expand_dims(x, axis = 1)  # Equivalent to x[:,np.newaxis]
            // >>> y
            // array([[1],
            //        [2]])
            // >>> y.shape
            // (2, 1)

            // Note that some examples may use ``None`` instead of ``np.newaxis``.  These
            // are the same objects:

            // >>> np.newaxis is None
            // True

            if (axis > a.ndim || axis < -a.ndim - 1)
            {
                throw new ValueError("axis value is out of range for this array");
            }

            axis = normalize_axis_index(axis, a.ndim + 1);


            npy_intp[] ExpandedDims = new npy_intp[a.dims.Length + 1];

            int j = 0;
            for (int i = 0; i < ExpandedDims.Length; i++)
            {
                if (i == axis)
                {
                    ExpandedDims[i] = 1;
                }
                else
                {
                    ExpandedDims[i] = a.Dim(j);
                    j++;
                }
            }

            return a.reshape(new shape(ExpandedDims));
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

            List<ndarray> arrays = new List<ndarray>();
            foreach (var v in tup)
            {
                var arr = array(v, copy:false, subok:true);
                if (arr.ndim < 2)
                {
                    arr = array(arr, copy: false, subok: true, ndmin: 2).T;
                }
                arrays.Add(arr);
            }

            return np.concatenate(arrays, 1);
        }

        public static ndarray row_stack(ICollection<object> tup)
        {
            return vstack(tup);
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

            return np.concatenate(atleast_3d(tup), 2);
        }

        public static ICollection<ndarray> array_split(ndarray ary, object indices_or_sections, int axis = 0)
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

            npy_intp Ntotal = 0;
            int Nsections = 0;

            if (axis < ary.dims.Length)
            {
                Ntotal = ary.Dim(axis);
            }
            else
            {
                Ntotal = len(ary);
            }

            List<npy_intp> div_points = new List<npy_intp>();
            if (indices_or_sections.GetType().IsArray)
            {
                int[] _indices_or_sections = indices_or_sections as int[];
                Nsections = _indices_or_sections.Length + 1;
                div_points.Add(0);
                foreach (var _indices in _indices_or_sections)
                    div_points.Add(_indices);
                div_points.Add(Ntotal);
            }
            else
            {
                Nsections = int.Parse(indices_or_sections.ToString());
                if (Nsections <= 0)
                    throw new ValueError("number sections must be larger than 0.");

                npy_intp Neach_section = Ntotal / Nsections;
                npy_intp extras = Ntotal % Nsections;

                List<npy_intp> section_sizes = new List<npy_intp>();
                section_sizes.Add(0);
                for (int i = 0; i < extras; i++)
                    section_sizes.Add(Neach_section + 1);
                for (int i = 0; i < Nsections - extras; i++)
                    section_sizes.Add(Neach_section);

                var dpoints = array(section_sizes.ToArray()).cumsum();
                div_points = dpoints.ToList<npy_intp>();
            }

            List<ndarray> sub_arys = new List<ndarray>();
            var sary = swapaxes(ary, axis, 0);
            for (int i = 0; i < Nsections; i++)
            {
                npy_intp st = div_points[i];
                npy_intp end = div_points[i + 1];
                sub_arys.Add(swapaxes(sary.A(st.ToString() + ":" + end.ToString()), axis, 0));
            }

            return sub_arys.ToArray();
        }

        public static ICollection<ndarray> split(ndarray ary, object indices_or_sections, int axis = 0)
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

            if (!indices_or_sections.GetType().IsArray)
            {
                int sections = int.Parse(indices_or_sections.ToString());
                var N = ary.shape.iDims[axis];
                if (N % sections != 0)
                    throw new ValueError("array split does not result in an equal division");
            }

            var res = array_split(ary, indices_or_sections, axis);
            return res;
        }

        public static ICollection<ndarray> hsplit(ndarray ary, object indices_or_sections)
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

            if (ndim(ary) == 0)
                throw new ValueError("hsplit only works on arrays of 1 or more dimensions");
            if (ary.ndim > 1)
                return split(ary, indices_or_sections, 1);
            else
                return split(ary, indices_or_sections, 0);
        }

        public static ICollection<ndarray> vsplit(ndarray ary, object indices_or_sections)
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

            if (ndim(ary) < 2)
                throw new ValueError("vsplit only works on arrays of 2 or more dimensions");

            return split(ary, indices_or_sections, 0);
        }

        public static ICollection<ndarray> dsplit(ndarray ary, object indices_or_sections)
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

            if (ndim(ary) < 3)
                throw new ValueError("dsplit only works on arrays of 3 or more dimensions");

            return split(ary, indices_or_sections, 2);
        }


        private delegate ndarray array_wrapper(ndarray r);
        private static array_wrapper get_array_wrap(ndarray a1, ndarray b1)
        {
            return null;
        }

        private static array_wrapper get_array_prepare(ndarray a1, ndarray b1)
        {
            return null;
        }
        public static ndarray kron(object a, object b)
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


            var b1 = asanyarray(b);
            var a1 = array(a, copy: false, subok: true, ndmin: b1.ndim);
            var ndb = b1.ndim;
            var nda = a1.ndim;
            if (nda == 0 || ndb == 0)
            {
                return np.multiply(a, b);
            }

            List<npy_intp> new_ashape = new List<npy_intp>();
            List<npy_intp> new_bshape = new List<npy_intp>();

            var as_ = a1.shape;
            var bs = b1.shape;

            new_ashape.AddRange(as_.iDims);
            new_bshape.AddRange(bs.iDims);

            if (!a1.flags.contiguous)
                a1 = reshape(a1, as_);
            if (!b1.flags.contiguous)
                b1 = reshape(b1, bs);

            var nd = ndb;
            if (ndb != nda)
            {
                if (ndb > nda)
                {
                    new_ashape = new List<npy_intp>();
                    for (int i = 0; i < ndb - nda; i++)
                        new_ashape.Add(1);
                    new_ashape.AddRange(as_.iDims);
                }
                else
                {
                    new_bshape = new List<npy_intp>();
                    for (int i = 0; i < nda - ndb; i++)
                        new_bshape.Add(1);
                    new_bshape.AddRange(bs.iDims);
                    nd = nda;
                }
            }

            new_ashape.AddRange(new_bshape);
            var result = outer(a, b).reshape(new_ashape.ToArray());
            var axis = nd - 1;
            for (int i = 0; i < nd; i++)
            {
               // Console.WriteLine("input = {0}", result);
                result = np.concatenate( result , axis : axis);
               // Console.WriteLine("output = {0}", result);
            }
            var wrapper = get_array_prepare(a1, b1);
            if (wrapper != null)
                result = wrapper(result);
            wrapper = get_array_wrap(a1, b1);
            if (wrapper != null)
                result = wrapper(result);
            return result;
        }

  

        public static ndarray tile(ndarray A, object reps)
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


            var tup = asanyarray(reps);

            var d = len(tup);

            if ((bool)(all(tup == 1).GetItem(0)) == true)
            {
                // Fixes the problem that the function does not make a copy if A is a
                // numpy array and the repetitions are 1 in all dimensions
                return np.array(A, copy: true, subok: true, ndmin: d);
            }

     
            // Note that no copy of zero-sized arrays is made. However since they
            // have no data there is no risk of an inadvertent overwrite.
            ndarray c = np.array(A, copy: false, subok: true, ndmin: d);

            List<npy_intp> tupList = new List<npy_intp>();
            foreach (var t1 in tup)
                tupList.Add(Convert.ToInt64(t1));

            if (d < c.ndim)
            {
                //tup = (1,) * (c.ndim - d) + tup
                tupList = new List<npy_intp>();
                for (int i = 0; i < c.ndim - d; i++)
                    tupList.Add(1);
                foreach (var t1 in tup)
                    tupList.Add(Convert.ToInt64(t1));
            }

            List<npy_intp> shape_out = new List<npy_intp>();
            for (int i = 0; i < c.ndim; i++)
            {
                shape_out.Add(c.shape.iDims[i] * tupList[i]);
            }

            var n = c.size;
            if (n > 0)
            {
                var originalShape = c.shape.iDims;
                for (int i = 0; i < originalShape.Length; i++)
                {
                    npy_intp dim_in = originalShape[i];
                    npy_intp nrep = tupList[i];

                    if (nrep != 1)
                    {
                        c = c.reshape(((npy_intp)(-1), n)).Repeat(nrep, 0);
                    }
                    n /= dim_in;

                }
            }
            return c.reshape(new shape(shape_out));
        }

        private static bool ValidateSameShapes(List<ndarray> arrays)
        {
            Dictionary<string, string> UniqueShapes = new Dictionary<string, string>();

            foreach (var arr in arrays)
            {
                string Key = "";

                foreach (var s in arr.dims)
                {
                    Key += s.ToString() + "_";
                }

                if (!UniqueShapes.ContainsKey(Key))
                {
                    UniqueShapes.Add(Key, Key);
                }
            }

            return UniqueShapes.Count == 1;
        }

    }
}
