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
using System.Text;
using System.Threading.Tasks;

namespace NumpyDotNet
{
    public static partial class np
    {
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

            source = normalize_axis_tuple(source, a.ndim, "source");
            destination = normalize_axis_tuple(destination, a.ndim, "destination");
            //if (len(source) != len(destination))
            //{
            //    throw new Exception("`source` and `destination` arguments must have the same number of elements");
            //}

            return null;
        }

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

            throw new NotImplementedException();
        }

        private static dynamic normalize_axis_tuple(dynamic axis, int ndim, string argname = null, bool allow_duplicates = false)
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

            return 0;
            throw new NotImplementedException();


        //    try:
        //    axis = [operator.index(axis)]
        //except TypeError:
        //    axis = tuple(axis)
        //axis = tuple(normalize_axis_index(ax, ndim, argname) for ax in axis)
        //        if not allow_duplicate and len(set(axis)) != len(axis):
        //    if argname:
        //        raise ValueError('repeated axis in `{}` argument'.format(argname))
        //    else:
        //        raise ValueError('repeated axis')
        //return axis

        }

        private static int normalize_axis_index(int axis, int ndim, string argname)
        {
            return 0;
        }
    }
}
