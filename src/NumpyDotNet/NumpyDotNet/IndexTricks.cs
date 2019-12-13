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
        #region mgrid/ogrid

        public static ndarray mgrid(Slice[] key)
        {
            return nd_grid(key, false);
        }

        public static ndarray ogrid(Slice[] key)
        {
            return nd_grid(key, true);
        }

        private static ndarray nd_grid(Slice[] key, bool Sparse)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region fill_diagonal

        public static void fill_diagonal(ndarray a, object val, bool wrap = false)
        {
            /*
            Fill the main diagonal of the given array of any dimensionality.

            For an array `a` with ``a.ndim >= 2``, the diagonal is the list of
            locations with indices ``a[i, ..., i]`` all identical. This function
            modifies the input array in-place, it does not return a value.

            Parameters
            ----------
            a : array, at least 2-D.
              Array whose diagonal is to be filled, it gets modified in-place.

            val : scalar
              Value to be written on the diagonal, its type must be compatible with
              that of the array a.

            wrap : bool
              For tall matrices in NumPy version up to 1.6.2, the
              diagonal "wrapped" after N columns. You can have this behavior
              with this option. This affects only tall matrices.
            */

            if (a.ndim < 2)
            {
                throw new ValueError("array must be at least 2-d");
            }

            npy_intp? end = null;
            npy_intp? step = null;

            if (a.ndim == 2)
            {
                // Explicit, fast formula for the common case.  For 2-d arrays, we
                // accept rectangular ones.
                step = a.shape.iDims[1] + 1;
                // This is needed to don't have tall matrix have the diagonal wrap.
                if (!wrap)
                {
                    end = a.shape.iDims[1] * a.shape.iDims[1];
                }
            }
            else
            {
                // For more than d=2, the strided formula is only valid for arrays with
                // all dimensions equal, so we check first.
                if (!allb(diff(a.shape.iDims) == 0))
                {
                    throw new ValueError("All dimensions of input must be of equal length");
                }
                step = 1 + (npy_intp)(cumprod(a.shape.iDims).A(":-1")).Sum().GetItem(0);
            }

            // Write the value out into the diagonal.
            a.Flat[":" + end.ToString()  + ":" + step.ToString()] = val;
        }

        #endregion

        #region diag_indices
        public static ndarray[] diag_indices(int n, int ndim = 2)
        {
            //Return the indices to access the main diagonal of an array.

            //This returns a tuple of indices that can be used to access the main
            //diagonal of an array `a` with ``a.ndim >= 2`` dimensions and shape
            //(n, n, ..., n).For ``a.ndim = 2`` this is the usual diagonal, for
            //``a.ndim > 2`` this is the set of indices to access ``a[i, i, ..., i]``
            //for ``i = [0..n - 1]``.

            //Parameters
            //----------
            //n : int
            //  The size, along each dimension, of the arrays for which the returned
            //  indices can be used.

            //ndim : int, optional
            //  The number of dimensions.

            //See also
            //--------
            //diag_indices_from

            //Notes
            //---- -
            //..versionadded:: 1.4.0

            //Examples
            //--------
            //Create a set of indices to access the diagonal of a(4, 4) array:

            //>>> di = np.diag_indices(4)
            //>>> di
            //(array([0, 1, 2, 3]), array([0, 1, 2, 3]))
            //>>> a = np.arange(16).reshape(4, 4)
            //>>> a
            //array([[0, 1, 2, 3],
            //       [ 4,  5,  6,  7],
            //       [ 8,  9, 10, 11],
            //       [12, 13, 14, 15]])
            //>>> a[di] = 100
            //>>> a
            //array([[100,   1,   2,   3],
            //       [  4, 100,   6,   7],
            //       [  8,   9, 100,  11],
            //       [ 12,  13,  14, 100]])

            //Now, we create indices to manipulate a 3-D array:

            //>>> d3 = np.diag_indices(2, 3)
            //>>> d3
            //(array([0, 1]), array([0, 1]), array([0, 1]))

            //And use it to set the diagonal of an array of zeros to 1:

            //>>> a = np.zeros((2, 2, 2), dtype=int)
            //>>> a[d3] = 1
            //>>> a
            //array([[[1, 0],
            //        [0, 0]],
            //       [[0, 0],
            //        [0, 1]]])

            var idx = arange(n);

            List<ndarray> lidx = new List<ndarray>();
            for (int i = 0; i < ndim; i++)
            {
                lidx.Add(np.copy(idx));
            }

            return lidx.ToArray();
        }
        #endregion

        #region diag_indices_from

        public static ndarray[] diag_indices_from(ndarray arr)
        {
            //Return the indices to access the main diagonal of an n - dimensional array.

            //See `diag_indices` for full details.


            //Parameters
            //----------

            //arr : array, at least 2 - D


            //See Also
            //--------

            //diag_indices

            if (arr.ndim < 2)
            {
                throw new ValueError("input array must be at least 2-d");
            }

            // For more than d=2, the strided formula is only valid for arrays with
            // all dimensions equal, so we check first.

            var FirstDim = arr.Dim(0);
            for (int i = 1; i < arr.ndim; i++)
            {
                if (arr.Dim(i) != FirstDim)
                {
                    throw new ValueError("All dimensions of input must be of equal length");
                }
            }

            return diag_indices((int)arr.Dim(0), arr.ndim);
        }

        #endregion
    }
}
