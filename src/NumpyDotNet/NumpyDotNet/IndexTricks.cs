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

        public static object mgrid(Slice[] key)
        {
            return nd_grid(key, false);
        }

        public static object ogrid(Slice[] key)
        {
            return nd_grid(key, true);
        }

        // Construct a multi-dimensional "meshgrid".

        // ``grid = nd_grid()`` creates an instance which will return a mesh-grid
        // when indexed.The dimension and number of the output arrays are equal
        // to the number of indexing dimensions.If the step length is not a
        // complex number, then the stop is not inclusive.

        // However, if the step length is a** complex number** (e.g. 5j), then the
        // integer part of its magnitude is interpreted as specifying the
        // number of points to create between the start and stop values, where
        // the stop value **is inclusive**.

        // If instantiated with an argument of ``sparse=True``, the mesh-grid is
        // open(or not fleshed out) so that only one-dimension of each returned
        // argument is greater than 1.

        // Parameters
        // ----------
        // sparse : bool, optional
        //     Whether the grid is sparse or not.Default is False.

        //Notes
        // -----
        // Two instances of `nd_grid` are made available in the NumPy namespace,
        // `mgrid` and `ogrid`::

        //     mgrid = nd_grid(sparse= False)
        //     ogrid = nd_grid(sparse= True)

        // Users should use these pre-defined instances instead of using `nd_grid`
        // directly.

        // Examples
        // --------
        // >>> mgrid = np.lib.index_tricks.nd_grid()
        // >>> mgrid[0:5, 0:5]
        // array([[[0, 0, 0, 0, 0],
        //         [1, 1, 1, 1, 1],
        //         [2, 2, 2, 2, 2],
        //         [3, 3, 3, 3, 3],
        //         [4, 4, 4, 4, 4]],
        //        [[0, 1, 2, 3, 4],
        //         [0, 1, 2, 3, 4],
        //         [0, 1, 2, 3, 4],
        //         [0, 1, 2, 3, 4],
        //         [0, 1, 2, 3, 4]]])
        // >>> mgrid[-1:1:5j]
        // array([-1. , -0.5,  0. ,  0.5,  1. ])

        // >>> ogrid = np.lib.index_tricks.nd_grid(sparse=True)
        // >>> ogrid[0:5, 0:5]
        // [array([[0],
        //         [1],
        //         [2],
        //         [3],
        //         [4]]), array([[0, 1, 2, 3, 4]])]

        private static object nd_grid(Slice[] key, bool Sparse)
        {
            ValidateKeyData(key);

            if (key.Length > 1)
            {
                List<npy_intp> size = new List<npy_intp>();
                dtype typ = np.Int32;

                for (int k = 0; k < key.Length; k++)
                {
                    var step = key[k].step;
                    var start = key[k].start;
                    if (start == null)
                        start = 0;
                    if (step == null)
                        step = 1;

                    if (IsComplex(step))
                    {
                        size.Add(Convert.ToInt32(Math.Abs(Convert.ToDouble(step))));
                        typ = np.Float64;
                    }
                    else
                    {
                        size.Add(Convert.ToInt32(Math.Ceiling((ConvertToDouble(key[k].stop) - ConvertToDouble(start)) / (ConvertToDouble(step) * 1.0))));
                    }

   
                    if (FloatingPointValue(step) ||
                        FloatingPointValue(start) ||
                        FloatingPointValue(key[k].Stop))
                    {
                        typ = np.Float64;
                    }
                    else
                    if (DecimalValue(step) ||
                        DecimalValue(start) ||
                        DecimalValue(key[k].Stop))
                    {
                        typ = np.Decimal;
                    }
                    else
                    if (ComplexValue(step) ||
                        ComplexValue(start) ||
                        ComplexValue(key[k].Stop))
                    {
                        typ = np.Complex;
                    }
                    else
                    if (BigIntValue(step) ||
                        BigIntValue(start) ||
                        BigIntValue(key[k].Stop))
                    {
                        typ = np.BigInt;
                    }

                }


                ndarray nn = null;
                ndarray []nnarray = null;
                if (Sparse)
                {
                    List<ndarray> _nnarray = new List<ndarray>();
                    for (int i = 0; i < size.Count; i++)
                    {
                        _nnarray.Add(arange(size[i], dtype: typ));
                    }
                    nnarray = _nnarray.ToArray();
                }
                else
                {
                    nn = np.indices(new shape(size), typ);
                }

                for (int k = 0; k < size.Count; k++)
                {
                    var step = key[k].step;
                    var start = key[k].start;
                    if (start == null)
                        start = 0;
                    if (step == null)
                        step = 1;

                    if (IsComplex(step))
                    {
                        //step = int(abs(step))
                        //if step != 1:
                        //    step = (key[k].stop - start) / float(step - 1)
                    }
                    if (Sparse)
                    {
                        if (typ.TypeNum == NPY_TYPES.NPY_FLOAT)
                        {
                            nnarray[k] = nnarray[k] * Convert.ToSingle(step) + Convert.ToSingle(start);
                        }
                        if (typ.TypeNum == NPY_TYPES.NPY_DOUBLE)
                        {
                            nnarray[k] = nnarray[k] * Convert.ToDouble(step) + Convert.ToDouble(start);
                        }
                        if (typ.TypeNum == NPY_TYPES.NPY_DECIMAL)
                        {
                            nnarray[k] = nnarray[k] * Convert.ToDecimal(step) + Convert.ToDecimal(start);
                        }
                        if (typ.TypeNum == NPY_TYPES.NPY_COMPLEX)
                        {
                            nnarray[k] = nnarray[k] *  ConvertToComplex(step) + ConvertToComplex(start);
                        }
                        if (typ.TypeNum == NPY_TYPES.NPY_BIGINT)
                        {
                            nnarray[k] = nnarray[k] * ConvertToBigInt(step) + ConvertToBigInt(start);
                        }
                        if (typ.TypeNum == NPY_TYPES.NPY_INT32)
                        {
                            nnarray[k] = nnarray[k] * Convert.ToInt32(step) + Convert.ToInt32(start);
                        }
                    }
                    else
                    {
                        if (typ.TypeNum == NPY_TYPES.NPY_FLOAT)
                        {
                            nn[k] = ((nn[k] as ndarray) * Convert.ToSingle(step) + Convert.ToSingle(start));
                        }
                        if (typ.TypeNum == NPY_TYPES.NPY_DOUBLE)
                        {
                            nn[k] = ((nn[k] as ndarray) * Convert.ToDouble(step) + Convert.ToDouble(start));
                        }
                        if (typ.TypeNum == NPY_TYPES.NPY_DECIMAL)
                        {
                            nn[k] = ((nn[k] as ndarray) * Convert.ToDecimal(step) + Convert.ToDecimal(start));
                        }
                        if (typ.TypeNum == NPY_TYPES.NPY_COMPLEX)
                        {
                            nn[k] = ((nn[k] as ndarray) * ConvertToComplex(step) + ConvertToComplex(start));
                        }
                        if (typ.TypeNum == NPY_TYPES.NPY_BIGINT)
                        {
                            nn[k] = ((nn[k] as ndarray) * ConvertToBigInt(step) + ConvertToBigInt(start));
                        }
                        if (typ.TypeNum == NPY_TYPES.NPY_INT32)
                        {
                            nn[k] = ((nn[k] as ndarray) * Convert.ToInt32(step) + Convert.ToInt32(start));
                        }
                    }
    
                }

                if (Sparse)
                {
                    var slobj = new object[size.Count];
                    for (int j = 0; j < size.Count; j++)
                        slobj[j] = np.newaxis;

                    for (int k = 0; k < size.Count; k++)
                    {
                        slobj[k] = new Slice(null, null);
                        nnarray[k] = (ndarray)nnarray[k][slobj];
                        slobj[k] = np.newaxis;
                    }

                    return nnarray;
                }
                else
                {
                    return nn;
                }

            }
            else
            {
                var step = key[0].step;
                var stop = key[0].stop;
                var start = key[0].start;
                if (start == null)
                    start = 0;

                if (IsComplex(step))
                {
                    //step = abs(step);
                    //length = int(step)
                    //if step != 1:
                    //    step = (key.stop - start) / float(step - 1)
                    //stop = key.stop + step
                    //return _nx.arange(0, length, 1, float) * step + start
                    throw new NotImplementedException();
                }
                else
                {
                    var tt = asanyarray(start);
                    if (tt.IsInexact)
                    {
                        if (tt.IsDecimal)
                        {
                            decimal? iStart = null;
                            decimal? iStop = null;
                            decimal? iStep = null;
                            if (start != null)
                                iStart = Convert.ToDecimal(start);
                            if (stop != null)
                                iStop = Convert.ToDecimal(stop);
                            if (step != null)
                                iStep = Convert.ToDecimal(step);

                            return arange(iStart.Value, iStop, iStep, dtype: np.Decimal);
                        }
                        else
                        if (tt.IsComplex)
                        {
                            System.Numerics.Complex? iStart = null;
                            System.Numerics.Complex? iStop = null;
                            System.Numerics.Complex? iStep = null;
                            if (start != null)
                                iStart = (System.Numerics.Complex)start;
                            if (stop != null)
                                iStop = (System.Numerics.Complex)stop;
                            if (step != null)
                                iStep = (System.Numerics.Complex)step;

                            return arange(iStart.Value, iStop, iStep, dtype: np.Complex);
                        }
                        else
                        {
                            double? iStart = null;
                            double? iStop = null;
                            double? iStep = null;
                            if (start != null)
                                iStart = Convert.ToDouble(start);
                            if (stop != null)
                                iStop = Convert.ToDouble(stop);
                            if (step != null)
                                iStep = Convert.ToDouble(step);

                            return arange(iStart.Value, iStop, iStep, dtype: np.Float64);
                        }
  
                    }
                    else
                    {
                        if (tt.IsBigInt)
                        {
                            System.Numerics.BigInteger? iStart = null;
                            System.Numerics.BigInteger? iStop = null;
                            System.Numerics.BigInteger? iStep = null;
                            if (start != null)
                                iStart = (System.Numerics.BigInteger)start;
                            if (stop != null)
                                iStop = (System.Numerics.BigInteger)stop;
                            if (step != null)
                                iStep = (System.Numerics.BigInteger)step;

                            return arange(iStart.Value, iStop, iStep);
                        }
                        else
                        {
                            Int64? iStart = null;
                            Int64? iStop = null;
                            Int64? iStep = null;
                            if (start != null)
                                iStart = Convert.ToInt64(start);
                            if (stop != null)
                                iStop = Convert.ToInt64(stop);
                            if (step != null)
                                iStep = Convert.ToInt64(step);

                            return arange(iStart.Value, iStop, iStep);
                        }
    
                    }
                }
    

            }
        }

        private static void ValidateKeyData(Slice[] key)
        {
            foreach (var k in key)
            {
                if (k.Start is string)
                    throw new Exception(string.Format("This function does not support Start values of {0}", k.Start.ToString()));
                if (k.Stop is string)
                    throw new Exception(string.Format("This function does not support Stop values of {0}", k.Stop.ToString()));
                if (k.Step is string)
                    throw new Exception(string.Format("This function does not support Step values of {0}", k.Step.ToString()));

            }
        }

        private static System.Numerics.Complex ConvertToComplex(object value)
        {
            if (value is System.Numerics.Complex)
            {
                return (System.Numerics.Complex)value;
            }
            if (value is IComparable)
            {
                return new System.Numerics.Complex(Convert.ToDouble(value), 0);
            }

            return System.Numerics.Complex.Zero;

        }

        private static System.Numerics.BigInteger ConvertToBigInt(object value)
        {
            if (value is System.Numerics.BigInteger)
            {
                return (System.Numerics.BigInteger)value;
            }
            if (value is IComparable)
            {
                return new System.Numerics.BigInteger(Convert.ToDouble(value));
            }

            return System.Numerics.BigInteger.Zero;

        }

        private static bool FloatingPointValue(object value)
        {
            if (value is float)
                return true;
            if (value is double)
                return true;
            return false;

        }
        private static bool DecimalValue(object value)
        {
            if (value is decimal)
                return true;
            return false;
        }
        private static bool ComplexValue(object value)
        {
            if (value is System.Numerics.Complex)
                return true;
            return false;
        }
        private static bool BigIntValue(object value)
        {
            if (value is System.Numerics.BigInteger)
                return true;
            return false;
        }

        private static bool IsComplex(object step)
        {
            return false;
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
