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
        private static ndarray _maybe_view_as_subclass(object original_array, object new_array)
        {
            ndarray array = null;
            if (original_array.GetType() != new_array.GetType())
            {
                // if input was an ndarray subclass and subclasses were OK,
                // then view the result as that subclass.
                array = asanyarray(new_array).view(asanyarray(original_array).Dtype);
                // Since we have done something akin to a view from original_array, we
                // should let the subclass finalize (if it has it implemented, i.e., is
                // not None).

                //if (array.__array_finalize__)
                //    array.__array_finalize__(original_array)
            }
            else
            {
                array = asanyarray(new_array);
            }

            return array;
        }

        public static ndarray as_strided(ndarray x, object oshape = null, object ostrides = null, bool subok = false, bool writeable = false)
        {
            npy_intp[] shape = null;
            npy_intp[] strides = null;

            if (oshape != null)
            {
                shape newshape = NumpyExtensions.ConvertTupleToShape(oshape);
                if (newshape == null)
                {
                    throw new Exception("Unable to convert shape object");
                }
                shape = newshape.iDims;
            }

            if (ostrides != null)
            {
                shape newstrides = NumpyExtensions.ConvertTupleToShape(ostrides);
                if (newstrides == null)
                {
                    throw new Exception("Unable to convert strides object");
                }
                strides = newstrides.iDims;
            }

            return as_strided(x, shape, strides, subok, writeable);

        }


        public static ndarray as_strided(ndarray x, npy_intp[] shape = null, npy_intp[] strides = null, bool subok = false, bool writeable = false)
        {
            var view = np.array(x, copy: false, subok: subok);

            if (shape != null)
                NpyCoreApi.SetArrayDimsOrStrides(view, shape, shape.Length, true);
            if (strides != null)
                NpyCoreApi.SetArrayDimsOrStrides(view, strides, strides.Length, false);

            if (view.flags.writeable && !writeable)
                view.flags.writeable = false;

            return view;
        }


        private static bool broadcastable(npy_intp[] adims, int and, npy_intp[] bdims, int bnd)
        {
            if (adims[and - 1] == 1 || bdims[bnd - 1] == 1)
                return true;

            if (adims[and - 1] == bdims[bnd - 1])
                return true;

            return false;
        }
        private static bool broadcastable(ndarray ao, ndarray bo)
        {
            return broadcastable(ao, bo.dims, bo.ndim);
        }
        private static bool broadcastable(ndarray ao, npy_intp[] bdims, int bnd)
        {
            return broadcastable(ao.dims, ao.ndim, bdims, bnd);
        }

        public static ndarray upscale_to(ndarray a, object oshape)
        {
            shape newshape = NumpyExtensions.ConvertTupleToShape(oshape);
            if (newshape == null)
            {
                throw new Exception("Unable to convert shape object");
            }

            if (!broadcastable(a, newshape.iDims, newshape.iDims.Length))
            {
                throw new Exception(string.Format("operands could not be broadcast together with shapes ({0}),({1})", a.shape.ToString(), newshape.ToString()));
            }

            ndarray ret = NpyCoreApi.NpyArray_UpscaleSourceArray(a, newshape);
            return ret.reshape(newshape);
        }

        private static ndarray _broadcast_to(object oarray, object oshape, bool subok, bool _readonly)
        {
            shape newshape = NumpyExtensions.ConvertTupleToShape(oshape);
            if (newshape == null)
            {
                throw new Exception("Unable to convert shape object");
            }

            if (newshape.iDims == null || newshape.iDims.Length == 0)
            {
                newshape = new shape(asanyarray(oarray).shape);
            }

            ndarray array = np.array(asanyarray(oarray), copy: false, subok: subok);

            if (array.dims == null)
            {
                throw new ValueError("cannot broadcast a non-scalar to a scalar array");
            }

            if (np.anyb(np.array(newshape.iDims) < 0))
            {
                throw new ValueError("all elements of broadcast shape must be non-negative");
            }

            var toshape = NpyCoreApi.BroadcastToShape(array, newshape.iDims, newshape.iDims.Length);

            var newStrides = new npy_intp[toshape.nd_m1 + 1];
            Array.Copy(toshape.strides, 0, newStrides, 0, newStrides.Length);
            var result = np.as_strided(array, shape: newshape.iDims, strides: newStrides);
 
            return result;
        }

        public static ndarray broadcast_to(object array, object shape, bool subok = false)
        {
            /*

            Parameters
            ----------
            array : array_like
                The array to broadcast.
            shape : tuple
                The shape of the desired array.
            subok : bool, optional
                If True, then sub-classes will be passed-through, otherwise
                the returned array will be forced to be a base-class array (default).

            Returns
            -------
            broadcast : array
                A readonly view on the original array with the given shape. It is
                typically not contiguous. Furthermore, more than one element of a
                broadcasted array may refer to a single memory location.

            Raises
            ------
            ValueError
                If the array is not compatible with the new shape according to NumPy's
                broadcasting rules.

            Notes
            -----
            .. versionadded:: 1.10.0

            Examples
            --------
            >>> x = np.array([1, 2, 3])
            >>> np.broadcast_to(x, (3, 3))
            array([[1, 2, 3],
                   [1, 2, 3],
                   [1, 2, 3]])             
            */

            //return upscale_to(asanyarray(array), shape);
            return _broadcast_to(array, shape, subok : subok, _readonly : true);
        }

        private static shape _broadcast_shape(ndarray [] args)
        {
            if (args == null)
            {
                return new shape(0);
            }

            var b = np.broadcast(args);

            ndarray b1 = np.broadcast_to(0, b.shape);
            for (int pos = 1; pos < args.Length; pos++)
            {
                b1 = np.broadcast_to(0, b1.shape);
            }

            return b1.shape;
        }

  
        public static IEnumerable<ndarray> broadcast_arrays(bool subok, IEnumerable<ndarray> args)
        {
            ndarray[] arrays = new ndarray[args.Count()];

            for (int i = 0; i < args.Count(); i++)
            {
                arrays[i] = np.array(args.ElementAt(i), copy: false, subok: subok);
            }

            var shape = _broadcast_shape(arrays);

            bool allsame = true;
            for (int i = 0; i < arrays.Length; i++)
            {
                if (shape != arrays[i].shape)
                {
                    allsame = false;
                    break;
                }

            }
            if (allsame)
            {
                return arrays;
            }

            ndarray[] broadcastedto = new ndarray[args.Count()];

            for (int i = 0; i < args.Count(); i++)
            {
                broadcastedto[i] = _broadcast_to(args.ElementAt(i), shape, subok: subok, _readonly: false);
            }

            return broadcastedto;
        }

    }
}
