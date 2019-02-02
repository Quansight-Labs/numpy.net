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

using NumpyDotNet;
using NumpyLib;
using System;
using System.Collections.Generic;
using System.IO;
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
    public static class NumpyExtensions
    {
        public static ndarray reshape(this ndarray a, shape newshape, NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            return np.reshape(a, newshape, order);
        }

        public static void tofile(this ndarray a, string fileName, string sep = null, string format = null)
        {
            np.tofile(a, fileName, sep, format);
        }

        public static void tofile(this ndarray a, Stream stream, string sep = null, string format = null)
        {
            np.tofile(a, stream, sep, format);
        }

        public static ndarray view(this ndarray a, dtype dtype = null, object type = null)
        {
            return np.view(a, dtype, type);
        }

        public static ndarray Flatten(this ndarray a, NPY_ORDER order)
        {
            return NpyCoreApi.Flatten(a, order);
        }

        public static ndarray Ravel(this ndarray a, NPY_ORDER order)
        {
            return np.ravel(a, order);
        }

        public static void Resize(this ndarray a, npy_intp[] newdims, bool refcheck, NPY_ORDER order)
        {
            np.resize(a, newdims, refcheck, order);
        }

        public static ndarray Squeeze(this ndarray a)
        {
            return np.squeeze(a);
        }

        public static ndarray SwapAxes(this ndarray a, int a1, int a2)
        {
            return np.swapaxes(a, a1, a2);
        }
 
        public static ndarray Transpose(this ndarray a, npy_intp[] permute = null)
        {
            return np.transpose(a, permute);
        }

        public static ndarray Choose(this ndarray a, IEnumerable<ndarray> choices, ndarray ret = null, NPY_CLIPMODE clipMode = NPY_CLIPMODE.NPY_RAISE)
        {
            return np.choose(a, choices, ret, clipMode);
        }


        public static ndarray Repeat(this ndarray a, ndarray repeats, int? axis)
        {
            return np.repeat(a, repeats, axis);
        }

        public static void PutTo(this ndarray a, ndarray values, ndarray indices, NPY_CLIPMODE mode)
        {
            int ret = np.put(a, indices, values, mode);
        }

        public static ndarray Sort(this ndarray a, int? axis = -1, NPY_SORTKIND sortkind = NPY_SORTKIND.NPY_QUICKSORT)
        {
            return np.sort(a, axis, sortkind, null);
        }


        public static ndarray ArgSort(this ndarray a, int? axis =-1, NPY_SORTKIND kind = NPY_SORTKIND.NPY_QUICKSORT)
        {
            return np.argsort(a, axis, kind);
        }

        public static ndarray ArgMax(this ndarray a, int? axis = -1, ndarray ret = null)
        {
            return np.argmax(a, axis, ret);
        }

        public static ndarray ArgMin(this ndarray a, int? axis = -1, ndarray ret = null)
        {
            return np.argmin(a, axis, ret);
        }

        public static ndarray SearchSorted(this ndarray a, ndarray keys, NPY_SEARCHSIDE side = NPY_SEARCHSIDE.NPY_SEARCHLEFT)
        {
            return np.searchsorted(a, keys, side);
        }

        public static ndarray diagonal(this ndarray a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            return np.diagonal(a, offset, axis1, axis2);
        }

        public static ndarray trace(this ndarray a, int offset = 0, int axis1 = 0, int axis2 = 1, dtype dtype = null, ndarray @out = null)
        {
            return np.trace(a, offset, axis1, axis2, dtype, @out);
        }


        public static ndarray trace(this ndarray a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            return np.diagonal(a, offset, axis1, axis2);
        }


        public static ndarray[] NonZero(this ndarray a)
        {
            return np.nonzero(a);
        }

        public static ndarray compress(this ndarray a, object condition, object axis = null, ndarray @out = null)
        {
            return np.compress(condition, a, axis, @out);
        }

        public static ndarray compress(this ndarray a, ndarray condition, int? axis = null, ndarray @out = null)
        {
            return np.compress(condition, a, axis, @out);
        }

        public static ndarray clip(this ndarray a, object a_min, object a_max, ndarray ret = null)
        {
            return np.clip(a, a_min, a_max, ret);
        }

        public static ndarray Sum(this ndarray a, int axis = -1, dtype dtype = null, ndarray ret = null)
        {
            return np.sum(a, axis, dtype, ret);
        }

        public static ndarray Any(this ndarray a, object axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.any(a, axis, @out, keepdims);
        }

        public static ndarray All(this ndarray a, object axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.all(a, axis, @out, keepdims);
        }

        public static ndarray cumsum(this ndarray a, int? axis, dtype dtype = null, ndarray ret = null)
        {
            return np.cumsum(a, axis, dtype, ret);
        }

        public static ndarray ptp(this ndarray a, object axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.ptp(a, axis, @out, keepdims);
        }

        public static ndarray AMax(this ndarray a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.amax(a, axis, @out, keepdims);
        }

        public static ndarray AMin(this ndarray a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {
            return np.amin(a, axis, @out, keepdims);
        }


        public static ndarray Prod(this ndarray a, int? axis = null, dtype dtype = null, ndarray ret = null)
        {
            return np.prod(a, axis, dtype, ret);
        }

        public static ndarray CumProd(this ndarray a, int axis, dtype dtype, ndarray ret = null)
        {
            return np.cumprod(a, axis, dtype, ret);
        }
        
        public static ndarray Mean(this ndarray a, int? axis = null, dtype dtype = null)
        {
            return np.mean(a, axis, dtype);
        }

        public static ndarray Std(this ndarray a, int? axis = null, dtype dtype = null)
        {
            return np.std(a, axis, dtype);
        }


        public static List<T> ToList<T>(this ndarray a)
        {
            T[] data = (T[])a.rawdata(0).datap;
            return data.ToList();
        }

        public static T[] ToArray<T>(this ndarray a)
        {
            T[] data = (T[])a.rawdata(0).datap;
            return data;
        }

    }
}
