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
            return ret;
        }

    }
}
