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
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Text;
using NumpyLib;
using npy_intp = System.Int32;


namespace NumpyDotNet
{
    public static partial class np
    {

        private static (ndarray a, ndarray mask) _replace_nan(ndarray a, float val)
        {
            /*
             If `a` is of inexact type, make a copy of `a`, replace NaNs with
             the `val` value, and return the copy together with a boolean mask
             marking the locations where NaNs were present. If `a` is not of
             inexact type, do nothing and return `a` together with a mask of None.

             Note that scalars will end up as array scalars, which is important
             for using the result as the value of the out argument in some
             operations.

             Parameters
             ----------
             a : array-like
                 Input array.
             val : float
                 NaN values are set to val before doing the operation.

             Returns
             -------
             y : ndarray
                 If `a` is of inexact type, return a copy of `a` with the NaNs
                 replaced by the fill value, otherwise return `a`.
             mask: {bool, None}
                 If `a` is of inexact type, return a boolean mask marking locations of
                 NaNs, otherwise return None.
             */

            a = np.array(a, subok: true, copy: true);

            ndarray mask;

            if (a.Dtype.TypeNum == NPY_TYPES.NPY_OBJECT)
            {
                // object arrays do not support `isnan` (gh-9009), so make a guess
                mask = a.NotEquals(a);
            }
            else if (a.IsFloatingPoint)
            {
                mask = np.isnan(a);
            }
            else
            {
                mask = null;
            }

            if (mask != null)
            {
                a[mask] = val;
                np.copyto(a, val, where: mask);
            }

            return (a: a, mask: mask);
        }
    }
}
