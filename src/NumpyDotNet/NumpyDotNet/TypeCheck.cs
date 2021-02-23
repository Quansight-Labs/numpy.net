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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Numerics;
using NumpyLib;

#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{
    public static partial class np
    {

        //Replace NaN with zero and infinity with large finite numbers.

        //If `x` is inexact, NaN is replaced by zero, and infinity and -infinity
        //replaced by the respectively largest and most negative finite floating
        //point values representable by ``x.dtype``.

        //For complex dtypes, the above is applied to each of the real and
        //imaginary components of `x` separately.

        //If `x` is not inexact, then no replacements are made.

        //Parameters
        //----------
        //x : scalar or array_like
        //    Input data.
        //copy : bool, optional
        //    Whether to create a copy of `x` (True) or to replace values
        //    in-place (False). The in-place operation only occurs if
        //    casting to an array does not require a copy.
        //    Default is True.

        //    ..versionadded:: 1.13

        //Returns
        //-------
        //out : ndarray
        //    `x`, with the non-finite values replaced.If `copy` is False, this may
        //    be `x` itself.

        //See Also
        //--------
        //isinf : Shows which elements are positive or negative infinity.
        //isneginf : Shows which elements are negative infinity.
        //isposinf : Shows which elements are positive infinity.
        //isnan : Shows which elements are Not a Number (NaN).
        //isfinite : Shows which elements are finite (not NaN, not infinity)

        //Notes
        //-----
        //NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
        //(IEEE 754). This means that Not a Number is not equivalent to infinity.

        //Examples
        //--------
        //>>> np.nan_to_num(np.inf)
        //1.7976931348623157e+308
        //>>> np.nan_to_num(-np.inf)
        //-1.7976931348623157e+308
        //>>> np.nan_to_num(np.nan)
        //0.0
        //>>> x = np.array([np.inf, -np.inf, np.nan, -128, 128])
        //>>> np.nan_to_num(x)
        //array([  1.79769313e+308,  -1.79769313e+308,   0.00000000e+000,
        //        -1.28000000e+002,   1.28000000e+002])
        //>>> y = np.array([complex(np.inf, np.nan), np.nan, complex(np.nan, np.inf)])
        //>>> np.nan_to_num(y)
        //array([  1.79769313e+308 +0.00000000e+000j,
        //         0.00000000e+000 +0.00000000e+000j,
        //         0.00000000e+000 +1.79769313e+308j])

        public static object nan_to_num(object x, bool copy=true)
        {
            ndarray xa = asanyarray(x);
            ndarray xn = nan_to_num(xa, copy);
            return xn.GetItem(0);
        }

        public static ndarray nan_to_num(ndarray x, bool copy = true)
        {
            x = array(x, subok: true, copy: copy);
            var xtype = x.Dtype.Descr.type_num;

            bool isscalar = (x.ndim == 0);

            // if not a floating point type number, just return.
            if (!x.IsInexact)
            {
                return x;
            }

            // decimal numbers don't have NAN or Infinity values
            if (x.IsDecimal)
            {
                return x;
            }

            if (x.IsComplex)
            {
                return x;
            }
            else
            {
                ndarray dest = x;
                object maxf = _getmax(dest);
                object minf = _getmin(dest);

                copyto(dest, 0, where: isnan(dest));
                copyto(dest, maxf, where: isposinf(dest));
                copyto(dest, minf, where: isneginf(dest));

                return dest;
            }

        }

        private static object _getmin(ndarray dest)
        {
            return DefaultArrayHandlers.GetArrayHandler(dest.TypeNum).GetArgSortMinValue();
        }

        private static object _getmax(ndarray dest)
        {
            return DefaultArrayHandlers.GetArrayHandler(dest.TypeNum).GetArgSortMaxValue();
        }
    }
}
