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
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
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
        class MathHelper
        {
            public ndarray a;
            public long[] offsets = null;
            public double[] dd = null;
            public double[] s = null;

            public MathHelper(object x)
            {
                a = asanyarray(x);
                if (a.Dtype.TypeNum != NPY_TYPES.NPY_DOUBLE)
                {
                    a = a.astype(dtype: np.Float64);
                }

                offsets = NpyCoreApi.GetViewOffsets(a);
                dd = a.Array.data.datap as double[];
                s = new double[offsets.Length];
            }
        }

        #region Trigonometric Functions

        public static ndarray sin(object x, object where = null)
        {
            MathHelper ch = new MathHelper(x);
            
            for (int i = 0; i < ch.offsets.Length; i++)
            {
                ch.s[i] = Math.Sin(ch.dd[ch.offsets[i]]);
            }

            var ret = np.array(ch.s).reshape(new shape(ch.a.dims));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray cos(object x, object where = null)
        {
            MathHelper ch = new MathHelper(x);

            for (int i = 0; i < ch.offsets.Length; i++)
            {
                ch.s[i] = Math.Cos(ch.dd[ch.offsets[i]]);
            }

            var ret = np.array(ch.s).reshape(new shape(ch.a.dims));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray tan(object x, object where = null)
        {
            MathHelper ch = new MathHelper(x);

            for (int i = 0; i < ch.offsets.Length; i++)
            {
                ch.s[i] = Math.Tan(ch.dd[ch.offsets[i]]);
            }

            var ret = np.array(ch.s).reshape(new shape(ch.a.dims));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray arcsin(object x, object where = null)
        {
            MathHelper ch = new MathHelper(x);

            for (int i = 0; i < ch.offsets.Length; i++)
            {
                ch.s[i] = Math.Asin(ch.dd[ch.offsets[i]]);
            }

            var ret = np.array(ch.s).reshape(new shape(ch.a.dims));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray arccos(object x, object where = null)
        {
            MathHelper ch = new MathHelper(x);

            for (int i = 0; i < ch.offsets.Length; i++)
            {
                ch.s[i] = Math.Acos(ch.dd[ch.offsets[i]]);
            }

            var ret = np.array(ch.s).reshape(new shape(ch.a.dims));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        public static ndarray arctan(object x, object where = null)
        {
            MathHelper ch = new MathHelper(x);

            for (int i = 0; i < ch.offsets.Length; i++)
            {
                ch.s[i] = Math.Atan(ch.dd[ch.offsets[i]]);
            }

            var ret = np.array(ch.s).reshape(new shape(ch.a.dims));
            if (where != null)
            {
                ret[np.invert(where)] = np.NaN;
            }

            return ret;
        }

        #endregion

    }
}
