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

#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{
    interface IArray
    {
        ndarray byteswap(bool inplace = false);
        object dtype { get; }
        void fill(object scalar);
        flagsobj flags { get; }
        object flat { get; set; }
        ndarray flatten(NumpyLib.NPY_ORDER order = NumpyLib.NPY_ORDER.NPY_CORDER);
        object item_byindex(Int32[] args);
        object item_byindex(Int64[] args);
        object item(params object[] args);
        void itemset_byindex(Int32[] args, object value);
        void itemset_byindex(Int64[] args, object value);
        void itemset(params object[] args);
        int ndim { get; }
        ndarray newbyteorder(string endian = null);
        ndarray ravel(NumpyLib.NPY_ORDER order = NumpyLib.NPY_ORDER.NPY_CORDER);
        void setflags(object write = null, object align = null, object uic = null);
        shape shape { get; }
        npy_intp size { get; }
        npy_intp[] strides { get; }
        object this[params object[] args] { get; set; }
        object this[int index] { get; }
        object this[long index] { get; }
        object this[System.Numerics.BigInteger index] { get; }
        byte[] tobytes(NumpyLib.NPY_ORDER order = NumpyLib.NPY_ORDER.NPY_ANYORDER);
    }
}
