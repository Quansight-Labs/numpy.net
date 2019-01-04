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
using NumpyLib;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif


namespace NumpyDotNet
{
    public class shape
    {
        public npy_intp[] iDims = null;

        public shape(IEnumerable<npy_intp> dim)
        {
            int nd = dim.Count();
            iDims = new npy_intp[nd];

            int i = 0;
            foreach (var d in dim)
            {
                iDims[i] = d;
                i++;
            }
        }

        public shape(npy_intp[] dim, int nd)
        {
            iDims = new npy_intp[nd];
            for (int i = 0; i < nd; i++)
            {
                iDims[i] = dim[i];
            }
        }

        public shape(shape s)
        {
            iDims = new npy_intp[s.iDims.Length];
            Array.Copy(s.iDims, iDims, s.iDims.Length);
        }

        public shape(int Dim1)
        {
            iDims = new npy_intp[1];
            iDims[0] = Dim1;
        }
        public shape(int Dim1, int Dim2)
        {
            iDims = new npy_intp[2];
            iDims[0] = Dim1;
            iDims[1] = Dim2;
        }
        public shape(int Dim1, int Dim2, int Dim3)
        {
            iDims = new npy_intp[3];
            iDims[0] = Dim1;
            iDims[1] = Dim2;
            iDims[2] = Dim3;
        }
        public shape(int Dim1, int Dim2, int Dim3, int Dim4)
        {
            iDims = new npy_intp[4];
            iDims[0] = Dim1;
            iDims[1] = Dim2;
            iDims[2] = Dim3;
            iDims[3] = Dim4;
        }

        public override string ToString()
        {
            StringBuilder buf = new StringBuilder();
            buf.Append("(");
            for (int i = 0; i < iDims.Length; i++)
            {
                if (i > 0) buf.Append(", ");
                buf.Append(iDims[i].ToString());
            }
            if (iDims.Length == 1) buf.Append(",");
            buf.Append(")");
            return buf.ToString();
        }

    }


    public enum order
    {
        C,
        F,
        A,
        K,
        DEFAULT = C,
    }

 
}
