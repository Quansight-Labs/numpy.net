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
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyLib
{
    internal partial class numpyinternal
    {
  
    
        internal static NpyArray NpyArray_GenericUnaryFunction(GenericReductionOp operation, NpyArray inputArray, NpyUFuncObject op, NpyArray retArray)
        {
            NpyArray[] mps = new NpyArray[npy_defs.NPY_MAXARGS];
            NpyArray result;

            if (retArray == null)
                retArray = inputArray;

            Debug.Assert(Validate(op));
            Debug.Assert(Validate(inputArray));
            Debug.Assert(Validate(retArray));

            mps[0] = inputArray;
            Npy_XINCREF(retArray);
            mps[1] = retArray;
            if (0 > NpyUFunc_GenericFunction(operation, op, 2, mps, 0, null, false, null, null))
            {
                result = null;
                goto finish;
            }

            if (retArray != null)
            {
                result = retArray;
            }
            else
            {
                result = mps[op.nin];
            }
            finish:
            Npy_XINCREF(result);
            if (mps[1] != null && (mps[1].flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) != 0)
            {
                NpyArray_ForceUpdate(mps[1]);
            }
            Npy_XDECREF(mps[1]);
            return result;
        }

        internal static int NpyArray_Bool(NpyArray mp)
        {
            npy_intp n;

            n = NpyArray_SIZE(mp);
            if (n == 1)
            {
                var vp = NpyArray_BYTES(mp);
                return NpyArray_DESCR(mp).f.nonzero(vp, vp.data_offset >> mp.ItemDiv) ? 1 : 0;
            }
            else if (n == 0)
            {
                return 0;
            }
            else
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                "The truth value of an array with more than one element is ambiguous. Use np.any() or np.all()");
                return -1;
            }
        }

 
 


    }
}
