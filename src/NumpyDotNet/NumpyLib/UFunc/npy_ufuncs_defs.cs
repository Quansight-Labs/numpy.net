﻿/*
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
        internal const int UFUNC_PARALLEL_DEST_MINSIZE = 1000;
        internal const int UFUNC_PARALLEL_DEST_ASIZE = 100;

        internal delegate void UFuncGeneralReductionHandler(VoidPtr[] bufPtr, npy_intp[] steps, UFuncOperation ops, npy_intp N);
        internal delegate void UFuncGeneralReductionHandler_XXX(UFuncOperation op, NpyUFuncReduceObject loop);
        internal delegate void UFuncGeneralAccumulateHandler_XXX(GenericReductionOp op, NpyUFuncReduceObject loop, UFuncOperation ufop);

        internal interface IUFUNC_Operations
        {
            void PerformOuterOpArrayIter(NpyArray a, NpyArray b, NpyArray destArray, NumericOperations operations, UFuncOperation op);
            void PerformReduceOpArrayIter(VoidPtr[] bufPtr, npy_intp[] steps, UFuncOperation ops, npy_intp N);
            void PerformAccumulateOpArrayIter(VoidPtr[] bufPtr, npy_intp[] steps, UFuncOperation ops, npy_intp N);
            void PerformReduceAtOpArrayIter(VoidPtr[] bufPtr, npy_intp[] steps, UFuncOperation ops, npy_intp N);
            void PerformScalarOpArrayIter(NpyArray destArray, NpyArray srcArray, NpyArray operArray, UFuncOperation op);

            void PerformReduceOpArrayIter_XXX(UFuncOperation op, NpyUFuncReduceObject loop);
            void PerformAccumulateOpArrayIter_XXX(GenericReductionOp op, NpyUFuncReduceObject loop, UFuncOperation ufop);


        }


    }
}
