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
        internal static void NpyArray_Item_INCREF(byte[] data, NpyArray_Descr descr)
        {
            // todo: not sure I need to do anything.  Am I going to ref count the buffers?
        }
        internal static void NpyArray_Item_INCREF(VoidPtr data, NpyArray_Descr descr)
        {
            // todo: not sure I need to do anything.  Am I going to ref count the buffers?
        }

        internal static void NpyArray_Item_XDECREF(byte[] data, NpyArray_Descr descr)
        {
            // todo: not sure I need to do anything.  Am I going to ref count the buffers?
        }
   
        internal static void NpyArray_Item_XDECREF(VoidPtr data, NpyArray_Descr descr)
        {
            // todo: not sure I need to do anything.  Am I going to ref count the buffers?
        }

        internal static int NpyArray_INCREF(NpyArray mp)
        {
            // todo: not sure I need to do anything.  Am I going to ref count the buffers?
            return 0;
        }

        internal static int NpyArray_XDECREF(NpyArray mp)
        {
            // todo: not sure I need to do anything.  Am I going to ref count the buffers?
            return 0;
        }
    }
}
