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

namespace NumpyDotNet {


    /// <summary>
    /// Records the type-specific get/set items for each descriptor type.
    /// </summary>
    public class ArrFuncs {
        internal Object GetItem(long offset, NpyArray arr) {
            return GetFunc(arr.data + offset, arr);
        }

        internal void SetItem(Object value, long offset, NpyArray arr) {
            SetFunc(value, arr.data + offset, arr);
        }

        internal Func<VoidPtr, NpyArray, Object> GetFunc { get; set; }
        internal Action<Object, VoidPtr, NpyArray> SetFunc { get; set; }
    }



    /// <summary>
    /// Collection of getitem/setitem functions and operations on object types.
    /// These are mostly used as callbacks from the core and operate on native
    /// memory.
    /// </summary>
    internal static class NumericOps {
        private static ArrFuncs[] ArrFuncs = null;
        private static Object ArrFuncsSyncRoot = new Object();

        /// <summary>
        /// Returns the array of functions appropriate to a given type.  The actual
        /// functions in the array will vary with the type sizes in the native code.
        /// </summary>
        /// <param name="t">Native array type</param>
        /// <returns>Functions matching that type</returns>
        internal static ArrFuncs FuncsForType(NPY_TYPES t)
        {
            if (ArrFuncs == null)
            {
                InitArrFuncs();
            }
            return ArrFuncs[(int)t];
        }


        /// <summary>
        /// Initializes the type-specific functions for each native type.
        /// </summary>
        internal static void InitArrFuncs()
        {

            lock (ArrFuncsSyncRoot)
            {
                if (ArrFuncs == null)
                {
                    ArrFuncs[] arr = new ArrFuncs[(int)NPY_TYPES.NPY_NTYPES];
                    // todo: see all the crazy code from the original numpydotnet

                    ArrFuncs = arr;
                }
            }
        }

  
    }
}
