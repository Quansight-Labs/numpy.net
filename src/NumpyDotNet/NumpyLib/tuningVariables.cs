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
using System.Text;
using System.Threading;
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
        private static npy_intp maxParallelIterators = 8;  // must be 1,2,4,8 or16
        private static npy_intp flatCopyParallelSize = 10000;
        internal static npy_intp maxIterOffsetCacheSize = 1000;

        private static npy_intp maxNumericOpParallelSize = 1000;
        private static npy_intp maxCopyFieldParallelSize = 1000;
        private static npy_intp maxSortOperationParallelSize = 1000;

        [ThreadStatic]
        internal static bool ?enableTryCatchOnCalculations = null;

        internal static bool getEnableTryCatchOnCalculations
        {
            get
            {
                if (numpyinternal.enableTryCatchOnCalculations.HasValue)
                    return numpyinternal.enableTryCatchOnCalculations.Value;
                numpyinternal.enableTryCatchOnCalculations = true;
                return numpyinternal.enableTryCatchOnCalculations.Value;
            }
        }

        internal static string GenerateTryCatchExceptionMessage(string ExMessage)
        {
            string Message = string.Format("This operation caused exception: {0}. Set np.tuning.EnableTryCatchOnCalculations = true to handle this exception cleanly.", ExMessage);
            return Message;
        }
    }
}
