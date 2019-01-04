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
using System.Diagnostics.Contracts;
using System.Linq;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Reflection;

namespace NumpyDotNet {
    /// <summary>
    /// ModuleMethods implements the module-level numpy functions.
    /// </summary>
    public static class ScalarMathModule
    {

        private static object GenericOper<T>(Func<T, ScalarGeneric> newScalar, Func<T, T, T> f, object aObj, object bObj) {
            T a, b;
            switch (Convert2ToNative<T>(aObj, bObj, out a, out b)) {
                case 0:
                    break;
                case -1:
                    // One can't be cast safely, must be mixed-type.
                    throw new NotImplementedException();
                case -2:
                    throw new NotImplementedException();
                case -3:
                    throw new NotImplementedException();
                default:
                    Contract.Assert(false, "Unhandled return value from Convert2ToNative");
                    return null;
            }
            return newScalar(f(a, b));
        }

        private static int Convert2ToNative<TReturn>(object a, object b, out TReturn aResult, out TReturn bResult) {
            // TODO: This is wrong, needs to be really implemented.
            try {
                aResult = (TReturn)a;
                bResult = (TReturn)b;
            } catch {
                aResult = default(TReturn);
                bResult = default(TReturn);
                return -1;
            }
            return 0;
        }
    }
}
