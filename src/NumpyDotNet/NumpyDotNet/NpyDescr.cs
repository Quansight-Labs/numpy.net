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
using System.Runtime.CompilerServices;
using System.Text;
using System.Numerics;
using NumpyLib;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet {
    /// <summary>
    /// Implements the descriptor (NpyArray_Descr) functionality.  This is not the
    /// public wrapper but a collection of funtionality to support the dtype class.
    /// </summary>
    public class NpyDescr {


        /// <summary>
        /// Checks to see if a given string matches any of date/time types.
        /// </summary>
        /// <param name="s">Type string</param>
        /// <returns>True if it's a date/time format, false if not</returns>
        private static bool CheckForDatetime(String s) {
            if (s.Length < 2) return false;
            if (s[1] == '8' && (s[0] == 'M' || s[0] == 'm')) return true;
            return s.StartsWith("datetime64") || s.StartsWith("timedelta64");
        }



        /// <summary>
        /// Comma strings are ones that start with an integer, are empty tuples,
        /// or contain commas.
        /// </summary>
        /// <param name="s">Datetime format string</param>
        /// <returns>True if a comma string</returns>
        private static bool CheckForCommaString(String s) {
            Func<char, bool> checkByteOrder =
                b => b == '>' || b == '<' || b == '|' || b == '=';

            // Check for ints at the start of a string.
            if (s[0] >= '0' && s[0] <= '9' ||
                s.Length > 1 && checkByteOrder(s[0]) && s[1] >= '0' && s[1] <= '9')
                return true;

            // Empty tuples
            if (s.Length > 1 && s[0] == '(' && s[1] == ')' ||
                s.Length > 3 && checkByteOrder(s[0]) && s[1] == '(' && s[2] == ')')
                return true;

            // Any commas in the string?
            return s.Contains(',');
        }

    }
}
