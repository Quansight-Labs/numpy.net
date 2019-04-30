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
        /*
        * Typestr converter
        */
        internal static NPY_TYPES NpyArray_TypestrConvert(int itemsize, NPY_TYPECHAR gentype)
        {
#if false
            NPY_TYPES newtype = NPY_TYPES.NPY_VOID;

            if (gentype == NPY_TYPECHAR.NPY_GENBOOLLTR)
            {
                if (itemsize == 1)
                {
                    newtype = NPY_TYPES.NPY_BOOL;
                }
                else
                {
                    newtype = NPY_TYPES.NPY_NOTYPE;
                }
            }
            else if (gentype == NPY_TYPECHAR.NPY_SIGNEDLTR)
            {
                switch (itemsize)
                {
                    case 1:
                        newtype = NPY_TYPES.NPY_INT8;
                        break;
                    case 2:
                        newtype = NPY_TYPES.NPY_INT16;
                        break;
                    case 4:
                        newtype = NPY_TYPES.NPY_INT32;
                        break;
                    case 8:
                        newtype = NPY_TYPES.NPY_INT64;
                        break;

                    case 16:
                        newtype = NPY_TYPES.NPY_INT128;
                        break;

                    default:
                        newtype = NPY_TYPES.NPY_NOTYPE;
                        break;
                }
            }
            else if (gentype == NPY_TYPECHAR.NPY_UNSIGNEDLTR)
            {
                switch (itemsize)
                {
                    case 1:
                        newtype = NPY_TYPES.NPY_UINT8;
                        break;
                    case 2:
                        newtype = NPY_TYPES.NPY_UINT16;
                        break;
                    case 4:
                        newtype = NPY_TYPES.NPY_UINT32;
                        break;
                    case 8:
                        newtype = NPY_TYPES.NPY_UINT64;
                        break;

                    case 16:
                        newtype = NPY_TYPES.NPY_UINT128;
                        break;

                    default:
                        newtype = NPY_TYPES.NPY_NOTYPE;
                        break;
                }
            }
            else if (gentype == NPY_TYPECHAR.NPY_FLOATINGLTR)
            {
                switch (itemsize)
                {
                    case 4:
                        newtype = NPY_TYPES.NPY_FLOAT32;
                        break;
                    case 8:
                        newtype = NPY_TYPES.NPY_FLOAT64;
                        break;

                    case 10:
                        newtype = NPY_TYPES.NPY_FLOAT80;
                        break;


                    case 12:
                        newtype = NPY_TYPES.NPY_FLOAT96;
                        break;


                    case 16:
                        newtype = NPY_TYPES.NPY_FLOAT128;
                        break;

                    default:
                        newtype = NPY_TYPES.NPY_NOTYPE;
                        break;
                }
            }
            else if (gentype == NPY_TYPECHAR.NPY_COMPLEXLTR)
            {
                switch (itemsize)
                {
                    case 8:
                        newtype = NPY_TYPES.NPY_COMPLEX64;
                        break;
                    case 16:
                        newtype = NPY_TYPES.NPY_COMPLEX128;
                        break;

                    case 20:
                        newtype = NPY_TYPES.NPY_COMPLEX160;
                        break;

                    case 24:
                        newtype = NPY_TYPES.NPY_COMPLEX192;
                        break;

                    case 32:
                        newtype = NPY_TYPES.NPY_COMPLEX256;
                        break;

                    default:
                        newtype = NPY_TYPES.NPY_NOTYPE;
                        break;
                }
            }
            return newtype;
#endif
            return NPY_TYPES.NPY_VOID;
        }

        static string npy_casting_to_string(NPY_CASTING casting)
        {
            switch (casting)
            {
                case NPY_CASTING.NPY_NO_CASTING:
                    return "'no'";
                case NPY_CASTING.NPY_EQUIV_CASTING:
                    return "'equiv'";
                case NPY_CASTING.NPY_SAFE_CASTING:
                    return "'safe'";
                case NPY_CASTING.NPY_SAME_KIND_CASTING:
                    return "'same_kind'";
                case NPY_CASTING.NPY_UNSAFE_CASTING:
                    return "'unsafe'";
                default:
                    return "<unknown>";
            }
        }
    }
}
