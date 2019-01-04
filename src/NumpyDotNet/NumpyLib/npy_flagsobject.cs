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
        internal static bool _IsContiguous(NpyArray ap)
        {
            npy_intp sd;
            npy_intp dim;
            int i;

            if (ap.nd == 0)
            {
                return true;
            }
            sd = (npy_intp)ap.descr.elsize;
            if (ap.nd == 1)
            {
                return ap.dimensions[0] == 1 || sd == ap.strides[0];
            }
            for (i = ap.nd - 1; i >= 0; --i)
            {
                dim = ap.dimensions[i];
                /* contiguous by definition */
                if (dim == 0)
                {
                    return true;
                }
                if (ap.strides[i] != sd)
                {
                    return false;
                }
                sd *= dim;
            }
            return true;
        }

        /* 0-strided arrays are not contiguous (even if dimension == 1) */
        static bool _IsFortranContiguous(NpyArray ap)
        {
            npy_intp sd;
            npy_intp dim;
            int i;

            if (ap.nd == 0)
            {
                return true;
            }
            sd = (npy_intp)ap.descr.elsize;
            if (ap.nd == 1)
            {
                return ap.dimensions[0] == 1 || sd == ap.strides[0];
            }
            for (i = 0; i < ap.nd; ++i)
            {
                dim = ap.dimensions[i];
                /* fortran contiguous by definition */
                if (dim == 0)
                {
                    return true;
                }
                if (ap.strides[i] != sd)
                {
                    return false;
                }
                sd *= dim;
            }
            return true;
        }


        /*
         * Update Several Flags at once.
         */
        internal static void NpyArray_UpdateFlags(NpyArray ret, NPYARRAYFLAGS flagmask)
        {
            if ((flagmask & NPYARRAYFLAGS.NPY_FORTRAN) > 0)
            {
                if (_IsFortranContiguous(ret))
                {
                    ret.flags |= NPYARRAYFLAGS.NPY_FORTRAN;
                    if (ret.nd > 1)
                    {
                        ret.flags &= ~NPYARRAYFLAGS.NPY_CONTIGUOUS;
                    }
                }
                else
                {
                    ret.flags &= ~NPYARRAYFLAGS.NPY_FORTRAN;
                }
            }
            if ((flagmask & NPYARRAYFLAGS.NPY_CONTIGUOUS) > 0)
            {
                if (_IsContiguous(ret))
                {
                    ret.flags |= NPYARRAYFLAGS.NPY_CONTIGUOUS;
                    if (ret.nd > 1)
                    {
                        ret.flags &= ~NPYARRAYFLAGS.NPY_FORTRAN;
                    }
                }
                else
                {
                    ret.flags &= ~NPYARRAYFLAGS.NPY_CONTIGUOUS;
                }
            }
            if ((flagmask & NPYARRAYFLAGS.NPY_ALIGNED) > 0)
            {
                if (Npy_IsAligned(ret))
                {
                    ret.flags |= NPYARRAYFLAGS.NPY_ALIGNED;
                }
                else
                {
                    ret.flags &= ~NPYARRAYFLAGS.NPY_ALIGNED;
                }
            }
            /*
             * This is not checked by default WRITEABLE is not
             * part of UPDATE_ALL
             */
            if ((flagmask & NPYARRAYFLAGS.NPY_WRITEABLE)  > 0)
            {
                if (Npy_IsWriteable(ret))
                {
                    ret.flags |= NPYARRAYFLAGS.NPY_WRITEABLE;
                }
                else
                {
                    ret.flags &= ~NPYARRAYFLAGS.NPY_WRITEABLE;
                }
            }
            return;
        }

    }
}
