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
        internal static NpyArray NpyArray_GetField(NpyArray self, NpyArray_Descr descr, int offset)
        {

            NpyArray ret = null;

            if (offset < 0 || (offset + descr.elsize) > self.descr.elsize)
            {
                string msg = string.Format("Need 0 <= offset <= {0} for requested type but received offset = {1}",
                    self.descr.elsize - descr.elsize, offset);
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                Npy_DECREF(descr);
                return null;
            }
            ret = NpyArray_NewView(descr, self.nd, self.dimensions, self.strides,
                                   self, offset, false);
            if (ret == null)
            {
                return null;
            }
            return ret;
        }

        internal static int NpyArray_SetField(NpyArray self, NpyArray_Descr descr, int offset, NpyArray val)
        {
            NpyArray ret = null;
            int retval = 0;

            if (offset < 0 || (offset + descr.elsize) > self.descr.elsize)
            {
                string msg = string.Format("Need 0 <= offset <= {0} for requested type but received offset = {1}",
                                self.descr.elsize - descr.elsize, offset);

                NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                Npy_DECREF(descr);
                return -1;
            }
            ret = NpyArray_NewView(descr, self.nd, self.dimensions, self.strides,
                                   self, offset, false);
            if (ret == null)
            {
                return -1;
            }
            retval = NpyArray_MoveInto(ret, val);
            Npy_DECREF(ret);
            return retval;
        }

        internal static NpyArray NpyArray_Byteswap(NpyArray self, bool inplace)
        {
            NpyArray ret;
            npy_intp size;
            NpyArrayIterObject it;

            if (inplace)
            {
                if (!NpyArray_ISWRITEABLE(self))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_RuntimeError, "Cannot byte-swap in-place on a read-only array");
                    return null;
                }
                size = NpyArray_SIZE(self);
                if (NpyArray_ISONESEGMENT(self))
                {
                    _default_copyswap(new VoidPtr(self), self.descr.elsize,  null, -1, size, true, self);
                }
                else
                { /* Use iterator */
                    int axis = -1;
                    npy_intp stride;
                    it = NpyArray_IterAllButAxis(self, ref axis);
                    stride = self.strides[axis];
                    size = self.dimensions[axis];
                    while (it.index < it.size)
                    {
                        _default_copyswap(it.dataptr, stride, null, -1, size, true, self);
                        NpyArray_ITER_NEXT(it);
                    }
                    Npy_DECREF(it);
                }

                Npy_INCREF(self);
                return self;
            }
            else
            {
                NpyArray newArray;
                if ((ret = NpyArray_NewCopy(self, NPY_ORDER.NPY_ANYORDER)) == null)
                {
                    return null;
                }
                newArray = NpyArray_Byteswap(ret, true);
                Npy_DECREF(newArray);
                return ret;
            }
        }

    }
}
