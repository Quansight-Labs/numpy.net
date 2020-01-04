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
        internal static int NpyArray_SetShape(NpyArray self, NpyArray_Dims newdims)
        {
            int nd;
            NpyArray ret;

            ret = NpyArray_Newshape(self, newdims, NPY_ORDER.NPY_CORDER);
            if (ret == null)
            {
                return -1;
            }
            if (NpyArray_DATA(ret).datap != NpyArray_DATA(self).datap)
            {
                Npy_XDECREF(ret);
                NpyErr_SetString(npyexc_type.NpyExc_AttributeError, "incompatible shape for a non-contiguous array");
                return -1;
            }

            /* Free old dimensions and strides */
            NpyDimMem_FREE(NpyArray_DIMS(self));
            nd = NpyArray_NDIM(ret);
            NpyArray_NDIM_Update(self,nd);
            if (nd > 0)
            {
                /* create new dimensions and strides */
                NpyArray_DIMS_Update(self,NpyDimMem_NEW(nd));
                if (NpyArray_DIMS(self) == null)
                {
                    Npy_XDECREF(ret);
                    NpyErr_MEMORY();
                    return -1;
                }
                NpyArray_STRIDES_Update(self, NpyDimMem_NEW(nd));
                memcpy(NpyArray_DIMS(self), NpyArray_DIMS(ret), nd * sizeof(npy_intp));
                memcpy(NpyArray_STRIDES(self), NpyArray_STRIDES(ret), nd * sizeof(npy_intp));
            }
            else
            {
                NpyArray_DIMS_Update(self, null);
                NpyArray_STRIDES_Update(self, null);
            }
            Npy_XDECREF(ret);
            NpyArray_UpdateFlags(self, NPYARRAYFLAGS.NPY_CONTIGUOUS | NPYARRAYFLAGS.NPY_FORTRAN);
            return 0;
        }

        internal static int NpyArray_SetStrides(NpyArray self, NpyArray_Dims newstrides)
        {
            NpyArray  newArray;
            npy_intp numbytes = 0, offset = 0;

            if (newstrides.len != NpyArray_NDIM(self))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "strides must be same length as shape");
                return -1;
            }
            newArray = NpyArray_BASE_ARRAY(self);
            while (null != NpyArray_BASE_ARRAY(newArray))
            {
                newArray = NpyArray_BASE_ARRAY(newArray);
            }

#if false
    /* TODO: Fix this so we can set strides on a buffer-backed array. */
    /* Get the available memory through the buffer interface on
     * new.base or if that fails from the current new
     * NOTE: PyObject_AsReadBuffer is never called during tests */
    if (newArray.base_obj != null && PyObject_AsReadBuffer(newArray.base_obj,
                                                       (const void **)&buf,
                                                       &buf_len) >= 0) {
        offset = NpyArray_BYTES(self) - buf;
        numbytes = buf_len - offset;
    }
#else
            if (newArray.base_obj != null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "strides cannot be set on array created from a buffer.");
                return -1;
            }
#endif
            else
            {
                NpyErr_Clear();

                numbytes = NpyArray_MultiplyList(NpyArray_DIMS(newArray), NpyArray_NDIM(newArray));
                numbytes = (npy_intp)(numbytes * NpyArray_ITEMSIZE(newArray));

                // todo: Kevin - this calculation may not be correct
                offset = (npy_intp)(NpyArray_BYTES_Length(self) - NpyArray_BYTES_Length(newArray));
            }

            if (!NpyArray_CheckStrides(NpyArray_ITEMSIZE(self),
                                       NpyArray_NDIM(self), numbytes,
                                       offset,
                                       NpyArray_DIMS(self), newstrides.ptr))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                 "strides is not compatible with available memory");
                return -1;
            }
            memcpy(NpyArray_STRIDES(self), newstrides.ptr, sizeof(npy_intp) * newstrides.len);
            NpyArray_UpdateFlags(self, NPYARRAYFLAGS.NPY_CONTIGUOUS | NPYARRAYFLAGS.NPY_FORTRAN);
            return 0;
        }

        internal static NpyArray NpyArray_GetReal(NpyArray self)
        {
            if (NpyArray_ISCOMPLEX(self))
            {
                return _get_part(self, false);
            }
            else
            {
                Npy_INCREF(self);
                return self;
            }
        }

        internal static NpyArray NpyArray_GetImag(NpyArray self)
        {
            if (NpyArray_ISCOMPLEX(self))
            {
                return _get_part(self, true);
            }
            else
            {
                Npy_INCREF(self);
                return self;
            }
        }

        static NpyArray _get_part(NpyArray self, bool imag)
        {
            return self;

            // .NET complex variables do not allow the ability to map into a view like python
            // Even if we could figure out a way, the real/imaginary fields are not changable directly.
            // The only way to access or modify them is a complete System.Numeric.Complex


            //NpyArray_Descr type;
            //int offset;


            //type = NpyArray_DescrFromType(NPY_TYPES.NPY_DOUBLE);

            //if (imag)
            //    offset = sizeof(double);
            //else
            //    offset = 0;


            //if (!NpyArray_ISNBO(self))
            //{
            //    NpyArray_Descr nw;
            //    nw = NpyArray_DescrNew(type);
            //    nw.byteorder = self.descr.byteorder;
            //    Npy_DECREF(type);
            //    type = nw;
            //}
            //return NpyArray_NewView(type,
            //                        self.nd,
            //                        self.dimensions,
            //                        self.strides,
            //                        self, (npy_intp)offset, false);
        }
    }

 
}
