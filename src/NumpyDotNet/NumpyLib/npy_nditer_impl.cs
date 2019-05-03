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
        /*** Global flags that may be passed to the iterator constructors ***/
        internal enum ITERFLAGS : UInt32
        {
            /* Track an index representing C order */
            NPY_ITER_C_INDEX = 0x00000001,
            /* Track an index representing Fortran order */
            NPY_ITER_F_INDEX = 0x00000002,
            /* Track a multi-index */
            NPY_ITER_MULTI_INDEX = 0x00000004,
            /* User code external to the iterator does the 1-dimensional innermost loop */
            NPY_ITER_EXTERNAL_LOOP = 0x00000008,
            /* Convert all the operands to a common data type */
            NPY_ITER_COMMON_DTYPE = 0x00000010,
            /* Operands may hold references, requiring API access during iteration */
            NPY_ITER_REFS_OK = 0x00000020,
            /* Zero-sized operands should be permitted, iteration checks IterSize for 0 */
            NPY_ITER_ZEROSIZE_OK = 0x00000040,
            /* Permits reductions (size-0 stride with dimension size > 1) */
            NPY_ITER_REDUCE_OK = 0x00000080,
            /* Enables sub-range iteration */
            NPY_ITER_RANGED = 0x00000100,
            /* Enables buffering */
            NPY_ITER_BUFFERED = 0x00000200,
            /* When buffering is enabled, grows the inner loop if possible */
            NPY_ITER_GROWINNER = 0x00000400,
            /* Delay allocation of buffers until first Reset* call */
            NPY_ITER_DELAY_BUFALLOC = 0x00000800,
            /* When NPY_KEEPORDER is specified, disable reversing negative-stride axes */
            NPY_ITER_DONT_NEGATE_STRIDES = 0x00001000,
            /*
             * If output operands overlap with other operands (based on heuristics that
             * has false positives but no false negatives), make temporary copies to
             * eliminate overlap.
             */
            NPY_ITER_COPY_IF_OVERLAP = 0x00002000,

            /*** Per-operand flags that may be passed to the iterator constructors ***/

            /* The operand will be read from and written to */
            NPY_ITER_READWRITE = 0x00010000,
            /* The operand will only be read from */
            NPY_ITER_READONLY = 0x00020000,
            /* The operand will only be written to */
            NPY_ITER_WRITEONLY = 0x00040000,
            /* The operand's data must be in native byte order */
            NPY_ITER_NBO = 0x00080000,
            /* The operand's data must be aligned */
            NPY_ITER_ALIGNED = 0x00100000,
            /* The operand's data must be contiguous (within the inner loop) */
            NPY_ITER_CONTIG = 0x00200000,
            /* The operand may be copied to satisfy requirements */
            NPY_ITER_COPY = 0x00400000,
            /* The operand may be copied with WRITEBACKIFCOPY to satisfy requirements */
            NPY_ITER_UPDATEIFCOPY = 0x00800000,
            /* Allocate the operand if it is NULL */
            NPY_ITER_ALLOCATE = 0x01000000,
            /* If an operand is allocated, don't use any subtype */
            NPY_ITER_NO_SUBTYPE = 0x02000000,
            /* This is a virtual array slot, operand is NULL but temporary data is there */
            NPY_ITER_VIRTUAL = 0x04000000,
            /* Require that the dimension match the iterator dimensions exactly */
            NPY_ITER_NO_BROADCAST = 0x08000000,
            /* A mask is being used on this array, affects buffer -> array copy */
            NPY_ITER_WRITEMASKED = 0x10000000,
            /* This array is the mask for all WRITEMASKED operands */
            NPY_ITER_ARRAYMASK = 0x20000000,
            /* Assume iterator order data access for COPY_IF_OVERLAP */
            NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE =  0x40000000,

            NPY_ITER_GLOBAL_FLAGS = 0x0000ffff,
            NPY_ITER_PER_OP_FLAGS = 0xffff0000,

        }

        internal class NpyIter
        {
            /* Initial fixed position data */
            public ITERFLAGS itflags;
            public byte ndim, nop;
            public sbyte maskop;
            public npy_intp itersize, iterstart, iterend;
            /* iterindex is only used if RANGED or BUFFERED is set */
            public npy_intp iterindex;
            /* The rest is variable */
            public char iter_flexdata;
        }

        private static ITERFLAGS NIT_ITFLAGS(NpyIter iter)
        {
            return iter.itflags;
        }

        private static byte NIT_NOP(NpyIter iter)
        {
            return iter.nop;
        }

        internal delegate bool NpyIter_IterNextFunc(NpyIter iter);

        /*NUMPY_API
        * Allocate a new iterator for one array object.
        */
        internal static NpyIter NpyIter_New(NpyArray op, ITERFLAGS flags,  NPY_ORDER order, NPY_CASTING casting, NpyArray_Descr dtype)
        {
            /* Split the flags into separate global and op flags */
            ITERFLAGS op_flags = flags & ITERFLAGS.NPY_ITER_PER_OP_FLAGS;
            flags &= ITERFLAGS.NPY_ITER_GLOBAL_FLAGS;

            return NpyIter_AdvancedNew(1, &op, flags, order, casting,
                                    &op_flags, &dtype,
                                    -1, null, null, 0);
        }


        /*NUMPY_API
        * Deallocate an iterator
        */
        internal static int NpyIter_Deallocate(NpyIter iter)
        {
            //ITERFLAGS itflags;
            ///*int ndim = NIT_NDIM(iter);*/
            //int iop, nop;
            //NpyArray_Descr[] dtype;
            //NpyArray[] _object;

            //if (iter == null)
            //{
            //    return npy_defs.NPY_SUCCEED;
            //}

            //itflags = NIT_ITFLAGS(iter);
            //nop = NIT_NOP(iter);
            //dtype = NIT_DTYPES(iter);
            //_object = NIT_OPERANDS(iter);

            ///* Deallocate any buffers and buffering data */
            //if (itflags & ITERFLAGS.NPY_ITFLAG_BUFFER)
            //{
            //    NpyIter_BufferData* bufferdata = NIT_BUFFERDATA(iter);
            //    VoidPtr[] buffers;
            //    NpyAuxData[] transferdata;

            //    /* buffers */
            //    buffers = NBF_BUFFERS(bufferdata);
            //    for (iop = 0; iop < nop; ++iop)
            //    {
            //        NpyArray_free(buffers[iop]);
            //    }
            //    /* read bufferdata */
            //    transferdata = NBF_READTRANSFERDATA(bufferdata);
            //    for (iop = 0; iop < nop; ++iop)
            //    {
            //        if (transferdata[iop]!= null)
            //        {
            //            NPY_AUXDATA_FREE(transferdata[iop]);
            //        }
            //    }
            //    /* write bufferdata */
            //    transferdata = NBF_WRITETRANSFERDATA(bufferdata);
            //    for (iop = 0; iop < nop; ++iop)
            //    {
            //        if (transferdata[iop] != null)
            //        {
            //            NPY_AUXDATA_FREE(transferdata[iop]);
            //        }
            //    }
            //}

            ///* Deallocate all the dtypes and objects that were iterated */
            //for (iop = 0; iop < nop; ++iop)
            //{
            //    Npy_XDECREF(dtype[iop]);
            //    Npy_XDECREF(_object[iop]);
            //}

            ///* Deallocate the iterator memory */
            //NpyObject_Free(iter);

            return npy_defs.NPY_SUCCEED;
        }

   

        internal static NpyIter_IterNextFunc NpyIter_GetIterNext(NpyIter dst_iter, object p)
        {
            throw new NotImplementedException();
        }


        private static VoidPtr[] NpyIter_GetDataPtrArray(NpyIter iter)
        {
            ITERFLAGS itflags = NIT_ITFLAGS(iter);
            /*int ndim = NIT_NDIM(iter);*/
            int nop = NIT_NOP(iter);

            if (itflags & ITERFLAGS.NPY_ITFLAG_BUFFER)
            {
                NpyIter_BufferData* bufferdata = NIT_BUFFERDATA(iter);
                return NBF_PTRS(bufferdata);
            }
            else
            {
                NpyIter_AxisData* axisdata = NIT_AXISDATA(iter);
                return NAD_PTRS(axisdata);
            }
        }

        private static npy_intp[] NpyIter_GetInnerStrideArray(NpyIter dst_iter)
        {
            throw new NotImplementedException();
        }


        private static VoidPtr NpyIter_GetInnerLoopSizePtr(NpyIter dst_iter)
        {
            throw new NotImplementedException();
        }

        private static bool NpyIter_IterationNeedsAPI(NpyIter dst_iter)
        {
            throw new NotImplementedException();
        }


    }
}
