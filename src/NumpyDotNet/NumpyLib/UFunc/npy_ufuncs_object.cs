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

#define BUFFER_UFUNCLOOP // doesn't seem to be used.  May only be useful if we have unaligned data which is not possible in .NET I think

using System;
using System.Collections.Concurrent;
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
    #region Data Structures

    internal delegate void NpyUFuncGenericFunction(GenericReductionOp op, VoidPtr[] s1, npy_intp i1, npy_intp[] i2, UFuncOperation ufop);

    internal class NpyUFuncObject : NpyObject_HEAD
    {
        public NpyUFuncObject(NpyUFuncGenericFunction f)
        {
            _function = f;
        }
        public UFuncOperation ops;
        public int nin, nout, nargs;
        public NpyUFuncIdentity identity;
 
        public byte[] data;
        public int ntypes
        {
            get
            {
                if (types == null)
                    return 0;
                return types.Length;
            }
        }
        public string name;
        public NPY_TYPES[] types;
        public object ptr;
        public NpyDict userloops;

        /* generalized ufunc */
        public int core_enabled;      /* 0 for scalar ufunc; 1 for generalized ufunc */
        public int core_num_dim_ix;   /* number of distinct dimension names in signature */

        /* dimension indices of input/output argument k are stored in
         core_dim_ixs[core_offsets[k]..core_offsets[k]+core_num_dims[k]-1] */
        public int[] core_num_dims;    /* numbers of core dimensions of each argument */
        public int[] core_dim_ixs;     /* dimension indices in a flatted form; indices
                            are in the range of [0,core_num_dim_ix) */
        public int[] core_offsets;     /* positions of 1st core dimensions of each
                            argument in core_dim_ixs */
        public string core_signature;  /* signature string for printing purpose */

        private NpyUFuncGenericFunction _function;
        public NpyUFuncGenericFunction GetFunction(int index)
        {
            return _function;
        }


    };

    internal class NpyUFuncLoopObject : NpyObject_HEAD
    {
 
        /* The iterators. */
        public NpyArrayMultiIterObject iter;

        /* The ufunc */
        public NpyUFuncObject ufunc;

        /* The error handling.  These fields are primary used by the interface
           layer to store info about what errors have occured. */
        public UFuncErrors errormask;         /* Integer showing desired error handling */
        public VoidPtr errobj;      /* currently a tuple with
                            (string, func or obj with write method or None) */
        public int first;


        /* Specific function and data to use */
        public NpyUFuncGenericFunction function;

        /* Loop method */
        public UFuncLoopMethod meth;

        /* Whether we need to copy to a buffer or not.*/
        public bool []needbuffer = new bool[npy_defs.NPY_MAXARGS];
        public int leftover;
        public int ninnerloops;
        public int lastdim;

        /* Whether or not to swap */
        public bool []swap = new bool[npy_defs.NPY_MAXARGS];

        /* Buffers for the loop */
        public VoidPtr []buffer = new VoidPtr[npy_defs.NPY_MAXARGS];
        public int bufsize;
        public npy_intp bufcnt;
        public VoidPtr []dptr = new VoidPtr[npy_defs.NPY_MAXARGS];

        /* For casting */
        public VoidPtr[] castbuf = new VoidPtr[npy_defs.NPY_MAXARGS];
        public NpyArray_VectorUnaryFunc []cast = new NpyArray_VectorUnaryFunc[npy_defs.NPY_MAXARGS];

        /* usually points to buffer but when a cast is to be
         done it switches for that argument to castbuf.
         */
        public VoidPtr []bufptr = new VoidPtr[npy_defs.NPY_MAXARGS];

        /* Steps filled in from iters or sizeof(item)
         depending on loop method.
         */
        public npy_intp []steps = new npy_intp[npy_defs.NPY_MAXARGS];

        public int obj;  /* This loop uses object arrays or needs the Python API */
                  /* Flags: UFUNC_OBJ_ISOBJECT, UFUNC_OBJ_NEEDS_API */
        public bool notimplemented; /* The loop caused notimplemented */
        public int objfunc; /* This loop calls object functions
                  (an inner-loop function with argument types */

        /* generalized ufunc */
        public npy_intp[] core_dim_sizes;   /* stores sizes of core dimensions;
                                    contains 1 + core_num_dim_ix elements */
        public npy_intp[] core_strides;     /* strides of loop and core dimensions */
    }

    internal class NpyUFuncReduceObject : NpyObject_HEAD
    {

        public NpyArrayIterObject it;
        public NpyArray ret;
        public NpyArrayIterObject rit;   /* Needed for Accumulate */
        public int outsize;
        public npy_intp index;
        public npy_intp size;
        public VoidPtr idptr = new VoidPtr(new sbyte[(int)UFuncsOptions.NPY_UFUNC_MAXIDENTITY], NPY_TYPES.NPY_BYTE);

        /* The ufunc */
        public NpyUFuncObject ufunc;

        /* The error handling */
        public UFuncErrors errormask;
        public VoidPtr errobj;
        public int first;

        public NpyUFuncGenericFunction function;
        public UFuncLoopMethod meth;
        public bool swap;

        public VoidPtr buffer;
        public int bufsize;

        public VoidPtr castbuf;
        public NpyArray_VectorUnaryFunc cast;

        public VoidPtr []bufptr = new VoidPtr[3];
        public npy_intp[] steps = new npy_intp[3];

        public npy_intp N;
        public long instrides;
        public int insize;
        public VoidPtr inptr;

        /* For copying small arrays */
        public NpyArray decref_arr;

        public int obj;
        public int retbase;

    }

    internal class UFUNCLoopWorkerParams
    {
        public UFUNCLoopWorkerParams(GenericReductionOp op, VoidPtr[] bufptr, npy_intp mm, npy_intp[] steps, UFuncOperation ops)
        {
            npy_intp i = 0;

            this.op = op;

            this.bufptr = new VoidPtr[bufptr.Length];
            foreach (var vp in bufptr)
            {
                this.bufptr[i] = new VoidPtr(vp);
                i++;
            }

            this.N = mm;

            this.steps = new npy_intp[steps.Length];
            Array.Copy(steps, this.steps, steps.Length);

            this.ops = ops;
        }

        public GenericReductionOp op;
        public VoidPtr[] bufptr;
        public npy_intp N;
        public npy_intp[] steps;
        public UFuncOperation ops;
    }

    /* A linked-list of function information for
     user-defined 1-d loops.
     */
    internal class NpyUFunc_Loop1d
    {
        public NpyUFuncGenericFunction func;
        public NPY_TYPES[] arg_types;
        public NpyUFunc_Loop1d next;
    }
    

    public enum UFuncOperation
    {
        add = 0,
        subtract,
        multiply,
        divide,
        remainder,
        fmod,
        power,
        square,
        reciprocal,
        ones_like,
        sqrt,
        negative,
        absolute,
        invert,
        left_shift,
        right_shift,
        bitwise_and,
        bitwise_xor,
        bitwise_or,
        less,
        less_equal,
        equal,
        not_equal,
        greater,
        greater_equal,
        floor_divide,
        true_divide,
        logical_or,
        logical_and,
        floor,
        ceil,
        maximum,
        minimum,
        rint,
        conjugate,
        isnan,
        fmax,
        fmin,
        heaviside,

        // special flags
        //special_operand_is_float,
        no_operation
    };

    internal enum NpyUFuncIdentity
    {
        NpyUFunc_One = 1,
        NpyUFunc_Zero = 0,
        NpyUFunc_None = -1,
    }

    internal enum GenericReductionOp
    {
        NPY_UFUNC_REDUCE = 1,
        NPY_UFUNC_ACCUMULATE,
        NPY_UFUNC_REDUCEAT,
        NPY_UFUNC_OUTER,
    }

    internal enum UFuncErrors
    {
        NPY_UFUNC_FPE_DIVIDEBYZERO = 1,
        NPY_UFUNC_FPE_OVERFLOW = 2,
        NPY_UFUNC_FPE_UNDERFLOW = 4,
        NPY_UFUNC_FPE_INVALID = 8,
        NPY_UFUNC_ERR_DEFAULT = 0,      /* Error mode that avoids look-up (no checking) */
    }

    internal enum UFuncLoopMethod
    {
        NO_UFUNCLOOP = 0,
        ZERO_EL_REDUCELOOP = 0,
        ONE_UFUNCLOOP = 1,
        ONE_EL_REDUCELOOP = 1,
        NOBUFFER_UFUNCLOOP = 2,
        NOBUFFER_REDUCELOOP = 2,
        BUFFER_UFUNCLOOP = 3,
        BUFFER_REDUCELOOP = 3,
        SIGNATURE_NOBUFFER_UFUNCLOOP = 4,
    }


    internal enum UFuncsOptions
    {
        /* Could make this more clever someday */
        NPY_UFUNC_MAXIDENTITY = 32,

        NPY_BUFSIZE = 10000,

        NPY_UFUNC_ERR_DEFAULT = 0, /* Error mode that avoids look-up (no checking) */

    }


    #endregion

    internal partial class numpyinternal
    {


        internal delegate int npy_prepare_outputs_func(NpyUFuncObject self, ref NpyArray mps, VoidPtr data);

        static void NpyErr_NoMemory()
        {
            NpyErr_SetString(npyexc_type.NpyExc_MemoryError, "no memory");
        }

        static void default_fp_error_state(NpyUFuncReduceObject loop)
        {
            loop.bufsize = (int)UFuncsOptions.NPY_BUFSIZE;
            loop.errormask = (int)UFuncsOptions.NPY_UFUNC_ERR_DEFAULT;
            loop.errobj = null;
        }

        private static void default_fp_error_state(NpyUFuncLoopObject loop)
        {
            loop.bufsize = (int)UFuncsOptions.NPY_BUFSIZE;
            loop.errormask = (int)UFuncsOptions.NPY_UFUNC_ERR_DEFAULT;
            loop.errobj = null;
        }

        static void default_fp_error_handler(string name, UFuncErrors errormask, VoidPtr errobj,
                                     int retstatus, ref int first)
        {
            string msg = "unknown";

            switch (errormask)
            {
                case UFuncErrors.NPY_UFUNC_FPE_DIVIDEBYZERO:
                    msg = "division by zero"; break;
                case UFuncErrors.NPY_UFUNC_FPE_OVERFLOW:
                    msg = "overflow"; break;
                case UFuncErrors.NPY_UFUNC_FPE_UNDERFLOW:
                    msg = "underflow"; break;
                case UFuncErrors.NPY_UFUNC_FPE_INVALID:
                    msg = "invalid"; break;
            }
            Console.WriteLine("libndarray floating point {0} warning.", msg);
        }


        internal static int NpyUFunc_GenericFunction(GenericReductionOp operation, NpyUFuncObject self, int nargs, NpyArray[] mps,
                             int ntypenums, NPY_TYPES[] rtypenums,
                             bool originalArgWasObjArray,
                             npy_prepare_outputs_func prepare_outputs,
                             VoidPtr prepare_out_args)
        {
            NpyUFuncLoopObject loop;
            string name = (null != self.name) ? self.name : "";
            int res;
            int i;

            Debug.Assert(Validate(self));

            NpyUFunc_clearfperr();

            /* Build the loop. */
            loop = construct_loop(self);
            if (loop == null)
            {
                return -1;
            }
            default_fp_error_state(loop);

            /* Setup the arrays */
            res = construct_arrays(loop, nargs, mps, ntypenums, rtypenums,
                                   prepare_outputs, prepare_out_args);

            if (res < 0)
            {
                ufuncloop_dealloc(loop);
                return -1;
            }

            /*
             * FAIL with NotImplemented if the other object has
             * the __r<op>__ method and has __array_priority__ as
             * an attribute (signalling it can handle ndarray's)
             * and is not already an ndarray or a subtype of the same type.
             */
            if (self.nin == 2 && self.nout == 1 &&
                NpyArray_TYPE(mps[1]) == NPY_TYPES.NPY_OBJECT && originalArgWasObjArray)
            {
                /* Return -2 for notimplemented. */
                ufuncloop_dealloc(loop);
                NpyErr_SetString(npyexc_type.NpyExc_NotImplementedError, "UFunc not implemented for object");
                return -2;
            }

            if (loop.notimplemented)
            {
                ufuncloop_dealloc(loop);
                NpyErr_SetString(npyexc_type.NpyExc_NotImplementedError, "UFunc not implemented for object");
                return -2;
            }
            if (self.core_enabled != 0 && loop.meth != UFuncLoopMethod.SIGNATURE_NOBUFFER_UFUNCLOOP)
            {
                NpyErr_SetString(npyexc_type.NpyExc_RuntimeError,
                                 "illegal loop method for ufunc with signature");
                goto fail;
            }

            switch (loop.meth)
            {
                case UFuncLoopMethod.ONE_UFUNCLOOP:
                    /*
                     * Everything is contiguous, notswapped, aligned,
                     * and of the right type.  -- Fastest.
                     * Or if not contiguous, then a single-stride
                     * increment moves through the entire array.
                     */
                    /*fprintf(stderr, "ONE...%d\n", loop.size);*/
                    loop.function(operation, loop.bufptr, loop.iter.size, loop.steps, self.ops);
                    if (!NPY_UFUNC_CHECK_ERROR(loop))
                        goto fail;

                    break;
                case UFuncLoopMethod.NOBUFFER_UFUNCLOOP:
                    /*
                     * Everything is notswapped, aligned and of the
                     * right type but not contiguous. -- Almost as fast.
                     */
                    while (loop.iter.index < loop.iter.size)
                    {
                        for (i = 0; i < self.nargs; i++)
                        {
                            loop.bufptr[i] = loop.iter.iters[i].dataptr;
                        }
                        loop.function(operation, loop.bufptr, loop.bufcnt,loop.steps, self.ops);
                        if (!NPY_UFUNC_CHECK_ERROR(loop))
                            goto fail;

                        /* Adjust loop pointers */
                        for (i = 0; i < self.nargs; i++)
                        {
                            NpyArray_ITER_NEXT(loop.iter.iters[i]);
                        }
                        loop.iter.index++;
                    }
                    break;
                case UFuncLoopMethod.SIGNATURE_NOBUFFER_UFUNCLOOP:
                    while (loop.iter.index < loop.iter.size)
                    {
                        for (i = 0; i < self.nargs; i++)
                        {
                            loop.bufptr[i] = loop.iter.iters[i].dataptr;
                        }
                        loop.function(operation, loop.bufptr, loop.core_dim_sizes[0],loop.core_strides, self.ops);
                        if (!NPY_UFUNC_CHECK_ERROR(loop))
                            goto fail;

                        /* Adjust loop pointers */
                        for (i = 0; i < self.nargs; i++)
                        {
                            NpyArray_ITER_NEXT(loop.iter.iters[i]);
                        }
                        loop.iter.index++;
                    }
                    break;
                case UFuncLoopMethod.BUFFER_UFUNCLOOP:
#if BUFFER_UFUNCLOOP
                    {

                        /* This should be a function */
                        NpyArray_CopySwapNFunc[] copyswapn = new NumpyLib.NpyArray_CopySwapNFunc[npy_defs.NPY_MAXARGS];
                        NpyArrayIterObject[] iters = loop.iter.iters;
                        bool[] swap = loop.swap;
                        VoidPtr[] dptr = loop.dptr;
                        int[] mpselsize = new int[npy_defs.NPY_MAXARGS];
                        npy_intp[] laststrides = new npy_intp[npy_defs.NPY_MAXARGS];
                        bool[] fastmemcpy = new bool[npy_defs.NPY_MAXARGS];
                        bool[] needbuffer = loop.needbuffer;
                        npy_intp index = loop.iter.index, size = loop.iter.size;
                        int bufsize;
                        npy_intp bufcnt;
                        int[] copysizes = new int[npy_defs.NPY_MAXARGS];
                        VoidPtr[] bufptr = loop.bufptr;
                        VoidPtr[] buffer = loop.buffer;
                        VoidPtr[] castbuf = loop.castbuf;
                        npy_intp[] steps = loop.steps;
                        VoidPtr[] tptr = new VoidPtr[npy_defs.NPY_MAXARGS];
                        int ninnerloops = loop.ninnerloops;
                        bool[] pyobject = new bool[npy_defs.NPY_MAXARGS];
                        int[] datasize = new int[npy_defs.NPY_MAXARGS];
                        int j, k, stopcondition;
                        VoidPtr myptr1, myptr2;

                        for (i = 0; i < self.nargs; i++)
                        {
                            copyswapn[i] = _default_copyswap;
                            mpselsize[i] = NpyArray_DESCR(mps[i]).elsize;
  
                            laststrides[i] = iters[i].strides[loop.lastdim];
                            if (steps[i] != 0 && laststrides[i] != mpselsize[i])
                            {
                                fastmemcpy[i] = false;
                            }
                            else
                            {
                                fastmemcpy[i] = true;
                            }
                        }
                        /* Do generic buffered looping here (works for any kind of
                         * arrays -- some need buffers, some don't.
                         *
                         *
                         * New algorithm: N is the largest dimension.  B is the buffer-size.
                         * quotient is loop.ninnerloops-1
                         * remainder is loop.leftover
                         *
                         * Compute N = quotient * B + remainder.
                         * quotient = N / B  # integer math
                         * (store quotient + 1) as the number of innerloops
                         * remainder = N % B # integer remainder
                         *
                         * On the inner-dimension we will have (quotient + 1) loops where
                         * the size of the inner function is B for all but the last when
                         * the niter size is remainder.
                         *
                         * So, the code looks very similar to NOBUFFER_LOOP except the
                         * inner-most loop is replaced with...
                         *
                         * for(i=0; i<quotient+1; i++) {
                         * if (i==quotient+1) make itersize remainder size
                         * copy only needed items to buffer.
                         * swap input buffers if needed
                         * cast input buffers if needed
                         * call loop_function()
                         * cast outputs in buffers if needed
                         * swap outputs in buffers if needed
                         * copy only needed items back to output arrays.
                         * update all data-pointers by strides*niter
                         * }
                         */

                        /*
                         * fprintf(stderr, "BUFFER...%d,%d,%d\n", loop.size,
                         * loop.ninnerloops, loop.leftover);
                         */
                        /*
                         * for(i=0; i<self.nargs; i++) {
                         * fprintf(stderr, "iters[%d].dataptr = %p, %p of size %d\n", i,
                         * iters[i], iters[i].ao.data, PyArray_NBYTES(iters[i].ao));
                         * }
                         */
                        stopcondition = ninnerloops;
                        if (loop.leftover == 0)
                        {
                            stopcondition--;
                        }
                        while (index < size)
                        {
                            bufsize = loop.bufsize;
                            for (i = 0; i < self.nargs; i++)
                            {
                                tptr[i] = loop.iter.iters[i].dataptr;
                                if (needbuffer[i])
                                {
                                    dptr[i] = bufptr[i];
                                    datasize[i] = (steps[i] != 0 ? bufsize : 1);
                                    copysizes[i] = datasize[i] * mpselsize[i];
                                }
                                else
                                {
                                    dptr[i] = tptr[i];
                                }
                            }

                            /* This is the inner function over the last dimension */
                            for (k = 1; k <= stopcondition; k++)
                            {
                                if (k == ninnerloops)
                                {
                                    bufsize = loop.leftover;
                                    for (i = 0; i < self.nargs; i++)
                                    {
                                        if (!needbuffer[i])
                                        {
                                            continue;
                                        }
                                        datasize[i] = (steps[i] != 0 ? bufsize : 1);
                                        copysizes[i] = datasize[i] * mpselsize[i];
                                    }
                                }

                                var helper = MemCopy.GetMemcopyHelper(buffer[0]);
                                for (i = 0; i < self.nin; i++)
                                {
                                    if (!needbuffer[i])
                                    {
                                        continue;
                                    }
                                    if (fastmemcpy[i])
                                    {
                                        helper.memmove_init(buffer[i], tptr[i]);
                                        helper.memcpy(buffer[i].data_offset, tptr[i].data_offset, copysizes[i]);
                                    }
                                    else
                                    {
                                        myptr1 = buffer[i];
                                        myptr2 = tptr[i];
                                        helper.memmove_init(myptr1, myptr2);
                                        for (j = 0; j < bufsize; j++)
                                        {
                                            helper.memcpy(myptr1.data_offset, myptr2.data_offset, mpselsize[i]);
                                            myptr1 += mpselsize[i];
                                            myptr2 += laststrides[i];
                                        }
                                    }

                                    /* swap the buffer if necessary */
                                    if (swap[i])
                                    {
                                        /* fprintf(stderr, "swapping...\n");*/
                                        copyswapn[i](buffer[i], mpselsize[i], null, -1,
                                                     (npy_intp)datasize[i], true,
                                                     mps[i]);
                                    }
                                    /* cast to the other buffer if necessary */
                                    if (loop.cast[i] != null)
                                    {
                                        /* fprintf(stderr, "casting... %d, %p %p\n", i, buffer[i]); */
                                        loop.cast[i](buffer[i], castbuf[i],
                                                      (npy_intp)datasize[i],
                                                      null, null);
                                    }
                                }

                                bufcnt = (npy_intp)bufsize;
                                loop.function(operation, dptr, bufcnt, steps, self.ops);
                                if (!NPY_UFUNC_CHECK_ERROR(loop))
                                    goto fail;

                                for (i = self.nin; i < self.nargs; i++)
                                {
                                    if (!needbuffer[i])
                                    {
                                        continue;
                                    }
                                    if (loop.cast[i] != null)
                                    {
                                        /* fprintf(stderr, "casting back... %d, %p", i,
                                           castbuf[i]); */
                                        loop.cast[i](castbuf[i],
                                                      buffer[i],
                                                      (npy_intp)datasize[i],
                                                      null, null);
                                    }
                                    if (swap[i])
                                    {
                                        copyswapn[i](buffer[i], mpselsize[i], null, -1,
                                                     (npy_intp)datasize[i], true,
                                                     mps[i]);
                                    }
                                    /* copy back to output arrays decref what's already
                                       there for object arrays */
                                    //if (pyobject[i])
                                    //{
                                    //    myptr1 = tptr[i];
                                    //    for (j = 0; j < datasize[i]; j++)
                                    //    {
                                    //        NpyInterface_DECREF(myptr1);
                                    //        myptr1 += laststrides[i];
                                    //    }
                                    //}
                                    if (fastmemcpy[i])
                                    {
                                        helper.memmove_init(tptr[i], buffer[i]);
                                        helper.memcpy(tptr[i].data_offset, buffer[i].data_offset, copysizes[i]);
                                    }
                                    else
                                    {
                                        myptr2 = buffer[i];
                                        myptr1 = tptr[i];
                                        helper.memmove_init(myptr1, myptr2);
                                        for (j = 0; j < bufsize; j++)
                                        {
                                            helper.memcpy(myptr1.data_offset, myptr2.data_offset, mpselsize[i]);
                                            myptr1 += laststrides[i];
                                            myptr2 += mpselsize[i];
                                        }
                                    }
                                }
                                if (k == stopcondition)
                                {
                                    continue;
                                }
                                for (i = 0; i < self.nargs; i++)
                                {
                                    tptr[i] += bufsize * laststrides[i];
                                    if (!needbuffer[i])
                                    {
                                        dptr[i] = tptr[i];
                                    }
                                }
                            }
                            /* end inner function over last dimension */

                            if (loop.objfunc != 0)
                            {
                                /*
                                 * DECREF castbuf when underlying function used
                                 * object arrays and casting was needed to get
                                 * to object arrays
                                 */
                                for (i = 0; i < self.nargs; i++)
                                {
                                    if (loop.cast[i] != null)
                                    {
                                        if (steps[i] == 0)
                                        {
                                            NpyInterface_DECREF(castbuf[i]);
                                        }
                                        else
                                        {
                                            int size2 = loop.bufsize;

                                            VoidPtr objptr = castbuf[i];
                                            /*
                                             * size is loop.bufsize unless there
                                             * was only one loop
                                             */
                                            if (ninnerloops == 1)
                                            {
                                                size2 = loop.leftover;
                                            }
                                            for (j = 0; j < size2; j++)
                                            {
                                                NpyInterface_DECREF(objptr);
                                                objptr.datap = null;
                                                objptr += 1;
                                            }
                                        }
                                    }
                                }
                                /* Prevent doing the decref twice on an error. */
                                loop.objfunc = 0;
                            }
                            /* fixme -- probably not needed here*/
                            if (!NPY_UFUNC_CHECK_ERROR(loop))
                                goto fail;

                            for (i = 0; i < self.nargs; i++)
                            {
                                NpyArray_ITER_NEXT(loop.iter.iters[i]);
                            }
                            index++;
                        }
                    } /* end of last case statement */
#endif
                    break;
            }

            for (i = 0; i < nargs; i++)
            {
                if (mps[i] != null && (mps[i].flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) != 0)
                {
                    NpyArray_ForceUpdate(mps[i]);
                }
            }

            ufuncloop_dealloc(loop);
            return 0;

            fail:
            if (loop != null)
            {
                if (loop.objfunc != 0)
                {
                    VoidPtr[] castbuf = loop.castbuf;
                    npy_intp[] steps = loop.steps;
                    int ninnerloops = loop.ninnerloops;
                    int j;

                    /*
                     * DECREF castbuf when underlying function used
                     * object arrays and casting was needed to get
                     * to object arrays
                     */
                    for (i = 0; i < self.nargs; i++)
                    {
                        if (loop.cast[i] != null)
                        {
                            if (steps[i] == 0)
                            {
                                NpyInterface_DECREF(castbuf[i]);
                            }
                            else
                            {
                                int size = loop.bufsize;

                                VoidPtr objptr = castbuf[i];
                                /*
                                 * size is loop.bufsize unless there
                                 * was only one loop
                                 */
                                if (ninnerloops == 1)
                                {
                                    size = loop.leftover;
                                }
                                for (j = 0; j < size; j++)
                                {
                                    NpyInterface_DECREF(objptr);
                                    objptr.datap = null;
                                    objptr += 1;
                                }
                            }
                        }
                    }
                }
                ufuncloop_dealloc(loop);
            }
            return -1;
        }

   

        internal static NpyArray NpyUFunc_GenericReduction(NpyUFuncObject self, NpyArray arr, NpyArray indices,
                         NpyArray _out, int axis, NpyArray_Descr otype, GenericReductionOp operation, bool keepdims)
        {
            if (self == null)
            {
                throw new Exception("UFunc does not exist for this operation");
            }


            switch (self.ops)
            {
                case UFuncOperation.add:
                    switch (arr.ItemType)
                    {
                        case NPY_TYPES.NPY_BOOL:
                            arr = NpyArray_CastToType(arr, NpyArray_DescrFromType(NPY_TYPES.NPY_INT32), false);
                            otype.type_num = arr.ItemType;
                            break;
                        default:
                            break;

                    }
                    break;
                default:
                    break;
            }


            if (otype.type_num != arr.ItemType)
            {
                arr = NpyArray_CastToType(arr, NpyArray_DescrFromType(otype.type_num), NpyArray_ISFORTRAN(arr));
            }


            NpyArray ret = null;

            /* Check to see if input is zero-dimensional */
            if (NpyArray_NDIM(arr) == 0)
            {
                string buf = string.Format("cannot %s on a scalar", operation.ToString());
                NpyErr_SetString(npyexc_type.NpyExc_TypeError, buf);
                Npy_DECREF(arr);
                return null;
            }
            /* Check to see that type (and otype) is not FLEXIBLE */
            if (NpyArray_ISFLEXIBLE(arr) ||
                (null != otype && NpyTypeNum_ISFLEXIBLE(otype.type_num)))
            {
                string buf = string.Format("cannot perform %s with flexible type", operation.ToString());

                NpyErr_SetString(npyexc_type.NpyExc_TypeError, buf);
                return null;
            }

            if (axis < 0)
            {
                axis += NpyArray_NDIM(arr);
            }
            if (axis < 0 || axis >= NpyArray_NDIM(arr))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "axis not in array");
                return null;
            }

            /*
             * If out is specified it determines otype
             * unless otype already specified.
             */
            if (otype == null && _out != null)
            {
                otype = NpyArray_DESCR(_out);
                Npy_INCREF(otype);
            }
            if (otype == null)
            {
                /*
                 * For integer types --- make sure at least a long
                 * is used for add and multiply reduction to avoid overflow
                 */
                NPY_TYPES typenum = NpyArray_TYPE(arr);
                if ((typenum < NPY_TYPES.NPY_FLOAT) &&
                    (self.name.Equals("add", StringComparison.CurrentCultureIgnoreCase) == true) ||
                    (self.name.Equals("multiply", StringComparison.CurrentCultureIgnoreCase) == true))
                {
                    if (NpyTypeNum_ISBOOL(typenum))
                    {
                        typenum = NPY_TYPES.NPY_INT64;
                    }
                    else if (NpyArray_ITEMSIZE(arr) < sizeof(long))
                    {
                        if (NpyTypeNum_ISUNSIGNED(typenum))
                        {
                            typenum = NPY_TYPES.NPY_UINT64;
                        }
                        else
                        {
                            typenum = NPY_TYPES.NPY_INT64;
                        }
                    }
                }
                otype = NpyArray_DescrFromType(typenum);
            }


            switch (operation)
            {
                case GenericReductionOp.NPY_UFUNC_REDUCE:
                    ret = NpyUFunc_Reduce(operation, self, arr, _out, axis,
                                          otype.type_num);
                    break;
                case GenericReductionOp.NPY_UFUNC_ACCUMULATE:
                    ret = NpyUFunc_Accumulate(operation, self, arr, _out, axis,
                                              otype.type_num);
                    break;
                case GenericReductionOp.NPY_UFUNC_REDUCEAT:
                    ret = NpyUFunc_Reduceat(operation, self, arr, indices, _out,
                                            axis, otype.type_num);
                    Npy_DECREF(indices);
                    break;
                default:
                    Debug.Assert(false, "unexpected GenericReductionOp value");
                    break;
            }

            if (keepdims)
            {
                npy_intp[] ExpandedDims = null;
                if (ret.dimensions == null || ret.nd <= 0)
                {
                    ExpandedDims = new npy_intp[1] { 1 };
                }
                else
                {
                    ExpandedDims = new npy_intp[ret.nd + 1];
                }


                int j = 0;
                for (int i = 0; i < ExpandedDims.Length; i++)
                {
                    if (i == axis)
                    {
                        ExpandedDims[i] = 1;
                    }
                    else
                    {
                        ExpandedDims[i] = ret.dimensions[j];
                        j++;
                    }
                }

                ret = NpyArray_Newshape(ret, new NpyArray_Dims() { len = ExpandedDims.Length, ptr = ExpandedDims }, NPY_ORDER.NPY_ANYORDER);
            }

            if (HasBoolReturn(self.ops))
            {
                ret = NpyArray_CastToType(ret, NpyArray_DescrFromType(NPY_TYPES.NPY_BOOL), false);
            }

            return ret;
        }


        private static void NpyUFunc_PerformUFunc(NpyArray srcArray, NpyArray destArray, ref object cumsum, npy_intp[] dimensions, int dimIdx, npy_intp src_offset, npy_intp dest_offset, NumericOperations operation)
        {
            if (numpyinternal.getEnableTryCatchOnCalculations)
            {
                if (dimIdx == destArray.nd)
                {
                    var srcValue = operation.srcGetItem(src_offset + srcArray.data.data_offset);

                    cumsum = operation.operation(srcValue, operation.ConvertOperand(cumsum));

                    try
                    {
                        operation.destSetItem(dest_offset + destArray.data.data_offset, cumsum);
                    }
                    catch
                    {
                        operation.destSetItem(dest_offset + destArray.data.data_offset, 0);
                    }
                }
                else
                {
                    for (int i = 0; i < dimensions[dimIdx]; i++)
                    {
                        npy_intp lsrc_offset = src_offset + srcArray.strides[dimIdx] * i;
                        npy_intp ldest_offset = dest_offset + destArray.strides[dimIdx] * i;

                        NpyUFunc_PerformUFunc(srcArray, destArray, ref cumsum, dimensions, dimIdx + 1, lsrc_offset, ldest_offset, operation);
                    }
                }
            }
            else
            {
                try
                {
                    if (dimIdx == destArray.nd)
                    {
                        var srcValue = operation.srcGetItem(src_offset + srcArray.data.data_offset);

                        cumsum = operation.operation(srcValue, operation.ConvertOperand(cumsum));
                        operation.destSetItem(dest_offset + destArray.data.data_offset, cumsum);
                    }
                    else
                    {
                        for (int i = 0; i < dimensions[dimIdx]; i++)
                        {
                            npy_intp lsrc_offset = src_offset + srcArray.strides[dimIdx] * i;
                            npy_intp ldest_offset = dest_offset + destArray.strides[dimIdx] * i;

                            NpyUFunc_PerformUFunc(srcArray, destArray, ref cumsum, dimensions, dimIdx + 1, lsrc_offset, ldest_offset, operation);
                        }
                    }
                }
                catch (Exception ex)
                {
                    string Message = numpyinternal.GenerateTryCatchExceptionMessage(ex.Message);
                    throw new Exception(Message);
                }

            }

   
        }



        /*
         * We have two basic kinds of loops. One is used when arr is not-swapped
         * and aligned and output type is the same as input type.  The other uses
         * buffers when one of these is not satisfied.
         *
         *  Zero-length and one-length axes-to-be-reduced are handled separately.
         */
        private static NpyArray NpyUFunc_Reduce(GenericReductionOp operation, NpyUFuncObject self, NpyArray arr, NpyArray _out, int axis, NPY_TYPES otype)
        {
            NpyArray ret = null;
            NpyUFuncReduceObject loop;
            npy_intp i, n;
            VoidPtr dptr;

            Debug.Assert(arr != null);
            Debug.Assert(Validate(arr));
            Debug.Assert(Validate(NpyArray_DESCR(arr)));
            Debug.Assert(Validate(self));


            /* Construct loop object */
            loop = construct_reduce(self, ref arr, _out, axis, otype, GenericReductionOp.NPY_UFUNC_REDUCE, 0);
            if (loop == null)
            {
                return null;
            }

            ICopyHelper helper = MemCopy.GetMemcopyHelper(loop.bufptr[0]);

            switch (loop.meth)
            {
                case UFuncLoopMethod.ZERO_EL_REDUCELOOP:
                    /* fprintf(stderr, "ZERO..%d\n", loop.size); */
                    helper.memmove_init(loop.bufptr[0], loop.idptr);
                    for (i = 0; i < loop.size; i++)
                    {
                        helper.memmove(loop.bufptr[0].data_offset, loop.idptr.data_offset, loop.outsize);
                        loop.bufptr[0] += loop.outsize;
                    }
                    break;
                case UFuncLoopMethod.ONE_EL_REDUCELOOP:
                    /*fprintf(stderr, "ONEDIM..%d\n", loop.size); */

                    // kevin - the loop count is not ever more than 2 or 3.  Probably not worth speeding up.

                    helper.memmove_init(loop.bufptr[0], loop.it.dataptr);
                    while (loop.index < loop.size)
                    {
                        helper.memmove(loop.bufptr[0].data_offset, loop.it.dataptr.data_offset, loop.outsize);
                        NpyArray_ITER_NEXT(loop.it);
                        loop.bufptr[0] += loop.outsize;
                        loop.index++;
                    }
                    break;
                case UFuncLoopMethod.NOBUFFER_UFUNCLOOP:
                    /*fprintf(stderr, "NOBUFFER..%d\n", loop.size); */

                    helper.memmove_init(loop.bufptr[0], loop.it.dataptr);

                    var UFuncHandler = GetGeneralReductionUFuncHandler(operation, loop.bufptr);

                    var loopcnt = loop.size - loop.index;
                    if (loopcnt <= 1 || UFuncHandler == null)
                    {
                        while (loop.index < loop.size)
                        {
                            helper.memmove(loop.bufptr[0].data_offset , loop.it.dataptr.data_offset, loop.outsize);
                            /* Adjust input pointer */
                            loop.bufptr[1] = loop.it.dataptr + loop.steps[1];

                            loop.function(operation, loop.bufptr, loop.N, loop.steps, self.ops);
                            if (!NPY_UFUNC_CHECK_ERROR(loop))
                                goto fail;

                            NpyArray_ITER_NEXT(loop.it);
                            loop.bufptr[0] += loop.outsize;
                            loop.bufptr[2] = loop.bufptr[0];
                            loop.index++;
                        }
                    }
                    else
                    {
                        ConcurrentQueue<UFUNCLoopWorkerParams> workToDo = new ConcurrentQueue<UFUNCLoopWorkerParams>();

                        bool HasError = false;
                        Task workerThread = null;
                        bool IsCompleted = false;

                        while (loop.index < loop.size)
                        {
                            helper.memmove(loop.bufptr[0].data_offset, loop.it.dataptr.data_offset, loop.outsize);
                            /* Adjust input pointer */
                            loop.bufptr[1] = loop.it.dataptr + loop.steps[1];

                            // put on queue for worker task to process
                            workToDo.Enqueue(new UFUNCLoopWorkerParams(operation, loop.bufptr, loop.N, loop.steps, self.ops));

                            if (workerThread == null)
                            {
                                // start worker thread to process the queued up work
                                workerThread = Task.Factory.StartNew(() =>
                                {
                                    while (true)
                                    {
                                        Parallel.For(0, workToDo.Count(), xxx =>
                                        {
                                            UFUNCLoopWorkerParams work = null;
                                            if (workToDo.TryDequeue(out work))
                                            {
                                                UFuncHandler(work.bufptr, work.steps, work.ops, work.N);
                                                if (!NPY_UFUNC_CHECK_ERROR(loop))
                                                {
                                                    HasError = true;
                                                }
                                            }

                                        });

                                        if (workToDo.Count() == 0 && IsCompleted)
                                            break;
                                    }

                                });
                            }


                            NpyArray_ITER_NEXT(loop.it);
                            loop.bufptr[0] += loop.outsize;
                            loop.bufptr[2] = loop.bufptr[0];
                            loop.index++;
                        }

                        IsCompleted = true;

                        workerThread.Wait();

                        if (HasError)
                            goto fail;
                    }

    

                    break;

                case UFuncLoopMethod.BUFFER_UFUNCLOOP:
#if BUFFER_UFUNCLOOP
                    /*
                     * use buffer for arr
                     *
                     * For each row to reduce
                     * 1. copy first item over to output (casting if necessary)
                     * 2. Fill inner buffer
                     * 3. When buffer is filled or end of row
                     * a. Cast input buffers if needed
                     * b. Call inner function.
                     * 4. Repeat 2 until row is done.
                     */
                    /* fprintf(stderr, "BUFFERED..%d %d\n", loop.size, loop.swap); */

                    while (loop.index < loop.size)
                    {
                        loop.inptr = loop.it.dataptr;
                        /* Copy (cast) First term over to output */
                        if (loop.cast != null)
                        {
                            /* A little tricky because we need to cast it first */
                            helper.copyswap(loop.buffer, loop.inptr, loop.swap);
                            loop.cast(loop.buffer, loop.castbuf, 1, null, null);

                            helper.memmove_init(loop.bufptr[0], loop.castbuf);
                            helper.memcpy(loop.bufptr[0].data_offset, loop.castbuf.data_offset, loop.outsize);
                        }
                        else
                        {
                            /* Simple copy */
                            helper.copyswap(loop.bufptr[0],loop.inptr,loop.swap);
                        }
                        loop.inptr += loop.instrides;
                        n = 1;
                        while (n < loop.N)
                        {
                            /* Copy up to loop.bufsize elements to buffer */
                            dptr = loop.buffer;
                            for (i = 0; i < loop.bufsize; i++, n++)
                            {
                                if (n == loop.N)
                                {
                                    break;
                                }
                                helper.copyswap(dptr, loop.inptr, loop.swap);
                                loop.inptr += loop.instrides;
                                dptr += loop.insize;
                            }
                            if (loop.cast != null)
                            {
                                loop.cast(loop.buffer, loop.castbuf, i, null, null);
                            }
                            loop.function(operation, loop.bufptr, i, loop.steps, self.ops);
                            loop.bufptr[0] += loop.steps[0] * i;
                            loop.bufptr[2] += loop.steps[2] * i;
                            if (!NPY_UFUNC_CHECK_ERROR(loop))
                                goto fail;
                        }
                        NpyArray_ITER_NEXT(loop.it);
                        loop.bufptr[0] += loop.outsize;
                        loop.bufptr[2] = loop.bufptr[0];
                        loop.index++;
                    }
#endif
                    break;
            }

            /* Hang on to this reference -- will be decref'd with loop */
            if (loop.retbase != 0)
            {
                ret = loop.ret.base_arr;
                NpyArray_ForceUpdate(loop.ret);
            }
            else
            {
                ret = loop.ret;
            }
            Npy_INCREF(ret);
            ufuncreduce_dealloc(loop);
            return ret;

            fail:
            if (loop != null)
            {
                ufuncreduce_dealloc(loop);
            }
            return null;
        }


        private static NpyArray NpyUFunc_Accumulate(GenericReductionOp operation, NpyUFuncObject self, NpyArray arr, NpyArray _out, int axis, NPY_TYPES otype)
        {
            NpyArray ret = null;
            NpyUFuncReduceObject loop;
            npy_intp i, n;
            VoidPtr dptr;

            Debug.Assert(Validate(self));

            /* Construct loop object */
            loop = construct_reduce(self, ref arr, _out, axis, otype, GenericReductionOp.NPY_UFUNC_ACCUMULATE, 0);
            if (loop == null)
            {
                return null;
            }

            ICopyHelper helper = MemCopy.GetMemcopyHelper(loop.bufptr[0]);

            switch (loop.meth)
            {
                case UFuncLoopMethod.ZERO_EL_REDUCELOOP:
                    /* Accumulate */
                    /* fprintf(stderr, "ZERO..%d\n", loop.size); */
                    for (i = 0; i < loop.size; i++)
                    {
                        helper.memmove_init(loop.bufptr[0], loop.idptr);
                        helper.memcpy(loop.bufptr[0].data_offset, loop.idptr.data_offset, loop.outsize);
                        loop.bufptr[0] += loop.outsize;
                    }
                    break;
                case UFuncLoopMethod.ONE_EL_REDUCELOOP:
                    /* Accumulate */
                    /* fprintf(stderr, "ONEDIM..%d\n", loop.size); */

                    helper.memmove_init(loop.bufptr[0], loop.it.dataptr);
                    while (loop.index < loop.size)
                    {
                        helper.memmove(loop.bufptr[0].data_offset, loop.it.dataptr.data_offset, loop.outsize);
                        NpyArray_ITER_NEXT(loop.it);
                        loop.bufptr[0] += loop.outsize;
                        loop.index++;
                    }
                    break;
                case UFuncLoopMethod.NOBUFFER_UFUNCLOOP:
                    /* Accumulate */
                    /* fprintf(stderr, "NOBUFFER..%d\n", loop.size); */

                    helper.memmove_init(loop.bufptr[0], loop.it.dataptr);

                    var UFuncHandler = GetGeneralReductionUFuncHandler(operation, loop.bufptr);

                    var loopcnt = loop.size - loop.index;
                    if (loopcnt <= 1 || UFuncHandler == null)
                    {
                        while (loop.index < loop.size)
                        {
                            helper.memmove(loop.bufptr[0].data_offset, loop.it.dataptr.data_offset, loop.outsize);
                            /* Adjust input pointer */
                            loop.bufptr[1] = loop.it.dataptr + loop.steps[1];
                            loop.function(operation, loop.bufptr, loop.N, loop.steps, self.ops);
                            if (!NPY_UFUNC_CHECK_ERROR(loop))
                                goto fail;
                            NpyArray_ITER_NEXT(loop.it);
                            NpyArray_ITER_NEXT(loop.rit);
                            loop.bufptr[0] = loop.rit.dataptr;
                            loop.bufptr[2] = loop.bufptr[0] + loop.steps[0];
                            loop.index++;
                        }
                    }
                    else
                    {
                        ConcurrentQueue<UFUNCLoopWorkerParams> workToDo = new ConcurrentQueue<UFUNCLoopWorkerParams>();

                        bool HasError = false;
                        Task workerThread = null;
                        bool IsCompleted = false;

                        while (loop.index < loop.size)
                        {
                            helper.memmove(loop.bufptr[0].data_offset, loop.it.dataptr.data_offset, loop.outsize);
                            /* Adjust input pointer */
                            loop.bufptr[1] = loop.it.dataptr + loop.steps[1];

                            workToDo.Enqueue(new UFUNCLoopWorkerParams(operation, loop.bufptr, loop.N, loop.steps, self.ops));

                            if (workerThread == null)
                            {
                                // start worker thread to process the queued up work
                                workerThread = Task.Factory.StartNew(() =>
                                {
                                    while (true)
                                    {
                                        Parallel.For(0, workToDo.Count(), xxx =>
                                        {
                                            UFUNCLoopWorkerParams work = null;
                                            if (workToDo.TryDequeue(out work))
                                            {
                                                UFuncHandler(work.bufptr, work.steps, work.ops, work.N);
                                                if (!NPY_UFUNC_CHECK_ERROR(loop))
                                                {
                                                    HasError = true;
                                                }
                                            }

                                        });

                                        if (workToDo.Count() == 0)
                                        {
                                            if (IsCompleted)
                                                break;

                                            //System.Threading.Thread.Sleep(100);
                                        }
                                    
                                    }

                                });
                            }

                            NpyArray_ITER_NEXT(loop.it);
                            NpyArray_ITER_NEXT(loop.rit);
                            loop.bufptr[0] = loop.rit.dataptr;
                            loop.bufptr[2] = loop.bufptr[0] + loop.steps[0];
                            loop.index++;
                        }

                        IsCompleted = true;

                        workerThread.Wait();

                        if (HasError)
                            goto fail;

                    }



                    break;
                case UFuncLoopMethod.BUFFER_UFUNCLOOP:
#if BUFFER_UFUNCLOOP
                    /* Accumulate
                     *
                     * use buffer for arr
                     *
                     * For each row to reduce
                     * 1. copy identity over to output (casting if necessary)
                     * 2. Fill inner buffer
                     * 3. When buffer is filled or end of row
                     * a. Cast input buffers if needed
                     * b. Call inner function.
                     * 4. Repeat 2 until row is done.
                     */
                    /* fprintf(stderr, "BUFFERED..%d %p\n", loop.size, loop.cast); */
                    while (loop.index < loop.size)
                    {
                        loop.inptr = loop.it.dataptr;
                        /* Copy (cast) First term over to output */
                        if (loop.cast != null)
                        {
                            /* A little tricky because we need to
                             cast it first */
                            helper.copyswap(loop.buffer, loop.inptr,loop.swap);
                            loop.cast(loop.buffer, loop.castbuf, 1, null, null);

                            helper.memmove_init(loop.bufptr[0], loop.castbuf);
                            helper.memcpy(loop.bufptr[0].data_offset, loop.castbuf.data_offset, loop.outsize);
                        }
                        else
                        {
                            /* Simple copy */
                            helper.copyswap(loop.bufptr[0],loop.inptr,loop.swap);
                        }
                        loop.inptr += loop.instrides;
                        n = 1;
                        while (n < loop.N)
                        {
                            /* Copy up to loop.bufsize elements to buffer */
                            dptr = loop.buffer;
                            for (i = 0; i < loop.bufsize; i++, n++)
                            {
                                if (n == loop.N)
                                {
                                    break;
                                }
                                helper.copyswap(dptr, loop.inptr,loop.swap);
                                loop.inptr += loop.instrides;
                                dptr += loop.insize;
                            }
                            if (loop.cast != null)
                            {
                                loop.cast(loop.buffer, loop.castbuf, i, null, null);
                            }
                            loop.function(operation, loop.bufptr, i, loop.steps, self.ops);
                            loop.bufptr[0] += loop.steps[0] * i;
                            loop.bufptr[2] += loop.steps[2] * i;
                            if (!NPY_UFUNC_CHECK_ERROR(loop))
                                goto fail;
                        }
                        NpyArray_ITER_NEXT(loop.it);
                        NpyArray_ITER_NEXT(loop.rit);
                        loop.bufptr[0] = loop.rit.dataptr;
                        loop.bufptr[2] = loop.bufptr[0] + loop.steps[0];
                        loop.index++;
                    }
#endif
                    break;

            }
            /* Hang on to this reference -- will be decref'd with loop */
            if (loop.retbase != 0)
            {
                ret = NpyArray_BASE_ARRAY(loop.ret);
                NpyArray_ForceUpdate(loop.ret);
            }
            else
            {
                ret = loop.ret;
            }
            Npy_INCREF(ret);
            ufuncreduce_dealloc(loop);
            return ret;

            fail:

            if (loop != null)
            {
                ufuncreduce_dealloc(loop);
            }
            return null;
        }
         

        private static NpyArray NpyUFunc_Reduceat(GenericReductionOp operation, NpyUFuncObject self, NpyArray arr, NpyArray ind, NpyArray _out, int axis, NPY_TYPES otype)
        {
            NpyArray ret;
            NpyUFuncReduceObject loop;
            npy_intp[] ptr = (npy_intp[])NpyArray_BYTES(ind).datap;
            npy_intp nn = NpyArray_DIM(ind, 0);
            npy_intp mm = NpyArray_DIM(arr, axis) - 1;
            npy_intp n, i, j;
            VoidPtr dptr;

            Debug.Assert(Validate(self));

            /* Check for out-of-bounds values in indices array */
            for (i = 0; i < nn; i++)
            {
                if ((ptr[i] < 0) || (ptr[i] > mm))
                {
                    string buf = string.Format("index out-of-bounds (0, {0})", (int)mm);
                    NpyErr_SetString(npyexc_type.NpyExc_IndexError, buf);
                    return null;
                }
            }

            ptr = (npy_intp[])NpyArray_BYTES(ind).datap;
            /* Construct loop object */
            loop = construct_reduce(self, ref arr, _out, axis, otype,
                                    GenericReductionOp.NPY_UFUNC_REDUCEAT, nn);
            if (loop == null)
            {
                return null;
            }

            var helper = MemCopy.GetMemcopyHelper(arr.data);

            switch (loop.meth)
            {
                case UFuncLoopMethod.ZERO_EL_REDUCELOOP:
                    /* zero-length index -- return array immediately */
                    /* fprintf(stderr, "ZERO..\n"); */
                    break;
                case UFuncLoopMethod.NOBUFFER_UFUNCLOOP:
                    /* Reduceat
                     * NOBUFFER -- behaved array and same type
                     */

                    var UFuncHandler = GetGeneralReductionUFuncHandler(operation, loop.bufptr);

                    var loopcnt = loop.size - loop.index;
                    if (loopcnt <= 1)
                    {
                        while (loop.index < loop.size)
                        {
                            ptr = (npy_intp[])NpyArray_BYTES(ind).datap;
                            for (i = 0; i < nn; i++)
                            {
                                loop.bufptr[1] = loop.it.dataptr + ptr[i] * loop.steps[1];

                                helper.memmove_init(loop.bufptr[0], loop.bufptr[1]);
                                helper.memcpy(loop.bufptr[0].data_offset, loop.bufptr[1].data_offset, loop.outsize);

                                mm = (i == nn - 1 ? NpyArray_DIM(arr, axis) - ptr[i] : ptr[i + 1] - ptr[i]) - 1;
                                if (mm > 0)
                                {
                                    loop.bufptr[1] += loop.steps[1];
                                    loop.bufptr[2] = loop.bufptr[0];

                                    if (UFuncHandler != null)
                                    {
                                        UFuncHandler(loop.bufptr, loop.steps, self.ops, mm);
                                    }
                                    else
                                    {
                                        loop.function(operation, loop.bufptr, mm, loop.steps, self.ops);
                                    }
                                    if (!NPY_UFUNC_CHECK_ERROR(loop))
                                    {
                                        goto fail;
                                    }
                                }
                                loop.bufptr[0] += NpyArray_STRIDE(loop.ret, axis);
                            }
                            NpyArray_ITER_NEXT(loop.it);
                            NpyArray_ITER_NEXT(loop.rit);
                            loop.bufptr[0] = loop.rit.dataptr;
                            loop.index++;
                        }
                    }
                    else
                    {
                        ConcurrentQueue<UFUNCLoopWorkerParams> workToDo = new ConcurrentQueue<UFUNCLoopWorkerParams>();

                        bool HasError = false;
                        Task workerThread = null;
                        bool IsCompleted = false;

                        while (loop.index < loop.size)
                        {
                            ptr = (npy_intp[])NpyArray_BYTES(ind).datap;
                            for (i = 0; i < nn; i++)
                            {
                                loop.bufptr[1] = loop.it.dataptr + ptr[i] * loop.steps[1];

                                helper.memmove_init(loop.bufptr[0], loop.bufptr[1]);
                                helper.memcpy(loop.bufptr[0].data_offset, loop.bufptr[1].data_offset, loop.outsize);

                                mm = (i == nn - 1 ? NpyArray_DIM(arr, axis) - ptr[i] : ptr[i + 1] - ptr[i]) - 1;
                                if (mm > 0)
                                {
                                    loop.bufptr[1] += loop.steps[1];
                                    loop.bufptr[2] = loop.bufptr[0];

                                    workToDo.Enqueue(new UFUNCLoopWorkerParams(operation, loop.bufptr, mm, loop.steps, self.ops));

                                    if (workerThread == null)
                                    {
                                        // start worker thread to process the queued up work
                                        workerThread = Task.Factory.StartNew(() =>
                                        {
                                            while (true)
                                            {
                                                Parallel.For(0, workToDo.Count(), xxx =>
                                                {
                                                    UFUNCLoopWorkerParams work = null;
                                                    if (workToDo.TryDequeue(out work))
                                                    {
                                                        if (UFuncHandler != null)
                                                        {
                                                            UFuncHandler(work.bufptr, work.steps, work.ops, work.N);
                                                        }
                                                        else
                                                        {
                                                            loop.function(work.op, work.bufptr, work.N, work.steps, work.ops);
                                                        }

                                                        if (!NPY_UFUNC_CHECK_ERROR(loop))
                                                        {
                                                            HasError = true;
                                                        }
                                                    }

                                                });

                                                if (workToDo.Count() == 0)
                                                {
                                                    if (IsCompleted)
                                                        break;

                                                    //System.Threading.Thread.Sleep(10);
                                                }

                                            }

                                        });
                                    }

                                }
                                loop.bufptr[0] += NpyArray_STRIDE(loop.ret, axis);
                            }
                            NpyArray_ITER_NEXT(loop.it);
                            NpyArray_ITER_NEXT(loop.rit);
                            loop.bufptr[0] = loop.rit.dataptr;
                            loop.index++;
                        }

                        IsCompleted = true;

                        if (workerThread != null)
                        {
                            workerThread.Wait();
                        }

                        if (HasError)
                            goto fail;
                    }

                    break;

                case UFuncLoopMethod.BUFFER_UFUNCLOOP:
#if BUFFER_UFUNCLOOP
                    /* Reduceat
                     * BUFFER -- misbehaved array or different types
                     */
                    /* fprintf(stderr, "BUFFERED..%d\n", loop.size); */
                    while (loop.index < loop.size)
                    {
                        ptr = (npy_intp[])NpyArray_BYTES(ind).datap;
                        for (i = 0; i < nn; i++)
                        {
                            helper.memmove_init(loop.bufptr[0], loop.idptr);
                            helper.memcpy(loop.bufptr[0].data_offset, loop.idptr.data_offset, loop.outsize);
                            n = 0;
                            mm = (i == nn - 1 ? NpyArray_DIM(arr, axis) - ptr[i] :
                                   ptr[i + 1] - ptr[i]) - 1;
                            if (mm < 1)
                            {
                                mm = 1;
                            }
                            loop.inptr = loop.it.dataptr + (ptr[i]) * loop.instrides;
                            while (n < mm)
                            {
                                /* Copy up to loop.bufsize elements to buffer */
                                dptr = loop.buffer;
                                for (j = 0; j < loop.bufsize; j++, n++)
                                {
                                    if (n == mm)
                                    {
                                        break;
                                    }
                                    helper.copyswap(dptr, loop.inptr,loop.swap);
                                    loop.inptr += loop.instrides;
                                    dptr += loop.insize;
                                }
                                if (loop.cast != null)
                                {
                                    loop.cast(loop.buffer, loop.castbuf, j,
                                               null, null);
                                }
                                loop.bufptr[2] = loop.bufptr[0];
                                loop.function(operation, loop.bufptr, j, loop.steps, self.ops);

                                if (!NPY_UFUNC_CHECK_ERROR(loop))
                                    goto fail;

                                loop.bufptr[0] += j * loop.steps[0];
                            }
                            loop.bufptr[0] += NpyArray_STRIDE(loop.ret, axis);
                        }
                        NpyArray_ITER_NEXT(loop.it);
                        NpyArray_ITER_NEXT(loop.rit);
                        loop.bufptr[0] = loop.rit.dataptr;
                        loop.index++;
                    }
#endif
                    break;
            }

            /* Hang on to this reference -- will be decref'd with loop */
            if (loop.retbase != 0)
            {
                ret = NpyArray_BASE_ARRAY(loop.ret);
                NpyArray_ForceUpdate(loop.ret);
            }
            else
            {
                ret = loop.ret;
            }
            Npy_INCREF(ret);
            ufuncreduce_dealloc(loop);
            return ret;

            fail:
            if (loop != null)
            {
                ufuncreduce_dealloc(loop);
            }
            return null;
        }
   
        private static NpyUFuncLoopObject construct_loop(NpyUFuncObject self)
        {
            NpyUFuncLoopObject loop;
            int i;
            string name;

            if (self == null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "function not supported");
                return null;
            }

            loop = new NpyUFuncLoopObject();
            if (loop == null)
            {
                NpyErr_MEMORY();
                return loop;
            }
            loop.nob_interface = null;
            loop.nob_refcnt = 1;
            loop.nob_type = null;
            loop.nob_magic_number = npy_defs.NPY_VALID_MAGIC;

            loop.iter = NpyArray_MultiIterNew();
            if (loop.iter == null)
            {
                npy_free(loop);
                return null;
            }

            loop.iter.index = 0;
            loop.iter.numiter = self.nargs;
            loop.ufunc = self;
            Npy_INCREF(loop.ufunc);
            loop.buffer[0] = null;
            for (i = 0; i < self.nargs; i++)
            {
                loop.iter.iters[i] = null;
                loop.cast[i] = null;
            }
            loop.errormask = 0;
            loop.errobj = null;
            loop.notimplemented = false;
            loop.first = 1;
            loop.core_dim_sizes = null;
            loop.core_strides = null;
            loop.leftover = 0;

            if (self.core_enabled != 0)
            {
                int num_dim_ix = 1 + self.core_num_dim_ix;
                int nstrides = self.nargs
                + self.core_offsets[self.nargs - 1]
                + self.core_num_dims[self.nargs - 1];
                loop.core_dim_sizes = new npy_intp[num_dim_ix];
                loop.core_strides = new npy_intp[nstrides];
                if (loop.core_dim_sizes == null || loop.core_strides == null)
                {
                    NpyErr_MEMORY();
                    goto fail;
                }
                memclr(new VoidPtr(loop.core_strides), nstrides);
                for (i = 0; i < num_dim_ix; i++)
                {
                    loop.core_dim_sizes[i] = 1;
                }
            }
            name = self.name != null ? self.name : "";

            return loop;

            fail:
            ufuncloop_dealloc(loop);
            return null;

        }

        private static int construct_arrays(NpyUFuncLoopObject loop, int nargs, NpyArray[] mps, int ntypenums, NPY_TYPES[] rtypenums, npy_prepare_outputs_func prepare, VoidPtr prepare_data)
        {
            int i;
            NPY_TYPES[] arg_types = new NPY_TYPES[npy_defs.NPY_MAXARGS];
            NPY_SCALARKIND[] scalars = new NPY_SCALARKIND[npy_defs.NPY_MAXARGS];
            NPY_SCALARKIND maxarrkind, maxsckind, newArr;
            NpyUFuncObject self = loop.ufunc;
            bool allscalars = true;
            bool flexible = false;
            bool isobject = false;

            npy_intp[] temp_dims = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp[] out_dims;
            int out_nd = 0;

            /* Get each input argument */
            maxarrkind = NPY_SCALARKIND.NPY_NOSCALAR;
            maxsckind = NPY_SCALARKIND.NPY_NOSCALAR;
            for (i = 0; i < self.nin; i++)
            {
                arg_types[i] = NpyArray_TYPE(mps[i]);
                if (!flexible && NpyTypeNum_ISFLEXIBLE(arg_types[i]))
                {
                    flexible = true;
                }
                if (!isobject && NpyTypeNum_ISOBJECT(arg_types[i]))
                {
                    isobject = true;
                }
                /*
                 * debug
                 * fprintf(stderr, "array %d has reference %d\n", i,
                 * (mps[i]).ob_refcnt);
                 */

                /*
                 * Scalars are 0-dimensional arrays at this point
                 */

                /*
                 * We need to keep track of whether or not scalars
                 * are mixed with arrays of different kinds.
                 */

                if (NpyArray_NDIM(mps[i]) > 0)
                {
                    scalars[i] = NPY_SCALARKIND.NPY_NOSCALAR;
                    allscalars = false;
                    newArr = NpyArray_ScalarKind(arg_types[i], null);
                    maxarrkind = NpyArray_MAX(newArr, maxarrkind);
                }
                else
                {
                    scalars[i] = NpyArray_ScalarKind(arg_types[i], mps[i]);
                    maxsckind = NpyArray_MAX(scalars[i], maxsckind);
                }
            }

            /* We don't do strings */
            if (flexible && !isobject)
            {
                loop.notimplemented = true;
                return nargs;
            }

            /*
             * If everything is a scalar, or scalars mixed with arrays of
             * different kinds of lesser kinds then use normal coercion rules
             */
            if (allscalars || (maxsckind > maxarrkind))
            {
                for (i = 0; i < self.nin; i++)
                {
                    scalars[i] = NPY_SCALARKIND.NPY_NOSCALAR;
                }
            }

            /* Select an appropriate function for these argument types. */
            if (select_types(loop.ufunc, arg_types, ref loop.function,
                             scalars, ntypenums,
                             rtypenums) == -1)
            {
                return -1;
            }

            /*
             * Create copies for some of the arrays if they are small
             * enough and not already contiguous
             */
            if (_create_copies(loop, arg_types, mps) < 0)
            {
                return -1;
            }

            /*
             * Only use loop dimensions when constructing Iterator:
             * temporarily replace mps[i] (will be recovered below).
             */
            if (self.core_enabled != 0)
            {
                for (i = 0; i < self.nin; i++)
                {
                    NpyArray ao;

                    if (_compute_dimension_size(loop, mps, i) < 0)
                    {
                        return -1;
                    }
                    ao = _trunc_coredim(mps[i], self.core_num_dims[i]);
                    if (ao == null)
                    {
                        return -1;
                    }
                    mps[i] = ao;
                }
            }

            /* Create Iterators for the Inputs */
            for (i = 0; i < self.nin; i++)
            {
                loop.iter.iters[i] = NpyArray_IterNew(mps[i]);
                if (loop.iter.iters[i] == null)
                {
                    return -1;
                }
            }

            /* Recover mps[i]. */
            if (self.core_enabled != 0)
            {
                for (i = 0; i < self.nin; i++)
                {
                    NpyArray ao = mps[i];
                    mps[i] = NpyArray_BASE_ARRAY(mps[i]);
                    Npy_DECREF(ao);
                }
            }

            /* Broadcast the result */
            loop.iter.numiter = self.nin;
            if (NpyArray_Broadcast(loop.iter) < 0)
            {
                return -1;
            }

            /* Get any return arguments */
            for (i = self.nin; i < nargs; i++)
            {
                if (self.core_enabled != 0)
                {
                    if (_compute_dimension_size(loop, mps, i) < 0)
                    {
                        return -1;
                    }
                }
                out_dims = _compute_output_dims(loop, i, ref out_nd, temp_dims);
                if (out_dims == null)
                {
                    return -1;
                }
                if (null != mps[i] && (NpyArray_NDIM(mps[i]) != out_nd ||
                                       !NpyArray_CompareLists(NpyArray_DIMS(mps[i]),
                                                              out_dims, out_nd)))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "invalid return array shape");
                    Npy_DECREF(mps[i]);
                    mps[i] = null;
                    return -1;
                }
                if (null != mps[i] && !NpyArray_ISWRITEABLE(mps[i]))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "return array is not writeable");
                    Npy_DECREF(mps[i]);
                    mps[i] = null;
                    return -1;
                }
            }

            /* construct any missing return arrays and make output iterators */
            for (i = self.nin; i < self.nargs; i++)
            {
                NpyArray_Descr ntype;

                if (mps[i] == null)
                {
                    out_dims = _compute_output_dims(loop, i, ref out_nd, temp_dims);
                    if (out_dims == null)
                    {
                        return -1;
                    }
                    mps[i] = NpyArray_New(null,
                                          out_nd,
                                          out_dims,
                                          arg_types[i],
                                          null, null,
                                          0, 0, null);
                    if (mps[i] == null)
                    {
                        return -1;
                    }
                }

                /*
                 * reset types for outputs that are equivalent
                 * -- no sense casting uselessly
                 */
                else
                {
                    if (NpyArray_TYPE(mps[i]) != arg_types[i])
                    {
                        NpyArray_Descr atype;
                        ntype = NpyArray_DESCR(mps[i]);
                        atype = NpyArray_DescrFromType(arg_types[i]);
                        if (NpyArray_EquivTypes(atype, ntype))
                        {
                            arg_types[i] = ntype.type_num;
                        }
                        Npy_DECREF(atype);
                    }

                    /* still not the same -- or will we have to use buffers?*/
                    if (NpyArray_TYPE(mps[i]) != arg_types[i]
                        || !NpyArray_ISBEHAVED_RO(mps[i]))
                    {
                        if (loop.iter.size < loop.bufsize || self.core_enabled != 0)
                        {
                            NpyArray newArr2;
                            /*
                             * Copy the array to a temporary copy
                             * and set the UPDATEIFCOPY flag
                             */
                            ntype = NpyArray_DescrFromType(arg_types[i]);
                            newArr2 = NpyArray_FromArray(mps[i], ntype,
                                     NPYARRAYFLAGS.NPY_FORCECAST | NPYARRAYFLAGS.NPY_ALIGNED | NPYARRAYFLAGS.NPY_UPDATEIFCOPY);
                            if (newArr2 == null)
                            {
                                return -1;
                            }
                            Npy_DECREF(mps[i]);
                            mps[i] = newArr2;
                        }
                    }
                }

                if (self.core_enabled != 0)
                {
                    NpyArray ao;

                    /* computer for all output arguments, and set strides in "loop" */
                    if (_compute_dimension_size(loop, mps, i) < 0)
                    {
                        return -1;
                    }
                    ao = _trunc_coredim(mps[i], self.core_num_dims[i]);
                    if (ao == null)
                    {
                        return -1;
                    }
                    /* Temporarily modify mps[i] for constructing iterator. */
                    mps[i] = ao;
                }

                loop.iter.iters[i] = NpyArray_IterNew(mps[i]);
                if (loop.iter.iters[i] == null)
                {
                    return -1;
                }

                /* Recover mps[i]. */
                if (self.core_enabled != 0)
                {
                    NpyArray ao = mps[i];
                    mps[i] = NpyArray_BASE_ARRAY(mps[i]);
                    Npy_DECREF(ao);
                }

            }

            /* wrap outputs */
            if (prepare != null)
            {
                if (prepare(self, ref mps[0], prepare_data) < 0)
                {
                    return -1;
                }
            }

            /*
             * If any of different type, or misaligned or swapped
             * then must use buffers
             */
            loop.bufcnt = 0;
            loop.obj = 0;
            /* Determine looping method needed */
            loop.meth = UFuncLoopMethod.NO_UFUNCLOOP;
            if (loop.iter.size == 0)
            {
                return nargs;
            }
            if (self.core_enabled != 0)
            {
                loop.meth = UFuncLoopMethod.SIGNATURE_NOBUFFER_UFUNCLOOP;
            }
            for (i = 0; i < self.nargs; i++)
            {
                loop.needbuffer[i] = false;
                if (arg_types[i] != NpyArray_TYPE(mps[i])
                    || !NpyArray_ISBEHAVED_RO(mps[i]))
                {
                    if (self.core_enabled != 0)
                    {
                        NpyErr_SetString(npyexc_type.NpyExc_RuntimeError,
                                        "never reached; copy should have been made");
                        return -1;
                    }
                    loop.meth = UFuncLoopMethod.BUFFER_UFUNCLOOP;
                    loop.needbuffer[i] = true;
                }
 
            }

            if (loop.meth == UFuncLoopMethod.NO_UFUNCLOOP)
            {
                loop.meth = UFuncLoopMethod.ONE_UFUNCLOOP;

                /* All correct type and BEHAVED */
                /* Check for non-uniform stridedness */
                for (i = 0; i < self.nargs; i++)
                {
                    if (!(loop.iter.iters[i].contiguous))
                    {
                        /*
                         * May still have uniform stride
                         * if (broadcast result) <= 1-d
                         */
                        if (NpyArray_NDIM(mps[i]) != 0 &&
                            (loop.iter.iters[i].nd_m1 > 0))
                        {
                            loop.meth = UFuncLoopMethod.NOBUFFER_UFUNCLOOP;
                            break;
                        }
                    }
                }
                if (loop.meth == UFuncLoopMethod.ONE_UFUNCLOOP)
                {
                    for (i = 0; i < self.nargs; i++)
                    {
                        loop.bufptr[i] = NpyArray_BYTES(mps[i]);
                    }
                }
            }

            loop.iter.numiter = self.nargs;

            /* Fill in steps  */
            if (loop.meth == UFuncLoopMethod.SIGNATURE_NOBUFFER_UFUNCLOOP && loop.iter.nd == 0)
            {
                /* Use default core_strides */
            }
            else if (loop.meth != UFuncLoopMethod.ONE_UFUNCLOOP)
            {
                int ldim;
                npy_intp minsum;
                npy_intp maxdim;
                NpyArrayIterObject it;
                npy_intp[] stride_sum = new npy_intp[npy_defs.NPY_MAXDIMS];
                int j;

                /* Fix iterators */

                /*
                 * Optimize axis the iteration takes place over
                 *
                 * The first thought was to have the loop go
                 * over the largest dimension to minimize the number of loops
                 *
                 * However, on processors with slow memory bus and cache,
                 * the slowest loops occur when the memory access occurs for
                 * large strides.
                 *
                 * Thus, choose the axis for which strides of the last iterator is
                 * smallest but non-zero.
                 */
                for (i = 0; i < loop.iter.nd; i++)
                {
                    stride_sum[i] = 0;
                    for (j = 0; j < loop.iter.numiter; j++)
                    {
                        stride_sum[i] += loop.iter.iters[j].strides[i];
                    }
                }

                ldim = loop.iter.nd - 1;
                minsum = stride_sum[loop.iter.nd - 1];
                for (i = loop.iter.nd - 2; i >= 0; i--)
                {
                    if (stride_sum[i] < minsum)
                    {
                        ldim = i;
                        minsum = stride_sum[i];
                    }
                }
                maxdim = loop.iter.dimensions[ldim];
                loop.iter.size /= maxdim;
                loop.bufcnt = maxdim;
                loop.lastdim = ldim;

                /*
                 * Fix the iterators so the inner loop occurs over the
                 * largest dimensions -- This can be done by
                 * setting the size to 1 in that dimension
                 * (just in the iterators)
                 */
                for (i = 0; i < loop.iter.numiter; i++)
                {
                    it = loop.iter.iters[i];
                    it.contiguous = false;
                    it.size /= (it.dims_m1[ldim] + 1);
                    it.dims_m1[ldim] = 0;
                    it.backstrides[ldim] = 0;

                    /*
                     * (won't fix factors because we
                     * don't use PyArray_ITER_GOTO1D
                     * so don't change them)
                     *
                     * Set the steps to the strides in that dimension
                     */
                    loop.steps[i] = it.strides[ldim];
                }

                /*
                 * Set looping part of core_dim_sizes and core_strides.
                 */
                if (loop.meth == UFuncLoopMethod.SIGNATURE_NOBUFFER_UFUNCLOOP)
                {
                    loop.core_dim_sizes[0] = maxdim;
                    for (i = 0; i < self.nargs; i++)
                    {
                        loop.core_strides[i] = loop.steps[i];
                    }
                }

                /*
                 * fix up steps where we will be copying data to
                 * buffers and calculate the ninnerloops and leftover
                 * values -- if step size is already zero that is not changed...
                 */
                if (loop.meth == UFuncLoopMethod.BUFFER_UFUNCLOOP)
                {
                    loop.leftover = (int)(maxdim % loop.bufsize);
                    loop.ninnerloops = (int)((maxdim / loop.bufsize) + 1);
                    for (i = 0; i < self.nargs; i++)
                    {
                        if (loop.needbuffer[i] && loop.steps[i] != 0)
                        {
                            loop.steps[i] = NpyArray_ITEMSIZE(mps[i]);
                        }
                        /* These are changed later if casting is needed */
                    }
                }
            }
            else if (loop.meth == UFuncLoopMethod.ONE_UFUNCLOOP)
            {
                /* uniformly-strided case */
                for (i = 0; i < self.nargs; i++)
                {
                    if (NpyArray_SIZE(mps[i]) == 1)
                    {
                        loop.steps[i] = 0;
                    }
                    else
                    {
                        loop.steps[i] = NpyArray_STRIDE(mps[i], NpyArray_NDIM(mps[i]) - 1);
                    }
                }
            }

            /* Finally, create memory for buffers if we need them */

            /*
             * Buffers for scalars are specially made small -- scalars are
             * not copied multiple times
             */
            if (loop.meth == UFuncLoopMethod.BUFFER_UFUNCLOOP)
            {
                int cnt = 0, cntcast = 0;
                int scnt = 0, scntcast = 0;
                VoidPtr castptr;
                VoidPtr bufptr;
                bool last_was_scalar = false;
                bool last_cast_was_scalar = false;
                int oldbufsize = 0;
                int oldsize = 0;
                int scbufsize = 4 * sizeof(double);
                int memsize;
                NpyArray_Descr descr;

                /* compute the element size */
                for (i = 0; i < self.nargs; i++)
                {
                    if (!loop.needbuffer[i])
                    {
                        continue;
                    }
                    if (arg_types[i] != mps[i].descr.type_num)
                    {
                        descr = NpyArray_DescrFromType(arg_types[i]);
                        if (loop.steps[i] != 0)
                        {
                            cntcast += descr.elsize;
                        }
                        else
                        {
                            scntcast += descr.elsize;
                        }
                        if (i < self.nin)
                        {
                            loop.cast[i] = NpyArray_GetCastFunc(NpyArray_DESCR(mps[i]),
                                                                 arg_types[i]);
                        }
                        else
                        {
                            loop.cast[i] = NpyArray_GetCastFunc(
                                descr, NpyArray_DESCR(mps[i]).type_num);
                        }
                        Npy_DECREF(descr);
                        if (loop.cast[i] == null)
                        {
                            return -1;
                        }
                    }
                    loop.swap[i] = !(NpyArray_ISNOTSWAPPED(mps[i]));
                    if (loop.steps[i] != 0)
                    {
                        cnt += NpyArray_ITEMSIZE(mps[i]);
                    }
                    else
                    {
                        scnt += NpyArray_ITEMSIZE(mps[i]);
                    }
                }
                memsize = loop.bufsize * (cnt + cntcast) + scbufsize * (scnt + scntcast);
                loop.buffer[0] = NpyDataMem_NEW(NPY_TYPES.NPY_UBYTE, (ulong)memsize);

                /*
                 * debug
                 * fprintf(stderr, "Allocated buffer at %p of size %d, cnt=%d, cntcast=%d\n",
                 *               loop->buffer[0], loop->bufsize * (cnt + cntcast), cnt, cntcast);
                 */
                if (loop.buffer[0] == null)
                {
                    NpyErr_MEMORY();
                    return -1;
                }
 
                castptr = loop.buffer[0] + loop.bufsize * cnt + scbufsize * scnt;
                bufptr = loop.buffer[0];
                loop.objfunc = 0;
                for (i = 0; i < self.nargs; i++)
                {
                    if (!loop.needbuffer[i])
                    {
                        continue;
                    }
                    loop.buffer[i] = bufptr + (last_was_scalar ? scbufsize :
                                                loop.bufsize) * oldbufsize;
                    last_was_scalar = (loop.steps[i] == 0);
                    bufptr = loop.buffer[i];
                    oldbufsize = NpyArray_ITEMSIZE(mps[i]);
                    /* fprintf(stderr, "buffer[%d] = %p\n", i, loop->buffer[i]); */
                    if (loop.cast[i] != null)
                    {
                        NpyArray_Descr descr2;
                        loop.castbuf[i] = castptr + (last_cast_was_scalar ? scbufsize :
                                                      loop.bufsize) * oldsize;
                        last_cast_was_scalar = last_was_scalar;
                        /* fprintf(stderr, "castbuf[%d] = %p\n", i, loop->castbuf[i]); */
                        descr2 = NpyArray_DescrFromType(arg_types[i]);
                        oldsize = descr2.elsize;
                        Npy_DECREF(descr2);
                        loop.bufptr[i] = loop.castbuf[i];
                        castptr = loop.castbuf[i];
                        if (loop.steps[i] != 0)
                        {
                            loop.steps[i] = oldsize;
                        }
                    }
                    else
                    {
                        loop.bufptr[i] = loop.buffer[i];
                    }
   
                }


                /* compute the element size */
                //for (i = 0; i < self.nargs; i++)
                //{
                //    if (loop.needbuffer[i])
                //    {
                //        int oldsize = mps[i].ItemSize;
                //        mps[i] = NpyArray_CastToType(mps[i], NpyArray_DescrFromType(arg_types[i]), NpyArray_ISFORTRAN(mps[i]));
                //        loop.needbuffer[i] = false;
                //        if (loop.steps[i] != 0)
                //            loop.steps[i]  = oldsize;
                //        continue;
                //    }
                //}

            }

            return nargs;
        }

        static NpyUFuncReduceObject construct_reduce(NpyUFuncObject self, ref NpyArray arr, NpyArray _out,
          int axis, NPY_TYPES otype, GenericReductionOp operation, npy_intp ind_size)
        {
            NpyUFuncReduceObject loop;
            NpyArray idarr;
            NpyArray aar;
            npy_intp[] loop_i = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp outsize = 0;
            NPY_TYPES[] arg_types = new NPY_TYPES[3];
            NPY_SCALARKIND[] scalars = new NPY_SCALARKIND[]
                { NPY_SCALARKIND.NPY_NOSCALAR, NPY_SCALARKIND.NPY_NOSCALAR, NPY_SCALARKIND.NPY_NOSCALAR };

            string name = self.name;
            int i, j, nd;
            NPYARRAYFLAGS flags;

            Debug.Assert(Validate(self));

            /* Reduce type is the type requested of the input during reduction */
            if (self.core_enabled != 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_RuntimeError, "construct_reduce not allowed on ufunc with signature");
                return null;
            }

            nd = NpyArray_NDIM(arr);
            arg_types[0] = otype;
            arg_types[1] = otype;
            arg_types[2] = otype;

            loop = new NpyUFuncReduceObject();
            if (loop == null)
            {
                NpyErr_MEMORY();
                return loop;
            }
            loop.nob_magic_number = (UInt32)npy_defs.NPY_VALID_MAGIC;

            loop.retbase = 0;
            loop.swap = false;
            loop.index = 0;
            loop.ufunc = self;
            Npy_INCREF(self);
            loop.cast = null;
            loop.buffer = null;
            loop.ret = null;
            loop.it = null;
            loop.rit = null;
            loop.errobj = null;
            loop.first = 1;
            loop.decref_arr = null;
            loop.N = NpyArray_DIM(arr, axis);
            loop.instrides = NpyArray_STRIDE(arr, axis);


            default_fp_error_state(loop);
            if (select_types(loop.ufunc, arg_types, ref loop.function,
                             scalars, 0, null) == -1)
            {
                goto fail;
            }

            /*
             * output type may change -- if it does
             * reduction is forced into that type
             * and we need to select the reduction function again
             */
            if (otype != arg_types[2])
            {
                otype = arg_types[2];
                arg_types[0] = otype;
                arg_types[1] = otype;
                if (select_types(loop.ufunc, arg_types, ref loop.function,
                                 scalars, 0, null) == -1)
                {
                    goto fail;
                }
            }

            /* Make copy if misbehaved or not otype for small arrays */
            if (_create_reduce_copy(loop, ref arr, otype) < 0)
            {
                goto fail;
            }
            aar = arr;

            if (loop.N == 0)
            {
                loop.meth = UFuncLoopMethod.ZERO_EL_REDUCELOOP;
            }
            else if (NpyArray_ISBEHAVED_RO(aar) && (otype == NpyArray_TYPE(aar)))
            {
                if (loop.N == 1)
                {
                    loop.meth = UFuncLoopMethod.ONE_EL_REDUCELOOP;
                }
                else
                {
                    loop.meth = UFuncLoopMethod.NOBUFFER_UFUNCLOOP;
                    loop.steps[1] = NpyArray_STRIDE(aar, axis);
                    loop.N -= 1;
                }
            }
            else
            {
                loop.meth = UFuncLoopMethod.BUFFER_UFUNCLOOP;
                loop.swap = !(NpyArray_ISNOTSWAPPED(aar));
            }

            loop.obj = 0;

            if ((loop.meth == UFuncLoopMethod.ZERO_EL_REDUCELOOP)
                || ((operation == GenericReductionOp.NPY_UFUNC_REDUCEAT)
                    && (loop.meth == UFuncLoopMethod.BUFFER_UFUNCLOOP)))
            {
                idarr = _getidentity(self, otype, name);
                if (idarr == null)
                {
                    goto fail;
                }

                if (NpyArray_ITEMSIZE(idarr) > (int)UFuncsOptions.NPY_UFUNC_MAXIDENTITY)
                {
                    string buf = string.Format("UFUNC_MAXIDENTITY ({0}) is too small (needs to be at least {1})",
                                   UFuncsOptions.NPY_UFUNC_MAXIDENTITY, NpyArray_ITEMSIZE(idarr));
                    NpyErr_SetString(npyexc_type.NpyExc_RuntimeError, buf);

                    Npy_DECREF(idarr);
                    goto fail;
                }

                var helper = MemCopy.GetMemcopyHelper(loop.idptr);
                var idarrvp = NpyArray_BYTES(idarr);
                helper.memmove_init(loop.idptr, idarrvp);
                helper.memcpy(loop.idptr.data_offset, idarrvp.data_offset, NpyArray_ITEMSIZE(idarr));
                Npy_DECREF(idarr);
            }

            /* Construct return array */
            flags = NPYARRAYFLAGS.NPY_CARRAY | NPYARRAYFLAGS.NPY_UPDATEIFCOPY | NPYARRAYFLAGS.NPY_FORCECAST;
            switch (operation)
            {
                case GenericReductionOp.NPY_UFUNC_REDUCE:
                    for (j = 0, i = 0; i < nd; i++)
                    {
                        if (i != axis)
                        {
                            loop_i[j++] = NpyArray_DIM(aar, i);
                        }
                    }
                    if (_out == null)
                    {
                        loop.ret = NpyArray_New(null, NpyArray_NDIM(aar) - 1, loop_i,
                                                 otype, null, null, 0, 0,
                                                 Npy_INTERFACE(aar));
                    }
                    else
                    {
                        outsize = NpyArray_MultiplyList(loop_i, NpyArray_NDIM(aar) - 1);
                    }
                    break;
                case GenericReductionOp.NPY_UFUNC_ACCUMULATE:
                    if (_out == null)
                    {
                        loop.ret = NpyArray_New(null, NpyArray_NDIM(aar),
                                                 NpyArray_DIMS(aar),
                                                 otype, null, null, 0, 0,
                                                 Npy_INTERFACE(aar));
                    }
                    else
                    {
                        outsize = NpyArray_MultiplyList(NpyArray_DIMS(aar),
                                                        NpyArray_NDIM(aar));
                    }
                    break;
                case GenericReductionOp.NPY_UFUNC_REDUCEAT:
                    copydims(loop_i, NpyArray_DIMS(aar), nd);
                    /* Index is 1-d array */
                    loop_i[axis] = ind_size;
                    if (_out == null)
                    {
                        loop.ret = NpyArray_New(null, NpyArray_NDIM(aar), loop_i, otype,
                                                 null, null, 0, 0, Npy_INTERFACE(aar));
                    }
                    else
                    {
                        outsize = NpyArray_MultiplyList(loop_i, NpyArray_NDIM(aar));
                    }
                    if (ind_size == 0)
                    {
                        loop.meth = UFuncLoopMethod.ZERO_EL_REDUCELOOP;
                        return loop;
                    }
                    if (loop.meth == UFuncLoopMethod.ONE_EL_REDUCELOOP)
                    {
                        loop.meth = UFuncLoopMethod.NOBUFFER_REDUCELOOP;
                    }
                    break;
            }
            if (_out != null)
            {
                if (NpyArray_SIZE(_out) != outsize)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                     "wrong shape for output");
                    goto fail;
                }
                loop.ret = NpyArray_FromArray(_out, NpyArray_DescrFromType(otype),
                                               flags);
                if ((loop.ret != null) && (loop.ret != _out))
                {
                    loop.retbase = 1;
                }
            }
            if (loop.ret == null)
            {
                goto fail;
            }
            loop.insize = NpyArray_ITEMSIZE(aar);
            loop.outsize = NpyArray_ITEMSIZE(loop.ret);
            loop.bufptr[0] = NpyArray_BYTES(loop.ret);

            if (loop.meth == UFuncLoopMethod.ZERO_EL_REDUCELOOP)
            {
                loop.size = NpyArray_SIZE(loop.ret);
                return loop;
            }

            loop.it = NpyArray_IterNew(aar);
            if (loop.it == null)
            {
                return null;
            }
            if (loop.meth == UFuncLoopMethod.ONE_EL_REDUCELOOP)
            {
                loop.size = loop.it.size;
                return loop;
            }

            /*
             * Fix iterator to loop over correct dimension
             * Set size in axis dimension to 1
             */
            loop.it.contiguous = false;
            loop.it.size /= (loop.it.dims_m1[axis] + 1);
            loop.it.dims_m1[axis] = 0;
            loop.it.backstrides[axis] = 0;
            loop.size = loop.it.size;
            if (operation == GenericReductionOp.NPY_UFUNC_REDUCE)
            {
                loop.steps[0] = 0;
            }
            else
            {
                loop.rit = NpyArray_IterNew(loop.ret);
                if (loop.rit == null)
                {
                    return null;
                }
                /*
                 * Fix iterator to loop over correct dimension
                 * Set size in axis dimension to 1
                 */
                loop.rit.contiguous = false;
                loop.rit.size /= (loop.rit.dims_m1[axis] + 1);
                loop.rit.dims_m1[axis] = 0;
                loop.rit.backstrides[axis] = 0;

                if (operation == GenericReductionOp.NPY_UFUNC_ACCUMULATE)
                {
                    loop.steps[0] = NpyArray_STRIDE(loop.ret, axis);
                }
                else
                {
                    loop.steps[0] = 0;
                }
            }
            loop.steps[2] = loop.steps[0];
            loop.bufptr[2] = loop.bufptr[0] + loop.steps[2];
            if (loop.meth == UFuncLoopMethod.BUFFER_UFUNCLOOP)
            {
                npy_intp _size;

                loop.steps[1] = loop.outsize;
                if (otype != NpyArray_TYPE(aar))
                {
                    _size = loop.bufsize * (loop.outsize + NpyArray_ITEMSIZE(aar)) / GetTypeSize(otype);
                    loop.buffer = NpyDataMem_NEW(otype, (ulong)_size);
                    if (loop.buffer == null)
                    {
                        goto fail;
                    }

                    loop.castbuf = loop.buffer + loop.bufsize * NpyArray_ITEMSIZE(aar);
                    loop.bufptr[1] = loop.castbuf;
                    loop.cast = NpyArray_GetCastFunc(NpyArray_DESCR(aar), otype);
                    if (loop.cast == null)
                    {
                        goto fail;
                    }
                }
                else
                {
                    _size = (loop.bufsize * loop.outsize) / GetTypeSize(otype);
                    loop.buffer = NpyDataMem_NEW(otype, (ulong)_size);
                    if (loop.buffer == null)
                    {
                        goto fail;
                    }

                    loop.bufptr[1] = loop.buffer;
                }
            }
            NpyUFunc_clearfperr();
            return loop;

            fail:
            ufuncreduce_dealloc(loop);
            return null;
        }

        static string _types_msg = "function not supported for these types, and can't coerce safely to supported types";


        /*
         * Called to determine coercion
         * Can change arg_types.
         */
        static int select_types(NpyUFuncObject self, NPY_TYPES[] arg_types,
                     ref NpyUFuncGenericFunction function,
                     NPY_SCALARKIND[] scalars,
                     int ntypenums, NPY_TYPES[] rtypenums)
        {
            int i, j;
            NPY_TYPES start_type;
            int userdef = -1;
            int userdef_ind = -1;

            if (self.userloops != null)
            {
                for (i = 0; i < self.nin; i++)
                {
                    if (NpyTypeNum_ISUSERDEF(arg_types[i]))
                    {
                        userdef = (int)arg_types[i];
                        userdef_ind = i;
                        break;
                    }
                }
            }

            if (rtypenums != null)
            {
                return extract_specified_loop(self, arg_types, function,
                                              ntypenums, rtypenums, userdef);
            }

            if (userdef > 0)
            {
                int ret = -1;

                /*
                 * Look through all the registered loops for all the user-defined
                 * types to find a match.
                 */
                while (ret == -1)
                {
                    NpyUFunc_Loop1d funcdata;
                    int userdefP;

                    if (userdef_ind >= self.nin)
                    {
                        break;
                    }
                    userdef = (int)arg_types[userdef_ind++];
                    if (!(NpyTypeNum_ISUSERDEF((NPY_TYPES)userdef)))
                    {
                        continue;
                    }
                    userdefP = userdef;
                    funcdata = (NpyUFunc_Loop1d)NpyDict_Get(self.userloops, userdefP);
                    /*
                     * extract the correct function
                     * data and argtypes for this user-defined type.
                     */
                    ret = _find_matching_userloop(funcdata, arg_types, scalars,
                                                  function, self.nargs,
                                                  self.nin);
                }
                if (ret == 0)
                {
                    return ret;
                }
                NpyErr_SetString(npyexc_type.NpyExc_TypeError, _types_msg);
                return ret;
            }

            start_type = arg_types[0];
            /*
             * If the first argument is a scalar we need to place
             * the start type as the lowest type in the class
             */
            if (scalars[0] != NPY_SCALARKIND.NPY_NOSCALAR)
            {
                start_type = _lowest_type(start_type);
            }

            i = 0;
            while (i < self.ntypes && start_type > self.types[i])
            {
                i++;
            }

            int start_ntype = i;

            for (j = 0; j < self.nin; j++)
            {
                for (i = start_ntype; i < self.ntypes; i++)
                {

                    if (NpyArray_CanCoerceScalar(arg_types[j], self.types[i], scalars[j]))
                    {
                        arg_types[j] = self.types[i];
                        break;
                    }
                }

                if (i >= self.ntypes)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_TypeError, _types_msg);
                    return -1;
                }

            }

            function = self.GetFunction(i);

            return 0;
        }


        private static int extract_specified_loop(NpyUFuncObject self, NPY_TYPES[] arg_types, 
            NpyUFuncGenericFunction function, int ntypenums, NPY_TYPES[] rtypenums, int userdef)
        {
            string msg = "loop written to specified type(s) not found";
            int nargs;
            int i, j;

            nargs = self.nargs;
            if (userdef > 0)
            {
                /* search in the user-defined functions */
                NpyUFunc_Loop1d funcdata;

                funcdata = (NpyUFunc_Loop1d)NpyDict_Get(self.userloops, userdef);
                if (null == funcdata)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_TypeError, "user-defined type used in ufunc with no registered loops");
                    return -1;
                }
                /*
                 * extract the correct function
                 * data and argtypes
                 */
                while (funcdata != null)
                {
                    if (ntypenums == 1)
                    {
                        if (rtypenums[0] == funcdata.arg_types[self.nin])
                        {
                            i = nargs;
                        }
                        else
                        {
                            i = -1;
                        }
                    }
                    else
                    {
                        for (i = 0; i < nargs; i++)
                        {
                            if (rtypenums[i] != funcdata.arg_types[i])
                            {
                                break;
                            }
                        }
                    }
                    if (i == nargs)
                    {
                        function = funcdata.func;
                        for (i = 0; i < nargs; i++)
                        {
                            arg_types[i] = funcdata.arg_types[i];
                        }
                        return 0;
                    }
                    funcdata = funcdata.next;
                }
                NpyErr_SetString(npyexc_type.NpyExc_TypeError, msg);
                return -1;
            }

            /* look for match in self.functions */
            for (j = 0; j < self.ntypes; j++)
            {
                if (rtypenums[0] == self.types[j * nargs + self.nin])
                {
                    i = nargs;
                }
                else
                {
                    i = -1;
                }
                if (i == nargs)
                {
                    function = self.GetFunction(j);
                    for (i = 0; i < nargs; i++)
                    {
                        arg_types[i] = self.types[j * nargs + i];
                    }
                    return 0;
                }
            }
            NpyErr_SetString(npyexc_type.NpyExc_TypeError, msg);

            return -1;

        }

        void npy_ufunc_dealloc(NpyUFuncObject self)
        {
            if (self.core_num_dims != null)
            {
                npy_free(self.core_num_dims);
            }
            if (self.core_dim_ixs != null)
            {
                npy_free(self.core_dim_ixs);
            }
            if (self.core_offsets != null)
            {
                npy_free(self.core_offsets);
            }
            if (self.core_signature != null)
            {
                npy_free(self.core_signature);
            }
            if (self.ptr != null)
            {
                npy_free(self.ptr);
            }
            if (null != self.userloops)
            {
                NpyDict_Destroy(self.userloops);
            }
            self.nob_magic_number = npy_defs.NPY_INVALID_MAGIC;
            npy_free(self);
        }


        private static void ufuncloop_dealloc(NpyUFuncLoopObject self)
        {
            if (self.ufunc != null)
            {
                if (self.core_dim_sizes != null)
                {
                    npy_free(self.core_dim_sizes);
                }
                if (self.core_strides != null)
                {
                    npy_free(self.core_strides);
                }
                self.iter.numiter = self.ufunc.nargs;
                Npy_DECREF(self.iter);
                if (self.buffer[0] != null)
                {
                    NpyDataMem_FREE(self.buffer[0]);
                }
                NpyInterface_DECREF(self.errobj);
                Npy_DECREF(self.ufunc);
            }

            self.nob_magic_number = npy_defs.NPY_INVALID_MAGIC;
            npy_free(self);
        }

        static void ufuncreduce_dealloc(NpyUFuncReduceObject self)
        {
            if (self.ufunc != null)
            {
                Npy_XDECREF(self.it);
                Npy_XDECREF(self.rit);
                Npy_XDECREF(self.ret);
                NpyInterface_DECREF(self.errobj);
                Npy_XDECREF(self.decref_arr);
                if (self.buffer != null)
                {
                    NpyDataMem_FREE(self.buffer);
                }
                Npy_DECREF(self.ufunc);
            }
            self.nob_magic_number = (UInt32)npy_defs.NPY_INVALID_MAGIC;
            npy_free(self);
        }


        private static int _create_copies(NpyUFuncLoopObject loop, NPY_TYPES[] arg_types, NpyArray[] mps)
        {
            int nin = loop.ufunc.nin;
            int i;
            npy_intp size;
            NpyArray_Descr ntype;
            NpyArray_Descr atype;

            for (i = 0; i < nin; i++)
            {
                size = NpyArray_SIZE(mps[i]);
                /*
                 * if the type of mps[i] is equivalent to arg_types[i]
                 * then set arg_types[i] equal to type of mps[i] for later checking....
                 */
                if (NpyArray_TYPE(mps[i]) != arg_types[i])
                {
                    ntype = mps[i].descr;
                    atype = NpyArray_DescrFromType(arg_types[i]);
                    if (NpyArray_EquivTypes(atype, ntype))
                    {
                        arg_types[i] = ntype.type_num;
                    }
                    Npy_DECREF(atype);
                }
                if (true ) //size < loop.bufsize || loop.ufunc.core_enabled != 0)
                {
                    if (!(NpyArray_ISBEHAVED_RO(mps[i]))
                        || NpyArray_TYPE(mps[i]) != arg_types[i])
                    {
                        NpyArray newArray;
                        ntype = NpyArray_DescrFromType(arg_types[i]);

                        /* Move reference to interface. */
                        newArray = NpyArray_FromArray(mps[i], ntype,
                                                  NPYARRAYFLAGS.NPY_FORCECAST | NPYARRAYFLAGS.NPY_ALIGNED);
                        if (newArray == null)
                        {
                            return -1;
                        }
                        Npy_DECREF(mps[i]);
                        mps[i] = newArray;
                    }
                }
            }
            return 0;
        }


        private static int _compute_dimension_size(NpyUFuncLoopObject loop, NpyArray[] mps, int i)
        {
            NpyUFuncObject ufunc = loop.ufunc;
            int j = ufunc.core_offsets[i];
            int k = NpyArray_NDIM(mps[i]) - ufunc.core_num_dims[i];
            int ind;

            for (ind = 0; ind < ufunc.core_num_dims[i]; ind++, j++, k++)
            {
                npy_intp dim = k < 0 ? 1 : NpyArray_DIM(mps[i], k);
                /* First element of core_dim_sizes will be used for looping */
                int dim_ix = ufunc.core_dim_ixs[j] + 1;
                if (loop.core_dim_sizes[dim_ix] == 1)
                {
                    /* broadcast core dimension  */
                    loop.core_dim_sizes[dim_ix] = dim;
                }
                else if (dim != 1 && dim != loop.core_dim_sizes[dim_ix])
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "core dimensions mismatch");
                    return -1;
                }
                /* First ufunc.nargs elements will be used for looping */
                loop.core_strides[ufunc.nargs + j] = dim == 1 ? 0 : NpyArray_STRIDE(mps[i], k);
            }
            return 0;
        }


        private static NpyArray _getidentity(NpyUFuncObject self, NPY_TYPES otype, string str)
        {
            NpyArray arr;
            NpyArray_Descr descr;
            NpyArray_Descr indescr;
            byte identity;
            NpyArray_VectorUnaryFunc castfunc;

            if (self.identity == NpyUFuncIdentity.NpyUFunc_None)
            {
                string buf = string.Format("zero-size array to ufunc.{0} without identity", str);
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, buf);
                return null;
            }

            /* Get the identity as an unsigned char. */
            if (self.identity == NpyUFuncIdentity.NpyUFunc_One)
            {
                identity = 1;
            }
            else
            {
                identity = 0;
            }

            /* Build the output 0-d array. */
            descr = NpyArray_DescrFromType(otype);
            if (descr == null)
            {
                return null;
            }
            arr = NpyArray_Alloc(descr, 0, null, false, null);
            if (arr == null)
            {
                return null;
            }

            indescr = NpyArray_DescrFromType(NPY_TYPES.NPY_UBYTE);
            Debug.Assert(indescr != null);

            castfunc = NpyArray_GetCastFunc(indescr, otype);
            Npy_DECREF(indescr);
            if (castfunc == null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                 "Can't cast identity to output type.");
                return null;
            }

            /* Use the castfunc to fill in the array. */
            castfunc(new VoidPtr(new byte[] { identity }), arr.data, 1, null, arr);

            return arr;
        }

        /* return 1 if arg1 > arg2, 0 if arg1 == arg2, and -1 if arg1 < arg2 */
        static int cmp_arg_types(NPY_TYPES[] arg1, NPY_TYPES[] arg2, int n)
        {
            int arg1Index = 0;
            int arg2Index = 0;
            for (; n > 0; n--, arg1Index++, arg2Index++)
            {
                if (NpyArray_EquivTypenums(arg1[arg1Index], arg2[arg2Index]))
                {
                    continue;
                }
                if (NpyArray_CanCastSafely(arg1[arg1Index], arg2[arg2Index]))
                {
                    return -1;
                }
                return 1;
            }
            return 0;
        }


        private static int _create_reduce_copy(NpyUFuncReduceObject loop, ref NpyArray arr, NPY_TYPES rtype)
        {
            npy_intp maxsize;
            NpyArray newArray;
            NpyArray_Descr ntype;

            maxsize = NpyArray_SIZE(arr);
            if (maxsize < loop.bufsize)
            {
                if (!(NpyArray_ISBEHAVED_RO(arr))
                    || NpyArray_TYPE(arr) != rtype)
                {
                    ntype = NpyArray_DescrFromType(rtype);

                    newArray = NpyArray_FromArray(arr, ntype, NPYARRAYFLAGS.NPY_FORCECAST | NPYARRAYFLAGS.NPY_ALIGNED);
                    if (newArray == null)
                    {
                        return -1;
                    }
                    arr = newArray;
                    loop.decref_arr = newArray;
                }
            }

            /*
             * Don't decref *arr before re-assigning
             * because it was not going to be DECREF'd anyway.
             *
             * If a copy is made, then the copy will be removed
             * on deallocation of the loop structure by setting
             * loop.decref_arr.
             */
            return 0;
        }

        private static npy_intp[] _compute_output_dims(NpyUFuncLoopObject loop, int iarg, ref int out_nd, npy_intp[] tmp_dims)
        {
            int i;
            NpyUFuncObject ufunc = loop.ufunc;

            if (ufunc.core_enabled == 0)
            {
                /* case of ufunc with trivial core-signature */
                out_nd = loop.iter.nd;
                return loop.iter.dimensions;
            }

            out_nd = loop.iter.nd + ufunc.core_num_dims[iarg];
            if (out_nd > npy_defs.NPY_MAXARGS)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                 "dimension of output variable exceeds limit");
                return null;
            }

            /* copy loop dimensions */
            copydims(tmp_dims, loop.iter.dimensions, loop.iter.nd);

            /* copy core dimension */
            for (i = 0; i < ufunc.core_num_dims[iarg]; i++)
            {
                tmp_dims[loop.iter.nd + i] = loop.core_dim_sizes[
                    1 + ufunc.core_dim_ixs[ufunc.core_offsets[iarg] + i]];
            }
            return tmp_dims;
        }

        private static NpyArray _trunc_coredim(NpyArray ap, int core_nd)
        {
            NpyArray ret;
            int nd = NpyArray_NDIM(ap) - core_nd;

            if (nd < 0)
            {
                nd = 0;
            }
            /* The following code is basically taken from PyArray_Transpose */
            /* NewFromDescr will steal this reference */
            Npy_INCREF(ap.descr);
            ret = NpyArray_NewFromDescr(ap.descr,
                                        nd, ap.dimensions,
                                        ap.strides,
                                        ap.data,
                                        ap.flags,
                                        false, null, Npy_INTERFACE(ap));
            if (ret == null)
            {
                return null;
            }
            /* point at true owner of memory: */
            NpyArray_BASE_ARRAY_Update(ret,ap);
            Npy_INCREF(ap);
            Debug.Assert(null == NpyArray_BASE(ret));
            NpyArray_UpdateFlags(ret,  NPYARRAYFLAGS.NPY_CONTIGUOUS | NPYARRAYFLAGS.NPY_FORTRAN);
            return ret;
        }

        private static int _find_matching_userloop(NpyUFunc_Loop1d funcdata, NPY_TYPES[] arg_types, 
                            NPY_SCALARKIND[] scalars, NpyUFuncGenericFunction function, 
                            int nargs, int nin)
        {
            int i;

            while (funcdata != null)
            {
                for (i = 0; i < nin; i++)
                {
                    if (!NpyArray_CanCoerceScalar(arg_types[i],
                                                  funcdata.arg_types[i],
                                                  scalars[i]))
                        break;
                }
                if (i == nin)
                {
                    /* match found */
                    function = funcdata.func;
                    /* Make sure actual arg_types supported by the loop are used */
                    for (i = 0; i < nargs; i++)
                    {
                        arg_types[i] = funcdata.arg_types[i];
                    }
                    return 0;
                }
                funcdata = funcdata.next;
            }
            return -1;
        }

 
        private static NPY_TYPES _lowest_type(NPY_TYPES intype)
        {
            switch (intype)
            {
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_INT64:
                    return NPY_TYPES.NPY_BYTE;

                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_UINT64:
                    return NPY_TYPES.NPY_UBYTE;

                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                    return NPY_TYPES.NPY_FLOAT;

                case NPY_TYPES.NPY_COMPLEX:
                    return NPY_TYPES.NPY_COMPLEX;

                case NPY_TYPES.NPY_BIGINT:
                    return NPY_TYPES.NPY_BIGINT;


                default:
                    return intype;
            }
        }

 
        /*
         * Floating point error handling.
         */
        //void NpyUFunc_SetFpErrFuncs(fpe_state_f state, fpe_handler_f handler)
        //{
        //    fp_error_state = state;
        //    fp_error_handler = handler;
        //}


        private static int NpyUFunc_getfperr()
        {
            return 0;
        }

        private static bool NpyUFunc_checkfperr(string name, UFuncErrors errmask, VoidPtr errobj, ref int first)
        {
            int retstatus;
            if (name == null)
            {
                name = "";
            }

            /* 1. check hardware flag --- this is platform dependent code */
            retstatus = NpyUFunc_getfperr();
            default_fp_error_handler(name, errmask, errobj, retstatus, ref first);
            return true;
        }


        private static void NpyUFunc_clearfperr()
        {
            NpyUFunc_getfperr();
        }

        private static bool NPY_UFUNC_CHECK_ERROR(NpyUFuncLoopObject loop)
        {
            do
            {
                if ((NpyErr_Occurred() ||
                      (loop.errormask != 0 && NpyUFunc_checkfperr(loop.ufunc.name, loop.errormask, loop.errobj, ref loop.first))))
                {
                    return false;
                }
            } while (false);

            return true;
        }

        private static bool NPY_UFUNC_CHECK_ERROR(NpyUFuncReduceObject arg)
        {
            //do
            //{
            //    if ((NpyErr_Occurred() ||
            //          (arg.errormask != 0 && NpyUFunc_checkfperr(arg.ufunc.name, arg.errormask, arg.errobj, ref arg.first))))
            //    {
            //        return false;
            //    }
            //} while (false);

            return true;
        }


    }



}
