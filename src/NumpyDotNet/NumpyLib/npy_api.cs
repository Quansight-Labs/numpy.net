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

    public class VoidPtr
    {
        // todo: take this out when done with performance
        public static int VoidPtrPlusOperatorCount = 0;

        public static VoidPtr operator +(VoidPtr v1, int i1)
        {
            return new VoidPtr(v1, i1);
        }
        public static VoidPtr operator -(VoidPtr v1, int i1)
        {
            return new VoidPtr(v1, -i1);
        }
        public static VoidPtr operator +(VoidPtr v1, long i1)
        {
            return new VoidPtr(v1, i1);
        }
        public static VoidPtr operator -(VoidPtr v1, long i1)
        {
            return new VoidPtr(v1, -i1);
        }

        public VoidPtr(object obj, NPY_TYPES typenum)
        {
            datap = obj;
            type_num = typenum;
        }
        public VoidPtr(NpyArray array) : this(array.data)
        {
        }
        public VoidPtr(NpyArray array, Int32 offset) : this(array.data, offset)
        {
        }
        public VoidPtr(NpyArray array, Int64 offset) : this(array.data, offset)
        {
        }
        public VoidPtr(VoidPtr vp)
        {
            datap = vp.datap;
            type_num = vp.type_num;
            data_offset = vp.data_offset;
        }
        public VoidPtr(VoidPtr vp, Int32 offset)
        {
            VoidPtrPlusOperatorCount++;

            datap = vp.datap;
            type_num = vp.type_num;
            data_offset = (npy_intp)vp.data_offset + offset;
        }
        public VoidPtr(VoidPtr vp, Int64 offset)
        {
            VoidPtrPlusOperatorCount++;

            datap = vp.datap;
            type_num = vp.type_num;
            data_offset = (npy_intp)(vp.data_offset + offset);
        }
   
        public VoidPtr(bool [] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_BOOL;
        }
        public VoidPtr(byte[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_UBYTE;
        }
        public VoidPtr(sbyte[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_BYTE;
        }
        public VoidPtr(Int16[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_INT16;
        }
        public VoidPtr(UInt16[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_UINT16;
        }
        public VoidPtr(Int32[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_INT32;
        }
        public VoidPtr(UInt32[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_UINT32;
        }
        public VoidPtr(Int64[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_INT64;
        }
        public VoidPtr(UInt64[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_UINT64;
        }
        public VoidPtr(float[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_FLOAT;
        }
        public VoidPtr(double[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_DOUBLE;
        }
        public VoidPtr(decimal[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_DECIMAL;
        }
        public VoidPtr(System.Numerics.Complex[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_COMPLEX;
        }
        public VoidPtr(System.Numerics.BigInteger[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_BIGINT;
        }
        public VoidPtr(object[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_OBJECT;
        }
        public VoidPtr(string[] obj)
        {
            datap = obj;
            type_num = NPY_TYPES.NPY_STRING;
        }
        public VoidPtr()
        {
            datap = null;
            type_num = NPY_TYPES.NPY_OBJECT;
        }

        public System.Object datap;
 
        public NPY_TYPES type_num = NPY_TYPES.NPY_OBJECT;

        public npy_intp data_offset = 0;
    }


    public delegate VoidPtr npy_iter_get_dataptr_t(NpyArrayIterObject iter, npy_intp[] i1);

    public class NpyArrayIterObject : NpyObject_HEAD
    {
        public int nd_m1;                                                   /* number of dimensions - 1 */
        public npy_intp index, size;
        public npy_intp[] coordinates = new npy_intp[npy_defs.NPY_MAXDIMS]; /* N-dimensional loop */
        public npy_intp[] dims_m1 = new npy_intp[npy_defs.NPY_MAXDIMS];     /* ao->dimensions - 1 */
        public npy_intp[] strides = new npy_intp[npy_defs.NPY_MAXDIMS];     /* ao->strides or fake */
        public npy_intp[] backstrides = new npy_intp[npy_defs.NPY_MAXDIMS]; /* how far to jump back */
        public npy_intp[] factors = new npy_intp[npy_defs.NPY_MAXDIMS];     /* shape factors */
        public NpyArray ao;
        public VoidPtr dataptr = new VoidPtr();                             /* pointer to current item*/
        public bool contiguous;
        public bool requiresIteration
        {
            get
            {
                if (ao.base_arr != null || !contiguous)
                    return true;
                return false;
            }
        }

        public npy_intp[,] bounds = new npy_intp[npy_defs.NPY_MAXDIMS, 2];
        public npy_intp[,] limits = new npy_intp[npy_defs.NPY_MAXDIMS, 2];
        public npy_intp[] limits_sizes = new npy_intp[npy_defs.NPY_MAXDIMS];
        public npy_iter_get_dataptr_t translate;
    }

    /*
     * Any object passed to NpyArray_Broadcast must be binary compatible
     * with this structure.
     */
    public class NpyArrayMultiIterObject : NpyObject_HEAD
    {
        /* DANGER - this must be in sync with NpyUFuncLoopObject in npy_ufunc_object.h */

        public int numiter;                                                                    /* number of iters */
        public npy_intp size;                                                                  /* broadcasted size */
        public npy_intp index;                                                                 /* current index */
        public int nd;                                                                         /* number of dims */
        public npy_intp[] dimensions = new npy_intp[npy_defs.NPY_MAXDIMS];                     /* dimensions */
        public NpyArrayIterObject[] iters = new NpyArrayIterObject[npy_defs.NPY_MAXARGS];      /* iterators */
    };

    public class NpyArrayMapIterObject : NpyArrayMultiIterObject
    {
        public NpyArrayMapIterObject()
        {
            for (int i = 0; i < npy_defs.NPY_MAXDIMS; i++)
            {
                indexes[i] = new NpyIndex();
            }
        }
        public NpyArrayIterObject ait;                                                     /* flat Iterator for underlying array */

        /* flat iterator for subspace (when numiter < nd) */
        public NpyArrayIterObject subspace;

        /* The index with any iterator indices changes to 0. */
        public NpyIndex[] indexes = new NpyIndex[npy_defs.NPY_MAXDIMS];
        public int n_indexes;

        /*
         * if subspace iteration, then this is the array of axes in
         * the underlying array represented by the index objects
         */
        public npy_intp[] iteraxes = new npy_intp[npy_defs.NPY_MAXDIMS];
        /*
         * if subspace iteration, the these are the coordinates to the
         * start of the subspace.
         */
        public npy_intp[] bscoord = new npy_intp[npy_defs.NPY_MAXDIMS];

        public int consec;
        public VoidPtr dataptr = new VoidPtr();
        public NpyArray_CopySwapFunc copyswap;
    }

    public class NpyArrayNeighborhoodIterObject
    {
        internal NpyObject_HEAD Head = new NpyObject_HEAD();

        /*
         * NpyArrayIterObject part: keep this in this exact order
         */
        int nd_m1;                                                      /* number of dimensions - 1 */
        npy_intp index, size;
        npy_intp[] coordinates = new npy_intp[npy_defs.NPY_MAXDIMS];    /* N-dimensional loop */
        npy_intp[] dims_m1 = new npy_intp[npy_defs.NPY_MAXDIMS];        /* ao->dimensions - 1 */
        npy_intp[] strides = new npy_intp[npy_defs.NPY_MAXDIMS];        /* ao->strides or fake */
        npy_intp[] backstrides = new npy_intp[npy_defs.NPY_MAXDIMS];    /* how far to jump back */
        npy_intp[] factors = new npy_intp[npy_defs.NPY_MAXDIMS];        /* shape factors */
        NpyArray ao;
        byte[] dataptr;                                                 /* pointer to current item*/
        bool contiguous;

        npy_intp[,] bounds = new npy_intp[npy_defs.NPY_MAXDIMS, 2];
        npy_intp[,] limits = new npy_intp[npy_defs.NPY_MAXDIMS, 2];
        npy_intp[] limits_sizes = new npy_intp[npy_defs.NPY_MAXDIMS];
        npy_iter_get_dataptr_t translate;

        /*
         * New members
         */
        npy_intp nd;

        /* Dimensions is the dimension of the array */
        npy_intp[] dimensions = new npy_intp[npy_defs.NPY_MAXDIMS];

        /*
         * Neighborhood points coordinates are computed relatively to the
         * point pointed by _internal_iter
         */
        NpyArrayIterObject _internal_iter;
        /*
         * To keep a reference to the representation of the constant value
         * for constant padding
         */
        byte[] constant;
        npy_free_func constant_free;

        int mode;
    }

    /*
    * Exception handling
    */
    public enum npyexc_type
    {
        NpyExc_NoError = 0,
        NpyExc_MemoryError,
        NpyExc_IOError,
        NpyExc_ValueError,
        NpyExc_TypeError,
        NpyExc_IndexError,
        NpyExc_RuntimeError,
        NpyExc_AttributeError,
        NpyExc_ComplexWarning,
        NpyExc_NotImplementedError,
        NpyExc_DotNetException,
        NpyExc_FloatingPointError,
        NpyExc_OverflowError,
    };

    public delegate bool npy_interface_array_new_wrapper(NpyArray newArray, bool ensureArray, bool customStrides, object subtype, object interfaceData, ref object interfaceRet);
    public delegate bool npy_interface_iter_new_wrapper(NpyArrayIterObject iter, ref object interfaceRet);
    public delegate bool npy_interface_multi_iter_new_wrapper(NpyArrayMultiIterObject iter, ref object interfaceRet);
    public delegate bool npy_interface_neighbor_iter_new_wrapper(NpyArrayNeighborhoodIterObject iter, ref object interfaceRet);
    public delegate bool npy_interface_descr_new_from_type(int type, NpyArray_Descr descr, ref object interfaceRet);
    public delegate bool npy_interface_descr_new_from_wrapper(object _base, NpyArray_Descr descr, ref object interfaceRet);
    public delegate bool npy_interface_ufunc_new_wrapper(object _base, ref object interfaceRet);

    public delegate void npy_free_func(object o1);
    public delegate void npy_tp_error_set(string FunctionName, npyexc_type et, string error);
    public delegate bool npy_tp_error_occurred(string FunctionName);
    public delegate void npy_tp_error_clear(string FunctionName);
    public delegate int npy_tp_cmp_priority(object o1, object o2);

    /*
     * Interface-provided reference management.  Note that even though these
     * mirror the Python routines they are slightly different because they also
     * work w/ garbage collected systems. Primarily, INCREF returns a possibly
     * different handle. This is the typical case and the second argument will
     * be NULL. When these are called from Npy_INCREF or Npy_DECREF and the
     * core object refcnt is going 0->1 or 1->0 the second argument is a pointer
     * to the nob_interface field.  This allows the interface routine to change
     * the interface pointer.  This is done instead of using the return value
     * to ensure that the switch is atomic.
     */
    public delegate object npy_interface_incref(object o1, ref object o2);
    public delegate object npy_interface_decref(object o1, ref object o2);

    public delegate object enable_threads();
    public delegate object disable_threads(object o1);


    public class NpyInterface_WrapperFuncs
    {
        public npy_interface_array_new_wrapper array_new_wrapper;
        public npy_interface_iter_new_wrapper iter_new_wrapper;
        public npy_interface_multi_iter_new_wrapper multi_iter_new_wrapper;
        public npy_interface_neighbor_iter_new_wrapper neighbor_iter_new_wrapper;
        public npy_interface_descr_new_from_type descr_new_from_type;
        public npy_interface_descr_new_from_wrapper descr_new_from_wrapper;
        public npy_interface_ufunc_new_wrapper ufunc_new_wrapper;
    };

    internal partial class numpyinternal
    {
        internal static void NpyErr_MEMORY()
        {
            NpyErr_SetString(npyexc_type.NpyExc_MemoryError, "memory error");
        }

        static NpyInterface_WrapperFuncs _NpyArrayWrapperFuncs;

        internal static bool NpyInterface_ArrayNewWrapper(NpyArray newArray, bool ensureArray, bool customStrides, object subtype, object interfaceData, ref object interfaceRet)
        {
            if (_NpyArrayWrapperFuncs.array_new_wrapper != null)
            {
               return _NpyArrayWrapperFuncs.array_new_wrapper(newArray, ensureArray, customStrides, subtype, interfaceData, ref interfaceRet);
            }
            return true;
        }


        internal static bool NpyArray_Check(NpyArray arr)
        {
            // todo: ???
            return true;
        }

        internal static npy_interface_neighbor_iter_new_wrapper NpyArrayNeighborhoodIter_Type;
  

    }


}
