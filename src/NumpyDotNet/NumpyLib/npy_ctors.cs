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
using size_t = System.UInt64;
using NpyArray_UCS4 = System.UInt32;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace NumpyLib
{
    internal partial class numpyinternal
    {
        internal delegate void strided_copy_func_t(VoidPtr dst, npy_intp outstrides, VoidPtr src, npy_intp instrides, npy_intp N, int elsize, NpyArray_Descr nad);

        /*
         * Reading from a file or a string.
         *
         * As much as possible, we try to use the same code for both files and strings,
         * so the semantics for fromstring and fromfile are the same, especially with
         * regards to the handling of text representations.
         */
        internal delegate int next_element(ref object o1, object o2, NpyArray_Descr ad, object o3);
        internal delegate int skip_separator(ref object o1, string s1, object o2);


        internal static void _strided_byte_copy(VoidPtr dst, npy_intp outstrides,
                                                   VoidPtr src, npy_intp instrides,
                                                   npy_intp N, int elsize, int eldiv)
        {

     
            try
            {
                if (dst.type_num == src.type_num)
                {
                    var helper = MemCopy.GetMemcopyHelper(dst);
                    helper.strided_byte_copy(dst, outstrides, src, instrides, N, elsize, eldiv);
                }
                else
                {
                    int tin_index = 0;
                    int tout_index = 0;

                    for (int i = 0; i < N; i++)
                    {
                        memmove(dst, tout_index, src, tin_index, elsize);

                        tin_index += (int)instrides;
                        tout_index += (int)outstrides;
                    }
                }
      
            }
            catch (Exception ex)
            {
                NpyErr_SetString(npyexc_type.NpyExc_DotNetException, string.Format("_strided_byte_copy: Exception: {0}", ex.Message));
            }

            return;

        }

        /*
         * This is the main array creation routine.
         *
         * Flags argument has multiple related meanings
         * depending on data and strides:
         *
         * If data is given, then flags is flags associated with data.
         * If strides is not given, then a contiguous strides array will be created
         * and the CONTIGUOUS bit will be set.  If the flags argument
         * has the FORTRAN bit set, then a FORTRAN-style strides array will be
         * created (and of course the FORTRAN flag bit will be set).
         *
         * If data is not given but created here, then flags will be DEFAULT
         * and a non-zero flags argument can be used to indicate a FORTRAN style
         * array is desired.
         */
        internal static size_t npy_array_fill_strides(npy_intp[] strides, npy_intp[] dims, int nd,
                                                      size_t itemsize, NPYARRAYFLAGS inflag, ref NPYARRAYFLAGS objflags)
        {
            int i;
            /* Only make Fortran strides if not contiguous as well */
            if (((inflag & NPYARRAYFLAGS.NPY_FORTRAN) != 0) && ((inflag & NPYARRAYFLAGS.NPY_CONTIGUOUS) == 0))
            {
                for (i = 0; i < nd; i++)
                {
                    strides[i] = (npy_intp)itemsize;
                    itemsize *= (ulong)(dims[i] > 0 ? dims[i] : 1);
                }
                objflags |= NPYARRAYFLAGS.NPY_FORTRAN;
                if (nd > 1)
                {
                    objflags &= ~NPYARRAYFLAGS.NPY_CONTIGUOUS;
                }
                else
                {
                    objflags |= NPYARRAYFLAGS.NPY_CONTIGUOUS;
                }
            }
            else
            {
                for (i = nd - 1; i >= 0; i--)
                {
                    strides[i] = (npy_intp)itemsize;
                    itemsize *= (ulong)(dims[i] > 0 ? dims[i] : 1);
                }
                objflags |= NPYARRAYFLAGS.NPY_CONTIGUOUS;
                if (nd > 1)
                {
                    objflags &= ~NPYARRAYFLAGS.NPY_FORTRAN;
                }
                else
                {
                    objflags |= NPYARRAYFLAGS.NPY_FORTRAN;
                }
            }
            return itemsize;
        }


        /*
         * Change a sub-array field to the base descriptor
         *
         * and update the dimensions and strides
         * appropriately.  Dimensions and strides are added
         * to the end unless we have a FORTRAN array
         * and then they are added to the beginning
         *
         * Strides are only added if given (because data is given).
         */
        internal static int _update_descr_and_dimensions(ref NpyArray_Descr des, npy_intp[] newdims,
                                                         npy_intp[] newstrides, int oldnd, bool isfortran)
        {
            NpyArray_Descr old;
            int newnd;
            int numnew;
            npy_intp[] mydim;
            int mydimindex = 0;

            int i;

            old = des;
            des = old.subarray._base;

            mydim = newdims;
            mydimindex = oldnd;
            numnew = old.subarray.shape_num_dims;

            newnd = oldnd + numnew;
            if (newnd > npy_defs.NPY_MAXDIMS)
            {
                goto finish;
            }
            //if (isfortran)
            //{
            //    memmove(new VoidPtr(newdims), numnew, new VoidPtr(newdims), 0, oldnd * sizeof(npy_intp));
            //    mydim = newdims;
            //}
            for (i = 0; i < numnew; i++)
            {
                mydim[i] = (npy_intp)old.subarray.shape_dims[i];
            }

            if (newstrides != null)
            {
                npy_intp tempsize;
                npy_intp[] mystrides;
                int mystridesindex = 0;

                mystrides = newstrides;
                mystridesindex = oldnd;
                //if (isfortran)
                //{
                //    memmove(new VoidPtr(newstrides), numnew, new VoidPtr(newstrides), 0, oldnd * sizeof(npy_intp));
                //    mystrides = newstrides;
                //}
                /* Make new strides -- alwasy C-contiguous */
                tempsize = (npy_intp)des.elsize;
                for (i = numnew - 1; i >= 0; i--)
                {
                    mystrides[i] = tempsize;
                    tempsize *= mydim[i] > 0 ? mydim[i] : 1;
                }
            }

            finish:
            Npy_INCREF(des);
            Npy_DECREF(old);
            return newnd;
        }

 

        /* If destination is not the right type, then src
           will be cast to destination -- this requires
           src and dest to have the same shape
        */

        /* Requires arrays to have broadcastable shapes

           The arrays are assumed to have the same number of elements
           They can be different sizes and have different types however.
        */

        internal static int _array_copy_into(NpyArray dest, NpyArray src, bool usecopy, bool alwaysstrided)
        {
            bool swap;
            bool simple;
            bool same;
        
            if (!NpyArray_EquivTypes(NpyArray_DESCR(dest), NpyArray_DESCR(src)))
            {
                return NpyArray_CastTo(dest, src);
            }
            if (!NpyArray_ISWRITEABLE(dest))
            {
                NpyErr_SetString(npyexc_type.NpyExc_RuntimeError,"cannot write to array");
                return -1;
            }
            same = NpyArray_SAMESHAPE(dest,src);
            simple = same && 
                (NpyArray_ISWRITEABLE(src) && NpyArray_ISWRITEABLE(dest)) &&
                ((NpyArray_ISCARRAY_RO(src) && NpyArray_ISCARRAY(dest)) ||
                 (NpyArray_ISFARRAY_RO(src) && NpyArray_ISFARRAY(dest)));

            if (simple && !alwaysstrided)
            {
                if (usecopy)
                {
                    memcpy(dest.data, src.data, NpyArray_NBYTES(dest));
                }
                else
                {
                    memmove(dest.data, 0, src.data, 0, NpyArray_NBYTES(dest));
                }
                return 0;
            }

            swap = NpyArray_ISNOTSWAPPED(dest) != NpyArray_ISNOTSWAPPED(src);

            if (src.nd == 0)
            {
                return _copy_from0d(dest, src, usecopy, swap);
            }

            /*
             * Could combine these because _broadcasted_copy would work as well.
             * But, same-shape copying is so common we want to speed it up.
             */
            if (same)
            {
                return _copy_from_same_shape(dest, src, swap);
            }
            else
            {
                return _broadcast_copy(dest, src,swap);
            }
        }

        /*
         * Move the memory of one array into another.
         */
        internal static int NpyArray_MoveInto(NpyArray dest, NpyArray src)
        {
            return _array_copy_into(dest, src, false, false);
        }

        /*
        * combine two arrays into one.
        */
        internal static NpyArray Combine(NpyArray arr1, NpyArray arr2)
        {
            VoidPtr p1 = arr1.data;
            VoidPtr p2 = arr2.data;

            if (p1.type_num != p2.type_num)
            {
                NpyErr_SetString(npyexc_type.NpyExc_RuntimeError, "array types are not the same");
                return null;
            }

            dynamic p1p = p1;
            dynamic p2p = p1;

            int p1len = p1p.datap.Length;
            int p2len = p1p.datap.Length;

            dynamic newData = NpyDataMem_NEW(arr1.descr.type_num, (ulong)(p1len + p2len), false);

            Array.Copy(p1p.datap, 0, newData.datap, 0, p1len);
            Array.Copy(p2p.datap, 0, newData.datap, p1len, p2len);

            npy_intp[] dims = new npy_intp[] { p1len + p2len };
            NpyArray appendedArray = NpyArray_NewFromDescr(arr1.descr, 1, dims, null, newData, arr1.flags, false, null, null);

            return appendedArray;
        }

        internal static int NpyArray_CombineInto(NpyArray dest, IEnumerable<NpyArray> ArraysToCombine)
        {

            foreach (var array in ArraysToCombine)
            {
                if (dest.ItemType != array.ItemType)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_RuntimeError, "array types are not the same");
                    return -1;
                }
            }

            try
            {
                npy_intp destoffset = 0;
                foreach (var array in ArraysToCombine)
                {
                    npy_intp BytesToCopy = NpyArray_SIZE(array) * array.ItemSize;
                    MemCopy.MemCpy(dest.data, destoffset, array.data, 0, BytesToCopy);
                    destoffset += BytesToCopy;
                }

            }
            catch (Exception ex)
            {
                NpyErr_SetString(npyexc_type.NpyExc_RuntimeError, ex.Message);
                return -1;
            }

            return 0;
        }

        /*
        * steals a reference to descr -- accepts null
        */


        internal static NpyArray NpyArray_CheckFromArray(NpyArray arr, NpyArray_Descr descr, NPYARRAYFLAGS requires)
        {
            NpyArray obj;

            Debug.Assert(Validate(arr) && Validate(arr.descr));
            
            if ((requires & NPYARRAYFLAGS.NPY_NOTSWAPPED) != 0)
            {
                if (descr == null && NpyArray_Check(arr) &&
                    !NpyArray_ISNBO(arr))
                {
                    descr = NpyArray_DescrNew(NpyArray_DESCR(arr));
                }
                else if (descr != null && !NpyArray_ISNBO(descr))
                {
                    NpyArray_DESCR_REPLACE(ref descr);
                }
                if (descr != null)
                {
                    descr.byteorder = NPY_NATIVE;
                }
            }

            obj = NpyArray_FromArray(arr, descr, requires);
            if (obj == null)
            {
                return null;
            }
            if (((requires & NPYARRAYFLAGS.NPY_ELEMENTSTRIDES) != 0) && NpyArray_ElementStrides(obj) != 0)
            {
                NpyArray newArray = NpyArray_NewCopy(obj, NPY_ORDER.NPY_ANYORDER);
                Npy_DECREF(obj);
                obj = newArray;
            }
            return obj;
        }

        internal static NpyArray NpyArray_CheckAxis(NpyArray arr, ref int axis, NPYARRAYFLAGS flags)
        {
            NpyArray temp1, temp2;
            int n = arr.nd;

            if (axis == npy_defs.NPY_MAXDIMS || n == 0)
            {
                if (n != 1)
                {
                    temp1 = NpyArray_Ravel(arr, 0);
                    if (temp1 == null)
                    {
                        axis = 0;
                        return null;
                    }
                    if (axis == npy_defs.NPY_MAXDIMS)
                    {
                        axis = NpyArray_NDIM(temp1) - 1;
                    }
                }
                else
                {
                    temp1 = arr;
                    Npy_INCREF(temp1);
                    axis = 0;
                }
                if (flags == 0 && axis == 0)
                {
                    return temp1;
                }
            }
            else
            {
                temp1 = arr;
                Npy_INCREF(temp1);
            }
            if (flags != 0)
            {
                temp2 = NpyArray_CheckFromArray(temp1, null, flags);
                Npy_DECREF(temp1);
                if (temp2 == null)
                {
                    return null;
                }
            }
            else
            {
                temp2 = temp1;
            }
            n = NpyArray_NDIM(temp2);
            if (axis < 0)
            {
                axis += n;
            }
            if ((axis < 0) || (axis >= n))
            {
                var msg = string.Format("axis(={0}) out of bounds", axis);
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                Npy_DECREF(temp2);
                return null;
            }
            return temp2;
        }


        /*NUMPY_API
         * Generic new array creation routine.
         *
         * Array type algorithm: IF
         *  ensureArray             - use base array type
         *  subtype != null         - use subtype
         *  interfaceData != null   - use type of interface data
         *  default                 - use base array type
         *
         * Steals a reference to descr (even on failure)
         */
        internal static NpyArray NpyArray_NewFromDescr(NpyArray_Descr descr, int nd, npy_intp[] dims, npy_intp[] strides, VoidPtr data,
                      NPYARRAYFLAGS flags, bool ensureArray, object subtype, object interfaceData)
        {
            NpyArray self;
            int i;
            size_t sd;
            npy_intp largest;
            npy_intp size;

            Debug.Assert(Validate(descr));
            Debug.Assert(0 < descr.nob_refcnt);

            if (descr.subarray != null)
            {
                NpyArray ret;
                npy_intp []newdims = new npy_intp[npy_defs.NPY_MAXDIMS];
                npy_intp []newstrides = new npy_intp[npy_defs.NPY_MAXDIMS];
                bool isfortran = false;
                isfortran = (data != null && ((flags & NPYARRAYFLAGS.NPY_FORTRAN) > 0) &&
                             !((flags & NPYARRAYFLAGS.NPY_CONTIGUOUS) > 0)) ||
                            ((data == null) && (flags != 0));
                Array.Copy(dims, newdims, nd);
                if (strides != null)
                {
                    Array.Copy(strides, newstrides, nd); 
                }
                nd = _update_descr_and_dimensions(ref descr, newdims,
                                                 newstrides, nd, isfortran);
                ret = NpyArray_NewFromDescr(descr, nd, newdims,
                                            newstrides, 
                                            data, flags, ensureArray, subtype,
                                            interfaceData);
                return ret;
            }
            if (nd < 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                 "number of dimensions must be >=0");
                Npy_DECREF(descr);
                return null;
            }
            if (nd > npy_defs.NPY_MAXDIMS)
            {
                var msg = string.Format("maximum number of dimensions is {0}", npy_defs.NPY_MAXDIMS);
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                Npy_DECREF(descr);
                return null;
            }

            /* Check dimensions */
            size = 1;
            sd = (size_t)descr.elsize;
            if (sd == 0)
            {
                if (!NpyDataType_ISSTRING(descr))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "Empty data-type");
                    Npy_DECREF(descr);
                    return null;
                }
                NpyArray_DESCR_REPLACE(ref descr);
                if (descr.type_num == NPY_TYPES.NPY_STRING)
                {
                    descr.elsize = 1;
                }
                else
                {
                    descr.elsize = sizeof(NpyArray_UCS4);
                }
                sd = (size_t)descr.elsize;
            }

            largest = (npy_intp)(npy_defs.NPY_MAX_INTP / sd);
            for (i = 0; i < nd; i++)
            {
                npy_intp dim = dims[i];

                if (dim == 0)
                {
                    /*
                     * Compare to NpyArray_OverflowMultiplyList that
                     * returns 0 in this case.
                     */
                    continue;
                }
                if (dim < 0)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                     "negative dimensions are not allowed");
                    Npy_DECREF(descr);
                    return null;
                }
                if (dim > largest)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                     "array is too big.");
                    Npy_DECREF(descr);
                    return null;
                }
                size *= dim;
                largest /= dim;
            }


            self = new NpyArray();
            if (self == null)
            {
                Npy_DECREF(descr);
                NpyErr_SetString(npyexc_type.NpyExc_MemoryError, "insufficient memory");
                return null;
            }
            NpyObject_Init(self, NpyArray_Type);
            self.nd = nd;
            self.dimensions = null;
            self.data = null;
            if (data == null || data.datap == null)
            {
                self.flags = NPYARRAYFLAGS.NPY_DEFAULT;
                if (flags != 0)
                {
                    self.flags |= NPYARRAYFLAGS.NPY_FORTRAN;
                    if (nd > 1)
                    {
                        self.flags &= ~NPYARRAYFLAGS.NPY_CONTIGUOUS;
                    }
                    flags = NPYARRAYFLAGS.NPY_FORTRAN;
                }
            }
            else
            {
                self.flags = (flags & ~NPYARRAYFLAGS.NPY_UPDATEIFCOPY);
            }
            self.nob_interface = null;
            self.descr = descr;
            self.SetBase(null);
            self.base_obj = null;

            if (nd > 0)
            {
                self.dimensions = NpyDimMem_NEW(nd);
                self.strides = NpyDimMem_NEW(nd);
                if (self.dimensions == null || self.strides == null)
                {
                    NpyErr_MEMORY();
                    goto fail;
                }

       
                memcpy(self.dimensions, dims, sizeof(npy_intp) * nd);
                if (strides == null)
                { /* fill it in */
                    sd = npy_array_fill_strides(self.strides, dims, nd, sd,
                                                flags, ref self.flags);
                }
                else
                {
                    /*
                     * we allow strides even when we create
                     * the memory, but be careful with this...
                     */
                    memcpy(self.strides, strides, sizeof(npy_intp) * nd);
                    sd *= (ulong)size;
                }
            }
            else
            {
                self.dimensions = self.strides = null;
            }

            if (data ==null || data.datap == null)
            {
                /*
                 * Allocate something even for zero-space arrays
                 * e.g. shape=(0,) -- otherwise buffer exposure
                 * (a.data) doesn't work as it should.
                 */

                if (sd == 0)
                {
                    sd = (size_t)descr.elsize;
                }
                if ((data = NpyDataMem_NEW(descr.type_num, sd)) == null)
                {
                    NpyErr_MEMORY();
                    goto fail;
                }
                self.flags |= NPYARRAYFLAGS.NPY_OWNDATA;
  
            }
            else
            {
                /*
                 * If data is passed in, this object won't own it by default.
                 * Caller must arrange for this to be reset if truly desired
                 */
                self.flags &= ~NPYARRAYFLAGS.NPY_OWNDATA;
            }

        
            self.data = data;

            /*
             * call the __array_finalize__
             * method if a subtype.
             * If obj is null, then call method with Py_None
             */
            if (false == NpyInterface_ArrayNewWrapper(self, ensureArray,
                                                          (null != strides),
                                                          subtype, interfaceData,
                                                          ref self.nob_interface))
            {
                self.nob_interface = null;
                Npy_DECREF(self);
                return null;
            }
            Debug.Assert(Validate(self) && Validate(self.descr));
            return self;

            fail:
            Npy_DECREF(self);
            return null;
        }

        /*NUMPY_API
         * Creates a new array with the same shape as the provided one,
         * with possible memory layout order and data type changes.
         *
         * prototype - The array the new one should be like.
         * order     - NPY_CORDER - C-contiguous result.
         *             NPY_FORTRANORDER - Fortran-contiguous result.
         *             NPY_ANYORDER - Fortran if prototype is Fortran, C otherwise.
         *             NPY_KEEPORDER - Keeps the axis ordering of prototype.
         * dtype     - If not NULL, overrides the data type of the result.
         * subok     - If 1, use the prototype's array subtype, otherwise
         *             always create a base-class array.
         *
         * NOTE: If dtype is not NULL, steals the dtype reference.  On failure or when
         * dtype->subarray is true, dtype will be decrefed.
         */
        internal static NpyArray NpyArray_NewLikeArray(NpyArray prototype, NPY_ORDER order,  NpyArray_Descr dtype, bool subok)
        {
            NpyArray ret = null;
            int ndim = NpyArray_NDIM(prototype);

            /* If no override data type, use the one from the prototype */
            if (dtype == null)
            {
                dtype = NpyArray_DESCR(prototype);
                Npy_INCREF(dtype);
            }

            NPYARRAYFLAGS flags = NPYARRAYFLAGS.NPY_DEFAULT;

            /* Handle ANYORDER and simple KEEPORDER cases */
            switch (order)
            {
                case NPY_ORDER.NPY_ANYORDER:
                    flags = NpyArray_ISFORTRAN(prototype) ?
                                            NPYARRAYFLAGS.NPY_FARRAY_RO : NPYARRAYFLAGS.NPY_CARRAY_RO;
                    break;
                case NPY_ORDER.NPY_KEEPORDER:
                    if (NpyArray_IS_C_CONTIGUOUS(prototype) || ndim <= 1)
                    {
                        flags = NPYARRAYFLAGS.NPY_CARRAY_RO;
                        break;
                    }
                    else if (NpyArray_IS_F_CONTIGUOUS(prototype))
                    {
                        flags = NPYARRAYFLAGS.NPY_FARRAY_RO;
                        break;
                    }
                    break;
                default:
                    break;
            }

            /* If it's not KEEPORDER, this is simple */
            if (order != NPY_ORDER.NPY_KEEPORDER)
            {
                ret = NpyArray_NewFromDescr(    dtype,
                                                ndim,
                                                NpyArray_DIMS(prototype),
                                                null,
                                                null,
                                                flags,
                                                true,
                                                null, null);


            }
            /* KEEPORDER needs some analysis of the strides */
            else
            {
                npy_intp[] strides = new npy_intp[npy_defs.NPY_MAXDIMS];
                npy_intp stride;
                npy_intp []shape = NpyArray_DIMS(prototype);
                npy_stride_sort_item []strideperm = new npy_stride_sort_item[npy_defs.NPY_MAXDIMS];
                int idim;

                PyArray_CreateSortedStridePerm(NpyArray_NDIM(prototype),
                                                NpyArray_STRIDES(prototype),
                                                strideperm);

                /* Build the new strides */
                stride = dtype.elsize;
                for (idim = ndim - 1; idim >= 0; --idim)
                {
                    npy_intp i_perm = strideperm[idim].perm;
                    strides[i_perm] = stride;
                    stride *= shape[i_perm];
                }

                /* Finally, allocate the array */
                ret = NpyArray_NewFromDescr(dtype,
                                                ndim,
                                                shape,
                                                strides,
                                                null,
                                                flags,
                                                true,
                                                null, null);
            }

            return ret;
        }


        /*
         * Generic new array creation routine.
         */
        internal static NpyArray NpyArray_New(object subtype, int nd, npy_intp[] dims, NPY_TYPES type_num, npy_intp[] strides, VoidPtr data, int itemsize, NPYARRAYFLAGS flags, object obj)
        {
            NpyArray_Descr descr;
            NpyArray newArr;

            descr = NpyArray_DescrFromType(type_num);
            if (descr == null)
            {
                return null;
            }
            if (descr.elsize == 0)
            {
                if (itemsize < 1)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                     "data type must provide an itemsize");
                    Npy_DECREF(descr);
                    return null;
                }
                NpyArray_DESCR_REPLACE(ref descr);
                descr.elsize = itemsize;
            }
            newArr = NpyArray_NewFromDescr(descr, nd, dims, strides, 
                                        data, flags, false, subtype, obj);
            if (null == newArr) return null;

            Debug.Assert(Validate(newArr) && Validate(newArr.descr));
            return newArr;
        }

        /*
         * Creates an array allocating new data.
         * Steals the reference to the descriptor.
         */
        internal static NpyArray NpyArray_Alloc(NpyArray_Descr descr, int nd, npy_intp[] dims, bool is_fortran, object interfaceData)
        {
             return NpyArray_NewFromDescr(descr, nd, dims,
                                         null, null, 
                                         (is_fortran ? NPYARRAYFLAGS.NPY_FORTRAN : 0),
                                         false, null, interfaceData);
        }


        /*
         * Creates a new array which is a view into the buffer of array.
         * Steals the reference to the descriptor.
         */
        internal static NpyArray NpyArray_NewView(NpyArray_Descr descr, int nd, npy_intp[] dims, npy_intp[] strides, 
                        NpyArray array, npy_intp offset, bool ensure_array)
        {
            /* TODO: Add some sanity checking. */
            NpyArray result;
            NPYARRAYFLAGS flags = array.flags & NPYARRAYFLAGS.NPY_WRITEABLE;

            if (strides == null)
            {
                flags |= (array.flags & (NPYARRAYFLAGS.NPY_CONTIGUOUS | NPYARRAYFLAGS.NPY_FORTRAN));
            }
            result = NpyArray_NewFromDescr(descr, nd, dims,
                                           strides,  array.data + offset,
                                           flags,
                                           ensure_array, null,
                                           Npy_INTERFACE(array));
            if (result == null)
            {
                return null;
            }
            result.SetBase(array);
            Npy_INCREF(array);
            NpyArray_UpdateFlags(result, NPYARRAYFLAGS.NPY_UPDATE_ALL);
            return result;
        }

        /*
         * steals reference to newtype --- acc. null
         */
        internal static NpyArray NpyArray_FromArray(NpyArray arr, NpyArray_Descr newtype, NPYARRAYFLAGS flags)
        {
            NpyArray ret = null;
            int itemsize;
            bool copy = false;
            NPYARRAYFLAGS arrflags;
            NpyArray_Descr oldtype;
            string msg = "cannot copy back to a read-only array";
            bool ensureArray = false;

            Debug.Assert(Validate(arr) && Validate(arr.descr));
            if (newtype != null)
            {
                Debug.Assert(Validate(newtype));
            }

            oldtype = NpyArray_DESCR(arr);
            if (newtype == null)
            {
                newtype = oldtype;
                Npy_INCREF(oldtype);
            }
            itemsize = newtype.elsize;
            if (itemsize == 0)
            {
                NpyArray_DESCR_REPLACE(ref newtype);
                if (newtype == null)
                {
                    return null;
                }
                newtype.elsize = oldtype.elsize;
                itemsize = newtype.elsize;
            }

            /*
             * Can't cast unless ndim-0 array, FORCECAST is specified
             * or the cast is safe.
             */
            if (!((flags & NPYARRAYFLAGS.NPY_FORCECAST) > 0) && NpyArray_NDIM(arr) != 0 &&
                !NpyArray_CanCastTo(oldtype, newtype))
            {
                Npy_DECREF(newtype);
                NpyErr_SetString(npyexc_type.NpyExc_TypeError,
                                "array cannot be safely cast to required type");
                return null;
            }

            /* Don't copy if sizes are compatible */
            if ((flags & NPYARRAYFLAGS.NPY_ENSURECOPY) > 0 || NpyArray_EquivTypes(oldtype, newtype))
            {
                arrflags = arr.flags;
                copy = (flags & NPYARRAYFLAGS.NPY_ENSURECOPY) > 0 ||
                ((flags & NPYARRAYFLAGS.NPY_CONTIGUOUS) > 0 && (0==(arrflags & NPYARRAYFLAGS.NPY_CONTIGUOUS)))
                || ((flags & NPYARRAYFLAGS.NPY_ALIGNED) > 0 && (0==(arrflags & NPYARRAYFLAGS.NPY_ALIGNED)))
                || (arr.nd > 1 &&
                    ((flags & NPYARRAYFLAGS.NPY_FORTRAN) > 0 && (0==(arrflags & NPYARRAYFLAGS.NPY_FORTRAN))))
                || ((flags & NPYARRAYFLAGS.NPY_WRITEABLE) > 0 && (0==(arrflags & NPYARRAYFLAGS.NPY_WRITEABLE)));

                if (copy)
                {
                    if ((flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) > 0 &&
                        (!NpyArray_ISWRITEABLE(arr)))
                    {
                        Npy_DECREF(newtype);
                        NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                        return null;
                    }
                    if ((flags & NPYARRAYFLAGS.NPY_ENSUREARRAY) > 0)
                    {
                        ensureArray = true;
                    }
                    ret = NpyArray_Alloc(newtype, arr.nd, arr.dimensions,
                                         (flags & NPYARRAYFLAGS.NPY_FORTRAN) > 0,
                                         ensureArray ? null : Npy_INTERFACE(arr));
                    if (ret == null)
                    {
                        return null;
                    }
                    if (NpyArray_CopyInto(ret, arr) == -1)
                    {
                        Npy_DECREF(ret);
                        return null;
                    }
                    if ((flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) > 0)
                    {
                        ret.flags |= NPYARRAYFLAGS.NPY_UPDATEIFCOPY;
                        ret.SetBase(arr);
                        Debug.Assert(null == ret.base_arr || null == ret.base_obj);
                        ret.base_arr.flags &= ~NPYARRAYFLAGS.NPY_WRITEABLE;
                        Npy_INCREF(arr);
                    }
                }
                /*
                 * If no copy then just increase the reference
                 * count and return the input
                 */
                else
                {
                    Npy_DECREF(newtype);
                    if (((flags & NPYARRAYFLAGS.NPY_ENSUREARRAY) > 0) /* &&
                !NpyArray_CheckExact(arr) --
                TODO: Would be nice to check this in the future */ )
                    {
                        Npy_INCREF(arr.descr);
                        ret = NpyArray_NewView(arr.descr,
                                               arr.nd, arr.dimensions, arr.strides,
                                               arr, 0, true);
                        if (ret == null)
                        {
                            return null;
                        }
                    }
                    else
                    {
                        ret = arr;
                        Npy_INCREF(arr);
                    }
                    return ret;
                }
            }

            /*
             * The desired output type is different than the input
             * array type and copy was not specified
             */
            else
            {
                if ((flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) > 0 && (!NpyArray_ISWRITEABLE(arr)))
                {
                    Npy_DECREF(newtype);
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                    return null;
                }
                if ((flags & NPYARRAYFLAGS.NPY_ENSUREARRAY) > 0)
                {
                    ensureArray = true;
                }
                ret = NpyArray_Alloc(newtype, arr.nd, arr.dimensions,
                                     (flags & NPYARRAYFLAGS.NPY_FORTRAN) > 0,
                                     ensureArray ? null : Npy_INTERFACE(arr));
                if (ret == null)
                {
                    return null;
                }
                if (NpyArray_CastTo(ret, arr) < 0)
                {
                    Npy_DECREF(ret);
                    return null;
                }
                if ((flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) > 0)
                {
                    ret.flags |= NPYARRAYFLAGS.NPY_UPDATEIFCOPY;
                    ret.SetBase(arr);
                    ret.base_arr.flags &= ~NPYARRAYFLAGS.NPY_WRITEABLE;
                    Npy_INCREF(arr);
                }
            }
            Debug.Assert(Validate(ret) && Validate(ret.descr));
            return ret;
        }

        /*
         * Copy an Array into another array -- memory must not overlap
         * Does not require src and dest to have "broadcastable" shapes
         * (only the same number of elements).
         */
        internal static int NpyArray_CopyAnyInto(NpyArray dest, NpyArray src)
        {
            int elsize; 
            bool simple;
            NpyArrayIterObject idest, isrc;
        
            if (!NpyArray_EquivArrTypes(dest, src))
            {
                return NpyArray_CastAnyTo(dest, src);
            }
            if (!NpyArray_ISWRITEABLE(dest))
            {
                NpyErr_SetString(npyexc_type.NpyExc_RuntimeError,
                                 "cannot write to array");
                return -1;
            }
            if (NpyArray_SIZE(dest) != NpyArray_SIZE(src))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                 "arrays must have the same number of elements for copy");
                return -1;
            }

            simple = 
                 (NpyArray_ISWRITEABLE(src) && NpyArray_ISWRITEABLE(dest)) &&
                ((NpyArray_ISCARRAY_RO(src) && NpyArray_ISCARRAY(dest)) ||
                 (NpyArray_ISFARRAY_RO(src) && NpyArray_ISFARRAY(dest)));
            if (simple)
            {
                memcpy(dest.data, src.data, NpyArray_NBYTES(dest));
                return 0;
            }

            if (NpyArray_SAMESHAPE(dest, src))
            {
                bool swap;

                swap = NpyArray_ISNOTSWAPPED(dest) != NpyArray_ISNOTSWAPPED(src);
                return _copy_from_same_shape(dest, src, swap);
            }

            /* Otherwise we have to do an iterator-based copy */
            idest = NpyArray_IterNew(dest);
            if (idest == null)
            {
                return -1;
            }
            isrc = NpyArray_IterNew(src);
            if (isrc == null)
            {
                Npy_DECREF(idest);
                return -1;
            }
            elsize = dest.descr.elsize;

            while (idest.index < idest.size)
            {
                memcpy(idest.dataptr, isrc.dataptr, elsize);
                NpyArray_ITER_NEXT(idest);
                NpyArray_ITER_NEXT(isrc);
            }

            return 0;
        }

        /* TODO: Put the order parameter in PyArray_CopyAnyInto and remove this */
        internal static int PyArray_CopyAsFlat(NpyArray dst, NpyArray src, NPY_ORDER order)
        {
            PyArray_StridedUnaryOp stransfer = null;
            NpyAuxData transferdata = null;
            NpyArrayIterObject dst_iter, src_iter;
            VoidPtr dst_dataptr, src_dataptr;
            npy_intp dst_stride, src_stride;
            VoidPtr dst_data, src_data;
            npy_intp dst_count, src_count, count;
            npy_intp src_itemsize;
            npy_intp dst_size, src_size;
            bool needs_api = false;
            npy_intp dst_countptr, src_countptr;

            NumericOperations operations = NumericOperations.GetOperations(null, src, dst, null);


            if (NpyArray_FailUnlessWriteable(dst, "destination array") < 0)
            {
                return -1;
            }

            /*
             * If the shapes match and a particular order is forced
             * for both, use the more efficient CopyInto
             */
            if (order != NPY_ORDER.NPY_ANYORDER && order != NPY_ORDER.NPY_KEEPORDER &&
                    NpyArray_NDIM(dst) == NpyArray_NDIM(src) &&
                    NpyArray_CompareLists(NpyArray_DIMS(dst), NpyArray_DIMS(src),
                                        NpyArray_NDIM(dst)))
            {
                return NpyArray_CopyInto(dst, src);
            }


            dst_size = NpyArray_SIZE(dst);
            src_size = NpyArray_SIZE(src);
            if (dst_size != src_size)
            {
                NpyErr_SetString(npyexc_type.NpyExc_RuntimeError,
                   string.Format("cannot copy from array of size {0} into an array of size {0}", src_size, dst_size));
                return -1;
            }

            /* Zero-sized arrays require nothing be done */
            if (dst_size == 0)
            {
                return 0;
            }

  

            /*
             * This copy is based on matching C-order traversals of src and dst.
             * By using two iterators, we can find maximal sub-chunks that
             * can be processed at once.
             */


            dst_iter = NpyArray_IterNew(dst);
            if (dst_iter == null)
            {
                return -1;
            }
            src_iter = NpyArray_IterNew(src);
            if (src_iter == null)
            {
                return -1;
            }


            /* Get all the values needed for the inner loop */
            dst_dataptr = dst_iter.dataptr;
            /* Since buffering is disabled, we can cache the stride */
            dst_stride = dst_iter.strides[0];
            dst_countptr = dst_iter.size;

            src_dataptr = src_iter.dataptr;
            /* Since buffering is disabled, we can cache the stride */
            src_stride = src_iter.strides[0];
            src_countptr = src_iter.size;
            src_itemsize = NpyArray_DESCR(src).elsize;





            /*
             * Because buffering is disabled in the iterator, the inner loop
             * strides will be the same throughout the iteration loop.  Thus,
             * we can pass them to this function to take advantage of
             * contiguous strides, etc.
             */
            if (PyArray_GetDTypeTransferFunction(
                            NpyArray_ISALIGNED(src) && NpyArray_ISALIGNED(dst),
                            src_iter.strides[0], dst_iter.strides[0],
                            NpyArray_DESCR(src), NpyArray_DESCR(dst),
                            false,
                            ref stransfer, ref transferdata,
                            ref needs_api) != npy_defs.NPY_SUCCEED)
            {
                return -1;
            }

            if (!needs_api)
            {
                //NPY_BEGIN_THREADS;
            }

            dst_count = dst_countptr;
            src_count = src_countptr;
            dst_data = new VoidPtr(dst_dataptr);
            src_data = new VoidPtr(src_dataptr);

            /* Transfer the biggest amount that fits both */
            count = (src_count < dst_count) ? src_count : dst_count;



            for (long j = 0; j < count; j++)
            {
                var bValue = operations.srcGetItem(src_iter.dataptr.data_offset - src.data.data_offset, src);

                try
                {
                    operations.destSetItem(dst_iter.dataptr.data_offset - dst.data.data_offset, bValue, dst);
                }
                catch
                {
                    operations.destSetItem(dst_iter.dataptr.data_offset - dst.data.data_offset, 0, dst);
                }
                NpyArray_ITER_NEXT(src_iter);
                NpyArray_ITER_NEXT(dst_iter);
            }

            //NPY_END_THREADS;

            NPY_AUXDATA_FREE(transferdata);

            return NpyErr_Occurred() ? -1 : 0;

        }

 

        internal static int NpyArray_CopyInto(NpyArray dest, NpyArray src)
        {
            return _array_copy_into(dest, src, true, false);
        }


        internal static void npy_byte_swap_vector(VoidPtr p, npy_intp n, int size)
        {
            _strided_byte_swap(p, (npy_intp)size, n, size);
        }

        /*
         * Special-case of NpyArray_CopyInto when dst is 1-d
         * and contiguous (and aligned).
         * NpyArray_CopyInto requires broadcastable arrays while
         * this one is a flattening operation...
         */
        internal static int _flat_copyinto(NpyArray dst, NpyArray src, NPY_ORDER order)
        {
            NpyArray orig_src = src;
            if (NpyArray_NDIM(src) == 0)
            {
                memcpy(NpyArray_BYTES(dst), NpyArray_BYTES(src), (long)NpyArray_BYTES_Length(src));
                return 0;
            }

            int axis = NpyArray_NDIM(src) - 1;

            if (order == NPY_ORDER.NPY_FORTRANORDER)
            {
                if (NpyArray_NDIM(src) <= 2)
                {
                    axis = 0;
                }
                /* fall back to a more general method */
                else
                {
                    src = NpyArray_Transpose(orig_src, null);
                }
            }

            NpyArrayIterObject it = NpyArray_IterAllButAxis(src, ref axis);
            if (it == null)
            {
                if (src != orig_src)
                {
                    Npy_DECREF(src);
                }
                return -1;
            }


            VoidPtr dptr = new VoidPtr(dst);
            NpyArray_Descr descr = dst.descr;
            int elsize = descr.elsize;
            npy_intp nbytes = (npy_intp)(elsize * NpyArray_DIM(src, axis));

            flat_copyinto(dptr, elsize, it,  NpyArray_STRIDE(src, axis), NpyArray_DIM(src, axis), nbytes);
 
            if (src != orig_src)
            {
                Npy_DECREF(src);
            }
            Npy_DECREF(it);
            return 0;
        }

        internal static void flat_copyinto(VoidPtr dest, npy_intp outstride, NpyArrayIterObject srcIter, npy_intp instride, npy_intp N, npy_intp destOffset)
        {

            npy_intp TotalLoops = srcIter.size - srcIter.index;
            npy_intp TotalCopies = TotalLoops * N;
            int eldiv = GetDivSize((int)outstride);

            var helper = MemCopy.GetMemcopyHelper(dest);
            helper.strided_byte_copy_init(dest, outstride, srcIter.dataptr, instride, (int)outstride, eldiv);

            if (TotalLoops < 2 || TotalCopies < numpyinternal.flatCopyParallelSize)
            {
                while (srcIter.index < srcIter.size)
                {
                    helper.strided_byte_copy(dest.data_offset, srcIter.dataptr.data_offset, N);
                    dest.data_offset += destOffset;
                    NpyArray_ITER_NEXT(srcIter);
                }
            }
            else
            {
                npy_intp SingleIterSize = N > numpyinternal.flatCopyParallelSize ? -1 : numpyinternal.maxCopyFieldParallelSize;
                var ParallelIters = NpyArray_ITER_ParallelSplit(srcIter, SingleIterSize);

                Parallel.For(0, ParallelIters.Count(), index =>
                //for (int index = 0; index < ParallelIters.Count(); index++)
                {
                    var ParallelDest = new VoidPtr(dest);
                    var ParallelIter = ParallelIters.ElementAt(index);
                    ParallelDest.data_offset += destOffset * index * ParallelIters.ElementAt(0).size;

                    while (ParallelIter.index < ParallelIter.size)
                    {
                        helper.strided_byte_copy(ParallelDest.data_offset, ParallelIter.dataptr.data_offset, N);
                        ParallelDest.data_offset += destOffset; /// * ParallelIters.Count();
                        NpyArray_ITER_NEXT(ParallelIter);
                    }
                });

            }

        }

        internal static void _strided_byte_swap(VoidPtr dest, npy_intp stride, npy_intp n, int size)
        {
            byte c = 0;
            npy_intp j, m;

            npy_intp a, b;
            VoidPtr ConvertedByteVP = ArrayConversions.ConvertToDesiredArrayType(dest, 0, (int)VoidPointer_BytesLength(dest), NPY_TYPES.NPY_UBYTE);
            byte[] p = (byte[])ConvertedByteVP.datap;

            switch (size)
            {
                case 1: /* no byteswap necessary */
                    break;
                case 4:
                    for (a = 0; n > 0; n--, a += stride - 1)
                    {
                        b = a + 3;
                        c = p[a]; p[a] = p[b]; a++; p[b] = c; b--;
                        c = p[a]; p[a] = p[b]; p[b] = c;
                    }
                    break;
                case 8:
                    for (a = 0; n > 0; n--, a += stride - 3)
                    {
                        b = a + 7;
                        c = p[a]; p[a] = p[b]; a++; p[b] = c; b--;
                        c = p[a]; p[a] = p[b]; a++; p[b] = c; b--;
                        c = p[a]; p[a] = p[b]; a++; p[b] = c; b--;
                        c = p[a]; p[a] = p[b]; p[b] = c;
                    }
                    break;
                case 2:
                    for (a = 0; n > 0; n--, a += stride)
                    {
                        b = a + 1;
                        c = p[a]; p[a] = p[b]; p[b] = c;
                    }
                    break;
                default:
                    m = (npy_intp)(size / 2);
                    for (a = 0; n > 0; n--, a += stride - m)
                    {
                        b = (npy_intp)(a + (size - 1));
                        for (j = 0; j < m; j++)
                        {
                            c = p[a]; p[a] = p[b]; a++; p[b] = c; b--;
                        }
                    }
                    break;

            }


            MemCopy.MemCpy(dest, 0, ConvertedByteVP, 0, (long)VoidPointer_BytesLength(ConvertedByteVP));
        }

        internal static int _copy_from_same_shape(NpyArray dest, NpyArray src, bool swap)
        {
            int maxaxis = -1, elsize, eldiv;
            npy_intp maxdim;
            NpyArrayIterObject dit, sit;

            dit = NpyArray_IterAllButAxis(dest, ref maxaxis);
            sit = NpyArray_IterAllButAxis(src, ref maxaxis);

            maxdim = dest.dimensions[maxaxis];

            if ((dit == null) || (sit == null))
            {
                Npy_XDECREF(dit);
                Npy_XDECREF(sit);
                return -1;
            }

            elsize = NpyArray_ITEMSIZE(dest);
            eldiv = GetDivSize(elsize);

            var srcParallelIters = NpyArray_ITER_ParallelSplit(sit, numpyinternal.maxCopyFieldParallelSize);
            var destParallelIters = NpyArray_ITER_ParallelSplit(dit, numpyinternal.maxCopyFieldParallelSize);

            var helper = MemCopy.GetMemcopyHelper(dest.data);
            helper.strided_byte_copy_init(dest.data, dest.strides[maxaxis], src.data, src.strides[maxaxis], elsize, eldiv);

            Parallel.For(0, destParallelIters.Count(), index =>
            //for (int index = 0; index < destParallelIters.Count(); index++) // 
            {
                var ldestIter = destParallelIters.ElementAt(index);
                var lsrcIter = srcParallelIters.ElementAt(index);

                //Parallel.For(0, taskSize, i =>
                while (ldestIter.index < ldestIter.size)
                {
                    helper.strided_byte_copy(ldestIter.dataptr.data_offset, lsrcIter.dataptr.data_offset, maxdim);
  
                    if (swap)
                    {
                        _strided_byte_swap(ldestIter.dataptr,
                                           ldestIter.strides[maxaxis],
                                           dest.dimensions[maxaxis], elsize);
                    }
                    NpyArray_ITER_NEXT(ldestIter);
                    NpyArray_ITER_NEXT(lsrcIter);
                }
            } );


            Npy_DECREF(sit);
            Npy_DECREF(dit);
            return 0;
        }


        internal static int _broadcast_copy(NpyArray dest, NpyArray src, bool swap)
        {
            NpyArrayMultiIterObject multi;
            int maxaxis;
            npy_intp maxdim;
            NpyArray_Descr descr = dest.descr;
            int elsize = descr.elsize;
            int eldiv = GetDivSize(elsize);

            multi = NpyArray_MultiIterFromArrays(null, 0, 2, dest, src);
            if (multi == null)
            {
                return -1;
            }

            if (multi.size != NpyArray_SIZE(dest))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                 "array dimensions are not compatible for copy");
                Npy_DECREF(multi);
                return -1;
            }

            maxaxis = NpyArray_RemoveSmallest(multi);
            if (maxaxis < 0)
            {
                memcpy(dest.data, src.data, elsize);
                if (swap)
                {
                    npy_byte_swap_vector(dest.data, 1, elsize);
                }
                return 0;
            }
            maxdim = multi.dimensions[maxaxis];

            /*
             * Increment the source and decrement the destination
             * reference counts
             *
             * Refcount note: src and dest may have different sizes
             */

            var srcIter = multi.iters[1];
            var destIter = multi.iters[0];

            npy_intp TotalSize = multi.size - multi.index;
            if (TotalSize < 4)
            {
                _broadcast_copy(destIter, srcIter, maxaxis, maxdim, elsize, eldiv, swap);
            }
            else
            {
                var parallelIters = NpyArray_ITER_ParallelSplit(destIter, srcIter);

                Parallel.For(0, parallelIters.Item1.Count(), i =>
                {
                    var _destIter = parallelIters.Item1.ElementAt(i);
                    var _srcIter = parallelIters.Item2.ElementAt(i);
                    _broadcast_copy(_destIter, _srcIter, maxaxis, maxdim, elsize, eldiv, swap);
                });
            }
    

            Npy_DECREF(multi);
            return 0;
        }

        private static void _broadcast_copy(NpyArrayIterObject destIter, NpyArrayIterObject srcIter, int maxaxis, npy_intp maxdim, int elsize, int eldiv, bool swap)
        {
            var helper = MemCopy.GetMemcopyHelper(destIter.dataptr);
            helper.strided_byte_copy_init(destIter.dataptr,
                             destIter.strides[maxaxis],
                             srcIter.dataptr,
                             srcIter.strides[maxaxis],
                             elsize, eldiv);

            while (destIter.index < destIter.size)
            {

                helper.strided_byte_copy(destIter.dataptr.data_offset, srcIter.dataptr.data_offset, maxdim);

                if (swap)
                {
                    _strided_byte_swap(destIter.dataptr,
                                       destIter.strides[maxaxis],
                                       maxdim, elsize);
                }
                NpyArray_ITER_NEXT(destIter);
                NpyArray_ITER_NEXT(srcIter);
            }
        }

        internal static int _copy_from0d(NpyArray dest, NpyArray src, bool usecopy, bool swap)
        {
            byte[] aligned = null;
            VoidPtr sptr;
            npy_intp numcopies;
            int elsize, eldiv;
            int retval = -1;

            numcopies = NpyArray_SIZE(dest);
            if (numcopies < 1)
            {
                return 0;
            }

            elsize = src.descr.elsize;
            eldiv = GetDivSize(elsize);

            if (!NpyArray_ISALIGNED(src))
            {
                aligned = new byte[elsize];
                if (aligned == null)
                {
                    NpyErr_MEMORY();
                    return -1;
                }
                memcpy(new VoidPtr(aligned), src.data, elsize);
                usecopy = true;
                sptr = new VoidPtr(aligned);
            }
            else
            {
                sptr = new VoidPtr(src);
            }

            if ((dest.nd < 2) || NpyArray_ISONESEGMENT(dest))
            {
                npy_intp dstride;

                if (dest.nd == 1)
                {
                    dstride = dest.strides[0];
                }
                else
                {
                    dstride = elsize;
                }
                var helper = MemCopy.GetMemcopyHelper(dest.data);
                helper.strided_byte_copy_init(dest.data, dstride, sptr, 0, elsize, eldiv);

                helper.strided_byte_copy(dest.data.data_offset, sptr.data_offset, numcopies);
                if (swap)
                {
                    _strided_byte_swap(dest.data, dstride, numcopies, (int)elsize);
                }
            }
            else
            {
                NpyArrayIterObject dit;
                int axis = -1;

                dit = NpyArray_IterAllButAxis(dest, ref axis);
                if (dit == null)
                {
                    goto finish;
                }

                var helper = MemCopy.GetMemcopyHelper(dit.dataptr);
                helper.strided_byte_copy_init(dit.dataptr, NpyArray_STRIDE(dest, axis), sptr, 0, elsize, eldiv);

                while (dit.index < dit.size)
                {
                    helper.strided_byte_copy(dit.dataptr.data_offset, sptr.data_offset, NpyArray_DIM(dest, axis));
                    if (swap)
                    {
                        _strided_byte_swap(dit.dataptr, NpyArray_STRIDE(dest, axis),
                                           NpyArray_DIM(dest, axis), (int)elsize);
                    }
                    NpyArray_ITER_NEXT(dit);
                }
                Npy_DECREF(dit);
            }
            retval = 0;

            finish:
            if (aligned != null)
            {
                aligned = null;
            }
            return retval;
        }
        
    }
}
