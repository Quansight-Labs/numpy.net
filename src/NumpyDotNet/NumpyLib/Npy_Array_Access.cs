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
using System.IO;
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
        internal static void NpyArrayAccess_Dealloc(NpyObject_HEAD obj)
        {
            Debug.Assert(npy_defs.NPY_VALID_MAGIC == obj.nob_magic_number);
            Debug.Assert(0 == obj.nob_refcnt);

            // Clear the interface pointer because some deallocators temporarily increment the
            // reference count on the object for some reason.  This causes callbacks into the
            // managed world that we do not want (and is added overhead).
            obj.nob_interface = null;

            // Calls the type-specific deallocation route.  This routine releases any references
            // held by the object itself prior to actually freeing the memory. 
            obj.nob_type.ntp_dealloc(obj);
        }

        internal static void  NpyArrayAccess_Incref(NpyObject_HEAD obj)
        {
            Debug.Assert(npy_defs.NPY_VALID_MAGIC == obj.nob_magic_number);
            Npy_INCREF(obj);
        }


        internal static void NpyArrayAccess_Decref(NpyObject_HEAD obj)
        {
            Debug.Assert(npy_defs.NPY_VALID_MAGIC == obj.nob_magic_number);
            Npy_DECREF(obj);
        }


        // This function is here because the Npy_INTERFACE macro does some
        // magic with creating interface objects on an as-needed basis so it's
        // more code than simply reading the nob_interface field.

        internal static object NpyArrayAccess_ToInterface(NpyObject_HEAD obj)
        {
            Debug.Assert(npy_defs.NPY_VALID_MAGIC == obj.nob_magic_number);
            return Npy_INTERFACE(obj);
        }

        internal static void NpyArrayAccess_SetState(NpyArray self, int ndim, npy_intp[] dims, NPY_ORDER order, string srcPtr, int srcLen)
        {
            //Debug.Assert(Validate(self));
            //Debug.Assert(null != dims);
            //Debug.Assert(0 <= ndim);

            //// Clear existing data and references.  Typically these will be empty.
            //if (NpyArray_CHKFLAGS(self,NPYARRAYFLAGS.NPY_OWNDATA))
            //{
            //    if (null != NpyArray_BYTES(self))
            //    {
            //        NpyArray_free(NpyArray_BYTES(self));
            //    }
            //    self.flags &= ~NPYARRAYFLAGS.NPY_OWNDATA;
            //}
            //Npy_XDECREF(NpyArray_BASE_ARRAY(self));
            //NpyInterface_DECREF(NpyArray_BASE(self));
            //self.base_arr = null;
            //self.base_obj = null;

            //if (null != NpyArray_DIMS(self))
            //{
            //    NpyDimMem_FREE(self.dimensions);
            //    self.dimensions = null;
            //}

            //self.flags = NPYARRAYFLAGS.NPY_DEFAULT;
            //self.nd = ndim;
            //if (0 < ndim)
            //{
            //    self.dimensions = NpyDimMem_NEW(ndim);
            //    self.strides = NpyDimMem_NEW(ndim);
            //    memcpy(NpyArray_DIMS(self), dims, sizeof(npy_intp) * ndim);
            //    npy_array_fill_strides(NpyArray_STRIDES(self), dims, ndim,
            //        NpyArray_ITEMSIZE(self), order, ref self.flags));
            //}

            //npy_intp bytes = NpyArray_ITEMSIZE(self) * NpyArray_SIZE(self);
            //NpyArray_BYTES(self) = (char*)NpyArray_malloc(bytes);
            //NpyArray_FLAGS(self) |= NPY_OWNDATA;

            //if (null != srcPtr)
            //{
            //    // This is unpleasantly inefficent.  The input is a .NET string, which is 16-bit
            //    // unicode. Thus the data is encoded into alternating bytes so we can't use memcpy.
            //    char* destPtr = NpyArray_BYTES(self);
            //    char* destEnd = destPtr + bytes;
            //    const wchar_t* srcEnd = srcPtr + srcLen;
            //    while (destPtr < destEnd && srcPtr < srcEnd) *(destPtr++) = (char)*(srcPtr++);
            //}
            //else
            //{
            //    memset(NpyArray_BYTES(self), 0, bytes);
            //}
        }

     

        internal static void NpyArrayAccess_ZeroFill(NpyArray arr, npy_intp offset)
        {
            int itemsize = NpyArray_ITEMSIZE(arr);
            npy_intp size = NpyArray_SIZE(arr) * itemsize;
            npy_intp off = offset * itemsize;
            npy_intp fill_size = size - off;
            memset(arr.data, 0, fill_size);
        }

        internal static void NpyArrayAccess_ClearUPDATEIFCOPY(NpyArray self)
        {
            if (NpyArray_CHKFLAGS(self, NPYARRAYFLAGS.NPY_UPDATEIFCOPY))
            {
                if (self.base_arr != null)
                {
                    self.base_arr.flags &= ~NPYARRAYFLAGS.NPY_WRITEABLE;
                    Npy_DECREF(self.base_arr);
                    self.SetBase(null);
                }
                self.flags &= ~NPYARRAYFLAGS.NPY_UPDATEIFCOPY;
            }
        }


        //
        // Returns a view of a using prototype as the interfaceData when creating the wrapper.
        // This will return the same subtype as prototype and use prototype in the __array_finalize__ call.
        //
        internal static NpyArray NpyArrayAccess_ViewLike(NpyArray a, NpyArray prototype)
        {
            Npy_INCREF(a.descr);
            NpyArray ret = NpyArray_NewFromDescr(a.descr, a.nd, a.dimensions, a.strides, a.data, a.flags, false, null, Npy_INTERFACE(prototype));
            ret.SetBase(a);
            Npy_INCREF(a);

            return ret;
        }

        internal static NpyArray NpyArrayAccess_FromFile(string fileName, NpyArray_Descr dtype, int count, string sep)
        {
            Debug.Assert(Validate(dtype));
            FileInfo fp = new FileInfo(fileName);
            NpyArray result;

            Npy_INCREF(dtype);
            result = string.IsNullOrEmpty(sep) ? NpyArray_FromBinaryFile(fp, dtype, count) :  NpyArray_FromTextFile(fp, dtype, count, sep);
            return result;
        }

        internal static NpyArray NpyArrayAccess_FromStream(Stream fileStream, NpyArray_Descr dtype, int count, string sep)
        {
            Debug.Assert(Validate(dtype));
            NpyArray result;

            Npy_INCREF(dtype);
            result = string.IsNullOrEmpty(sep) ? NpyArray_FromBinaryStream(fileStream, dtype, count) : NpyArray_FromTextStream(fileStream, dtype, count, sep);
            return result;
        }


        internal static void NpyArrayAccess_ToFile(NpyArray array, string fileName, string sep, string format)
        {
            FileInfo fp = new FileInfo(fileName);

            if (string.IsNullOrEmpty(sep))
            {
                NpyArray_ToBinaryFile(array, fp);
            }
            else
            {
                NpyArray_ToTextFile(array, fp, sep, format);
            }
  
        }

        internal static void NpyArrayAccess_ToStream(NpyArray array, Stream fs, string sep, string format)
        {
            if (string.IsNullOrEmpty(sep))
            {
                NpyArray_ToBinaryStream(array, fs);
            }
            else
            {
                NpyArray_ToTextStream(array, fs, sep, format);
            }
        }

        internal static int NpyArrayAccess_Fill(NpyArray arr)
        {
            NpyArray_FillFunc fill = arr.descr.f.fill;
            if (fill == null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "no fill-function for data-type");
                return -1;
            }
            fill(arr.data, NpyArray_SIZE(arr), arr);
            if (NpyErr_Occurred())
            {
                return -1;
            }
            return 0;
        }

        internal static void NpyArrayAccess_DescrReplaceFields(NpyArray_Descr descr, List<string> nameslist, NpyDict fields)
        {
            Debug.Assert(Validate(descr));

            if (null != descr.names)
            {
                NpyArray_DescrDeallocNamesAndFields(descr);
            }
            descr.names = nameslist;
            descr.fields = fields;
        }


        internal static int NpyArrayAccess_AddField(NpyDict fields, List<string> names, int i, string name, NpyArray_Descr fieldType, int offset, string title)
        {
            if (fields.bucketArray.ContainsKey(name))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "two fields with the same name");
                return -1;
            }
            names[i] = name.ToString();
            NpyArray_DescrSetField(fields, names[i], fieldType, offset, title);
            return 0;
        }

        internal static int NpyArrayAccess_GetDescrField(NpyArray_Descr descr, string fieldName, ref NpyArray_DescrField pField)
        {
            if (descr.names == null)
            {
                return -1;
            }
            NpyArray_DescrField value = (NpyArray_DescrField)NpyDict_Get(descr.fields, fieldName);
            if (value == null)
            {
                return -1;
            }
            pField = value;
            return 0;
        }

 
        internal static void NpyArrayAccess_DescrDestroyNames(List<string> names, int n)
        {
            for (int i = 0; i < n; i++)
            {
                if (names[i] != null)
                {
                    names[i] = null;
                }
            }
            npy_free(names);
        }

        internal static int NpyArrayAccess_GetFieldOffset(NpyArray_Descr descr, string fieldName, ref NpyArray_Descr pDescr)
        {
            if (descr.names == null)
            {
                return -1;
            }
            NpyArray_DescrField value = (NpyArray_DescrField)NpyDict_Get(descr.fields, fieldName);
            if (value == null)
            {
                return -1;
            }
            pDescr = value.descr;
            return value.offset;
        }

        internal static NpyDict_Iter NpyArrayAccess_DictAllocIter()
        {
            NpyDict_Iter iter = new NpyDict_Iter();
            NpyDict_IterInit(iter);
            return iter;
        }


        internal static bool NpyArrayAccess_DictNext(NpyDict dict, NpyDict_Iter iter, NpyDict_KVPair KVPair)
        {
            return NpyDict_IterNext(dict, iter, KVPair);
        }


        internal static void NpyArrayAccess_DictFreeIter(NpyDict_Iter iter)
        {
            npy_free(iter);
        }

        internal static NpyArray_Descr NpyArrayAccess_InheritDescriptor(NpyArray_Descr type, NpyArray_Descr conv)
        {
            NpyArray_Descr nw = NpyArray_DescrNew(type);
            if (nw.elsize > 0 && nw.elsize != conv.elsize)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "mismatch in size of old and new data-descriptor");
                Npy_DECREF(nw);
                return null;
            }
            nw.elsize = conv.elsize;
            if (conv.names != null)
            {
                nw.names = NpyArray_DescrNamesCopy(conv.names);
                nw.fields = NpyArray_DescrFieldsCopy(conv.fields);
            }
            nw.flags = conv.flags;
            return nw;
        }


        internal static VoidPtr NpyArrayAccess_IterGoto1D(NpyArrayIterObject it, npy_intp index)
        {
            if (index < 0)
            {
                index += it.size;
            }
            if (index < 0 || index >= it.size)
            {
                string buf = string.Format("index out of bounds 0<=index<{0}", (long)it.size);
                NpyErr_SetString( npyexc_type.NpyExc_IndexError, buf);
                return null;
            }
            NpyArray_ITER_RESET(it);
            NpyArray_ITER_GOTO1D(it, index);
            return it.dataptr;
        }

        internal static npy_intp[] NpyArrayAccess_IterCoords(NpyArrayIterObject self)
        {
            if (self.contiguous)
            {
                /*
                 * coordinates not kept track of ---
                 * need to generate from index
                 */
                npy_intp val;
                int nd = self.ao.nd;
                int i;
                val = self.index;
                for (i = 0; i < nd; i++)
                {
                    if (self.factors[i] != 0)
                    {
                        self.coordinates[i] = val / self.factors[i];
                        val = val % self.factors[i];
                    }
                    else
                    {
                        self.coordinates[i] = 0;
                    }
                }
            }
            return self.coordinates;
        }

        internal static VoidPtr NpyArrayAccess_DupZeroElem(NpyArray arr)
        {
            VoidPtr mem = NpyDataMem_NEW(arr.ItemType, (ulong)NpyArray_ITEMSIZE(arr));
            memcpy(mem, NpyArray_DATA(arr), NpyArray_ITEMSIZE(arr));
            return mem;
        }

        internal static  NpyArrayMultiIterObject NpyArrayAccess_MultiIterFromArrays(NpyArray []arrays, int n)
        {
            return NpyArray_MultiIterFromArrays(arrays, n, 0);
        }

        internal static void NpyArrayAccess_CopySwapIn(NpyArray arr, Int64 offset, VoidPtr data, bool swap)
        {
            arr.descr.f.copyswap(arr.data + offset, data, swap, arr);
        }

        internal static void NpyArrayAccess_CopySwapOut(NpyArray arr, Int64 offset, VoidPtr data, bool swap)
        {
            arr.descr.f.copyswap(data, arr.data + offset, swap, arr);
        }

        // Similar to above, but does not handle string, void or other flexibly sized types because it can't pass
        // an array pointer in.  This is specifically used for fixed scalar types.
        internal static void NpyArrayAccess_CopySwapScalar(NpyArray_Descr descr, VoidPtr dest, VoidPtr src, bool swap)
        {
            descr.f.copyswap(dest, src, swap, null);
        }

    }
}
