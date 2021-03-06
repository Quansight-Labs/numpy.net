﻿/*
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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
using npy_ucs4 = System.Int64;
#else
using npy_intp = System.Int32;
using npy_ucs4 = System.Int32;
#endif

using size_t = System.UInt64;
using npy_datetime = System.Int64;
using npy_timedelta = System.Int64;

using System.IO;
using System.Diagnostics;

namespace NumpyLib
{
    public enum NPY_ORDER : int
    {
        NPY_ANYORDER = -1,
        NPY_CORDER = 0,
        NPY_FORTRANORDER = 1,
        NPY_KORDER = 2,
        NPY_KEEPORDER = 2,
    };


    internal class numpyAPI
    {
        #region npy_arrayobject
        internal static npy_intp NpyArray_Size(NpyArray op)
        {
            return numpyinternal.NpyArray_Size(op);
        }

        internal static int NpyArray_ElementStrides(NpyArray arr)
        {
            return numpyinternal.NpyArray_ElementStrides(arr);
        }

        //*
        //* This routine checks to see if newstrides(of length nd) will not
        //* ever be able to walk outside of the memory implied numbytes and offset.
        //*
        //* The available memory is assumed to start at -offset and proceed
        //* to numbytes-offset.The strides are checked to ensure
        //* that accessing memory using striding will not try to reach beyond
        //* this memory for any of the axes.
        //*
        //* If numbytes is 0 it will be calculated using the dimensions and
        //* element-size.
        //*
        //* This function checks for walking beyond the beginning and right-end
        //* of the buffer and therefore works for any integer stride(positive
        //* or negative).
        //*
        internal static bool NpyArray_CheckStrides(int elsize, int nd, npy_intp numbytes, npy_intp offset, npy_intp[] dims, npy_intp[] newstrides)
        {
            return numpyinternal.NpyArray_CheckStrides(elsize, nd, numbytes, offset, dims, newstrides);
        }


        internal static void NpyArray_ForceUpdate(NpyArray self)
        {
            numpyinternal.NpyArray_ForceUpdate(self);
        }

        internal static NpyArray NpyArray_CompareStringArrays(NpyArray a1, NpyArray a2, int cmp_op, int rstrip)
        {
            //return numpyinternal.NpyArray_CompareStringArrays(a1, a2, cmp_op, rstrip);
            
            // not currently supporting strings
            return null;
        }

        ///* Deallocs & destroy's the array object.
        //*  Returns whether or not we did an artificial incref
        //*  so we can keep track of the total refcount for debugging.
        //*/
        internal static int NpyArray_dealloc(NpyArray self)
        {
            return numpyinternal.NpyArray_dealloc(self);
        }

        internal static npy_intp[] GetViewOffsets(NpyArray self)
        {
            return numpyinternal.GetViewOffsets(self);
        }

        internal static npy_intp[] GetViewOffsets(NpyArrayIterObject iter, npy_intp count)
        {
            return numpyinternal.GetViewOffsets(iter, count);
        }

        #endregion

        #region npy_array_access

        internal static NpyDict_Iter NpyArrayAccess_DictAllocIter()
        {
            return numpyinternal.NpyArrayAccess_DictAllocIter();
        }

        internal static bool NpyArrayAccess_DictNext(NpyDict dict, NpyDict_Iter iter, NpyDict_KVPair KVPair)
        {
            return numpyinternal.NpyArrayAccess_DictNext(dict, iter, KVPair);
        }

        internal static void NpyArrayAccess_DictFreeIter(NpyDict_Iter iter)
        {
            numpyinternal.NpyArrayAccess_DictFreeIter(iter);
        }

        internal static void NpyArrayAccess_Dealloc(NpyObject_HEAD obj)
        {
            numpyinternal.NpyArrayAccess_Dealloc(obj);
        }

        internal static void NpyArrayAccess_Incref(NpyObject_HEAD obj)
        {
            numpyinternal.NpyArrayAccess_Incref(obj);
        }


        internal static void NpyArrayAccess_Decref(NpyObject_HEAD obj)
        {
            numpyinternal.NpyArrayAccess_Decref(obj);
        }


        // This function is here because the Npy_INTERFACE macro does some
        // magic with creating interface objects on an as-needed basis so it's
        // more code than simply reading the nob_interface field.

        internal static object NpyArrayAccess_ToInterface(NpyObject_HEAD obj)
        {
            return numpyinternal.NpyArrayAccess_ToInterface(obj);
        }

        internal static void NpyArrayAccess_SetState(NpyArray self, int ndim, npy_intp[] dims, NPY_ORDER order, string srcPtr, int srcLen)
        {
            numpyinternal.NpyArrayAccess_SetState(self, ndim, dims, order, srcPtr, srcLen);
        }

        internal static void NpyArrayAccess_ZeroFill(NpyArray arr, npy_intp offset)
        {
            numpyinternal.NpyArrayAccess_ZeroFill(arr, offset);
        }

        internal static void NpyArrayAccess_ClearUPDATEIFCOPY(NpyArray self)
        {
            numpyinternal.NpyArrayAccess_ClearUPDATEIFCOPY(self);
        }

        internal static NpyArray NpyArrayAccess_ViewLike(NpyArray a, NpyArray prototype)
        {
            return numpyinternal.NpyArrayAccess_ViewLike(a, prototype);
        }

        internal static NpyArray NpyArrayAccess_FromFile(string fileName, NpyArray_Descr dtype, int count, string sep)
        {
            return numpyinternal.NpyArrayAccess_FromFile(fileName, dtype, count, sep);
        }

        internal static NpyArray NpyArrayAccess_FromStream(Stream fileStream, NpyArray_Descr dtype, int count, string sep)
        {
            return numpyinternal.NpyArrayAccess_FromStream(fileStream, dtype, count, sep);
        }

        internal static void NpyArrayAccess_ToFile(NpyArray array, string fileName, string sep, string format)
        {
            numpyinternal.NpyArrayAccess_ToFile(array, fileName, sep, format);
        }

        internal static void NpyArrayAccess_ToStream(NpyArray array, Stream fileName, string sep, string format)
        {
            numpyinternal.NpyArrayAccess_ToStream(array, fileName, sep, format);
        }

  
        internal static int NpyArrayAccess_Fill(NpyArray arr)
        {
            return numpyinternal.NpyArrayAccess_Fill(arr);
        }

        internal static void NpyArrayAccess_DescrReplaceFields(NpyArray_Descr descr, List<string> nameslist, NpyDict fields)
        {
            numpyinternal.NpyArrayAccess_DescrReplaceFields(descr, nameslist, fields);
        }

        internal static int NpyArrayAccess_AddField(NpyDict fields, List<string> names, int i, string name, NpyArray_Descr fieldType, int offset, string title)
        {
            return numpyinternal.NpyArrayAccess_AddField(fields, names, i, name, fieldType, offset, title);
        }

        internal static int NpyArrayAccess_GetDescrField(NpyArray_Descr descr, string fieldName, ref NpyArray_DescrField pField)
        {
            return numpyinternal.NpyArrayAccess_GetDescrField(descr, fieldName, ref pField);
        }


        internal static void NpyArrayAccess_DescrDestroyNames(List<string> names, int n)
        {
            numpyinternal.NpyArrayAccess_DescrDestroyNames(names, n);
        }


        internal static int NpyArrayAccess_GetFieldOffset(NpyArray_Descr descr, string fieldName, ref NpyArray_Descr pDescr)
        {
            return numpyinternal.NpyArrayAccess_GetFieldOffset(descr, fieldName, ref pDescr);
        }

#endregion

        #region npy_arraytypes
        internal static NpyArray_Descr NpyArray_DescrFromType(NPY_TYPES type)
        {
            return numpyinternal.NpyArray_DescrFromType(type);
        }
        #endregion

        #region npy_buffer

        internal static int npy_array_getsegcount(NpyArray self, ref size_t lenp)
        {
            return numpyinternal.npy_array_getsegcount(self, ref lenp);
        }

        internal static int npy_array_getreadbuf(NpyArray self, size_t segment, ref VoidPtr ptrptr)
        {
            return numpyinternal.npy_array_getreadbuf(self, segment, ref ptrptr);
        }

        internal static int npy_array_getwritebuf(NpyArray self, size_t segment, ref VoidPtr ptrptr)
        {
            return numpyinternal.npy_array_getwritebuf(self, segment, ref ptrptr);
        }


        #endregion

        #region npy_calculation

 
        internal static NpyArray NpyArray_PerformNumericOperation(UFuncOperation operationType, NpyArray x1Array, NpyArray x2Array, NpyArray outArray, NpyArray whereFilter)
        {
            return numpyinternal.NpyArray_PerformNumericOperation(operationType, x1Array, x2Array, outArray, whereFilter);
        }

        internal static NpyArray NpyArray_PerformOuterOp(NpyArray a, NpyArray b, NpyArray dest, UFuncOperation operationType)
        {
            return numpyinternal.PerformOuterOpArray(a, b, dest, operationType);
        }

        internal static NpyArray NpyArray_PerformReduceOp(NpyArray a, int axis, UFuncOperation ops, NPY_TYPES rtype, NpyArray @out, bool keepdims)
        {
            return numpyinternal.PerformReduceOpArray(a, axis, ops, rtype, @out, keepdims);
        }

        internal static NpyArray NpyArray_PerformReduceAtOp(NpyArray a, NpyArray indices, int axis, UFuncOperation ops, NPY_TYPES rtype, NpyArray @out)
        {
            return numpyinternal.PerformReduceAtOpArray(a, indices, axis, ops, rtype, @out);
        }

        internal static NpyArray NpyArray_PerformAccumulateOp(NpyArray a, int axis, UFuncOperation ops, NPY_TYPES rtype, NpyArray @out)
        {
            return numpyinternal.PerformAccumulateOpArray(a, axis, ops, rtype, @out);
        }

        internal static NpyArray NpyArray_ArgMax(NpyArray op, int axis, NpyArray _out)
        {
            return numpyinternal.NpyArray_ArgMax(op, axis, _out);
        }

        internal static NpyArray NpyArray_ArgMin(NpyArray op, int axis, NpyArray _out)
        {
            return numpyinternal.NpyArray_ArgMin(op, axis, _out);
        }

        internal static NpyArray NpyArray_Max(NpyArray self, int axis, NpyArray _out, bool keepdims)
        {
            return numpyinternal.NpyArray_Max(self, axis, _out, keepdims);
        }

        internal static NpyArray NpyArray_Min(NpyArray self, int axis, NpyArray _out, bool keepdims)
        {
            return numpyinternal.NpyArray_Min(self, axis, _out, keepdims);
        }

        internal static NpyArray NpyArray_Sum(NpyArray self, int axis, NPY_TYPES rtype, NpyArray _out, bool keepdims)
        {
            return numpyinternal.NpyArray_Sum(self, axis, rtype, _out, keepdims);
        }

        internal static NpyArray NpyArray_Prod(NpyArray self, int axis, NPY_TYPES rtype, NpyArray _out, bool keepdims)
        {
            return numpyinternal.NpyArray_Prod(self, axis, rtype, _out, keepdims);
        }

        internal static NpyArray NpyArray_CumSum(NpyArray self, int axis, NPY_TYPES rtype, NpyArray _out)
        {
            return numpyinternal.NpyArray_CumSum(self, axis, rtype, _out);
        }

        internal static NpyArray NpyArray_CumProd(NpyArray self, int axis, NPY_TYPES rtype, NpyArray _out)
        {
            return numpyinternal.NpyArray_CumProd(self, axis, rtype, _out);
        }

        internal static NpyArray NpyArray_Floor(NpyArray self, NpyArray _out)
        {
            return numpyinternal.NpyArray_Floor(self, _out);
        }

        internal static NpyArray NpyArray_IsNaN(NpyArray self)
        {
            return numpyinternal.NpyArray_IsNaN(self);
        }

        internal static NpyArray NpyArray_Any(NpyArray self, int axis, NpyArray _out, bool keepdims)
        {
            return numpyinternal.NpyArray_Any(self, axis, _out, keepdims);
        }

        internal static NpyArray NpyArray_All(NpyArray self, int axis, NpyArray _out, bool keepdims)
        {
            return numpyinternal.NpyArray_All(self, axis, _out, keepdims);
        }

        internal static NpyArray NpyArray_NumericOpUpscaleSourceArray(NpyArray srcArray, NpyArray operandArray)
        {
            return numpyinternal.NpyArray_NumericOpUpscaleSourceArray(srcArray, operandArray);
        }

        internal static NpyArray NpyArray_NumericOpUpscaleSourceArray(NpyArray srcArray, npy_intp[] dimensions, int nd)
        {
            return numpyinternal.NpyArray_NumericOpUpscaleSourceArray(srcArray, dimensions, nd);
        }

        #endregion

        #region npy_common

        internal static bool Npy_IsAligned(NpyArray ap)
        {
            return numpyinternal.Npy_IsAligned(ap);
        }

        internal static bool Npy_IsWriteable(NpyArray ap)
        {
            return numpyinternal.Npy_IsWriteable(ap);
        }

        internal static VoidPtr NpyArray_Index2Ptr(NpyArray mp, npy_intp i)
        {
            return numpyinternal.NpyArray_Index2Ptr(mp, i);
        }

#endregion


#region npy_convert

        internal static NpyArray NpyArray_View(NpyArray self, NpyArray_Descr type, object subtype)
        {
            return numpyinternal.NpyArray_View(self, type, subtype);
        }

        internal static int NpyArray_SetDescr(NpyArray self, NpyArray_Descr newtype)
        {
            return numpyinternal.NpyArray_SetDescr(self, newtype);
        }

        internal static NpyArray NpyArray_NewCopy(NpyArray m1, NPY_ORDER order)
        {
            return numpyinternal.NpyArray_NewCopy(m1, order);
        }

        internal static int NpyArray_ToBinaryFile(NpyArray self, FileInfo fp)
        {
            return numpyinternal.NpyArray_ToBinaryFile(self, fp);
        }


        internal static int NpyArray_FillWithScalar(NpyArray arr, NpyArray zero_d_array)
        {
            return numpyinternal.NpyArray_FillWithScalar(arr, zero_d_array);
        }

#endregion

#region MemUtilities


        internal static bool MemCpy(VoidPtr Dest, npy_intp DestOffset, VoidPtr Src, npy_intp SrcOffset, long Length)
        {
            return MemCopy.MemCpy(Dest, DestOffset, Src, SrcOffset, Length);
        }

        internal static VoidPtr Alloc_NewBuffer(NPY_TYPES type_num, ulong size)
        {
            return numpyinternal.NpyDataMem_NEW(type_num, size, true);
        }

        internal static VoidPtr Alloc_NewArray(NPY_TYPES type_num, size_t size)
        {
            return numpyinternal.NpyDataMem_NEW(type_num, size, false);
        }

        private class NpyArrayOffsetHelper
        {
            NpyArray srcArray;
            long index;

            internal NpyArrayOffsetHelper(NpyArray srcArray)
            {
                this.srcArray = srcArray;
                this.index = 0;
            }

            internal npy_intp CalculateOffset(npy_intp index)
            {
                if (index < 0)
                {
                    index = srcArray.dimensions[0] + index;
                }

                this.index = index;
                return CalculateOffset(srcArray, 0, 0);
            }

            private npy_intp CalculateOffset(NpyArray arr, int dimIdx, npy_intp offset)
            {
                npy_intp CalculatedIndex = 0;

                //Console.WriteLine(string.Format("dimIdx:{0}, offset:{1}, arr.nd:{2}", dimIdx, offset, arr.nd));

                if (dimIdx == arr.nd)
                {
                    //Console.WriteLine("Found offset:{0}, {1}", offset, offset / arr.ItemSize);
                    index--;
                    return (offset >> arr.ItemDiv);
                }
                else
                {
                    for (int i = 0; i < arr.dimensions[dimIdx]; i++)
                    {
                        npy_intp lsrc_offset = offset + arr.strides[dimIdx] * i;
                        CalculatedIndex = CalculateOffset(arr, dimIdx + 1, lsrc_offset);
                        if (index < 0)
                        {
                            return CalculatedIndex;
                        }
                    }
                }
                return CalculatedIndex;
            }

        }

        internal static object GetItem(NpyArray arr, npy_intp index)
        {
            var offsetHelper = new NpyArrayOffsetHelper(arr);
            npy_intp CalculatedOffset = offsetHelper.CalculateOffset(index);

            return arr.descr.f.getitem(CalculatedOffset * arr.ItemSize, arr);
        }

        internal static void GetItems(NpyArray arr, object[] buffer, npy_intp index, npy_intp length)
        {
            var offsetHelper = new NpyArrayOffsetHelper(arr);

            int i = 0;
            for (; index < length; index++)
            {
                npy_intp CalculatedOffset = offsetHelper.CalculateOffset(index);

                buffer[i++] = arr.descr.f.getitem(CalculatedOffset * arr.ItemSize, arr);
            }
  
        }

        internal static npy_intp SetItem(NpyArray arr, npy_intp index, object value)
        {
            var offsetHelper = new NpyArrayOffsetHelper(arr);
            npy_intp CalculatedOffset = offsetHelper.CalculateOffset(index);

            return arr.descr.f.setitem(CalculatedOffset * arr.ItemSize, value, arr);
        }

        internal static npy_intp SetIndex(VoidPtr obj, npy_intp index, object value)
        {
            return numpyinternal.SetIndex(obj, index, value);
        }

        #endregion

        #region npy_convert_datatype

        internal static void DefaultCastFunction(VoidPtr Src, VoidPtr Dest, npy_intp srclen, NpyArray srcArray, NpyArray destArray)
        {
            CastFunctions.DefaultCastFunction(Src, Dest, srclen, srcArray, destArray);
            return;
        }

        internal static VoidPtr ConvertToDesiredArrayType(VoidPtr Src, int SrcOffset, int Length, NPY_TYPES type_num)
        {
            return ArrayConversions.ConvertToDesiredArrayType(Src, SrcOffset, Length, type_num);
        }

        internal static NpyArray_VectorUnaryFunc NpyArray_GetCastFunc(NpyArray_Descr descr, NPY_TYPES type_num)
        {
            return numpyinternal.NpyArray_GetCastFunc(descr, type_num);
        }

        internal static int NpyArray_CastTo(NpyArray dest, NpyArray src)
        {
            return numpyinternal.NpyArray_CastTo(dest, src);
        }

        internal static NpyArray NpyArray_CastToType(NpyArray mp, NpyArray_Descr at, bool fortran)
        {
            return numpyinternal.NpyArray_CastToType(mp, at, fortran);
        }

        internal static int NpyArray_CastAnyTo(NpyArray _out, NpyArray mp)
        {
            return numpyinternal.NpyArray_CastAnyTo(_out, mp);
        }

        internal static bool NpyArray_CanCastSafely(NPY_TYPES fromtype, NPY_TYPES totype)
        {
            return numpyinternal.NpyArray_CanCastSafely(fromtype, totype);
        }

        internal static bool NpyArray_CanCastTo(NpyArray_Descr from, NpyArray_Descr to)
        {
            return numpyinternal.NpyArray_CanCastTo(from, to);
        }

        internal static bool NpyArray_ValidType(NPY_TYPES type)
        {
            return numpyinternal.NpyArray_ValidType(type);
        }

#endregion

#region npy_ctors

        internal static void npy_byte_swap_vector(VoidPtr p, npy_intp n, int size)
        {
           numpyinternal.npy_byte_swap_vector(p, n, size);
        }

        internal static int _flat_copyinto(NpyArray dst, NpyArray src, NPY_ORDER order)
        {
            return numpyinternal._flat_copyinto(dst, src, order);
        }
        
        internal static int NpyArray_MoveInto(NpyArray dest, NpyArray src)
        {
            return numpyinternal.NpyArray_MoveInto(dest, src);
        }

        internal static NpyArray NpyArray_Combine(NpyArray arr1, NpyArray arr2)
        {
            return numpyinternal.Combine(arr1, arr2);
        }

        internal static int NpyArray_CombineInto(NpyArray dest, IEnumerable<NpyArray> ArraysToCombine)
        {
            return numpyinternal.NpyArray_CombineInto(dest, ArraysToCombine);
        }

        internal static NpyArray NpyArray_CheckFromArray(NpyArray arr, NpyArray_Descr descr, NPYARRAYFLAGS requires)
        {
            return numpyinternal.NpyArray_CheckFromArray(arr, descr, requires);
        }

        internal static NpyArray NpyArray_CheckAxis(NpyArray arr, ref int axis, NPYARRAYFLAGS flags)
        {
            return numpyinternal.NpyArray_CheckAxis(arr, ref axis, flags);
        }

        internal static NpyArray NpyArray_NewFromDescr(NpyArray_Descr descr, int nd, npy_intp[] dims, npy_intp[] strides, VoidPtr data,
                  NPYARRAYFLAGS flags, bool ensureArray, object subtype, object interfaceData)
        {
            return numpyinternal.NpyArray_NewFromDescr(descr, nd, dims, strides, data, flags, ensureArray, subtype, interfaceData);
        }

        internal static NpyArray NpyArray_New(object subtype, int nd, npy_intp[] dims, NPY_TYPES type_num, npy_intp[] strides, VoidPtr data, int itemsize, NPYARRAYFLAGS flags, object obj)
        {
            return numpyinternal.NpyArray_New(subtype, nd, dims, type_num, strides, data, itemsize, flags, obj);
        }

        internal static NpyArray NpyArray_Alloc(NpyArray_Descr descr, int nd, npy_intp[] dims, bool is_fortran, object interfaceData)
        {
            return numpyinternal.NpyArray_Alloc(descr, nd, dims, is_fortran, interfaceData);
        }

        internal static NpyArray NpyArray_NewView(NpyArray_Descr descr, int nd, npy_intp[] dims, npy_intp[] strides, NpyArray array, npy_intp offset, bool ensure_array)
        {
            return numpyinternal.NpyArray_NewView(descr, nd, dims, strides, array, offset, ensure_array);
        }

        internal static NpyArray NpyArray_FromArray(NpyArray arr, NpyArray_Descr newtype, NPYARRAYFLAGS flags)
        {
            return numpyinternal.NpyArray_FromArray(arr, newtype, flags);
        }

        internal static int NpyArray_CopyAnyInto(NpyArray dest, NpyArray src)
        {
            return numpyinternal.NpyArray_CopyAnyInto(dest, src);
        }

        internal static int NpyArray_CopyInto(NpyArray dest, NpyArray src)
        {
            return numpyinternal.NpyArray_CopyInto(dest, src);
        }

        internal static NpyArray NpyArray_FromTextFile(FileInfo fp, NpyArray_Descr dtype, npy_intp num, string sep)
        {
            return numpyinternal.NpyArray_FromTextFile(fp, dtype, num, sep);
        }

        internal static NpyArray NpyArray_FromString(string data, npy_intp slen, NpyArray_Descr dtype, npy_intp num, string sep)
        {
            return numpyinternal.NpyArray_FromString(data, slen, dtype, num, sep);
        }

        internal static NpyArray NpyArray_FromBinaryFile(FileInfo fp, NpyArray_Descr dtype, npy_intp num)
        {
            return numpyinternal.NpyArray_FromBinaryFile(fp, dtype, num);
        }

 
#endregion


#region npy_descriptor

        internal static NpyArray_Descr NpyArray_DescrNewFromType(NPY_TYPES type_num)
        {
            return numpyinternal.NpyArray_DescrNewFromType(type_num);
        }

        internal static NpyArray_Descr NpyArray_DescrNewSubarray(NpyArray_Descr baseDescr, int ndim, npy_intp[] dims)
        {
            return numpyinternal.NpyArray_DescrNewSubarray(baseDescr, ndim, dims);
        }


        internal static void NpyArray_DescrReplaceSubarray(NpyArray_Descr descr, NpyArray_Descr baseDescr, int ndim, npy_intp[] dims)
        {
            numpyinternal.NpyArray_DescrReplaceSubarray(descr, baseDescr, ndim, dims);
        }

        internal static NpyArray_Descr NpyArray_DescrNew(NpyArray_Descr _base)
        {
            return numpyinternal.NpyArray_DescrNew(_base);
        }

        internal static NpyArray_Descr NpyArray_SmallType(NpyArray_Descr chktype, NpyArray_Descr mintype)
        {
            return numpyinternal.NpyArray_SmallType(chktype, mintype);
        }

        internal static NpyArray_Descr NpyArray_DescrFromArray(NpyArray ap, NpyArray_Descr mintype)
        {
            return numpyinternal.NpyArray_DescrFromArray(ap, mintype);
        }

        internal static NpyArray_ArrayDescr NpyArray_DupSubarray(NpyArray_ArrayDescr src)
        {
            return numpyinternal.NpyArray_DupSubarray(src);
        }

        internal static void NpyArray_DescrDestroy(NpyArray_Descr self)
        {
            numpyinternal.NpyArray_DescrDestroy(self);
        }
        internal static void NpyArray_DestroySubarray(NpyArray_ArrayDescr self)
        {
            numpyinternal.NpyArray_DestroySubarray(self);
        }
        internal static NpyArray_Descr NpyArray_DescrNewByteorder(NpyArray_Descr self, char newendian)
        {
            return numpyinternal.NpyArray_DescrNewByteorder(self, newendian);
        }
        internal static List<string> NpyArray_DescrAllocNames(int n)
        {
            return numpyinternal.NpyArray_DescrAllocNames(n);
        }
        internal static NpyDict NpyArray_DescrAllocFields()
        {
            return numpyinternal.NpyArray_DescrAllocFields();
        }
        internal static void NpyArray_DescrDeallocNamesAndFields(NpyArray_Descr self)
        {
            numpyinternal.NpyArray_DescrDeallocNamesAndFields(self);
        }
        internal static int NpyArray_DescrReplaceNames(NpyArray_Descr self, List<string> nameslist)
        {
            return numpyinternal.NpyArray_DescrReplaceNames(self, nameslist);
        }
        internal static void NpyArray_DescrSetNames(NpyArray_Descr self, List<string> nameslist)
        {
            numpyinternal.NpyArray_DescrSetNames(self, nameslist);
        }
        internal static void NpyArray_DescrSetField(NpyDict self, string key, NpyArray_Descr descr, int offset, string title)
        {
            numpyinternal.NpyArray_DescrSetField(self, key, descr, offset, title);
        }
        internal static NpyDict NpyArray_DescrFieldsCopy(NpyDict fields)
        {
            return numpyinternal.NpyArray_DescrFieldsCopy(fields);
        }
        internal static List<string> NpyArray_DescrNamesCopy(List<string> names)
        {
            return numpyinternal.NpyArray_DescrNamesCopy(names);
        }

        internal static bool npy_arraydescr_isnative(NpyArray_Descr self)
        {
            return numpyinternal.npy_arraydescr_isnative(self);
        }
  

        internal static NpyArray_Descr NpyArrayAccess_InheritDescriptor(NpyArray_Descr type, NpyArray_Descr conv)
        {
            return numpyinternal.NpyArrayAccess_InheritDescriptor(type, conv);
        }

        internal static VoidPtr NpyArrayAccess_IterGoto1D(NpyArrayIterObject it, npy_intp index)
        {
            return numpyinternal.NpyArrayAccess_IterGoto1D(it, index);
        }

        internal static npy_intp[] NpyArrayAccess_IterCoords(NpyArrayIterObject self)
        {
            return numpyinternal.NpyArrayAccess_IterCoords(self);
        }

        internal static NpyArrayMultiIterObject NpyArrayAccess_MultiIterFromArrays(NpyArray[] arrays, int n)
        {
            return numpyinternal.NpyArrayAccess_MultiIterFromArrays(arrays, n);
        }

#endregion

#region npy_flagsobject
        internal static void NpyArray_UpdateFlags(NpyArray ret, NPYARRAYFLAGS flagmask)
        {
            numpyinternal.NpyArray_UpdateFlags(ret, flagmask);
        }
#endregion

#region npy_getset

        internal static int NpyArray_SetShape(NpyArray self, NpyArray_Dims newdims)
        {
            return numpyinternal.NpyArray_SetShape(self, newdims);
        }
        internal static int NpyArray_SetStrides(NpyArray self, NpyArray_Dims newstrides)
        {
            return numpyinternal.NpyArray_SetStrides(self, newstrides);
        }
        internal static NpyArray NpyArray_GetReal(NpyArray self)
        {
            return numpyinternal.NpyArray_GetReal(self);
        }
        internal static NpyArray NpyArray_GetImag(NpyArray self)
        {
            return numpyinternal.NpyArray_GetImag(self);
        }
#endregion

#region npy_index

        internal static void NpyArray_IndexDealloc(NpyIndex[] indexes, int n)
        {
            numpyinternal.NpyArray_IndexDealloc(indexes, n);
        }
        internal static int NpyArray_IndexExpandBool(NpyIndex []indexes, int n, NpyIndex[] out_indexes)
        {
            return numpyinternal.NpyArray_IndexExpandBool(indexes, n, out_indexes);
        }
        internal static int NpyArray_IndexBind(NpyIndex[] indexes, int n, npy_intp[] dimensions, int nd, NpyIndex[] out_indexes)
        {
            return numpyinternal.NpyArray_IndexBind(indexes, n, dimensions, nd, out_indexes);
        }
        internal static int NpyArray_IndexToDimsEtc(NpyArray array, NpyIndex[] indexes, int n,
                            npy_intp[] dimensions, npy_intp[] strides, ref npy_intp offset_ptr, bool allow_arrays)
        {
            return numpyinternal.NpyArray_IndexToDimsEtc(array, indexes, n, dimensions, strides, ref offset_ptr, allow_arrays);
        }
        internal static npy_intp NpyArray_SliceSteps(NpyIndexSlice slice)
        {
            return numpyinternal.NpyArray_SliceSteps(slice);
        }

#endregion

#region npy_item_selection

        internal static NpyArray NpyArray_TakeFrom(NpyArray self0, NpyArray indices0, int axis, NpyArray ret, NPY_CLIPMODE clipmode)
        {
            return numpyinternal.NpyArray_TakeFrom(self0, indices0, axis, ret, clipmode);
        }
        internal static int NpyArray_PutTo(NpyArray self, NpyArray values0, NpyArray indices0, NPY_CLIPMODE clipmode)
        {
            return numpyinternal.NpyArray_PutTo(self, values0, indices0, clipmode);
        }
        internal static int NpyArray_PutMask(NpyArray self, NpyArray values0, NpyArray mask0)
        {
            return numpyinternal.NpyArray_PutMask(self, values0, mask0);
        }
        internal static NpyArray NpyArray_Repeat(NpyArray aop, NpyArray op, int axis)
        {
            return numpyinternal.NpyArray_Repeat(aop, op, axis);
        }
        internal static NpyArray NpyArray_Choose(NpyArray ip, NpyArray []mps, int n, NpyArray ret, NPY_CLIPMODE clipmode)
        {
            return numpyinternal.NpyArray_Choose(ip, mps, n, ret, clipmode);
        }
        internal static int NpyArray_Partition(NpyArray op, NpyArray ktharray, int axis, NPY_SELECTKIND which)
        {
            return numpyinternal.NpyArray_Partition(op, ktharray, axis, which);
        }
        internal static NpyArray NpyArray_ArgPartition(NpyArray op, NpyArray ktharray, int axis, NPY_SELECTKIND which)
        {
            return numpyinternal.NpyArray_ArgPartition(op, ktharray, axis, which);
        }
        internal static int NpyArray_Sort(NpyArray op, int axis, NPY_SORTKIND which)
        {
            return numpyinternal.NpyArray_Sort(op, axis, which);
        }
        internal static NpyArray NpyArray_ArgSort(NpyArray op, int axis, NPY_SORTKIND which)
        {
            return numpyinternal.NpyArray_ArgSort(op, axis, which);
        }
        internal static NpyArray NpyArray_LexSort(NpyArray []mps, int n, int axis)
        {
            return numpyinternal.NpyArray_LexSort(mps, n, axis);
        }
        internal static NpyArray NpyArray_SearchSorted(NpyArray op1, NpyArray op2, NPY_SEARCHSIDE side)
        {
            return numpyinternal.NpyArray_SearchSorted(op1, op2, side);
        }
        internal static int NpyArray_NonZero(NpyArray self, NpyArray[] index_arrays, object obj)
        {
            return numpyinternal.NpyArray_NonZero(self, index_arrays, obj);
        }
        internal static NpyArray NpyArray_Subarray(NpyArray self, VoidPtr dataptr)
        {
            return numpyinternal.NpyArray_Subarray(self, dataptr);
        }
        internal static IList<npy_intp> NpyArray_IndexesFromAxis(NpyArray op, int axis)
        {
            return numpyinternal.NpyArray_IndexesFromAxis(op, axis);
        }


#endregion

        #region npy_iterators


        internal static VoidPtr NpyArray_IterNext(NpyArrayIterObject it)
        {
            return numpyinternal.NpyArray_IterNext(it);
        }

        // Resets the iterator to the first element in the array.

        internal static void NpyArray_IterReset(NpyArrayIterObject it)
        {
            numpyinternal.NpyArray_IterReset(it);
        }

        internal static NpyArrayIterObject NpyArray_IterNew(NpyArray ao)
        {
            return numpyinternal.NpyArray_IterNew(ao);
        }
        internal static NpyArrayIterObject NpyArray_BroadcastToShape(NpyArray ao, npy_intp []dims, int nd)
        {
            return numpyinternal.NpyArray_BroadcastToShape(ao, dims, nd);
        }
        internal static NpyArray NpyArray_IterSubscript(NpyArrayIterObject self, NpyIndex []indexes, int n)
        {
            return numpyinternal.NpyArray_IterSubscript(self, indexes, n);
        }
        internal static int NpyArray_IterSubscriptAssign(NpyArrayIterObject self, NpyIndex []indexes, int n, NpyArray value)
        {
            return numpyinternal.NpyArray_IterSubscriptAssign(self, indexes, n, value);
        }
        internal static NpyArrayIterObject NpyArray_IterAllButAxis(NpyArray obj, ref int inaxis)
        {
            return numpyinternal.NpyArray_IterAllButAxis(obj, ref inaxis);
        }
        internal static int NpyArray_RemoveSmallest(NpyArrayMultiIterObject multi)
        {
            return numpyinternal.NpyArray_RemoveSmallest(multi);
        }

        internal static void NpyArray_Reset(NpyArrayMultiIterObject multi)
        {
            numpyinternal.NpyArray_MultiIter_RESET(multi);
        }

        internal static void NpyArray_Next(NpyArrayMultiIterObject multi)
        {
            numpyinternal.NpyArray_MultiIter_NEXT(multi);
        }

        internal static bool NpyArray_NotDone(NpyArrayMultiIterObject multi)
        {
            return numpyinternal.NpyArray_MultiIter_NOTDONE(multi);
        }

        internal static VoidPtr NpyArray_MultiIter_DATA(NpyArrayMultiIterObject multi, int index)
        {
            return numpyinternal.NpyArray_MultiIter_DATA(multi, index);
        }

        internal static int NpyArray_Broadcast(NpyArrayMultiIterObject mit)
        {
            return numpyinternal.NpyArray_Broadcast(mit);
        }
        internal static NpyArrayMultiIterObject NpyArray_MultiIterNew()
        {
            return numpyinternal.NpyArray_MultiIterNew();
        }
        internal static NpyArrayMultiIterObject NpyArray_MultiIterFromArrays(NpyArray[] mps, int n, int nadd, params object[] p)
        {
            return numpyinternal.NpyArray_MultiIterFromArrays(mps, n, nadd, p);
        }
#endregion

#region npy_mapping

        internal static NpyArrayMapIterObject NpyArray_MapIterNew(NpyIndex[] indexes, int n)
        {
            return numpyinternal.NpyArray_MapIterNew(indexes, n);
        }

        internal static int NpyArray_MapIterBind(NpyArrayMapIterObject mit, NpyArray arr, NpyArray true_array)
        {
            return numpyinternal.NpyArray_MapIterBind(mit, arr, true_array);
        }

        internal static void NpyArray_MapIterReset(NpyArrayMapIterObject mit)
        {
            numpyinternal.NpyArray_MapIterReset(mit);
        }

        internal static void NpyArray_MapIterNext(NpyArrayMapIterObject mit)
        {
            numpyinternal.NpyArray_MapIterNext(mit);
        }

        internal static NpyArray NpyArray_GetMap(NpyArrayMapIterObject mit)
        {
            return numpyinternal.NpyArray_GetMap(mit);
        }

        internal static int NpyArray_SetMap(NpyArrayMapIterObject mit, NpyArray arr)
        {
            return numpyinternal.NpyArray_SetMap(mit, arr);
        }

        internal static NpyArray NpyArray_ArrayItem(NpyArray self, npy_intp i)
        {
            return numpyinternal.NpyArray_ArrayItem(self, i);
        }

        internal static NpyArray NpyArray_IndexSimple(NpyArray self, NpyIndex []indexes, int n)
        {
            return numpyinternal.NpyArray_IndexSimple(self, indexes, n);
        }

        internal static int NpyArray_IndexFancyAssign(NpyArray self, NpyIndex []indexes, int n, NpyArray value)
        {
            return numpyinternal.NpyArray_IndexFancyAssign(self, indexes, n, value);
        }

        internal static NpyArray NpyArray_Subscript(NpyArray self, NpyIndex []indexes, int n)
        {
            return numpyinternal.NpyArray_Subscript(self, indexes, n);
        }

        internal static int NpyArray_SubscriptAssign(NpyArray self, NpyIndex []indexes, int n, NpyArray value)
        {
            return numpyinternal.NpyArray_SubscriptAssign(self, indexes, n, value);
        }

#endregion

#region npy_methods

        internal static NpyArray NpyArray_GetField(NpyArray self, NpyArray_Descr typed, int offset)
        {
            return numpyinternal.NpyArray_GetField(self, typed, offset);
        }

        internal static int NpyArray_SetField(NpyArray self, NpyArray_Descr dtype, int offset, NpyArray val)
        {
            return numpyinternal.NpyArray_SetField(self, dtype, offset, val);
        }

        internal static NpyArray NpyArray_Byteswap(NpyArray self, bool inplace)
        {
            return numpyinternal.NpyArray_Byteswap(self, inplace);
        }

#endregion

#region npy_multiarray


        /* Initializes the library at startup. This functions must be called exactly once by the interface layer.*/
        internal static void npy_initlib(NpyArray_FunctionDefs functionDefs, NpyInterface_WrapperFuncs wrapperFuncs,
            npy_tp_error_set error_set,
            npy_tp_error_occurred error_occurred,
            npy_tp_error_clear error_clear,
            npy_tp_cmp_priority cmp_priority,
            npy_interface_incref incref, npy_interface_decref decref,
            enable_threads et, disable_threads dt)
        {
            numpyinternal.npy_initlib(functionDefs, wrapperFuncs,
            error_set,
            error_occurred,
            error_clear,
            cmp_priority,
            incref, decref,
            et, dt);

            numpyinternal._intialize_builtin_descrs();
        }

        internal static npy_intp NpyArray_MultiplyIntList(npy_intp[] l1, int n)
        {
            return numpyinternal.NpyArray_MultiplyIntList(l1, n);
        }

        internal static npy_intp NpyArray_MultiplyList(npy_intp[] l1, int n)
        {
            return numpyinternal.NpyArray_MultiplyIntList(l1, n);
        }

        internal static npy_intp NpyArray_OverflowMultiplyList(npy_intp[] l1, int n)
        {
            return numpyinternal.NpyArray_OverflowMultiplyList(l1, n);
        }

        internal static VoidPtr NpyArray_GetPtr(NpyArray arr, npy_intp []indexes)
        {
            return numpyinternal.NpyArray_GetPtr(arr, indexes);
        }

        internal static bool NpyArray_CompareLists(npy_intp[] l1, npy_intp[] l2, int n)
        {
            return numpyinternal.NpyArray_CompareLists(l1, l2, n);
        }

        internal static int NpyArray_AsCArray(ref NpyArray apIn, ref byte[] ptr, npy_intp[] dims, int nd, NpyArray_Descr typedescr)
        {
            return numpyinternal.NpyArray_AsCArray(ref apIn, ref ptr, dims, nd, typedescr);
        }

        internal static int NpyArray_Free(NpyArray ap, object ptr)
        {
            return numpyinternal.NpyArray_Free(ap, ptr);
        }

        internal static NpyArray NpyArray_Concatenate(NpyArray[] arrays, int? axis, NpyArray ret)
        {
            return numpyinternal.NpyArray_Concatenate(arrays, axis, ret);
        }

        internal static NPY_SCALARKIND NpyArray_ScalarKind(NPY_TYPES typenum, ref NpyArray arr)
        {
            return numpyinternal.NpyArray_ScalarKind(typenum, arr);
        }

        internal static bool NpyArray_CanCoerceScalar(NPY_TYPES thistype, NPY_TYPES neededtype, NPY_SCALARKIND scalar)
        {
            return numpyinternal.NpyArray_CanCoerceScalar(thistype,  neededtype, scalar);
        }

        internal static NpyArray NpyArray_InnerProduct(NpyArray ap1, NpyArray ap2, NPY_TYPES typenum)
        {
            return numpyinternal.NpyArray_InnerProduct(ap1, ap2, typenum);
        }

        internal static NpyArray NpyArray_MatrixProduct(NpyArray ap1, NpyArray ap2, NPY_TYPES typenum)
        {
            return numpyinternal.NpyArray_MatrixProduct(ap1, ap2, typenum);
        }

        internal static NpyArray NpyArray_CopyAndTranspose(NpyArray arr)
        {
            return numpyinternal.NpyArray_CopyAndTranspose(arr);
        }

        internal static NpyArray NpyArray_Correlate(NpyArray ap1, NpyArray ap2, NPY_TYPES typenum, NPY_CONVOLE_MODE mode)
        {
            return numpyinternal.NpyArray_Correlate(ap1, ap2, typenum, mode);
        }

        internal static bool NpyArray_EquivTypes(NpyArray_Descr typ1, NpyArray_Descr typ2)
        {
            return numpyinternal.NpyArray_EquivTypes(typ1, typ2);
        }

        internal static void NpyArray_CopyTo(NpyArray dst, NpyArray src, NPY_CASTING casting, NpyArray wheremask_in)
        {
            numpyinternal.NpyArray_CopyTo(dst, src, casting, wheremask_in);
        }

        internal static void NpyArray_Place(NpyArray arr, NpyArray mask,  NpyArray vals)
        {
            numpyinternal.NpyArray_Place(arr, mask, vals);
        }

        internal static bool NpyArray_EquivTypenums(NPY_TYPES typenum1, NPY_TYPES typenum2)
        {
            return numpyinternal.NpyArray_EquivTypenums(typenum1, typenum2);
        }

        internal static int NpyArray_GetEndianness()
        {
            return numpyinternal.NpyArray_GetEndianness();
        }

#endregion

#region npy_number


 
        internal static NpyUFuncObject NpyArray_GetNumericOp(UFuncOperation op)
        {
            return numpyinternal.NpyArray_GetNumericOp(op);
        }

        internal static NpyUFuncObject NpyArray_SetNumericOp(UFuncOperation op, NpyUFuncObject func)
        {
            return numpyinternal.NpyArray_SetNumericOp(op, func);
        }

  
        internal static int NpyArray_Bool(NpyArray mp)
        {
            return numpyinternal.NpyArray_Bool(mp);
        }

#endregion

#region npy_refcount

        internal static void NpyArray_Item_INCREF(byte[] data, NpyArray_Descr descr)
        {
            numpyinternal.NpyArray_Item_INCREF(data, descr);
        }

        internal static void NpyArray_Item_XDECREF(byte [] data, NpyArray_Descr descr)
        {
            numpyinternal.NpyArray_Item_XDECREF(data, descr);
        }

        internal static int NpyArray_INCREF(NpyArray mp)
        {
            return numpyinternal.NpyArray_INCREF(mp);
        }

        internal static int NpyArray_XDECREF(NpyArray mp)
        {
            return numpyinternal.NpyArray_XDECREF(mp);
        }

#endregion

#region npy_shape

        internal static int NpyArray_Resize(NpyArray self, NpyArray_Dims newshape, bool refcheck, NPY_ORDER fortran)
        {
            return numpyinternal.NpyArray_Resize(self, newshape, refcheck, fortran);
        }

        internal static NpyArray NpyArray_Newshape(NpyArray self, NpyArray_Dims newdims, NPY_ORDER fortran)
        {
            return numpyinternal.NpyArray_Newshape(self, newdims, fortran);
        }

        internal static NpyArray NpyArray_Squeeze(NpyArray self)
        {
            return numpyinternal.NpyArray_Squeeze(self);
        }

        internal static NpyArray NpyArray_SqueezeSelected(NpyArray self, int axis)
        {
            return numpyinternal.NpyArray_SqueezeSelected(self, axis);
        }

        internal static NpyArray NpyArray_SwapAxes(NpyArray ap, int a1, int a2)
        {
            return numpyinternal.NpyArray_SwapAxes(ap, a1, a2);
        }

        internal static NpyArray NpyArray_Ravel(NpyArray a, NPY_ORDER order)
        {
            return numpyinternal.NpyArray_Ravel(a, order);
        }

        internal static NpyArray NpyArray_Flatten(NpyArray a, NPY_ORDER order)
        {
            return numpyinternal.NpyArray_Flatten(a, order);
        }

        internal static NpyArray NpyArray_FlatView(NpyArray a)
        {
            return numpyinternal.NpyArray_FlatView(a);
        }

        internal static NpyArray NpyArray_Transpose(NpyArray ap, NpyArray_Dims permute)
        {
            return numpyinternal.NpyArray_Transpose(ap, permute);
        }

#endregion

#region npy_unfunc_object

        internal static NpyArray NpyUFunc_GenericReduction(NpyUFuncObject self, NpyArray arr, NpyArray indices,
                          NpyArray _out, int axis, NpyArray_Descr otype, GenericReductionOp operation)
        {
            return numpyinternal.NpyUFunc_GenericReduction(self, arr, indices, _out, axis, otype, operation, false);
        }

        #endregion

        #region DefaultArrayHandlers

        internal static void SetArrayHandler(NPY_TYPES ItemType, IArrayHandlers Handlers)
        {
            DefaultArrayHandlers.SetArrayHandler(ItemType, Handlers);
        }
        internal static IArrayHandlers GetArrayHandler(NPY_TYPES ItemType)
        {
            return DefaultArrayHandlers.GetArrayHandler(ItemType);
        }
        #endregion

        #region npy_usertypes

        internal static void NpyArray_InitArrFuncs(NpyArray_ArrFuncs f)
        {
            numpyinternal.NpyArray_InitArrFuncs(f);
        }

        internal static int NpyArray_GetNumusertypes()
        {
            return numpyinternal.NpyArray_GetNumusertypes();
        }

        internal static bool NpyArray_RegisterDataType(NpyArray_Descr descr)
        {
            return numpyinternal.NpyArray_RegisterDataType(descr);
        }

        internal static int NpyArray_RegisterCastFunc(NpyArray_Descr descr, NPY_TYPES totype, NpyArray_VectorUnaryFunc castfunc)
        {
            return numpyinternal.NpyArray_RegisterCastFunc(descr, totype, castfunc);
        }

        internal static int NpyArray_RegisterCanCast(NpyArray_Descr descr, NPY_TYPES totype, NPY_SCALARKIND scalar)
        {
            return numpyinternal.NpyArray_RegisterCanCast(descr, totype, scalar);
        }

        internal static NpyArray_Descr NpyArray_UserDescrFromTypeNum(NPY_TYPES typenum)
        {
            return numpyinternal.NpyArray_UserDescrFromTypeNum(typenum);
        }
#endregion

#region npy_dict

        internal static void NpyDict_Destroy(NpyDict hashTable)
        {
            numpyinternal.NpyDict_Destroy(hashTable);
        }

#endregion

    }
}
